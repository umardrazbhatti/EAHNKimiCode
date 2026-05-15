"""
scripts/train_real.py — Phase 2: FF++ fine-tuning with stratified batches.
MUST load synthetic_pretrained.pth.
"""

import os
import math
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader, Sampler
from torch.amp import GradScaler, autocast

from config import EAHNConfig, parse_args
from data.datasets import DeepfakeDataset
from data.collate import deepfake_collate_fn
from models.eahn import EAHN
from losses.classification import build_classification_loss
from losses.explanation import ExplanationLoss
from losses.temporal import TemporalConsistencyLoss
from metrics.detection import DetectionMetrics
from utils.checkpointing import save_checkpoint, load_checkpoint
from utils.logging_utils import Logger


class StratifiedBatchSampler(Sampler):
    """
    Guarantees every batch contains exactly num_real real samples and
    (batch_size - num_real) fake samples. Eliminates all-real or all-fake batches.
    """
    def __init__(self, dataset, batch_size, num_real_per_batch=None):
        labels = np.array([s["label"] for s in dataset.samples], dtype=int)
        self.real_idx = np.where(labels == 0)[0].tolist()
        self.fake_idx = np.where(labels == 1)[0].tolist()
        self.batch_size = batch_size

        # Default: at least 1 real per batch (critical for 5:1 imbalance with batch=4)
        if num_real_per_batch is None:
            self.num_real = max(1, batch_size // 4)
        else:
            self.num_real = num_real_per_batch
        self.num_fake = batch_size - self.num_real

        self.num_batches = min(
            len(self.real_idx) // self.num_real,
            len(self.fake_idx) // self.num_fake,
        )
        self.length = self.num_batches

    def __iter__(self):
        np.random.shuffle(self.real_idx)
        np.random.shuffle(self.fake_idx)
        for i in range(self.num_batches):
            batch = (
                self.real_idx[i * self.num_real : (i + 1) * self.num_real] +
                self.fake_idx[i * self.num_fake : (i + 1) * self.num_fake]
            )
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.length


def main(config: EAHNConfig):
    device = torch.device(config.device)
    print(f"[FF++ Phase] Device: {device}")
    os.makedirs(config.output_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    train_ds = DeepfakeDataset(config, "train", config.dataset_name)
    val_ds = DeepfakeDataset(config, "val", config.dataset_name)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # STRATIFIED SAMPLER: 1 real + 3 fake per batch (batch_size=4)
    sampler = StratifiedBatchSampler(train_ds, config.batch_size, num_real_per_batch=1)
    train_loader = DataLoader(
        train_ds, batch_sampler=sampler,
        num_workers=config.num_workers, collate_fn=deepfake_collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        num_workers=config.num_workers, collate_fn=deepfake_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # ── Smoke check ───────────────────────────────────────────────────────
    _sb = next(iter(train_loader))
    _bl = _sb["label"].cpu().numpy().astype(int)
    _n_real, _n_fake = int((_bl == 0).sum()), int((_bl == 1).sum())
    print(f"[Smoke] First batch: real={_n_real} fake={_n_fake}")
    assert _n_real > 0 and _n_fake > 0, "Stratified sampler failed."

    # ── Model ─────────────────────────────────────────────────────────────
    model = EAHN(config).to(device)

    # Load synthetic pre-trained weights
    synth_ckpt = os.path.join(config.output_dir, "synthetic_pretrained.pth")
    if config.resume_checkpoint:
        synth_ckpt = config.resume_checkpoint
    if os.path.exists(synth_ckpt):
        ckpt = torch.load(synth_ckpt, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"[INIT] Loaded synthetic checkpoint: {synth_ckpt}")
    else:
        print("[WARN] No synthetic checkpoint found. Training from scratch — expect poor results.")

    # Backbone unfrozen immediately with FULL LR
    if config.freeze_backbone:
        model.spatial_stream.set_frozen(False)
    backbone_lr = config.lr * config.backbone_lr_ratio
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if n.startswith("spatial_stream.backbone.")],
         "lr": backbone_lr, "weight_decay": config.weight_decay},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("spatial_stream.backbone.")],
         "lr": config.lr, "weight_decay": config.weight_decay},
    ])

    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return (epoch + 1) / (config.warmup_epochs + 1)
        progress = (epoch - config.warmup_epochs) / max(config.epochs - config.warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = config.mixed_precision and device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 7
    scaler = GradScaler("cuda") if use_amp else None
    logger = Logger(config.output_dir)

    # ── Losses ────────────────────────────────────────────────────────────
    cls_loss_fn = build_classification_loss(config)
    exp_loss_fn = ExplanationLoss(
        alpha=config.alpha,
        beta=config.beta,
        diversity_weight=config.attn_diversity_weight,
        class_sep_weight=config.class_sep_weight,
    )
    temp_loss_fn = TemporalConsistencyLoss(gamma=config.gamma)

    ckpt_path = os.path.join(config.output_dir, "best_model.pth")

    metrics_csv_path = os.path.join(config.output_dir, "train_metrics.csv")
    with open(metrics_csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_real_acc", "train_fake_acc",
            "val_auc", "val_f1", "val_real_acc", "val_fake_acc", "lr"
        ])

    # ── Training loop ───────────────────────────────────────────────────
    import contextlib
    best_auc = -1.0
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        epoch_real_correct = epoch_real_total = 0
        epoch_fake_correct = epoch_fake_total = 0

        # Anneal attention temp: 1.5 → 0.5 over epochs
        target_temp = max(0.5, 1.5 * math.exp(-epoch / 3.0))
        model.set_attention_temp(target_temp)

        for batch_idx, batch in enumerate(train_loader):
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"].to(device)
            has_mask = batch["has_mask"].to(device)

            ctx = autocast("cuda") if use_amp else contextlib.nullcontext()
            with ctx:
                out = model(frames)

                if torch.isnan(out.logit).any() or torch.isinf(out.logit).any():
                    optimizer.zero_grad()
                    continue

                l_cls = cls_loss_fn(out.logit, labels)
                exp_out = exp_loss_fn(out.M_t, masks, has_mask, labels=labels)
                l_exp = exp_out.loss
                l_temp = temp_loss_fn(out.M_t, out.low_level)

                # CRITICAL: lambda1 is now 0.05. Classification dominates.
                _global_step = epoch * len(train_loader) + batch_idx
                _lambda1_eff = config.lambda1 * min(1.0, _global_step / 2000.0)
                l_total = l_cls + _lambda1_eff * l_exp + config.lambda2 * l_temp

                if torch.isnan(l_total) or torch.isinf(l_total):
                    optimizer.zero_grad()
                    continue

                loss = l_total / config.grad_accum_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % config.grad_accum_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                preds = (out.prob >= 0.5).float()
                real_mask = (labels == 0)
                fake_mask = (labels == 1)
                if real_mask.any():
                    epoch_real_correct += (preds[real_mask] == labels[real_mask]).sum().item()
                    epoch_real_total += real_mask.sum().item()
                if fake_mask.any():
                    epoch_fake_correct += (preds[fake_mask] == labels[fake_mask]).sum().item()
                    epoch_fake_total += fake_mask.sum().item()

            running_loss += l_total.item()

        # Leftover step
        if len(train_loader) % config.grad_accum_steps != 0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        real_acc = epoch_real_correct / max(epoch_real_total, 1)
        fake_acc = epoch_fake_correct / max(epoch_fake_total, 1)
        avg_train_loss = running_loss / max(len(train_loader), 1)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        probs_list, labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                frames = batch["frames"].to(device)
                out = model(frames)
                probs_list.extend(out.prob.cpu().tolist())
                labels_list.extend(batch["label"].cpu().tolist())

        metrics = DetectionMetrics.compute(probs_list, labels_list)
        val_auc = metrics.get("auc_roc", float("nan"))
        val_f1 = metrics.get("f1", 0.0)

        # Per-class val accuracy
        probs_arr = np.array(probs_list)
        labels_arr = np.array(labels_list, dtype=int)
        preds_arr = (probs_arr >= 0.5).astype(int)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels_arr, preds_arr)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            val_real_acc = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            val_fake_acc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            val_real_acc = val_fake_acc = 0.0

        print(f"Epoch {epoch+1}/{config.epochs} | "
              f"TrainLoss: {avg_train_loss:.4f} | RealAcc: {real_acc:.3f} | FakeAcc: {fake_acc:.3f} | "
              f"ValAUC: {val_auc:.4f} | ValF1: {val_f1:.4f} | "
              f"ValReal: {val_real_acc:.3f} | ValFake: {val_fake_acc:.3f}")

        with open(metrics_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, f"{avg_train_loss:.4f}", f"{real_acc:.4f}", f"{fake_acc:.4f}",
                f"{val_auc:.4f}", f"{val_f1:.4f}", f"{val_real_acc:.4f}", f"{val_fake_acc:.4f}",
                f"{optimizer.param_groups[0]['lr']:.2e}",
            ])

        # Early stopping
        if not math.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch, best_auc, config, ckpt_path)
            print(f"--> Best model saved (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            print(f"--> No improvement. Patience: {patience_counter}/{config.patience}")
            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    logger.close()
    print(f"\nTraining complete. Best Val AUC-ROC: {best_auc:.4f}")

    if config.eval_after_train:
        from scripts.evaluate import run_evaluation
        print("\n--- Starting evaluation ---")
        run_evaluation(config)


if __name__ == "__main__":
    args = parse_args()
    config = EAHNConfig.from_args(args)
    main(config)