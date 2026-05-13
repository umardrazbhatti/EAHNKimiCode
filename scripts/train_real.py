"""
scripts/train_real.py — Phase 2 GPU training on FF++/Celeb-DF/DFDC.

FIXES vs previous version:
 1. Backbone freezing now freezes BN stats (prevents NaN cascade).
 2. Removed cls_dropout dependency.
 3. Gradient-alignment loss fixed (no .backward() contamination).
 4. Temperature clamped to [1.0, 2.0] (escapes uniform trap).
 5. Focal loss now uses label smoothing.
 6. NaN/Inf guards on model outputs and loss — skip bad batches.
 7. lambda1_eff warmup increased 200 → 2000 steps so explanation loss
    does not overwhelm the classifier while attention is still learning.
 8. Per-class accuracy logging preserved.
 9. Differential learning rates: backbone gets 0.1× head LR.
10. Unfreeze LR logic fixed for multi-group optimizer.
11. FIX: scheduler.base_lrs updated on unfreeze so CosineAnnealingLR
    respects the reduced backbone LR.
12. FIX: torch.cuda.empty_cache() before unfreeze to prevent OOM.
13. FIX: step leftover gradients at end of epoch to prevent stale grads.
14. FIX: wrap unfreeze in try/except with clear diagnostics.
15. FIX: cumulative 100-batch logging to prevent Kaggle log bloat.
"""

import os
import math
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
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


def main(config: EAHNConfig):
    device = torch.device(config.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(device)
        name = torch.cuda.get_device_name(device)
        print(f"[Device] {name} | CUDA capability sm_{cap[0]}{cap[1]}")
        if cap[0] < 7:
            print(
                f"[WARNING] sm_{cap[0]}{cap[1]} is below PyTorch minimum "
                f"(sm_70). Switch Kaggle accelerator to T4. "
                "Falling back to CPU for MTCNN. AMP disabled."
            )
    os.makedirs(config.output_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = DeepfakeDataset(config, "train", config.dataset_name)
    val_ds = DeepfakeDataset(config, "val", config.dataset_name)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    labels_arr = np.array([s["label"] for s in train_ds.samples], dtype=int)
    class_counts = np.bincount(labels_arr, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels_arr]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    print(
        f"[Sampler] class_counts={class_counts.tolist()} "
        f"class_weights={class_weights.tolist()}"
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, sampler=sampler,
        num_workers=config.num_workers, collate_fn=deepfake_collate_fn,
        drop_last=True, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        num_workers=config.num_workers, collate_fn=deepfake_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # ── First-batch class-balance smoke check ─────────────────────────────────
    _sb = next(iter(train_loader))
    _bl = _sb["label"].cpu().numpy().astype(int)
    _n_real, _n_fake = int((_bl == 0).sum()), int((_bl == 1).sum())
    print(f"[Smoke] First batch: real={_n_real} fake={_n_fake}")
    assert _n_real > 0 and _n_fake > 0, (
        f"First batch is single-class (real={_n_real}, fake={_n_fake}). "
        "Sampler or split is broken — check DeepfakeDataset._split()."
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EAHN(config).to(device)

    # FIX: Differential learning rates — backbone gets 0.1× head LR
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if name.startswith("spatial_stream.backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": config.lr * 0.1},
        {"params": head_params, "lr": config.lr},
    ], weight_decay=config.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )

    use_amp = (
        config.mixed_precision
        and device.type == "cuda"
        and torch.cuda.get_device_capability(device)[0] >= 7
    )
    scaler = GradScaler("cuda") if use_amp else None
    logger = Logger(config.output_dir)

    # ── Progressive backbone freezing ─────────────────────────────────────────
    if config.freeze_backbone:
        model.spatial_stream.set_frozen(True)
        print(f"[Backbone] Frozen for first {config.unfreeze_backbone_epoch} epochs")

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_auc = -1.0
    if config.resume_checkpoint and os.path.exists(config.resume_checkpoint):
        ckpt = load_checkpoint(config.resume_checkpoint, model, optimizer, scheduler)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_auc = ckpt.get("best_metric", 0.0)
        print(f"Resumed from epoch {start_epoch}, best AUC {best_auc:.4f}")

    # ── Losses ────────────────────────────────────────────────────────────────
    cls_loss_fn = build_classification_loss(config)
    exp_loss_fn = ExplanationLoss(
        alpha=config.alpha,
        beta=config.beta,
        diversity_weight=config.attn_diversity_weight,
        class_sep_weight=0.5,
    )
    temp_loss_fn = TemporalConsistencyLoss(gamma=config.gamma)

    ckpt_path = os.path.join(config.output_dir, "best_model.pth")

    # ── Training loop ─────────────────────────────────────────────────────────
    import contextlib
    total_batches = len(train_loader)
    epoch_w = len(str(config.epochs))
    batch_w = len(str(total_batches))

    for epoch in range(start_epoch, config.epochs):
        # ── Unfreeze backbone at scheduled epoch ─────────────────────────────
        if config.freeze_backbone and epoch == config.unfreeze_backbone_epoch:
            print(f"[Backbone] Preparing to unfreeze at epoch {epoch} ...")
            if device.type == "cuda":
                torch.cuda.empty_cache()
                print(f"[VRAM] Cached before empty: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            try:
                model.spatial_stream.set_frozen(False)
                print(f"[Backbone] Unfrozen at epoch {epoch}")
            except Exception as e:
                raise RuntimeError(f"set_frozen(False) failed: {e}")

            old_lr = optimizer.param_groups[0]["lr"]
            new_lr = old_lr * 0.1
            optimizer.param_groups[0]["lr"] = new_lr
            scheduler.base_lrs[0] = new_lr
            print(f"[LR] Backbone group 0: {old_lr:.2e} → {new_lr:.2e}")
            print(f"[LR] Head group 1:    {optimizer.param_groups[1]['lr']:.2e}")

        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        epoch_real_correct = 0
        epoch_real_total = 0
        epoch_fake_correct = 0
        epoch_fake_total = 0

        # ── Cumulative window accumulators (100-batch) ───────────────────────
        win_loss = 0.0
        win_cls = 0.0
        win_exp = 0.0
        win_temp = 0.0
        win_sim = 0.0
        win_count = 0

        for batch_idx, batch in enumerate(train_loader):
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"].to(device)
            has_mask = batch["has_mask"].to(device)

            ctx = autocast("cuda") if use_amp else contextlib.nullcontext()

            with ctx:
                out = model(frames)

                # ── NaN/Inf guard on model outputs ──────────────────────────
                if (torch.isnan(out.logit).any() or torch.isinf(out.logit).any() or
                    torch.isnan(out.M_t).any() or torch.isinf(out.M_t).any()):
                    print(
                        f"[WARNING] NaN/Inf in model outputs at "
                        f"E{epoch+1}B{batch_idx}. Skipping batch."
                    )
                    optimizer.zero_grad()
                    continue

                l_cls = cls_loss_fn(out.logit, labels)
                exp_out = exp_loss_fn(out.M_t, masks, has_mask, labels=labels)
                l_exp = exp_out.loss
                l_temp = temp_loss_fn(out.M_t, out.low_level)

                l_grad_align = torch.tensor(0.0, device=device)
                if config.lambda_grad_align > 0 and batch_idx % 5 == 0:
                    grad_saliency = model.compute_gradient_saliency(frames)
                    l_grad_align = torch.nn.functional.mse_loss(out.M_t, grad_saliency)

                _global_step = epoch * len(train_loader) + batch_idx
                _lambda1_eff = config.lambda1 * min(1.0, _global_step / 2000.0)
                l_total = (
                    l_cls
                    + _lambda1_eff * l_exp
                    + config.lambda2 * l_temp
                    + config.lambda_grad_align * l_grad_align
                )

                # ── NaN/Inf guard on loss ───────────────────────────────────
                if torch.isnan(l_total) or torch.isinf(l_total):
                    print(
                        f"[WARNING] NaN/Inf loss at E{epoch+1}B{batch_idx}. "
                        f"L_cls={l_cls.item():.4f} L_exp={l_exp.item():.4f}. "
                        "Skipping batch."
                    )
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

            # Per-class accuracy tracking
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

            # ── First-batch diagnostics (keep) ───────────────────────────────
            if epoch == 0 and batch_idx == 0:
                print(f"[DIAG] M_t mean={out.M_t.mean():.4f} std={out.M_t.std():.4f}")
                print(f"[DIAG] L_cls={l_cls.item():.4f} L_exp={l_exp.item():.4f} L_temp={l_temp.item():.4f}")
                print(f"[DIAG] attn_temp=exp({model.cross_attention.log_temp.item():.3f})={torch.exp(model.cross_attention.log_temp).item():.3f}")

            # ── LIVE sparse diagnostic every 20 batches (keep) ───────────────
            if batch_idx % 20 == 0:
                _live_std = out.M_t.std().item()
                _live_tau = model.cross_attention.log_temp.exp().item()
                print(
                    f"[LIVE E{epoch+1} B{batch_idx:03d}] "
                    f"M_t_std={_live_std:.3f} "
                    f"tau={_live_tau:.2f} "
                    f"L_cls={l_cls.item():.2f} "
                    f"L_H={exp_out.l_h:.2f} "
                    f"L_TV={exp_out.l_tv:.2f} "
                    f"L_div={exp_out.l_div:.2f} "
                    f"L_class_sep={exp_out.l_class_sep:.2f} "
                    f"L_temp={l_temp.item():.4f} "
                    f"L_grad={l_grad_align.item():.4f} "
                    f"sample_sim={exp_out.inter_sample_sim:.2f}"
                )

            # ── Accumulate for 100-batch window ──────────────────────────────
            win_loss += l_total.item()
            win_cls += l_cls.item()
            win_exp += l_exp.item()
            win_temp += l_temp.item()
            win_sim += exp_out.inter_sample_sim
            win_count += 1

            # ── Print cumulative summary every 100 batches ───────────────────
            is_last = (batch_idx == total_batches - 1)
            if (batch_idx + 1) % 100 == 0 or is_last:
                n = max(win_count, 1)
                print(
                    f"Epoch {epoch + 1:>{epoch_w}}/{config.epochs} | "
                    f"Batch {batch_idx + 1:>{batch_w}}/{total_batches} | "
                    f"AvgLoss: {win_loss/n:.4f} | "
                    f"Cls: {win_cls/n:.4f} | "
                    f"Exp: {win_exp/n:.4f} | "
                    f"Temp: {win_temp/n:.4f} | "
                    f"sim: {win_sim/n:.2f}"
                )
                # reset window
                win_loss = win_cls = win_exp = win_temp = win_sim = 0.0
                win_count = 0

            running_loss += l_total.item()

        # FIX: Step any leftover gradients
        if total_batches % config.grad_accum_steps != 0:
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
        print(
            f"[Epoch {epoch+1} Class Acc] Real: {real_acc:.3f} ({epoch_real_correct}/{epoch_real_total}) "
            f"Fake: {fake_acc:.3f} ({epoch_fake_correct}/{epoch_fake_total})"
        )

        avg_train_loss = running_loss / max(len(train_loader), 1)
        logger.log_scalars("train", {
            "loss": avg_train_loss,
            "real_acc": real_acc,
            "fake_acc": fake_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }, epoch)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        probs_list, labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                frames = batch["frames"].to(device)
                out = model(frames)
                probs_list.extend(out.prob.cpu().tolist())
                labels_list.extend(batch["label"].cpu().tolist())

        metrics = DetectionMetrics.compute(probs_list, labels_list)
        logger.log_scalars("val", metrics, epoch)
        print(
            f"Epoch {epoch + 1:>{epoch_w}}/{config.epochs} | "
            f"Val AUC-ROC: {metrics['auc_roc']:.4f} | "
            f"F1: {metrics['f1']:.4f}"
        )

        val_auc = metrics.get("auc_roc", float("nan"))
        if not math.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(model, optimizer, scheduler, epoch, best_auc,
                            config, ckpt_path)
            print(f"--> Best model saved (AUC-ROC: {best_auc:.4f})")

        last_ckpt = os.path.join(config.output_dir, f"checkpoint_epoch{epoch:03d}.pth")
        save_checkpoint(model, optimizer, scheduler, epoch, val_auc, config, last_ckpt)

    logger.close()
    print(f"\nTraining complete. Best AUC-ROC: {best_auc:.4f}")

    if config.eval_after_train:
        from scripts.evaluate import run_evaluation
        print("\n--- Starting evaluation ---")
        run_evaluation(config)


if __name__ == "__main__":
    args = parse_args()
    config = EAHNConfig.from_args(args)
    main(config)