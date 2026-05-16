"""
scripts/train_real.py — Phase 2: FF++ fine-tuning.
FIXED: Now matches actual repo APIs, loads synthetic checkpoint, uses
TemporalConsistencyLoss(M_t, low_level) correctly, and handles class imbalance.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import EAHNConfig, parse_args
from models.eahn import EAHN
from losses.explanation import ExplanationLoss
from losses.temporal import TemporalConsistencyLoss
from data.datasets import DeepfakeDataset
from data.collate import deepfake_collate_fn
from utils.checkpointing import save_checkpoint, load_checkpoint
from metrics.detection import DetectionMetrics
from utils.logging_utils import Logger


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: ExplanationLoss,
    temporal_criterion: TemporalConsistencyLoss,
    device: torch.device,
    epoch: int,
    lambda1: float,
    lambda2: float,
    grad_accum_steps: int,
    scaler=None,
) -> dict:
    model.train()
    running_loss = 0.0
    running_cls = 0.0
    running_exp = 0.0
    running_temp = 0.0
    global_step = epoch * len(dataloader)

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f"Real Train {epoch}")

    for batch_idx, batch in enumerate(pbar):
        step = global_step + batch_idx
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)
        masks = batch.get("mask", None)
        if masks is not None:
            masks = masks.to(device).unsqueeze(1)

        B, T, C, H, W = frames.shape

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(frames)
            logits = output.logit
            M_t = output.M_t
            low_level = output.low_level
            attn = M_t.mean(dim=1).unsqueeze(1)

            loss_dict = criterion(logits, labels, attn, masks)
            L_cls = loss_dict["L_cls"]
            L_exp = loss_dict["L_exp"]

            # FIXED: TemporalConsistencyLoss takes (M_t, low_level)
            L_temp = temporal_criterion(M_t, low_level)

            total_loss = L_cls + lambda1 * L_exp + lambda2 * L_temp
            total_loss = total_loss / grad_accum_steps

        if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss.item() > 100.0 / grad_accum_steps:
            print(f"[SKIP] Bad loss {total_loss.item():.4f} at step {step}")
            continue

        if scaler:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        running_loss += total_loss.item() * grad_accum_steps
        running_cls += L_cls.item()
        running_exp += L_exp.item()
        running_temp += L_temp.item()

        pbar.set_postfix({
            "loss": f"{total_loss.item() * grad_accum_steps:.4f}",
            "L_cls": f"{L_cls.item():.4f}",
            "L_exp": f"{L_exp.item():.4f}",
            "L_temp": f"{L_temp.item():.4f}",
        })

    n = len(dataloader)
    return {
        "loss": running_loss / n,
        "L_cls": running_cls / n,
        "L_exp": running_exp / n,
        "L_temp": running_temp / n,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: ExplanationLoss,
    temporal_criterion: TemporalConsistencyLoss,
    device: torch.device,
) -> dict:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Real Val"):
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)
        masks = batch.get("mask", None)
        if masks is not None:
            masks = masks.to(device).unsqueeze(1)

        output = model(frames)
        logits = output.logit
        M_t = output.M_t
        low_level = output.low_level
        attn = M_t.mean(dim=1).unsqueeze(1)

        loss_dict = criterion(logits, labels, attn, masks)
        L_temp = temporal_criterion(M_t, low_level)
        total_loss = loss_dict["loss"] + L_temp

        running_loss += total_loss.item()
        probs = torch.sigmoid(logits)
        all_preds.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    metrics = DetectionMetrics()
    results = metrics.compute(all_preds, all_labels)

    return {
        "loss": running_loss / len(dataloader),
        "auc": results.get("auc_roc", 0.0),
        "ap": results.get("auc_pr", 0.0),
        "f1": results.get("f1", 0.0),
    }


def main(cfg=None):
    if cfg is None:
        args = parse_args()
        cfg = EAHNConfig.from_args(args)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[Phase 2] Real-data Fine-tuning on {device}")
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Build model
    model = EAHN(config=cfg).to(device)

    # Load synthetic checkpoint if present
    synth_ckpt = os.path.join(cfg.output_dir, "synthetic_final.pth")
    if os.path.exists(synth_ckpt):
        print(f"Loading synthetic checkpoint: {synth_ckpt}")
        ckpt = load_checkpoint(synth_ckpt, model, strict=False)
        print(f"  -> Resumed from epoch {ckpt.get('epoch', '?')}")
    elif cfg.resume_checkpoint:
        print(f"Loading checkpoint: {cfg.resume_checkpoint}")
        load_checkpoint(cfg.resume_checkpoint, model, strict=False)

    # Progressive backbone freezing
    if cfg.freeze_backbone and hasattr(model.spatial_stream, "set_frozen"):
        model.spatial_stream.set_frozen(True)
        print("Backbone frozen for initial epochs")

    # Criteria
    criterion = ExplanationLoss(
        diversity_weight=cfg.attn_diversity_weight,
        entropy_weight=cfg.alpha,
        class_sep_weight=cfg.class_sep_weight,
    )
    temporal_criterion = TemporalConsistencyLoss(gamma=cfg.gamma)

    # Datasets
    train_dataset = DeepfakeDataset(cfg, "train", cfg.dataset_name)
    val_dataset = DeepfakeDataset(cfg, "val", cfg.dataset_name)

    # Class-imbalance sampler
    labels = [s["label"] for s in train_dataset.samples]
    class_counts = np.bincount(labels, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=deepfake_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=deepfake_collate_fn,
    )

    # Optimizer with optional backbone LR ratio
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if "spatial_stream.backbone" in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    if backbone_params and cfg.backbone_lr_ratio != 1.0:
        param_groups = [
            {"params": backbone_params, "lr": cfg.lr * cfg.backbone_lr_ratio},
            {"params": head_params, "lr": cfg.lr},
        ]
    else:
        param_groups = model.parameters()

    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if cfg.mixed_precision and device.type == "cuda" else None
    logger = Logger(os.path.join(cfg.output_dir, "logs_real"))

    best_auc = 0.0
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        # Progressive unfreeze
        if (
            cfg.freeze_backbone
            and epoch == cfg.unfreeze_backbone_epoch
            and hasattr(model.spatial_stream, "set_frozen")
        ):
            model.spatial_stream.set_frozen(False)
            print(f"  -> Unfroze backbone at epoch {epoch}")
            for param_group in optimizer.param_groups:
                if param_group["params"] == backbone_params:
                    param_group["lr"] *= 0.1

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, temporal_criterion,
            device, epoch, cfg.lambda1, cfg.lambda2, cfg.grad_accum_steps, scaler,
        )
        print(f"[Epoch {epoch}] Train Loss: {train_metrics['loss']:.4f} | "
              f"L_cls: {train_metrics['L_cls']:.4f} | L_exp: {train_metrics['L_exp']:.4f} | "
              f"L_temp: {train_metrics['L_temp']:.4f}")
        logger.log_scalars("real/train", train_metrics, epoch)

        val_metrics = validate(model, val_loader, criterion, temporal_criterion, device)
        print(f"[Epoch {epoch}] Val Loss: {val_metrics['loss']:.4f} | "
              f"AUC: {val_metrics['auc']:.4f} | F1: {val_metrics['f1']:.4f}")
        logger.log_scalars("real/val", val_metrics, epoch)

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            patience_counter = 0
            save_checkpoint(
                model, optimizer, None, epoch, best_auc, cfg,
                os.path.join(cfg.output_dir, "best_model.pth"),
            )
            print(f"  -> Saved best model (AUC={best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"  -> Early stopping at epoch {epoch}")
                break

    save_checkpoint(
        model, optimizer, None, cfg.epochs, best_auc, cfg,
        os.path.join(cfg.output_dir, "real_final.pth"),
    )
    print("[Phase 2] Complete.")
    logger.close()


if __name__ == "__main__":
    main()