"""
scripts/train_synthetic.py — Phase 1: Synthetic pre-training.
FIXED: Now matches actual repo APIs (EAHN(config), DeepfakeDataset, etc.)
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import EAHNConfig, parse_args
from models.eahn import EAHN
from losses.explanation import ExplanationLoss
from data.datasets import DeepfakeDataset
from data.collate import deepfake_collate_fn
from utils.checkpointing import save_checkpoint
from metrics.detection import DetectionMetrics
from utils.logging_utils import Logger


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: ExplanationLoss,
    device: torch.device,
    epoch: int,
    lambda1: float,
    grad_accum_steps: int,
    scaler=None,
) -> dict:
    model.train()
    running_loss = 0.0
    running_cls = 0.0
    running_exp = 0.0
    global_step = epoch * len(dataloader)

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f"Syn Train {epoch}")

    for batch_idx, batch in enumerate(pbar):
        step = global_step + batch_idx
        frames = batch["frames"].to(device)          # (B, T, C, H, W)
        labels = batch["label"].to(device)           # (B,)
        masks = batch.get("mask", None)
        if masks is not None:
            masks = masks.to(device).unsqueeze(1)    # (B, 1, H, W)

        B, T, C, H, W = frames.shape

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(frames)
            logits = output.logit                      # (B,)
            M_t = output.M_t                           # (B, T, feat_h, feat_w)
            attn = M_t.mean(dim=1).unsqueeze(1)        # (B, 1, feat_h, feat_w)

            loss_dict = criterion(logits, labels, attn, masks)
            # Apply lambda1 to explanation portion; keep L_cls raw
            total_loss = loss_dict["L_cls"] + lambda1 * loss_dict["L_exp"]
            total_loss = total_loss / grad_accum_steps

        # NaN / explosion guard
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
        running_cls += loss_dict["L_cls"].item()
        running_exp += loss_dict["L_exp"].item()

        pbar.set_postfix({
            "loss": f"{total_loss.item() * grad_accum_steps:.4f}",
            "L_cls": f"{loss_dict['L_cls'].item():.4f}",
            "L_exp": f"{loss_dict['L_exp'].item():.4f}",
        })

    n = len(dataloader)
    return {
        "loss": running_loss / n,
        "L_cls": running_cls / n,
        "L_exp": running_exp / n,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: ExplanationLoss,
    device: torch.device,
) -> dict:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Syn Val"):
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)
        masks = batch.get("mask", None)
        if masks is not None:
            masks = masks.to(device).unsqueeze(1)

        output = model(frames)
        logits = output.logit
        M_t = output.M_t
        attn = M_t.mean(dim=1).unsqueeze(1)

        loss_dict = criterion(logits, labels, attn, masks)
        running_loss += loss_dict["loss"].item()

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
    print(f"[Phase 1] Synthetic Pre-training on {device}")
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Build model — API is EAHN(config=cfg), NOT EAHN(num_classes=...)
    model = EAHN(config=cfg).to(device)

    # Explanation loss (alpha mapped to entropy weight)
    criterion = ExplanationLoss(
        diversity_weight=cfg.attn_diversity_weight,
        entropy_weight=cfg.alpha,
        class_sep_weight=cfg.class_sep_weight,
    )

    # Synthetic dataset — generates on-the-fly, no image files needed
    train_dataset = DeepfakeDataset(cfg, "train", "synthetic")
    val_dataset = DeepfakeDataset(cfg, "val", "synthetic")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler() if cfg.mixed_precision and device.type == "cuda" else None
    logger = Logger(os.path.join(cfg.output_dir, "logs_synthetic"))

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            cfg.lambda1, cfg.grad_accum_steps, scaler,
        )
        print(f"[Epoch {epoch}] Train Loss: {train_metrics['loss']:.4f} | "
              f"L_cls: {train_metrics['L_cls']:.4f} | L_exp: {train_metrics['L_exp']:.4f}")
        logger.log_scalars("synthetic/train", train_metrics, epoch)

        val_metrics = validate(model, val_loader, criterion, device)
        print(f"[Epoch {epoch}] Val Loss: {val_metrics['loss']:.4f} | "
              f"AUC: {val_metrics['auc']:.4f} | F1: {val_metrics['f1']:.4f}")
        logger.log_scalars("synthetic/val", val_metrics, epoch)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            save_checkpoint(
                model, optimizer, None, epoch, val_metrics["auc"], cfg,
                os.path.join(cfg.output_dir, "best_synthetic.pth"),
            )
            print(f"  -> Saved best synthetic checkpoint (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"  -> Early stopping at epoch {epoch}")
                break

    save_checkpoint(
        model, optimizer, None, cfg.epochs, 0.0, cfg,
        os.path.join(cfg.output_dir, "synthetic_final.pth"),
    )
    print("[Phase 1] Complete. Saved to", cfg.output_dir)
    logger.close()


if __name__ == "__main__":
    main()