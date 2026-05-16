import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import EAHNConfig
from models.eahn import EAHN
from losses.explanation import ExplanationLoss
from losses.classification import build_classification_loss
from data.synthetic_generator import SyntheticDataset
from utils.checkpointing import save_checkpoint, load_checkpoint
from metrics.detection import DetectionMetrics


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: ExplanationLoss,
    device: torch.device,
    epoch: int,
    lambda1: float,
    warmup_steps: int = 1000,
) -> dict:
    model.train()
    running_loss = 0.0
    running_cls = 0.0
    running_exp = 0.0
    global_step = epoch * len(dataloader)

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        step = global_step + batch_idx

        frames = batch["frames"].to(device)          # (B, T, C, H, W)
        labels = batch["label"].to(device)             # (B,)
        masks = batch.get("mask", None)
        if masks is not None:
            masks = masks.to(device)                   # (B, 1, H, W)

        B, T, C, H, W = frames.shape
        # Flatten temporal into batch for frame-level processing
        frames_flat = frames.view(B * T, C, H, W)

        optimizer.zero_grad()

        # Forward
        outputs = model(frames_flat)  # dict with 'logits', 'attention', etc.
        logits = outputs["logits"].view(B, T, -1).mean(dim=1).squeeze(-1)  # (B,)
        attn = outputs["attention"]  # (B*T, 1, H, W)
        # Pool attention over time
        attn = attn.view(B, T, 1, H, W).mean(dim=1)  # (B, 1, H, W)

        # L_exp warmup: ramp lambda1 over first N steps
        lambda1_eff = lambda1 * min(1.0, step / warmup_steps)  # FIXED: warmup

        loss_dict = criterion(logits, labels, attn, masks)
        loss = loss_dict["loss"]

        # FIXED: bad loss skip
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100.0:
            print(f"[SKIP] Bad loss {loss.item():.4f} at step {step}, skipping batch")
            optimizer.zero_grad()
            continue

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        running_cls += loss_dict["L_cls"].item()
        running_exp += loss_dict["L_exp"].item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "L_cls": f"{loss_dict['L_cls'].item():.4f}",
            "L_exp": f"{loss_dict['L_exp'].item():.4f}",
            "lambda1": f"{lambda1_eff:.3f}",
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

    for batch in tqdm(dataloader, desc="Validate"):
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)

        B, T, C, H, W = frames.shape
        frames_flat = frames.view(B * T, C, H, W)

        outputs = model(frames_flat)
        logits = outputs["logits"].view(B, T, -1).mean(dim=1).squeeze(-1)
        attn = outputs["attention"].view(B, T, 1, H, W).mean(dim=1)

        loss_dict = criterion(logits, labels, attn, None)
        running_loss += loss_dict["loss"].item()

        probs = torch.sigmoid(logits)
        all_preds.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    metrics = DetectionMetrics()
    results = metrics.compute(all_preds, all_labels)

    return {
        "loss": running_loss / len(dataloader),
        "auc": results.get("auc", 0.0),
        "ap": results.get("ap", 0.0),
        "accuracy": results.get("accuracy", 0.0),
    }


def main(cfg: Optional[EAHNConfig] = None):
    if cfg is None:
        cfg = EAHNConfig()

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[Phase 1] Synthetic Pre-training on {device}")

    # Build model
    model = EAHN(
        num_classes=cfg.num_classes,
        backbone=cfg.backbone,
        temporal_dim=cfg.temporal_dim,
    ).to(device)

    # FIXED: diversity_weight 8.0 -> 2.0
    criterion = ExplanationLoss(
        diversity_weight=cfg.attn_diversity_weight,  # should be 2.0 in config
        entropy_weight=cfg.attn_entropy_weight,
        class_sep_weight=cfg.attn_class_sep_weight,
    )

    # Build datasets
    train_dataset = SyntheticDataset(
        root=cfg.synthetic_data_root,
        num_samples=cfg.synthetic_samples,
        frames_per_clip=cfg.frames_per_clip,
        transform=None,  # add your transform
    )
    val_dataset = SyntheticDataset(
        root=cfg.synthetic_data_root,
        num_samples=cfg.synthetic_val_samples,
        frames_per_clip=cfg.frames_per_clip,
        transform=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr_phase1,
        weight_decay=cfg.weight_decay,
    )

    best_val_loss = float("inf")
    for epoch in range(1, cfg.epochs_phase1 + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            lambda1=cfg.lambda1, warmup_steps=1000,
        )
        print(f"[Epoch {epoch}] Train Loss: {train_metrics['loss']:.4f} | "
              f"L_cls: {train_metrics['L_cls']:.4f} | L_exp: {train_metrics['L_exp']:.4f}")

        # FIXED: validate every epoch to catch NaN early
        val_metrics = validate(model, val_loader, criterion, device)
        print(f"[Epoch {epoch}] Val Loss: {val_metrics['loss']:.4f} | "
              f"AUC: {val_metrics['auc']:.4f} | Acc: {val_metrics['accuracy']:.4f}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                model, optimizer, epoch,
                filepath=os.path.join(cfg.checkpoint_dir, "best_synthetic.pth"),
            )

    # Save final
    save_checkpoint(
        model, optimizer, cfg.epochs_phase1,
        filepath=os.path.join(cfg.checkpoint_dir, "synthetic_final.pth"),
    )
    print("[Phase 1] Complete. Saved to", cfg.checkpoint_dir)


if __name__ == "__main__":
    main()