import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import EAHNConfig
from models.eahn import EAHN
from losses.explanation import ExplanationLoss
from losses.temporal import TemporalConsistencyLoss
from data.datasets import DeepfakeDataset
from utils.checkpointing import save_checkpoint, load_checkpoint
from metrics.detection import DetectionMetrics


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
    warmup_steps: int = 500,
) -> dict:
    model.train()
    running_loss = 0.0
    running_cls = 0.0
    running_exp = 0.0
    running_temp = 0.0
    global_step = epoch * len(dataloader)

    pbar = tqdm(dataloader, desc=f"Real Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        step = global_step + batch_idx

        frames = batch["frames"].to(device)   # (B, T, C, H, W)
        labels = batch["label"].to(device)    # (B,)

        B, T, C, H, W = frames.shape
        frames_flat = frames.view(B * T, C, H, W)

        optimizer.zero_grad()

        outputs = model(frames_flat)
        logits = outputs["logits"].view(B, T, -1).mean(dim=1).squeeze(-1)  # (B,)
        attn = outputs["attention"].view(B, T, 1, H, W).mean(dim=1)        # (B, 1, H, W)

        # Warmup for L_exp
        lambda1_eff = lambda1 * min(1.0, step / warmup_steps)

        loss_dict = criterion(logits, labels, attn, None)
        L_cls_exp = loss_dict["loss"]

        # Temporal consistency across frames
        attn_temporal = outputs["attention"].view(B, T, 1, H, W)
        L_temp = temporal_criterion(attn_temporal)

        total_loss = L_cls_exp + lambda2 * L_temp

        if torch.isnan(total_loss) or total_loss.item() > 100.0:
            print(f"[SKIP] Bad loss {total_loss.item():.4f} at step {step}")
            optimizer.zero_grad()
            continue

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += total_loss.item()
        running_cls += loss_dict["L_cls"].item()
        running_exp += loss_dict["L_exp"].item()
        running_temp += L_temp.item()

        pbar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "L_cls": f"{loss_dict['L_cls'].item():.4f}",
            "L_exp": f"{loss_dict['L_exp'].item():.4f}",
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
def validate(model, dataloader, criterion, temporal_criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="Val"):
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)
        B, T, C, H, W = frames.shape
        frames_flat = frames.view(B * T, C, H, W)

        outputs = model(frames_flat)
        logits = outputs["logits"].view(B, T, -1).mean(dim=1).squeeze(-1)
        attn = outputs["attention"].view(B, T, 1, H, W).mean(dim=1)

        loss_dict = criterion(logits, labels, attn, None)
        L_temp = temporal_criterion(outputs["attention"].view(B, T, 1, H, W))
        total_loss = loss_dict["loss"] + L_temp

        running_loss += total_loss.item()
        probs = torch.sigmoid(logits)
        all_preds.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    metrics = DetectionMetrics()
    results = metrics.compute(all_preds, all_labels)

    return {
        "loss": running_loss / len(dataloader),
        "auc": results.get("auc", 0.0),
        "accuracy": results.get("accuracy", 0.0),
    }


def main(cfg: Optional[EAHNConfig] = None):
    if cfg is None:
        cfg = EAHNConfig()

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[Phase 2] Real-data Fine-tuning on {device}")

    # Load synthetic checkpoint
    synth_ckpt = os.path.join(cfg.checkpoint_dir, "synthetic_final.pth")
    if not os.path.exists(synth_ckpt):
        raise FileNotFoundError(f"Synthetic checkpoint not found: {synth_ckpt}")

    model = EAHN(
        num_classes=cfg.num_classes,
        backbone=cfg.backbone,
        temporal_dim=cfg.temporal_dim,
    ).to(device)

    # FIXED: PyTorch 2.6 weights_only=False
    ckpt = torch.load(synth_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"Loaded synthetic checkpoint from {synth_ckpt}")

    # FIXED: diversity_weight 8.0 -> 2.0
    criterion = ExplanationLoss(
        diversity_weight=cfg.attn_diversity_weight,
        entropy_weight=cfg.attn_entropy_weight,
        class_sep_weight=cfg.attn_class_sep_weight,
    )
    temporal_criterion = TemporalConsistencyLoss()

    # Build FF++ dataset
    train_dataset = DeepfakeDataset(
        root=cfg.ffpp_root,
        split="train",
        frames_per_clip=cfg.frames_per_clip,
    )
    val_dataset = DeepfakeDataset(
        root=cfg.ffpp_root,
        split="val",
        frames_per_clip=cfg.frames_per_clip,
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
        lr=cfg.lr_phase2,
        weight_decay=cfg.weight_decay,
    )

    best_auc = 0.0
    for epoch in range(1, cfg.epochs_phase2 + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, temporal_criterion,
            device, epoch, lambda1=cfg.lambda1, lambda2=cfg.lambda2,
        )
        print(f"[Epoch {epoch}] Train Loss: {train_metrics['loss']:.4f} | "
              f"L_cls: {train_metrics['L_cls']:.4f} | L_exp: {train_metrics['L_exp']:.4f} | "
              f"L_temp: {train_metrics['L_temp']:.4f}")

        val_metrics = validate(model, val_loader, criterion, temporal_criterion, device)
        print(f"[Epoch {epoch}] Val Loss: {val_metrics['loss']:.4f} | AUC: {val_metrics['auc']:.4f}")

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            save_checkpoint(
                model, optimizer, epoch,
                filepath=os.path.join(cfg.checkpoint_dir, "best_real.pth"),
            )

    save_checkpoint(
        model, optimizer, cfg.epochs_phase2,
        filepath=os.path.join(cfg.checkpoint_dir, "real_final.pth"),
    )
    print("[Phase 2] Complete.")


if __name__ == "__main__":
    main()