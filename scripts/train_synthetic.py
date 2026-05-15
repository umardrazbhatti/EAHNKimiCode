"""
scripts/train_synthetic.py — Phase 1: Pre-train on synthetic data with mask supervision.
Run this BEFORE train_real.py.

FIXES:
- sys.path manipulation for standalone execution
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to prevent fragmentation OOM
- Auto batch_size 4→2 / grad_accum 4→8 for Kaggle T4 safety
- torch.cuda.empty_cache() between epochs
- Explicit del out/loss after backward
"""

import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import math
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from config import EAHNConfig, parse_args
from data.synthetic_generator import SyntheticDataset
from data.collate import deepfake_collate_fn
from models.eahn import EAHN
from losses.classification import build_classification_loss
from losses.explanation import ExplanationLoss
from losses.temporal import TemporalConsistencyLoss
from utils.checkpointing import save_checkpoint
from utils.logging_utils import Logger


def main(config: EAHNConfig):
    device = torch.device(config.device)
    print(f"[Synthetic Phase] Device: {device}")
    os.makedirs(config.output_dir, exist_ok=True)

    # ── Memory safeguard for Kaggle T4 (16 GB) ───────────────────────────
    effective_batch = config.batch_size * config.grad_accum_steps
    if config.batch_size >= 4 and config.num_frames >= 16:
        config.batch_size = 2
        config.grad_accum_steps = max(1, effective_batch // config.batch_size)
        print(f"[MEMORY] Auto-adjusted for T4: batch_size={config.batch_size}, "
              f"grad_accum={config.grad_accum_steps} (effective={effective_batch})")

    # ── Dataset ─────────────────────────────────────────────────────────────
    synth_ds = SyntheticDataset(
        source_image_dir="/kaggle/working/synth_source",
        num_frames=config.num_frames,
        frame_size=config.frame_size,
        length=20000,  # 10k real + 10k fake
    )
    n_train = int(0.9 * len(synth_ds))
    n_val = len(synth_ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(synth_ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, collate_fn=deepfake_collate_fn,
        drop_last=True, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, collate_fn=deepfake_collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Synthetic train: {len(train_ds)} | val: {len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = EAHN(config).to(device)

    if config.freeze_backbone:
        model.spatial_stream.set_frozen(False)
        print("[Backbone] Unfrozen from start (synthetic phase)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return (epoch + 1) / (config.warmup_epochs + 1)
        progress = (epoch - config.warmup_epochs) / max(config.epochs - config.warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = config.mixed_precision and device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 7
    scaler = GradScaler("cuda") if use_amp else None
    logger = Logger(config.output_dir)

    # ── Losses ──────────────────────────────────────────────────────────────
    cls_loss_fn = build_classification_loss(config)
    exp_loss_fn = ExplanationLoss(
        alpha=config.alpha,
        beta=config.beta,
        diversity_weight=config.attn_diversity_weight,
        class_sep_weight=config.class_sep_weight,
    )
    temp_loss_fn = TemporalConsistencyLoss(gamma=config.gamma)

    ckpt_path = os.path.join(config.output_dir, "synthetic_pretrained.pth")
    best_val = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────
    import contextlib
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0

        target_temp = max(0.5, 2.0 * math.exp(-epoch / 2.5))
        model.set_attention_temp(target_temp)

        for batch_idx, batch in enumerate(train_loader):
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"].to(device)
            has_mask = batch["has_mask"].to(device)

            ctx = autocast("cuda") if use_amp else contextlib.nullcontext()
            with ctx:
                out = model(frames)

                l_cls = cls_loss_fn(out.logit, labels)
                exp_out = exp_loss_fn(out.M_t, masks, has_mask, labels=labels)
                l_exp = exp_out.loss
                l_temp = temp_loss_fn(out.M_t, out.low_level)

                l_total = l_cls + config.lambda1 * l_exp + config.lambda2 * l_temp

                if torch.isnan(l_total) or torch.isinf(l_total):
                    print(f"[WARN] NaN loss at E{epoch+1}B{batch_idx}. Skipping.")
                    optimizer.zero_grad()
                    del out, l_total, l_cls, l_exp, l_temp
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
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

            running_loss += l_total.item()

            if batch_idx % 100 == 0:
                print(f"[E{epoch+1} B{batch_idx}] L_total={l_total.item():.3f} "
                      f"L_cls={l_cls.item():.3f} L_exp={l_exp.item():.3f} "
                      f"tau={target_temp:.2f} M_std={out.M_t.std().item():.3f}")

            # Explicit cleanup to prevent CUDA fragmentation
            del out, loss, l_total, l_cls, l_exp, l_temp
            if device.type == 'cuda' and batch_idx % 50 == 0:
                torch.cuda.empty_cache()

        # Step leftover gradients
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
        avg_train_loss = running_loss / max(len(train_loader), 1)

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                frames = batch["frames"].to(device)
                labels = batch["label"].to(device)
                masks = batch["mask"].to(device)
                has_mask = batch["has_mask"].to(device)
                out = model(frames)
                l_cls = cls_loss_fn(out.logit, labels)
                exp_out = exp_loss_fn(out.M_t, masks, has_mask, labels=labels)
                l_total = l_cls + config.lambda1 * exp_out.loss + config.lambda2 * temp_loss_fn(out.M_t, out.low_level)
                val_loss += l_total.item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        print(f"[Epoch {epoch+1}/{config.epochs}] TrainLoss: {avg_train_loss:.4f} | "
              f"ValLoss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_val_loss < best_val:
            best_val = avg_val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, best_val, config, ckpt_path)
            print(f"--> Best synthetic checkpoint saved")

        # Free memory between epochs
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    logger.close()
    print(f"\nSynthetic pre-training complete. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    args = parse_args()
    config = EAHNConfig.from_args(args)
    main(config)