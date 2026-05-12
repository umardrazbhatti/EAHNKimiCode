"""
config.py — single source of truth for all EAHN hyperparameters.
CLI overrides via argparse; no hardcoded paths anywhere else.

FIX: attn_temp_init changed from 1.386 (τ=4.0) to 0.0 (τ=1.0).
A temperature of 4.0 forces uniform attention maps (M_t ≈ 1/49 everywhere),
which creates a flat loss landscape with zero gradients for the temperature
parameter — the "uniform attention trap" that locked τ at 4.0 forever.
"""

import argparse
import warnings
import torch
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EAHNConfig:
    # ── Paths ─────────────────────────────────────────────────────────────────
    data_root: str = "/kaggle/input/"
    output_dir: str = "/kaggle/working/outputs/"
    cache_dir: str = "/kaggle/working/.face_cache/"
    resume_checkpoint: str = ""

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_name: Literal["synthetic", "ff++", "celeb_df", "dfdc"] = "ff++"
    dataset_compression: str = "c23"
    num_frames: int = 16
    frame_size: int = 224
    train_split: float = 0.8
    val_split: float = 0.1

    # ── Model ─────────────────────────────────────────────────────────────────
    backbone: str = "efficientnet_b4"
    backbone_pretrained: bool = True
    transformer_layers: int = 4
    transformer_heads: int = 8
    d_model: int = 256
    dropout: float = 0.1

    # ── Loss weights ──────────────────────────────────────────────────────────
    lambda1: float = 1.0          # L_exp weight
    lambda2: float = 0.05         # L_temp weight (reduced to avoid freezing heatmaps)
    alpha: float = 0.5            # entropy weight — INCREASED to prevent one-hot collapse
    beta: float = 0.5             # TV weight in weak supervision
    gamma: float = 0.1            # gate decay rate in L_temp
    attn_temp_init: float = 0.0   # FIX: was 1.386 (τ=4.0). Now 0.0 → τ=1.0 (peaked, trainable)
    attn_diversity_weight: float = 2.5
    cls_dropout_p: float = 0.0    # DISABLED — caused train/test distribution mismatch
    lambda_grad_align: float = 0.1  # gradient-alignment loss weight
    label_smoothing: float = 0.05   # label smoothing for classification

    # ── Backbone freezing ─────────────────────────────────────────────────────
    freeze_backbone: bool = True
    unfreeze_backbone_epoch: int = 3

    # ── Classification loss ───────────────────────────────────────────────────
    cls_loss_type: str = "focal"  # "bce" | "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 50
    batch_size: int = 4
    grad_accum_steps: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-2
    mixed_precision: bool = True
    num_workers: int = 0

    # ── Evaluation / Visualisation ────────────────────────────────────────────
    eval_after_train: bool = True
    save_heatmaps: bool = True
    heatmap_samples: int = 20

    # ── Device ────────────────────────────────────────────────────────────────
    device: str = "auto"

    def __post_init__(self):
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                warnings.warn("No GPU found. Switching to CPU with reduced settings.")
                self._apply_cpu_safe_overrides()

    def _apply_cpu_safe_overrides(self):
        self.num_frames = 4
        self.transformer_layers = 2
        self.transformer_heads = 2
        self.batch_size = 2
        self.mixed_precision = False
        self.num_workers = 0
        if "efficientnet_b4" in self.backbone:
            self.backbone = "efficientnet_b0"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "EAHNConfig":
        cfg = cls()
        for key, val in vars(args).items():
            if hasattr(cfg, key) and val is not None:
                setattr(cfg, key, val)
        return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EAHN Training and Evaluation")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None,
                        choices=["synthetic", "ff++", "celeb_df", "dfdc"])
    parser.add_argument("--dataset_compression", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda1", type=float, default=None)
    parser.add_argument("--lambda2", type=float, default=None)
    parser.add_argument("--heatmap_samples", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--eval_after_train", action="store_true", default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--attn_temp_init", type=float, default=None)
    parser.add_argument("--attn_diversity_weight", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--cls_dropout_p", type=float, default=None)
    parser.add_argument("--lambda_grad_align", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--freeze_backbone", action="store_true", default=None)
    parser.add_argument("--unfreeze_backbone_epoch", type=int, default=None)
    parser.add_argument("--cls_loss_type", type=str, default=None,
                        choices=["bce", "focal"])
    parser.add_argument("--focal_alpha", type=float, default=None)
    parser.add_argument("--focal_gamma", type=float, default=None)
    return parser.parse_args()
