"""
config.py — Two-phase defaults: synthetic pre-training + FF++ fine-tuning.
"""

import argparse
import warnings
import torch
from dataclasses import dataclass
from typing import Literal


@dataclass
class EAHNConfig:
    # ── Paths ─────────────────────────────────────────────────────────────
    data_root: str = "/kaggle/input/"
    output_dir: str = "/kaggle/working/outputs/"
    cache_dir: str = "/kaggle/working/.face_cache/"
    resume_checkpoint: str = ""          # For FF++ phase: path to synthetic_pretrained.pth

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset_name: Literal["synthetic", "ff++", "celeb_df", "dfdc"] = "ff++"
    dataset_compression: str = "c23"
    num_frames: int = 16
    frame_size: int = 224
    train_split: float = 0.8
    val_split: float = 0.1

    # ── Model ─────────────────────────────────────────────────────────────
    backbone: str = "efficientnet_b4"
    backbone_pretrained: bool = True
    transformer_layers: int = 4
    transformer_heads: int = 8
    d_model: int = 256
    dropout: float = 0.1

    # ── Loss weights ──────────────────────────────────────────────────────
    # PHASE 1 (Synthetic): lambda1=1.0  (strong mask supervision)
    # PHASE 2 (FF++):     lambda1=0.05 (weakly-supervised nudge)
    lambda1: float = 0.05
    lambda2: float = 0.02
    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 0.1
    attn_temp_init: float = 0.0
    attn_diversity_weight: float = 8.0
    cls_dropout_p: float = 0.0
    lambda_grad_align: float = 0.0       # REMOVED
    label_smoothing: float = 0.02
    class_sep_weight: float = 0.5

    # ── Backbone ──────────────────────────────────────────────────────────
    freeze_backbone: bool = False        # CHANGED: unfreeze immediately
    unfreeze_backbone_epoch: int = 0
    backbone_lr_ratio: float = 1.0       # CHANGED: backbone gets same LR as head

    # ── Classification loss ───────────────────────────────────────────────
    cls_loss_type: str = "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # ── Training ──────────────────────────────────────────────────────────
    epochs: int = 15
    batch_size: int = 4
    grad_accum_steps: int = 4
    lr: float = 5e-4                     # CHANGED: 5e-4 (backbone must learn)
    weight_decay: float = 1e-2
    mixed_precision: bool = True
    num_workers: int = 0
    warmup_epochs: int = 2
    patience: int = 3

    # ── Evaluation ────────────────────────────────────────────────────────
    eval_after_train: bool = True
    save_heatmaps: bool = True
    heatmap_samples: int = 20

    # ── Device ────────────────────────────────────────────────────────────
    device: str = "auto"

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu":
                warnings.warn("No GPU. Using CPU-safe overrides.")
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
    parser = argparse.ArgumentParser(description="EAHN Training")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None, choices=["synthetic", "ff++", "celeb_df", "dfdc"])
    parser.add_argument("--dataset_compression", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone_lr_ratio", type=float, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
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
    parser.add_argument("--cls_loss_type", type=str, default=None, choices=["bce", "focal"])
    parser.add_argument("--focal_alpha", type=float, default=None)
    parser.add_argument("--focal_gamma", type=float, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--class_sep_weight", type=float, default=None)
    return parser.parse_args()