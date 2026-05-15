"""
models/eahn.py — EAHN with non-linear classifier head and temperature annealing.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from config import EAHNConfig
from models.spatial_stream import SpatialStream
from models.temporal_stream import TemporalStream
from models.cross_attention import CrossAttentionFusion


@dataclass
class EAHNOutput:
    logit: torch.Tensor
    prob: torch.Tensor
    M_t: torch.Tensor
    M_t_up: torch.Tensor
    S: torch.Tensor
    low_level: torch.Tensor
    attn_pool: torch.Tensor


class EAHN(nn.Module):
    def __init__(self, config: EAHNConfig):
        super().__init__()
        self.config = config
        d = config.d_model

        self.spatial_stream = SpatialStream(
            backbone_name=config.backbone,
            pretrained=config.backbone_pretrained,
            d_model=d,
            freeze_backbone=config.freeze_backbone,
        )

        dummy = torch.zeros(1, 3, config.frame_size, config.frame_size)
        with torch.no_grad():
            dummy_tokens = self.spatial_stream(dummy)
            N = dummy_tokens.shape[1]
            self.N = N
            self.feat_h = self.spatial_stream.feat_h
            self.feat_w = self.spatial_stream.feat_w

        max_seq = config.num_frames * N + 1
        self.temporal_stream = TemporalStream(
            d_model=d,
            num_heads=config.transformer_heads,
            num_layers=config.transformer_layers,
            dropout=config.dropout,
            max_seq_len=max_seq,
        )

        self.cross_attention = CrossAttentionFusion(
            d_model=d,
            num_heads=config.transformer_heads,
            temp_init=math.log(2.0),  # Start warmer (τ=2.0) for exploration
        )

        # MLP classifier: allows non-linear interaction of cls_out + attn_pool
        self.classifier = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def set_attention_temp(self, temp: float):
        """External temperature control (annealing)."""
        if hasattr(self.cross_attention, 'log_temp'):
            with torch.no_grad():
                self.cross_attention.log_temp.fill_(math.log(max(temp, 0.1)))

    def forward(self, frames: torch.Tensor) -> EAHNOutput:
        B, T, C, H, W = frames.shape
        frames_flat = frames.reshape(B * T, C, H, W)

        spatial_tokens = self.spatial_stream(frames_flat)
        low_feat = self.spatial_stream.low_level_features()

        N = spatial_tokens.shape[1]
        d = self.config.d_model
        C_low, Hl, Wl = low_feat.shape[1], low_feat.shape[2], low_feat.shape[3]

        spatial_tokens = spatial_tokens.view(B, T, N, d)
        low_level = low_feat.view(B, T, C_low, Hl, Wl)

        Q, cls_out = self.temporal_stream(spatial_tokens.reshape(B, T * N, d))
        Q = Q.reshape(B, T, N, d)

        # Clamp temperature to prevent explosion
        if hasattr(self.cross_attention, 'log_temp'):
            with torch.no_grad():
                self.cross_attention.log_temp.clamp_(math.log(0.3), math.log(4.0))

        M_t, attn_pool = self.cross_attention(Q, spatial_tokens)

        M_t_up = F.interpolate(
            M_t.reshape(B * T, 1, self.feat_h, self.feat_w),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).reshape(B, T, H, W)

        final_feat = torch.cat([cls_out, attn_pool], dim=-1)  # (B, 2d)
        logit = self.classifier(final_feat).squeeze(-1)
        prob = torch.sigmoid(logit)

        return EAHNOutput(
            logit=logit, prob=prob,
            M_t=M_t, M_t_up=M_t_up,
            S=spatial_tokens, low_level=low_level,
            attn_pool=attn_pool,
        )

    def compute_gradient_saliency(self, frames: torch.Tensor):
        B, T, C, H, W = frames.shape
        param_states = [p.requires_grad for p in self.parameters()]
        try:
            for p in self.parameters():
                p.requires_grad = False
            frames_in = frames.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                out = self(frames_in)
                score = out.logit.sum() if out.logit.ndim == 1 else out.logit[:, 1].sum()
                grad = torch.autograd.grad(score, frames_in, create_graph=False, retain_graph=False)[0]
                grad_spatial = grad.abs().mean(dim=2)
                grad_7 = F.interpolate(
                    grad_spatial.reshape(B * T, 1, H, W),
                    size=(self.feat_h, self.feat_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(B, T, self.feat_h, self.feat_w)
        finally:
            for p, state in zip(self.parameters(), param_states):
                p.requires_grad = state
        return grad_7.detach()