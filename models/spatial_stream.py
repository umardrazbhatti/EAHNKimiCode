"""
models/spatial_stream.py — Feature extraction from raw frames.

FIXES:
 1. set_frozen now safely checks module types before calling eval()/train().
 2. Added explicit requires_grad reset on all backbone parameters.
"""

import math
from typing import Optional

import timm
import torch
import torch.nn as nn


class SpatialStream(nn.Module):
    def __init__(
        self,
        backbone_name: str = "efficientnet_b4",
        pretrained: bool = True,
        d_model: int = 256,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.d_model = d_model

        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True
        )

        if "efficientnet" in backbone_name:
            self.low_level_extractor = nn.Sequential(
                self.backbone.conv_stem,
                self.backbone.bn1,
                nn.SiLU(inplace=True),
            )
        else:
            self.low_level_extractor = None

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feats = self.backbone(dummy)
            self.feat_channels = feats[-1].shape[1]
            self.feat_h = feats[-1].shape[2]
            self.feat_w = feats[-1].shape[3]

        self.proj = nn.Conv2d(self.feat_channels, d_model, kernel_size=1)

        self._cached_low_level: Optional[torch.Tensor] = None

        if freeze_backbone:
            self.set_frozen(True)

    def set_frozen(self, freeze: bool):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if freeze:
                    m.eval()
                    if hasattr(m, 'track_running_stats'):
                        m.track_running_stats = False
                else:
                    m.train()
                    if hasattr(m, 'track_running_stats'):
                        m.track_running_stats = True

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.low_level_extractor is not None:
                low = self.low_level_extractor(frames)
            else:
                low = self.backbone(frames)[0]
            self._cached_low_level = low.detach()

        feats = self.backbone(frames)
        last = feats[-1]
        self.feat_h, self.feat_w = last.shape[-2], last.shape[-1]

        proj = self.proj(last)
        tokens = proj.flatten(2).transpose(1, 2)
        return tokens

    def low_level_features(self) -> torch.Tensor:
        if self._cached_low_level is None:
            raise RuntimeError("Call forward() before low_level_features().")
        return self._cached_low_level

    @property
    def grad_cam_target_layer(self):
        if "efficientnet" in self.backbone_name:
            return self.backbone.blocks[-1]
        return list(self.backbone.modules())[-1]