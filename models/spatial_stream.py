"""
models/spatial_stream.py — EfficientNet/ConvNeXt backbone wrapper.

FIX: set_frozen() now forces BN layers into eval() mode and disables
running-stat tracking when frozen. This prevents BatchNorm corruption
with batch_size=4, which was the root cause of the Epoch-2 NaN cascade.
"""

import torch
import torch.nn as nn
import timm


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
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True
        )
        self.feat_channels = self.backbone.feature_info.channels()[-1]
        self.proj = nn.Conv2d(self.feat_channels, d_model, kernel_size=1)

        if "efficientnet" in backbone_name:
            self.low_level_extractor = nn.Sequential(
                self.backbone.conv_stem,
                self.backbone.bn1,
                nn.SiLU(inplace=True),
            )
        elif "convnext" in backbone_name:
            self.low_level_extractor = self.backbone.stem
        else:
            self.low_level_extractor = None

        if freeze_backbone:
            self.set_frozen(True)

        self._cached_low_level: torch.Tensor = None
        self.feat_h: int = None
        self.feat_w: int = None

    def set_frozen(self, freeze: bool):
        """
        Freeze or unfreeze backbone parameters.
        CRITICAL FIX: When frozen, BN layers are forced to eval() mode and
        track_running_stats is disabled. This prevents running mean/var
        corruption caused by batch_size=4.
        """
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if freeze:
                    m.eval()
                    m.track_running_stats = False
                else:
                    m.train()
                    m.track_running_stats = True

    @property
    def grad_cam_target_layer(self):
        if hasattr(self.backbone, "blocks"):
            return self.backbone.blocks[-1]
        elif hasattr(self.backbone, "stages"):
            return self.backbone.stages[-1]
        else:
            return self.proj

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
