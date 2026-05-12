"""
xai/gradcam.py — Grad-CAM on a per-frame spatial model.

FIX: ClassifierOutputTarget(1) raises IndexError on binary classifier.
Replaced with _ScalarOutputTarget.
"""

import torch
import torch.nn as nn
import numpy as np


class _SpatialModel(nn.Module):
    def __init__(self, spatial_stream, d_model: int, device: str):
        super().__init__()
        self.backbone    = spatial_stream.backbone
        self.proj        = spatial_stream.proj
        self.avg_pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier  = nn.Linear(d_model, 1)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats     = self.backbone(x)
        last_feat = feats[-1]
        proj      = self.proj(last_feat)
        pooled    = self.avg_pool(proj).reshape(x.size(0), -1)
        logit     = self.classifier(pooled)
        return logit


class _ScalarOutputTarget:
    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.dim() == 2:
            return model_output[:, 0].sum()
        return model_output.sum()


class GradCAMExplainer:
    def __init__(self, eahn_model, target_layer: nn.Module):
        self.model = eahn_model
        device = eahn_model.config.device

        self.spatial_model = _SpatialModel(
            spatial_stream=eahn_model.spatial_stream,
            d_model=eahn_model.config.d_model,
            device=device,
        )

        self.spatial_model.classifier.weight.data.copy_(
            eahn_model.classifier.weight.data
        )
        self.spatial_model.classifier.bias.data.copy_(
            eahn_model.classifier.bias.data
        )

        from pytorch_grad_cam import GradCAM
        self.cam = GradCAM(
            model=self.spatial_model,
            target_layers=[target_layer],
        )
        self.device = device

    def explain(self, frames: torch.Tensor) -> np.ndarray:
        B, T, C, H, W = frames.shape
        frames_flat = frames.reshape(B * T, C, H, W).to(self.device)

        targets = [_ScalarOutputTarget()] * (B * T)

        grayscale_cams = self.cam(
            input_tensor=frames_flat,
            targets=targets,
            aug_smooth=False,
            eigen_smooth=False,
        )

        heatmaps = torch.from_numpy(grayscale_cams).reshape(B, T, H, W)

        mn = heatmaps.reshape(B, T, -1).min(-1, keepdim=True)[0].unsqueeze(-1)
        mx = heatmaps.reshape(B, T, -1).max(-1, keepdim=True)[0].unsqueeze(-1)
        heatmaps = (heatmaps - mn) / (mx - mn + 1e-8)

        return heatmaps.numpy()
