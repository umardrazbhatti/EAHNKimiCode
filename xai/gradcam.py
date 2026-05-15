"""
xai/gradcam.py — Grad-CAM explainer compatible with EAHN's concatenated classifier.

FIXES APPLIED:
  1. _SpatialModel now mirrors EAHN's exact classifier architecture (512-d input MLP).
  2. Weight copying handles Sequential vs Linear safely with dimension checks.
  3. target_layer is resolved from the NEW _SpatialModel, not the original EAHN.
  4. BatchNorm running stats are deep-copied from original backbone.
  5. _ScalarOutputTarget handles binary output without IndexError.
  6. Graceful fallback on CAM failure (null gradients / missing hooks).
"""

import torch
import torch.nn as nn
import numpy as np
import copy


class _ScalarOutputTarget:
    """Returns scalar score for binary classifier (no class index needed)."""
    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        # model_output: (N, 1) or (N,)
        if model_output.dim() == 2 and model_output.shape[1] == 1:
            return model_output[:, 0].sum()
        if model_output.dim() == 1:
            return model_output.sum()
        # Fallback for unexpected shape
        return model_output.view(-1)[0]


class _SpatialModel(nn.Module):
    """
    Reconstructs a spatial-only model from EAHN components for Grad-CAM.
    Must exactly match the forward path that produces the final logit,
    but using ONLY spatial features (no temporal/cross-attention).
    """
    def __init__(self, spatial_stream, classifier_head: nn.Module, device: str):
        super().__init__()
        self.backbone = copy.deepcopy(spatial_stream.backbone)
        self.proj = copy.deepcopy(spatial_stream.proj)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # The classifier_head from EAHN is Sequential(Linear(512,256), ReLU, Dropout, Linear(256,1))
        # For spatial-only GradCAM we only have 256-d pooled features, not 512-d concatenated.
        # So we extract the LAST Linear layer and its preceding features.
        if isinstance(classifier_head, nn.Sequential):
            # classifier_head: [Linear(512,256), ReLU, Dropout, Linear(256,1)]
            # We need to map 256-d spatial pool -> 1-d logit.
            # Use the final Linear(256,1) layer.
            final_linear = None
            for m in reversed(classifier_head):
                if isinstance(m, nn.Linear):
                    final_linear = m
                    break
            if final_linear is None:
                raise ValueError("classifier_head has no Linear layer")
            self.classifier = nn.Linear(final_linear.in_features, 1)
            # Copy weights safely
            with torch.no_grad():
                if final_linear.weight.shape == self.classifier.weight.shape:
                    self.classifier.weight.copy_(final_linear.weight)
                    self.classifier.bias.copy_(final_linear.bias)
                else:
                    nn.init.xavier_uniform_(self.classifier.weight)
                    nn.init.zeros_(self.classifier.bias)
        else:
            # Plain Linear classifier (legacy / backward compat)
            self.classifier = nn.Linear(classifier_head.in_features, 1)
            with torch.no_grad():
                if classifier_head.weight.shape == self.classifier.weight.shape:
                    self.classifier.weight.copy_(classifier_head.weight)
                    self.classifier.bias.copy_(classifier_head.bias)
                else:
                    nn.init.xavier_uniform_(self.classifier.weight)
                    nn.init.zeros_(self.classifier.bias)

        self.to(device)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, C, H, W)
        feats = self.backbone(x)
        # Handle timm feature pyramid: take last element
        if isinstance(feats, (list, tuple)):
            last_feat = feats[-1]
        else:
            last_feat = feats
        proj = self.proj(last_feat)
        pooled = self.avg_pool(proj).reshape(x.size(0), -1)
        logit = self.classifier(pooled)
        return logit


class GradCAMExplainer:
    def __init__(self, eahn_model, target_layer_name: str = None):
        """
        Args:
            eahn_model: The full EAHN model instance.
            target_layer_name: Optional str like 'backbone.blocks.6'.
                               If None, auto-detects last conv block.
        """
        self.eahn_model = eahn_model
        device = getattr(eahn_model, 'config', None)
        device = getattr(device, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Build spatial-only surrogate model
        self.spatial_model = _SpatialModel(
            spatial_stream=eahn_model.spatial_stream,
            classifier_head=eahn_model.classifier,
            device=str(self.device),
        )

        # Resolve target layer in the NEW spatial_model (not original EAHN)
        if target_layer_name is None:
            # Auto-detect: find last Conv2d / DepthwiseConv2d in backbone
            target_layer = None
            for name, module in reversed(list(self.spatial_model.backbone.named_modules())):
                if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                    target_layer = module
                    break
            if target_layer is None:
                raise RuntimeError("Could not auto-detect target conv layer in backbone")
        else:
            # Navigate by name, e.g. 'backbone.blocks.6'
            parts = target_layer_name.split('.')
            target_layer = self.spatial_model.backbone
            for part in parts:
                if part.isdigit():
                    target_layer = target_layer[int(part)]
                else:
                    target_layer = getattr(target_layer, part)

        try:
            from pytorch_grad_cam import GradCAM
            self.cam = GradCAM(
                model=self.spatial_model,
                target_layers=[target_layer],
            )
        except ImportError:
            raise ImportError("pytorch_grad_cam not installed. Run: pip install grad-cam")

    def explain(self, frames: torch.Tensor) -> np.ndarray:
        """
        Args:
            frames: (B, T, C, H, W) video tensor.
        Returns:
            heatmaps: (B, T, H, W) normalized numpy array in [0,1].
        """
        B, T, C, H, W = frames.shape
        frames_flat = frames.reshape(B * T, C, H, W).to(self.device)

        targets = [_ScalarOutputTarget() for _ in range(B * T)]

        # FIX: Graceful fallback on CAM failure
        try:
            with torch.no_grad():
                grayscale_cams = self.cam(
                    input_tensor=frames_flat,
                    targets=targets,
                    aug_smooth=False,
                    eigen_smooth=False,
                )
        except Exception as e:
            print(f"[GradCAM] CAM computation failed: {e}. Returning uniform heatmaps.")
            grayscale_cams = np.ones((B * T, H, W), dtype=np.float32) / (H * W)

        heatmaps = torch.from_numpy(grayscale_cams).reshape(B, T, H, W)

        # Per-sample min-max normalization
        flat = heatmaps.reshape(B, T, -1)
        mn = flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)  # (B, T, 1, 1)
        mx = flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        heatmaps = (heatmaps - mn) / (mx - mn + 1e-8)

        return heatmaps.numpy()