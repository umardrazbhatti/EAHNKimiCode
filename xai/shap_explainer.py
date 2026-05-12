"""
xai/shap_explainer.py — Integrated Gradients (Captum) as SHAP approximation.
"""

import torch
import numpy as np


class SHAPExplainer:
    def __init__(self, model, method: str = "integratedgrads"):
        self.model  = model
        self.method = method
        if method == "integratedgrads":
            from captum.attr import IntegratedGradients
            self.ig = IntegratedGradients(self._forward_wrapper)

    def _forward_wrapper(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).logit

    def explain(self, frames: torch.Tensor) -> np.ndarray:
        frames = frames.float().requires_grad_(True)
        try:
            attributions = self.ig.attribute(
                frames, target=None, n_steps=20, internal_batch_size=1
            )
        except Exception:
            out = self.model(frames)
            out.logit.backward()
            attributions = frames.grad

        saliency = attributions.abs().mean(dim=2, keepdim=True)
        saliency  = saliency.squeeze(0).squeeze(1)

        mn = saliency.min()
        mx = saliency.max()
        saliency = (saliency - mn) / (mx - mn + 1e-8)
        return saliency.detach().cpu().numpy()
