import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """BCE with optional label smoothing."""
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        target = label.float()
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + (1 - target) * self.label_smoothing
        return F.binary_cross_entropy_with_logits(logit, target)


class FocalLoss(nn.Module):
    """
    Focal loss for class-imbalanced binary classification.
    Safe under torch.autocast because it uses BCEWithLogits.

    FIX: alpha is now class-conditional (P2).
         alpha_t = alpha      if y == 1 (fake)
         alpha_t = 1 - alpha  if y == 0 (real)
         With alpha=0.25, real (minority) gets 0.75 weight, fake gets 0.25.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = label_smoothing

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        if self.eps > 0:
            target = target * (1 - self.eps) + (1 - target) * self.eps

        bce = F.binary_cross_entropy_with_logits(logit, target, reduction='none')

        with torch.no_grad():
            prob = torch.sigmoid(logit)
            pt = torch.where(target > 0.5, prob, 1 - prob)
            # P2 FIX: class-conditional alpha
            alpha_t = torch.where(
                target > 0.5,
                torch.tensor(self.alpha, device=logit.device, dtype=logit.dtype),
                torch.tensor(1.0 - self.alpha, device=logit.device, dtype=logit.dtype),
            )
            focal_weight = alpha_t * (1 - pt).pow(self.gamma)

        return (focal_weight * bce).mean()


def build_classification_loss(config) -> nn.Module:
    """Factory: FocalLoss (default) or ClassificationLoss based on config."""
    if getattr(config, "cls_loss_type", "bce") == "focal":
        return FocalLoss(
            alpha=getattr(config, "focal_alpha", 0.25),
            gamma=getattr(config, "focal_gamma", 2.0),
            label_smoothing=getattr(config, 'label_smoothing', 0.0),
        )
    return ClassificationLoss(label_smoothing=getattr(config, 'label_smoothing', 0.0))