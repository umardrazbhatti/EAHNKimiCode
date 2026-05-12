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
            # 0 -> eps, 1 -> 1-eps
            target = target * (1 - self.label_smoothing) + (1 - target) * self.label_smoothing
        return F.binary_cross_entropy_with_logits(logit, target)


class FocalLoss(nn.Module):
    """
    Focal loss for class-imbalanced binary classification.
    Safe under torch.autocast because it uses BCEWithLogits (no manual sigmoid + BCE).

    alpha : weight for positive class (fake). 0.25 down-weights the easy majority.
    gamma : focusing parameter — higher = more focus on hard examples.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Use BCEWithLogits instead of sigmoid + BCE (unsafe under autocast)
        bce = F.binary_cross_entropy_with_logits(logit, target.float(), reduction='none')

        # Compute probability for focal weighting
        with torch.no_grad():
            prob = torch.sigmoid(logit)
            pt = torch.where(target.bool(), prob, 1 - prob)
            focal_weight = self.alpha * (1 - pt).pow(self.gamma)

        focal = focal_weight * bce
        return focal.mean()


def build_classification_loss(config) -> nn.Module:
    """Factory: FocalLoss (default) or ClassificationLoss based on config."""
    if getattr(config, "cls_loss_type", "bce") == "focal":
        return FocalLoss(
            alpha=getattr(config, "focal_alpha", 0.25),
            gamma=getattr(config, "focal_gamma", 2.0),
        )
    return ClassificationLoss(label_smoothing=getattr(config, 'label_smoothing', 0.0))
