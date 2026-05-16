import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ExplanationLoss(nn.Module):
    """
    Explanation loss for EAHN with numerical stability fixes.
    Computes:
      - L_cls: classification loss (BCE)
      - L_exp: explanation loss = entropy + diversity + class_separation
    """
    def __init__(
        self,
        diversity_weight: float = 2.0,      # FIXED: 8.0 -> 2.0
        entropy_weight: float = 1.0,
        class_sep_weight: float = 1.0,
        diversity_hinge: float = 0.3,         # FIXED: 0.2 -> 0.3
        class_sep_hinge: float = 0.1,         # FIXED: 0.05 -> 0.1
        eps: float = 1e-6,                    # FIXED: added global eps
    ):
        super().__init__()
        self.diversity_weight = diversity_weight
        self.entropy_weight = entropy_weight
        self.class_sep_weight = class_sep_weight
        self.diversity_hinge = diversity_hinge
        self.class_sep_hinge = class_sep_hinge
        self.eps = eps

    def forward(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        attention_maps: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: (B,) logits or probabilities
            labels: (B,) binary labels {0, 1}
            attention_maps: (B, 1, H, W) attention weights (after softmax)
            masks: (B, 1, H, W) ground-truth masks [0, 1]
        Returns:
            dict with 'loss', 'L_cls', 'L_exp', 'L_entropy', 'L_diversity', 'L_class_sep'
        """
        eps = self.eps

        # --- Classification Loss ---
        L_cls = F.binary_cross_entropy_with_logits(
            predictions, labels.float(), reduction="mean"
        )

        B, _, H, W = attention_maps.shape
        attn = attention_maps.view(B, -1)  # (B, H*W)

        # --- Entropy Loss (encourage peaky attention) ---
        # Normalize by map size so it doesn't scale with 224x224=50176
        attn_safe = torch.clamp(attn, min=eps, max=1.0)
        entropy = -(attn_safe * torch.log(attn_safe + eps)).sum(dim=1).mean()
        L_entropy = entropy / (H * W)  # FIXED: normalize by spatial size

        # --- Diversity Loss (push different samples apart) ---
        if B > 1:
            # Pairwise cosine similarity matrix
            attn_norm = F.normalize(attn, p=2, dim=1)  # (B, H*W)
            sim_matrix = torch.mm(attn_norm, attn_norm.t())  # (B, B)

            # Mask out diagonal
            mask_diag = torch.eye(B, device=sim_matrix.device).bool()
            sim_offdiag = sim_matrix.masked_fill(mask_diag, 0.0)

            # Hinge: penalize if similarity > threshold
            diversity = F.relu(sim_offdiag - self.diversity_hinge).mean()
        else:
            diversity = torch.tensor(0.0, device=attn.device)

        L_diversity = diversity

        # --- Class Separation Loss (push real/fake attention apart) ---
        if B > 1 and labels.unique().numel() > 1:
            real_mask = (labels == 0)
            fake_mask = (labels == 1)
            if real_mask.sum() > 0 and fake_mask.sum() > 0:
                real_attn = attn[real_mask].mean(dim=0)   # (H*W,)
                fake_attn = attn[fake_mask].mean(dim=0)     # (H*W,)

                real_norm = F.normalize(real_attn.unsqueeze(0), p=2, dim=1)
                fake_norm = F.normalize(fake_attn.unsqueeze(0), p=2, dim=1)
                class_sim = torch.mm(real_norm, fake_norm.t()).squeeze()

                L_class_sep = F.relu(class_sim - self.class_sep_hinge)
            else:
                L_class_sep = torch.tensor(0.0, device=attn.device)
        else:
            L_class_sep = torch.tensor(0.0, device=attn.device)

        # --- Combine ---
        L_exp = (
            self.entropy_weight * L_entropy
            + self.diversity_weight * L_diversity
            + self.class_sep_weight * L_class_sep
        )

        total_loss = L_cls + L_exp

        # --- Numerical Safety ---
        total_loss = torch.nan_to_num(total_loss, nan=100.0, posinf=100.0, neginf=-100.0)
        total_loss = torch.clamp(total_loss, min=-10.0, max=10.0)  # FIXED: gradient safety

        return {
            "loss": total_loss,
            "L_cls": L_cls.detach(),
            "L_exp": L_exp.detach(),
            "L_entropy": L_entropy.detach(),
            "L_diversity": L_diversity.detach(),
            "L_class_sep": L_class_sep.detach(),
        }