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
        diversity_weight: float = 2.0,
        entropy_weight: float = 1.0,
        class_sep_weight: float = 1.0,
        diversity_hinge: float = 0.3,
        class_sep_hinge: float = 0.1,
        eps: float = 1e-6,
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
        eps = self.eps

        # --- Classification Loss ---
        L_cls = F.binary_cross_entropy_with_logits(
            predictions, labels.float(), reduction="mean"
        )

        B, _, H, W = attention_maps.shape
        attn = attention_maps.view(B, -1)

        # --- Entropy Loss ---
        attn_safe = torch.clamp(attn, min=eps, max=1.0)
        entropy = -(attn_safe * torch.log(attn_safe + eps)).sum(dim=1).mean()
        L_entropy = entropy / (H * W)

        # --- Diversity Loss ---
        if B > 1:
            attn_norm = F.normalize(attn, p=2, dim=1)
            sim_matrix = torch.mm(attn_norm, attn_norm.t())
            mask_diag = torch.eye(B, device=sim_matrix.device).bool()
            sim_offdiag = sim_matrix.masked_fill(mask_diag, 0.0)
            diversity = F.relu(sim_offdiag - self.diversity_hinge).mean()
        else:
            diversity = torch.tensor(0.0, device=attn.device)

        L_diversity = diversity

        # --- Class Separation Loss ---
        if B > 1 and labels.unique().numel() > 1:
            real_mask = (labels == 0)
            fake_mask = (labels == 1)
            if real_mask.sum() > 0 and fake_mask.sum() > 0:
                real_attn = attn[real_mask].mean(dim=0)
                fake_attn = attn[fake_mask].mean(dim=0)
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

        # Numerical Safety
        total_loss = torch.nan_to_num(total_loss, nan=100.0, posinf=100.0, neginf=-100.0)
        total_loss = torch.clamp(total_loss, min=-10.0, max=10.0)

        return {
            "loss": total_loss,
            "L_cls": L_cls,           # FIXED: removed .detach() so trainer can weight it
            "L_exp": L_exp,           # FIXED: removed .detach()
            "L_entropy": L_entropy.detach(),
            "L_diversity": L_diversity.detach(),
            "L_class_sep": L_class_sep.detach(),
        }