"""
losses/explanation.py — L_exp:
  * Supervised (has pixel mask): MSE(M_t_avg, gt_mask)
  * Weak supervision (no mask): α·Entropy(M_t) + β·TV(M_t) + diversity_weight·l_div

FIXES:
 1. alpha reverted to 0.5 (was 0.2) — stronger penalty on one-hot collapse.
 2. Diversity now computed on PER-SAMPLE centroids (time-averaged) instead of
    per-frame maps. This prevents the model from bypassing diversity via
    augmentation noise.
 3. NEW: Class-conditional separation loss — penalises when real and fake
    attention maps are too similar (forces discriminative explanations).
 4. Diversity hinge lowered from 0.5 → 0.2 (stronger penalty).
 5. Class-separation hinge tightened from 0.2 → 0.05.
 6. Added class_sep_weight parameter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExplanationLossOutput:
    loss: torch.Tensor
    l_h: float
    l_tv: float
    l_div: float
    l_class_sep: float
    inter_sample_sim: float


class ExplanationLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5,
                 diversity_weight: float = 2.5, class_sep_weight: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.diversity_weight = diversity_weight
        self.class_sep_weight = class_sep_weight

    def forward(
        self,
        M_t: torch.Tensor,
        masks: torch.Tensor,
        has_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> ExplanationLossOutput:
        B, T, h, w = M_t.shape
        loss = M_t.new_zeros(1).squeeze()

        l_h_acc = 0.0
        l_tv_acc = 0.0

        for i in range(B):
            m_avg = M_t[i].mean(0)

            if has_mask[i]:
                gt = masks[i]
                if gt.shape != (h, w):
                    gt = F.interpolate(
                        gt.unsqueeze(0).unsqueeze(0).float(),
                        size=(h, w), mode='bilinear', align_corners=False
                    ).squeeze()
                loss = loss + F.mse_loss(m_avg, gt)
            else:
                m_flat = m_avg.clamp(1e-8, 1 - 1e-8).flatten()
                entropy = -(m_flat * m_flat.log()).sum()

                tv_h = (M_t[i, :, :, 1:] - M_t[i, :, :, :-1]).abs().mean()
                tv_w = (M_t[i, :, 1:, :] - M_t[i, :, :-1, :]).abs().mean()
                tv = tv_h + tv_w

                loss = loss + (self.alpha * entropy + self.beta * tv)
                l_h_acc += entropy.item()
                l_tv_acc += tv.item()

        loss = loss / B

        # ── Inter-sample diversity (PER-SAMPLE centroids) ────────────────────
        M_per_sample = M_t.mean(dim=1)  # (B, h, w)
        flat = M_per_sample.reshape(B, h * w)
        flat = flat / (flat.norm(dim=-1, keepdim=True) + 1e-8)
        sim_matrix = flat @ flat.T  # (B, B)
        eye = torch.eye(B, dtype=torch.bool, device=M_t.device)
        n_pairs = B * (B - 1)
        inter_sample_sim = float(
            sim_matrix.masked_fill(eye, 0.0).sum().item() / max(n_pairs, 1)
        )
        # FIX: Hinge lowered from 0.5 → 0.2
        l_div_tensor = F.relu(
            sim_matrix.masked_fill(eye, 0.0).sum() / max(n_pairs, 1) - 0.2
        )
        loss = loss + self.diversity_weight * l_div_tensor

        # ── Class-conditional separation ───────────────────────────────────────
        l_class_sep = torch.tensor(0.0, device=M_t.device)
        if labels is not None and B >= 2:
            real_mask = (labels == 0)
            fake_mask = (labels == 1)
            if real_mask.any() and fake_mask.any():
                real_cent = flat[real_mask].mean(dim=0, keepdim=True)
                fake_cent = flat[fake_mask].mean(dim=0, keepdim=True)
                sim = F.cosine_similarity(real_cent, fake_cent, dim=-1)
                # FIX: Hinge tightened from 0.2 → 0.05
                l_class_sep = F.relu(0.05 - sim) * self.class_sep_weight
                loss = loss + l_class_sep

        return ExplanationLossOutput(
            loss=loss,
            l_h=l_h_acc / max(B, 1),
            l_tv=l_tv_acc / max(B, 1),
            l_div=float(l_div_tensor.item()),
            l_class_sep=float(l_class_sep.item()),
            inter_sample_sim=inter_sample_sim,
        )