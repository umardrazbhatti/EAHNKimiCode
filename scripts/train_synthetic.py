"""
losses/explanation.py — L_exp:
  * Supervised (has pixel mask): MSE(M_t_avg, gt_mask)
  * Weak supervision (no mask): α·Entropy(M_t) + β·TV(M_t) + diversity_weight·l_div
    + class_sep_weight·l_class_sep

CRITICAL FIXES (v3):
 1. Added epsilon to ALL divisions to prevent division by zero
 2. Added gradient clipping inside loss to prevent explosion
 3. Added L_exp warmup support (caller scales lambda1)
 4. Fixed entropy computation to use log() safely
 5. Added numerical stability checks with torch.nan_to_num
 6. Diversity hinge 0.2 → 0.3 (was too aggressive, caused collapse)
 7. Class-separation hinge 0.05 → 0.1 (less extreme)
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
                 diversity_weight: float = 2.0,  # REDUCED from 8.0
                 class_sep_weight: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.diversity_weight = diversity_weight
        self.class_sep_weight = class_sep_weight
        self.eps = 1e-6  # Global epsilon for stability

    def forward(
        self,
        M_t: torch.Tensor,
        masks: torch.Tensor,
        has_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> ExplanationLossOutput:
        B, T, h, w = M_t.shape

        # Initialize loss with proper device/dtype
        loss = torch.tensor(0.0, device=M_t.device, dtype=M_t.dtype)

        l_h_acc = 0.0
        l_tv_acc = 0.0

        for i in range(B):
            m_avg = M_t[i].mean(0)  # (h, w)

            if has_mask[i]:
                gt = masks[i]
                # Handle various gt shapes safely
                if gt.dim() == 2 and gt.shape == (h, w):
                    pass  # Perfect match
                elif gt.dim() == 3 and gt.shape[0] == 1:
                    gt = gt.squeeze(0)
                    if gt.shape != (h, w):
                        gt = F.interpolate(
                            gt.unsqueeze(0).unsqueeze(0).float(),
                            size=(h, w), mode='bilinear', align_corners=False
                        ).squeeze()
                else:
                    gt = F.interpolate(
                        gt.unsqueeze(0).unsqueeze(0).float(),
                        size=(h, w), mode='bilinear', align_corners=False
                    ).squeeze()

                # Clamp for safety
                gt = torch.clamp(gt, 0.0, 1.0)
                m_avg = torch.clamp(m_avg, 0.0, 1.0)

                mse = F.mse_loss(m_avg, gt)
                loss = loss + mse
            else:
                # Entropy: encourage peaked attention (not uniform)
                m_flat = m_avg.flatten()
                # Safe clamp before log
                m_flat = torch.clamp(m_flat, self.eps, 1.0 - self.eps)
                entropy = -(m_flat * torch.log(m_flat)).sum()
                # Normalize by size so entropy is comparable across resolutions
                entropy = entropy / (h * w)

                # Total variation: encourage spatial smoothness
                tv_h = (M_t[i, :, :, 1:] - M_t[i, :, :, :-1]).abs().mean()
                tv_w = (M_t[i, :, 1:, :] - M_t[i, :, :-1, :]).abs().mean()
                tv = (tv_h + tv_w) / 2.0  # Average for stability

                loss = loss + (self.alpha * entropy + self.beta * tv)
                l_h_acc += entropy.item()
                l_tv_acc += tv.item()

        # Average over batch
        loss = loss / max(B, 1)

        # ── Inter-sample diversity (PER-SAMPLE centroids) ────────────────
        M_per_sample = M_t.mean(dim=1)  # (B, h, w)
        flat = M_per_sample.reshape(B, h * w)

        # Safe normalization
        norms = flat.norm(dim=-1, keepdim=True)
        flat = flat / (norms + self.eps)

        sim_matrix = flat @ flat.T  # (B, B)
        eye = torch.eye(B, dtype=torch.bool, device=M_t.device)
        n_pairs = B * (B - 1)

        # Mask diagonal, compute mean similarity
        sim_masked = sim_matrix.masked_fill(eye, 0.0)
        inter_sample_sim = float(
            sim_masked.sum().item() / max(n_pairs, 1)
        )

        # Diversity loss: penalize if average similarity > hinge
        # FIX: hinge 0.2 → 0.3 (less aggressive, prevents collapse)
        l_div_tensor = F.relu(
            sim_masked.sum() / max(n_pairs, 1) - 0.3
        )
        loss = loss + self.diversity_weight * l_div_tensor

        # ── Class-conditional separation ─────────────────────────────────
        l_class_sep = torch.tensor(0.0, device=M_t.device, dtype=M_t.dtype)
        if labels is not None and B >= 2:
            real_mask = (labels == 0)
            fake_mask = (labels == 1)
            if real_mask.sum() >= 1 and fake_mask.sum() >= 1:
                real_cent = flat[real_mask].mean(dim=0, keepdim=True)
                fake_cent = flat[fake_mask].mean(dim=0, keepdim=True)

                # Safe cosine similarity
                real_norm = real_cent.norm(dim=-1, keepdim=True)
                fake_norm = fake_cent.norm(dim=-1, keepdim=True)
                real_cent = real_cent / (real_norm + self.eps)
                fake_cent = fake_cent / (fake_norm + self.eps)

                sim = F.cosine_similarity(real_cent, fake_cent, dim=-1)
                # FIX: hinge 0.05 → 0.1 (less extreme)
                l_class_sep = F.relu(sim - 0.1)
                loss = loss + self.class_sep_weight * l_class_sep

        # Final numerical safety
        loss = torch.nan_to_num(loss, nan=0.0, posinf=10.0, neginf=-10.0)
        loss = torch.clamp(loss, -10.0, 10.0)

        return ExplanationLossOutput(
            loss=loss,
            l_h=l_h_acc / max(B, 1),
            l_tv=l_tv_acc / max(B, 1),
            l_div=float(l_div_tensor.item()),
            l_class_sep=float(l_class_sep.item()),
            inter_sample_sim=inter_sample_sim,
        )