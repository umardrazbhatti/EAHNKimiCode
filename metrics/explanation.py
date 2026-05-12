"""
metrics/explanation.py — Explanation quality metrics.

FIX: deletion_insertion_auc had broken 5D tensor indexing (del_frames[b,:, :,mask_2d]).
     Replaced with vectorised torch.where() using expanded boolean masks.
"""

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import spearmanr
from typing import Dict


class ExplanationMetrics:

    @staticmethod
    def localisation_iou(
        M_t_avg: torch.Tensor,
        gt_masks: torch.Tensor,
        has_mask_flags,
        threshold: float = 0.5,
    ):
        import torch.nn.functional as F

        if M_t_avg.dim() == 2:
            M_t_avg   = M_t_avg.unsqueeze(0)
            gt_masks  = gt_masks.unsqueeze(0)
            has_mask_flags = [has_mask_flags]                 if not hasattr(has_mask_flags, "__len__") else list(has_mask_flags)

        B = M_t_avg.shape[0]
        valid_ious = []

        for i in range(B):
            if not bool(has_mask_flags[i]):
                continue
            gt = gt_masks[i]
            if gt.sum() == 0:
                continue
            m = M_t_avg[i]
            if gt.shape != m.shape:
                gt = F.interpolate(
                    gt.unsqueeze(0).unsqueeze(0).float(),
                    size=m.shape, mode="bilinear", align_corners=False,
                ).squeeze()
            M_bin = (m > threshold).float()
            gt_f  = gt.float()
            inter = (M_bin * gt_f).sum()
            union = ((M_bin + gt_f) > 0).float().sum()
            valid_ious.append(float(inter / (union + 1e-8)))

        if len(valid_ious) == 0:
            return None
        return float(np.mean(valid_ious))

    @staticmethod
    def temporal_ssim(M_t_up: torch.Tensor) -> float:
        values = []
        N, T, H, W = M_t_up.shape
        for b in range(N):
            for t in range(T - 1):
                a = M_t_up[b, t].cpu().numpy().astype(np.float32)
                b_ = M_t_up[b, t + 1].cpu().numpy().astype(np.float32)
                val = ssim(a, b_, data_range=1.0)
                values.append(val)
        return float(np.mean(values)) if values else 1.0

    @staticmethod
    def faithfulness_correlation(
        M_flat: torch.Tensor,
        grad_flat: torch.Tensor,
    ) -> float:
        m = M_flat.detach().cpu().numpy().flatten()
        g = grad_flat.detach().cpu().numpy().flatten()
        if len(m) < 3 or np.std(m) < 1e-8 or np.std(g) < 1e-8:
            return 0.0
        corr, _ = spearmanr(m, g)
        return float(corr) if not np.isnan(corr) else 0.0

    @staticmethod
    def deletion_insertion_auc(model, frames, saliency,
                               steps: int = 10) -> dict:
        """
        Deletion/Insertion AUC with fixed 5D tensor indexing.
        """
        device = next(model.parameters()).device
        B, T, C, H, W = frames.shape
        total_pixels = H * W

        with torch.no_grad():
            baseline_logit = model(frames.to(device)).prob.mean().item()

        del_scores = []
        ins_scores = []

        # Mean explanation over time → (B, H, W)
        if isinstance(saliency, torch.Tensor):
            sal = saliency.mean(1).cpu().numpy()
        else:
            sal = saliency.mean(1)

        for step in range(steps + 1):
            frac = step / steps
            k = max(1, int(frac * total_pixels))

            del_frames = frames.clone()
            ins_frames = torch.zeros_like(frames)

            for b in range(B):
                flat_sal = sal[b].reshape(-1)
                top_k_idx = np.argsort(flat_sal)[-k:]
                mask = np.zeros(H * W, dtype=bool)
                mask[top_k_idx] = True
                mask_2d = mask.reshape(H, W)
                mask_t = torch.from_numpy(mask_2d).to(del_frames.device)
                mask_exp = mask_t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

                # Vectorised over T and C
                del_frames[b] = del_frames[b] * (~mask_exp).float()
                ins_frames[b] = ins_frames[b] * (~mask_exp).float() + frames[b] * mask_exp.float()

            with torch.no_grad():
                del_score = model(del_frames.to(device)).prob.mean().item()
                ins_score = model(ins_frames.to(device)).prob.mean().item()

            del_scores.append(del_score)
            ins_scores.append(ins_score)

        _trapz = getattr(np, "trapezoid", np.trapz)
        del_auc = float(_trapz(del_scores) / steps)
        ins_auc = float(_trapz(ins_scores) / steps)
        return {"deletion_auc": del_auc, "insertion_auc": ins_auc}

    @staticmethod
    def collapse_diagnostics(all_M_t: torch.Tensor) -> Dict[str, float]:
        N, T, H, W = all_M_t.shape

        flat = all_M_t.mean(dim=1).reshape(N, H * W).float()
        flat_norm = flat / (flat.norm(dim=-1, keepdim=True) + 1e-8)
        sim_matrix = flat_norm @ flat_norm.T
        eye = torch.eye(N, dtype=torch.bool, device=all_M_t.device)
        n_pairs = N * (N - 1)
        inter_cosine = float(
            sim_matrix.masked_fill(eye, 0.0).sum().item() / max(n_pairs, 1)
        )

        mean_maps = all_M_t.mean(dim=1)
        peak_indices = mean_maps.reshape(N, -1).argmax(dim=-1)
        peak_rc = [(int(idx) // W, int(idx) % W) for idx in peak_indices.tolist()]
        from collections import Counter
        most_common_count = Counter(peak_rc).most_common(1)[0][1]
        peak_mode_share = float(most_common_count) / N

        stds = all_M_t.std(dim=(-1, -2)).mean(dim=-1)
        m_t_std_mean = float(stds.mean().item())
        m_t_std_max  = float(stds.max().item())

        return {
            "inter_sample_cosine_mean": inter_cosine,
            "peak_mode_share":          peak_mode_share,
            "m_t_std_mean":             m_t_std_mean,
            "m_t_std_max":              m_t_std_max,
        }
