"""
losses/temporal.py — Gated Temporal Consistency loss L_temp.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConsistencyLoss(nn.Module):
    def __init__(self, gamma: float = 0.1):
        super().__init__()
        self.gamma = gamma

    def forward(self, M_t: torch.Tensor, low_level: torch.Tensor) -> torch.Tensor:
        B, T = M_t.shape[:2]
        if T < 2:
            return torch.tensor(0.0, device=M_t.device)

        phi = low_level.detach().reshape(B, T, -1)
        phi = F.normalize(phi, p=2, dim=-1)

        total_loss = torch.tensor(0.0, device=M_t.device)
        n_pairs    = 0

        for t in range(T - 1):
            diff_norm = (phi[:, t] - phi[:, t + 1]).norm(dim=-1)
            w_t = torch.exp(-self.gamma * diff_norm)
            map_diff = (M_t[:, t] - M_t[:, t + 1]).pow(2).mean(dim=(-1, -2))
            total_loss = total_loss + (w_t * map_diff).mean()
            n_pairs   += 1

        return total_loss / n_pairs
