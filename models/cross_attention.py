"""
models/cross_attention.py — Cross-Attention Fusion with learnable temperature.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 8, temp_init: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        # FIX: Use passed temp_init instead of hardcoded log(4.0)
        self.log_temp = nn.Parameter(torch.tensor(float(temp_init)))

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, Q: torch.Tensor, S: torch.Tensor):
        B, T, L, d = Q.shape
        h = w = int(math.sqrt(L))

        Q_flat = Q.reshape(B * T, L, d)
        S_flat = S.reshape(B * T, L, d)

        Qp = self.q_proj(Q_flat)
        Kp = self.k_proj(S_flat)
        Vp = self.v_proj(S_flat)

        tau = torch.exp(self.log_temp).clamp(min=0.5, max=10.0)
        scores = torch.bmm(Qp, Kp.transpose(-2, -1)) / (self.scale * tau)
        A = F.softmax(scores, dim=-1)

        attended = self.out_proj(torch.bmm(A, Vp))

        M_flat = A.mean(dim=-2)
        M_t = M_flat.reshape(B, T, h, w)

        M_weights = M_flat.unsqueeze(-1)
        S_pool = (M_weights * Vp).sum(dim=1) / (M_weights.sum(dim=1) + 1e-8)
        attn_pool = S_pool.reshape(B, T, d).mean(dim=1)

        return M_t, attn_pool