"""
xai/sanity_checks.py — Adebayo et al. 2018 sanity checks for explanation faithfulness.
"""

import copy
import torch
import numpy as np
from typing import Optional


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    norm_a = np.linalg.norm(a_flat) + 1e-8
    norm_b = np.linalg.norm(b_flat) + 1e-8
    return float(np.dot(a_flat / norm_a, b_flat / norm_b))


def _get_M_t(model, frames: torch.Tensor) -> np.ndarray:
    device = next(model.parameters()).device
    with torch.no_grad():
        out = model(frames.to(device))
    return out.M_t_up[0].cpu().numpy()


def model_randomization_check(
    model,
    frames: torch.Tensor,
    n_random: int = 3,
) -> float:
    original_M_t = _get_M_t(model, frames)

    model_copy = copy.deepcopy(model)
    model_copy.eval()

    named_params = [
        (name, param)
        for name, param in model_copy.named_parameters()
        if param.requires_grad and param.dim() >= 1
    ]

    if not named_params:
        return 1.0

    step_size = max(1, len(named_params) // max(n_random, 1))
    cascade_positions = list(range(step_size - 1, len(named_params), step_size))[:n_random]

    sims = []
    for pos in cascade_positions:
        for i in range(pos + 1):
            name, param = named_params[i]
            torch.nn.init.normal_(param.data)

        randomized_M_t = _get_M_t(model_copy, frames)
        sims.append(_cosine_sim(original_M_t, randomized_M_t))

    return float(np.mean(sims)) if sims else 1.0


def label_randomization_check(
    model,
    train_loader,
    config,
    n_batches: int = 5,
) -> Optional[float]:
    return None
