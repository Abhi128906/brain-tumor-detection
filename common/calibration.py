"""Temperature scaling: makes the softmax confidence an honest probability.

Fit ONE scalar T on validation logits (never on test), then use softmax(logits / T) everywhere.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits: np.ndarray | torch.Tensor, labels: np.ndarray | torch.Tensor,
                    max_iter: int = 200) -> float:
    """Minimise NLL over log T with LBFGS. Returns T clamped to a sane range [0.25, 10]."""
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(np.clip(log_t.exp().item(), 0.25, 10.0))


def softmax_with_temperature(logits, temperature: float = 1.0):
    """Works for torch tensors (returns tensor)."""
    return F.softmax(logits / max(float(temperature), 1e-3), dim=-1)
