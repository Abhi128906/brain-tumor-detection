"""Test-time augmentation (TTA) helpers shared by the backend and the evaluation scripts."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def classifier_probs_tta(model, x: torch.Tensor, temperature: float = 1.0, tta: bool = True) -> np.ndarray:
    """x: (B,3,H,W) normalised. Returns (B,C) calibrated probabilities averaged over TTA views.

    Views: identity + horizontal flip (tumour *type* does not depend on left/right).
    """
    model.eval()
    views = [x, torch.flip(x, dims=[3])] if tta else [x]
    t = max(float(temperature), 1e-3)
    probs = torch.stack([F.softmax(model(v) / t, dim=1) for v in views]).mean(dim=0)
    return probs.cpu().numpy()


@torch.no_grad()
def segmenter_prob_tta_batch(model, x: torch.Tensor, tta: bool = True) -> np.ndarray:
    """x: (B,3,H,W). Returns (B,H,W) tumour probability averaged over identity + horizontal-flip views.

    Only the horizontal flip is used: brains are approximately left/right symmetric, whereas an
    up/down-flipped brain is out-of-distribution for models trained without vertical-flip augmentation.
    """
    model.eval()
    p = torch.sigmoid(model(x))
    n = 1
    if tta:
        p = p + torch.flip(torch.sigmoid(model(torch.flip(x, dims=[3]))), dims=[3])
        n = 2
    return (p / n).squeeze(1).cpu().numpy()


def segmenter_prob_tta(model, x: torch.Tensor, tta: bool = True) -> np.ndarray:
    """Single image convenience wrapper: x (1,3,H,W) -> (H,W)."""
    return segmenter_prob_tta_batch(model, x, tta)[0]
