"""Segmentation / classification metrics shared by training + evaluation scripts."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def dice_iou(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6):
    """Return (dice, iou) for binary masks. Empty-vs-empty counts as a perfect 1.0."""
    p = pred > 0
    t = target > 0
    inter = float(np.logical_and(p, t).sum())
    p_sum, t_sum = float(p.sum()), float(t.sum())
    if p_sum == 0 and t_sum == 0:
        return 1.0, 1.0
    dice = (2 * inter + eps) / (p_sum + t_sum + eps)
    iou = (inter + eps) / (p_sum + t_sum - inter + eps)
    return dice, iou


def hd95(pred: np.ndarray, target: np.ndarray):
    """95th-percentile symmetric Hausdorff distance in pixels. None if either mask is empty."""
    p, t = pred > 0, target > 0
    if not p.any() or not t.any():
        return None
    dt_t = ndi.distance_transform_edt(~t)
    dt_p = ndi.distance_transform_edt(~p)
    p_edge = p & ~ndi.binary_erosion(p)
    t_edge = t & ~ndi.binary_erosion(t)
    d_pt = dt_t[p_edge]
    d_tp = dt_p[t_edge]
    if d_pt.size == 0 or d_tp.size == 0:
        return None
    return float(max(np.percentile(d_pt, 95), np.percentile(d_tp, 95)))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE of the top-1 prediction. probs: (N, C), labels: (N,)"""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI of the mean of per-sample values (e.g. correctness or Dice)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    means = np.array([values[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))
