import io
import base64
from contextlib import contextmanager

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from backend.app import config

_DROPOUT_TYPES = (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)


@contextmanager
def mc_dropout_mode(model: nn.Module):
    """Put ONLY the dropout layers in train mode; everything else (BatchNorm, InstanceNorm) stays in eval.

    The previous implementation called model.train(), which for BatchNorm-based models (FastBrain-UNet)
    switches to batch statistics with batch size 1 and also overwrites the stored running statistics.
    """
    was_training = model.training
    model.eval()
    droppers = [m for m in model.modules() if isinstance(m, _DROPOUT_TYPES)]
    for m in droppers:
        m.train()
    try:
        yield len(droppers)
    finally:
        model.train(was_training)


def estimate_mc_uncertainty(model, input_tensor, num_passes=None, device=None, target_shape=None, original_image=None):
    """
    Monte Carlo Dropout predictive uncertainty.
    Runs T stochastic forward passes (dropout active), then computes pixel-wise variance.
    Variance is mapped to [0,1] with a FIXED scale (config.UNCERTAINTY_VAR_SCALE) so scores are comparable
    between images. (Previously each image was rescaled by its own 99th percentile, which made every image
    look uncertain and the review gate meaningless.)

    Returns None when the model contains no dropout layers (uncertainty cannot be estimated).
    """
    num_passes = int(num_passes or config.MC_PASSES)

    preds = []
    with mc_dropout_mode(model) as n_dropout:
        if n_dropout == 0:
            return None
        with torch.no_grad():
            for _ in range(num_passes):
                preds.append(torch.sigmoid(model(input_tensor)).cpu().numpy().squeeze())

    preds_stack = np.stack(preds, axis=0)               # (T, H, W)
    mean_pred_224 = preds_stack.mean(axis=0)
    var_map_224 = preds_stack.var(axis=0)

    if target_shape is not None:
        h_out, w_out = target_shape[:2]
    elif original_image is not None:
        h_out, w_out = original_image.shape[:2]
    else:
        h_out, w_out = mean_pred_224.shape

    norm_var_224 = np.clip(var_map_224 / config.UNCERTAINTY_VAR_SCALE, 0.0, 1.0)
    norm_uncertainty = cv2.resize(norm_var_224, (w_out, h_out), interpolation=cv2.INTER_LINEAR)
    mean_pred = cv2.resize(mean_pred_224, (w_out, h_out), interpolation=cv2.INTER_LINEAR)

    # Uncertainty score focused on the predicted tumour boundary band
    binary_mask = (mean_pred > 0.5).astype(np.uint8)
    if binary_mask.sum() > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(binary_mask, kernel, iterations=3)
        eroded = cv2.erode(binary_mask, kernel, iterations=3)
        boundary_zone = (dilated - eroded) > 0
        boundary_uncertainty = float(norm_uncertainty[boundary_zone].mean()) if boundary_zone.sum() > 0 else 0.0
    else:
        boundary_uncertainty = float(norm_uncertainty.mean())

    active = norm_uncertainty > 0.08
    overall_uncertainty_score = float(norm_uncertainty[active].mean()) if active.any() else 0.0

    uncertainty_uint8 = np.uint8(255 * norm_uncertainty)
    heatmap_rgb = cv2.cvtColor(cv2.applyColorMap(uncertainty_uint8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

    overlay_base64 = None
    if original_image is not None:
        gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY) if original_image.ndim == 3 else original_image
        if gray.shape[:2] != (h_out, w_out):
            gray = cv2.resize(gray, (w_out, h_out))
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        alpha = np.clip((norm_uncertainty - 0.06) / 0.35, 0.0, 0.88)[:, :, np.newaxis]
        overlay_rgb = (heatmap_rgb * alpha + gray_3ch * (1.0 - alpha)).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(overlay_rgb).save(buf, format="PNG")
        overlay_base64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    buf = io.BytesIO()
    Image.fromarray(heatmap_rgb).save(buf, format="PNG")
    heatmap_base64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    boundary_pct = round(boundary_uncertainty * 100, 2)
    return {
        "available": True,
        "mean_prediction": mean_pred,
        "normalized_map": norm_uncertainty,
        "heatmap_base64": heatmap_base64,
        "overlay_base64": overlay_base64,
        "boundary_uncertainty": boundary_pct,
        "overall_uncertainty_score": round(overall_uncertainty_score * 100, 2),
        "num_stochastic_passes": num_passes,
        "threshold_pct": config.UNCERTAINTY_THRESHOLD_PCT,
        "review_recommended": bool(boundary_pct > config.UNCERTAINTY_THRESHOLD_PCT),
    }
