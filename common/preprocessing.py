"""Single source of truth for model input preprocessing.

Classifier  : PIL bilinear resize -> [0,1] -> ImageNet mean/std       (same as training)
Segmenter   : PIL default resize  -> [0,1] -> per-image z-score        (same as notebook training)

Previously the backend resized with cv2 and fed BOTH models the segmenter's z-scored tensor,
which silently mismatched the classifier's training distribution.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
IMAGE_SIZE = (224, 224)          # classifier input
SEG_SIZE = (224, 224)            # segmenter input

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_pil_rgb(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return Image.fromarray(arr).convert("RGB")


def classifier_tensor(img, size=IMAGE_SIZE) -> torch.Tensor:
    """-> (1, 3, H, W) float tensor, ImageNet-normalised. Matches training transforms."""
    pil = _to_pil_rgb(img).resize((size[1], size[0]), Image.BILINEAR)
    x = np.asarray(pil, dtype=np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).contiguous()


def segmenter_tensor(img, size=SEG_SIZE) -> torch.Tensor:
    """-> (1, 3, H, W) float tensor, per-image z-score. Matches nnU-Net training."""
    pil = _to_pil_rgb(img).resize((size[1], size[0]))       # PIL default, as in training
    x = np.asarray(pil, dtype=np.float32) / 255.0
    x = (x - x.mean()) / (x.std() + 1e-6)
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).contiguous()


def segmenter_normalize_np(img_np_01: np.ndarray) -> np.ndarray:
    """z-score for an already-resized float image in [0,1] (used by the seg Dataset)."""
    return (img_np_01 - img_np_01.mean()) / (img_np_01.std() + 1e-6)
