"""Classifier loading + inference for the API.

Behavioural changes (see CHANGES.md):
  * Uses the SAME architecture as training (common/classifier_net.py). Before, the backend defined its own
    layout, and `load_state_dict(strict=False)` silently loaded nothing while reporting "configured".
  * Strict loading; class order is validated against the checkpoint.
  * NO fabricated predictions: if no valid checkpoint exists the classifier is reported as unavailable.
  * Temperature-calibrated probabilities with horizontal-flip TTA.
"""
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from common.classifier_net import BrainTumorClassifier  # noqa: E402
from common.preprocessing import CLASS_NAMES, classifier_tensor  # noqa: E402
from common.inference import classifier_probs_tta  # noqa: E402
from backend.app import config  # noqa: E402

CLASS_DISPLAY = {
    'glioma': 'Glioma',
    'meningioma': 'Meningioma',
    'no_tumor': 'No Tumour (Healthy)',
    'pituitary': 'Pituitary Tumour'
}

TUMOR_DETAILS = {
    'glioma': {
        'name': 'Glioma',
        'severity': 'High (Requires Oncological Staging)',
        'origin': 'Glial cells (Astrocytes, Oligodendrocytes, Ependymoma)',
        'description': 'Gliomas originate in the supportive glial cells of the central nervous system. They frequently exhibit diffuse, infiltrative borders with variable surrounding vasogenic edema.',
        'clinical_actions': 'Contrast MRI/Spectroscopy, neurosurgical oncology consult, stereotactic biopsy or maximal safe resection.'
    },
    'meningioma': {
        'name': 'Meningioma',
        'severity': 'Moderate (Typically Benign, Monitor Mass Effect)',
        'origin': 'Meningeal layers (Dura mater / Arachnoid cap cells)',
        'description': 'Meningiomas arise from the meningeal protective coverings of the brain. The majority (~85%) are WHO Grade I slow-growing lesions with distinct dural tails.',
        'clinical_actions': 'High-resolution volumetric MRI follow-up. Microsurgical excision or stereotactic radiosurgery for symptomatic or expanding tumors.'
    },
    'pituitary': {
        'name': 'Pituitary Tumour',
        'severity': 'Moderate (Endocrine & Visual Chiasm Evaluation)',
        'origin': 'Anterior pituitary gland (Sella turcica)',
        'description': 'Pituitary adenomas originate within the sella turcica. They may secrete active hormones (Prolactin, ACTH, GH) or cause bitemporal hemianopsia via optic chiasm compression.',
        'clinical_actions': 'Endocrinological hormonal panel, formal Goldman perimetry visual field mapping, transsphenoidal endoscopic surgery or medical therapy.'
    },
    'no_tumor': {
        'name': 'No Tumour (Normal Brain)',
        'severity': 'Normal / Negative',
        'origin': 'Healthy cerebral parenchyma',
        'description': 'No evidence of intracranial mass lesion, abnormal contrast enhancement, midline shift, or ventricular compression detected in this MRI slice.',
        'clinical_actions': 'Routine screening or symptom correlation if neurological complaints persist.'
    }
}


@dataclass
class ClassifierBundle:
    model: Optional[torch.nn.Module]
    is_configured: bool
    device: torch.device
    temperature: float = 1.0
    backbone: str = ""
    error: Optional[str] = None
    val_acc: Optional[float] = None


_cache: dict = {}


def resolve_device(device=None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if config.DEVICE in ("cpu", "cuda"):
        if config.DEVICE == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(config.DEVICE)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_classifier(device=None, path: Optional[str] = None) -> ClassifierBundle:
    device = resolve_device(device)
    path = path or config.CLASSIFIER_WEIGHTS_PATH

    if not os.path.exists(path):
        msg = f"Classifier checkpoint not found at {path}. Train one with training/train.py."
        print(f"[NeuroVision AI] {msg}")
        return ClassifierBundle(None, False, device, error=msg)
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
        names = list(ckpt.get("class_names", CLASS_NAMES))
        if names != list(CLASS_NAMES):
            raise ValueError(f"checkpoint class order {names} != expected {list(CLASS_NAMES)}")
        backbone = ckpt.get("backbone", "mobilenet_v3")
        model = BrainTumorClassifier(num_classes=len(CLASS_NAMES), pretrained=False, backbone=backbone)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)   # strict: never half-load silently
        model.to(device).eval()
        T = float(ckpt.get("temperature", 1.0))
        print(f"[NeuroVision AI] Loaded classifier '{backbone}' (T={T:.3f}) from {path}")
        return ClassifierBundle(model, True, device, T, backbone, None, ckpt.get("val_acc"))
    except Exception as e:
        msg = f"Classifier checkpoint at {path} could not be loaded: {e}"
        print(f"[NeuroVision AI] {msg}")
        return ClassifierBundle(None, False, device, error=msg)


def get_classifier_bundle(device=None) -> ClassifierBundle:
    if "bundle" not in _cache:
        _cache["bundle"] = load_classifier(device)
    return _cache["bundle"]


def reset_classifier_cache():
    _cache.clear()


def get_classifier(device=None):
    """Backward-compatible accessor -> (model_or_None, is_configured, device)."""
    b = get_classifier_bundle(device)
    return b.model, b.is_configured, b.device


def classify_image(img_rgb: np.ndarray, device=None, tta: Optional[bool] = None):
    """Returns calibrated class probabilities (np.ndarray, len 4) or None if no classifier is available."""
    b = get_classifier_bundle(device)
    if not b.is_configured:
        return None
    x = classifier_tensor(img_rgb).to(b.device)
    probs = classifier_probs_tta(b.model, x, b.temperature, config.USE_TTA if tta is None else tta)
    return probs[0]
