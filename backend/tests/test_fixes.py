"""Regression tests for the accuracy / honesty fixes (run: pytest backend/tests)."""
import os
import sys
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (BASE_DIR, os.path.join(BASE_DIR, "training")):
    if p not in sys.path:
        sys.path.insert(0, p)

SAMPLE = os.path.join(BASE_DIR, "app", "static", "samples", "sample_glioma.jpg")


def _sample():
    if not os.path.exists(SAMPLE):
        pytest.skip("sample image missing")
    return np.array(Image.open(SAMPLE).convert("RGB"))


# ---- fix: classifier preprocessing identical between training and inference -------------------
def test_classifier_preprocessing_matches_training_transform():
    from common.preprocessing import classifier_tensor
    from dataset import get_transforms
    img = _sample()
    _, eval_tf = get_transforms()
    expected = eval_tf(Image.fromarray(img)).unsqueeze(0)
    got = classifier_tensor(img)
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-4), float((got - expected).abs().max())


def test_segmenter_preprocessing_is_zscored_per_image():
    from common.preprocessing import segmenter_tensor
    x = segmenter_tensor(_sample())
    assert x.shape == (1, 3, 224, 224)
    assert abs(float(x.mean())) < 1e-3 and abs(float(x.std()) - 1.0) < 1e-2


# ---- fix: patient-level split without leakage ---------------------------------------------------
def test_group_split_has_no_patient_leakage():
    from dataset import split_train_val
    rng = np.random.default_rng(0)
    n_patients, per = 40, 6
    groups = [f"P{p}" for p in range(n_patients) for _ in range(per)]
    labels = np.array([p % 4 for p in range(n_patients) for _ in range(per)])
    samples = list(range(len(groups)))
    tr, va = split_train_val(samples, labels, groups, 0.2, seed=1)
    g = np.array(groups)
    assert set(g[tr]).isdisjoint(set(g[va]))
    assert len(va) > 0 and len(tr) > len(va)


# ---- fix: calibration --------------------------------------------------------------------------
def test_temperature_scaling_reduces_nll_for_overconfident_model():
    from common.calibration import fit_temperature
    torch.manual_seed(0)
    labels = torch.randint(0, 4, (500,))
    logits = torch.randn(500, 4) * 4.0
    logits[torch.arange(500), labels] += 6.0
    flip = torch.rand(500) < 0.25                     # 25% wrong but very confident -> overconfident
    logits[flip] = torch.randn(int(flip.sum()), 4) * 4.0 + 6.0 * torch.nn.functional.one_hot(
        torch.randint(0, 4, (int(flip.sum()),)), 4)
    T = fit_temperature(logits, labels)
    nll = lambda t: torch.nn.functional.cross_entropy(logits / t, labels).item()
    assert T > 1.0
    assert nll(T) < nll(1.0)


# ---- fix: backend loads classifier strictly, never half-loads ------------------------------------
def test_classifier_checkpoint_round_trip_and_strictness():
    from common.classifier_net import BrainTumorClassifier
    from backend.app.models.classifier import load_classifier
    m = BrainTumorClassifier(4, pretrained=False, backbone="mobilenet_v3")
    with tempfile.TemporaryDirectory() as d:
        good = os.path.join(d, "good.pt")
        torch.save({"model_state_dict": m.state_dict(), "backbone": "mobilenet_v3",
                    "class_names": ["glioma", "meningioma", "no_tumor", "pituitary"], "temperature": 1.7}, good)
        b = load_classifier("cpu", good)
        assert b.is_configured and abs(b.temperature - 1.7) < 1e-9

        bad_sd = {("model." + k): v for k, v in m.state_dict().items()}      # the OLD mismatching key layout
        bad = os.path.join(d, "bad.pt")
        torch.save({"model_state_dict": bad_sd, "backbone": "mobilenet_v3",
                    "class_names": ["glioma", "meningioma", "no_tumor", "pituitary"]}, bad)
        b2 = load_classifier("cpu", bad)
        assert not b2.is_configured and b2.error          # previously: silently "configured" with random weights

        wrong_order = os.path.join(d, "order.pt")
        torch.save({"model_state_dict": m.state_dict(), "backbone": "mobilenet_v3",
                    "class_names": ["meningioma", "glioma", "no_tumor", "pituitary"]}, wrong_order)
        assert not load_classifier("cpu", wrong_order).is_configured


def test_missing_checkpoint_reports_unavailable():
    from backend.app.models.classifier import load_classifier
    b = load_classifier("cpu", "/nonexistent/best_model.pt")
    assert not b.is_configured and b.model is None


# ---- fix: MC dropout must not touch BatchNorm ------------------------------------------------
def test_mc_dropout_keeps_batchnorm_frozen():
    from backend.app.models.lightweight_segmenter import FastBrainUNet
    from backend.app.services.uncertainty import estimate_mc_uncertainty
    m = FastBrainUNet().eval()
    bns = [x for x in m.modules() if isinstance(x, torch.nn.BatchNorm2d)]
    before = [b.running_mean.clone() for b in bns]
    out = estimate_mc_uncertainty(m, torch.randn(1, 3, 224, 224), num_passes=3, original_image=np.zeros((224, 224, 3), np.uint8))
    assert out is not None and out["num_stochastic_passes"] == 3
    assert all(torch.equal(a, b.running_mean) for a, b in zip(before, bns))
    assert not m.training


def test_mc_dropout_returns_none_without_dropout_layers():
    from backend.app.services.uncertainty import estimate_mc_uncertainty
    m = torch.nn.Sequential(torch.nn.Conv2d(3, 1, 3, padding=1))
    assert estimate_mc_uncertainty(m, torch.randn(1, 3, 32, 32)) is None


# ---- fix: mask post-processing --------------------------------------------------------------
def test_postprocess_removes_speckle_keeps_main_lesion_and_fills_holes():
    from backend.app.services.postprocess import postprocess_mask
    p = np.zeros((100, 100), np.float32)
    p[30:70, 30:70] = 0.9
    p[45:50, 45:50] = 0.1          # hole inside the lesion
    p[5:7, 5:7] = 0.95             # isolated speckle
    out = postprocess_mask(p, 0.5, min_area_px=20)
    assert out[50, 50] == 1 and out[6, 6] == 0 and out[47, 47] == 1


def test_untrained_fast_model_is_refused_not_served_with_random_weights():
    from backend.app.models import lightweight_segmenter as ls
    from backend.app.models.segmenter import ModelUnavailableError
    from backend.app import config
    if os.path.exists(config.FASTBRAIN_WEIGHTS_PATH):
        pytest.skip("fast model weights exist")
    ls._lightweight_cache.clear()
    with pytest.raises(ModelUnavailableError):
        ls.get_lightweight_segmenter("cpu")
