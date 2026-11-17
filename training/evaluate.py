"""
Evaluation of a trained Brain Tumor classifier on the TEST split.

Reports (all computed, nothing hardcoded):
  accuracy with 95% bootstrap CI, macro-F1, per-class precision/recall/F1, one-vs-rest macro AUC,
  expected calibration error (raw vs temperature-scaled), confusion matrix, with and without TTA.
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dataset import get_dataloaders, CLASS_NAMES  # noqa: E402
from model import BrainTumorClassifier, CLASS_DISPLAY  # noqa: E402
from common.inference import classifier_probs_tta  # noqa: E402
from common.metrics import expected_calibration_error, bootstrap_ci  # noqa: E402


def load_checkpoint(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    names = list(ckpt.get("class_names", CLASS_NAMES))
    if names != list(CLASS_NAMES):
        raise ValueError(f"Checkpoint class order {names} != expected {CLASS_NAMES}")
    model = BrainTumorClassifier(num_classes=len(names), pretrained=False, backbone=ckpt.get("backbone", "mobilenet_v3"))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)      # strict: never silently half-load
    return model.to(device).eval(), ckpt


def predict_all(model, loader, device, temperature, tta):
    probs, labels = [], []
    for images, y in loader:
        probs.append(classifier_probs_tta(model, images.to(device), temperature, tta))
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def evaluate_model(model_path="models/best_model.pt", data_dir="dataset/classification", output_dir="models",
                   group_regex=None, no_tta=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at {model_path}. Train the model first (training/train.py).")
    model, ckpt = load_checkpoint(model_path, device)
    T = float(ckpt.get("temperature", 1.0))
    print(f"Loaded {ckpt.get('backbone')} | temperature {T:.3f} | device {device}")

    d = get_dataloaders(data_dir=data_dir, batch_size=32, group_regex=group_regex, pin_memory=device.type == "cuda")
    test_loader = d["test_loader"]
    display = [CLASS_DISPLAY.get(c, c) for c in CLASS_NAMES]
    results = {}

    for tta in ([False] if no_tta else [False, True]):
        probs, labels = predict_all(model, test_loader, device, T, tta)
        preds = probs.argmax(1)
        correct = (preds == labels).astype(float)
        lo, hi = bootstrap_ci(correct)
        try:
            auc = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
        except ValueError:
            auc = None
        tag = "tta" if tta else "single"
        results[tag] = {
            "accuracy": float(correct.mean()), "accuracy_ci95": [lo, hi],
            "macro_f1": float(f1_score(labels, preds, average="macro")), "macro_auc_ovr": auc,
            "ece": expected_calibration_error(probs, labels),
        }
        print(f"\n=== TEST ({'with TTA' if tta else 'single view'}) ===")
        print(f"accuracy {correct.mean()*100:.2f}%  (95% CI {lo*100:.2f}-{hi*100:.2f})  macro-F1 {results[tag]['macro_f1']:.4f}  "
              f"AUC {auc if auc is None else round(auc,4)}  ECE {results[tag]['ece']:.4f}")
        if tta or no_tta:
            print(classification_report(labels, preds, target_names=display, digits=4))
            cm = confusion_matrix(labels, preds)
            results[tag]["confusion_matrix"] = cm.tolist()
            results[tag]["per_class"] = classification_report(labels, preds, target_names=CLASS_NAMES, output_dict=True)
            import seaborn as sns
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=display, yticklabels=display, cbar=False)
            plt.title(f"Confusion Matrix - Test (acc {correct.mean()*100:.2f}%)", fontweight="bold")
            plt.xlabel("Predicted"); plt.ylabel("Ground truth"); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=200); plt.close()

    out = {"checkpoint": os.path.basename(model_path), "backbone": ckpt.get("backbone"), "temperature": T,
           "split": d["split_info"], "classes": list(CLASS_NAMES), "results": results}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "test_evaluation.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {os.path.join(output_dir, 'test_evaluation.json')} and confusion_matrix.png")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate Brain Tumor Classifier")
    ap.add_argument("--model-path", default="models/best_model.pt")
    ap.add_argument("--data-dir", default="dataset/classification")
    ap.add_argument("--output-dir", default="models")
    ap.add_argument("--group-regex", default=None)
    ap.add_argument("--no-tta", action="store_true")
    a = ap.parse_args()
    evaluate_model(a.model_path, a.data_dir, a.output_dir, a.group_regex, a.no_tta)
