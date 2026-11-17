"""Reads REAL, measured evaluation metrics written by training/train_segmentation.py.

The /benchmark endpoint used to return hardcoded Dice/IoU/memory numbers. Now every number is either
measured (latency, parameter count, parameter memory) or read from models/segmentation_metrics.json,
and is reported as null / "not evaluated" when no evaluation has been run.
"""
import json
import os

from backend.app import config


def load_segmentation_metrics() -> dict:
    if not os.path.isfile(config.SEG_METRICS_PATH):
        return {}
    try:
        with open(config.SEG_METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_model_metrics(model_id: str):
    """-> dict with dice/iou/... or None."""
    return load_segmentation_metrics().get(model_id)


def load_classifier_metadata() -> dict:
    p = os.path.join(config.MODELS_DIR, "model_metadata.json")
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
