"""Central runtime configuration. Values come from environment variables (or a .env file at the repo root).

Previously .env.example listed settings that nothing actually read; now they are honoured.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_dotenv(path):
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(os.path.join(BASE_DIR, ".env"))


def _path(env_name, default_rel):
    p = os.environ.get(env_name, default_rel)
    return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)


def _float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return int(default)


def _bool(name, default):
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


MODELS_DIR = os.path.join(BASE_DIR, "models")
NNUNET_WEIGHTS_PATH = _path("NNUNET_WEIGHTS_PATH", "models/best_nnunet_model.pt")
FASTBRAIN_WEIGHTS_PATH = _path("FASTBRAIN_WEIGHTS_PATH", "models/fast_brain_unet.pt")
CLASSIFIER_WEIGHTS_PATH = _path("CLASSIFIER_WEIGHTS_PATH", "models/best_model.pt")
SEG_METRICS_PATH = os.path.join(MODELS_DIR, "segmentation_metrics.json")

DEVICE = os.environ.get("DEVICE", "auto")                    # auto | cpu | cuda
MC_PASSES = _int("DEFAULT_MC_PASSES", 8)
UNCERTAINTY_THRESHOLD_PCT = _float("UNCERTAINTY_THRESHOLD", 35.0)
UNCERTAINTY_VAR_SCALE = _float("UNCERTAINTY_VAR_SCALE", 0.06)  # fixed variance->[0,1] scale (same for every image)
SEG_THRESHOLD = _float("SEG_THRESHOLD", 0.5)
ABSTAIN_THRESHOLD = _float("ABSTAIN_THRESHOLD", 0.60)         # calibrated top-1 prob below this => "uncertain"
LESION_MIN_FRACTION = _float("LESION_MIN_FRACTION", 0.002)    # fraction of the image that counts as a real lesion
USE_TTA = _bool("USE_TTA", True)
MAX_UPLOAD_MB = _int("MAX_UPLOAD_SIZE_MB", 100)
MAX_STORED_RESULTS = _int("MAX_STORED_RESULTS", 50)
