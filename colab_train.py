"""
Google Colab / one-command launcher for BRISC 2025.

This file used to contain a full copy of the old training recipe (its own architecture, test set used
for checkpoint selection, from-scratch training). It now just downloads the data and delegates to the
maintained scripts, so there is exactly ONE implementation of the model and the training loop:

    training/train.py               -> classifier  (pretrained backbone, validation split, calibration)
    training/train_segmentation.py  -> nnU-Net / FastBrain-UNet

Colab usage (run from the repo root, with the whole repository uploaded or cloned):
    !pip -q install -r requirements.txt
    !python colab_train.py                       # classifier only
    !python colab_train.py --with-segmentation   # + nnU-Net + distilled FastBrain-UNet
"""
import os
import sys
import zipfile
import argparse
import subprocess
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_URL = "https://ndownloader.figshare.com/files/59298329"


def ensure_dataset():
    if os.path.isdir("brisc2025"):
        return
    print("Downloading BRISC 2025 ...")
    urllib.request.urlretrieve(DATA_URL, "brisc2025.zip")
    with zipfile.ZipFile("brisc2025.zip") as z:
        z.extractall(".")


def sh(*cmd):
    print("\n$", " ".join(cmd))
    subprocess.run([sys.executable, *cmd], check=True, cwd=ROOT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--with-segmentation", action="store_true")
    ap.add_argument("--group-regex", default=None, help="regex (1 capture group) extracting the patient id from file names")
    a = ap.parse_args()

    os.chdir(ROOT)
    ensure_dataset()
    extra = ["--group-regex", a.group_regex] if a.group_regex else []
    sh("training/train.py", "--data-dir", "brisc2025/classification_task", "--backbone", a.backbone,
       "--epochs", str(a.epochs), "--class-weights", *extra)
    sh("training/evaluate.py", "--data-dir", "brisc2025/classification_task", *extra)
    if a.with_segmentation:
        sh("training/train_segmentation.py", "--model", "nnunet", "--data-dir", "brisc2025/segmentation_task", *extra)
        sh("training/train_segmentation.py", "--model", "fast", "--teacher", "models/best_nnunet_model.pt",
           "--data-dir", "brisc2025/segmentation_task", *extra)
    print("\nDone. Download models/ (best_model.pt, best_nnunet_model.pt, fast_brain_unet.pt, *.json) into your project's models/ folder.")
