"""
Training script for the Brain Tumor MRI classifier (BRISC 2025).

Key improvements over the original script
  * ImageNet-PRETRAINED backbone by default (biggest single accuracy gain; use --no-pretrained to disable)
  * Warm-up stage: backbone frozen for --freeze-epochs, then full fine-tuning with a lower backbone LR
  * Checkpoint selection + early stopping on a held-out VALIDATION split (test set touched once, at the end)
  * Label smoothing, optional class-weighted loss, stronger augmentation, mixed precision on GPU
  * Temperature scaling fitted on validation logits -> calibrated confidences (stored in the checkpoint)
  * Honest final report: test accuracy, macro-F1, ECE (before / after calibration)

Example
  python training/train.py --backbone efficientnet_b0 --epochs 30 --device auto
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

from dataset import get_dataloaders, CLASS_NAMES  # noqa: E402
from model import BrainTumorClassifier, BACKBONES  # noqa: E402
from common.calibration import fit_temperature  # noqa: E402
from common.metrics import expected_calibration_error  # noqa: E402
from common.preprocessing import IMAGE_SIZE  # noqa: E402


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but CUDA is unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(name)


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        all_logits.append(model(images.to(device)).float().cpu())
        all_labels.append(labels)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = scaler is not None
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def eval_split(model, loader, device):
    """Plain (un-smoothed) CE + accuracy + macro-F1 for model selection."""
    logits, labels = collect_logits(model, loader, device)
    ce = nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels)).item()
    preds = logits.argmax(1)
    return ce, float((preds == labels).mean()), float(f1_score(labels, preds, average="macro")), logits, labels


def plot_metrics(history, output_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", color="#2563eb", linewidth=2)
    plt.plot(history["val_loss"], label="Val Loss", color="#dc2626", linewidth=2, linestyle="--")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.grid(True, linestyle=":", alpha=0.6); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc", color="#16a34a", linewidth=2)
    plt.plot(history["val_acc"], label="Val Acc", color="#9333ea", linewidth=2, linestyle="--")
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.grid(True, linestyle=":", alpha=0.6); plt.legend()
    plt.tight_layout(); plt.savefig(output_path, dpi=200); plt.close()


def run_training(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = pick_device(args.device)
    print("=" * 64)
    print("Brain Tumor Classifier Training")
    print(f"Device: {device} | Backbone: {args.backbone} | Pretrained: {not args.no_pretrained} | Epochs: {args.epochs}")
    print("=" * 64)
    os.makedirs(args.model_dir, exist_ok=True)

    # 1. Data ---------------------------------------------------------------------------
    d = get_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.workers,
                        subset_fraction=args.subset, val_fraction=args.val_fraction, seed=args.seed,
                        group_regex=args.group_regex, pin_memory=device.type == "cuda")
    train_loader, val_loader, test_loader = d["train_loader"], d["val_loader"], d["test_loader"]
    print(f"Split: {d['split_info']}")
    print(f"Train class counts: {dict(zip(CLASS_NAMES, d['class_counts']))}")

    # 2. Model --------------------------------------------------------------------------
    model = BrainTumorClassifier(num_classes=len(CLASS_NAMES), pretrained=not args.no_pretrained,
                                 backbone=args.backbone).to(device)
    pretrained_ok = bool(getattr(model, "pretrained_loaded", False))

    # 3. Loss / optimiser ---------------------------------------------------------------
    weight = None
    if args.class_weights:
        counts = np.maximum(np.array(d["class_counts"], dtype=np.float64), 1)
        weight = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32, device=device)
        print(f"Class weights: {weight.cpu().numpy().round(3).tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)

    optimizer = optim.AdamW([
        {"params": model.backbone_parameters(), "lr": args.lr * args.backbone_lr_mult},
        {"params": model.head_parameters(), "lr": args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # 4. Loop ---------------------------------------------------------------------------
    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "val_f1")}
    best = {"acc": -1.0, "loss": float("inf"), "state": None, "epoch": 0}
    patience_left = args.patience
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.set_backbone_trainable(epoch > args.freeze_epochs)
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        va_loss, va_acc, va_f1, _, _ = eval_split(model, val_loader, device)
        scheduler.step()

        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc, va_f1)):
            history[k].append(v)
        phase = "head-only" if epoch <= args.freeze_epochs else "full"
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] {phase:9s} ({time.time()-t0:.1f}s) "
              f"train loss {tr_loss:.4f} acc {tr_acc*100:.2f}% | val loss {va_loss:.4f} acc {va_acc*100:.2f}% F1 {va_f1:.3f}")

        improved = (va_acc > best["acc"] + 1e-9) or (abs(va_acc - best["acc"]) <= 1e-9 and va_loss < best["loss"])
        if improved:
            best.update(acc=va_acc, loss=va_loss, epoch=epoch,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            patience_left = args.patience
            print(f"  * new best on VALIDATION (acc {va_acc*100:.2f}%)")
        elif epoch > args.freeze_epochs:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch} (no val improvement for {args.patience} epochs)")
                break

    total_time = time.time() - start
    model.load_state_dict(best["state"])
    plot_metrics(history, os.path.join(args.model_dir, "training_curves.png"))

    # 5. Calibration on VALIDATION logits ---------------------------------------------------
    _, val_acc, val_f1, val_logits, val_labels = eval_split(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)
    print(f"Fitted temperature on validation set: T = {temperature:.3f}")
    if temperature <= 0.26 or temperature >= 9.9:
        print("WARNING: temperature hit its clamp boundary - the model is probably under-trained or the "
              "validation set is too small. Falling back to T = 1.0 (no calibration).")
        temperature = 1.0

    # 6. Single, final TEST evaluation ----------------------------------------------------
    _, test_acc, test_f1, test_logits, test_labels = eval_split(model, test_loader, device)
    sm = lambda z, T: torch.softmax(torch.from_numpy(z) / T, dim=1).numpy()
    ece_raw = expected_calibration_error(sm(test_logits, 1.0), test_labels)
    ece_cal = expected_calibration_error(sm(test_logits, temperature), test_labels)
    print(f"TEST  acc {test_acc*100:.2f}% | macro-F1 {test_f1:.4f} | ECE {ece_raw:.4f} -> {ece_cal:.4f} (after temperature scaling)")

    # 7. Save (plain python types + tensors only, so torch.load(weights_only=True) works) ------
    ckpt = {
        "epoch": int(best["epoch"]),
        "model_state_dict": best["state"],
        "backbone": args.backbone,
        "class_names": list(CLASS_NAMES),
        "image_size": list(IMAGE_SIZE),
        "temperature": float(temperature),
        "val_acc": float(val_acc),
        "pretrained_init": pretrained_ok,
    }
    torch.save(ckpt, os.path.join(args.model_dir, "best_model.pt"))

    metadata = {
        "backbone": args.backbone,
        "pretrained_init": pretrained_ok,
        "epochs_requested": args.epochs,
        "best_epoch": int(best["epoch"]),
        "best_val_acc": float(val_acc),
        "val_macro_f1": float(val_f1),
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_f1),
        "test_ece_raw": ece_raw,
        "test_ece_calibrated": ece_cal,
        "temperature": float(temperature),
        "label_smoothing": args.label_smoothing,
        "class_weights": bool(args.class_weights),
        "split": d["split_info"],
        "note": "Checkpoint chosen on validation split; test split evaluated once after training.",
        "classes": list(CLASS_NAMES),
        "image_size": list(IMAGE_SIZE),
        "total_training_time_sec": round(total_time, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.model_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved best_model.pt + model_metadata.json to {args.model_dir}/")
    print("Next: python training/evaluate.py   (full report with confusion matrix and confidence intervals)")
    return metadata


def build_parser():
    p = argparse.ArgumentParser(description="Train Brain Tumor Classifier")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4, help="LR for the new head (backbone uses lr * backbone-lr-mult)")
    p.add_argument("--backbone-lr-mult", type=float, default=0.3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--backbone", type=str, default="efficientnet_b0", choices=list(BACKBONES))
    p.add_argument("--no-pretrained", action="store_true", help="Train from random init (not recommended)")
    p.add_argument("--freeze-epochs", type=int, default=2, help="Epochs with the backbone frozen (head warm-up)")
    p.add_argument("--patience", type=int, default=8, help="Early-stopping patience (epochs) on validation accuracy")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--class-weights", action="store_true", help="Weight the loss by inverse class frequency")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--group-regex", type=str, default=None,
                   help="Regex with ONE capture group that extracts the patient id from a file stem "
                        "(used for patient-level splitting when there is no manifest.csv)")
    p.add_argument("--data-dir", type=str, default="dataset/classification")
    p.add_argument("--model-dir", type=str, default="models")
    p.add_argument("--subset", type=float, default=1.0, help="Fraction of data for quick smoke tests")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p


if __name__ == "__main__":
    run_training(build_parser().parse_args())
