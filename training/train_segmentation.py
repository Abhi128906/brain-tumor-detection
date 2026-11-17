"""
Train / evaluate the segmentation models (nnU-Net 2D and FastBrain-UNet) on BRISC 2025.

Why this script exists
  * The repo only had a Colab notebook for segmentation: no augmentation, test set used for checkpoint
    selection, no patient-level split, and FastBrain-UNet was never trained at all (random weights).
  * It writes REAL measured metrics to models/segmentation_metrics.json, which the API's /benchmark
    endpoint reads (it used to return hardcoded Dice / IoU / memory numbers).

Usage
  # 1) retrain the main model (augmentation, Dice+BCE, MC-dropout-ready, early stopping on VALIDATION)
  python training/train_segmentation.py --model nnunet --epochs 40

  # 2) train the lightweight edge model, distilled from the nnU-Net teacher
  python training/train_segmentation.py --model fast --teacher models/best_nnunet_model.pt --epochs 40

  # 3) just measure the weights you already have (writes models/segmentation_metrics.json)
  python training/train_segmentation.py --model nnunet --eval-only
"""

import os
import sys
import json
import time
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.preprocessing import SEG_SIZE, segmenter_normalize_np  # noqa: E402
from common.inference import segmenter_prob_tta_batch  # noqa: E402
from common.metrics import dice_iou, hd95, bootstrap_ci  # noqa: E402
from backend.app.models.segmenter import nnUNet2D  # noqa: E402
from backend.app.models.lightweight_segmenter import FastBrainUNet  # noqa: E402
from backend.app.services.postprocess import postprocess_mask  # noqa: E402
from dataset import infer_groups, split_train_val  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png")
DATA_CANDIDATES = ["brisc2025/segmentation_task", "dataset/segmentation", "segmentation_task"]


# ----------------------------------------------------------------------------- data
def find_data_dir(user_dir=None):
    for d in ([user_dir] if user_dir else DATA_CANDIDATES):
        if d and os.path.isdir(os.path.join(d, "train", "images")) and os.path.isdir(os.path.join(d, "train", "masks")):
            return d
    raise FileNotFoundError(
        f"Segmentation dataset not found. Looked for train/images + train/masks under: "
        f"{[user_dir] if user_dir else DATA_CANDIDATES}. Pass --data-dir.")


def list_pairs(split_dir):
    img_dir, mask_dir = os.path.join(split_dir, "images"), os.path.join(split_dir, "masks")
    pairs = []
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith(IMG_EXT):
            continue
        base = os.path.splitext(f)[0]
        cands = [base + e for e in (".png", ".jpg", ".jpeg")] + [base + "_mask" + e for e in (".png", ".jpg")]
        mp = next((os.path.join(mask_dir, c) for c in cands if os.path.exists(os.path.join(mask_dir, c))), None)
        pairs.append((os.path.join(img_dir, f), mp))     # mp None => genuinely empty mask (healthy slice)
    return pairs


class SegDataset(Dataset):
    """Loads (image, mask) pairs. Preprocessing matches inference: PIL resize -> [0,1] -> per-image z-score."""

    def __init__(self, pairs, size=SEG_SIZE, train=False, seed=0):
        self.pairs, self.size, self.train = pairs, size, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.pairs)

    def _load(self, idx):
        ip, mp = self.pairs[idx]
        img = np.asarray(Image.open(ip).convert("RGB").resize(self.size), dtype=np.float32) / 255.0
        if mp:
            m = np.asarray(Image.open(mp).convert("L").resize(self.size, Image.NEAREST), dtype=np.float32)
            mask = (m > 127).astype(np.float32)
        else:
            mask = np.zeros(self.size[::-1], dtype=np.float32)
        return img, mask

    def _augment(self, img, mask):
        r = self.rng
        h, w = mask.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), r.uniform(-15, 15), r.uniform(0.9, 1.1))
        M[:, 2] += (r.uniform(-0.05, 0.05) * w, r.uniform(-0.05, 0.05) * h)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if r.random() < 0.5:                                   # left/right flip only (anatomically plausible)
            img, mask = img[:, ::-1], mask[:, ::-1]
        img = np.clip(img, 0, 1) ** r.uniform(0.8, 1.25)         # gamma
        img = np.clip(img * r.uniform(0.85, 1.15) + r.uniform(-0.05, 0.05), 0, 1)
        img = np.clip(img + r.normal(0, r.uniform(0, 0.02), img.shape), 0, 1)
        return np.ascontiguousarray(img, dtype=np.float32), np.ascontiguousarray(mask, dtype=np.float32)

    def __getitem__(self, idx):
        img, mask = self._load(idx)
        if self.train:
            img, mask = self._augment(img, mask)
        img = segmenter_normalize_np(img)
        return (torch.from_numpy(img).permute(2, 0, 1).float(), torch.from_numpy(mask).unsqueeze(0).float())


def make_splits(data_dir, val_fraction, seed, subset, group_regex):
    train_pairs = list_pairs(os.path.join(data_dir, "train"))
    test_dir = os.path.join(data_dir, "test")
    test_pairs = list_pairs(test_dir) if os.path.isdir(os.path.join(test_dir, "images")) else []
    labels = np.array([int(mp is not None and np.asarray(Image.open(mp).convert("L")).max() > 127) for _, mp in train_pairs])
    samples = [(ip, 0) for ip, _ in train_pairs]
    groups = infer_groups(data_dir, samples, group_regex)
    if groups is None:
        print("WARNING: no patient ids (manifest.csv / --group-regex) - validation split is slice-level and may be optimistic.")
    tr, va = split_train_val(samples, labels, groups, val_fraction, seed)
    if subset < 1.0:
        rng = np.random.default_rng(seed)
        tr = np.sort(rng.choice(tr, max(1, int(len(tr) * subset)), replace=False))
        va = np.sort(rng.choice(va, max(1, int(len(va) * subset)), replace=False))
        test_pairs = [test_pairs[i] for i in sorted(rng.choice(len(test_pairs), max(1, int(len(test_pairs) * subset)), replace=False))] if test_pairs else []
    info = {"group_aware": groups is not None, "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": len(test_pairs)}
    return [train_pairs[i] for i in tr], [train_pairs[i] for i in va], test_pairs, info


# ----------------------------------------------------------------------------- loss / eval
def dice_loss(logits, target, smooth=1e-5):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(2, 3))
    union = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1.0 - ((2 * inter + smooth) / (union + smooth)).mean()


@torch.no_grad()
def evaluate(model, loader, device, tta=True, postprocess=True, with_hd=False):
    """Evaluates the DEPLOYED pipeline (flip-TTA + connected-component clean-up) at 224x224."""
    model.eval()
    dices, ious, hds, tumour_dices, has_tumour = [], [], [], [], []
    for x, y in loader:
        probs = segmenter_prob_tta_batch(model, x.to(device), tta)
        y = y.squeeze(1).numpy()
        for p, t in zip(probs, y):
            pred = postprocess_mask(p, 0.5, min_area_px=20) if postprocess else (p > 0.5).astype(np.uint8)
            d, i = dice_iou(pred, t)
            dices.append(d); ious.append(i); has_tumour.append(bool(t.sum() > 0))
            if t.sum() > 0:
                tumour_dices.append(d)
                if with_hd:
                    h = hd95(pred, t)
                    if h is not None:
                        hds.append(h)
    return {"dices": np.array(dices), "ious": np.array(ious), "tumour_dices": np.array(tumour_dices),
            "hd95": np.array(hds), "has_tumour": np.array(has_tumour)}


def summarize(r):
    lo, hi = bootstrap_ci(r["dices"])
    fp_rate = None
    neg = ~r["has_tumour"]
    if neg.any():                                  # healthy slices where the model still predicted a lesion
        fp_rate = float((r["dices"][neg] < 1.0).mean())
    return {
        "dice": float(r["dices"].mean()), "dice_ci95": [lo, hi],
        "dice_tumour_only": float(r["tumour_dices"].mean()) if len(r["tumour_dices"]) else None,
        "iou": float(r["ious"].mean()),
        "hd95_px": float(np.median(r["hd95"])) if len(r["hd95"]) else None,
        "false_positive_rate_on_empty_slices": fp_rate,
        "n_slices": int(len(r["dices"])), "n_tumour_slices": int(r["has_tumour"].sum()),
    }


def update_metrics_file(path, model_id, entry):
    data = {}
    if os.path.isfile(path):
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception:
            data = {}
    data[model_id] = entry
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ----------------------------------------------------------------------------- main
def build_model(kind):
    return nnUNet2D(in_channels=3, num_classes=1, mc_dropout=0.2) if kind == "nnunet" else FastBrainUNet(3, 1)


def load_weights(model, path, device):
    obj = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj, strict=True)


def run(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    data_dir = find_data_dir(args.data_dir)
    model_id = "nnunet-2d" if args.model == "nnunet" else "fast-brain-unet"
    out_path = args.out or os.path.join("models", "best_nnunet_model.pt" if args.model == "nnunet" else "fast_brain_unet.pt")
    metrics_path = os.path.join(os.path.dirname(out_path) or ".", "segmentation_metrics.json")
    print(f"Device {device} | model {args.model} | data {data_dir}")

    train_pairs, val_pairs, test_pairs, split = make_splits(data_dir, args.val_fraction, args.seed, args.subset, args.group_regex)
    print(f"Split: {split}")
    mk = lambda p, tr, sh: DataLoader(SegDataset(p, train=tr, seed=args.seed), batch_size=args.batch_size, shuffle=sh,
                                      num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=tr and len(p) > args.batch_size)
    train_loader, val_loader = mk(train_pairs, True, True), mk(val_pairs, False, False)
    test_loader = mk(test_pairs, False, False) if test_pairs else None

    model = build_model(args.model).to(device)

    if not args.eval_only:
        teacher = None
        if args.teacher:
            teacher = nnUNet2D(3, 1).to(device); load_weights(teacher, args.teacher, device); teacher.eval()
            print(f"Distilling from teacher {args.teacher} (kd_weight={args.kd_weight})")
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
        bce = nn.BCEWithLogitsLoss()
        best, best_state, patience = -1.0, None, args.patience
        t_start = time.time()
        for epoch in range(1, args.epochs + 1):
            model.train(); run_loss = 0.0; t0 = time.time()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=scaler is not None):
                    logits = model(x)
                    loss = 0.5 * bce(logits, y) + 0.5 * dice_loss(logits, y)
                    if teacher is not None:
                        with torch.no_grad():
                            soft = torch.sigmoid(teacher(x))
                        loss = loss + args.kd_weight * F.binary_cross_entropy_with_logits(logits, soft)
                if scaler:
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                else:
                    loss.backward(); opt.step()
                run_loss += loss.item()
            sched.step()
            v = summarize(evaluate(model, val_loader, device, tta=False, postprocess=False))
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({time.time()-t0:.1f}s) loss {run_loss/max(len(train_loader),1):.4f} "
                  f"| VAL dice {v['dice']:.4f} (tumour-only {v['dice_tumour_only'] if v['dice_tumour_only'] is None else round(v['dice_tumour_only'],4)}) IoU {v['iou']:.4f}")
            if v["dice"] > best:
                best = v["dice"]; patience = args.patience
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
                print(f"  * new best on VALIDATION (dice {best:.4f})")
            else:
                patience -= 1
                if patience <= 0:
                    print(f"Early stopping at epoch {epoch}"); break
        model.load_state_dict(best_state)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        torch.save(best_state, out_path)                        # plain state_dict: matches the backend loader
        print(f"Saved weights -> {out_path} (trained {(time.time()-t_start)/60:.1f} min)")
    else:
        if not os.path.exists(out_path):
            raise FileNotFoundError(f"--eval-only: no weights at {out_path}")
        load_weights(model, out_path, device)

    # ---- single, final evaluation on the held-out TEST split (deployment pipeline: TTA + post-processing)
    if test_loader is None:
        print("No test split found; evaluating on validation split instead."); test_loader, which = val_loader, "val"
    else:
        which = "test"
    entry = summarize(evaluate(model, test_loader, device, tta=True, postprocess=True, with_hd=True))
    entry.update({
        "split_evaluated": which, "tta": "hflip", "postprocessing": "largest-components + hole filling",
        "resolution": list(SEG_SIZE), "weights": os.path.basename(out_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "data_split": split,
        "provenance": ("eval-only on existing weights: if they were originally selected on this split, "
                       "the score is optimistic" if args.eval_only else
                       "trained here; checkpoint selected on validation, test evaluated once"),
    })
    print(json.dumps({k: entry[k] for k in ("dice", "dice_ci95", "dice_tumour_only", "iou", "hd95_px",
                                            "false_positive_rate_on_empty_slices", "n_slices")}, indent=2))
    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
    update_metrics_file(metrics_path, model_id, entry)
    print(f"Metrics written to {metrics_path}")


def build_parser():
    p = argparse.ArgumentParser(description="Train / evaluate segmentation models")
    p.add_argument("--model", choices=["nnunet", "fast"], default="nnunet")
    p.add_argument("--data-dir", default=None, help="folder containing train/{images,masks} and test/{images,masks}")
    p.add_argument("--out", default=None, help="weights path (default models/best_nnunet_model.pt or models/fast_brain_unet.pt)")
    p.add_argument("--teacher", default=None, help="nnU-Net weights to distill from (for --model fast)")
    p.add_argument("--kd-weight", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--group-regex", default=None)
    p.add_argument("--subset", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--eval-only", action="store_true", help="skip training; measure existing weights and write metrics")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
