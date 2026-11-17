"""
Dataset and DataLoader utilities for the BRISC Brain Tumor MRI classification task.

What changed vs. the original version (and why):
  * A real VALIDATION split is carved out of the training folder. Checkpoints / early stopping are
    chosen on it, so the TEST folder is touched exactly once, at the very end. (Previously the test
    set was used for model selection, which makes the reported accuracy optimistic.)
  * The split is group-aware when patient ids are available (manifest.csv or --group-regex), so
    slices from one patient never appear on both sides of the split (patient-level leakage).
  * Stronger augmentation (random resized crop, affine, intensity jitter, random erasing).
  * Class counts are returned so the loss can be class-weighted.
"""

import os
import re
import sys
import csv
import warnings
from collections import Counter

import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.preprocessing import CLASS_NAMES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

# Kept for backwards compatibility (app/server.py imports these names)
NORM_MEAN = list(IMAGENET_MEAN)
NORM_STD = list(IMAGENET_STD)

_FILE_COLS = ("filename", "file_name", "file", "image", "image_name", "image_path", "path")
_GROUP_COLS = ("patient_id", "patient", "subject_id", "subject", "case_id", "study_id")


def get_transforms(image_size=IMAGE_SIZE):
    """Returns (train_transform, eval_transform)."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.25),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.08), value="random"),
    ])
    # Must stay identical to common.preprocessing.classifier_tensor (verified in the unit tests)
    eval_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    return train_transform, eval_transform


# ----------------------------------------------------------------------------- patient groups
def _read_manifest(data_dir):
    for cand in (os.path.join(data_dir, "manifest.csv"),
                 os.path.join(os.path.dirname(data_dir.rstrip("/\\")), "manifest.csv"),
                 os.path.join(os.path.dirname(os.path.dirname(data_dir.rstrip("/\\"))), "manifest.csv")):
        if os.path.isfile(cand):
            return cand
    return None


def infer_groups(data_dir, samples, group_regex=None):
    """Return a list of patient/group ids aligned with `samples` (list of (path, label)), or None."""
    stems = [os.path.splitext(os.path.basename(p))[0] for p, _ in samples]

    if group_regex:
        rx = re.compile(group_regex)
        groups = []
        for s in stems:
            m = rx.search(s)
            if not m:
                warnings.warn(f"--group-regex did not match '{s}'; treating it as its own group")
                groups.append(s)
            else:
                groups.append(m.group(1) if m.groups() else m.group(0))
        return groups

    manifest = _read_manifest(data_dir)
    if manifest:
        with open(manifest, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            cols = {c.lower(): c for c in rows[0].keys()}
            fcol = next((cols[c] for c in _FILE_COLS if c in cols), None)
            gcol = next((cols[c] for c in _GROUP_COLS if c in cols), None)
            if fcol and gcol:
                lookup = {os.path.splitext(os.path.basename(r[fcol]))[0]: r[gcol] for r in rows if r.get(fcol)}
                groups = [lookup.get(s) for s in stems]
                matched = sum(g is not None for g in groups)
                if matched >= 0.9 * len(stems):
                    return [g if g is not None else s for g, s in zip(groups, stems)]
                warnings.warn(f"manifest.csv matched only {matched}/{len(stems)} files; ignoring it")
    return None


def split_train_val(samples, labels, groups, val_fraction=0.15, seed=42):
    """Stratified (and group-aware, if groups given) split. Returns (train_idx, val_idx)."""
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split
    idx = np.arange(len(samples))
    if groups is not None:
        n_splits = max(2, int(round(1.0 / val_fraction)))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        tr, va = next(sgkf.split(idx, labels, groups))
        return tr, va
    tr, va = train_test_split(idx, test_size=val_fraction, stratify=labels, random_state=seed)
    return np.sort(tr), np.sort(va)


# ----------------------------------------------------------------------------- loaders
def get_dataloaders(data_dir="dataset/classification", batch_size=32, num_workers=0,
                    subset_fraction=1.0, val_fraction=0.15, seed=42, group_regex=None,
                    image_size=IMAGE_SIZE, pin_memory=False):
    """
    Returns a dict:
      train_loader, val_loader, test_loader, class_names, class_counts (train),
      split_info (dict describing how the split was made - saved into model_metadata.json)
    """
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    train_tf, eval_tf = get_transforms(image_size)

    # Two views of the same folder: augmented for fitting, clean for validation
    full_train_aug = datasets.ImageFolder(train_dir, transform=train_tf)
    full_train_eval = datasets.ImageFolder(train_dir, transform=eval_tf)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_tf)

    if full_train_aug.classes != list(CLASS_NAMES):
        raise ValueError(f"Class folders {full_train_aug.classes} do not match expected {CLASS_NAMES}")

    samples = full_train_aug.samples
    labels = np.array([y for _, y in samples])
    groups = infer_groups(data_dir, samples, group_regex)
    tr_idx, va_idx = split_train_val(samples, labels, groups, val_fraction, seed)

    split_info = {"val_fraction": val_fraction, "seed": seed,
                  "group_aware": groups is not None,
                  "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)), "n_test": int(len(test_dataset))}
    if groups is not None:
        g = np.array(groups)
        overlap = set(g[tr_idx]) & set(g[va_idx])
        split_info["train_val_group_overlap"] = len(overlap)
        assert not overlap, "BUG: patient groups leaked across train/val"
        # also report train/test patient overlap if the manifest covers the test folder
        test_groups = infer_groups(data_dir, test_dataset.samples, group_regex)
        if test_groups is not None:
            split_info["train_test_group_overlap"] = len(set(groups) & set(test_groups))
            if split_info["train_test_group_overlap"]:
                warnings.warn(f"{split_info['train_test_group_overlap']} patient ids appear in BOTH train and test "
                              f"folders - test accuracy will be optimistic.")
    else:
        warnings.warn(
            "No patient ids found (no usable manifest.csv and no --group-regex): validation split is "
            "SLICE-level. Slices of the same patient may land in both train and val, so val accuracy can be "
            "optimistic. Provide --group-regex (one capture group = patient id) to fix this.")

    if subset_fraction < 1.0:
        rng = np.random.default_rng(seed)
        def sub(a):
            return np.sort(rng.choice(a, max(1, int(len(a) * subset_fraction)), replace=False))
        tr_idx, va_idx = sub(tr_idx), sub(va_idx)
        test_dataset = Subset(test_dataset, sub(np.arange(len(test_dataset))))
        split_info.update(n_train=int(len(tr_idx)), n_val=int(len(va_idx)), n_test=int(len(test_dataset)))

    train_ds = Subset(full_train_aug, tr_idx)
    val_ds = Subset(full_train_eval, va_idx)

    counts = Counter(int(labels[i]) for i in tr_idx)
    class_counts = [counts.get(i, 0) for i in range(len(CLASS_NAMES))]

    kw = dict(num_workers=num_workers, pin_memory=pin_memory)
    return {
        "train_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_ds) > batch_size, **kw),
        "val_loader": DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw),
        "test_loader": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kw),
        "class_names": list(CLASS_NAMES),
        "class_counts": class_counts,
        "split_info": split_info,
    }
