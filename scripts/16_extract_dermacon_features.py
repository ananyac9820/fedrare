#!/usr/bin/env python3
"""
DermaCon-IN through the frozen DenseNet-121, once - the Indian counterpart of scripts/05.

Same backbone, same evaluation transform, same feature definition as Fed-ISIC2019, so the
two datasets differ in their data and nothing else. The 1,024-d feature is what DenseNet's
own classifier layer receives, and the extraction is checked against torchvision's forward
pass on the first batch exactly as script 05 does.

Also writes the two centre splits fixed in docs/DERMACON_PLAN.md before any training:
    D0  patient id -> 6 groups by CRC32 (near-IID control)
    D1  six age bands (naturally uneven; the 60-80 band holds 1.95x its share of rare cases)
and the dataset's own subject-wise train/test split, after checking no patient is on both
sides of it.

Labels are Main_class with "No Definite Diagnosis" dropped (it is not a disease), leaving
7 classes. Rows for dropped images are kept in the file with label -1 so that row i here is
row i of Skin_Metadata.tab; everything downstream filters on label >= 0.

    python scripts/16_extract_dermacon_features.py          # CPU, ~15-25 min
    python scripts/16_extract_dermacon_features.py --limit 64

Output: data/features/dermacon_densenet121.npz + .json provenance.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zlib import crc32

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torchvision
from PIL import Image

from src.data.loader import build_transforms
from src.models.densenet import build_model, densenet_features
from src.utils.seed import get_device

DATA = ROOT / "data" / "dermacon_in"
OUT_DIR = ROOT / "data" / "features"
PREFIX = "dermacon_densenet121"
FEATURE_DIM = 1024
IMAGE_SIZE = 224
N_CENTRES = 6
DROP_CLASS = "No Definite Diagnosis"
AGE_BANDS = ["0 - 10", "10 - 20", "20 - 40", "40 - 60", "60 - 80", "80 - 100"]


class _Images(torch.utils.data.Dataset):
    """Just enough Dataset to let worker processes do the decoding."""

    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB"))


def read_tab(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def index_images() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for archive in ("DATASET_0", "DATASET_1"):
        for path in (DATA / archive).rglob("*"):
            if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                found.setdefault(path.name, path)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract DermaCon-IN features.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    rows = read_tab(DATA / "Skin_Metadata.tab")
    if args.limit:
        rows = rows[:args.limit]
    files = index_images()
    missing = [r["Image_name"] for r in rows if r["Image_name"] not in files]
    if missing:
        raise SystemExit(f"{len(missing)} images in the metadata are not on disk, e.g. "
                         f"{missing[:3]}\nRun: python scripts/14_fetch_dermacon.py")

    # Labels: Main_class minus the non-disease bucket, in a fixed alphabetical order so the
    # mapping does not depend on the order rows happen to appear in.
    classes = sorted({r["Main_class"] for r in rows} - {DROP_CLASS})
    class_index = {name: i for i, name in enumerate(classes)}
    labels = np.array([class_index.get(r["Main_class"], -1) for r in rows], dtype=np.int64)

    # D0: stable hash of the patient id. crc32, not hash(), which Python salts per process.
    d0 = np.array([crc32(r["Subject_ID"].encode()) % N_CENTRES for r in rows], dtype=np.int64)
    # D1: age bands, in the order given above.
    band_index = {b: i for i, b in enumerate(AGE_BANDS)}
    d1 = np.array([band_index.get(r["Age"].strip(), -1) for r in rows], dtype=np.int64)
    if (d1 < 0).any():
        unknown = sorted({r["Age"] for r, c in zip(rows, d1) if c < 0})
        raise SystemExit(f"age values outside the fixed bands: {unknown}")

    # The dataset's own split, but verified rather than trusted.
    train_names = {r["Image_name"] for r in read_tab(DATA / "train_split.tab")}
    test_names = {r["Image_name"] for r in read_tab(DATA / "test_split.tab")}
    train_subjects = {r["Subject_ID"] for r in rows if r["Image_name"] in train_names}
    test_subjects = {r["Subject_ID"] for r in rows if r["Image_name"] in test_names}
    leaked = train_subjects & test_subjects
    if leaked:
        raise SystemExit(f"{len(leaked)} patients appear in both shipped splits, e.g. "
                         f"{sorted(leaked)[:3]} - do not use this split")
    is_train = np.array([r["Image_name"] in train_names for r in rows], dtype=bool)
    unassigned = sum(1 for r in rows if r["Image_name"] not in train_names | test_names)

    device = get_device("cuda")
    torch.set_grad_enabled(False)
    model = build_model(len(classes), pretrained=True).to(device).eval()
    transform = build_transforms(IMAGE_SIZE, train=False)
    print(f"Device: {device} | torch {torch.__version__}")
    print(f"{len(rows):,} images | {len(classes)} classes: {classes}")
    print(f"train {int(is_train.sum()):,} / test {int((~is_train).sum()):,} "
          f"(no patient on both sides; {unassigned} rows in neither shipped file)")

    # These are full-resolution phone photographs (up to 4,608 px), so JPEG decode, not the
    # network, is the bottleneck. Decoding in worker processes takes this from ~1 image a
    # second to something that finishes over lunch. Order is preserved: shuffle stays off.
    loader = torch.utils.data.DataLoader(
        _Images([files[r["Image_name"]] for r in rows], transform),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=False)

    features = np.empty((len(rows), FEATURE_DIM), dtype=np.float32)
    t0, max_diff, start = time.time(), None, 0
    for batch in loader:
        batch = batch.to(device)
        stop = start + len(batch)

        if max_diff is None:   # same check as script 05: our path must equal torchvision's
            head = model.classifier
            model.classifier = torch.nn.Identity()
            reference = model(batch)
            model.classifier = head
            max_diff = float((densenet_features(model, batch) - reference).abs().max())
            if max_diff > 1e-4:
                raise RuntimeError(f"feature path disagrees with torchvision by {max_diff}")

        features[start:stop] = densenet_features(model, batch).float().cpu().numpy()
        if stop % (args.batch_size * 20) == 0 or stop == len(rows):
            rate = stop / (time.time() - t0)
            print(f"  {stop:>6,}/{len(rows):,} | {rate:5.1f} img/s | "
                  f"ETA {(len(rows) - stop) / rate / 60:4.1f} min", flush=True)
        start = stop

    if not np.isfinite(features).all():
        raise RuntimeError("non-finite feature values")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{args.limit}" if args.limit else ""
    path = OUT_DIR / f"{PREFIX}{suffix}.npz"
    np.savez(path, features=features, labels=labels, d0=d0, d1=d1, is_train=is_train,
             image_names=np.array([r["Image_name"] for r in rows]))

    kept = labels >= 0
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "DermaCon-IN (doi:10.7910/DVN/W7OUZM, CC BY-NC-SA 4.0)",
        "citation": "Madarkar et al., NeurIPS 2025 Datasets & Benchmarks",
        "model": "torchvision densenet121, weights IMAGENET1K_V1, frozen, eval mode",
        "preprocessing": "src/data/loader.build_transforms(224, train=False)",
        "torchvision_max_abs_diff": max_diff,
        "n_images": len(rows),
        "n_used": int(kept.sum()),
        "dropped_class": DROP_CLASS,
        "classes": classes,
        "per_class": {c: int((labels == i).sum()) for c, i in class_index.items()},
        "splits": {
            "note": "constructed - DermaCon-IN does not record which hospital took an image",
            "D0": "patient id -> 6 groups by crc32 (near-IID control)",
            "D1": "six age bands " + str(AGE_BANDS),
            "D0_sizes": [int((d0[kept] == k).sum()) for k in range(N_CENTRES)],
            "D1_sizes": [int((d1[kept] == k).sum()) for k in range(N_CENTRES)],
        },
        "train_test": {"source": "the dataset's own subject-wise split",
                       "train": int(is_train[kept].sum()),
                       "test": int((~is_train[kept]).sum()),
                       "patients_in_both": 0},
        "seconds": round(time.time() - t0, 1),
        "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                     "numpy": np.__version__, "python": platform.python_version()},
    }
    (OUT_DIR / f"{PREFIX}{suffix}.json").write_text(json.dumps(meta, indent=2),
                                                    encoding="utf-8")
    print(f"\nwrote {path} ({path.stat().st_size / 1e6:.1f} MB) in "
          f"{meta['seconds'] / 60:.1f} min")
    print(f"  usable images {meta['n_used']:,} (dropped {len(rows) - meta['n_used']} "
          f"'{DROP_CLASS}')")
    print(f"  D0 sizes {meta['splits']['D0_sizes']}")
    print(f"  D1 sizes {meta['splits']['D1_sizes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
