#!/usr/bin/env python3
"""
Tier A, step 1 - run a frozen, ImageNet-pretrained DenseNet-121 over every Fed-ISIC2019 image
once and save the result.

Every Tier A experiment then trains only the classifier head on these saved features, which
is what makes seeds, attacks and ablations affordable on a laptop CPU (design doc, Section 7).

The feature for an image is exactly what DenseNet-121's own classifier layer receives:
features -> ReLU -> global average pool -> 1024 numbers. The script checks this against
torchvision's forward pass on the first batch, so a Tier A head is a drop-in replacement for
DenseNet's classifier.

Preprocessing matches the evaluation transform in src/data/loader.py (images are already
224px in the mirror; ImageNet normalisation; no augmentation).

Outputs to --out-dir (default data/features/, gitignored - never commit features):
    fed_isic2019_densenet121_train.npz   features (N, 1024) float32, labels (N,), centers (N,)
    fed_isic2019_densenet121_test.npz    same for the test split
    fed_isic2019_densenet121.json        provenance: weights, versions, counts, timings

Row order in each file is the Hugging Face split order, the same order load_split_columns()
returns, so row i here is row i everywhere else in the repo.

    python scripts/05_extract_features.py                 # CPU ~35 min, GPU a few minutes
    python scripts/05_extract_features.py --limit 64      # quick smoke test, separate files
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.data.loader import build_transforms
from src.models.densenet import build_model, densenet_features
from src.utils.config import load_config
from src.utils.seed import get_device

FEATURE_DIM = 1024
PREFIX = "fed_isic2019_densenet121"
# Published per-centre training counts (same check as scripts/01_verify_setup.py).
EXPECTED_TRAIN = {0: 9930, 1: 3163, 2: 2691, 3: 1807, 4: 655, 5: 351}


def check_against_torchvision(model: nn.Module, x: torch.Tensor) -> float:
    """Max abs difference between our features and torchvision's own pre-classifier output."""
    head = model.classifier
    model.classifier = nn.Identity()
    try:
        reference = model(x)
    finally:
        model.classifier = head
    return float((densenet_features(model, x) - reference).abs().max())


def extract_split(split: str, cfg: dict, model: nn.Module, device: torch.device,
                  batch_size: int, limit: int | None) -> dict:
    from datasets import load_dataset

    ds = load_dataset(cfg["dataset"]["hf_repo"], split=split)
    n = len(ds) if limit is None else min(limit, len(ds))
    transform = build_transforms(cfg["dataset"]["image_size"], train=False)
    features = np.empty((n, FEATURE_DIM), dtype=np.float32)
    labels = np.asarray(ds["label"][:n], dtype=np.int64)
    centers = np.asarray(ds["center"][:n], dtype=np.int64)

    t0 = time.time()
    max_diff = None
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        images = ds[start:stop]["image"]
        x = torch.stack([transform(img.convert("RGB")) for img in images]).to(device)
        with torch.inference_mode():
            if max_diff is None:
                max_diff = check_against_torchvision(model, x)
                if max_diff > 1e-4:
                    raise RuntimeError(f"feature path disagrees with torchvision by {max_diff}")
            features[start:stop] = densenet_features(model, x).float().cpu().numpy()

        done = stop
        if done == n or done // batch_size % 20 == 0:
            rate = done / (time.time() - t0)
            eta = (n - done) / rate / 60
            print(f"  {split}: {done:>6,}/{n:,} images | {rate:5.1f} img/s | ETA {eta:5.1f} min",
                  flush=True)

    if not np.isfinite(features).all():
        raise RuntimeError(f"non-finite values in {split} features")
    return {"features": features, "labels": labels, "centers": centers,
            "seconds": time.time() - t0, "torchvision_max_abs_diff": max_diff}


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract frozen DenseNet-121 features.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "features")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N images per split (smoke test; writes *_limitN files)")
    args = parser.parse_args()

    cfg = load_config(ROOT / "configs" / "default.yaml")
    device = get_device("cuda")
    torch.set_grad_enabled(False)
    model = build_model(cfg["dataset"]["n_classes"], pretrained=True).to(device).eval()
    print(f"Device: {device} | torch {torch.__version__} | threads {torch.get_num_threads()}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{args.limit}" if args.limit else ""
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "torchvision densenet121, weights IMAGENET1K_V1, frozen, eval mode",
        "feature": "ReLU(features(x)) -> adaptive_avg_pool2d(1) -> flatten; input to classifier",
        "feature_dim": FEATURE_DIM,
        "preprocessing": "src/data/loader.py build_transforms(train=False): resize 224, "
                         "ToTensor, ImageNet mean/std; no augmentation",
        "source": cfg["dataset"]["hf_repo"],
        "row_order": "Hugging Face split order (same as load_split_columns)",
        "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                     "numpy": np.__version__, "python": platform.python_version()},
        "device": str(device),
        "limit": args.limit,
        "splits": {},
    }

    for split in ("train", "test"):
        print(f"\nExtracting {split} split...")
        out = extract_split(split, cfg, model, device, args.batch_size, args.limit)
        path = args.out_dir / f"{PREFIX}_{split}{suffix}.npz"
        np.savez(path, features=out["features"], labels=out["labels"], centers=out["centers"])

        per_centre = {int(k): int((out["centers"] == k).sum()) for k in np.unique(out["centers"])}
        meta["splits"][split] = {
            "file": path.name, "n_images": int(len(out["labels"])),
            "per_centre": per_centre,
            "per_centre_per_class": np.stack([
                np.bincount(out["labels"][out["centers"] == k],
                            minlength=cfg["dataset"]["n_classes"])
                for k in range(cfg["dataset"]["n_centers"])]).tolist(),
            "seconds": round(out["seconds"], 1),
            "torchvision_max_abs_diff": out["torchvision_max_abs_diff"],
            "feature_mean": float(out["features"].mean()),
            "feature_std": float(out["features"].std()),
        }
        print(f"  wrote {path} ({path.stat().st_size / 1e6:.1f} MB) in {out['seconds'] / 60:.1f} min")

        if split == "train" and args.limit is None and per_centre != EXPECTED_TRAIN:
            print(f"  WARNING: train per-centre counts {per_centre} differ from the published "
                  f"{EXPECTED_TRAIN}. The mirror may have changed.")

    meta_path = args.out_dir / f"{PREFIX}{suffix}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n  wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
