"""Loading the DenseNet-121 features saved by scripts/05_extract_features.py (Tier A)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "data" / "features"
PREFIX = "fed_isic2019_densenet121"


@dataclass(frozen=True)
class FeatureSplit:
    features: np.ndarray   # (N, 1024) float32
    labels: np.ndarray     # (N,) int64
    centers: np.ndarray    # (N,) int64, the real Fed-ISIC2019 centre of each image

    def counts(self, n_centers: int, n_classes: int, centers: np.ndarray | None = None) -> np.ndarray:
        """(n_centers, n_classes) image counts, optionally under a different centre assignment."""
        ctr = self.centers if centers is None else centers
        return np.stack([np.bincount(self.labels[ctr == k], minlength=n_classes)
                         for k in range(n_centers)])


def load_features(split: str, features_dir: Path | str | None = None) -> FeatureSplit:
    path = Path(features_dir or FEATURES_DIR) / f"{PREFIX}_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/05_extract_features.py first.")
    with np.load(path) as z:
        return FeatureSplit(z["features"], z["labels"].astype(np.int64),
                            z["centers"].astype(np.int64))
