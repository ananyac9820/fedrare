"""Loading the DenseNet-121 features saved by scripts/05_extract_features.py (Tier A)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "data" / "features"
PREFIX = "fed_isic2019_densenet121"
# Feature sets: "frozen" = ImageNet DenseNet-121 as is (scripts/05_extract_features.py);
# "ft4" = last dense block fine-tuned federated, the G0b retry (scripts/08_g0b_retry.py,
# docs/DEVIATIONS.md D3). Experiments from D4 on use "ft4".
VARIANTS = {"frozen": PREFIX, "ft4": f"{PREFIX}_ft4"}


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


def load_features(split: str, features_dir: Path | str | None = None,
                  variant: str = "frozen") -> FeatureSplit:
    if variant not in VARIANTS:
        raise ValueError(f"unknown feature variant {variant!r}; expected one of {list(VARIANTS)}")
    path = Path(features_dir or FEATURES_DIR) / f"{VARIANTS[variant]}_{split}.npz"
    if not path.exists():
        script = "05_extract_features.py" if variant == "frozen" else "08_g0b_retry.py"
        raise FileNotFoundError(f"{path} not found. Run scripts/{script} first.")
    with np.load(path) as z:
        return FeatureSplit(z["features"], z["labels"].astype(np.int64),
                            z["centers"].astype(np.int64))
