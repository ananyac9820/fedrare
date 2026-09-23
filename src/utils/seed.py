"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed every RNG this project touches.

    Matters more than usual here: with rare classes at under 1% of the data, run-to-run
    variance on rare-class F1 is large. Without seeding you cannot tell a real improvement
    from noise.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "cuda") -> torch.device:
    """CUDA if asked for and present, else Apple-silicon MPS if present, else CPU."""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred in ("cuda", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
