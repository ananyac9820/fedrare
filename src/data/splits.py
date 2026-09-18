"""
Data splits for the EARN experiments (design doc Section 6.1).

A split is a centre assignment for the TRAINING images: an int array aligned with the train
split's row order (the order load_split_columns() and the saved features use). The test
split is never re-assigned - evaluation uses the pooled test set either way.

    S1  natural   Fed-ISIC2019's real 6-centre split, unchanged.
    S2  specialist (constructed, disclosed in the paper). Most rare-class training images are
                  moved into centre 2, so coverage - the number of centres holding at least
                  HOLDER_MIN_IMAGES images of a class - falls to 1-2 for the rare classes.
                  Models a single specialist clinic holding nearly all cases of an
                  ultra-rare condition.

S2 definition: for every rare class and every centre other than the specialist, a fixed
fraction (MOVE_FRACTION) of that centre's training images of that class, chosen at random
with a fixed seed (S2_SEED), is re-assigned to the specialist. Nothing else changes: common
classes stay where they are, labels are untouched, and donor centres simply shrink by the
images they gave up. The seed is fixed in this module - not the experiment seed - so every
run and every team member gets the identical S2.

    from src.data.splits import make_split
    centers = make_split("s2", train.centers, train.labels, rare_ids=[5, 6])
"""

from __future__ import annotations

import numpy as np

HOLDER_MIN_IMAGES = 20       # a centre "holds" a class with at least this many images (doc S1)
SPECIALIST_CENTRE = 2        # the measured rare-class specialist (scripts/02_explore_data.py)
MOVE_FRACTION = 0.9          # share of each donor's rare-class images moved to the specialist
S2_SEED = 0                  # fixed: S2 is one specific split, the same for everyone


def class_counts(centers: np.ndarray, labels: np.ndarray, n_centers: int,
                 n_classes: int) -> np.ndarray:
    """(n_centers, n_classes) image counts under a centre assignment."""
    return np.stack([np.bincount(labels[centers == k], minlength=n_classes)
                     for k in range(n_centers)])


def coverage(centers: np.ndarray, labels: np.ndarray, n_centers: int, n_classes: int,
             min_images: int = HOLDER_MIN_IMAGES) -> np.ndarray:
    """Number of centres holding at least min_images of each class. Shape (n_classes,)."""
    return (class_counts(centers, labels, n_centers, n_classes) >= min_images).sum(axis=0)


def split_s1(centers: np.ndarray) -> np.ndarray:
    """The natural split: a copy of the real centre assignment."""
    return np.asarray(centers, dtype=np.int64).copy()


def split_s2(centers: np.ndarray, labels: np.ndarray, rare_ids: list[int],
             specialist: int = SPECIALIST_CENTRE, move_fraction: float = MOVE_FRACTION,
             seed: int = S2_SEED) -> np.ndarray:
    """Move move_fraction of every other centre's rare-class training images to specialist.

    For each (donor centre, rare class) pair, round(move_fraction * n) images move (half
    rounds up), chosen uniformly at random without replacement. Returns a new centre array;
    the inputs are not modified.
    """
    if not 0.0 <= move_fraction <= 1.0:
        raise ValueError(f"move_fraction must be in [0, 1], got {move_fraction}")
    centers = np.asarray(centers, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if centers.shape != labels.shape:
        raise ValueError("centers and labels must be aligned")

    rng = np.random.default_rng(seed)
    out = centers.copy()
    for c in sorted(rare_ids):
        for k in sorted(set(centers.tolist()) - {specialist}):
            idx = np.flatnonzero((centers == k) & (labels == c))
            n_move = int(np.floor(move_fraction * len(idx) + 0.5))
            if n_move:
                out[rng.choice(idx, size=n_move, replace=False)] = specialist
    return out


def make_split(name: str, centers: np.ndarray, labels: np.ndarray, rare_ids: list[int],
               **kwargs) -> np.ndarray:
    """Centre assignment for split "s1" or "s2" (kwargs go to split_s2)."""
    name = name.lower()
    if name == "s1":
        return split_s1(centers)
    if name == "s2":
        return split_s2(centers, labels, rare_ids, **kwargs)
    raise ValueError(f"unknown split {name!r}; expected 's1' or 's2'")
