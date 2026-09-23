"""Tests for the S1 / S2 data splits, on the real Fed-ISIC2019 train-split counts."""

import numpy as np
import pytest

from src.data.splits import (HOLDER_MIN_IMAGES, S2_SEED, class_counts, coverage, make_split,
                             split_s2)

# Real train-split counts (rows = centre 0..5, cols = class 0..7), from load_split_columns.
TRAIN_COUNTS = np.array([
    [2279, 3363, 2245, 606, 923, 97, 87, 330],
    [20, 2977, 2, 0, 96, 23, 45, 0],
    [533, 1470, 174, 20, 379, 40, 64, 11],
    [276, 649, 223, 86, 406, 20, 3, 144],
    [166, 349, 0, 0, 140, 0, 0, 0],
    [58, 276, 5, 0, 6, 4, 2, 0],
])
RARE = [5, 6]
N_CENTERS, N_CLASSES = TRAIN_COUNTS.shape


def arrays_from_counts(counts, seed=0):
    centers = np.repeat(np.arange(N_CENTERS), counts.sum(1))
    labels = np.concatenate([np.repeat(np.arange(N_CLASSES), row) for row in counts])
    order = np.random.default_rng(seed).permutation(len(labels))   # realistic interleaving
    return centers[order], labels[order]


CENTERS, LABELS = arrays_from_counts(TRAIN_COUNTS)


def test_s1_is_the_natural_split():
    s1 = make_split("s1", CENTERS, LABELS, RARE)
    assert np.array_equal(s1, CENTERS) and s1 is not CENTERS
    assert np.array_equal(class_counts(s1, LABELS, N_CENTERS, N_CLASSES), TRAIN_COUNTS)
    # coverage on S1 matches the design doc: DF held by 4 centres, VASC by 3
    assert coverage(s1, LABELS, N_CENTERS, N_CLASSES).tolist()[5:7] == [4, 3]


def test_s2_moves_only_rare_classes_into_centre_2():
    s2 = make_split("s2", CENTERS, LABELS, RARE)
    before = class_counts(CENTERS, LABELS, N_CENTERS, N_CLASSES)
    after = class_counts(s2, LABELS, N_CENTERS, N_CLASSES)
    common = [c for c in range(N_CLASSES) if c not in RARE]
    assert np.array_equal(after[:, common], before[:, common])       # common classes untouched
    assert np.array_equal(after.sum(0), before.sum(0))                # no image lost or duplicated
    changed = s2 != CENTERS
    assert np.all(s2[changed] == 2) and np.all(np.isin(LABELS[changed], RARE))
    for c in RARE:
        for k in range(N_CENTERS):
            if k != 2:
                expected_kept = before[k, c] - int(np.floor(0.9 * before[k, c] + 0.5))
                assert after[k, c] == expected_kept


def test_s2_coverage_drops_to_one_or_two():
    s2 = make_split("s2", CENTERS, LABELS, RARE)
    cov = coverage(s2, LABELS, N_CENTERS, N_CLASSES)
    assert all(1 <= cov[c] <= 2 for c in RARE)
    counts = class_counts(s2, LABELS, N_CENTERS, N_CLASSES)
    for c in RARE:
        assert counts[2, c] >= HOLDER_MIN_IMAGES and counts[2, c] == counts[:, c].max()
    common = [c for c in range(N_CLASSES) if c not in RARE]
    assert np.array_equal(cov[common], coverage(CENTERS, LABELS, N_CENTERS, N_CLASSES)[common])


def test_s2_is_deterministic_and_independent_of_input_order_seed():
    a = split_s2(CENTERS, LABELS, RARE)
    b = split_s2(CENTERS, LABELS, RARE, seed=S2_SEED)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, split_s2(CENTERS, LABELS, RARE, seed=S2_SEED + 1))


def test_s2_edge_fractions_and_input_safety():
    orig = CENTERS.copy()
    assert np.array_equal(split_s2(CENTERS, LABELS, RARE, move_fraction=0.0), CENTERS)
    everything = split_s2(CENTERS, LABELS, RARE, move_fraction=1.0)
    assert np.all(everything[np.isin(LABELS, RARE)] == 2)
    assert np.array_equal(CENTERS, orig)                                 # input not modified
    with pytest.raises(ValueError):
        split_s2(CENTERS, LABELS, RARE, move_fraction=1.5)
    with pytest.raises(ValueError):
        make_split("s3", CENTERS, LABELS, RARE)
