"""
Building blocks shared by Camp A, EARN and the G0a gate, so all three measure the same thing.

Each follows the design doc's pseudocode (EARN Project Design v2, Section 4.4):

    U[k] = clip(local_train(global, data_k), median_norm)       -> clip_to_median_norm
    e[k][c] = norm(U[k].head_row[c])                            -> head_evidence
    e[k][c] / sum_j e[j][c]                                     -> evidence_share
    cap(norm(w), 0.5)                                           -> cap_weights

Everything works on flattened updates: a (K, D) float64 tensor, one row per client, laid out
by a FlatLayout built from the model's state keys.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .interface import DEFAULT_HEAD_PREFIX, body_keys, float_keys, head_keys


@dataclass(frozen=True)
class FlatLayout:
    """Where each parameter lives in a flattened update vector."""
    keys: tuple[str, ...]            # all floating-point keys, in flattening order
    head_weight_key: str             # e.g. "classifier.weight", shape (C, D_head)
    n_classes: int
    body_cols: np.ndarray            # columns of body parameters
    head_cols: tuple[np.ndarray, ...]    # head_cols[c] = columns of head row c (weight + bias)
    head_weight_cols: tuple[np.ndarray, ...]  # head_weight_cols[c] = columns of weight row c only

    @classmethod
    def from_state(cls, state: dict, prefix: str = DEFAULT_HEAD_PREFIX) -> "FlatLayout":
        keys = tuple(float_keys(state))
        heads = set(head_keys(state, prefix))
        weight_key = f"{prefix}.weight"
        if weight_key not in state:
            raise KeyError(f"no head weight '{weight_key}' in state")
        n_classes = state[weight_key].shape[0]

        body, rows, weight_rows = [], [[] for _ in range(n_classes)], [None] * n_classes
        pos = 0
        for k in keys:
            n = state[k].numel()
            if k in heads:
                if state[k].shape[0] != n_classes:
                    raise ValueError(f"head tensor {k} has leading dim {state[k].shape[0]}, "
                                     f"expected {n_classes} (one row per class)")
                per_row = n // n_classes
                for c in range(n_classes):
                    cols = np.arange(pos + c * per_row, pos + (c + 1) * per_row)
                    rows[c].append(cols)
                    if k == weight_key:
                        weight_rows[c] = cols
            else:
                body.append(np.arange(pos, pos + n))
            pos += n
        assert set(body_keys(state, prefix)) | heads == set(keys)
        return cls(keys=keys, head_weight_key=weight_key, n_classes=n_classes,
                   body_cols=np.concatenate(body) if body else np.zeros(0, dtype=np.int64),
                   head_cols=tuple(np.concatenate(r) for r in rows),
                   head_weight_cols=tuple(weight_rows))


def clip_to_median_norm(updates: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
    """Scale each client's whole update down to the median update norm of this round.

    Updates at or below the median are unchanged. With an even number of clients the median
    is the mean of the middle two norms. Returns (clipped updates, per-client scale).
    """
    norms = updates.norm(dim=1).double().cpu().numpy()
    median = float(np.median(norms))
    scale = np.where(norms > median, median / np.maximum(norms, 1e-300), 1.0)
    return updates * torch.as_tensor(scale, dtype=updates.dtype)[:, None], scale


def head_evidence(updates: torch.Tensor, layout: FlatLayout) -> np.ndarray:
    """e[k][c] = L2 norm of client k's update to head weight row c. Shape (K, C).

    Weight row only; the bias entry is excluded (the design doc's G0a retry option is to use
    the bias instead).
    """
    return np.stack([updates[:, cols].norm(dim=1).double().cpu().numpy()
                     for cols in layout.head_weight_cols], axis=1)


def evidence_share(evidence: np.ndarray) -> np.ndarray:
    """e[k][c] / sum_j e[j][c], per class. A class with no evidence at all gets zeros."""
    totals = evidence.sum(axis=0, keepdims=True)
    return np.divide(evidence, totals, out=np.zeros_like(evidence, dtype=np.float64),
                     where=totals > 0)


def cap_weights(weights: np.ndarray, cap: float = 0.5) -> np.ndarray:
    """Normalise to sum 1, then cap every weight at `cap`, redistributing the excess.

    Excess goes to the uncapped clients in proportion to their weights (equally if they all
    have zero weight). Needs cap * K >= 1 to be feasible.
    """
    w = np.asarray(weights, dtype=np.float64).copy()
    if (w < 0).any():
        raise ValueError("weights must be non-negative")
    if w.sum() <= 0:
        raise ValueError("weights must not all be zero")
    if cap * len(w) < 1 - 1e-12:
        raise ValueError(f"cap {cap} infeasible for {len(w)} clients")
    w /= w.sum()
    capped = np.zeros(len(w), dtype=bool)
    for _ in range(len(w)):
        over = (w > cap + 1e-12) & ~capped
        if not over.any():
            break
        capped |= over
        excess = (w[capped] - cap).sum()
        w[capped] = cap
        free = ~capped
        base = w[free].sum()
        w[free] += excess * (w[free] / base if base > 0 else np.full(free.sum(), 1 / free.sum()))
    return w
