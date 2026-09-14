"""
Byzantine-robust aggregation rules.

Every rule takes one round's client updates stacked as a (n_clients, n_params) tensor and
returns an AggregationResult: the aggregated update, plus for each client the effective
weight it carried into that aggregate and whether it counts as selected.

Effective weights sum to 1 across clients, so they are comparable between rules:
  - fedavg                  sample-count share
  - krum, multi_krum        1/m for each of the m selected clients, 0 for the rest
  - trimmed_mean,
    coordinate_wise_median  the fraction of all coordinate values in the aggregate that came
                            from that client

The coordinate-wise rules never drop a client outright - they drop individual coordinates -
so for them "selected" is derived from the weight: a client counts as rejected when its
effective weight falls below NEAR_ZERO_FRACTION of the uniform share 1/n.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# A coordinate-wise rule counts a client as rejected when its effective weight is under
# this fraction of the uniform share 1/n.
NEAR_ZERO_FRACTION = 0.5


@dataclass
class AggregationResult:
    aggregate: torch.Tensor  # (n_params,)
    weights: np.ndarray      # (n_clients,) effective weight, sums to 1
    selected: np.ndarray     # (n_clients,) bool


def _flags_from_weights(weights: np.ndarray,
                        near_zero_fraction: float = NEAR_ZERO_FRACTION) -> np.ndarray:
    return weights >= near_zero_fraction / len(weights)


def _select_and_average(updates: torch.Tensor, chosen: np.ndarray) -> AggregationResult:
    n = updates.shape[0]
    selected = np.zeros(n, dtype=bool)
    selected[chosen] = True
    weights = selected / selected.sum()
    aggregate = updates[torch.as_tensor(chosen)].mean(dim=0)
    return AggregationResult(aggregate, weights.astype(np.float64), selected)


def fedavg(updates: torch.Tensor, n_samples) -> AggregationResult:
    """Sample-count weighted mean. The non-robust baseline: every client is selected."""
    counts = torch.as_tensor(np.asarray(n_samples), dtype=updates.dtype)
    w = counts / counts.sum()
    aggregate = (w[:, None] * updates).sum(dim=0)
    return AggregationResult(aggregate, w.double().numpy(), np.ones(len(w), dtype=bool))


def krum_scores(updates: torch.Tensor, f: int) -> np.ndarray:
    """Krum score per client: summed squared distance to its n - f - 2 nearest neighbours.

    Krum's guarantee needs n >= 2f + 3 (Blanchard et al., 2017).
    """
    n = updates.shape[0]
    if n < 2 * f + 3:
        raise ValueError(f"Krum needs n >= 2f + 3; got n={n}, f={f}")
    k = n - f - 2
    x = updates.double()
    dists = torch.cdist(x, x).pow(2)
    dists.fill_diagonal_(float("inf"))
    nearest = torch.sort(dists, dim=1).values[:, :k]
    return nearest.sum(dim=1).numpy()


def krum(updates: torch.Tensor, f: int) -> AggregationResult:
    """Select the single client with the lowest Krum score and use its update alone."""
    scores = krum_scores(updates, f)
    return _select_and_average(updates, np.argsort(scores, kind="stable")[:1])


def multi_krum(updates: torch.Tensor, f: int, m: int | None = None) -> AggregationResult:
    """Average the m clients with the lowest Krum scores (default m = n - f)."""
    n = updates.shape[0]
    m = n - f if m is None else m
    scores = krum_scores(updates, f)
    return _select_and_average(updates, np.argsort(scores, kind="stable")[:m])


def trimmed_mean(updates: torch.Tensor, trim: int) -> AggregationResult:
    """Per coordinate, drop the `trim` largest and `trim` smallest values, then average."""
    n = updates.shape[0]
    if n - 2 * trim < 1:
        raise ValueError(f"trim={trim} leaves no values with n={n} clients")
    order = torch.argsort(updates, dim=0)  # order[i, j] = client at rank i for coordinate j
    kept = order[trim:n - trim]
    aggregate = torch.gather(updates, 0, kept).mean(dim=0)
    counts = torch.bincount(kept.reshape(-1), minlength=n).double()
    weights = (counts / counts.sum()).numpy()
    return AggregationResult(aggregate, weights, _flags_from_weights(weights))


def coordinate_wise_median(updates: torch.Tensor) -> AggregationResult:
    """Per-coordinate median. With an even number of clients, the mean of the middle two."""
    n = updates.shape[0]
    order = torch.argsort(updates, dim=0)
    middle = order[n // 2:n // 2 + 1] if n % 2 else order[n // 2 - 1:n // 2 + 1]
    aggregate = torch.gather(updates, 0, middle).mean(dim=0)
    counts = torch.bincount(middle.reshape(-1), minlength=n).double()
    weights = (counts / counts.sum()).numpy()
    return AggregationResult(aggregate, weights, _flags_from_weights(weights))
