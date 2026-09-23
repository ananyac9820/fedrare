"""
Baseline aggregation rules, in the aggregate() interface of src/federated/interface.py.

    fedavg                  size-weighted mean (optionally with median-norm clipping)
    fedavg_clipped          fedavg with clip=True - the design doc's "FedAvg with clipping"
    krum, multi_krum,       the existing rules in src/federated/robust.py, unchanged,
    trimmed_mean,           wrapped to take and return model states
    coordinate_wise_median
    camp_a                  evidence-proportional per-class weighting: EARN with trust
                            switched off (design doc Section 6.3)

All are stateless functions: pass them directly, or `lambda: fedavg` as a factory.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from . import robust
from .interface import (DEFAULT_HEAD_PREFIX, INTERFACE_VERSION, State, apply_delta, flatten,
                        state_delta, unflatten)
from .primitives import (FlatLayout, cap_weights, clip_to_median_norm, evidence_share,
                         head_evidence)


def _stack_updates(client_states: Sequence[State], global_state: State, layout: FlatLayout):
    return torch.stack([flatten(state_delta(cs, global_state, layout.keys), layout.keys)
                        for cs in client_states])


def _new_state(global_state: State, aggregate: torch.Tensor, layout: FlatLayout) -> State:
    return apply_delta(global_state, unflatten(aggregate, global_state, layout.keys))


def _info(rule: str, round_num: int, body_w, head_w, clip_scale, **extra) -> dict:
    head_w = np.asarray(head_w, dtype=np.float64)
    info = {
        "interface_version": INTERFACE_VERSION, "rule": rule, "round": int(round_num),
        "selected": (head_w > 0).any(axis=1) | (np.asarray(body_w) > 0),
        "body_weights": np.asarray(body_w, dtype=np.float64),
        "head_row_weights": head_w,
        "clip_scale": np.asarray(clip_scale, dtype=np.float64),
    }
    info.update(extra)
    return info


def fedavg(client_states, client_sizes, global_state, round_num, *, clip: bool = False,
           head_prefix: str = DEFAULT_HEAD_PREFIX, **kwargs):
    """Size-weighted mean of the updates. clip=True clips each to the median norm first."""
    layout = FlatLayout.from_state(global_state, head_prefix)
    updates = _stack_updates(client_states, global_state, layout)
    scale = np.ones(len(client_states))
    if clip:
        updates, scale = clip_to_median_norm(updates)
    w = np.asarray(client_sizes, dtype=np.float64)
    w /= w.sum()
    agg = (torch.as_tensor(w)[:, None] * updates).sum(dim=0)
    head_w = np.tile(w[:, None], (1, layout.n_classes))
    return _new_state(global_state, agg, layout), _info(
        "fedavg_clipped" if clip else "fedavg", round_num, w, head_w, scale)


def fedavg_clipped(client_states, client_sizes, global_state, round_num, **kwargs):
    kwargs.pop("clip", None)
    return fedavg(client_states, client_sizes, global_state, round_num, clip=True, **kwargs)


def _client_level(rule_name: str, rule_fn, client_states, global_state, round_num, head_prefix):
    """Krum-style rules: one selection for the whole update, same weights on every row."""
    layout = FlatLayout.from_state(global_state, head_prefix)
    updates = _stack_updates(client_states, global_state, layout)
    res = rule_fn(updates)
    head_w = np.tile(res.weights[:, None], (1, layout.n_classes))
    return _new_state(global_state, res.aggregate, layout), _info(
        rule_name, round_num, res.weights, head_w, np.ones(len(client_states)))


def _coordinate_wise(rule_name: str, rule_fn, client_states, global_state, round_num,
                     head_prefix):
    """Trimmed mean / median act per coordinate, so per-row weights are computed per row."""
    layout = FlatLayout.from_state(global_state, head_prefix)
    updates = _stack_updates(client_states, global_state, layout)
    res = rule_fn(updates)
    # Each coordinate is aggregated independently, so running the rule on one row's columns
    # gives exactly that row's aggregate and that row's effective client weights.
    head_w = np.stack([rule_fn(updates[:, cols]).weights for cols in layout.head_cols], axis=1)
    body_w = rule_fn(updates[:, layout.body_cols]).weights if len(layout.body_cols) else res.weights
    return _new_state(global_state, res.aggregate, layout), _info(
        rule_name, round_num, body_w, head_w, np.ones(len(client_states)))


def krum(client_states, client_sizes, global_state, round_num, *, byzantine_f: int = 1,
         head_prefix: str = DEFAULT_HEAD_PREFIX, **kwargs):
    return _client_level("krum", lambda u: robust.krum(u, byzantine_f),
                         client_states, global_state, round_num, head_prefix)


def multi_krum(client_states, client_sizes, global_state, round_num, *, byzantine_f: int = 1,
               head_prefix: str = DEFAULT_HEAD_PREFIX, **kwargs):
    return _client_level("multi_krum", lambda u: robust.multi_krum(u, byzantine_f),
                         client_states, global_state, round_num, head_prefix)


def trimmed_mean(client_states, client_sizes, global_state, round_num, *, trim: int = 1,
                 head_prefix: str = DEFAULT_HEAD_PREFIX, **kwargs):
    return _coordinate_wise("trimmed_mean", lambda u: robust.trimmed_mean(u, trim),
                            client_states, global_state, round_num, head_prefix)


def coordinate_wise_median(client_states, client_sizes, global_state, round_num, *,
                           head_prefix: str = DEFAULT_HEAD_PREFIX, **kwargs):
    return _coordinate_wise("coordinate_wise_median", robust.coordinate_wise_median,
                            client_states, global_state, round_num, head_prefix)


def camp_a(client_states, client_sizes, global_state, round_num, *, lam: float = 1.0,
           cap: float = 0.5, clip: bool = True, head_prefix: str = DEFAULT_HEAD_PREFIX,
           evidence_fn=None, evidence_kind: str = "weight", rule_name: str = "camp_a",
           **kwargs):
    """Evidence-proportional per-class weighting - EARN's weighting with trust fixed at 1.

    Per the design doc's pseudocode with T[k][c] = 1 for everyone:
        U[k]  = clip(update_k, median_norm)                       (clip=True)
        body  = size-weighted mean of U[k].body                   (plain FedAvg)
        w[k]  = size_share_k + lam * e[k][c] / sum_j e[j][c]      for each head row c
        row c = sum_k cap(norm(w), cap)[k] * U[k].row[c]

    With T = 0 this would reduce to FedAvg; with T = 1 every client that looks like a holder
    gets the full evidence bonus without having earned it - the Camp A behaviour EARN is
    meant to fix. lam = 1 weighs size share and evidence share equally; the design doc
    leaves lam unspecified, so it is a named constant here, not a tuned value.

    evidence_fn(clipped_updates, layout, round_num) -> (K, C) replaces the head-row evidence;
    the oracle-evidence runs (docs/DEVIATIONS.md D6) use it for a Camp A that weights by
    reported class counts - the cwFedAvg / FedSat style of the literature.
    """
    layout = FlatLayout.from_state(global_state, head_prefix)
    updates = _stack_updates(client_states, global_state, layout)
    scale = np.ones(len(client_states))
    if clip:
        updates, scale = clip_to_median_norm(updates)

    size_share = np.asarray(client_sizes, dtype=np.float64)
    size_share /= size_share.sum()
    evidence = (np.asarray(evidence_fn(updates, layout, round_num), dtype=np.float64)
                if evidence_fn is not None else head_evidence(updates, layout, evidence_kind))
    share = evidence_share(evidence)

    agg = torch.zeros(updates.shape[1], dtype=updates.dtype)
    if len(layout.body_cols):
        agg[layout.body_cols] = (torch.as_tensor(size_share)[:, None]
                                 * updates[:, layout.body_cols]).sum(dim=0)
    head_w = np.zeros((len(client_states), layout.n_classes))
    for c, cols in enumerate(layout.head_cols):
        head_w[:, c] = cap_weights(size_share + lam * share[:, c], cap)
        agg[cols] = (torch.as_tensor(head_w[:, c])[:, None] * updates[:, cols]).sum(dim=0)

    return _new_state(global_state, agg, layout), _info(
        rule_name, round_num, size_share, head_w, scale,
        evidence=evidence, evidence_share=share)


RULES = {
    "fedavg": fedavg,
    "fedavg_clipped": fedavg_clipped,
    "krum": krum,
    "multi_krum": multi_krum,
    "trimmed_mean": trimmed_mean,
    "coordinate_wise_median": coordinate_wise_median,
    "camp_a": camp_a,
}
