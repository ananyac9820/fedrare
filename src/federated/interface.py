"""
The aggregation interface: the one seam between the training loop, the aggregation rules
(FedAvg, robust rules, Camp A, EARN) and the attacks that test them.

Pipeline & EARN (feature/pipeline) and Attacks & Eval (feature/attacks) build against this
contract independently. Do not change it without telling the other track. A human-readable
copy with examples lives in docs/aggregation_interface.md.

INTERFACE_VERSION = 1


The signature
-------------

    new_global_state, round_info = aggregate(client_states, client_sizes, global_state,
                                             round_num, **kwargs)

An aggregator is any callable with that signature. Stateless rules (fedavg, krum, camp_a)
are plain functions. Rules that carry state across rounds (EARN: trust table, history) are
instances of a class whose __call__ has this signature; the training loop builds a fresh
instance per run from a zero-argument factory, so no state leaks between runs or seeds.

Inputs
------
client_states : list[dict[str, torch.Tensor]], length K
    Client k's full model state_dict after its local training this round - weights, not
    updates. Same keys and shapes as global_state. Index k is the client (centre) id and
    stays the same in every round, so client 2 is always centre 2.
client_sizes : sequence[int], length K
    Number of local training images per client. Used only for size-weighted parts (FedAvg,
    and EARN's body). EARN never uses it to judge a class - no self-reported class counts.
global_state : dict[str, torch.Tensor]
    The state every client started from this round. Updates are client - global.
round_num : int
    1-based round index.
**kwargs
    Rule-specific options (documented on each rule). Every aggregator must accept and
    ignore keyword arguments it does not use, so the loop can pass one common set.
    The loop always passes: n_classes (int), head_prefix (str, default "classifier").

Model layout convention
-----------------------
The classifier head is every state key starting with head_prefix + "." - for both Tier A
(a single nn.Linear named `classifier`) and Tier B (torchvision DenseNet-121, whose last
layer is also `classifier`). Head row c is `classifier.weight[c]` together with
`classifier.bias[c]`: everything the model uses to score disease c. All other floating-point
keys are the body (Tier A has no body). Non-floating-point entries (e.g. BatchNorm's
num_batches_tracked) are copied from global_state unchanged.

Outputs
-------
new_global_state : dict[str, torch.Tensor]
    Same keys, shapes and dtypes as global_state. Tensors are new objects (never views of
    client or global tensors).

round_info : dict. Required keys, for every aggregator:
    "interface_version"  int                 INTERFACE_VERSION
    "rule"               str                 e.g. "fedavg", "krum", "camp_a", "earn"
    "round"              int                 round_num
    "selected"           bool   (K,)         client influenced the new global state at all
    "body_weights"       float  (K,)         effective weight on body parameters; sums to 1.
                                             For rules without a body/head split this is the
                                             client-level weight.
    "head_row_weights"   float  (K, C)       effective weight client k received on head
                                             row c; every column sums to 1. For client-level
                                             rules each column equals the client weights
                                             (per-row for coordinate-wise rules). This is
                                             what "attacker's captured weight on a rare
                                             row" is read from.
    "clip_scale"         float  (K,)         factor each update was multiplied by before
                                             aggregation; 1.0 = not clipped.

Optional keys - added by evidence-based rules (Camp A, EARN). If present they must have
exactly these shapes and meanings:
    "evidence"           float  (K, C)       e(k,c) = L2 norm of client k's (clipped) update
                                             to head weight row c. >= 0.
    "evidence_share"     float  (K, C)       e(k,c) / sum_j e(j,c); columns sum to 1
                                             (all-zero column -> all zeros).
    "holders"            bool   (K, C)       e(k,c) > tau                          (EARN)
    "coverage"           int    (C,)         n(c) = holders per class              (EARN)
    "peer_confidence"    float  (C,)         p(c) = clip((n(c) - 1) / 4, 0, 1)     (EARN)
    "agreement"          float  (K, C)       a(k,c); NaN where not computed        (EARN)
    "trust"              float  (K, C)       T(k,c) in [0, 1] AFTER this round     (EARN)
    "trust_prev"         float  (K, C)       T(k,c) before this round's update     (EARN)
    "history_hash"       str                 hex digest committed to the ledger    (EARN)

All arrays are numpy, float64 / bool / int64. Use validate_round_info() in tests.
"""

from __future__ import annotations

from typing import Callable, Mapping, Protocol, Sequence

import numpy as np
import torch

INTERFACE_VERSION = 1
DEFAULT_HEAD_PREFIX = "classifier"

State = dict[str, torch.Tensor]

REQUIRED_KEYS = ("interface_version", "rule", "round", "selected", "body_weights",
                 "head_row_weights", "clip_scale")
# key -> (dtype kind, shape spec using K and C)
OPTIONAL_KEYS = {
    "evidence": ("f", ("K", "C")),
    "evidence_share": ("f", ("K", "C")),
    "holders": ("b", ("K", "C")),
    "coverage": ("i", ("C",)),
    "peer_confidence": ("f", ("C",)),
    "agreement": ("f", ("K", "C")),
    "trust": ("f", ("K", "C")),
    "trust_prev": ("f", ("K", "C")),
}


class Aggregator(Protocol):
    def __call__(self, client_states: Sequence[State], client_sizes: Sequence[int],
                 global_state: State, round_num: int, **kwargs) -> tuple[State, dict]: ...


AggregatorFactory = Callable[[], Aggregator]


# ------------------------------------------------------------------ state helpers

def head_keys(state: Mapping[str, torch.Tensor], prefix: str = DEFAULT_HEAD_PREFIX) -> list[str]:
    return [k for k in state if k.startswith(prefix + ".")]


def float_keys(state: Mapping[str, torch.Tensor]) -> list[str]:
    return [k for k, v in state.items() if v.is_floating_point()]


def body_keys(state: Mapping[str, torch.Tensor], prefix: str = DEFAULT_HEAD_PREFIX) -> list[str]:
    heads = set(head_keys(state, prefix))
    return [k for k in float_keys(state) if k not in heads]


def state_delta(client_state: State, global_state: State, keys: Sequence[str]) -> State:
    """client - global for the given keys, as float64 CPU tensors."""
    return {k: client_state[k].detach().double().cpu() - global_state[k].detach().double().cpu()
            for k in keys}


def apply_delta(global_state: State, delta: Mapping[str, torch.Tensor]) -> State:
    """New state = global + delta on delta's keys; every other key copied unchanged."""
    out = {}
    for k, v in global_state.items():
        if k in delta:
            out[k] = (v.detach().double().cpu() + delta[k]).to(dtype=v.dtype, device=v.device)
        else:
            out[k] = v.detach().clone()
    return out


def flatten(delta: Mapping[str, torch.Tensor], keys: Sequence[str]) -> torch.Tensor:
    return torch.cat([delta[k].reshape(-1) for k in keys]) if keys else torch.zeros(0, dtype=torch.float64)


def unflatten(vec: torch.Tensor, like: Mapping[str, torch.Tensor], keys: Sequence[str]) -> State:
    out, pos = {}, 0
    for k in keys:
        n = like[k].numel()
        out[k] = vec[pos:pos + n].reshape(like[k].shape)
        pos += n
    return out


# ------------------------------------------------------------------ validation

def validate_round_info(info: dict, n_clients: int, n_classes: int, atol: float = 1e-6) -> None:
    """Raise ValueError if round_info breaks the contract. Use this in every rule's tests."""
    missing = [k for k in REQUIRED_KEYS if k not in info]
    if missing:
        raise ValueError(f"round_info missing required keys {missing}")
    if info["interface_version"] != INTERFACE_VERSION:
        raise ValueError(f"interface_version {info['interface_version']} != {INTERFACE_VERSION}")
    K, C = n_clients, n_classes

    def arr(key, kind, shape):
        a = info[key]
        if not isinstance(a, np.ndarray):
            raise ValueError(f"{key} must be a numpy array, got {type(a).__name__}")
        if a.shape != shape:
            raise ValueError(f"{key} shape {a.shape} != {shape}")
        if a.dtype.kind != kind and not (kind == "i" and a.dtype.kind == "u"):
            raise ValueError(f"{key} dtype {a.dtype} is not kind '{kind}'")
        return a

    arr("selected", "b", (K,))
    bw = arr("body_weights", "f", (K,))
    hw = arr("head_row_weights", "f", (K, C))
    cs = arr("clip_scale", "f", (K,))
    if (bw < -atol).any() or abs(bw.sum() - 1) > atol:
        raise ValueError(f"body_weights must be >= 0 and sum to 1, got {bw}")
    if (hw < -atol).any() or not np.allclose(hw.sum(0), 1, atol=atol):
        raise ValueError(f"head_row_weights columns must be >= 0 and sum to 1, got {hw.sum(0)}")
    if (cs <= 0).any() or (cs > 1 + atol).any():
        raise ValueError(f"clip_scale must be in (0, 1], got {cs}")

    sizes = {"K": K, "C": C}
    for key, (kind, spec) in OPTIONAL_KEYS.items():
        if key in info:
            arr(key, kind, tuple(sizes[s] for s in spec))
    if "trust" in info:
        t = info["trust"]
        if (t < -atol).any() or (t > 1 + atol).any():
            raise ValueError("trust must be in [0, 1]")
    if "history_hash" in info and not isinstance(info["history_hash"], str):
        raise ValueError("history_hash must be a hex string")
