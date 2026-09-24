"""
Tier A federated training loop: every client trains only the classifier head on frozen
DenseNet-121 features (design doc Section 7). Any aggregator in the aggregate() interface
(src/federated/interface.py) plugs in unchanged.

    result = run_federated(lambda: baselines.fedavg, clients, test, TierAConfig(), seed=42,
                           rare_ids=[5, 6])

Attacks plug in through client_hooks (see ClientHook) without editing this file.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable, Protocol, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.loader import class_balanced_weights
from src.utils.metrics import confusion, summarise
from src.utils.seed import set_seed

from .interface import DEFAULT_HEAD_PREFIX, AggregatorFactory, State, validate_round_info


@dataclass(frozen=True)
class TierAConfig:
    """Local training and schedule. Fixed before any Tier A result exists."""
    rounds: int = 40
    local_steps: int = 50          # same number of steps for every client each round
    batch_size: int = 32
    lr: float = 5e-4               # configs/default.yaml training.lr, Adam
    class_balanced_loss: bool = True
    feature_dim: int = 1024
    n_classes: int = 8


class TierAHead(nn.Module):
    """DenseNet-121's classifier layer on its own. State keys: classifier.weight / .bias."""

    def __init__(self, feature_dim: int = 1024, n_classes: int = 8):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        return self.classifier(x)


@dataclass
class ClientData:
    x: torch.Tensor   # (n, feature_dim) float32
    y: torch.Tensor   # (n,) int64

    @property
    def size(self) -> int:
        return len(self.y)


def make_clients(features: np.ndarray, labels: np.ndarray, centers: np.ndarray,
                 n_centers: int) -> list[ClientData]:
    """One ClientData per centre id 0..n_centers-1, under the given centre assignment."""
    x, y = torch.from_numpy(np.ascontiguousarray(features)), torch.from_numpy(labels)
    return [ClientData(x[torch.from_numpy(np.flatnonzero(centers == k))],
                       y[torch.from_numpy(np.flatnonzero(centers == k))]) for k in range(n_centers)]


def local_train(global_state: State, x: torch.Tensor, y: torch.Tensor, cfg: TierAConfig,
                rng: np.random.Generator) -> State:
    """Train the head from global_state on (x, y); return the new state (detached copies).

    Class-balanced loss weights come from the labels this client actually trains on.
    """
    model = TierAHead(cfg.feature_dim, cfg.n_classes)
    model.load_state_dict({k: v.detach().clone() for k, v in global_state.items()})
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    weight = None
    if cfg.class_balanced_loss:
        weight = class_balanced_weights(np.bincount(y.numpy(), minlength=cfg.n_classes))

    n = len(y)
    perm, pos = rng.permutation(n), 0
    for _ in range(cfg.local_steps):
        if pos + cfg.batch_size > n:
            perm, pos = rng.permutation(n), 0
        idx = torch.from_numpy(perm[pos:pos + cfg.batch_size])
        pos += cfg.batch_size
        opt.zero_grad()
        F.cross_entropy(model(x[idx]), y[idx], weight=weight).backward()
        opt.step()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


class HonestTrain(Protocol):
    def __call__(self, x: torch.Tensor | None = None, y: torch.Tensor | None = None) -> State: ...


class ClientHook(Protocol):
    """Replaces one client's local training - how attacks plug in.

    Called as hook(round_num=r, client_id=k, global_state=..., data=ClientData,
    honest_train=fn) and must return the client's state_dict for this round.
    honest_train(x, y) runs the normal local training from this round's global state, on
    (x, y) if given or the client's own data otherwise - so a label-flipping attack calls
    honest_train(data.x, flipped_y), and a scaling attack post-processes honest_train().
    """

    def __call__(self, *, round_num: int, client_id: int, global_state: State,
                 data: ClientData, honest_train: HonestTrain) -> State: ...


@torch.no_grad()
def evaluate(state: State, x: torch.Tensor, y: np.ndarray, rare_ids: list[int],
             n_classes: int) -> dict:
    w, b = state[f"{DEFAULT_HEAD_PREFIX}.weight"], state[f"{DEFAULT_HEAD_PREFIX}.bias"]
    pred = (x @ w.T + b).argmax(1).numpy()
    out = summarise(y, pred, rare_ids, n_classes)
    out["confusion"] = confusion(y, pred, n_classes).tolist()
    return out


def run_federated(aggregator_factory: AggregatorFactory, clients: Sequence[ClientData],
                  test_x: torch.Tensor, test_y: np.ndarray, cfg: TierAConfig, seed: int,
                  rare_ids: list[int], client_hooks: dict[int, ClientHook] | None = None,
                  aggregate_kwargs: dict | None = None, verbose: bool = False,
                  on_round: Callable[[int, dict, dict], None] | None = None) -> dict:
    """Run cfg.rounds of Tier A federated training. Returns a result dict:

        config, seed, rule
        round_infos     list of round_info dicts, one per round (validated)
        round_metrics   list of test metrics of the global head after each round
        final           test metrics after the last round
        final_state     the global head after the last round (tensors, not JSON-serialisable)
        seconds         wall time
    """
    set_seed(seed)
    client_hooks = client_hooks or {}
    aggregate_kwargs = dict(aggregate_kwargs or {})
    aggregate_kwargs.setdefault("n_classes", cfg.n_classes)
    aggregate_kwargs.setdefault("head_prefix", DEFAULT_HEAD_PREFIX)

    global_state = {k: v.detach().clone()
                    for k, v in TierAHead(cfg.feature_dim, cfg.n_classes).state_dict().items()}
    aggregator = aggregator_factory()
    sizes = [c.size for c in clients]
    round_infos, round_metrics = [], []
    t0 = time.time()

    for r in range(1, cfg.rounds + 1):
        client_states = []
        for k, data in enumerate(clients):
            rng = np.random.default_rng([seed, r, k])

            def honest_train(x=None, y=None, _data=data, _rng=rng, _g=global_state):
                return local_train(_g, _data.x if x is None else x, _data.y if y is None else y,
                                   cfg, _rng)

            hook = client_hooks.get(k)
            client_states.append(
                hook(round_num=r, client_id=k, global_state=global_state, data=data,
                     honest_train=honest_train) if hook else honest_train())

        global_state, info = aggregator(client_states, sizes, global_state, r, **aggregate_kwargs)
        validate_round_info(info, len(clients), cfg.n_classes)
        metrics = evaluate(global_state, test_x, test_y, rare_ids, cfg.n_classes)
        round_infos.append(info)
        round_metrics.append(metrics)
        if on_round:
            on_round(r, info, metrics)
        if verbose:
            print(f"    round {r:>2}/{cfg.rounds} | balanced acc {metrics['balanced_accuracy']:.3f}"
                  f" | rare macro-F1 {metrics['rare_macro_f1']:.3f}", flush=True)

    return {"config": asdict(cfg), "seed": seed, "rule": round_infos[-1]["rule"],
            "round_infos": round_infos, "round_metrics": round_metrics,
            "final": round_metrics[-1], "final_state": global_state,
            "seconds": time.time() - t0}
