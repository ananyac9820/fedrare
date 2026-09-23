"""
EARN - Earned, Audited, Rare-class-aware aggregatioN (EARN Project Design v2, Section 4).

STATUS: implemented and unit-tested, but NOT validated. Gate G0a failed after its one retry
(docs/DEVIATIONS.md R1): the evidence signal the design doc specifies does not track which
hospitals hold a disease. EARN is therefore evaluated only as an exploratory, labelled
analysis on an ORACLE evidence signal (D6) - "what the mechanism would do if a working,
spoofable signal existed" - and never presented as a validated method.

One round, per the design doc's pseudocode (Section 4.4):

    U[k]   = clip(update_k, median_norm)
    e[k,c] = evidence of client k on class c           (evidence_fn; see below)
    holders(c) = {k : e[k,c] > tau};  n(c) = |holders(c)|
    p(c)   = clip((n(c) - 1) / divisor, 0, 1)                           peer confidence
    for k in holders(c):
        a(k,c) = p(c) * cos(U[k].row_c, median_{j in holders, j != k} U[j].row_c)
               + (1 - p(c)) * cos(U[k].row_c, history(k,c))              history is locked
        T[k,c] = min(1, T + step) if a >= agree else T * decay         slow up, fast down
    body     = size-weighted mean of U[k].body                          plain FedAvg
    w[k,c]   = size_share_k + lam * T[k,c] * e[k,c] / sum_j e[j,c]
    row c    = sum_k cap(norm(w[:,c]), cap)[k] * U[k].row_c
    ledger.commit(round, T, n, hash(history))   - rejects any T rise > step

Choices the design doc leaves open, fixed here before any EARN result (D6):
  - history(k,c): running mean of client k's clipped row-c updates over all its previous rounds.
    It is updated after this round's check. The ledger stores its hash every round.
  - "no history" (round 1, or an ablation) and "no peers" (n(c) = 1) each contribute the
    neutral score NEUTRAL = 0.5 in place of the missing cosine.
  - Clients that are not holders of c this round keep their T[k,c] unchanged.
  - Trust starts at 0 for everyone. With all trust at 0, EARN is clipped FedAvg with the 50%
    row cap applied (centre 0 holds 53.4% of the training images, so "reduces exactly to
    FedAvg" and "no hospital may exceed 50% of any row" cannot both hold; we keep the cap).

Ablations (design doc Section 6.3), as constructor flags:
  blend=False       no coverage blend: peer check only (p = 1 for every class)
  slow_ramp=False   trust jumps straight to 1 on agreement
  ledger=False      history editable: every client in `editable_history` rewrites its stored
                    history to equal its current update before the check (the attack a locked
                    history prevents); no chain is kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch

from src.ledger.chain import HashChainLedger, hash_arrays

from .interface import DEFAULT_HEAD_PREFIX, INTERFACE_VERSION, State, apply_delta, flatten, \
    state_delta, unflatten
from .primitives import FlatLayout, cap_weights, clip_to_median_norm, evidence_share, head_evidence

NEUTRAL = 0.5

# evidence_fn(clipped_updates (K, D), layout, round_num) -> (K, C) array, >= 0
EvidenceFn = Callable[[torch.Tensor, FlatLayout, int], np.ndarray]


@dataclass
class EARNConfig:
    step: float = 0.1            # trust rise per agreeing round
    decay: float = 0.5           # trust multiplier on a disagreeing round
    agree: float = 0.5           # agreement threshold
    cap: float = 0.5             # no client above 50% of any row
    divisor: float = 4.0         # p(c) = clip((n - 1) / divisor, 0, 1)
    lam: float = 1.0             # evidence bonus scale (same as Camp A)
    tau: float = 0.0             # holder threshold on evidence
    blend: bool = True
    slow_ramp: bool = True
    ledger: bool = True
    editable_history: tuple[int, ...] = field(default_factory=tuple)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = float(a.norm()), float(b.norm())
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b) / (na * nb)


class EARN:
    """Stateful aggregator: build a fresh instance per run (run_federated takes a factory)."""

    def __init__(self, cfg: EARNConfig | None = None, evidence_fn: EvidenceFn | None = None,
                 evidence_kind: str = "weight", name: str = "earn"):
        self.cfg = cfg or EARNConfig()
        self.evidence_fn = evidence_fn
        self.evidence_kind = evidence_kind
        self.name = name
        self.trust: np.ndarray | None = None
        self.hist_sum: np.ndarray | None = None      # (K, C, row_len) running sums
        self.hist_n: np.ndarray | None = None        # (K, C) rounds recorded
        self.chain: HashChainLedger | None = None

    # ------------------------------------------------------------------ helpers
    def _init(self, K: int, layout: FlatLayout):
        C, row_len = layout.n_classes, len(layout.head_cols[0])
        self.trust = np.zeros((K, C))
        self.hist_sum = np.zeros((K, C, row_len))
        self.hist_n = np.zeros((K, C), dtype=np.int64)
        if self.cfg.ledger:
            # the no-slow-ramp ablation jumps trust to 1, which the step rule exists to forbid
            self.chain = HashChainLedger(K, C, max_step=self.cfg.step if self.cfg.slow_ramp else 1.0)

    def history(self, k: int, c: int) -> np.ndarray | None:
        n = self.hist_n[k, c]
        return None if n == 0 else self.hist_sum[k, c] / n

    def history_hash(self) -> str:
        return hash_arrays(self.hist_sum, self.hist_n)

    # ------------------------------------------------------------------ the round
    def __call__(self, client_states: Sequence[State], client_sizes, global_state: State,
                 round_num: int, *, head_prefix: str = DEFAULT_HEAD_PREFIX, **kwargs):
        cfg = self.cfg
        layout = FlatLayout.from_state(global_state, head_prefix)
        K, C = len(client_states), layout.n_classes
        if self.trust is None:
            self._init(K, layout)

        updates = torch.stack([flatten(state_delta(cs, global_state, layout.keys), layout.keys)
                               for cs in client_states])
        updates, scale = clip_to_median_norm(updates)
        evidence = (self.evidence_fn(updates, layout, round_num) if self.evidence_fn
                    else head_evidence(updates, layout, self.evidence_kind))
        evidence = np.asarray(evidence, dtype=np.float64)
        holders = evidence > cfg.tau
        coverage = holders.sum(axis=0).astype(np.int64)
        p = (np.ones(C) if not cfg.blend
             else np.clip((coverage - 1) / cfg.divisor, 0.0, 1.0))

        rows = [updates[:, cols] for cols in layout.head_cols]         # C x (K, row_len)
        trust_prev = self.trust.copy()
        agreement = np.full((K, C), np.nan)
        for c in range(C):
            members = np.flatnonzero(holders[:, c])
            for k in members:
                row = rows[c][k]
                peers = [j for j in members if j != k]
                peer_score = (_cos(row, rows[c][peers].median(dim=0).values) if peers
                              else NEUTRAL)
                if k in cfg.editable_history:          # ledger=False ablation only
                    hist_score = 1.0
                else:
                    hist = self.history(k, c)
                    hist_score = NEUTRAL if hist is None else _cos(row, torch.as_tensor(hist))
                a = p[c] * peer_score + (1 - p[c]) * hist_score
                agreement[k, c] = a
                if a >= cfg.agree:
                    self.trust[k, c] = min(1.0, self.trust[k, c] + cfg.step) if cfg.slow_ramp else 1.0
                else:
                    self.trust[k, c] = self.trust[k, c] * cfg.decay

        # record this round into each client's history (after the check)
        for c in range(C):
            self.hist_sum[:, c] += rows[c].double().cpu().numpy()
        self.hist_n += 1

        # aggregate
        size_share = np.asarray(client_sizes, dtype=np.float64)
        size_share = size_share / size_share.sum()
        share = evidence_share(evidence)
        agg = torch.zeros(updates.shape[1], dtype=updates.dtype)
        if len(layout.body_cols):
            agg[layout.body_cols] = (torch.as_tensor(size_share)[:, None]
                                     * updates[:, layout.body_cols]).sum(dim=0)
        head_w = np.zeros((K, C))
        for c, cols in enumerate(layout.head_cols):
            head_w[:, c] = cap_weights(size_share + cfg.lam * self.trust[:, c] * share[:, c], cfg.cap)
            agg[cols] = (torch.as_tensor(head_w[:, c])[:, None] * updates[:, cols]).sum(dim=0)

        history_hash = self.history_hash()
        block_hash = None
        if self.chain is not None:
            block_hash = self.chain.commit(round_num, self.trust, coverage, history_hash).block_hash

        new_state = apply_delta(global_state, unflatten(agg, global_state, layout.keys))
        info = {
            "interface_version": INTERFACE_VERSION, "rule": self.name, "round": int(round_num),
            "selected": (head_w > 0).any(axis=1) | (size_share > 0),
            "body_weights": size_share, "head_row_weights": head_w,
            "clip_scale": np.asarray(scale, dtype=np.float64),
            "evidence": evidence, "evidence_share": share, "holders": holders,
            "coverage": coverage, "peer_confidence": p.astype(np.float64),
            "agreement": agreement, "trust": self.trust.copy(), "trust_prev": trust_prev,
            "history_hash": history_hash,
        }
        if block_hash is not None:
            info["block_hash"] = block_hash
        return new_state, info
