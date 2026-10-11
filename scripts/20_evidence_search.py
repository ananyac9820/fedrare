#!/usr/bin/env python3
"""
A pre-registered search for a replacement evidence signal (docs/DEVIATIONS.md D11).

G0a is decided and stays failed. This measures five candidate definitions of evidence, all
declared in D11 before any was computed, against the same yardstick the gate used: round 1
of the Tier A loop, seed 42, natural split S1, updates clipped to the median norm, Spearman
per class between the evidence and the true per-class training counts across the six
centres, statistic = mean over the 8 classes, bar 0.7.

    C1  signed bias change, per-client normalised
    C2  signed weight-row change, per-client normalised
    C3  cosine with the size-weighted mean row-c update (peer consensus direction)
    C4  probe-based, CELM-lite: per-class probe by gradient ascent on the global head
    C5  the positive part of C1

Every candidate is reported, pass or fail, and a candidate that clears the bar proves only
that a faithful signal exists - D11 rule 2 requires a fresh pre-registration with an attack
check before anything here could become EARN's signal.

    python scripts/20_evidence_search.py

Writes docs/results/evidence_search.json.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scipy.stats import spearmanr

from src.data.features import load_features
from src.federated.interface import flatten, state_delta
from src.federated.primitives import FlatLayout, clip_to_median_norm, head_evidence
from src.federated.tier_a import TierAConfig, TierAHead, local_train, make_clients
from src.utils.config import load_config
from src.utils.seed import set_seed

OUT = ROOT / "docs" / "results"
SEED, ROUND, BAR = 42, 1, 0.7
N_CENTRES = 6


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    rho = spearmanr(a, b).statistic
    return 0.0 if np.isnan(rho) else float(rho)


def rows_and_bias(updates: torch.Tensor, layout: FlatLayout, n_classes: int):
    """Per-client classifier rows (K, C, D) and bias changes (K, C) from flat updates."""
    # FlatLayout stores column indices per head row, not slices.
    w = np.stack([updates[:, layout.head_weight_cols[c]].numpy() for c in range(n_classes)],
                 axis=1)                                              # (K, C, D)
    if layout.head_bias_cols is None:
        return w, np.zeros((len(updates), n_classes))
    b = np.stack([updates[:, layout.head_bias_cols[c]].numpy() for c in range(n_classes)],
                 axis=1)                                              # (K, C)
    return w, b


def candidates(updates, layout, global_state, clients, n_classes):
    """The five definitions from D11, in the order they were declared."""
    w, b = rows_and_bias(updates, layout, n_classes)
    sizes = np.array([c.size for c in clients], dtype=float)
    out = {}

    # C1 - signed bias change, per-client normalised by that client's total bias movement.
    denom = np.abs(b).sum(axis=1, keepdims=True)
    out["C1"] = b / np.where(denom > 0, denom, 1.0)

    # C2 - signed weight-row change (sum along the row), per-client normalised.
    row_sum = w.sum(axis=2)
    denom = np.abs(row_sum).sum(axis=1, keepdims=True)
    out["C2"] = row_sum / np.where(denom > 0, denom, 1.0)

    # C3 - cosine between each client's row c and the size-weighted mean row c.
    share = sizes / sizes.sum()
    consensus = (share[:, None, None] * w).sum(axis=0)              # (C, D)
    cos = np.zeros((len(w), n_classes))
    for k in range(len(w)):
        for c in range(n_classes):
            a, m = w[k, c], consensus[c]
            na, nm = np.linalg.norm(a), np.linalg.norm(m)
            cos[k, c] = float(a @ m / (na * nm)) if na > 1e-12 and nm > 1e-12 else 0.0
    out["C3"] = cos

    # C4 - CELM-lite. One probe per class, made by pushing a feature vector up the global
    # head's logit c, then ask how much each client's head raises that logit.
    gw = global_state["classifier.weight"].detach().clone()
    gb = global_state["classifier.bias"].detach().clone()
    probes = []
    for c in range(n_classes):
        torch.manual_seed(SEED + c)
        x = torch.zeros(1, gw.shape[1], requires_grad=True)
        opt = torch.optim.Adam([x], lr=0.1)
        for _ in range(100):
            opt.zero_grad()
            (-(x @ gw.T + gb)[0, c]).backward()
            opt.step()
        probes.append(x.detach()[0])
    probe = torch.stack(probes)                                     # (C, D)
    base = (probe @ gw.T + gb).diagonal().numpy()                   # global logit c on probe c
    c4 = np.zeros((len(w), n_classes))
    for k in range(len(w)):
        kw = gw + torch.as_tensor(w[k], dtype=gw.dtype)
        kb = gb + torch.as_tensor(b[k], dtype=gb.dtype)
        c4[k] = (probe @ kw.T + kb).diagonal().numpy() - base
    out["C4"] = c4

    # C5 - the positive part of C1: a push down is not evidence of holdings.
    out["C5"] = np.maximum(0.0, out["C1"])
    return out


def main() -> int:
    cfg_yaml = load_config(ROOT / "configs" / "default.yaml")
    n_classes = cfg_yaml["dataset"]["n_classes"]
    names = [cfg_yaml["dataset"]["class_names"][c] for c in range(n_classes)]
    cfg = TierAConfig(n_classes=n_classes)

    train = load_features("train")
    counts = train.counts(N_CENTRES, n_classes)
    clients = make_clients(train.features, train.labels, train.centers, N_CENTRES)
    sizes = np.array([c.size for c in clients], dtype=float)

    # Round 1, exactly as the gate ran it.
    set_seed(SEED)
    global_state = {k: v.detach().clone()
                    for k, v in TierAHead(cfg.feature_dim, n_classes).state_dict().items()}
    states = [local_train(global_state, c.x, c.y, cfg, np.random.default_rng([SEED, ROUND, k]))
              for k, c in enumerate(clients)]
    layout = FlatLayout.from_state(global_state)
    updates = torch.stack([flatten(state_delta(s, global_state, layout.keys), layout.keys)
                           for s in states])
    clipped, _ = clip_to_median_norm(updates)

    # The gate's own signal, recomputed here so the comparison is on one screen.
    tested = {"G0a (the design doc's signal)": head_evidence(clipped, layout)}
    tested.update(candidates(clipped, layout, global_state, clients, n_classes))

    print("=" * 78)
    print("D11 - candidate evidence signals, scored exactly as gate G0a scores")
    print(f"seed {SEED}, round {ROUND}, split S1, bar {BAR}")
    print("=" * 78)

    results = {}
    for name, ev in tested.items():
        ev = np.asarray(ev, dtype=float)
        per_class = np.array([spearman(ev[:, c], counts[:, c]) for c in range(n_classes)])
        vs_size = np.array([spearman(ev[:, c], sizes) for c in range(n_classes)])
        stat = float(per_class.mean())
        results[name] = {
            "statistic": stat, "passed": bool(stat >= BAR),
            "per_class": {names[c]: float(per_class[c]) for c in range(n_classes)},
            "mean_spearman_vs_size": float(vs_size.mean()),
        }
        print(f"\n  {name}")
        print(f"    mean Spearman vs true counts : {stat:+.3f}   "
              f"{'CLEARS THE BAR' if stat >= BAR else 'under the bar'}")
        print(f"    mean Spearman vs centre size : {vs_size.mean():+.3f}")
        print("    per class: " + "  ".join(
            f"{names[c][:12]} {per_class[c]:+.2f}" for c in range(n_classes)))

    best = max(results, key=lambda k: results[k]["statistic"])
    record = {
        "entry": "D11", "fixed_in": "docs/DEVIATIONS.md D11, before any candidate was computed",
        "measured_like": "gate G0a: round 1, seed 42, S1, clipped updates, mean Spearman over 8 classes",
        "bar": BAR, "seed": SEED,
        "candidates": results,
        "best": best, "best_statistic": results[best]["statistic"],
        "any_passed": any(r["passed"] for r in results.values()),
        "note": ("D11 rule 2: clearing the bar here would show only that a faithful signal exists. "
                 "It would need a fresh pre-registration including an A1 spoofability check before "
                 "becoming EARN's signal, because R8 showed faithful-on-honest-data and "
                 "safe-against-a-liar are different properties."),
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evidence_search.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"  best: {best} at {results[best]['statistic']:+.3f} against a bar of {BAR}")
    print(f"  {'A candidate clears it.' if record['any_passed'] else 'None clears it.'} "
          f"Fallback F1 stands either way until a fresh pre-registration says otherwise.")
    print("=" * 78)
    print(f"wrote {(OUT / 'evidence_search.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
