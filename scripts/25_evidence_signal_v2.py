#!/usr/bin/env python3
"""
Evidence signal, second pass (docs/DEVIATIONS.md D15).

G0a stays decided. This is a search for a signal, with every candidate declared in D15
before any was computed, scored exactly as the gate scores: Tier A round updates on the
natural split S1, clipped to the median norm, Spearman per class between the evidence and
the true per-class training counts across the six centres, statistic = mean over 8 classes,
bar 0.7.

Two differences from R9, both fixed in D15: three seeds (42/43/44) reported as mean +- sd
rather than one, and multi-round candidates averaged over rounds 1-3.

    E1  cosine with the leave-one-out consensus (drops self-inclusion)
    E2  E1 with an unweighted consensus (drops size from the consensus too)
    E3  C1 averaged over rounds 1-3
    E4  E1 averaged over rounds 1-3
    E5  per-class rank average of C1 and E1

R9's candidates and the design doc's original signal are recomputed under the same protocol.

    python scripts/25_evidence_signal_v2.py

Writes docs/results/evidence_signal_v2.json.
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
from scipy.stats import rankdata, spearmanr

from src.data.features import load_features
from src.federated import baselines
from src.federated.interface import flatten, state_delta
from src.federated.primitives import FlatLayout, clip_to_median_norm, head_evidence
from src.federated.tier_a import TierAConfig, TierAHead, local_train, make_clients
from src.utils.config import load_config
from src.utils.seed import set_seed

OUT = ROOT / "docs" / "results"
SEEDS = (42, 43, 44)
ROUNDS = 3                      # candidates E3/E4 average over rounds 1..3
BAR = 0.7
N_CENTRES = 6


def spearman(a, b):
    rho = spearmanr(a, b).statistic
    return 0.0 if np.isnan(rho) else float(rho)


def rows_and_bias(updates, layout, n_classes):
    w = np.stack([updates[:, layout.head_weight_cols[c]].numpy() for c in range(n_classes)],
                 axis=1)
    if layout.head_bias_cols is None:
        return w, np.zeros((len(updates), n_classes))
    b = np.stack([updates[:, layout.head_bias_cols[c]].numpy() for c in range(n_classes)],
                 axis=1)
    return w, b


def normalised_bias(b):
    """R9's C1: signed bias change, divided by that client's total bias movement."""
    denom = np.abs(b).sum(axis=1, keepdims=True)
    return b / np.where(denom > 0, denom, 1.0)


def loo_cosine(w, sizes, weighted=True):
    """E1/E2: cosine against the consensus of the OTHER clients only."""
    K, C, _ = w.shape
    out = np.zeros((K, C))
    for k in range(K):
        others = [j for j in range(K) if j != k]
        if weighted:
            wt = sizes[others] / sizes[others].sum()
        else:
            wt = np.full(len(others), 1.0 / len(others))
        for c in range(C):
            consensus = (wt[:, None] * w[others, c, :]).sum(axis=0)
            a, m = w[k, c], consensus
            na, nm = np.linalg.norm(a), np.linalg.norm(m)
            out[k, c] = float(a @ m / (na * nm)) if na > 1e-12 and nm > 1e-12 else 0.0
    return out


def rank_average(x, y):
    """E5: per class, average the two candidates' ranks across clients."""
    out = np.zeros_like(x)
    for c in range(x.shape[1]):
        out[:, c] = (rankdata(x[:, c]) + rankdata(y[:, c])) / 2
    return out


def round_updates(clients, cfg, seed, n_classes, rounds):
    """Clipped updates for rounds 1..rounds, replaying the loop exactly as the gate does."""
    set_seed(seed)
    global_state = {k: v.detach().clone()
                    for k, v in TierAHead(cfg.feature_dim, n_classes).state_dict().items()}
    layout = FlatLayout.from_state(global_state)
    sizes = [c.size for c in clients]
    per_round = []
    for r in range(1, rounds + 1):
        states = [local_train(global_state, c.x, c.y, cfg, np.random.default_rng([seed, r, k]))
                  for k, c in enumerate(clients)]
        updates = torch.stack([flatten(state_delta(s, global_state, layout.keys), layout.keys)
                               for s in states])
        clipped, _ = clip_to_median_norm(updates)
        per_round.append(clipped)
        global_state, _ = baselines.fedavg(states, sizes, global_state, r,
                                           n_classes=n_classes)
    return per_round, layout


def candidates_for(per_round, layout, sizes, n_classes):
    """Every candidate, from one seed's updates. R9's are recomputed for comparability."""
    first = per_round[0]
    w1, b1 = rows_and_bias(first, layout, n_classes)
    c1 = normalised_bias(b1)
    e1 = loo_cosine(w1, sizes, weighted=True)

    out = {
        "G0a (design doc)": np.asarray(head_evidence(first, layout), dtype=float),
        "C1 (R9 best clean)": c1,
        "C3 (R9 best raw)": None,     # filled below, kept for the like-for-like comparison
        "E1 leave-one-out": e1,
        "E2 LOO unweighted": loo_cosine(w1, sizes, weighted=False),
        "E5 rank(C1, E1)": rank_average(c1, e1),
    }

    # C3: cosine with the size-weighted consensus INCLUDING self - R9's version.
    share = sizes / sizes.sum()
    consensus = (share[:, None, None] * w1).sum(axis=0)
    c3 = np.zeros((len(w1), n_classes))
    for k in range(len(w1)):
        for c in range(n_classes):
            a, m = w1[k, c], consensus[c]
            na, nm = np.linalg.norm(a), np.linalg.norm(m)
            c3[k, c] = float(a @ m / (na * nm)) if na > 1e-12 and nm > 1e-12 else 0.0
    out["C3 (R9 best raw)"] = c3

    # Multi-round versions.
    c1_rounds, e1_rounds = [], []
    for upd in per_round:
        w, b = rows_and_bias(upd, layout, n_classes)
        c1_rounds.append(normalised_bias(b))
        e1_rounds.append(loo_cosine(w, sizes, weighted=True))
    out["E3 C1 over 3 rounds"] = np.mean(c1_rounds, axis=0)
    out["E4 E1 over 3 rounds"] = np.mean(e1_rounds, axis=0)
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

    per_seed = {}
    for seed in SEEDS:
        per_round, layout = round_updates(clients, cfg, seed, n_classes, ROUNDS)
        per_seed[seed] = candidates_for(per_round, layout, sizes, n_classes)
        print(f"  seed {seed} done", flush=True)

    labels = list(per_seed[SEEDS[0]])
    results = {}
    for label in labels:
        stats, size_rhos, per_class = [], [], []
        for seed in SEEDS:
            ev = np.asarray(per_seed[seed][label], dtype=float)
            rhos = np.array([spearman(ev[:, c], counts[:, c]) for c in range(n_classes)])
            stats.append(float(rhos.mean()))
            size_rhos.append(float(np.mean([spearman(ev[:, c], sizes)
                                            for c in range(n_classes)])))
            per_class.append(rhos)
        results[label] = {
            "mean": float(np.mean(stats)), "sd": float(np.std(stats, ddof=1)),
            "per_seed": stats, "vs_size_mean": float(np.mean(size_rhos)),
            "per_class_mean": {names[c]: float(np.mean([p[c] for p in per_class]))
                               for c in range(n_classes)},
            "passed": bool(np.mean(stats) >= BAR),
        }

    print("\n" + "=" * 78)
    print(f"D15 - evidence signal, second pass | {len(SEEDS)} seeds | bar {BAR}")
    print("=" * 78)
    print(f"\n  {'candidate':<24}{'mean':>9}{'sd':>8}{'vs size':>10}   verdict")
    for label, r in sorted(results.items(), key=lambda kv: -kv[1]["mean"]):
        print(f"  {label:<24}{r['mean']:>+9.3f}{r['sd']:>8.3f}{r['vs_size_mean']:>+10.3f}   "
              f"{'CLEARS THE BAR' if r['passed'] else 'under the bar'}")

    best = max(results, key=lambda k: results[k]["mean"])
    record = {
        "entry": "D15", "fixed_in": "docs/DEVIATIONS.md D15, before any candidate was computed",
        "bar": BAR, "seeds": list(SEEDS), "rounds_for_multiround": ROUNDS,
        "measured_like": "gate G0a, on S1, clipped updates, mean Spearman over 8 classes",
        "candidates": results, "best": best,
        "any_passed": any(r["passed"] for r in results.values()),
        "note": "D11/D15 rule: clearing the bar shows a faithful signal exists, nothing more. "
                "Use in EARN needs its own pre-registration including an A1 spoofability check.",
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evidence_signal_v2.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\n  best: {best} at {results[best]['mean']:+.3f} "
          f"(+-{results[best]['sd']:.3f}) against a bar of {BAR}")
    print(f"  {'A candidate clears it.' if record['any_passed'] else 'None clears it.'}")
    print(f"\nwrote {(OUT / 'evidence_signal_v2.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
