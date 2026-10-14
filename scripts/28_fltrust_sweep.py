#!/usr/bin/env python3
"""
FLTrust root-dataset sweep (docs/DEVIATIONS.md D18).

FLTrust (NDSS 2021) gives the server a small clean root dataset, trains a server update on
it, and scores each client update by how closely it agrees in direction - handling 40-60%
malicious clients. Our boundary argument says this protects a class only when the root set
covers it, which for a rare disease is the assumption the problem denies.

This sweeps exactly that: common classes contribute 10 root examples each, the rare classes
contribute r each for r in {0, 1, 2, 5, 10, 20}, under no attack and under A1. Root rows are
removed from the clients, so the server's data is its own.

A study, not a gate. Both outcomes - including the one that would sink the design note - were
fixed in D18 before this ran.

    python scripts/28_fltrust_sweep.py                 # ~1.5-2 h on a laptop CPU
    python scripts/28_fltrust_sweep.py --rounds 3      # shape check

Writes docs/results/fltrust_sweep.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.attacks.attacks import DEFAULT_ATTACKS
from src.data.features import load_features
from src.federated import baselines
from src.federated.baselines import _info
from src.federated.interface import INTERFACE_VERSION, apply_delta, flatten, state_delta, unflatten
from src.federated.primitives import FlatLayout
from src.federated.tier_a import ClientData, TierAConfig, local_train, make_clients, run_federated

OUT = ROOT / "docs" / "results"
SEEDS = (42, 43, 44)
RARE_IDS = [5, 6]
N_CENTRES, N_CLASSES = 6, 8
COMMON_PER_CLASS = 10                 # fixed, D18
RARE_SWEEP = (0, 1, 2, 5, 10, 20, 50)   # swept, D18 as amended
WORKS_MARGIN = 0.10                     # D18 amendment: 'works on a class'
ATTACKS = ("none", "A1")


def build_root(features, labels, centers, rare_per_class, seed=0):
    """Carve the server's root set out of the training pool and take it away from the clients."""
    rng = np.random.default_rng(seed)
    chosen = []
    for c in range(N_CLASSES):
        want = rare_per_class if c in RARE_IDS else COMMON_PER_CLASS
        if want == 0:
            continue
        pool = np.flatnonzero(labels == c)
        chosen.append(rng.choice(pool, size=min(want, len(pool)), replace=False))
    idx = np.concatenate(chosen) if chosen else np.array([], dtype=int)
    mask = np.zeros(len(labels), dtype=bool)
    mask[idx] = True
    client_centers = centers.copy()
    client_centers[mask] = -1          # make_clients only collects centres 0..K-1
    return (torch.from_numpy(np.ascontiguousarray(features[mask])),
            torch.from_numpy(labels[mask]), client_centers, int(mask.sum()))


def fltrust_factory(root_x, root_y, cfg, seed):
    """FLTrust as an aggregate() rule: server update, ReLU cosine trust, norm rescaling."""
    state = {"round": 0}

    def rule(client_states, client_sizes, global_state, round_num, **kwargs):
        layout = FlatLayout.from_state(global_state)
        keys = layout.keys
        updates = torch.stack([flatten(state_delta(s, global_state, keys), keys)
                               for s in client_states])

        # The server trains on its own root set, exactly as a client would.
        rng = np.random.default_rng([seed, round_num, 9999])
        server_state = local_train(global_state, root_x, root_y, cfg, rng)
        g0 = flatten(state_delta(server_state, global_state, keys), keys)
        g0_norm = g0.norm().clamp_min(1e-12)

        cos = torch.stack([(u @ g0) / (u.norm().clamp_min(1e-12) * g0_norm) for u in updates])
        trust = torch.relu(cos)                                   # FLTrust's ReLU clipping
        scale = g0_norm / updates.norm(dim=1).clamp_min(1e-12)    # rescale onto the server's sphere
        weights = trust / trust.sum().clamp_min(1e-12)

        agg = (weights[:, None] * scale[:, None] * updates).sum(dim=0)
        new_state = apply_delta(global_state, unflatten(agg, global_state, keys))

        w = weights.numpy().astype(np.float64)
        head_w = np.repeat(w[:, None], N_CLASSES, axis=1)
        # FLTrust rescales each update onto the server's sphere, which can amplify a small
        # one - so it is not a clipping factor and is not reported as clip_scale, which the
        # interface requires to be in (0, 1]. It is carried separately.
        info = _info("fltrust", round_num, w, head_w, np.ones(len(updates)),
                     trust=np.repeat(trust.numpy().astype(np.float64)[:, None],
                                     N_CLASSES, axis=1))
        info["fltrust_rescale"] = scale.numpy().astype(np.float64)
        info["interface_version"] = INTERFACE_VERSION
        return new_state, info

    return lambda: rule


def per_class_recall(confusion):
    cm = np.asarray(confusion, dtype=float)
    return [float(cm[c, c] / cm[c].sum()) if cm[c].sum() else 0.0 for c in range(N_CLASSES)]


def rare_recall(confusion):
    rec = per_class_recall(confusion)
    return float(np.mean([rec[c] for c in RARE_IDS]))


def main() -> int:
    ap = argparse.ArgumentParser(description="D18: FLTrust root-dataset sweep.")
    ap.add_argument("--rounds", type=int, default=40)
    args = ap.parse_args()

    train, test = load_features("train"), load_features("test")
    test_x, test_y = torch.from_numpy(test.features), test.labels
    cfg = TierAConfig(rounds=args.rounds, n_classes=N_CLASSES)

    rows, t0 = [], time.time()
    print(f"D18 | FLTrust sweep | rare root examples {RARE_SWEEP} | {args.rounds} rounds")
    for r in RARE_SWEEP:
        root_x, root_y, client_centers, root_n = build_root(
            train.features, train.labels, train.centers, r)
        clients = make_clients(train.features, train.labels, client_centers, N_CENTRES)
        for attack in ATTACKS:
            spec = DEFAULT_ATTACKS[attack]
            for seed in SEEDS:
                for rule_name in ("fltrust", "fedavg"):
                    factory = (fltrust_factory(root_x, root_y, cfg, seed)
                               if rule_name == "fltrust" else (lambda: baselines.fedavg))
                    res = run_federated(factory, clients, test_x, test_y, cfg, seed,
                                        RARE_IDS, client_hooks=spec.hooks(RARE_IDS))
                    final = res["final"]
                    info = res["round_infos"][-1]
                    w = np.asarray(info["body_weights"], dtype=float)
                    rows.append({
                        "rare_root_per_class": r, "root_size": root_n, "attack": attack,
                        "rule": rule_name, "seed": seed,
                        "balanced_accuracy": final["balanced_accuracy"],
                        "rare_macro_f1": final["rare_macro_f1"],
                        "rare_recall": rare_recall(final["confusion"]),
                        "per_class_recall": per_class_recall(final["confusion"]),
                        "attacker": spec.attacker,
                        "attacker_weight": (float(w[spec.attacker])
                                            if spec.attacker is not None else None),
                    })
            print(f"  r={r:<3} {attack:<5} done ({time.time() - t0:.0f}s)", flush=True)

    def avg(r, attack, rule, key):
        vals = [x[key] for x in rows if x["rare_root_per_class"] == r and x["attack"] == attack
                and x["rule"] == rule and x[key] is not None]
        return float(np.mean(vals)) if vals else None

    # D18's two pre-registered readings.
    base_r0 = avg(0, "A1", "fedavg", "rare_recall")
    fl_r0 = avg(0, "A1", "fltrust", "rare_recall")
    fl_r20 = avg(20, "A1", "fltrust", "rare_recall")
    base_r20 = avg(20, "A1", "fedavg", "rare_recall")
    supports = bool(abs(fl_r0 - base_r0) <= 0.05 and (fl_r20 - base_r20) >= 0.10)
    refutes = bool((fl_r0 - base_r0) >= 0.10)

    def class_recall(r, attack, rule, c):
        vals = [x["per_class_recall"][c] for x in rows
                if x["rare_root_per_class"] == r and x["attack"] == attack
                and x["rule"] == rule]
        return float(np.mean(vals)) if vals else None

    # D18 amendment: "works on class c" = recall under A1 within WORKS_MARGIN of the same
    # arm's no-attack recall. FLTrust has one trust per client, not one per class, so
    # protection downstream is the only honest per-class reading.
    protection = {}
    for rr in RARE_SWEEP:
        per_class = {}
        for c in range(N_CLASSES):
            clean = class_recall(rr, "none", "fltrust", c)
            attacked = class_recall(rr, "A1", "fltrust", c)
            per_class[c] = {"no_attack": clean, "under_A1": attacked,
                            "drop": clean - attacked,
                            "works": bool(attacked >= clean - WORKS_MARGIN),
                            "rare": c in RARE_IDS}
        protection[rr] = per_class

    threshold = next((rr for rr in RARE_SWEEP
                      if all(protection[rr][c]["works"] for c in RARE_IDS)), None)

    record = {
        "entry": "D18",
        "fixed_in": "docs/DEVIATIONS.md D18 and its amendment, before the first run",
        "works_margin": WORKS_MARGIN,
        "per_class_protection": protection,
        "headline_threshold_rare_examples_per_class": threshold,
        "headline_note": ("smallest number of rare-class examples the server needs before "
                          "FLTrust protects both rare classes; null means not reached by "
                          f"r = {max(RARE_SWEEP)}"),
        "rounds": args.rounds, "seeds": list(SEEDS),
        "common_root_per_class": COMMON_PER_CLASS, "rare_sweep": list(RARE_SWEEP),
        "reading": {
            "fltrust_rare_recall_at_r0_under_A1": fl_r0,
            "fedavg_rare_recall_under_A1": base_r0,
            "fltrust_rare_recall_at_r20_under_A1": fl_r20,
            "supports_boundary_claim": supports,
            "refutes_boundary_claim": refutes,
        },
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fltrust_sweep.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    print("\n" + "=" * 86)
    print("D18 - FLTrust under A1, as the server's rare-class examples increase")
    print("=" * 86)
    print(f"\n  {'rare root':>10}{'root size':>11}{'FLTrust rare recall':>21}"
          f"{'FedAvg rare recall':>20}{'FLTrust bal acc':>17}")
    for r in RARE_SWEEP:
        n = next(x["root_size"] for x in rows if x["rare_root_per_class"] == r)
        print(f"  {r:>10}{n:>11}{avg(r, 'A1', 'fltrust', 'rare_recall'):>21.4f}"
              f"{avg(r, 'A1', 'fedavg', 'rare_recall'):>20.4f}"
              f"{avg(r, 'A1', 'fltrust', 'balanced_accuracy'):>17.4f}")
    print("\n  no attack, for reference:")
    for r in RARE_SWEEP:
        print(f"  {r:>10}{'':>11}{avg(r, 'none', 'fltrust', 'rare_recall'):>21.4f}"
              f"{avg(r, 'none', 'fedavg', 'rare_recall'):>20.4f}"
              f"{avg(r, 'none', 'fltrust', 'balanced_accuracy'):>17.4f}")
    print(f"\n  per-class protection under A1 (ok = recall within {WORKS_MARGIN} of no-attack)")
    print("    r     " + "  ".join(f"c{c}" + ("*" if c in RARE_IDS else " ")
                                   for c in range(N_CLASSES)))
    for rr in RARE_SWEEP:
        marks = "  ".join(("ok  " if protection[rr][c]["works"] else "FAIL")
                          for c in range(N_CLASSES))
        print(f"    {rr:<5} {marks}")
    print("    (* = rare class)")
    print(f"  HEADLINE - rare examples the server needs before both rare classes are "
          f"protected: {threshold if threshold is not None else 'not reached by r=' + str(max(RARE_SWEEP))}")
    print(f"  supports the boundary claim: {supports}")
    print(f"  refutes the boundary claim:  {refutes}")
    print(f"\nwrote {(OUT / 'fltrust_sweep.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
