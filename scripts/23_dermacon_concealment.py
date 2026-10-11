#!/usr/bin/env python3
"""
Replicate the F1-concealment split on DermaCon-IN (docs/DEVIATIONS.md D14).

D13 found, on Fed-ISIC2019, that rare-class F1 moves opposite to the damage metric in 26.9%
of cases under selection rules (Krum, Multi-Krum) and in 0 of 320 under every averaging
rule. This runs the same battery on DermaCon-IN's D1 split to see whether that holds.

96 runs: 8 methods x 4 attacks x 3 seeds, 40 rounds, oracle evidence, frozen features ->
144 comparable (pair, rare class) cases. Unlike the original grid, every run stores its
full final-round confusion matrix, so recall and precision are exact for every case.

Attack roles are ported, not centre numbers (D14): target is the head class, A1/A3 come
from the weakest rare holder, the sleeper from the largest genuine one.

    python scripts/23_dermacon_concealment.py          # ~1.5-2 h on a laptop CPU
    python scripts/23_dermacon_concealment.py --rounds 2   # shape check

Writes docs/results/dermacon_concealment.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.attacks.attacks import Scaling, Sleeper, SuddenExpert
from src.data.splits import HOLDER_MIN_IMAGES
from src.federated import baselines
from src.federated.earn import EARN, EARNConfig
from src.federated.tier_a import TierAConfig, make_clients, run_federated

OUT = ROOT / "docs" / "results"
FEATURES = ROOT / "data" / "features" / "dermacon_densenet121.npz"
META = ROOT / "data" / "features" / "dermacon_densenet121.json"

RARE_NAMES = ("Keratanisation Disorders", "Neoplasms and tumors")
TARGET_NAME = "Infectious Disorders"          # the head class, standing in for nevus
A1_A3_ATTACKER = 0                            # weakest rare holder (D14)
A2_ATTACKER = 3                               # largest genuine rare holder (D14)
SELECTION_RULES = {"krum", "multi_krum"}
METHODS = ("fedavg", "fedavg_clipped", "trimmed_mean", "coordinate_wise_median",
           "camp_a_reported", "earn", "krum", "multi_krum")
ATTACKS = ("none", "A1", "A2", "A3")
SEEDS = (42, 43, 44)
N_CENTRES = 6
ORACLE_TAU = HOLDER_MIN_IMAGES - 0.5


def load():
    d = np.load(FEATURES, allow_pickle=True)
    keep = d["labels"] >= 0
    classes = json.loads(META.read_text(encoding="utf-8"))["classes"]
    return {k: d[k][keep] for k in ("features", "labels", "d1", "is_train")}, classes


def hooks_for(attack, rare_ids, target):
    """The same three attacks, pointed at this dataset's roles."""
    if attack == "none":
        return {}, None
    rare = tuple(rare_ids)
    if attack == "A1":
        return {A1_A3_ATTACKER: SuddenExpert(rare_ids=rare, target=target)}, A1_A3_ATTACKER
    if attack == "A2":
        return {A2_ATTACKER: Sleeper(rare_ids=rare, target=target)}, A2_ATTACKER
    if attack == "A3":
        return {A1_A3_ATTACKER: Scaling(rare_ids=rare, target=target)}, A1_A3_ATTACKER
    raise ValueError(attack)


def aggregator(method, counts, claims_holder):
    """Oracle evidence, with a claiming attacker reporting the top honest count (as the grid does)."""
    honest_top = counts.max(axis=0)

    def oracle(updates, layout, round_num, _counts=counts, _k=claims_holder):
        ev = counts.astype(np.float64).copy()
        if _k is not None:
            for c in range(ev.shape[1]):
                ev[_k, c] = max(ev[_k, c], honest_top[c])
        return ev

    if method == "camp_a_reported":
        return lambda: partial(baselines.camp_a, evidence_fn=oracle,
                               rule_name="camp_a_reported")
    if method == "earn":
        return lambda: EARN(EARNConfig(tau=ORACLE_TAU), evidence_fn=oracle, name="earn")
    return lambda: baselines.RULES[method]


def main() -> int:
    parser = argparse.ArgumentParser(description="D14 replication on DermaCon-IN.")
    parser.add_argument("--rounds", type=int, default=40)
    args = parser.parse_args()

    data, classes = load()
    rare_ids = [classes.index(n) for n in RARE_NAMES]
    target = classes.index(TARGET_NAME)
    train = data["is_train"]
    y, c = data["labels"][train], data["d1"][train]
    clients = make_clients(data["features"][train], y, c, N_CENTRES)
    counts = np.stack([np.bincount(y[c == k], minlength=len(classes))
                       for k in range(N_CENTRES)])
    test = ~train
    test_x = torch.from_numpy(np.ascontiguousarray(data["features"][test]))
    test_y = data["labels"][test]
    cfg = TierAConfig(rounds=args.rounds, n_classes=len(classes))

    print(f"D14 replication | DermaCon-IN D1 | {len(METHODS)} methods x {len(ATTACKS)} "
          f"attacks x {len(SEEDS)} seeds = {len(METHODS) * len(ATTACKS) * len(SEEDS)} runs")
    print(f"  rare {RARE_NAMES} -> {rare_ids} | target '{TARGET_NAME}' -> {target}")
    print(f"  A1/A3 attacker centre {A1_A3_ATTACKER} "
          f"(rare held: {counts[A1_A3_ATTACKER, rare_ids].tolist()}), "
          f"A2 centre {A2_ATTACKER} (rare held: {counts[A2_ATTACKER, rare_ids].tolist()})")

    runs, t_start = [], time.time()
    for method in METHODS:
        for attack in ATTACKS:
            hooks, attacker = hooks_for(attack, rare_ids, target)
            claims = attacker if attack in ("A1", "A2", "A3") else None
            for seed in SEEDS:
                t0 = time.time()
                res = run_federated(aggregator(method, counts, claims), clients, test_x,
                                    test_y, cfg, seed, rare_ids, client_hooks=hooks)
                final = res["final"]
                runs.append({
                    "method": method, "family": ("selection" if method in SELECTION_RULES
                                                 else "averaging"),
                    "attack": attack, "seed": seed, "attacker": attacker,
                    "balanced_accuracy": final["balanced_accuracy"],
                    "accuracy": final["accuracy"],
                    "rare_macro_f1": final["rare_macro_f1"],
                    "per_class_f1": final["per_class_f1"],
                    "confusion": final["confusion"],      # D14: stored this time
                    "seconds": round(time.time() - t0, 1),
                })
            print(f"  {method:<24} {attack:<5} done "
                  f"({time.time() - t_start:.0f}s elapsed)", flush=True)

    record = {
        "entry": "D14", "fixed_in": "docs/DEVIATIONS.md D14, before the first run",
        "dataset": "DermaCon-IN", "split": "D1 (age bands, constructed)",
        "rounds": args.rounds, "seeds": list(SEEDS),
        "classes": classes, "rare_ids": rare_ids, "target": target,
        "attackers": {"A1": A1_A3_ATTACKER, "A2": A2_ATTACKER, "A3": A1_A3_ATTACKER},
        "selection_rules": sorted(SELECTION_RULES),
        "note": "full final-round confusion matrices are stored, so recall and precision are "
                "exact for every case - the thing the Fed-ISIC2019 grid could not support",
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": runs,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dermacon_concealment.json").write_text(json.dumps(record, indent=2),
                                                   encoding="utf-8")
    print(f"\nwrote {(OUT / 'dermacon_concealment.json').relative_to(ROOT)} "
          f"({len(runs)} runs, {time.time() - t_start:.0f}s)")
    print("Now run: python scripts/24_compare_concealment.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
