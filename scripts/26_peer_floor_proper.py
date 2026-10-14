#!/usr/bin/env python3
"""
The peer-confidence floor, tested on its own terms (docs/DEVIATIONS.md D16).

R10 found, while testing something else, that flooring EARN's peer confidence cuts a
claiming attacker's captured rare-row weight at coverage 1 from 0.263 to 0.037 with the
honest specialist untouched - and refused to call it a result, because the conditions it
cleared only asked whether the floor did no harm. This tests it.

60 runs: 2 arms (floor off, floor 0.5) x 2 splits x 5 attacks (none, A1, A2, A3, A2s) x 3
seeds, 100 rounds, oracle evidence, absolute holder threshold throughout. Confusion
matrices are stored, so rare-class recall is exact - after R13 this project does not judge
attack damage on F1 alone.

Plus two runs outside the battery at 200 rounds (S2, no attack, floor on and off) to check
an honest sole holder is not slowly penalised on a longer horizon. Reported separately.

    python scripts/26_peer_floor_proper.py          # ~2 h on a laptop CPU
    python scripts/26_peer_floor_proper.py --rounds 3 --skip-long   # shape check

Writes docs/results/peer_floor_proper.json.
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

from src.attacks.attacks import DEFAULT_ATTACKS
from src.experiments.grid import ORACLE_TAU, RARE_IDS, SPECIALIST, load_context, reported_counts_fn
from src.federated.earn import EARN, EARNConfig
from src.federated.tier_a import run_federated

OUT = ROOT / "docs" / "results"
P_FLOOR = 0.5
SEEDS = (42, 43, 44)
ATTACKS = ("none", "A1", "A2", "A3", "A2s")
ARMS = {"floor off": 0.0, "floor on": P_FLOOR}
MARGIN = 0.05
PREVIOUSLY_SEEN = {("s2", "A1")}        # the R10 cell; cannot carry the verdict on its own


def rare_recall(confusion, rare_ids):
    cm = np.asarray(confusion, dtype=float)
    out = []
    for c in rare_ids:
        support = cm[c].sum()
        out.append(cm[c, c] / support if support else 0.0)
    return float(np.mean(out))


def one_run(ctx, split, arm_floor, attack, seed):
    spec = DEFAULT_ATTACKS[attack]
    oracle = reported_counts_fn(ctx.counts[split], attack)
    res = run_federated(
        lambda: EARN(EARNConfig(tau=ORACLE_TAU, p_floor=arm_floor), evidence_fn=oracle,
                     name="earn"),
        ctx.clients[split], ctx.test_x, ctx.test_y, ctx.cfg, seed, RARE_IDS,
        client_hooks=spec.hooks(RARE_IDS))
    final = res["final"]
    info = res["round_infos"][-1]
    rare_w = info["head_row_weights"][:, RARE_IDS].mean(axis=1)
    sizes = np.array([c.size for c in ctx.clients[split]], dtype=float)
    return {
        "balanced_accuracy": final["balanced_accuracy"],
        "rare_macro_f1": final["rare_macro_f1"],
        "rare_recall": rare_recall(final["confusion"], RARE_IDS),
        "specialist_rare_weight": float(rare_w[SPECIALIST]),
        "specialist_size_share": float(sizes[SPECIALIST] / sizes.sum()),
        "attacker": spec.attacker,
        "attacker_rare_weight": (float(rare_w[spec.attacker])
                                 if spec.attacker is not None else None),
        "confusion": final["confusion"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="D16: the peer-confidence floor on its own terms.")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--skip-long", action="store_true")
    args = ap.parse_args()

    ctx = load_context(rounds=args.rounds, variant="frozen")
    rows, t0 = [], time.time()
    print(f"D16 | floor {P_FLOOR} | {args.rounds} rounds | seeds {SEEDS}")
    for split in ("s1", "s2"):
        for arm, floor in ARMS.items():
            for attack in ATTACKS:
                for seed in SEEDS:
                    r = one_run(ctx, split, floor, attack, seed)
                    rows.append({"split": split, "arm": arm, "p_floor": floor,
                                 "attack": attack, "seed": seed,
                                 "previously_seen": (split, attack) in PREVIOUSLY_SEEN, **r})
                print(f"  {split} / {arm:<9} / {attack:<4} done "
                      f"({time.time() - t0:.0f}s)", flush=True)

    def avg(split, arm, attack, key):
        vals = [r[key] for r in rows if r["split"] == split and r["arm"] == arm
                and r["attack"] == attack and r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    checks, detail = {}, {}

    # 1. reduces capture on S2 wherever there is something to reduce
    reductions, failures = {}, []
    for attack in ATTACKS:
        if attack == "none":
            continue
        off = avg("s2", "floor off", attack, "attacker_rare_weight")
        on = avg("s2", "floor on", attack, "attacker_rare_weight")
        if off is None:
            continue
        reductions[attack] = {"floor_off": off, "floor_on": on, "reduction": off - on,
                              "applicable": off > 0.10,
                              "previously_seen": ("s2", attack) in PREVIOUSLY_SEEN}
        if off > 0.10 and (off - on) < MARGIN:
            failures.append(attack)
    applicable = [a for a, v in reductions.items() if v["applicable"]]
    fresh = [a for a in applicable if not reductions[a]["previously_seen"]]
    checks["1. reduces capture on S2 where there is something to reduce"] = {
        "applicable_attacks": applicable, "of_which_not_previously_seen": fresh,
        "failed_on": failures, "met": bool(applicable and not failures)}
    detail["s2_capture"] = reductions

    # 2. never helps an attacker, either split
    worst = None
    for split in ("s1", "s2"):
        for attack in ATTACKS:
            off = avg(split, "floor off", attack, "attacker_rare_weight")
            on = avg(split, "floor on", attack, "attacker_rare_weight")
            if off is None:
                continue
            delta = on - off
            if worst is None or delta > worst[0]:
                worst = (delta, split, attack)
    checks["2. never increases an attacker's capture by more than 0.01"] = {
        "largest_increase": worst[0], "where": f"{worst[1]}/{worst[2]}",
        "met": bool(worst[0] <= 0.01)}

    # 3. costs the honest case nothing
    honest = {}
    ok3 = True
    for split in ("s1", "s2"):
        f_off = avg(split, "floor off", "none", "rare_macro_f1")
        f_on = avg(split, "floor on", "none", "rare_macro_f1")
        honest[f"{split} rare_macro_f1"] = {"floor_off": f_off, "floor_on": f_on,
                                            "difference": f_on - f_off}
        ok3 &= abs(f_on - f_off) <= MARGIN
    s_off = avg("s2", "floor off", "none", "specialist_rare_weight")
    s_on = avg("s2", "floor on", "none", "specialist_rare_weight")
    honest["s2 honest specialist weight"] = {"floor_off": s_off, "floor_on": s_on,
                                             "difference": s_on - s_off}
    ok3 &= (s_on >= s_off - MARGIN)
    checks["3. costs the honest case nothing"] = {**honest, "met": bool(ok3)}

    # 4. inert on S1
    s1_deltas = {}
    ok4 = True
    for attack in ATTACKS:
        off = avg("s1", "floor off", attack, "attacker_rare_weight")
        on = avg("s1", "floor on", attack, "attacker_rare_weight")
        if off is None:
            continue
        s1_deltas[attack] = {"floor_off": off, "floor_on": on, "difference": on - off}
        ok4 &= abs(on - off) <= MARGIN
    checks["4. inert on S1, where peers are plentiful"] = {**s1_deltas, "met": bool(ok4)}

    passed = all(c["met"] for c in checks.values())

    # The longer-horizon check, outside the verdict.
    long_horizon = None
    if not args.skip_long:
        print("\n  long-horizon check: S2, no attack, 200 rounds, seed 42", flush=True)
        long_ctx = load_context(rounds=200, variant="frozen")
        long_horizon = {arm: one_run(long_ctx, "s2", floor, "none", 42)
                        for arm, floor in ARMS.items()}
        for arm, r in long_horizon.items():
            r.pop("confusion", None)
            print(f"    {arm:<9} specialist weight {r['specialist_rare_weight']:.3f} | "
                  f"rare F1 {r['rare_macro_f1']:.4f} | rare recall {r['rare_recall']:.4f}")

    record = {
        "entry": "D16", "fixed_in": "docs/DEVIATIONS.md D16, before the first run",
        "p_floor": P_FLOOR, "rounds": args.rounds, "seeds": list(SEEDS),
        "margin": MARGIN, "attacks": list(ATTACKS),
        "previously_seen_cells": [list(c) for c in PREVIOUSLY_SEEN],
        "checks": checks, "detail": detail, "passed": bool(passed),
        "long_horizon_200_rounds": long_horizon,
        "decision": ("the floor is EARN's first validated positive result and becomes the "
                     "proposed default for coverage-1 regimes" if passed else
                     "the floor does not clear its conditions; see which one failed"),
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "peer_floor_proper.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    print("\n" + "=" * 76)
    for label, c in checks.items():
        print(f"  {'met    ' if c['met'] else 'NOT MET'}  {label}")
    print(f"\nD16 -> {'ALL CONDITIONS MET' if passed else 'NOT MET'}")
    print(f"  {record['decision']}")
    print("=" * 76)
    print("\n  S2 attacker capture, floor off -> floor on:")
    for attack, v in detail["s2_capture"].items():
        tag = "  (seen in R10)" if v["previously_seen"] else ""
        print(f"    {attack:<4} {v['floor_off']:.3f} -> {v['floor_on']:.3f}  "
              f"(reduction {v['reduction']:+.3f}){tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
