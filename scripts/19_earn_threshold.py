#!/usr/bin/env python3
"""
Amendment M2 on Fed-ISIC2019: absolute holder threshold vs scale-relative (docs/DEVIATIONS.md D10).

Two arms, identical except for who counts as a holder of a class:

    absolute   EARNConfig(tau=19.5)    a centre must hold 20 images of the class
    relative   EARNConfig(tau="mean")  a centre's evidence must beat this round's mean

across splits S1 and S2 and attacks none and A1, three seeds. A1 is in deliberately: a lower
bar for holding could make a claiming attacker easier to admit, which is the risk this
amendment has to rule out rather than assume away.

The three criteria and what each outcome means were fixed in D10 before this first ran.

Feature set: the published grid uses ft4; those files are not on this machine, so both arms
run on the frozen features by default. Like for like, and never to be quoted beside a grid
number. Pass --variant ft4 to re-run unchanged if they become available.

    python scripts/19_earn_threshold.py                 # ~2 h on a laptop CPU
    python scripts/19_earn_threshold.py --rounds 10     # a quick shape check

Writes docs/results/earn_threshold_amendment.csv and docs/results/amendment_m2_fedisic.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src.attacks.attacks import DEFAULT_ATTACKS
from src.experiments.grid import (N_CLASSES, ORACLE_TAU, RARE_IDS, SPECIALIST,
                                  load_context, reported_counts_fn)
from src.federated.earn import EARN, EARNConfig
from src.federated.tier_a import run_federated

OUT = ROOT / "docs" / "results"
ARMS = {"absolute": ORACLE_TAU, "relative": "mean"}
MARGIN = 0.05           # D10, all three criteria use the same margin
SEEDS = (42, 43, 44)


def holders_per_class(counts: np.ndarray, tau) -> list[int]:
    """How many centres clear the threshold for each class - the mechanism-level explanation."""
    if tau == "mean":
        bar = counts.mean(axis=0)
        return [int((counts[:, c] > bar[c]).sum()) for c in range(counts.shape[1])]
    return [int((counts[:, c] > tau).sum()) for c in range(counts.shape[1])]


def main() -> int:
    parser = argparse.ArgumentParser(description="M2 on Fed-ISIC2019 (D10).")
    parser.add_argument("--variant", default="frozen", choices=["frozen", "ft4"])
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--attacks", nargs="+", default=["none", "A1"])
    parser.add_argument("--splits", nargs="+", default=["s1", "s2"])
    args = parser.parse_args()

    ctx = load_context(rounds=args.rounds, variant=args.variant)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, finals = [], []

    print(f"M2 on Fed-ISIC2019 | {args.variant} features | {args.rounds} rounds | "
          f"seeds {SEEDS}")
    for split in args.splits:
        counts = ctx.counts[split]
        sizes = np.array([c.size for c in ctx.clients[split]], dtype=float)
        size_share = sizes / sizes.sum()
        for arm, tau in ARMS.items():
            print(f"\n{split.upper()} / {arm}: holders per class "
                  f"{holders_per_class(counts, tau)} (rare: "
                  f"{[holders_per_class(counts, tau)[c] for c in RARE_IDS]})")
            for attack in args.attacks:
                spec = DEFAULT_ATTACKS[attack]
                oracle = reported_counts_fn(counts, attack)
                for seed in SEEDS:
                    t0 = time.time()
                    res = run_federated(
                        lambda: EARN(EARNConfig(tau=tau), evidence_fn=oracle,
                                     name=f"earn_{arm}"),
                        ctx.clients[split], ctx.test_x, ctx.test_y, ctx.cfg, seed,
                        RARE_IDS, client_hooks=spec.hooks(RARE_IDS))
                    final = res["final"]
                    head_w = res["round_infos"][-1]["head_row_weights"]
                    rare_w = head_w[:, RARE_IDS].mean(axis=1)

                    record = {
                        "split": split, "arm": arm, "attack": attack, "seed": seed,
                        "balanced_accuracy": final["balanced_accuracy"],
                        "rare_macro_f1": final["rare_macro_f1"],
                        "accuracy": final["accuracy"],
                        "specialist_rare_weight": float(rare_w[SPECIALIST]),
                        "specialist_size_share": float(size_share[SPECIALIST]),
                        "attacker_rare_weight": (float(rare_w[spec.attacker])
                                                 if spec.attacker is not None else None),
                        "attacker": spec.attacker,
                        "seconds": round(time.time() - t0, 1),
                    }
                    finals.append(record)
                    rows.append(record)
                    print(f"    {attack:<5} seed {seed}: rare F1 "
                          f"{final['rare_macro_f1']:.4f} | balanced acc "
                          f"{final['balanced_accuracy']:.4f} | specialist weight "
                          f"{rare_w[SPECIALIST]:.3f}"
                          + (f" | attacker weight {rare_w[spec.attacker]:.3f}"
                             if spec.attacker is not None else "")
                          + f" | {record['seconds']:.0f}s", flush=True)

    csv_path = OUT / "earn_threshold_amendment.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- the three criteria from D10 ---------------------------------------------------
    def avg(split, arm, attack, key):
        vals = [r[key] for r in finals
                if r["split"] == split and r["arm"] == arm and r["attack"] == attack
                and r[key] is not None]
        return float(np.mean(vals)) if vals else None

    checks, details = [], {}
    for split in args.splits:
        a = avg(split, "absolute", "none", "rare_macro_f1")
        r = avg(split, "relative", "none", "rare_macro_f1")
        ok = abs(r - a) <= MARGIN
        checks.append(ok)
        details[f"{split}: no harm without attack"] = {
            "absolute": a, "relative": r, "difference": r - a, "met": ok}

    if "A1" in args.attacks:
        for split in args.splits:
            a = avg(split, "absolute", "A1", "attacker_rare_weight")
            r = avg(split, "relative", "A1", "attacker_rare_weight")
            ok = (r - a) <= MARGIN
            checks.append(ok)
            details[f"{split}: attacker gains no more weight under A1"] = {
                "absolute": a, "relative": r, "difference": r - a, "met": ok}

    if "s2" in args.splits:
        a = avg("s2", "absolute", "none", "specialist_rare_weight")
        r = avg("s2", "relative", "none", "specialist_rare_weight")
        ok = r >= a - MARGIN
        checks.append(ok)
        details["s2: the specialist is still served"] = {
            "absolute": a, "relative": r, "difference": r - a, "met": ok}

    passed = all(checks)
    record = {
        "amendment": "M2 on Fed-ISIC2019",
        "fixed_in": "docs/DEVIATIONS.md D10, before the first run",
        "features": args.variant,
        "note": ("frozen features, not the ft4 set the published grid uses - both arms are "
                 "the same, so the comparison holds, but these numbers do not restate grid "
                 "results"),
        "rounds": args.rounds, "seeds": list(SEEDS), "margin": MARGIN,
        "attacks": args.attacks, "splits": args.splits,
        "holders_per_class": {
            f"{split}/{arm}": holders_per_class(ctx.counts[split], tau)
            for split in args.splits for arm, tau in ARMS.items()},
        "criteria": details,
        "passed": bool(passed),
        "decision": ("adopt the relative threshold as EARN's default (D10)" if passed else
                     "keep the absolute threshold; R6/R7 stand as a dataset-specific finding"),
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": finals,
    }
    (OUT / "amendment_m2_fedisic.json").write_text(json.dumps(record, indent=2),
                                                   encoding="utf-8")

    print("\n" + "=" * 74)
    for label, c in details.items():
        print(f"  {'met    ' if c['met'] else 'not met'}  {label}: "
              f"absolute {c['absolute']:.4f} -> relative {c['relative']:.4f}")
    print(f"M2 ON FED-ISIC2019 -> {'ALL CRITERIA MET' if passed else 'NOT MET'}")
    print(f"  {record['decision']}")
    print("=" * 74)
    print(f"wrote {csv_path.relative_to(ROOT)} and amendment_m2_fedisic.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
