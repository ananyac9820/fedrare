#!/usr/bin/env python3
"""
Amendment M3: a floor under peer confidence (docs/DEVIATIONS.md D12).

R8 rejected the scale-relative holder threshold because a claiming attacker captured 20.7%
of the rare rows against 4.8% under the absolute rule. The diagnosis was that the relative
bar shrank the holder set, which lowered p = (n-1)/4, which shifted EARN's agreement check
onto the self-history branch - the branch a consistent liar passes. M3 puts a floor of 0.5
under p so the peer check always carries at least half.

Four arms over D10's battery (S1/S2 x none/A1 x seeds 42/43/44, 100 rounds, oracle
evidence, frozen features):

    absolute            tau = 19.5      floor off     <- EARN as it stands, the reference
    relative            tau = "mean"    floor off     <- reproduces R8, as a regression check
    absolute + floor    tau = 19.5      floor 0.5
    relative + floor    tau = "mean"    floor 0.5

The four conditions and what each outcome means were fixed in D12 before this first ran.

    python scripts/21_peer_floor.py        # ~2 h on a laptop CPU

Writes docs/results/peer_floor.csv and docs/results/amendment_m3.json.
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
from src.experiments.grid import ORACLE_TAU, RARE_IDS, SPECIALIST, load_context, reported_counts_fn
from src.federated.earn import EARN, EARNConfig
from src.federated.tier_a import run_federated

OUT = ROOT / "docs" / "results"
P_FLOOR = 0.5           # D12, fixed before the run
MARGIN = 0.05
SEEDS = (42, 43, 44)
ARMS = {
    "absolute": {"tau": ORACLE_TAU, "p_floor": 0.0},
    "relative": {"tau": "mean", "p_floor": 0.0},
    "absolute + floor": {"tau": ORACLE_TAU, "p_floor": P_FLOOR},
    "relative + floor": {"tau": "mean", "p_floor": P_FLOOR},
}
REFERENCE = "absolute"


def main() -> int:
    parser = argparse.ArgumentParser(description="M3: a floor under peer confidence (D12).")
    parser.add_argument("--rounds", type=int, default=100,
                        help="D12 fixes this at 100; lower values are shape checks only, and "
                             "the regression condition will fail because R8 is a 100-round result")
    args = parser.parse_args()
    ctx = load_context(rounds=args.rounds, variant="frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    print(f"M3 - peer-confidence floor {P_FLOOR} | 100 rounds | seeds {SEEDS}")
    for split in ("s1", "s2"):
        counts = ctx.counts[split]
        sizes = np.array([c.size for c in ctx.clients[split]], dtype=float)
        size_share = sizes / sizes.sum()
        for arm, knobs in ARMS.items():
            print(f"\n{split.upper()} / {arm}")
            for attack in ("none", "A1"):
                spec = DEFAULT_ATTACKS[attack]
                oracle = reported_counts_fn(counts, attack)
                for seed in SEEDS:
                    t0 = time.time()
                    res = run_federated(
                        lambda: EARN(EARNConfig(tau=knobs["tau"], p_floor=knobs["p_floor"]),
                                     evidence_fn=oracle, name=arm),
                        ctx.clients[split], ctx.test_x, ctx.test_y, ctx.cfg, seed,
                        RARE_IDS, client_hooks=spec.hooks(RARE_IDS))
                    final = res["final"]
                    info = res["round_infos"][-1]
                    rare_w = info["head_row_weights"][:, RARE_IDS].mean(axis=1)
                    rows.append({
                        "split": split, "arm": arm, "tau": str(knobs["tau"]),
                        "p_floor": knobs["p_floor"], "attack": attack, "seed": seed,
                        "balanced_accuracy": final["balanced_accuracy"],
                        "rare_macro_f1": final["rare_macro_f1"],
                        "accuracy": final["accuracy"],
                        "specialist_rare_weight": float(rare_w[SPECIALIST]),
                        "specialist_size_share": float(size_share[SPECIALIST]),
                        "attacker_rare_weight": (float(rare_w[spec.attacker])
                                                 if spec.attacker is not None else None),
                        "seconds": round(time.time() - t0, 1),
                    })
                    print(f"    {attack:<5} seed {seed}: rare F1 {final['rare_macro_f1']:.4f}"
                          + (f" | attacker weight {rare_w[spec.attacker]:.3f}"
                             if spec.attacker is not None else "")
                          + f" | {rows[-1]['seconds']:.0f}s", flush=True)

    csv_path = OUT / "peer_floor.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def avg(split, arm, attack, key):
        vals = [r[key] for r in rows if r["split"] == split and r["arm"] == arm
                and r["attack"] == attack and r[key] is not None]
        return float(np.mean(vals)) if vals else None

    checks = {}
    # 1 - the floor fixes the spoofability it was written for (S1, A1).
    ref = avg("s1", REFERENCE, "A1", "attacker_rare_weight")
    got = avg("s1", "relative + floor", "A1", "attacker_rare_weight")
    checks["1. relative + floor gives the attacker no more weight than EARN does today"] = {
        "reference (absolute)": ref, "relative + floor": got,
        "difference": got - ref, "met": bool(got - ref <= MARGIN),
        "for scale": {"relative, floor off (R8's failure)":
                      avg("s1", "relative", "A1", "attacker_rare_weight")},
    }
    # 2 - no cost on honest data.
    for split in ("s1", "s2"):
        a = avg(split, REFERENCE, "none", "rare_macro_f1")
        b = avg(split, "relative + floor", "none", "rare_macro_f1")
        checks[f"2. {split}: relative + floor costs nothing without an attack"] = {
            "reference (absolute)": a, "relative + floor": b,
            "difference": b - a, "met": bool(abs(b - a) <= MARGIN)}
    # 3 - the floor does not harm EARN as it stands.
    for split in ("s1", "s2"):
        a = avg(split, REFERENCE, "none", "rare_macro_f1")
        b = avg(split, "absolute + floor", "none", "rare_macro_f1")
        checks[f"3a. {split}: the floor does not cost the current rule accuracy"] = {
            "reference (absolute)": a, "absolute + floor": b,
            "difference": b - a, "met": bool(abs(b - a) <= MARGIN)}
        ra = avg(split, REFERENCE, "A1", "attacker_rare_weight")
        rb = avg(split, "absolute + floor", "A1", "attacker_rare_weight")
        checks[f"3b. {split}: the floor does not weaken the current rule under A1"] = {
            "reference (absolute)": ra, "absolute + floor": rb,
            "difference": rb - ra, "met": bool(rb - ra <= MARGIN)}

    # 4 - regression against R8's recorded numbers for the two floor-off arms.
    r8_path = OUT / "amendment_m2_fedisic.json"
    regression = {"checked": False}
    if r8_path.exists():
        r8 = json.loads(r8_path.read_text(encoding="utf-8"))
        def r8_avg(split, arm, attack, key):
            vals = [r[key] for r in r8["runs"] if r["split"] == split and r["arm"] == arm
                    and r["attack"] == attack and r.get(key) is not None]
            return float(np.mean(vals)) if vals else None
        diffs = {}
        for split in ("s1", "s2"):
            for arm in ("absolute", "relative"):
                for attack in ("none", "A1"):
                    now = avg(split, arm, attack, "rare_macro_f1")
                    then = r8_avg(split, arm, attack, "rare_macro_f1")
                    if now is not None and then is not None:
                        diffs[f"{split}/{arm}/{attack}"] = abs(now - then)
        worst = max(diffs.values()) if diffs else None
        regression = {"checked": True, "largest_difference": worst,
                      "met": bool(worst is not None and worst < 1e-6), "per_cell": diffs}
    checks["4. the floor-off arms reproduce R8"] = regression

    passed = all(c.get("met") for c in checks.values())
    decision = ("relative + floor becomes the proposed default: scale-free, and no more "
                "spoofable than today's rule" if passed else
                "the floor does not do what M3 was written to do; the absolute threshold stays")
    record = {
        "amendment": "M3 - peer-confidence floor", "p_floor": P_FLOOR,
        "fixed_in": "docs/DEVIATIONS.md D12, before the first run",
        "features": "frozen", "rounds": args.rounds, "seeds": list(SEEDS), "margin": MARGIN,
        "reference_arm": REFERENCE, "checks": checks, "passed": bool(passed),
        "decision": decision,
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": rows,
    }
    (OUT / "amendment_m3.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    print("\n" + "=" * 76)
    for label, c in checks.items():
        if "largest_difference" in c:
            print(f"  {'met    ' if c.get('met') else 'not met'}  {label}"
                  + (f" (largest difference {c['largest_difference']:.2e})"
                     if c.get("largest_difference") is not None else " (no R8 file)"))
        else:
            print(f"  {'met    ' if c['met'] else 'not met'}  {label}")
            print(f"            {c.get('reference (absolute)'):.4f} -> "
                  f"{[v for k, v in c.items() if k.startswith(('relative', 'absolute +'))][0]:.4f}")
    print(f"\nM3 -> {'ALL CONDITIONS MET' if passed else 'NOT MET'}")
    print(f"  {decision}")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
