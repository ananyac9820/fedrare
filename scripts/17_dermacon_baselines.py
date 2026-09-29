#!/usr/bin/env python3
"""
Tier A baselines on DermaCon-IN, and gate DG0 - the Indian counterpart of scripts/07.

Runs the same federated loop as Fed-ISIC2019 (frozen features, one linear head, FedAvg,
class-balanced local loss) on the two constructed splits fixed in docs/DERMACON_PLAN.md:

    D0  patient id hashed into 6 groups   near-IID control
    D1  six age bands                     naturally uneven

and checks DG0, whose threshold was written down before this script ever ran:

    FedAvg on D1, 40 rounds, seed 42, balanced accuracy >= 0.40 over the 7 classes.
    One retry, also specified in advance: 100 rounds, same seed.

If DG0 does not clear, this script says so and stops - the plan says DermaCon-IN is then
used for external validation only, and the shortfall is reported rather than tuned away.

    python scripts/17_dermacon_baselines.py
    python scripts/17_dermacon_baselines.py --rounds 100 --split d1   # the pre-set retry

Writes docs/results/dermacon_baselines.csv and docs/results/gate_dg0.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.federated import baselines
from src.federated.tier_a import TierAConfig, make_clients, run_federated

FEATURES = ROOT / "data" / "features" / "dermacon_densenet121.npz"
META = ROOT / "data" / "features" / "dermacon_densenet121.json"
OUT = ROOT / "docs" / "results"

RARE_NAMES = ("Keratanisation Disorders", "Neoplasms and tumors")  # dataset's own spelling
DG0_THRESHOLD = 0.40      # docs/DERMACON_PLAN.md, fixed before any run
DG0_ROUNDS = 40
DG0_RETRY_ROUNDS = 100
SEED = 42


def load():
    if not FEATURES.exists():
        raise SystemExit(f"missing {FEATURES}\n"
                         f"Run: python scripts/16_extract_dermacon_features.py")
    d = np.load(FEATURES, allow_pickle=True)
    keep = d["labels"] >= 0                       # drops "No Definite Diagnosis"
    classes = json.loads(META.read_text(encoding="utf-8"))["classes"]
    return ({k: d[k][keep] for k in ("features", "labels", "d0", "d1", "is_train")},
            classes)


def split_clients(data, split_key, n_centres=6):
    """Training clients for one constructed split, dropping any centre with no data."""
    train = data["is_train"]
    x, y, c = data["features"][train], data["labels"][train], data[split_key][train]
    clients, kept = [], []
    for k in range(n_centres):
        if (c == k).sum() == 0:
            continue
        kept.append(k)
        clients.append(k)
    made = make_clients(x, y, c, n_centres)
    return [made[k] for k in kept], kept


def specialist_ratio(data, split_key, rare_ids, n_centres=6):
    """Rare-share divided by size-share per centre - the project's headline quantity."""
    train = data["is_train"]
    y, c = data["labels"][train], data[split_key][train]
    rare = np.isin(y, rare_ids)
    out = {}
    for k in range(n_centres):
        size = int((c == k).sum())
        if not size:
            continue
        r = int((rare & (c == k)).sum())
        out[k] = {"images": size, "rare": r,
                  "ratio": (r / max(rare.sum(), 1)) / (size / len(y))}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="DermaCon-IN Tier A baselines and DG0.")
    parser.add_argument("--rounds", type=int, default=DG0_ROUNDS)
    parser.add_argument("--split", choices=["d0", "d1", "both"], default="both")
    parser.add_argument("--rule", default="fedavg", choices=sorted(baselines.RULES))
    args = parser.parse_args()

    data, classes = load()
    rare_ids = [classes.index(n) for n in RARE_NAMES if n in classes]
    if len(rare_ids) != len(RARE_NAMES):
        raise SystemExit(f"rare classes {RARE_NAMES} not all present in {classes}")

    test = ~data["is_train"]
    test_x = torch.from_numpy(np.ascontiguousarray(data["features"][test]))
    test_y = data["labels"][test]
    cfg = TierAConfig(rounds=args.rounds, n_classes=len(classes))

    print(f"DermaCon-IN | {len(data['labels']):,} usable images | {len(classes)} classes")
    print(f"  rare: {', '.join(RARE_NAMES)} -> ids {rare_ids}")
    print(f"  train {int(data['is_train'].sum()):,} / test {int(test.sum()):,}")

    OUT.mkdir(parents=True, exist_ok=True)
    rows, results = [], {}
    for split_key in (["d0", "d1"] if args.split == "both" else [args.split]):
        clients, kept = split_clients(data, split_key)
        ratios = specialist_ratio(data, split_key, rare_ids)
        print(f"\n{split_key.upper()}  {len(clients)} centres with data "
              f"(sizes {[c.size for c in clients]})")
        for k, v in ratios.items():
            print(f"    centre {k}: {v['images']:>5} images, {v['rare']:>3} rare, "
                  f"ratio {v['ratio']:.2f}x")

        res = run_federated(lambda: baselines.RULES[args.rule], clients, test_x, test_y,
                            cfg, seed=SEED, rare_ids=rare_ids, verbose=False)
        final = res["final"]
        results[split_key] = final
        print(f"  {args.rule}, {cfg.rounds} rounds, seed {SEED}: "
              f"balanced accuracy {final['balanced_accuracy']:.4f} | "
              f"rare macro-F1 {final['rare_macro_f1']:.4f} | "
              f"accuracy {final['accuracy']:.4f}")
        # per_class_f1 is a dict keyed by class index, not a list - iterating it directly
        # yields the keys, which silently prints (and stores) indices instead of scores.
        for i, name in enumerate(classes):
            print(f"      {name[:38]:<38} F1 {final['per_class_f1'][i]:.3f}")

        for r, m in enumerate(res["round_metrics"], start=1):
            rows.append({"split": split_key, "rule": args.rule, "seed": SEED, "round": r,
                         "accuracy": m["accuracy"],
                         "balanced_accuracy": m["balanced_accuracy"],
                         "macro_f1": m["macro_f1"], "rare_macro_f1": m["rare_macro_f1"],
                         **{f"f1_{i}_{classes[i].replace(' ', '_')}": m["per_class_f1"][i]
                            for i in range(len(classes))}})

    csv_path = OUT / f"dermacon_baselines_{args.rule}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {csv_path.relative_to(ROOT)}")

    # ---- DG0, exactly as written down beforehand -------------------------------------
    if "d1" in results and args.rule == "fedavg":
        got = results["d1"]["balanced_accuracy"]
        passed = got >= DG0_THRESHOLD
        is_retry = args.rounds == DG0_RETRY_ROUNDS
        record = {
            "gate": "DG0",
            "question": "is the DermaCon-IN baseline good enough to study?",
            "threshold": DG0_THRESHOLD,
            "fixed_in": "docs/DERMACON_PLAN.md, before any run",
            "split": "D1", "rule": "fedavg", "rounds": args.rounds, "seed": SEED,
            "statistic": got,
            "passed": bool(passed),
            "attempt": "retry" if is_retry else "first",
            "rare_macro_f1": results["d1"]["rare_macro_f1"],
            "accuracy": results["d1"]["accuracy"],
            "n_classes": len(classes),
            "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        path = OUT / ("gate_dg0_retry.json" if is_retry else "gate_dg0.json")
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")

        print("\n" + "=" * 74)
        print(f"GATE DG0: balanced accuracy {got:.4f} against a bar of {DG0_THRESHOLD:.2f} "
              f"-> {'CLEARED' if passed else 'NOT CLEARED'}")
        if not passed and not is_retry:
            print("  The plan allows one retry, already specified: 100 rounds, same seed.")
            print("  Run: python scripts/17_dermacon_baselines.py --rounds 100 --split d1")
        elif not passed:
            print("  The retry did not clear it either. Under the plan, DermaCon-IN is now")
            print("  used for external validation only, and this shortfall is reported.")
        print("=" * 74)
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
