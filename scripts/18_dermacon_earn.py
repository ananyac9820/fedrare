#!/usr/bin/env python3
"""
EARN and Camp A on DermaCon-IN, and gate DG2 (docs/DERMACON_PLAN.md).

Same exploratory setting as the Fed-ISIC2019 EARN analysis: the evidence signal is an
ORACLE - each centre's true per-class training counts - because the measured signal did not
survive gate G0a (docs/DEVIATIONS.md R1). This says what the mechanism would do given a
working signal. It is not a validated method, here or on Fed-ISIC2019.

DG2, fixed before this ran:
  - on D1, no attack: EARN's rare-class macro-F1 must be no more than 0.05 below FedAvg's,
    and the specialist centre (the 60-80 age band, centre 4) must end with a weight share on
    the rare rows above its size share;
  - on D0 (near-IID): EARN must stay within 0.05 of FedAvg in either direction - there is no
    specialist to find, so finding one would be the method misfiring.

    python scripts/18_dermacon_earn.py

Writes docs/results/dermacon_earn.csv and docs/results/gate_dg2.json.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.data.splits import HOLDER_MIN_IMAGES
from src.federated import baselines
from src.federated.earn import EARN, EARNConfig
from src.federated.tier_a import TierAConfig, make_clients, run_federated

FEATURES = ROOT / "data" / "features" / "dermacon_densenet121.npz"
META = ROOT / "data" / "features" / "dermacon_densenet121.json"
OUT = ROOT / "docs" / "results"

RARE_NAMES = ("Keratanisation Disorders", "Neoplasms and tumors")
SPECIALIST = {"d1": 4, "d0": None}   # D1 centre 4 = the 60-80 band, named in the plan
ORACLE_TAU = HOLDER_MIN_IMAGES - 0.5
ROUNDS, SEED, MARGIN = 40, 42, 0.05
N_CENTRES = 6


def load():
    d = np.load(FEATURES, allow_pickle=True)
    keep = d["labels"] >= 0
    classes = json.loads(META.read_text(encoding="utf-8"))["classes"]
    return {k: d[k][keep] for k in ("features", "labels", "d0", "d1", "is_train")}, classes


def counts_matrix(y, c, n_classes):
    """True per-centre class counts - the oracle evidence, and what Camp A is given."""
    return np.stack([np.bincount(y[c == k], minlength=n_classes) for k in range(N_CENTRES)])


def main() -> int:
    data, classes = load()
    n_classes = len(classes)
    rare_ids = [classes.index(n) for n in RARE_NAMES]
    test = ~data["is_train"]
    test_x = torch.from_numpy(np.ascontiguousarray(data["features"][test]))
    test_y = data["labels"][test]
    cfg = TierAConfig(rounds=ROUNDS, n_classes=n_classes)

    rows, summary = [], {}
    for split_key in ("d0", "d1"):
        train = data["is_train"]
        y, c = data["labels"][train], data[split_key][train]
        clients = make_clients(data["features"][train], y, c, N_CENTRES)
        counts = counts_matrix(y, c, n_classes)
        sizes = np.array([cl.size for cl in clients], dtype=float)
        size_share = sizes / sizes.sum()

        def oracle(updates, layout, round_num, _counts=counts):
            return _counts.astype(np.float64)

        methods = {
            "fedavg": lambda: baselines.fedavg,
            "camp_a_reported": lambda: partial(baselines.camp_a, evidence_fn=oracle,
                                               rule_name="camp_a_reported"),
            "earn": lambda: EARN(EARNConfig(tau=ORACLE_TAU), evidence_fn=oracle, name="earn"),
        }

        print(f"\n{split_key.upper()}  sizes {sizes.astype(int).tolist()}  "
              f"rare counts per centre {counts[:, rare_ids].sum(axis=1).tolist()}")
        summary[split_key] = {}
        for name, factory in methods.items():
            res = run_federated(factory, clients, test_x, test_y, cfg, seed=SEED,
                                rare_ids=rare_ids, verbose=False)
            final = res["final"]

            # Weight the method ended up giving each centre on the rare rows.
            last = res["round_infos"][-1]
            head_w = last.get("head_row_weights")
            rare_w = (head_w[:, rare_ids].mean(axis=1) if head_w is not None else size_share)
            summary[split_key][name] = {
                "balanced_accuracy": final["balanced_accuracy"],
                "rare_macro_f1": final["rare_macro_f1"],
                "accuracy": final["accuracy"],
                "rare_weight_share": rare_w.tolist(),
                "size_share": size_share.tolist(),
            }
            print(f"  {name:<16} balanced acc {final['balanced_accuracy']:.4f} | "
                  f"rare macro-F1 {final['rare_macro_f1']:.4f}")
            spec = SPECIALIST[split_key]
            if spec is not None:
                print(f"      centre {spec}: size share {size_share[spec]:.3f} -> "
                      f"rare-row weight {rare_w[spec]:.3f}")
            for r, m in enumerate(res["round_metrics"], start=1):
                rows.append({"split": split_key, "method": name, "seed": SEED, "round": r,
                             "accuracy": m["accuracy"],
                             "balanced_accuracy": m["balanced_accuracy"],
                             "macro_f1": m["macro_f1"],
                             "rare_macro_f1": m["rare_macro_f1"]})

    csv_path = OUT / "dermacon_earn.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- DG2, exactly as written down -------------------------------------------------
    d1, d0 = summary["d1"], summary["d0"]
    spec = SPECIALIST["d1"]
    not_worse = d1["earn"]["rare_macro_f1"] >= d1["fedavg"]["rare_macro_f1"] - MARGIN
    lifts_specialist = (d1["earn"]["rare_weight_share"][spec]
                        > d1["earn"]["size_share"][spec])
    quiet_on_iid = abs(d0["earn"]["rare_macro_f1"]
                       - d0["fedavg"]["rare_macro_f1"]) <= MARGIN
    passed = bool(not_worse and lifts_specialist and quiet_on_iid)

    record = {
        "gate": "DG2", "question": "does EARN help on DermaCon-IN?",
        "fixed_in": "docs/DERMACON_PLAN.md, before any run",
        "setting": "oracle evidence (exploratory) - the measured signal did not clear G0a",
        "margin": MARGIN, "rounds": ROUNDS, "seed": SEED,
        "conditions": {
            "D1 rare macro-F1 not worse than FedAvg by more than the margin": {
                "earn": d1["earn"]["rare_macro_f1"],
                "fedavg": d1["fedavg"]["rare_macro_f1"], "met": bool(not_worse)},
            "D1 specialist weight share on rare rows above its size share": {
                "centre": spec, "size_share": d1["earn"]["size_share"][spec],
                "rare_weight_share": d1["earn"]["rare_weight_share"][spec],
                "met": bool(lifts_specialist)},
            "D0 within the margin of FedAvg (no effect expected)": {
                "earn": d0["earn"]["rare_macro_f1"],
                "fedavg": d0["fedavg"]["rare_macro_f1"], "met": bool(quiet_on_iid)},
        },
        "passed": passed,
        "all_methods": summary,
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (OUT / "gate_dg2.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    for label, cond in record["conditions"].items():
        print(f"  {'met    ' if cond['met'] else 'not met'}  {label}")
    print(f"GATE DG2 -> {'CLEARED' if passed else 'NOT CLEARED'}")
    print("=" * 74)
    print(f"wrote {csv_path.relative_to(ROOT)} and gate_dg2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
