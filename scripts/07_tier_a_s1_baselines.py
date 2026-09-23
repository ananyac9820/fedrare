#!/usr/bin/env python3
"""
Tier A baselines on the natural split S1, and Gate G0b.

1. FedAvg on S1, 3 seeds. Gate G0b (design doc Section 6.6, fixed before any result):
   Tier A FedAvg reaches balanced accuracy >= 0.45. Measured here as the final-round global
   head's balanced accuracy on the pooled test split, averaged over the 3 seeds.
   If G0b fails the script stops: the design doc's retry (unfreeze the last dense block and
   re-extract features) is a team decision.
2. Only if G0b passes: Camp A (evidence-proportional per-class weighting, design doc
   Section 6.3) on S1, same 3 seeds.

Both use TierAConfig defaults (src/federated/tier_a.py) - the same local training G0a
measured - and the aggregators in src/federated/baselines.py.

Outputs to results/, one pair of CSVs per rule run (rule = fedavg, camp_a):
    tier_a_s1_<rule>.csv          seed, rule, round, test metrics of the global head each round
    tier_a_s1_<rule>_weights.csv  seed, rule, round, client_id, class_id, head_row_weight, evidence
    gate_g0b.json                 the gate record
    tier_a_s1.md                  summary

    python scripts/07_tier_a_s1_baselines.py              # FedAvg + G0b, then Camp A if it passes
    python scripts/07_tier_a_s1_baselines.py --no-camp-a  # FedAvg + G0b only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.data.features import load_features
from src.data.splits import make_split
from src.federated import baselines
from src.federated.tier_a import TierAConfig, make_clients, run_federated
from src.utils.config import load_config

# --- Pre-registered. Do not change after seeing results. ---
SEEDS = (42, 43, 44)
G0B_THRESHOLD = 0.45
SPECIALIST = 2


def load_s02():
    spec = importlib.util.spec_from_file_location("s02", ROOT / "scripts" / "02_explore_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_rule(rule: str, clients, test, cfg, rare_ids, names) -> tuple[list[dict], list[dict], list[dict]]:
    metric_rows, weight_rows, results = [], [], []
    for seed in SEEDS:
        def progress(r, info, m, _seed=seed):
            if r % 10 == 0 or r == cfg.rounds:
                print(f"  {rule:<8} seed {_seed} round {r:>2}/{cfg.rounds} | balanced acc "
                      f"{m['balanced_accuracy']:.3f} | rare macro-F1 {m['rare_macro_f1']:.3f}",
                      flush=True)

        res = run_federated(lambda: baselines.RULES[rule], clients,
                            torch.from_numpy(test.features), test.labels, cfg, seed, rare_ids,
                            on_round=progress)
        results.append(res)
        for r, (info, m) in enumerate(zip(res["round_infos"], res["round_metrics"]), 1):
            row = {"seed": seed, "rule": rule, "round": r,
                   **{k: m[k] for k in ("accuracy", "balanced_accuracy", "macro_f1",
                                        "rare_macro_f1")}}
            row.update({f"f1_{c}_{names[c].replace(' ', '_')}": m["per_class_f1"][c]
                        for c in range(cfg.n_classes)})
            metric_rows.append(row)
            ev = info.get("evidence")
            for k in range(len(clients)):
                for c in range(cfg.n_classes):
                    weight_rows.append({
                        "seed": seed, "rule": rule, "round": r, "client_id": k, "class_id": c,
                        "head_row_weight": float(info["head_row_weights"][k, c]),
                        "evidence": float(ev[k, c]) if ev is not None else np.nan})
        print(f"  {rule} seed {seed}: final balanced acc {res['final']['balanced_accuracy']:.3f}, "
              f"rare macro-F1 {res['final']['rare_macro_f1']:.3f} ({res['seconds']:.0f}s)")
    return metric_rows, weight_rows, results


def summary_row(results: list[dict], rule: str, rare_ids: list[int], names: list[str]) -> list:
    f = lambda key: np.array([r["final"][key] for r in results])  # noqa: E731
    per_class = np.array([[r["final"]["per_class_f1"][c] for c in rare_ids] for r in results])
    cells = [rule]
    for key in ("balanced_accuracy", "rare_macro_f1", "macro_f1", "accuracy"):
        cells.append(f"{f(key).mean():.3f} ± {f(key).std():.3f}")
    cells += [f"{per_class[:, i].mean():.3f}" for i in range(len(rare_ids))]
    return cells


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main() -> int:
    parser = argparse.ArgumentParser(description="Tier A baselines on S1 and Gate G0b.")
    parser.add_argument("--no-camp-a", action="store_true",
                        help="run FedAvg and check G0b only; skip Camp A")
    args = parser.parse_args()
    cfg_yaml = load_config(ROOT / "configs" / "default.yaml")
    n_centers, n_classes = cfg_yaml["dataset"]["n_centers"], cfg_yaml["dataset"]["n_classes"]
    names = [cfg_yaml["dataset"]["class_names"][c] for c in range(n_classes)]
    out_dir = ROOT / cfg_yaml.get("output_dir", "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = TierAConfig(n_classes=n_classes)

    train, test = load_features("train"), load_features("test")
    both = pd.DataFrame(train.counts(n_centers, n_classes) + test.counts(n_centers, n_classes),
                        index=[f"centre_{k}" for k in range(n_centers)], columns=names)
    rare_ids, _ = load_s02().identify_rare(both, cfg_yaml["dataset"]["head_ratio_divisor"])
    centers = make_split("s1", train.centers, train.labels, rare_ids)
    clients = make_clients(train.features, train.labels, centers, n_centers)
    sizes = np.array([c.size for c in clients])
    print(f"S1 clients: {sizes.tolist()} training images; rare classes {rare_ids}")
    print(f"Tier A config: {cfg}\n")

    metric_rows, weight_rows, all_results = [], [], {}

    # ---------------------------------------------------------------- FedAvg + G0b
    print("FedAvg on S1")
    m, w, all_results["fedavg"] = run_rule("fedavg", clients, test, cfg, rare_ids, names)
    metric_rows += m
    weight_rows += w
    bal = np.array([r["final"]["balanced_accuracy"] for r in all_results["fedavg"]])
    g0b_passed = bool(bal.mean() >= G0B_THRESHOLD)
    gate = {"gate": "G0b", "passed": g0b_passed, "statistic": float(bal.mean()),
            "threshold": G0B_THRESHOLD, "per_seed": dict(zip(map(str, SEEDS), bal.tolist())),
            "definition": "Tier A FedAvg on S1, final-round balanced accuracy on the pooled "
                          "test split, mean over seeds", "tier_a_config": cfg.__dict__}
    (out_dir / "gate_g0b.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    bar = "=" * 78
    print(f"\n{bar}\nG0b: {'PASS' if g0b_passed else 'FAIL'}  (FedAvg balanced accuracy "
          f"{bal.mean():.3f} ± {bal.std():.3f} over seeds {SEEDS}; threshold {G0B_THRESHOLD})\n{bar}")

    # ---------------------------------------------------------------- Camp A
    if g0b_passed and args.no_camp_a:
        print("\nCamp A skipped (--no-camp-a).")
    elif g0b_passed:
        print("\nCamp A on S1")
        m, w, all_results["camp_a"] = run_rule("camp_a", clients, test, cfg, rare_ids, names)
        metric_rows += m
        weight_rows += w
    else:
        print("  STOP: Camp A not run. Per the design doc, the retry (unfreeze the last dense "
              "block and re-extract features) is a team decision.")

    metrics = pd.DataFrame(metric_rows)
    weights = pd.DataFrame(weight_rows)
    for rule in all_results:
        metrics[metrics["rule"] == rule].to_csv(out_dir / f"tier_a_s1_{rule}.csv", index=False)
        weights[weights["rule"] == rule].to_csv(out_dir / f"tier_a_s1_{rule}_weights.csv", index=False)

    # ---------------------------------------------------------------- summary
    rare_names = [names[c] for c in rare_ids]
    md = [
        "# Tier A baselines on S1 (natural split)", "",
        f"**G0b: {'PASS' if g0b_passed else 'FAIL'}** - FedAvg balanced accuracy "
        f"{bal.mean():.3f} ± {bal.std():.3f} (seeds {', '.join(map(str, SEEDS))}), threshold "
        f"{G0B_THRESHOLD}.", "",
        f"Tier A: frozen DenseNet-121 features, linear head, {cfg.rounds} rounds, "
        f"{cfg.local_steps} local Adam steps per centre per round (batch {cfg.batch_size}, "
        f"lr {cfg.lr}), class-balanced loss: {cfg.class_balanced_loss}. Final-round global "
        "head on the pooled test split; mean ± std over seeds.", "",
        md_table(["rule", "balanced acc", "rare macro-F1", "macro-F1", "accuracy"]
                 + [f"F1 {n}" for n in rare_names],
                 [summary_row(res, rule, rare_ids, names) for rule, res in all_results.items()]),
        "",
    ]
    if "camp_a" in all_results:
        spec = weights[weights["client_id"] == SPECIALIST]
        rows = []
        for c in rare_ids:
            per_rule = spec[spec["class_id"] == c].groupby("rule")["head_row_weight"].mean()
            rows.append([names[c], f"{per_rule['fedavg']:.3f}", f"{per_rule['camp_a']:.3f}",
                         f"{per_rule['camp_a'] / per_rule['fedavg']:.2f}x"])
        md += [f"Centre {SPECIALIST}'s mean weight on each rare head row (all rounds, all seeds):",
               "", md_table(["row", "FedAvg", "Camp A", "Camp A / FedAvg"], rows), ""]
    md += ["Per-round data: " + ", ".join(f"`tier_a_s1_{r}.csv`, `tier_a_s1_{r}_weights.csv`"
                                          for r in all_results) + ".", ""]
    (out_dir / "tier_a_s1.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n  wrote {out_dir / 'tier_a_s1.md'} and the CSVs")
    return 0 if g0b_passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
