#!/usr/bin/env python3
"""
D7 follow-up (docs/DEVIATIONS.md): does EARN's locked history matter when the specialist itself turns?

Attack A2s = sleeper behaviour by centre 2 (honest rounds 1-15, A1 from round 16). Runs fedavg,
camp_a_reported, earn and earn_no_ledger on S1 and S2 with seeds 42-44 through the same
src/experiments/grid.run_one as the main study, then evaluates the pre-registered criterion:

    lock matters on a split  <=>  rareF1(earn) - rareF1(earn_no_ledger) >= 0.05
                                  AND attacker weight(earn) <= 0.5 * attacker weight(earn_no_ledger)

No-attack reference rows are read from the D6 grid (results/grid/runs.csv).

Output: results/followup/*.json.gz (one per run), results/followup_ledger.json

    python scripts/11_ledger_followup.py
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.experiments import grid

OUT = ROOT / "results" / "followup"
METHODS = ("fedavg", "camp_a_reported", "earn", "earn_no_ledger")
ATTACK = "A2s"
F1_MARGIN, WEIGHT_FACTOR = 0.05, 0.5          # pre-registered in D7
_CTX = None


def _init():
    global _CTX
    import torch
    torch.set_num_threads(1)
    _CTX = grid.load_context()


def _work(job):
    split, method, seed = job
    path = OUT / f"{split}__{method}__{ATTACK}__{seed}.json.gz"
    if not path.exists():
        out = grid.run_one(_CTX, split, method, ATTACK, seed)
        out.pop("weights", None)          # keep the files small; trust/weights summarised in "run"
        with gzip.open(path, "wt") as f:
            json.dump(out, f)
    with gzip.open(path, "rt") as f:
        return json.load(f)["run"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, m, seed) for s in grid.SPLITS for m in METHODS for seed in grid.SEEDS]
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1), initializer=_init) as pool:
        runs = pd.DataFrame(list(pool.map(_work, jobs)))
    base = pd.read_csv(ROOT / "results" / "grid" / "runs.csv")
    base = base[(base.attack == "none") & base.method.isin(METHODS)]

    def mean(df, split, method, col):
        return float(df[(df.split == split) & (df.method == method)][col].mean())

    record = {"entry": "D7", "attack": ATTACK, "attacker": 2, "seeds": list(grid.SEEDS),
              "criterion": {"f1_margin": F1_MARGIN, "weight_factor": WEIGHT_FACTOR}, "splits": {}}
    print(f"\nD7 - specialist sleeper (centre 2 turns at round 16), mean of seeds {grid.SEEDS}\n")
    for split in grid.SPLITS:
        rows = {}
        for m in METHODS:
            rows[m] = {"rare_f1_no_attack": mean(base, split, m, "rare_macro_f1"),
                       "rare_f1_A2s": mean(runs, split, m, "rare_macro_f1"),
                       "balanced_accuracy_A2s": mean(runs, split, m, "balanced_accuracy"),
                       "to_nevus_A2s": mean(runs, split, m, "to_target_rare"),
                       "attacker_weight_5": mean(runs, split, m, "attacker_weight_5"),
                       "attacker_weight_6": mean(runs, split, m, "attacker_weight_6"),
                       "attacker_fedavg_weight": mean(runs, split, m, "attacker_fedavg_weight")}
            if m.startswith("earn"):
                for c in (5, 6):
                    rows[m][f"attacker_trust_{c}_final"] = mean(runs, split, m, f"attacker_trust_{c}_final")
        e, nl = rows["earn"], rows["earn_no_ledger"]
        d_f1 = e["rare_f1_A2s"] - nl["rare_f1_A2s"]
        w_e = np.mean([e["attacker_weight_5"], e["attacker_weight_6"]])
        w_nl = np.mean([nl["attacker_weight_5"], nl["attacker_weight_6"]])
        met = bool(d_f1 >= F1_MARGIN and w_e <= WEIGHT_FACTOR * w_nl)
        record["splits"][split] = {"methods": rows, "rare_f1_gain_from_lock": d_f1,
                                   "attacker_weight_earn": w_e, "attacker_weight_no_ledger": w_nl,
                                   "lock_matters": met}
        print(f"  {split.upper()}")
        for m, r in rows.items():
            extra = (f"  attacker trust end {r['attacker_trust_5_final']:.2f}/{r['attacker_trust_6_final']:.2f}"
                     if "attacker_trust_5_final" in r else "")
            print(f"    {m:<16} rare F1 none {r['rare_f1_no_attack']:.3f} -> A2s {r['rare_f1_A2s']:.3f}"
                  f"   attacker weight DF/VL {r['attacker_weight_5']:.3f}/{r['attacker_weight_6']:.3f}{extra}")
        print(f"    lock gain in rare F1 {d_f1:+.3f}; attacker weight earn {w_e:.3f} vs no-ledger {w_nl:.3f}"
              f"  -> lock matters: {'YES' if met else 'no'}\n")
    (ROOT / "results" / "followup_ledger.json").write_text(json.dumps(record, indent=2))
    print(f"wrote {ROOT / 'results' / 'followup_ledger.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
