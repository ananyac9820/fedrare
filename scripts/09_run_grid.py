#!/usr/bin/env python3
"""
Run the experiment grid pre-registered in docs/DEVIATIONS.md D6, in parallel.

    python scripts/09_run_grid.py --grid f1       # Fallback F1: 8 methods x 4 attacks x S1/S2 x 3 seeds
    python scripts/09_run_grid.py --grid earn     # exploratory EARN (oracle evidence), same axes
    python scripts/09_run_grid.py --grid all

Each run is written to results/grid/parts/ as soon as it finishes, so an interrupted grid resumes
where it stopped. At the end the parts are combined into (docs/results_format.md):

    results/grid/runs.csv          one row per (split, method, attack, seed)
    results/grid/rounds.csv.gz     per-round test metrics
    results/grid/weights.csv.gz    per-round rare-row weight / evidence / trust of every client
    results/ledger/earn_rounds_<split>_<attack>_seed<seed>.json   EARN trust tables for the chain
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.experiments import grid
from src.ledger.chain import to_bps

OUT = ROOT / "results" / "grid"
PARTS = OUT / "parts"
LEDGER_DIR = ROOT / "results" / "ledger"

_CTX = None


def _init_worker():
    global _CTX
    import torch
    torch.set_num_threads(1)
    _CTX = grid.load_context()


def _key(split, method, attack, seed) -> str:
    return f"{split}__{method}__{attack}__{seed}"


def _work(job):
    split, method, attack, seed = job
    out = grid.run_one(_CTX, split, method, attack, seed)
    path = PARTS / f"{_key(*job)}.json.gz"
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(out, f)
    tmp.rename(path)
    return job, out["run"]


def combine():
    runs, rounds, weights = [], [], []
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    for p in sorted(PARTS.glob("*.json.gz")):
        with gzip.open(p, "rt") as f:
            out = json.load(f)
        runs.append(out["run"])
        rounds += out["rounds"]
        weights += out["weights"]
        if "ledger" in out:
            r = out["run"]
            for block in out["ledger"]:
                block["trust_bps"] = to_bps(block["trust"]).ravel().tolist()
            (LEDGER_DIR / f"earn_rounds_{r['split']}_{r['attack']}_seed{r['seed']}.json").write_text(
                json.dumps({"split": r["split"], "attack": r["attack"], "seed": r["seed"],
                            "n_clients": 6, "n_classes": 8, "max_step": 0.1,
                            "rounds": out["ledger"]}))
    pd.DataFrame(runs).to_csv(OUT / "runs.csv", index=False)
    pd.DataFrame(rounds).to_csv(OUT / "rounds.csv.gz", index=False)
    pd.DataFrame(weights).to_csv(OUT / "weights.csv.gz", index=False)
    print(f"combined {len(runs)} runs -> {OUT}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=("f1", "earn", "all"), default="all")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--combine-only", action="store_true")
    args = ap.parse_args()
    PARTS.mkdir(parents=True, exist_ok=True)
    if args.combine_only:
        combine()
        return 0

    methods = {"f1": grid.F1_METHODS, "earn": grid.EARN_METHODS,
               "all": grid.F1_METHODS + grid.EARN_METHODS}[args.grid]
    jobs = [(s, m, a, seed) for s in grid.SPLITS for m in methods for a in grid.ATTACKS
            for seed in grid.SEEDS]
    todo = [j for j in jobs if not (PARTS / f"{_key(*j)}.json.gz").exists()]
    print(f"{len(jobs)} runs in grid '{args.grid}', {len(todo)} to do, {args.workers} workers",
          flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as pool:
        futures = [pool.submit(_work, j) for j in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            job, run = fut.result()
            print(f"  [{i:>3}/{len(todo)}] {'/'.join(map(str, job)):<45} bal acc "
                  f"{run['balanced_accuracy']:.3f}  rare F1 {run['rare_macro_f1']:.3f}  "
                  f"({run['seconds']:.0f}s, elapsed {time.time() - t0:.0f}s)", flush=True)
    combine()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
