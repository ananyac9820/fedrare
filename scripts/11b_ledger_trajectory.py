#!/usr/bin/env python3
"""
Diagnostic for D7/R4 (docs/DEVIATIONS.md): round-by-round trust, agreement and rare F1 when the
specialist (centre 2) turns at round 16 on S2, with a locked history (`earn`) vs an editable one
(`earn_no_ledger`). Explains the D7 result; it is not a new variant and has no pass/fail criterion.

Output: results/followup_ledger_trajectory.json (mean over seeds 42-44, per round)

    python scripts/11b_ledger_trajectory.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src.attacks.attacks import DEFAULT_ATTACKS
from src.experiments import grid
from src.federated.tier_a import run_federated

SPLIT, ATTACK, ATTACKER = "s2", "A2s", 2


def main() -> int:
    ctx = grid.load_context()
    out = {"split": SPLIT, "attack": ATTACK, "attacker": ATTACKER, "seeds": list(grid.SEEDS),
           "note": "diagnostic for D7/R4; mean over seeds; trust/agreement on the vascular-lesion (6) "
                   "and dermatofibroma (5) rows of the attacker", "methods": {}}
    for method in ("earn", "earn_no_ledger"):
        trust, agree, f1 = [], [], []
        for seed in grid.SEEDS:
            res = run_federated(grid.aggregator_factory(method, ctx.counts[SPLIT], ATTACK),
                                ctx.clients[SPLIT], ctx.test_x, ctx.test_y, ctx.cfg, seed, grid.RARE_IDS,
                                client_hooks=DEFAULT_ATTACKS[ATTACK].hooks(grid.RARE_IDS))
            trust.append([[i["trust"][ATTACKER, c] for c in (5, 6)] for i in res["round_infos"]])
            agree.append([[i["agreement"][ATTACKER, c] for c in (5, 6)] for i in res["round_infos"]])
            f1.append([m["rare_macro_f1"] for m in res["round_metrics"]])
        t, a, r = np.mean(trust, 0), np.nanmean(agree, 0), np.mean(f1, 0)
        out["methods"][method] = {
            "trust_5": t[:, 0].round(4).tolist(), "trust_6": t[:, 1].round(4).tolist(),
            "agreement_5": a[:, 0].round(4).tolist(), "agreement_6": a[:, 1].round(4).tolist(),
            "rare_f1": r.round(4).tolist(),
            "min_trust_6_after_turn": float(t[15:, 1].min()),
            "round_of_min": int(16 + np.argmin(t[15:, 1])),
            "trust_6_round_30": float(t[29, 1]), "rare_f1_round_20": float(r[19])}
        m = out["methods"][method]
        print(f"{method:<15} VL trust: r15 {t[14,1]:.2f} -> min {m['min_trust_6_after_turn']:.2f} "
              f"(round {m['round_of_min']}) -> r30 {m['trust_6_round_30']:.2f} | rare F1 r15 {r[14]:.3f} r20 {r[19]:.3f}")
    (ROOT / "results" / "followup_ledger_trajectory.json").write_text(json.dumps(out))
    print("wrote results/followup_ledger_trajectory.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
