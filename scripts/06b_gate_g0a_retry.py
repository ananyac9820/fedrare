#!/usr/bin/env python3
"""
Gate G0a retry (docs/DEVIATIONS.md D1) and amendment M1 (D2). Pre-registered before first run.

G0a failed (scripts/06_gate_g0a.py: mean Spearman -0.210 against 0.7). The design doc allows one
retry. This script measures three evidence definitions under G0a's exact protocol - round 1 of
the Tier A loop, seed 42, TierAConfig defaults, updates clipped to the median norm, per class the
Spearman correlation across the 6 centres between evidence and true training counts, gate
statistic = mean over the 8 classes, pass at >= 0.7:

    retry  (D1, decides G0a)   e(k,c) = |change of bias c|
    avg3   (reported only)     weight-row evidence averaged over rounds 1-3 of FedAvg
    M1     (D2, amendment)     e(k,c) = max(0, change of bias c)   - sign-aware

Verdicts, fixed in D2 before running:
    retry passes                      -> G0a PASS
    retry fails, M1 passes            -> G0a FAIL (Fallback F1 by the design doc's letter);
                                         EARN continues on M1 as a labelled amended track
    both fail                         -> G0a FAIL, Fallback F1 only

Reported alongside, not part of any gate: holder-detection AUROC (does evidence separate
centres holding >= 20 images of a class from those that do not), and each definition's value
in rounds 2 and 3.

Output: results/gate_g0a_retry.json

    python scripts/06b_gate_g0a_retry.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.data.features import load_features
from src.data.splits import HOLDER_MIN_IMAGES
from src.federated import baselines
from src.federated.interface import flatten, state_delta
from src.federated.primitives import FlatLayout, clip_to_median_norm, head_evidence
from src.federated.tier_a import TierAConfig, TierAHead, local_train, make_clients
from src.utils.config import load_config
from src.utils.seed import set_seed

# --- Pre-registered. Do not change after seeing the result. ---
GATE_THRESHOLD = 0.7
SEED = 42
N_ROUNDS = 3                 # avg3 averages rounds 1..3; retry and M1 use round 1
RETRY_KIND = "bias"          # D1
AMENDMENT_KIND = "signed_bias"  # D2


def load_s02():
    spec = importlib.util.spec_from_file_location("s02", ROOT / "scripts" / "02_explore_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    rho = spearmanr(a, b).statistic
    return 0.0 if np.isnan(rho) else float(rho)


def gate_stat(evidence: np.ndarray, counts: np.ndarray) -> tuple[float, list[float]]:
    per_class = [spearman(evidence[:, c], counts[:, c]) for c in range(counts.shape[1])]
    return float(np.mean(per_class)), per_class


def holder_auroc(evidence: np.ndarray, counts: np.ndarray) -> float:
    holders = (counts >= HOLDER_MIN_IMAGES).ravel()
    return float(roc_auc_score(holders, evidence.ravel()))


def main() -> int:
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
    counts = train.counts(n_centers, n_classes)
    clients = make_clients(train.features, train.labels, train.centers, n_centers)
    sizes = [c.size for c in clients]

    # Rounds 1..N_ROUNDS exactly as run_federated() runs FedAvg for this seed.
    set_seed(SEED)
    global_state = {k: v.detach().clone()
                    for k, v in TierAHead(cfg.feature_dim, n_classes).state_dict().items()}
    layout = FlatLayout.from_state(global_state)
    per_round = {kind: [] for kind in ("weight", "bias", "signed_bias")}
    for r in range(1, N_ROUNDS + 1):
        states = [local_train(global_state, c.x, c.y, cfg, np.random.default_rng([SEED, r, k]))
                  for k, c in enumerate(clients)]
        updates = torch.stack([flatten(state_delta(s, global_state, layout.keys), layout.keys)
                               for s in states])
        clipped, _ = clip_to_median_norm(updates)
        for kind in per_round:
            per_round[kind].append(head_evidence(clipped, layout, kind))
        global_state, _ = baselines.fedavg(states, sizes, global_state, r)

    definitions = {
        "retry_bias_round1": ("D1 - decides G0a", per_round[RETRY_KIND][0]),
        "avg3_weight_rounds1to3": ("reported only", np.mean(per_round["weight"], axis=0)),
        "M1_signed_bias_round1": ("D2 - amendment", per_round[AMENDMENT_KIND][0]),
    }
    record = {"gate": "G0a-retry", "threshold": GATE_THRESHOLD, "seed": SEED,
              "tier_a_config": cfg.__dict__, "rare_ids": rare_ids, "class_names": names,
              "train_counts": counts.tolist(), "holder_min_images": HOLDER_MIN_IMAGES,
              "definitions": {}}

    bar = "=" * 78
    print(bar + "\nGATE G0a RETRY (D1) and AMENDMENT M1 (D2) - natural split S1\n" + bar)
    for key, (role, ev) in definitions.items():
        stat, per_class = gate_stat(ev, counts)
        auroc = holder_auroc(ev, counts)
        later = {}
        if key != "avg3_weight_rounds1to3":
            kind = RETRY_KIND if key.startswith("retry") else AMENDMENT_KIND
            later = {f"round_{r + 1}": gate_stat(per_round[kind][r], counts)[0]
                     for r in range(1, N_ROUNDS)}
        record["definitions"][key] = {
            "role": role, "passed": stat >= GATE_THRESHOLD, "statistic": stat,
            "per_class_spearman": per_class, "holder_auroc": auroc,
            "statistic_later_rounds": later, "evidence": ev.tolist()}
        print(f"\n  {key}  [{role}]")
        for c in range(n_classes):
            tag = " (rare)" if c in rare_ids else ""
            print(f"    {names[c] + tag:<30}{per_class[c]:>8.3f}   "
                  f"{' '.join(f'{v:8.5f}' for v in ev[:, c])}")
        print(f"    mean Spearman {stat:.3f} ({'PASS' if stat >= GATE_THRESHOLD else 'fail'}), "
              f"holder AUROC {auroc:.3f}"
              + (", later rounds " + ", ".join(f"{k} {v:.3f}" for k, v in later.items())
                 if later else ""))

    retry_pass = record["definitions"]["retry_bias_round1"]["passed"]
    m1_pass = record["definitions"]["M1_signed_bias_round1"]["passed"]
    if retry_pass:
        verdict, track = "PASS", "design-doc evidence (bias row)"
    elif m1_pass:
        verdict, track = "FAIL", "amended track on M1 (signed bias), labelled; Fallback F1 by the letter"
    else:
        verdict, track = "FAIL", "Fallback F1 only"
    record.update({"passed": retry_pass, "verdict": verdict, "amendment_M1_passed": m1_pass,
                   "next": track})
    print(f"\n{bar}\nG0a after retry: {verdict}   |   M1: {'PASS' if m1_pass else 'fail'}"
          f"\n  next: {track}\n{bar}")
    path = out_dir / "gate_g0a_retry.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
