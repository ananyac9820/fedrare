#!/usr/bin/env python3
"""
Gate G0a - does the evidence signal e(k,c) track which hospitals actually hold each disease?
(EARN Project Design v2, Section 6.6. Checked first, because every later step depends on it.)

PASS condition (fixed in the design doc before any result): with the frozen backbone,
evidence e(k,c) ranks hospitals' true per-class counts with Spearman correlation >= 0.7 on
the natural split S1.

How it is measured - fixed here before the first run, and identical to what EARN will see:
  - Round 1 of the Tier A loop (src/federated/tier_a.py): the head is initialised exactly as
    run_federated() does for this seed, and every centre runs the same local training
    (TierAConfig defaults) from it.
  - U[k] = that centre's update, clipped to the median update norm of the round.
  - e(k,c) = L2 norm of U[k]'s change to head weight row c (src/federated/primitives.py).
  - For each class c: Spearman correlation, across the 6 centres, between e(.,c) and the true
    number of training images of class c each centre holds. A class where it is undefined
    (a constant column) counts as 0.
  - Gate statistic = the mean of those 8 per-class correlations. PASS if >= 0.7.

Reported alongside, NOT part of the gate: the per-class correlations, and how strongly
evidence tracks centre size alone. Bigger centres hold more of almost every class, so a high
correlation could come from size rather than from class content - worth knowing before
building on a PASS.

If the gate fails, the design doc's retry (bias row, or evidence averaged over 3 rounds) is a
team decision, not something this script does on its own.

    python scripts/06_gate_g0a.py          # needs scripts/05_extract_features.py output
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

from src.data.features import load_features
from src.federated.interface import flatten, state_delta
from src.federated.primitives import FlatLayout, clip_to_median_norm, head_evidence
from src.federated.tier_a import TierAConfig, TierAHead, local_train, make_clients
from src.utils.config import load_config
from src.utils.seed import set_seed

# --- Pre-registered. Do not change after seeing the result. ---
GATE_THRESHOLD = 0.7          # design doc Section 6.6
SEED = 42                     # configs/default.yaml seed
ROUND = 1


def load_s02():
    spec = importlib.util.spec_from_file_location("s02", ROOT / "scripts" / "02_explore_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    rho = spearmanr(a, b).statistic
    return 0.0 if np.isnan(rho) else float(rho)


def main() -> int:
    cfg_yaml = load_config(ROOT / "configs" / "default.yaml")
    n_centers, n_classes = cfg_yaml["dataset"]["n_centers"], cfg_yaml["dataset"]["n_classes"]
    names = [cfg_yaml["dataset"]["class_names"][c] for c in range(n_classes)]
    out_dir = ROOT / cfg_yaml.get("output_dir", "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = TierAConfig(n_classes=n_classes)

    train, test = load_features("train"), load_features("test")
    # Rare classes by script 02's rule, on train + test counts exactly as script 02 does.
    s02 = load_s02()
    both = pd.DataFrame(train.counts(n_centers, n_classes) + test.counts(n_centers, n_classes),
                        index=[f"centre_{k}" for k in range(n_centers)], columns=names)
    rare_ids, _ = s02.identify_rare(both, cfg_yaml["dataset"]["head_ratio_divisor"])

    counts = train.counts(n_centers, n_classes)          # what each centre trains on
    clients = make_clients(train.features, train.labels, train.centers, n_centers)
    sizes = np.array([c.size for c in clients])

    # Round 1, exactly as run_federated() does it for this seed.
    set_seed(SEED)
    global_state = {k: v.detach().clone()
                    for k, v in TierAHead(cfg.feature_dim, n_classes).state_dict().items()}
    states = [local_train(global_state, c.x, c.y, cfg, np.random.default_rng([SEED, ROUND, k]))
              for k, c in enumerate(clients)]
    layout = FlatLayout.from_state(global_state)
    updates = torch.stack([flatten(state_delta(s, global_state, layout.keys), layout.keys)
                           for s in states])
    clipped, scale = clip_to_median_norm(updates)
    evidence = head_evidence(clipped, layout)

    per_class = np.array([spearman(evidence[:, c], counts[:, c]) for c in range(n_classes)])
    size_rho = np.array([spearman(evidence[:, c], sizes) for c in range(n_classes)])
    statistic = float(per_class.mean())
    passed = statistic >= GATE_THRESHOLD

    # ---------------------------------------------------------------- console
    bar = "=" * 78
    print(bar)
    print("GATE G0a - evidence e(k,c) vs true per-class training counts, natural split S1")
    print(bar)
    print(f"  Tier A config: {cfg}")
    print(f"  seed {SEED}, round {ROUND}; clip scale per centre {np.round(scale, 3).tolist()}")
    print(f"\n  {'class':<26}{'Spearman':>10}{'  vs size':>10}   evidence by centre 0..5  |  counts")
    for c in range(n_classes):
        tag = "  (rare)" if c in rare_ids else ""
        print(f"  {names[c] + tag:<26}{per_class[c]:>10.3f}{size_rho[c]:>10.3f}   "
              f"{' '.join(f'{v:6.3f}' for v in evidence[:, c])}  |  "
              f"{' '.join(f'{int(v):>4}' for v in counts[:, c])}")
    print(f"\n  Gate statistic (mean over {n_classes} classes): {statistic:.3f}   "
          f"threshold {GATE_THRESHOLD}")
    print(f"  Rare classes only (reported, not the gate): "
          + ", ".join(f"{names[c]} {per_class[c]:.3f}" for c in rare_ids))
    print(f"  Evidence vs centre size (reported, not the gate): mean {size_rho.mean():.3f}")
    print("\n" + bar)
    print(f"G0a: {'PASS' if passed else 'FAIL'}  (mean Spearman {statistic:.3f} "
          f"{'>=' if passed else '<'} {GATE_THRESHOLD})")
    print(bar)
    if not passed:
        print("  STOP: per the design doc, do not build further on this signal. The retry "
              "(bias row, or evidence averaged over 3 rounds) is a team decision.")

    # ---------------------------------------------------------------- record
    record = {
        "gate": "G0a", "passed": passed, "statistic": statistic, "threshold": GATE_THRESHOLD,
        "definition": "mean over classes of Spearman(e(.,c), train counts(.,c)) across the 6 "
                      "centres; round 1 of the Tier A loop; updates clipped to median norm; "
                      "e = L2 norm of head weight row update",
        "seed": SEED, "round": ROUND, "tier_a_config": cfg.__dict__,
        "rare_ids": rare_ids, "class_names": names,
        "per_class_spearman": per_class.tolist(),
        "per_class_spearman_vs_size": size_rho.tolist(),
        "evidence": evidence.tolist(), "train_counts": counts.tolist(),
        "centre_sizes": sizes.tolist(), "clip_scale": scale.tolist(),
    }
    path = out_dir / "gate_g0a.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\n  wrote {path}")
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
