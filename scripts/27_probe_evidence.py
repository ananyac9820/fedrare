#!/usr/bin/env python3
"""
Gate G0c: a probe-based evidence signal, after CELM (docs/DEVIATIONS.md D17).

G0a stays failed. This is a different signal from a different family with its own gate:
evidence obtained by probing each client's trained head, class by class, as CELM does -
rather than from the geometry of the update, which is what G0a measured and what R14 showed
tops out at 0.391.

On Tier A the head is linear, so "the input that maximises client k's logit for class c" has
an exact optimum on the unit ball, ||w_c|| + b_c. We compute that exactly and also run
CELM's literal gradient procedure as an agreement check.

Three things are measured for every candidate, all fixed in D17 before running:
  1. faithfulness - mean Spearman against true per-class counts, 3 seeds, bar 0.7
  2. size        - the same against centre size; a candidate correlating at least as well
                   with size as with counts is disqualified as a size proxy
  3. spoofability - where the A1 attacker (holds no rare images, inflates its rare rows)
                   ranks among the six centres; top-two on either rare class = unusable

    python scripts/27_probe_evidence.py

Writes docs/results/gate_g0c_probe.json.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from src.attacks.attacks import DEFAULT_ATTACKS
from src.data.features import load_features
from src.federated.tier_a import TierAConfig, TierAHead, local_train, make_clients
from src.utils.config import load_config
from src.utils.seed import set_seed

OUT = ROOT / "docs" / "results"
SEEDS = (42, 43, 44)
ROUND = 1
BAR = 0.7
RARE_IDS = [5, 6]
N_CENTRES = 6
PROBE_STEPS = 200          # CELM's literal procedure, for the agreement check


def spearman(a, b):
    rho = spearmanr(a, b).statistic
    return 0.0 if np.isnan(rho) else float(rho)


def client_heads(clients, cfg, seed, n_classes, hooks=None):
    """Each client's trained head after one round of local training, plus the update."""
    set_seed(seed)
    global_state = {k: v.detach().clone()
                    for k, v in TierAHead(cfg.feature_dim, n_classes).state_dict().items()}
    heads, updates = [], []
    for k, data in enumerate(clients):
        rng = np.random.default_rng([seed, ROUND, k])

        def honest_train(x=None, y=None, _d=data, _rng=rng):
            return local_train(global_state, _d.x if x is None else x,
                               _d.y if y is None else y, cfg, _rng)

        hook = (hooks or {}).get(k)
        state = (hook(round_num=ROUND, client_id=k, global_state=global_state, data=data,
                      honest_train=honest_train) if hook else honest_train())
        w = state["classifier.weight"].detach().numpy()
        b = state["classifier.bias"].detach().numpy()
        heads.append((w, b))
        updates.append((w - global_state["classifier.weight"].numpy(),
                        b - global_state["classifier.bias"].numpy()))
    return heads, updates, global_state


def p1(heads):
    """Exact probe optimum on the unit ball: max_x (w_c . x + b_c) = ||w_c|| + b_c."""
    return np.array([[np.linalg.norm(w[c]) + b[c] for c in range(len(b))] for w, b in heads])


def p2(heads):
    """Margin against the hardest competitor: min_j ( ||w_c - w_j|| + b_c - b_j )."""
    out = []
    for w, b in heads:
        C = len(b)
        row = []
        for c in range(C):
            row.append(min(float(np.linalg.norm(w[c] - w[j]) + b[c] - b[j])
                           for j in range(C) if j != c))
        out.append(row)
    return np.array(out)


def normalise_per_client(ev):
    denom = np.abs(ev).sum(axis=1, keepdims=True)
    return ev / np.where(denom > 0, denom, 1.0)


def p5_gradient_probe(heads, n_classes, steps=PROBE_STEPS):
    """CELM's literal procedure: optimise a probe per class per client model."""
    out = np.zeros((len(heads), n_classes))
    for k, (w, b) in enumerate(heads):
        W = torch.tensor(w, dtype=torch.float32)
        B = torch.tensor(b, dtype=torch.float32)
        for c in range(n_classes):
            torch.manual_seed(1000 + 10 * k + c)
            x = torch.randn(W.shape[1], requires_grad=True)
            opt = torch.optim.Adam([x], lr=0.05)
            for _ in range(steps):
                opt.zero_grad()
                unit = x / x.norm().clamp_min(1e-9)      # keep the probe on the unit ball
                (-(unit @ W[c] + B[c])).backward()
                opt.step()
            with torch.no_grad():
                unit = x / x.norm().clamp_min(1e-9)
                out[k, c] = float(unit @ W[c] + B[c])
    return out


def candidates(heads, updates, n_classes):
    ev = {}
    ev["P1 probe optimum"] = p1(heads)
    ev["P2 margin"] = p2(heads)
    ev["P3 P1 per-client normalised"] = normalise_per_client(p1(heads))
    ev["P4 P1 on the update"] = p1(updates)
    ev["P5 gradient probe"] = p5_gradient_probe(heads, n_classes)
    return ev


def main() -> int:
    cfg_yaml = load_config(ROOT / "configs" / "default.yaml")
    n_classes = cfg_yaml["dataset"]["n_classes"]
    names = [cfg_yaml["dataset"]["class_names"][c] for c in range(n_classes)]
    cfg = TierAConfig(n_classes=n_classes)

    train = load_features("train")
    counts = train.counts(N_CENTRES, n_classes)
    clients = make_clients(train.features, train.labels, train.centers, N_CENTRES)
    sizes = np.array([c.size for c in clients], dtype=float)

    per_seed = {}
    for seed in SEEDS:
        heads, updates, _ = client_heads(clients, cfg, seed, n_classes)
        per_seed[seed] = candidates(heads, updates, n_classes)
        print(f"  seed {seed} done", flush=True)

    results = {}
    for label in per_seed[SEEDS[0]]:
        stats, size_rhos, per_class = [], [], []
        for seed in SEEDS:
            ev = np.asarray(per_seed[seed][label], dtype=float)
            rhos = np.array([spearman(ev[:, c], counts[:, c]) for c in range(n_classes)])
            stats.append(float(rhos.mean()))
            size_rhos.append(float(np.mean([spearman(ev[:, c], sizes)
                                            for c in range(n_classes)])))
            per_class.append(rhos)
        mean, size_mean = float(np.mean(stats)), float(np.mean(size_rhos))
        results[label] = {
            "mean": mean, "sd": float(np.std(stats, ddof=1)), "per_seed": stats,
            "vs_size_mean": size_mean,
            "size_proxy": bool(abs(size_mean) >= abs(mean)),      # D17 disqualification rule
            "clears_bar": bool(mean >= BAR),
            "per_class_mean": {names[c]: float(np.mean([p[c] for p in per_class]))
                               for c in range(n_classes)},
        }

    # ---- spoofability: where does the A1 attacker rank? (D17, required) -----------------
    spec = DEFAULT_ATTACKS["A1"]
    hooks = spec.hooks(RARE_IDS)
    attacker = spec.attacker
    spoof = {}
    for label in results:
        ranks = {c: [] for c in RARE_IDS}
        for seed in SEEDS:
            heads, updates, _ = client_heads(clients, cfg, seed, n_classes, hooks=hooks)
            ev = np.asarray(candidates(heads, updates, n_classes)[label], dtype=float)
            for c in RARE_IDS:
                # rank 1 = highest evidence of holding this class
                order = rankdata(-ev[:, c], method="min")
                ranks[c].append(int(order[attacker]))
        best_rank = min(min(v) for v in ranks.values())
        spoof[label] = {
            "attacker": attacker,
            "attacker_true_rare_counts": [int(counts[attacker, c]) for c in RARE_IDS],
            "ranks_per_rare_class": {names[c]: ranks[c] for c in RARE_IDS},
            "best_rank_achieved": best_rank,
            "spoofable": bool(best_rank <= 2),                    # D17 condition
        }
        print(f"  spoof check {label}: best rank {best_rank} of 6", flush=True)

    print("\n" + "=" * 86)
    print(f"GATE G0c - probe-based evidence after CELM | {len(SEEDS)} seeds | bar {BAR}")
    print("G0a remains failed; this is a separate gate for a separate signal.")
    print("=" * 86)
    print(f"\n  {'candidate':<30}{'mean':>8}{'sd':>7}{'vs size':>9}  {'proxy?':>7} "
          f"{'A1 rank':>8}  verdict")
    for label, r in sorted(results.items(), key=lambda kv: -kv[1]["mean"]):
        s = spoof[label]
        verdict = ("PASSES" if (r["clears_bar"] and not r["size_proxy"]
                                and not s["spoofable"]) else
                   "unusable (spoofable)" if r["clears_bar"] and s["spoofable"] else
                   "unusable (size proxy)" if r["clears_bar"] and r["size_proxy"] else
                   "under the bar")
        print(f"  {label:<30}{r['mean']:>+8.3f}{r['sd']:>7.3f}{r['vs_size_mean']:>+9.3f}  "
              f"{'yes' if r['size_proxy'] else 'no':>7} {s['best_rank_achieved']:>8}  {verdict}")

    passing = [k for k, r in results.items()
               if r["clears_bar"] and not r["size_proxy"] and not spoof[k]["spoofable"]]
    record = {
        "gate": "G0c", "entry": "D17",
        "fixed_in": "docs/DEVIATIONS.md D17, before any candidate was computed",
        "not_a_rescoring_of": "G0a, which stays failed",
        "bar": BAR, "seeds": list(SEEDS), "round": ROUND,
        "candidates": results, "spoofability": spoof,
        "passing_candidates": passing, "passed": bool(passing),
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate_g0c_probe.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nG0c -> {'PASSED by ' + ', '.join(passing) if passing else 'NOT PASSED'}")
    print(f"wrote {(OUT / 'gate_g0c_probe.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
