"""
The experiment grid pre-registered in docs/DEVIATIONS.md D6: every (split, method, attack, seed)
run of the Fallback F1 study and of the exploratory oracle-evidence EARN analysis.

    rows = run_one(ctx, split="s1", method="camp_a", attack="A1", seed=42)

Output rows follow docs/results_format.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from functools import partial

import numpy as np
import torch

from src.attacks.attacks import DEFAULT_ATTACKS, TARGET_CLASS
from src.data.features import load_features
from src.data.splits import HOLDER_MIN_IMAGES, class_counts, make_split
from src.federated import baselines
from src.federated.earn import EARN, EARNConfig
from src.federated.tier_a import TierAConfig, make_clients, run_federated

N_CENTERS, N_CLASSES = 6, 8
RARE_IDS = [5, 6]
SEEDS = (42, 43, 44)
SPLITS = ("s1", "s2")
ROUNDS = 100                     # D4
FEATURES = "ft4"                 # D3
SPECIALIST = 2
ATTACKS = ("none", "A1", "A2", "A3")

F1_METHODS = ("fedavg", "fedavg_clipped", "krum", "multi_krum", "trimmed_mean",
              "coordinate_wise_median", "camp_a", "camp_a_reported")
EARN_METHODS = ("earn", "earn_no_blend", "earn_no_ramp", "earn_no_ledger", "earn_step0.05",
                "earn_step0.2", "earn_tau10", "earn_tau40", "earn_real_signal")
ORACLE_TAU = HOLDER_MIN_IMAGES - 0.5


@dataclass
class Context:
    """Features and per-split clients, loaded once per worker process."""
    train: object
    test_x: torch.Tensor
    test_y: np.ndarray
    clients: dict
    counts: dict            # split -> (K, C) true training counts
    cfg: TierAConfig


def load_context(rounds: int = ROUNDS, variant: str = FEATURES) -> Context:
    train, test = load_features("train", variant=variant), load_features("test", variant=variant)
    clients, counts = {}, {}
    for split in SPLITS:
        centers = make_split(split, train.centers, train.labels, RARE_IDS)
        clients[split] = make_clients(train.features, train.labels, centers, N_CENTERS)
        counts[split] = class_counts(centers, train.labels, N_CENTERS, N_CLASSES)
    return Context(train, torch.from_numpy(test.features), test.labels, clients, counts,
                   replace(TierAConfig(), rounds=rounds))


def reported_counts_fn(counts: np.ndarray, attack):
    """Evidence = reported class counts; a claiming attacker reports the top honest count."""
    spec = DEFAULT_ATTACKS[attack]
    hook = spec.hooks(RARE_IDS).get(spec.attacker) if spec.attacker is not None else None
    honest_top = counts.max(axis=0)

    def fn(updates, layout, round_num):
        ev = counts.astype(np.float64).copy()
        if hook is not None and getattr(hook, "claims_holder", False) and hook.active(round_num):
            for c in RARE_IDS:
                ev[spec.attacker, c] = honest_top[c]
        return ev
    return fn


def aggregator_factory(method: str, counts: np.ndarray, attack: str):
    """Zero-argument factory for run_federated."""
    if method in baselines.RULES:
        return lambda: baselines.RULES[method]
    oracle = reported_counts_fn(counts, attack)
    if method == "camp_a_reported":
        return lambda: partial(baselines.camp_a, evidence_fn=oracle, rule_name="camp_a_reported")
    spec = DEFAULT_ATTACKS[attack]
    base = EARNConfig(tau=ORACLE_TAU)
    variants = {
        "earn": base,
        "earn_no_blend": replace(base, blend=False),
        "earn_no_ramp": replace(base, slow_ramp=False),
        "earn_no_ledger": replace(base, ledger=False,
                                  editable_history=() if spec.attacker is None else (spec.attacker,)),
        "earn_step0.05": replace(base, step=0.05),
        "earn_step0.2": replace(base, step=0.2),
        "earn_tau10": replace(base, tau=10 - 0.5),
        "earn_tau40": replace(base, tau=40 - 0.5),
    }
    if method == "earn_real_signal":
        return lambda: EARN(EARNConfig(tau="mean"), evidence_kind="weight", name=method)
    if method not in variants:
        raise ValueError(f"unknown method {method!r}")
    cfg = variants[method]
    return lambda: EARN(cfg, evidence_fn=oracle, name=method)


def attack_success(confusion: list, target: int = TARGET_CLASS) -> dict:
    cm = np.asarray(confusion)
    out = {f"to_target_{c}": float(cm[c, target] / max(cm[c].sum(), 1)) for c in RARE_IDS}
    out["to_target_rare"] = float(cm[RARE_IDS, target].sum() / max(cm[RARE_IDS].sum(), 1))
    return out


def run_one(ctx: Context, split: str, method: str, attack: str, seed: int) -> dict:
    """One run. Returns {"run": row, "rounds": [...], "weights": [...]} (results_format.md)."""
    spec = DEFAULT_ATTACKS[attack]
    counts = ctx.counts[split]
    clients = ctx.clients[split]
    t0 = time.time()
    res = run_federated(aggregator_factory(method, counts, attack), clients, ctx.test_x,
                        ctx.test_y, ctx.cfg, seed, RARE_IDS, client_hooks=spec.hooks(RARE_IDS))
    seconds = time.time() - t0
    infos, metrics = res["round_infos"], res["round_metrics"]
    sizes = np.array([c.size for c in clients], dtype=np.float64)
    size_share = sizes / sizes.sum()
    hook = spec.hooks(RARE_IDS).get(spec.attacker) if spec.attacker is not None else None
    active = [r for r in range(1, ctx.cfg.rounds + 1) if hook is None or hook.active(r)]
    hw = np.stack([i["head_row_weights"] for i in infos])            # (R, K, C)
    final = res["final"]

    run = {"split": split, "method": method, "attack": attack, "seed": seed,
           "attacker": -1 if spec.attacker is None else spec.attacker,
           "rounds": ctx.cfg.rounds, "seconds": round(seconds, 2),
           "balanced_accuracy": final["balanced_accuracy"], "accuracy": final["accuracy"],
           "macro_f1": final["macro_f1"], "rare_macro_f1": final["rare_macro_f1"],
           **{f"f1_{c}": final["per_class_f1"][c] for c in range(N_CLASSES)},
           **attack_success(final["confusion"])}
    for c in RARE_IDS:
        run[f"specialist_weight_{c}"] = float(hw[:, SPECIALIST, c].mean())
        run[f"specialist_fedavg_weight"] = float(size_share[SPECIALIST])
        if spec.attacker is not None:
            idx = np.asarray(active) - 1
            run[f"attacker_weight_{c}"] = float(hw[idx, spec.attacker, c].mean())
            run["attacker_fedavg_weight"] = float(size_share[spec.attacker])
    if "trust" in infos[-1]:
        for c in RARE_IDS:
            run[f"specialist_trust_{c}_r15"] = float(infos[min(14, len(infos) - 1)]["trust"][SPECIALIST, c])
            run[f"specialist_trust_{c}_final"] = float(infos[-1]["trust"][SPECIALIST, c])
            if spec.attacker is not None:
                run[f"attacker_trust_{c}_final"] = float(infos[-1]["trust"][spec.attacker, c])
                run[f"attacker_trust_{c}_max"] = float(max(i["trust"][spec.attacker, c] for i in infos))

    key = {"split": split, "method": method, "attack": attack, "seed": seed}
    rounds = [{**key, "round": r, "balanced_accuracy": m["balanced_accuracy"],
               "rare_macro_f1": m["rare_macro_f1"], "f1_5": m["per_class_f1"][5],
               "f1_6": m["per_class_f1"][6], **attack_success(m["confusion"])}
              for r, m in enumerate(metrics, 1)]
    weights = []
    for r, info in enumerate(infos, 1):
        for k in range(len(clients)):
            for c in RARE_IDS:
                row = {**key, "round": r, "client": k, "class": c,
                       "weight": float(info["head_row_weights"][k, c])}
                if "evidence" in info:
                    row["evidence"] = float(info["evidence"][k, c])
                if "trust" in info:
                    row["trust"] = float(info["trust"][k, c])
                    row["holder"] = bool(info["holders"][k, c])
                    row["coverage"] = int(info["coverage"][c])
                weights.append(row)
    out = {"run": run, "rounds": rounds, "weights": weights}
    if method == "earn" and "block_hash" in infos[-1]:
        out["ledger"] = [{"round": r, "trust": i["trust"].tolist(),
                          "coverage": i["coverage"].tolist(), "history_hash": i["history_hash"],
                          "block_hash": i["block_hash"]} for r, i in enumerate(infos, 1)]
    return out
