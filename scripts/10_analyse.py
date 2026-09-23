#!/usr/bin/env python3
"""
Analyse the grid (scripts/09_run_grid.py) exactly as pre-registered in docs/DEVIATIONS.md D6.

    Gate G1 (per split, Camp A under A1), the S1/S2 framing decision (design doc 6.2),
    the Fallback F1 tables, fairness to the specialist centre 2, the exploratory
    oracle-evidence EARN tables and G2-oracle, aggregation overhead per round, and figures.

Outputs:
    results/analysis.json        everything below, machine-readable (the website reads it)
    results/analysis.md          the same as readable tables
    results/figures/*.png        figures for the paper

    python scripts/10_analyse.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.experiments import grid

RES = ROOT / "results"
FIG = RES / "figures"
NAMES = {5: "Dermatofibroma", 6: "Vascular lesion"}
METHOD_LABEL = {
    "fedavg": "FedAvg", "fedavg_clipped": "FedAvg + clipping", "krum": "Krum",
    "multi_krum": "Multi-Krum", "trimmed_mean": "Trimmed mean",
    "coordinate_wise_median": "Median", "camp_a": "Camp A (update evidence)",
    "camp_a_reported": "Camp A (reported counts)", "earn": "EARN (oracle)",
    "earn_no_blend": "EARN - no blend", "earn_no_ramp": "EARN - no slow ramp",
    "earn_no_ledger": "EARN - no ledger", "earn_step0.05": "EARN step 0.05",
    "earn_step0.2": "EARN step 0.2", "earn_tau10": "EARN holder >= 10",
    "earn_tau40": "EARN holder >= 40", "earn_real_signal": "EARN (real signal)",
}
# G1 thresholds - design doc Section 6.6, fixed before any result
G1_RATIO, G1_DF1, G1_DBA = 2.0, 0.15, 0.03
# G2 wording - design doc Section 6.6 (evaluated on oracle evidence only; not the gate)
G2_ATTACK_TOL, G2_CAMP_A_TOL, G2_TRUST = 0.05, 0.02, 0.8


def ms(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {"mean": float(x.mean()), "sd": float(x.std()), "n": int(len(x))}


def fmt(d: dict, nd: int = 3) -> str:
    return f"{d['mean']:.{nd}f} ± {d['sd']:.{nd}f}"


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def cell(runs, split, method, attack, col):
    sel = runs[(runs.split == split) & (runs.method == method) & (runs.attack == attack)]
    return ms(sel[col]) if len(sel) else None


# ------------------------------------------------------------------ G1

def gate_g1(runs: pd.DataFrame, method: str = "camp_a") -> dict:
    out = {}
    for split in grid.SPLITS:
        per_class = {}
        base_ba = cell(runs, split, method, "none", "balanced_accuracy")["mean"]
        att_ba = cell(runs, split, method, "A1", "balanced_accuracy")["mean"]
        d_ba = base_ba - att_ba
        for c in grid.RARE_IDS:
            w = cell(runs, split, method, "A1", f"attacker_weight_{c}")["mean"]
            fed = cell(runs, split, method, "A1", "attacker_fedavg_weight")["mean"]
            f1_none = cell(runs, split, method, "none", f"f1_{c}")["mean"]
            f1_att = cell(runs, split, method, "A1", f"f1_{c}")["mean"]
            ratio, d_f1 = w / fed, f1_none - f1_att
            per_class[NAMES[c]] = {
                "attacker_weight": w, "attacker_fedavg_weight": fed, "ratio": ratio,
                "f1_no_attack": f1_none, "f1_under_A1": f1_att, "f1_drop": d_f1,
                "weight_condition": ratio >= G1_RATIO, "f1_condition": d_f1 >= G1_DF1}
        passed = d_ba < G1_DBA and any(v["weight_condition"] and v["f1_condition"]
                                       for v in per_class.values())
        out[split] = {"passed": bool(passed), "balanced_accuracy_drop": d_ba,
                      "balanced_accuracy_condition": d_ba < G1_DBA, "per_class": per_class}
    return out


def framing(g1: dict) -> dict:
    s1, s2 = g1["s1"]["passed"], g1["s2"]["passed"]
    if s1:
        text = ("The attack succeeds on S1 (natural split): primary claim on real hospital data; "
                "S2 becomes the stress test.")
    elif s2:
        text = ("The attack succeeds only on S2: narrower but valid. The vulnerability requires "
                "coverage of 1-2; S2 is the primary setting with the specialist-clinic "
                "justification, and the S1 null result is reported in the same table.")
    else:
        text = ("The attack succeeds on neither split: Fallback F1 - an empirical study of how "
                "rare-class-aware aggregation behaves on a real hospital split.")
    return {"s1": s1, "s2": s2, "decision": text}


# ------------------------------------------------------------------ tables

def method_table(runs, methods, split) -> list[dict]:
    rows = []
    for m in methods:
        for a in grid.ATTACKS:
            sel = runs[(runs.split == split) & (runs.method == m) & (runs.attack == a)]
            if not len(sel):
                continue
            r = {"method": m, "label": METHOD_LABEL.get(m, m), "attack": a,
                 "balanced_accuracy": ms(sel.balanced_accuracy), "rare_macro_f1": ms(sel.rare_macro_f1),
                 "macro_f1": ms(sel.macro_f1),
                 **{f"f1_{c}": ms(sel[f"f1_{c}"]) for c in grid.RARE_IDS},
                 "to_target_rare": ms(sel.to_target_rare),
                 **{f"specialist_weight_{c}": ms(sel[f"specialist_weight_{c}"]) for c in grid.RARE_IDS}}
            if a != "none":
                for c in grid.RARE_IDS:
                    r[f"attacker_ratio_{c}"] = ms(sel[f"attacker_weight_{c}"] / sel.attacker_fedavg_weight)
            for c in grid.RARE_IDS:
                for key in (f"specialist_trust_{c}_r15", f"specialist_trust_{c}_final",
                            f"attacker_trust_{c}_max"):
                    if key in sel and sel[key].notna().all():
                        r[key] = ms(sel[key])
            rows.append(r)
    return rows


def rare_shares(ctx) -> dict:
    out = {}
    for split in grid.SPLITS:
        cnt = ctx.counts[split]
        sizes = cnt.sum(axis=1)
        out[split] = {"size_share": (sizes / sizes.sum()).tolist(),
                      "rare_share": {NAMES[c]: (cnt[:, c] / cnt[:, c].sum()).tolist() for c in grid.RARE_IDS},
                      "coverage_ge20": {NAMES[c]: int((cnt[:, c] >= 20).sum()) for c in grid.RARE_IDS},
                      "counts": cnt.tolist()}
    return out


def g2_oracle(runs, split) -> dict:
    e = lambda a, col: cell(runs, split, "earn", a, col)["mean"]  # noqa: E731
    base = e("none", "rare_macro_f1")
    checks = {
        "A1_within_0.05": abs(e("A1", "rare_macro_f1") - base) <= G2_ATTACK_TOL,
        "A2_within_0.05": abs(e("A2", "rare_macro_f1") - base) <= G2_ATTACK_TOL,
        "within_0.02_of_camp_a_reported": abs(base - cell(runs, split, "camp_a_reported", "none",
                                                          "rare_macro_f1")["mean"]) <= G2_CAMP_A_TOL,
        "above_fedavg": base > cell(runs, split, "fedavg", "none", "rare_macro_f1")["mean"],
        "specialist_trust_r15_ge_0.8": all(e("none", f"specialist_trust_{c}_r15") >= G2_TRUST
                                           for c in grid.RARE_IDS),
    }
    return {"passed": all(checks.values()), "checks": checks,
            "values": {"earn_none": base, "earn_A1": e("A1", "rare_macro_f1"),
                       "earn_A2": e("A2", "rare_macro_f1"),
                       "camp_a_reported_none": cell(runs, split, "camp_a_reported", "none", "rare_macro_f1")["mean"],
                       "fedavg_none": cell(runs, split, "fedavg", "none", "rare_macro_f1")["mean"],
                       **{f"specialist_trust_{c}_r15": e("none", f"specialist_trust_{c}_r15")
                          for c in grid.RARE_IDS}}}


# ------------------------------------------------------------------ overhead

def overhead(ctx, rounds: int = 20) -> dict:
    """Milliseconds per aggregation call on real Tier A client states (S1, no attack, seed 42)."""
    from src.federated.tier_a import run_federated
    out = {}
    for method in ("fedavg", "fedavg_clipped", "krum", "trimmed_mean", "camp_a",
                   "camp_a_reported", "earn"):
        factory = grid.aggregator_factory(method, ctx.counts["s1"], "none")
        times = []

        def timed_factory(_f=factory):
            agg = _f()

            def wrapped(*a, **k):
                t = time.perf_counter()
                res = agg(*a, **k)
                times.append((time.perf_counter() - t) * 1000)
                return res
            return wrapped
        cfg = ctx.cfg.__class__(**{**ctx.cfg.__dict__, "rounds": rounds})
        run_federated(timed_factory, ctx.clients["s1"], ctx.test_x, ctx.test_y, cfg, 42, grid.RARE_IDS)
        out[method] = ms(times[1:])      # first call includes warm-up
    return out


# ------------------------------------------------------------------ figures

def figures(runs, rounds_df, weights_df, shares):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    colours = {"none": "#2c6a64", "A1": "#b5543c", "A2": "#c9962b", "A3": "#5b6f95"}

    # 1. rare macro-F1 by method and attack, per split
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, split in zip(axes, grid.SPLITS):
        methods = list(grid.F1_METHODS) + ["earn"]
        x = np.arange(len(methods))
        for i, a in enumerate(grid.ATTACKS):
            vals = [cell(runs, split, m, a, "rare_macro_f1") for m in methods]
            ax.bar(x + (i - 1.5) * 0.2, [v["mean"] for v in vals], 0.2,
                   yerr=[v["sd"] for v in vals], label=a, color=colours[a], capsize=2)
        ax.set_xticks(x, [METHOD_LABEL[m].replace(" (", "\n(") for m in methods], rotation=45,
                      ha="right", fontsize=8)
        ax.set_title(f"{split.upper()}: rare-class macro-F1 (3 seeds)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("rare macro-F1")
    axes[0].legend(title="attack", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "rare_f1_by_method_attack.png", dpi=160)
    plt.close(fig)

    # 2. attacker's captured weight on rare rows under A1, ratio to its FedAvg weight
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, split in zip(axes, grid.SPLITS):
        methods = list(grid.F1_METHODS) + ["earn"]
        x = np.arange(len(methods))
        for j, c in enumerate(grid.RARE_IDS):
            sel = [runs[(runs.split == split) & (runs.method == m) & (runs.attack == "A1")] for m in methods]
            vals = [(s[f"attacker_weight_{c}"] / s.attacker_fedavg_weight).mean() for s in sel]
            ax.bar(x + (j - 0.5) * 0.35, vals, 0.35, label=NAMES[c])
        ax.axhline(G1_RATIO, color="k", ls="--", lw=1, label="G1 bar (2x)")
        ax.set_xticks(x, [METHOD_LABEL[m].replace(" (", "\n(") for m in methods], rotation=45,
                      ha="right", fontsize=8)
        ax.set_title(f"{split.upper()}: attacker (centre 4) weight on rare row / FedAvg weight, A1")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "attacker_weight_ratio_A1.png", dpi=160)
    plt.close(fig)

    # 3. specialist weight on rare rows vs its knowledge share (no attack)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, split in zip(axes, grid.SPLITS):
        methods = list(grid.F1_METHODS) + ["earn"]
        x = np.arange(len(methods))
        for j, c in enumerate(grid.RARE_IDS):
            vals = [cell(runs, split, m, "none", f"specialist_weight_{c}")["mean"] for m in methods]
            ax.bar(x + (j - 0.5) * 0.35, vals, 0.35, label=f"{NAMES[c]} weight")
            ax.axhline(shares[split]["rare_share"][NAMES[c]][grid.SPECIALIST], ls="--", lw=1,
                       color=f"C{j}", label=f"{NAMES[c]}: centre 2's share of the images")
        ax.set_xticks(x, [METHOD_LABEL[m].replace(" (", "\n(") for m in methods], rotation=45,
                      ha="right", fontsize=8)
        ax.set_title(f"{split.upper()}: centre 2's weight on rare rows (no attack)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "specialist_weight.png", dpi=160)
    plt.close(fig)

    # 4. EARN (oracle) trust of centre 2 and of the attacker over rounds
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, split in zip(axes, grid.SPLITS):
        for a, k, style in (("none", 2, "-"), ("A1", 4, "--"), ("A2", 1, ":")):
            sel = weights_df[(weights_df.split == split) & (weights_df.method == "earn")
                             & (weights_df.attack == a) & (weights_df.client == k)]
            for c in grid.RARE_IDS:
                curve = sel[sel["class"] == c].groupby("round")["trust"].mean()
                who = "centre 2 (honest)" if k == 2 else f"centre {k} ({a} attacker)"
                ax.plot(curve.index, curve.values, style, label=f"{who}, {NAMES[c]}")
        ax.axhline(G2_TRUST, color="k", lw=0.8)
        ax.axvline(15, color="grey", lw=0.8, ls=":")
        ax.set_title(f"{split.upper()}: EARN trust (oracle evidence, exploratory)")
        ax.set_xlabel("round")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("trust T(k, c)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "earn_trust_curves.png", dpi=160)
    plt.close(fig)

    # 5. learning curves, no attack
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, split in zip(axes, grid.SPLITS):
        for m in ("fedavg", "krum", "coordinate_wise_median", "camp_a", "camp_a_reported", "earn"):
            sel = rounds_df[(rounds_df.split == split) & (rounds_df.method == m) & (rounds_df.attack == "none")]
            curve = sel.groupby("round")["rare_macro_f1"].mean()
            ax.plot(curve.index, curve.values, label=METHOD_LABEL[m])
        ax.set_title(f"{split.upper()}: rare macro-F1 per round (no attack)")
        ax.set_xlabel("round")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "learning_curves.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ main

def main() -> int:
    runs = pd.read_csv(RES / "grid" / "runs.csv")
    rounds_df = pd.read_csv(RES / "grid" / "rounds.csv.gz")
    weights_df = pd.read_csv(RES / "grid" / "weights.csv.gz")
    expected = len(grid.SPLITS) * len(grid.F1_METHODS + grid.EARN_METHODS) * len(grid.ATTACKS) * len(grid.SEEDS)
    if len(runs) != expected:
        print(f"WARNING: {len(runs)} runs, expected {expected}")
    ctx = grid.load_context()
    shares = rare_shares(ctx)

    g1 = gate_g1(runs, "camp_a")
    g1_reported = gate_g1(runs, "camp_a_reported")
    frame = framing(g1)
    f1_tables = {s: method_table(runs, grid.F1_METHODS, s) for s in grid.SPLITS}
    earn_tables = {s: method_table(runs, ("fedavg", "camp_a_reported") + grid.EARN_METHODS, s)
                   for s in grid.SPLITS}
    g2 = {s: g2_oracle(runs, s) for s in grid.SPLITS}
    print("timing aggregation overhead...", flush=True)
    over = overhead(ctx)
    figures(runs, rounds_df, weights_df, shares)
    ledger_path = RES / "ledger" / "ledger_rounds.json"
    ledger = json.loads(ledger_path.read_text())["summary"] if ledger_path.exists() else None

    analysis = {"_meta": {"created": time.strftime("%Y-%m-%d %H:%M"), "runs": int(len(runs)),
                          "seeds": list(grid.SEEDS), "rounds": grid.ROUNDS, "features": grid.FEATURES,
                          "preregistration": "docs/DEVIATIONS.md D6"},
                "shares": shares, "gate_g1": g1, "gate_g1_camp_a_reported": g1_reported,
                "framing": frame, "f1_tables": f1_tables, "earn_tables": earn_tables,
                "g2_oracle": g2, "overhead_ms_per_round": over, "ledger": ledger}
    (RES / "analysis.json").write_text(json.dumps(analysis, indent=2, default=float))

    # ---------------------------------------------------------------- markdown
    md = ["# Results - Fallback F1 study and exploratory EARN (oracle evidence)", "",
          f"{len(runs)} runs; Tier A on ft4 features, {grid.ROUNDS} rounds, seeds "
          f"{', '.join(map(str, grid.SEEDS))}; mean ± sd over seeds. Pre-registered in "
          "docs/DEVIATIONS.md D6.", "", "## Gate G1 (Camp A = update evidence, attack A1)", ""]
    rows = []
    for s in grid.SPLITS:
        for name, v in g1[s]["per_class"].items():
            rows.append([s.upper(), name, f"{v['ratio']:.2f}x", f"{v['f1_drop']:+.3f}",
                         f"{g1[s]['balanced_accuracy_drop']:+.3f}",
                         "PASS" if g1[s]["passed"] else "fail"])
    md += [md_table(["split", "rare class", "attacker weight / FedAvg", "F1 drop",
                     "balanced-acc drop", "G1 on split"], rows), "",
           f"**Framing (design doc 6.2):** {frame['decision']}", "",
           "Same numbers for Camp A on reported counts (not gating):", ""]
    rows = []
    for s in grid.SPLITS:
        for name, v in g1_reported[s]["per_class"].items():
            rows.append([s.upper(), name, f"{v['ratio']:.2f}x", f"{v['f1_drop']:+.3f}",
                         f"{g1_reported[s]['balanced_accuracy_drop']:+.3f}",
                         "would pass" if g1_reported[s]["passed"] else "would fail"])
    md += [md_table(["split", "rare class", "attacker weight / FedAvg", "F1 drop",
                     "balanced-acc drop", "G1 criteria"], rows), ""]
    for s in grid.SPLITS:
        md += [f"## Fallback F1 - {s.upper()}", ""]
        rows = [[r["label"], r["attack"], fmt(r["balanced_accuracy"]), fmt(r["rare_macro_f1"]),
                 fmt(r["f1_5"]), fmt(r["f1_6"]), fmt(r["to_target_rare"]),
                 f"{r['specialist_weight_5']['mean']:.3f} / {r['specialist_weight_6']['mean']:.3f}",
                 (f"{r['attacker_ratio_5']['mean']:.2f}x / {r['attacker_ratio_6']['mean']:.2f}x"
                  if "attacker_ratio_5" in r else "-")] for r in f1_tables[s]]
        md += [md_table(["method", "attack", "balanced acc", "rare macro-F1", "F1 DF", "F1 VL",
                         "rare -> nevus", "centre 2 weight DF / VL", "attacker ratio DF / VL"], rows), ""]
    for s in grid.SPLITS:
        md += [f"## Exploratory EARN, oracle evidence - {s.upper()} (NOT a validated result)", ""]
        rows = [[r["label"], r["attack"], fmt(r["balanced_accuracy"]), fmt(r["rare_macro_f1"]),
                 fmt(r["to_target_rare"]),
                 (f"{r['attacker_ratio_5']['mean']:.2f}x / {r['attacker_ratio_6']['mean']:.2f}x"
                  if "attacker_ratio_5" in r else "-"),
                 (f"{r['specialist_trust_5_r15']['mean']:.2f} / {r['specialist_trust_6_r15']['mean']:.2f}"
                  if "specialist_trust_5_r15" in r else "-")] for r in earn_tables[s]]
        md += [md_table(["method", "attack", "balanced acc", "rare macro-F1", "rare -> nevus",
                         "attacker ratio DF / VL", "centre 2 trust @15 DF / VL"], rows), "",
               f"G2-oracle on {s.upper()}: **{'all conditions met' if g2[s]['passed'] else 'not met'}** - "
               + ", ".join(f"{k}: {'yes' if v else 'no'}" for k, v in g2[s]["checks"].items()), ""]
    md += ["## Aggregation overhead (ms per round, S1, 20 rounds)", "",
           md_table(["method", "ms / round"], [[METHOD_LABEL[m], fmt(v, 2)] for m, v in over.items()]), ""]
    if ledger:
        md += ["## Ledger (local Hardhat chain)", "",
               f"{ledger['rounds_committed']} EARN rounds from {ledger['runs']} runs committed; gas per "
               f"round mean {ledger['gas_per_round']['mean']:.0f} (max {ledger['gas_per_round']['max']}); "
               f"latency mean {ledger['latency_ms_per_round']['mean']:.2f} ms; deploy gas "
               f"{ledger['deploy_gas']}; final on-chain trust matches Python in every run: "
               f"{ledger['all_final_trust_match']}; forged boost rejected in every run: "
               f"{ledger['all_tamper_rejected']}.", ""]
    (RES / "analysis.md").write_text("\n".join(md))
    print(f"wrote {RES / 'analysis.json'}, {RES / 'analysis.md'}, figures in {FIG}")
    print(f"G1: S1 {'PASS' if g1['s1']['passed'] else 'fail'}, S2 {'PASS' if g1['s2']['passed'] else 'fail'}"
          f" -> {frame['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
