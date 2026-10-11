#!/usr/bin/env python3
"""
Does rare-class F1 conceal attack damage? (docs/DEVIATIONS.md D13)

Analyses result files that already exist - no training runs. Three tests, scoped by what
the files actually hold:

    1  grid, 612 cases: does F1 move opposite to to_target under attack?
    2  Tier B, 16 cases: does F1 move opposite to true recall?
    3  Tier B, 16 cases: is disagreement where the honest baseline over-predicts?
       Declared descriptive in D13 before running - sixteen cells cannot carry a correlation.

Disagreement is dTT > eps and dF1 > eps: the damage metric says worse, F1 says better.
eps = 0.01, fixed in D13. Cases moving by eps or less on either metric are excluded from
the rate and counted separately.

    python scripts/22_f1_concealment.py

Writes docs/results/f1_concealment.json, docs/results/f1_concealment.md, and two plots
under docs/results/figures/.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = ROOT / "docs" / "results"
FIGS = OUT / "figures"
EPS = 0.01                       # D13
RARE = {5: "Dermatofibroma", 6: "Vascular lesion"}
TARGET = 1                       # nevus, the class the attacks push rare images into

INK, MUTED, FAINT = "#1d2929", "#56615f", "#8a918d"
ACCENT, WARN, CREAM, GRID = "#2c6a64", "#ae2330", "#faf7f0", "#e4dccb"


# ---------------------------------------------------------------- test 1: the grid
def grid_cases():
    rows = list(csv.DictReader(open(OUT / "runs.csv", encoding="utf-8")))
    honest = {(r["split"], r["method"], r["seed"]): r for r in rows if r["attack"] == "none"}
    cases = []
    for r in rows:
        if r["attack"] == "none":
            continue
        base = honest.get((r["split"], r["method"], r["seed"]))
        if base is None:
            continue
        for c in RARE:
            d_f1 = float(r[f"f1_{c}"]) - float(base[f"f1_{c}"])
            d_tt = float(r[f"to_target_{c}"]) - float(base[f"to_target_{c}"])
            cases.append({
                "source": "grid", "split": r["split"], "method": r["method"],
                "attack": r["attack"], "seed": int(r["seed"]), "class": RARE[c],
                "f1_honest": float(base[f"f1_{c}"]), "f1_attacked": float(r[f"f1_{c}"]),
                "tt_honest": float(base[f"to_target_{c}"]),
                "tt_attacked": float(r[f"to_target_{c}"]),
                "d_f1": d_f1, "d_damage": d_tt, "damage_metric": "to_target",
            })
    return cases


# ---------------------------------------------------------------- test 2 and 3: Tier B
def tier_b_cases():
    data = json.loads((OUT / "tier_b.json").read_text(encoding="utf-8"))
    runs = {f"{r['rule']}_{r['attack']}": r for r in data["runs"]}
    honest = runs["fedavg_none"]
    cases = []
    for name, run in runs.items():
        if run["attack"] == "none":
            continue
        for rnd in sorted(honest["metrics_by_round"], key=int):
            if rnd not in run["metrics_by_round"]:
                continue
            hm, am = honest["metrics_by_round"][rnd], run["metrics_by_round"][rnd]
            hc, ac = np.array(hm["confusion"]), np.array(am["confusion"])
            for c in RARE:
                def stats(cm):
                    support = cm[c].sum()
                    predicted = cm[:, c].sum()
                    return (cm[c, c] / support if support else 0.0,
                            cm[c, c] / predicted if predicted else 0.0,
                            cm[c, TARGET] / support if support else 0.0,
                            int(predicted), int(support))
                h_rec, h_prec, h_tt, h_pred, support = stats(hc)
                a_rec, a_prec, a_tt, a_pred, _ = stats(ac)
                cases.append({
                    "source": "tier_b", "run": name, "round": int(rnd), "class": RARE[c],
                    "support": support,
                    "f1_honest": hm["per_class_f1"][str(c)],
                    "f1_attacked": am["per_class_f1"][str(c)],
                    "recall_honest": h_rec, "recall_attacked": a_rec,
                    "precision_honest": h_prec, "precision_attacked": a_prec,
                    "predicted_honest": h_pred, "predicted_attacked": a_pred,
                    "tt_honest": h_tt, "tt_attacked": a_tt,
                    "d_f1": am["per_class_f1"][str(c)] - hm["per_class_f1"][str(c)],
                    "d_damage": h_rec - a_rec,          # positive = recall lost = damage
                    "d_recall": a_rec - h_rec,
                    "d_tt": a_tt - h_tt,
                    "damage_metric": "recall",
                })
    return cases


def classify(case):
    """D13: disagreement = the damage metric says worse while F1 says better."""
    d_f1, d_dmg = case["d_f1"], case["d_damage"]
    if abs(d_f1) <= EPS or abs(d_dmg) <= EPS:
        return "no clear movement"
    if d_dmg > 0 and d_f1 > 0:
        return "disagreement (damage up, F1 up)"
    if d_dmg < 0 and d_f1 < 0:
        return "converse (damage down, F1 down)"
    return "agreement"


def rate(cases):
    kinds = [classify(c) for c in cases]
    moved = [k for k in kinds if k != "no clear movement"]
    dis = sum(k.startswith("disagreement") for k in kinds)
    con = sum(k.startswith("converse") for k in kinds)
    return {
        "cases": len(cases), "with_clear_movement": len(moved),
        "excluded_no_movement": len(cases) - len(moved),
        "disagreements": dis,
        "disagreement_rate": dis / len(moved) if moved else 0.0,
        "converse": con, "converse_rate": con / len(moved) if moved else 0.0,
    }


SELECTION_RULES = {"krum", "multi_krum"}   # pick one client's update; the rest average


def wilson(k, n):
    """95% interval for a rate - small subsets need their error bars shown."""
    if not n:
        return [None, None]
    z, phat = 1.96, k / n
    d = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / d
    half = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [max(0.0, centre - half), min(1.0, centre + half)]


def family(cases):
    """The split that turned out to matter: selection rules against averaging rules."""
    out = {}
    for name, keep in (("selection (krum, multi_krum)", True),
                       ("averaging (all others)", False)):
        group = [c for c in cases if (c["method"] in SELECTION_RULES) == keep]
        r = rate(group)
        r["disagreement_rate_95ci"] = wilson(r["disagreements"], r["with_clear_movement"])
        out[name] = r
    return out


def breakdown(cases, key):
    out = {}
    groups = defaultdict(list)
    for c in cases:
        groups[c[key]].append(c)
    for name, group in sorted(groups.items()):
        out[str(name)] = rate(group)
    return out


# ---------------------------------------------------------------- plots
def plots(grid, tb):
    FIGS.mkdir(parents=True, exist_ok=True)

    # 1. dF1 against the damage metric, disagreement quadrant marked.
    fig, ax = plt.subplots(figsize=(7.6, 6.2), dpi=150)
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor("#fffdf9")
    x = [c["d_damage"] for c in grid]
    y = [c["d_f1"] for c in grid]
    colours = [WARN if classify(c).startswith("disagreement") else ACCENT for c in grid]
    lim = max(0.35, max(abs(v) for v in x + y) * 1.05)
    ax.add_patch(plt.Rectangle((0, 0), lim, lim, facecolor=WARN, alpha=0.07, zorder=0))
    ax.text(lim * 0.97, lim * 0.95, "damage up, F1 up\n(F1 conceals it)", ha="right", va="top",
            fontsize=9, color=WARN)
    ax.scatter(x, y, c=colours, s=16, alpha=0.65, linewidths=0, zorder=3)
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("change in to_target under attack  (right = more rare images pushed to nevus)",
                  fontsize=9, color=MUTED)
    ax.set_ylabel("change in rare-class F1 under attack", fontsize=9, color=MUTED)
    ax.set_title(f"Grid: {len(grid)} paired cases, 408 pre-registered runs",
                 fontsize=12, color=INK, weight="bold", loc="left", pad=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "f1_concealment_scatter.png", facecolor=CREAM)

    # 2. Tier B: honest precision against whether F1 and recall disagree.
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150)
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor("#fffdf9")
    for c in tb:
        dis = classify(c).startswith("disagreement")
        ax.scatter(c["precision_honest"], c["d_f1"], s=80,
                   c=WARN if dis else ACCENT, alpha=0.8, linewidths=0)
        ax.annotate(f"{c['class'][:4]} r{c['round']}", (c["precision_honest"], c["d_f1"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7, color=FAINT)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("honest baseline's precision on that class  (left = over-predicting)",
                  fontsize=9, color=MUTED)
    ax.set_ylabel("change in F1 under attack", fontsize=9, color=MUTED)
    ax.set_title(f"Tier B: {len(tb)} cases - the only data with recall and precision",
                 fontsize=12, color=INK, weight="bold", loc="left", pad=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "f1_concealment_precision.png", facecolor=CREAM)
    print(f"  wrote {FIGS.relative_to(ROOT)}/f1_concealment_scatter.png and _precision.png")


def main() -> int:
    grid, tb = grid_cases(), tier_b_cases()
    overall = rate(grid)
    tb_rate = rate(tb)

    # Test 3, descriptive only (D13).
    dis = [c for c in tb if classify(c).startswith("disagreement")]
    agree = [c for c in tb if classify(c) == "agreement"]
    precision_note = {
        "disagreement_cases": len(dis), "agreement_cases": len(agree),
        "mean_honest_precision_when_disagreeing":
            float(np.mean([c["precision_honest"] for c in dis])) if dis else None,
        "mean_honest_precision_when_agreeing":
            float(np.mean([c["precision_honest"] for c in agree])) if agree else None,
        "declared": "descriptive only - D13 fixed this before running; 16 cells cannot "
                    "support a correlation and none is computed",
    }

    # Does to_target agree with recall where both exist? (D13's recorded caveat.)
    both = [c for c in tb if abs(c["d_recall"]) > EPS or abs(c["d_tt"]) > EPS]
    same_story = sum(1 for c in both if (c["d_recall"] < 0) == (c["d_tt"] > 0))
    tt_vs_recall = {"cases": len(both), "pointing_the_same_way": same_story,
                    "share": same_story / len(both) if both else None}

    record = {
        "entry": "D13", "fixed_in": "docs/DEVIATIONS.md D13, before the analysis ran",
        "epsilon": EPS,
        "test_1_grid_f1_vs_to_target": {
            **overall,
            "by_attack": breakdown(grid, "attack"),
            "by_class": breakdown(grid, "class"),
            "by_split": breakdown(grid, "split"),
            "by_method": breakdown(grid, "method"),
            "by_rule_family": family(grid),
        },
        "test_2_tier_b_f1_vs_recall": {**tb_rate, "cases_detail": tb},
        "test_3_precision_descriptive": precision_note,
        "to_target_vs_recall": tt_vs_recall,
        "scope": {
            "confusion_matrices_available_in": ["docs/results/tier_b.json"],
            "grid_has": "per-class F1 and to_target only; recall and precision are not "
                        "recoverable without re-running",
        },
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "grid_cases": grid,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "f1_concealment.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    print("=" * 76)
    print("D13 - does rare-class F1 conceal attack damage?")
    print("=" * 76)
    print(f"\nTest 1, the grid: {overall['cases']} comparable cases "
          f"({overall['excluded_no_movement']} moved by <= {EPS} and are excluded)")
    print(f"  disagreement (to_target up, F1 up): {overall['disagreements']} of "
          f"{overall['with_clear_movement']} = {100 * overall['disagreement_rate']:.1f}%")
    print(f"  converse    (to_target down, F1 down): {overall['converse']} = "
          f"{100 * overall['converse_rate']:.1f}%")
    for key in ("by_attack", "by_class", "by_split"):
        print(f"\n  {key}:")
        for name, r in record["test_1_grid_f1_vs_to_target"][key].items():
            print(f"    {name:<22} {r['disagreements']:>3} of {r['with_clear_movement']:>3}"
                  f" = {100 * r['disagreement_rate']:>5.1f}%")
    print("\n  by rule family - the split that matters:")
    for name, r in record["test_1_grid_f1_vs_to_target"]["by_rule_family"].items():
        lo, hi = r["disagreement_rate_95ci"]
        print(f"    {name:<28} {r['disagreements']:>3} of "
              f"{r['with_clear_movement']:>3} = {100 * r['disagreement_rate']:>5.1f}%"
              + (f"  [95% {100 * lo:.0f}-{100 * hi:.0f}%]" if lo is not None else ""))
    print(f"\nTest 2, Tier B (exact recall): {tb_rate['disagreements']} of "
          f"{tb_rate['with_clear_movement']} = {100 * tb_rate['disagreement_rate']:.1f}%"
          f"  [{tb_rate['cases']} cases total]")
    print(f"\nTest 3 (descriptive): honest precision when F1 and recall disagree "
          f"{precision_note['mean_honest_precision_when_disagreeing']}, when they agree "
          f"{precision_note['mean_honest_precision_when_agreeing']}")
    print(f"\nto_target vs recall where both exist: "
          f"{tt_vs_recall['pointing_the_same_way']}/{tt_vs_recall['cases']} point the same way")

    plots(grid, tb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
