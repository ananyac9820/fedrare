#!/usr/bin/env python3
"""
Does the F1-concealment split replicate? Fed-ISIC2019 against DermaCon-IN (D14).

Reads `f1_concealment.json` (the Fed-ISIC2019 study, D13) and `dermacon_concealment.json`
(the replication battery) and puts the two side by side, by rule family, with Wilson 95%
intervals and the number of cases behind every rate.

The DermaCon runs stored confusion matrices, so the same comparison is made twice there:
against `to_target`, for like-for-like with Fed-ISIC2019, and against **exact recall**,
which is the test the original grid could not support at scale.

    python scripts/24_compare_concealment.py

Writes docs/results/concealment_replication.json and .md, and one figure.
"""

from __future__ import annotations

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
EPS = 0.01                                  # D13, unchanged
SELECTION = {"krum", "multi_krum"}

INK, MUTED, FAINT = "#1d2929", "#56615f", "#8a918d"
ACCENT, WARN, CREAM, GRID = "#2c6a64", "#ae2330", "#faf7f0", "#e4dccb"


def wilson(k, n):
    if not n:
        return (None, None)
    z, phat = 1.96, k / n
    d = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / d
    half = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def classify(d_f1, d_damage):
    """D13: disagreement = damage metric says worse while F1 says better."""
    if abs(d_f1) <= EPS or abs(d_damage) <= EPS:
        return "flat"
    if d_damage > 0 and d_f1 > 0:
        return "disagreement"
    if d_damage < 0 and d_f1 < 0:
        return "converse"
    return "agreement"


def summarise(cases):
    kinds = [classify(c["d_f1"], c["d_damage"]) for c in cases]
    moved = [k for k in kinds if k != "flat"]
    dis = sum(k == "disagreement" for k in kinds)
    lo, hi = wilson(dis, len(moved))
    return {"cases": len(cases), "with_clear_movement": len(moved), "disagreements": dis,
            "rate": dis / len(moved) if moved else None, "ci95": [lo, hi],
            "converse": sum(k == "converse" for k in kinds)}


def by_family(cases):
    groups = defaultdict(list)
    for c in cases:
        groups[c["family"]].append(c)
    return {f: summarise(groups[f]) for f in sorted(groups)}


def fedisic_cases():
    """Re-derive the Fed-ISIC2019 cases from the D13 study file."""
    d = json.loads((OUT / "f1_concealment.json").read_text(encoding="utf-8"))
    out = []
    for c in d["grid_cases"]:
        out.append({**c, "family": "selection" if c["method"] in SELECTION else "averaging"})
    return out


def dermacon_cases():
    """Pair each attacked run with the honest run of the same method and seed."""
    d = json.loads((OUT / "dermacon_concealment.json").read_text(encoding="utf-8"))
    rare_ids, target = d["rare_ids"], d["target"]
    honest = {(r["method"], r["seed"]): r for r in d["runs"] if r["attack"] == "none"}

    def stats(cm, c):
        cm = np.asarray(cm)
        support, predicted = cm[c].sum(), cm[:, c].sum()
        return {"recall": cm[c, c] / support if support else 0.0,
                "precision": cm[c, c] / predicted if predicted else 0.0,
                "to_target": cm[c, target] / support if support else 0.0,
                "predicted": int(predicted), "support": int(support)}

    tt, rec = [], []
    for r in d["runs"]:
        if r["attack"] == "none":
            continue
        base = honest[(r["method"], r["seed"])]
        for c in rare_ids:
            h, a = stats(base["confusion"], c), stats(r["confusion"], c)
            f1_h = base["per_class_f1"][str(c)]
            f1_a = r["per_class_f1"][str(c)]
            common = {"dataset": "DermaCon-IN", "method": r["method"], "family": r["family"],
                      "attack": r["attack"], "seed": r["seed"],
                      "class": d["classes"][c], "d_f1": f1_a - f1_h,
                      "f1_honest": f1_h, "f1_attacked": f1_a,
                      "precision_honest": h["precision"], "support": h["support"]}
            tt.append({**common, "damage_metric": "to_target",
                       "d_damage": a["to_target"] - h["to_target"]})
            rec.append({**common, "damage_metric": "recall",
                        "d_damage": h["recall"] - a["recall"],   # positive = recall lost
                        "recall_honest": h["recall"], "recall_attacked": a["recall"]})
    return tt, rec


def figure(fed, derm_tt, derm_rec):
    FIGS.mkdir(parents=True, exist_ok=True)
    panels = [("Fed-ISIC2019\n(vs to_target)", by_family(fed)),
              ("DermaCon-IN\n(vs to_target)", by_family(derm_tt)),
              ("DermaCon-IN\n(vs exact recall)", by_family(derm_rec))]

    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=150)
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor("#fffdf9")
    width, xs = 0.36, np.arange(len(panels))
    for i, (family, colour) in enumerate((("selection", WARN), ("averaging", ACCENT))):
        rates, los, his, labels = [], [], [], []
        for _, fam in panels:
            r = fam.get(family, {"rate": 0.0, "ci95": [0, 0], "disagreements": 0,
                                 "with_clear_movement": 0})
            rate = (r["rate"] or 0.0) * 100
            lo, hi = [(v or 0.0) * 100 for v in r["ci95"]]
            rates.append(rate)
            los.append(max(0.0, rate - lo))
            his.append(max(0.0, hi - rate))
            labels.append(f"{r['disagreements']}/{r['with_clear_movement']}")
        pos = xs + (i - 0.5) * width
        ax.bar(pos, rates, width, color=colour, alpha=0.85,
               label=f"{family} rules", zorder=3)
        ax.errorbar(pos, rates, yerr=[los, his], fmt="none", ecolor=INK, elinewidth=1.2,
                    capsize=4, zorder=4)
        for x, rate, lab in zip(pos, rates, labels):
            ax.text(x, rate + 1.5, lab, ha="center", fontsize=8, color=MUTED)

    ax.set_xticks(xs, [p[0] for p in panels], fontsize=9, color=MUTED)
    ax.set_ylabel("cases where F1 moved opposite to the damage metric (%)",
                  fontsize=9, color=MUTED)
    ax.set_title("Does the concealment split replicate?", fontsize=13, color=INK,
                 weight="bold", loc="left", pad=14)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    fig.tight_layout()
    fig.savefig(FIGS / "concealment_replication.png", facecolor=CREAM)
    print(f"  wrote {(FIGS / 'concealment_replication.png').relative_to(ROOT)}")


def main() -> int:
    if not (OUT / "dermacon_concealment.json").exists():
        raise SystemExit("run scripts/23_dermacon_concealment.py first")
    fed = fedisic_cases()
    derm_tt, derm_rec = dermacon_cases()

    panels = {
        "fed_isic2019_vs_to_target": by_family(fed),
        "dermacon_vs_to_target": by_family(derm_tt),
        "dermacon_vs_recall": by_family(derm_rec),
    }

    def line(name, fam):
        r = fam.get(name)
        if not r or not r["with_clear_movement"]:
            return f"{name:<10} no cases"
        lo, hi = r["ci95"]
        return (f"{name:<10} {r['disagreements']:>3} of {r['with_clear_movement']:>3} = "
                f"{100 * r['rate']:>5.1f}%  [95% {100 * lo:>4.1f}-{100 * hi:>4.1f}%]")

    print("=" * 74)
    print("D14 - does the concealment split replicate?")
    print("=" * 74)
    for title, fam in panels.items():
        print(f"\n{title}")
        for family in ("selection", "averaging"):
            print("  " + line(family, fam))

    # The pre-registered reading.
    d_sel = panels["dermacon_vs_to_target"].get("selection", {})
    d_avg = panels["dermacon_vs_to_target"].get("averaging", {})
    sel_rate = d_sel.get("rate") or 0.0
    avg_rate = d_avg.get("rate") or 0.0
    if avg_rate > 0.10:
        verdict = ("refuted: averaging rules disagree substantially here, so the "
                   "'averaging never does this' claim does not hold")
    elif sel_rate > avg_rate and (d_sel.get("ci95") or [0, 0])[0] > avg_rate:
        verdict = "replicates: selection rules disagree, averaging rules stay near zero"
    elif sel_rate <= 0.05 and avg_rate <= 0.05:
        verdict = ("does not replicate: both families are near zero, so the Fed-ISIC2019 "
                   "split is dataset-specific and is reported as such")
    else:
        verdict = ("inconclusive: the two families are not clearly separated on this "
                   "dataset at this number of cases")

    record = {
        "entry": "D14", "epsilon": EPS,
        "fixed_in": "docs/DEVIATIONS.md D14, before the battery ran",
        "panels": panels, "verdict": verdict,
        "differences_not_controlled": [
            "40 rounds here against 100 in the Fed-ISIC2019 grid",
            "7 classes against 8",
            "constructed age-band centres against six real hospitals",
            "the Fed-ISIC2019 grid had 17 methods, this battery 8",
        ],
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (OUT / "concealment_replication.json").write_text(json.dumps(record, indent=2),
                                                      encoding="utf-8")
    print(f"\nVERDICT: {verdict}")
    figure(fed, derm_tt, derm_rec)
    print(f"wrote {(OUT / 'concealment_replication.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
