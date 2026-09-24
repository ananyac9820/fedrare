"""Draw the trust-score figure (docs/figures/trust_scores.*).

    python docs/figures/make_trust_figure.py

Four panels, all read from result files - nothing is typed in by hand:
  A  the trust table EARN ends on: 6 hospitals x 8 diseases, S1, no attack, seed 42
  B  what happens to a sleeper attacker's trust, with and without the locked history
  C  how many hospitals hold each disease - which is what makes checking possible
  D  rare-class F1 through the same attack, which is the part that does not hold up

Sources: web/data/ledger.json (per-round trust tables committed on chain) and
docs/results/followup_ledger_trajectory.json (the D7 diagnostic).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

INK, MUTED, FAINT = "#1d2929", "#56615f", "#8a918d"
ACCENT, ACCENT_SOFT = "#2c6a64", "#dcebe7"
CREAM, SAND, LINE = "#faf7f0", "#e7dcc7", "#d9d0bd"
WARN = "#ae2330"

SHORT = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
RARE = {5, 6}

ledger = json.loads((ROOT / "web" / "data" / "ledger.json").read_text(encoding="utf-8"))
traj = json.loads((ROOT / "docs" / "results" / "followup_ledger_trajectory.json")
                  .read_text(encoding="utf-8"))

last = ledger["rounds"][-1]
trust = np.array(last["trustBps"], dtype=float).reshape(6, 8) / 10000
coverage = np.array(last["coverage"], dtype=int)
earn, plain = traj["methods"]["earn"], traj["methods"]["earn_no_ledger"]
TURN = 16  # A2s: honest for 15 rounds, then turns

greens = LinearSegmentedColormap.from_list("earn", ["#ffffff", ACCENT_SOFT, ACCENT])

fig = plt.figure(figsize=(13.4, 8.0), dpi=150)
fig.patch.set_facecolor(CREAM)
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1, 0.92],
                      left=0.055, right=0.985, top=0.835, bottom=0.085,
                      wspace=0.22, hspace=0.42)

fig.text(0.055, 0.955, "EARN trust scores, as measured", fontsize=18, color=INK,
         weight="bold", va="center")
fig.text(0.055, 0.915, "Trust is per hospital and per disease - a hospital can be trusted on "
                       "one disease and not another. 0 means no claim to check.",
         fontsize=9, color=MUTED, va="center")


def style(ax, title, subtitle=""):
    ax.set_facecolor("#fffdf9")
    for side, on in (("top", False), ("right", False), ("left", True), ("bottom", True)):
        ax.spines[side].set_visible(on)
        if on:
            ax.spines[side].set_color(LINE)
    ax.set_title(title, fontsize=11.5, color=INK, weight="bold", loc="left", pad=30)
    if subtitle:
        ax.text(0, 1.012, subtitle, transform=ax.transAxes, fontsize=8.2, color=FAINT,
                va="bottom", ha="left")
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)


# ------------------------------------------------------------------ A: trust table
axA = fig.add_subplot(gs[0, 0])
axA.imshow(trust, cmap=greens, vmin=0, vmax=1, aspect="auto")
axA.set_xticks(range(8), SHORT)
axA.set_yticks(range(6), [f"H{i}" for i in range(6)])
for i in range(6):
    for j in range(8):
        v = trust[i, j]
        label = "-" if v == 0 else ("<0.01" if v < 0.005 else f"{v:.2f}")
        axA.text(j, i, label, ha="center", va="center",
                 fontsize=7.6 if label == "<0.01" else 8.2,
                 color="white" if v > 0.55 else (FAINT if v == 0 else INK))
for lbl in axA.get_xticklabels():
    if lbl.get_text() in ("DF", "VASC"):
        lbl.set_color(ACCENT)
        lbl.set_weight("bold")
style(axA, "A  The trust table after 100 rounds",
      "S1, no attack, seed 42 - the table committed to the ledger each round")
axA.set_xlabel("disease", fontsize=8.5, color=MUTED)
axA.spines["left"].set_visible(False)
axA.spines["bottom"].set_visible(False)

# ------------------------------------------------------------------ B: sleeper attack
axB = fig.add_subplot(gs[0, 1])
rounds = np.arange(1, len(earn["trust_6"]) + 1)
axB.plot(rounds, earn["trust_6"], color=ACCENT, lw=2.2, label="with locked history")
axB.plot(rounds, plain["trust_6"], color=FAINT, lw=1.8, ls="--",
         label="without it (no ledger)")
axB.axvline(TURN, color=WARN, lw=1.2, ls=":")
axB.text(TURN + 1.5, 0.93, "attacker turns", fontsize=8, color=WARN)
axB.annotate(f"falls to {earn['min_trust_6_after_turn']:.3f}\nby round "
             f"{earn['round_of_min']}",
             xy=(earn["round_of_min"], earn["min_trust_6_after_turn"]),
             xytext=(30, 0.30), fontsize=8, color=ACCENT,
             arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1))
axB.set_ylim(-0.04, 1.08)
axB.set_xlim(1, 60)
style(axB, "B  A sleeper attacker's trust on Vascular lesion",
      "S2, attack A2s, attacker = hospital 2, mean of 3 seeds")
axB.set_xlabel("round", fontsize=8.5, color=MUTED)
axB.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=MUTED)

# ------------------------------------------------------------------ C: coverage
axC = fig.add_subplot(gs[1, 0])
colours = [ACCENT if i in RARE else SAND for i in range(8)]
axC.bar(range(8), coverage, color=colours, edgecolor=LINE, width=0.64)
for i, c in enumerate(coverage):
    axC.text(i, c + 0.12, str(c), ha="center", fontsize=8.6,
             color=ACCENT if i in RARE else MUTED,
             weight="bold" if i in RARE else "normal")
axC.set_xticks(range(8), SHORT)
axC.set_yticks(range(0, 7, 2))
axC.set_ylim(0, 6.9)
style(axC, "C  How many hospitals hold each disease",
      "coverage - with 2 or 3 holders there is almost nobody to check a claim against")
axC.set_ylabel("hospitals", fontsize=8.5, color=MUTED)

# ------------------------------------------------------------------ D: rare F1
axD = fig.add_subplot(gs[1, 1])
axD.plot(rounds, earn["rare_f1"], color=ACCENT, lw=2.2, label="with locked history")
axD.plot(rounds, plain["rare_f1"], color=FAINT, lw=1.8, ls="--", label="without it")
axD.axvline(TURN, color=WARN, lw=1.2, ls=":")
axD.text(TURN + 1.5, 0.55, "attacker turns", fontsize=8, color=WARN)
axD.set_xlim(1, 60)
axD.set_ylim(-0.03, 0.72)
style(axD, "D  Rare-class F1 through the same attack",
      "the honest part: the lock spots the turn, but the damage still lands")
axD.set_xlabel("round", fontsize=8.5, color=MUTED)
axD.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=MUTED)

fig.text(0.055, 0.022,
         "Read from web/data/ledger.json and docs/results/followup_ledger_trajectory.json.  "
         "These EARN rounds use the oracle evidence signal (exploratory) - measuring that "
         "signal from the updates is the open problem.",
         fontsize=7.6, color=FAINT, va="center")

for name, bg in (("trust_scores", CREAM), ("trust_scores_white", "white")):
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"{name}.{ext}", facecolor=bg)
    print("wrote", f"{name}.png and .svg")
