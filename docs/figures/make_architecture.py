"""Draw the EARN system architecture diagram (docs/figures/system_architecture.*).

    python docs/figures/make_architecture.py

Writes a cream version (matching the website) and a white one (for slides), each as PNG
and SVG. The numbers on the diagram are measured ones - the dataset counts, 408 runs, and
the ledger's gas and latency - so if a result changes, change them here too.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent

INK, MUTED, FAINT = "#1d2929", "#56615f", "#8a918d"
ACCENT, ACCENT_SOFT = "#2c6a64", "#dcebe7"
LINE, PAPER, CREAM, SAND = "#d9d0bd", "#fffdf9", "#faf7f0", "#f2ebdd"
WARN, WARN_SOFT = "#ae2330", "#f9e1e1"

fig, ax = plt.subplots(figsize=(13.33, 8.8), dpi=150)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
fig.patch.set_facecolor(CREAM)


def box(x, y, w, h, title, lines=(), *, face=PAPER, edge=LINE, tcol=INK, bcol=MUTED,
        tsize=10.5, bsize=8.2, lw=1.2, dashed=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
        facecolor=face, edgecolor=edge, linewidth=lw,
        linestyle=(0, (4, 3)) if dashed else "solid", zorder=2))
    ty = y + h - 3.1 if lines else y + h / 2 - 0.6
    ax.text(x + 2.2, ty, title, fontsize=tsize, color=tcol, weight="bold",
            va="center", ha="left", zorder=3)
    for i, line in enumerate(lines):
        ax.text(x + 2.2, ty - 3.0 - i * 2.55, line, fontsize=bsize, color=bcol,
                va="center", ha="left", zorder=3)


def chip(x, y, w, h, top, bottom, face=SAND, edge=LINE, tcol=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9",
                                facecolor=face, edgecolor=edge, linewidth=1, zorder=3))
    ax.text(x + w / 2, y + h * 0.62, top, fontsize=8.4, color=tcol, weight="bold",
            ha="center", va="center", zorder=4)
    ax.text(x + w / 2, y + h * 0.26, bottom, fontsize=7.0, color=MUTED, ha="center",
            va="center", zorder=4)


def arrow(x1, y1, x2, y2, color=ACCENT, lw=1.6, style="-|>", dashed=False):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13, linewidth=lw,
        color=color, zorder=5, linestyle=(0, (4, 3)) if dashed else "solid",
        shrinkA=0, shrinkB=0))


def band(y, text):
    ax.text(1.0, y, text, fontsize=8.0, color=FAINT, weight="bold", rotation=90,
            va="center", ha="center", zorder=3)


# ---------------------------------------------------------------- title
ax.text(4, 96.6, "EARN — system architecture", fontsize=19, color=INK, weight="bold",
        va="center")
ax.text(4, 93.2, "Coverage-aware, ledger-anchored federated aggregation for rare skin disease  ·  "
                 "every number below comes from a result file in the repository",
        fontsize=8.6, color=MUTED, va="center")

# ---------------------------------------------------------------- data
band(85.5, "DATA")
box(4, 79.5, 56, 11.5, "Fed-ISIC2019  ·  23,247 dermoscopy images  ·  6 hospitals",
    ["Hugging Face mirror flwrlabs/fed-isic2019 (FLamby-derived, CC BY-NC 4.0)",
     "train 18,597   test 4,650   ·   images never leave the hospital"])
box(63, 79.5, 33, 11.5, "Two splits",
    ["S1  the natural hospital split",
     "S2  specialist: 90% of rare images moved to one centre"],
    face=SAND)

arrow(32, 79.5, 32, 74.5)

# ---------------------------------------------------------------- features
band(69.5, "FEATURES")
box(4, 63.5, 56, 11, "DenseNet-121 (ImageNet, frozen)  →  1,024-d features",
    ["extracted once for every image · scripts/05_extract_features.py",
     "only the classifier head is federated — the study runs on a laptop"])
box(63, 63.5, 33, 11, "Held out",
    ["the 4,650 test images are never trained on",
     "they are what the live demo predicts"], face=SAND)

arrow(32, 63.5, 32, 58.5)

# ---------------------------------------------------------------- federated loop
band(50, "FEDERATED LOOP")
box(4, 40.5, 56, 17.5, "Tier A federated round  ·  6 clients  ·  100 rounds  ·  3 seeds",
    ["each hospital trains the head on its own features (class-balanced loss),",
     "then sends only the update, clipped to the median size"])
for i, (cid, n) in enumerate([(0, "9,930"), (1, "3,163"), (2, "2,691"),
                              (3, "1,807"), (4, "655"), (5, "351")]):
    chip(6.4 + i * 8.9, 42.2, 7.8, 5.6, f"H{cid}", n)

box(63, 40.5, 31, 17.5, "Attacks — one dishonest hospital",
    ["A1  sudden expert: claims rare data it has none of",
     "A2  sleeper: honest 15 rounds, then turns",
     "A3  scaling: sends a 10x oversized update",
     "injected as client hooks — no other code changes"],
    face=WARN_SOFT, edge=WARN, tcol=WARN, bcol="#7d2028")
arrow(63, 49, 60, 49, color=WARN)

arrow(32, 40.5, 32, 35.5)

# ---------------------------------------------------------------- aggregation
band(26, "AGGREGATION")
box(4, 30.5, 90, 4.6, "aggregate(client_states, client_sizes, global_state, round)  →  "
                      "new global model + round_info", face=SAND, tsize=9.5)
box(4, 16.5, 44, 12.5, "Baselines",
    ["FedAvg  ·  Krum  ·  Multi-Krum  ·  trimmed mean",
     "coordinate-wise median  ·  Camp A (evidence-proportional)"])
box(52, 16.5, 42, 12.5, "EARN",
    ["evidence → coverage → check the claim → trust",
     "weight = size share + trust x evidence share, capped at 50%",
     "trust rises 0.1 a round, halves on a bad one"],
    face=ACCENT_SOFT, edge=ACCENT, tcol=ACCENT, bcol="#245a55")
arrow(26, 30.5, 26, 29.0)
arrow(74, 30.5, 74, 29.0)

arrow(74, 16.5, 74, 15.0)
arrow(26, 16.5, 26, 15.0)

# ---------------------------------------------------------------- outputs
band(8, "OUTPUTS")
box(4, 1.0, 28, 14.0, "Gates + results",
    ["G0a · G0b · G1 · G2, all fixed before running",
     "408 pre-registered runs · 44 Python tests",
     "results/*.csv  ·  docs/results/*.json"])
box(36, 1.0, 28, 14.0, "EarnLedger.sol",
    ["Hardhat chain, solc 0.8.24",
     "2,400 rounds · ~184k gas · ~1.0 ms a round",
     "forged trust steps rejected on chain"],
    face=ACCENT_SOFT, edge=ACCENT, tcol=ACCENT, bcol="#245a55")
box(66, 1.0, 28, 14.0, "Site + live demo",
    ["sync-results.mjs → web/data/*.json → Next.js",
     "demo/server.py serves the saved head",
     "/demo predicts a held-out image in ~0.2 s"])

# demo path: test images go straight to the demo
ax.plot([96, 97.8], [69, 69], color=FAINT, lw=1.2, linestyle=(0, (4, 3)), zorder=1)
ax.plot([97.8, 97.8], [69, 8], color=FAINT, lw=1.2, linestyle=(0, (4, 3)), zorder=1)
arrow(97.8, 8, 94, 8, color=FAINT, lw=1.2)
ax.text(97.0, 38.5, "held-out images", fontsize=7.0, color=FAINT, rotation=90,
        va="center", ha="center")

ax.text(4, -3.2, "fedrare.vercel.app  ·  github.com/ananyac9820/fedrare", fontsize=7.6,
        color=FAINT, va="center")

OUT.mkdir(parents=True, exist_ok=True)
# Two backgrounds: cream matches the website, white drops onto a plain slide.
for name, bg in (("system_architecture", CREAM), ("system_architecture_white", "white")):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", facecolor=bg, bbox_inches="tight", pad_inches=0.35)
    print("wrote", (OUT / f"{name}.png").name, "and .svg")
