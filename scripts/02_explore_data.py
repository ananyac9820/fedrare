#!/usr/bin/env python3
"""
Week 1-2 - measure the real per-centre, per-class distribution.

This is the most important script in the early phase. Everything downstream depends on
knowing which classes are genuinely rare and which centres actually hold them. It is not
optional exploratory work: the rare class IDs it produces get written into the config and
drive the primary evaluation metric.

Outputs to results/:
    class_distribution.csv        counts per centre per class
    class_distribution_pct.csv    row-normalised percentages
    class_distribution.png        heatmap
    rare_classes.yaml             measured rare class IDs, to paste into the config

    python scripts/02_explore_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yaml

from src.data.loader import load_split_columns
from src.utils.config import load_config

# A centre must hold at least this many images of a rare class to be named its specialist,
# so a centre with a handful of images cannot top the over-representation table on noise.
MIN_SPECIALIST_IMAGES = 20


def collect(cfg: dict) -> pd.DataFrame:
    n_centers = cfg["dataset"]["n_centers"]
    n_classes = cfg["dataset"]["n_classes"]
    names = cfg["dataset"]["class_names"]

    # One pass per split, reading only the two integer columns, rather than filtering the
    # whole dataset once per centre per split.
    counts = np.zeros((n_centers, n_classes), dtype=int)
    for split in ("train", "test"):
        centers, labels = load_split_columns(split=split)
        print(f"  {split}: {len(labels):>6,} images")
        for c in range(n_centers):
            counts[c] += np.bincount(labels[centers == c], minlength=n_classes)

    for c in range(n_centers):
        print(f"  centre {c}: {int(counts[c].sum()):>6,} images")

    df = pd.DataFrame(
        counts,
        index=[f"centre_{c}" for c in range(n_centers)],
        columns=[names[i] for i in range(n_classes)],
    )
    return df


def identify_rare(df: pd.DataFrame, head_ratio_divisor: float) -> tuple[list[int], float]:
    """Tail classes are those under 1/head_ratio_divisor of the head class's share.

    Returns (rare class ids, cutoff as a percentage of all samples).
    """
    totals = df.sum(axis=0)
    share = totals / totals.sum() * 100.0
    cutoff_pct = float(share.max()) / head_ratio_divisor
    return [i for i, col in enumerate(df.columns) if share[col] < cutoff_pct], cutoff_pct


def rare_over_representation(df: pd.DataFrame, rare_ids: list[int],
                             min_images: int = MIN_SPECIALIST_IMAGES):
    """How over-represented each rare class is at each centre, relative to centre size.

    over_rep[k][c] = (share of class c held by centre k) / (share of all images held by centre k)

    1.0 means the centre holds exactly its proportional share; above 1.0 it is a specialist
    in that class relative to its size; below 1.0 it is under-represented. The specialist
    for a class is the highest-ratio centre among those holding at least min_images of it.

    Returns (over_rep DataFrame of centres x rare classes, {class name: centre or None}).
    """
    names = [df.columns[i] for i in rare_ids]
    centre_share = df.sum(axis=1) / df.values.sum()
    class_share = df[names] / df[names].sum(axis=0)
    over_rep = class_share.div(centre_share, axis=0)

    specialists = {}
    for name in names:
        eligible = df[name] >= min_images
        specialists[name] = over_rep.loc[eligible, name].idxmax() if eligible.any() else None
    return over_rep, specialists


def plot(df: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  (matplotlib/seaborn unavailable, skipping plot)")
        return

    pct = df.div(df.sum(axis=1), axis=0) * 100.0
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pct, annot=True, fmt=".1f", cmap="rocket_r", ax=ax,
                cbar_kws={"label": "% of that centre's images"})
    ax.set_title("Fed-ISIC2019: class distribution per centre (%)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}")


def main() -> int:
    cfg = load_config("configs/default.yaml")
    out_dir = Path(cfg.get("output_dir", "./results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading centres...")
    try:
        df = collect(cfg)
    except Exception as exc:
        print(f"\nCould not load the dataset: {exc}")
        print("Run scripts/01_verify_setup.py first.")
        return 1

    totals = df.sum(axis=0)
    share = totals / totals.sum() * 100.0
    divisor = cfg["dataset"]["head_ratio_divisor"]
    rare_ids, cutoff_pct = identify_rare(df, divisor)
    rare_names = [df.columns[i] for i in rare_ids]

    print("\n" + "=" * 68)
    print("OVERALL CLASS DISTRIBUTION")
    print("=" * 68)
    for name in df.columns:
        flag = "  <-- RARE" if share[name] < cutoff_pct else ""
        print(f"  {name:<26} {int(totals[name]):>7,}  ({share[name]:5.2f}%){flag}")
    print(f"  {'TOTAL':<26} {int(totals.sum()):>7,}")

    print("\n" + "=" * 68)
    print("RARE CLASSES (measured, not assumed)")
    print("=" * 68)
    head = share.idxmax()
    print(f"  head class: {head} ({share[head]:.2f}%)")
    print(f"  cutoff: {share[head]:.2f}% / {divisor} = {cutoff_pct:.2f}% of all samples")
    for i, name in zip(rare_ids, rare_names):
        print(f"  class {i}: {name}  ({share[name]:.2f}%)")

    # Which centre holds the largest share of each rare class? This is the hospital whose
    # contribution a scalar reputation scheme would most likely under-weight, and therefore
    # the one to watch throughout the experiments.
    print("\n" + "=" * 68)
    print("RARE-CLASS CONCENTRATION BY CENTRE")
    print("=" * 68)
    for i, name in zip(rare_ids, rare_names):
        col = df[name]
        if col.sum() == 0:
            continue
        pct_of_class = col / col.sum() * 100.0
        top = pct_of_class.idxmax()
        print(f"  {name}:")
        for centre in df.index:
            marker = "  <-- largest holder" if centre == top else ""
            print(f"     {centre}: {int(col[centre]):>4} ({pct_of_class[centre]:5.1f}%){marker}")

    # Raw concentration favours big centres. Dividing by each centre's overall share shows
    # which centres hold more of a rare class than their size alone would predict.
    over_rep, specialists = rare_over_representation(df, rare_ids)
    print("\n" + "=" * 68)
    print("RARE-CLASS OVER-REPRESENTATION BY CENTRE")
    print("=" * 68)
    print("  ratio = (centre's share of the class) / (centre's share of all images)")
    print("  1.00 = proportional, >1 = specialist for its size, <1 = under-represented")
    print("  cells show: ratio (images of that class)\n")
    print(f"  {'':<10}" + "".join(f"{name:>20}" for name in over_rep.columns))
    for centre in df.index:
        cells = "".join(
            f"{f'{over_rep.at[centre, name]:.2f} ({int(df.at[centre, name])})':>20}"
            for name in over_rep.columns
        )
        print(f"  {centre:<10}{cells}")
    print(f"\n  Most over-represented centre per class (min {MIN_SPECIALIST_IMAGES} images):")
    for name in over_rep.columns:
        centre = specialists[name]
        if centre is None:
            print(f"     {name}: no centre holds >= {MIN_SPECIALIST_IMAGES} images")
        else:
            print(f"     {name}: {centre}  (ratio {over_rep.at[centre, name]:.2f}, "
                  f"{int(df.at[centre, name])} images)")

    df.to_csv(out_dir / "class_distribution.csv")
    (df.div(df.sum(axis=1), axis=0) * 100).round(2).to_csv(out_dir / "class_distribution_pct.csv")

    with open(out_dir / "rare_classes.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({
            "rare_class_ids": [int(i) for i in rare_ids],
            "rare_class_names": rare_names,
            "head_ratio_divisor": divisor,
            "rare_cutoff_pct": round(cutoff_pct, 3),
            "class_share_pct": {str(n): round(float(share[n]), 3) for n in df.columns},
            "specialist_min_images": MIN_SPECIALIST_IMAGES,
            "rare_over_representation": {
                str(n): {str(k): round(float(over_rep.at[k, n]), 3) for k in df.index}
                for n in over_rep.columns
            },
            "rare_specialist_centre": {str(n): specialists[n] for n in over_rep.columns},
        }, f, sort_keys=False)

    plot(df, out_dir / "class_distribution.png")

    print("\n" + "=" * 68)
    print("NEXT STEP")
    print("=" * 68)
    print(f"  Copy rare_class_ids {rare_ids} from results/rare_classes.yaml")
    print("  into configs/default.yaml under dataset.rare_class_ids.")
    print("  Verify the class names match the config - do not assume the ordering.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
