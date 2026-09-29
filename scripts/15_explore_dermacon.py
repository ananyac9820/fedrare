#!/usr/bin/env python3
"""
What is in DermaCon-IN, measured - the Indian counterpart of scripts/02_explore_data.py.

Answers the questions that decide whether this project's method transfers to it:
  - how long is the tail, and which classes does our rare rule pick out?
  - is a patient ever split across train and test? (the dataset ships a subject-wise split)
  - what could stand in for hospitals, given that the release has no site column?

The last one matters most. Fed-ISIC2019 gives six real hospitals; DermaCon-IN does not say
which of its three hospitals took which photograph, so any centre split here is one we
construct. This script measures how uneven each candidate would be, so the choice is made
on numbers rather than convenience, and it prints the rare-class holdings each one implies.

    python scripts/15_explore_dermacon.py
    python scripts/15_explore_dermacon.py --level Sub_class   # 19 labels instead of 8

Writes results/dermacon_summary.yaml.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from zlib import crc32
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

DATA = ROOT / "data" / "dermacon_in"
HEAD_RATIO_DIVISOR = 25          # configs/default.yaml - the same rule as Fed-ISIC2019
N_CENTRES = 6                    # so a split is comparable with Fed-ISIC2019's six
SPECIALIST_MIN = 20              # a centre must hold this many to count as a holder


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def distribution(rows, key):
    return Counter(r[key] for r in rows)


def rare_by_project_rule(counts: Counter, total: int) -> list[str]:
    """Our rule: a class is rare if its share is under 1/25th of the head class's share."""
    head = counts.most_common(1)[0][1]
    cutoff = head / HEAD_RATIO_DIVISOR
    return sorted([c for c, n in counts.items() if n < cutoff], key=lambda c: counts[c])


def show(title, counts: Counter, total: int, limit=None):
    print(f"\n{title}  ({len(counts)} labels)")
    items = counts.most_common()
    shown = items if limit is None else items[:limit]
    for name, n in shown:
        bar = "#" * max(1, round(40 * n / items[0][1]))
        print(f"  {name[:46]:<46} {n:>5}  {100 * n / total:5.2f}%  {bar}")
    if limit and len(items) > limit:
        rest = sum(n for _, n in items[limit:])
        print(f"  {'... ' + str(len(items) - limit) + ' more labels':<46} {rest:>5}  "
              f"{100 * rest / total:5.2f}%")


def candidate_splits(rows, label_key):
    """Ways to cut this dataset into centres, each one an attribute the dataset records."""
    by = {}
    by["skin tone (Monk)"] = [r["Monk_skin_tone"] or "N/A" for r in rows]
    by["age band"] = [r["Age"] or "N/A" for r in rows]
    by["body region (first listed)"] = [(r["Body_part"].split(",")[0].strip() or "N/A")
                                        for r in rows]
    # zlib.crc32, not hash(): Python salts hash() per process, so the groups would
    # differ between runs and the split would not be reproducible.
    by["patient id (hash)"] = [f"c{crc32(r['Subject_ID'].encode()) % N_CENTRES}"
                               for r in rows]
    return by


def split_report(rows, assignment, label_key, rare):
    """Per-centre sizes and rare-class holdings under one candidate split."""
    groups = defaultdict(list)
    for row, centre in zip(rows, assignment):
        groups[centre].append(row)

    sizes = {k: len(v) for k, v in groups.items()}
    rare_counts = {k: sum(1 for r in v if r[label_key] in rare) for k, v in groups.items()}
    total, rare_total = len(rows), sum(rare_counts.values())

    # The project's headline quantity: rare-disease share divided by size share.
    ratios = {k: ((rare_counts[k] / rare_total) / (sizes[k] / total))
              if rare_total and sizes[k] else 0.0 for k in groups}
    holders = sum(1 for k in groups if rare_counts[k] >= SPECIALIST_MIN)
    return sizes, rare_counts, ratios, holders


def main() -> int:
    parser = argparse.ArgumentParser(description="Explore DermaCon-IN.")
    parser.add_argument("--level", default="Main_class",
                        choices=["Main_class", "Sub_class", "Disease_label"])
    args = parser.parse_args()

    meta = DATA / "Skin_Metadata.tab"
    if not meta.exists():
        raise SystemExit(f"missing {meta}\nRun: python scripts/14_fetch_dermacon.py --metadata")
    rows = read(meta)
    total = len(rows)
    subjects = {r["Subject_ID"] for r in rows}

    print("=" * 78)
    print("DERMACON-IN  (Indian dermatology, 3 tertiary hospitals, North Karnataka)")
    print("=" * 78)
    print(f"images {total:,}   patients {len(subjects):,}   "
          f"images per patient {total / len(subjects):.2f}")
    gradable = sum(1 for r in rows if r["Gradability"].strip().lower().startswith("y"))
    print(f"gradable for diagnosis: {gradable:,} ({100 * gradable / total:.1f}%)")

    for key, limit in (("Main_class", None), ("Sub_class", None), ("Disease_label", 12)):
        show(key.replace("_", " ").upper(), distribution(rows, key), total, limit)

    counts = distribution(rows, args.level)
    rare = rare_by_project_rule(counts, total)
    head_name, head_n = counts.most_common(1)[0]
    print(f"\nRARE CLASSES AT {args.level} under the project's rule "
          f"(share < head share / {HEAD_RATIO_DIVISOR})")
    print(f"  head class: {head_name} at {100 * head_n / total:.2f}%  "
          f"-> cutoff {100 * head_n / total / HEAD_RATIO_DIVISOR:.2f}%")
    if not rare:
        print("  none - this level is too coarse for the rule to bite")
    for c in rare[:12]:
        print(f"  {c[:50]:<50} {counts[c]:>5}  {100 * counts[c] / total:5.2f}%")
    if len(rare) > 12:
        print(f"  ... and {len(rare) - 12} more")

    tail = sum(1 for _, n in counts.items() if n < 20)
    print(f"\n  labels with fewer than 20 images: {tail} of {len(counts)}")

    print("\nSKIN TONE AND PATIENTS")
    show("FITZPATRICK", distribution(rows, "Fitzpatrick"), total)
    show("MONK SKIN TONE", distribution(rows, "Monk_skin_tone"), total)

    # Is a patient ever split across the shipped train/test files?
    train_p = DATA / "train_split.tab"
    test_p = DATA / "test_split.tab"
    if train_p.exists() and test_p.exists():
        tr = {r["Subject_ID"] for r in read(train_p)}
        te = {r["Subject_ID"] for r in read(test_p)}
        overlap = tr & te
        print(f"\nSHIPPED SPLIT  train {len(tr):,} patients / test {len(te):,} patients  "
              f"-> patients in both: {len(overlap)}")

    print("\n" + "=" * 78)
    print(f"CANDIDATE CENTRE SPLITS  (no site column exists - these are constructed)")
    print("=" * 78)
    print(f"Rare classes at {args.level}: {', '.join(rare) if rare else '(none)'}")
    for name, assignment in candidate_splits(rows, args.level).items():
        sizes, rare_counts, ratios, holders = split_report(rows, assignment, args.level, rare)
        order = sorted(sizes, key=lambda k: -sizes[k])[:8]
        biggest = max(ratios.values()) if ratios else 0
        print(f"\n  by {name}: {len(sizes)} groups, "
              f"{holders} of them hold >= {SPECIALIST_MIN} rare images")
        print(f"    largest rare-share / size-share ratio: {biggest:.2f}x "
              f"(Fed-ISIC2019 centre 2 is 1.87x)")
        for k in order:
            print(f"      {str(k)[:28]:<28} {sizes[k]:>5} images  "
                  f"{rare_counts[k]:>4} rare  ratio {ratios[k]:.2f}x")

    out = ROOT / "results" / "dermacon_summary.yaml"
    out.parent.mkdir(exist_ok=True)
    lines = [
        "# Written by scripts/15_explore_dermacon.py - measured, not hand-entered.",
        "dataset: DermaCon-IN (doi:10.7910/DVN/W7OUZM, CC BY-NC-SA 4.0)",
        f"images: {total}", f"patients: {len(subjects)}", f"gradable: {gradable}",
        f"level: {args.level}", f"n_labels: {len(counts)}",
        f"head_class: \"{head_name}\"", f"head_share_pct: {100 * head_n / total:.3f}",
        f"rare_rule: share < head_share / {HEAD_RATIO_DIVISOR}",
        "rare_classes:",
        *[f"  - \"{c}\"  # {counts[c]} images" for c in rare],
        "site_column_available: false  # the release does not say which hospital took an image",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
