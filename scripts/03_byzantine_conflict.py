#!/usr/bin/env python3
"""
Feasibility check - do standard Byzantine-robust aggregation rules reject the rare-class
specialist centre?

scripts/02_explore_data.py found one centre most over-represented for every rare class
relative to its size (read from results/rare_classes.yaml). A robust rule that treats
"different from the majority" as suspicious would be expected to reject that centre more
often than the others. This script measures whether it does.

There are no attackers in this run: every rejection is a rejection of an honest centre.

A negative result is a valid outcome. The settings and the thresholds that turn rejection
rates into a verdict are fixed constants below, set before the first run. Do not adjust them
after seeing results.

Outputs to results/:
    byzantine_conflict.csv    round, rule, client_id, selected, weight
    byzantine_conflict.md     rejection rates, specialist comparison, rankings, verdict

    python scripts/03_byzantine_conflict.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from src.data.loader import IMAGENET_MEAN, IMAGENET_STD, load_split_columns
from src.federated import robust
from src.utils.config import load_config
from src.utils.metrics import summarise
from src.utils.seed import set_seed

# --- Fixed settings. CPU-only feasibility run: speed over accuracy. ---
IMAGE_SIZE = 32
N_ROUNDS = 15
# Every client runs the same number of local steps per round (as in FLamby's Fed-ISIC2019
# benchmark). With one local epoch instead, centre 0 would take ~30x more steps than centre
# 5 and update magnitude - i.e. dataset size - would dominate what the robust rules react to.
LOCAL_STEPS = 50
BATCH_SIZE = 32
LR = 1e-3
F_BYZANTINE = 1  # the largest f Krum tolerates with 6 clients (needs n >= 2f + 3)
TRIM = 1         # trimmed mean drops this many values from each end, per coordinate

RULES = ["fedavg", "krum", "multi_krum", "trimmed_mean", "coordinate_wise_median"]
ROBUST_RULES = RULES[1:]

# --- Verdict thresholds, fixed before the first run. ---
# Per rule: shortfall = 1 - (specialist's survival rate / other centres' mean survival rate),
# where survival = 1 - rejection rate. 0 means the specialist survives as often as the
# others; 1 means it is always rejected while the others are not. Normalising by survival
# keeps the measure comparable across rules with very different base rejection rates
# (Krum rejects 5 of 6 clients every round; multi-Krum rejects 1).
CONFLICT_THRESHOLD = 0.50
WEAK_THRESHOLD = 0.25


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(32 * (IMAGE_SIZE // 4) ** 2, n_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def load_images(split: str, cfg: dict, cache_dir: Path):
    """Downsampled images for one split, cached as .npz after the first decode."""
    path = cache_dir / f"fed_isic2019_{split}_{IMAGE_SIZE}px.npz"
    if path.exists():
        z = np.load(path)
        return z["images"], z["centers"], z["labels"]

    from datasets import load_dataset
    from PIL import Image

    ds = load_dataset(cfg["dataset"]["hf_repo"], split=split)
    images = np.empty((len(ds), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    for i, row in enumerate(ds):
        img = row["image"].convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        images[i] = np.asarray(img)
    centers, labels = load_split_columns(split=split)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, images=images, centers=centers, labels=labels)
    return images, centers, labels


def to_tensor(images: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def load_specialist(out_dir: Path) -> tuple[int, dict, dict]:
    path = out_dir / "rare_classes.yaml"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run scripts/02_explore_data.py first.")
    with open(path, "r", encoding="utf-8") as f:
        info = yaml.safe_load(f)
    spec = info.get("rare_specialist_centre") or {}
    centres = {v for v in spec.values() if v}
    if len(centres) != 1:
        raise SystemExit(f"Expected one specialist centre across rare classes, got {spec}")
    return int(centres.pop().split("_")[1]), info, spec


def local_update(model: nn.Module, global_params: torch.Tensor, x: torch.Tensor,
                 y: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    # vector_to_parameters makes each parameter a view into the vector it is given, so pass a
    # copy: otherwise training writes straight into global_params and every update is zero.
    vector_to_parameters(global_params.clone(), model.parameters())
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = len(y)
    perm, pos = rng.permutation(n), 0
    for _ in range(LOCAL_STEPS):
        if pos + BATCH_SIZE > n:
            perm, pos = rng.permutation(n), 0
        idx = torch.from_numpy(perm[pos:pos + BATCH_SIZE])
        pos += BATCH_SIZE
        opt.zero_grad()
        F.cross_entropy(model(x[idx]), y[idx]).backward()
        opt.step()
    return parameters_to_vector(model.parameters()).detach() - global_params


def aggregate(rule: str, updates: torch.Tensor, n_samples) -> robust.AggregationResult:
    if rule == "fedavg":
        return robust.fedavg(updates, n_samples)
    if rule == "krum":
        return robust.krum(updates, F_BYZANTINE)
    if rule == "multi_krum":
        return robust.multi_krum(updates, F_BYZANTINE)
    if rule == "trimmed_mean":
        return robust.trimmed_mean(updates, TRIM)
    if rule == "coordinate_wise_median":
        return robust.coordinate_wise_median(updates)
    raise ValueError(rule)


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    return torch.cat([model(x[i:i + 2048]).argmax(1) for i in range(0, len(x), 2048)]).numpy()


def rule_verdict(shortfall: float) -> str:
    if shortfall >= CONFLICT_THRESHOLD:
        return "CONFLICT"
    if shortfall >= WEAK_THRESHOLD:
        return "WEAK"
    return "NO"


def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> int:
    cfg = load_config("configs/default.yaml")
    seed = int(cfg["seed"])
    n_classes = cfg["dataset"]["n_classes"]
    n_centers = cfg["dataset"]["n_centers"]
    out_dir = Path(cfg.get("output_dir", "./results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    target, rare_info, spec = load_specialist(out_dir)
    rare_ids = [int(i) for i in rare_info["rare_class_ids"]]
    over_rep = rare_info.get("rare_over_representation", {})
    print(f"Specialist centre under test: centre {target} ({spec})")

    print("Loading images...")
    t0 = time.time()
    cache_dir = Path("data/cache")
    tr_img, tr_ctr, tr_lab = load_images("train", cfg, cache_dir)
    te_img, _, te_lab = load_images("test", cfg, cache_dir)
    x_train, x_test = to_tensor(tr_img), to_tensor(te_img)
    y_train = torch.from_numpy(tr_lab.astype(np.int64))
    client_idx = [np.flatnonzero(tr_ctr == k) for k in range(n_centers)]
    xs = [x_train[torch.from_numpy(i)] for i in client_idx]
    ys = [y_train[torch.from_numpy(i)] for i in client_idx]
    n_samples = [len(i) for i in client_idx]
    print(f"  {len(tr_lab):,} train / {len(te_lab):,} test images at {IMAGE_SIZE}px "
          f"({time.time() - t0:.0f}s); train per centre: {n_samples}")

    rows, evals = [], {}
    for rule in RULES:
        t0 = time.time()
        set_seed(seed)
        model = SmallCNN(n_classes)
        global_params = parameters_to_vector(model.parameters()).detach().clone()
        for rnd in range(1, N_ROUNDS + 1):
            # Same batch order per (round, client) under every rule, so rules differ only in
            # how they aggregate.
            updates = torch.stack([
                local_update(model, global_params, xs[k], ys[k],
                             np.random.default_rng([seed, rnd, k]))
                for k in range(n_centers)
            ])
            res = aggregate(rule, updates, n_samples)
            global_params = global_params + res.aggregate
            for k in range(n_centers):
                rows.append({"round": rnd, "rule": rule, "client_id": k,
                             "selected": bool(res.selected[k]),
                             "weight": float(res.weights[k])})

        vector_to_parameters(global_params, model.parameters())
        evals[rule] = summarise(te_lab.astype(int), predict(model, x_test), rare_ids, n_classes)
        print(f"  {rule:<24} {time.time() - t0:5.0f}s   test acc {evals[rule]['accuracy']:.3f}"
              f"   balanced acc {evals[rule]['balanced_accuracy']:.3f}")

    log = pd.DataFrame(rows)
    log.to_csv(out_dir / "byzantine_conflict.csv", index=False)

    # --- Analysis ---
    rejection = 1.0 - log.groupby(["rule", "client_id"])["selected"].mean().unstack()
    mean_weight = log.groupby(["rule", "client_id"])["weight"].mean().unstack()
    rejection, mean_weight = rejection.loc[RULES], mean_weight.loc[RULES]
    centres = list(range(n_centers))

    comparison = {}
    for rule in ROBUST_RULES:
        r_t = float(rejection.at[rule, target])
        r_others = float(rejection.loc[rule].drop(target).mean())
        r_all = float(rejection.loc[rule].mean())
        s_others = 1.0 - r_others
        shortfall = 1.0 - (1.0 - r_t) / s_others if s_others > 0 else 0.0
        comparison[rule] = {"r_t": r_t, "r_others": r_others, "r_all": r_all,
                            "shortfall": shortfall, "verdict": rule_verdict(shortfall)}

    ranks = {}
    for rule in RULES:
        order = sorted(centres, key=lambda k: (-rejection.at[rule, k], mean_weight.at[rule, k]))
        ranks[rule] = {k: order.index(k) + 1 for k in centres}

    verdicts = [c["verdict"] for c in comparison.values()]
    n_conflict = verdicts.count("CONFLICT")
    n_weak_or_more = n_conflict + verdicts.count("WEAK")
    if n_conflict >= 2:
        overall = "CONFLICT CONFIRMED"
    elif n_conflict >= 1 or n_weak_or_more >= 2:
        overall = "WEAK CONFLICT"
    else:
        overall = "NO CONFLICT"

    # --- Console ---
    print("\n" + "=" * 78)
    print("PER-CENTRE REJECTION RATE (fraction of rounds rejected or near-zero weight)")
    print("=" * 78)
    print(f"  {'rule':<24}" + "".join(f"{f'centre {k}':>9}" for k in centres))
    print(f"  {'(train images)':<24}" + "".join(f"{n_samples[k]:>9,}" for k in centres))
    for rule in RULES:
        print(f"  {rule:<24}" + "".join(f"{pct(rejection.at[rule, k]):>9}" for k in centres))

    print("\n" + "=" * 78)
    print(f"CENTRE {target} VS OTHER CENTRES")
    print("=" * 78)
    print(f"  {'rule':<24}{f'centre {target}':>10}{'others':>9}{'diff':>9}"
          f"{'shortfall':>11}{'rank':>6}   verdict")
    for rule, c in comparison.items():
        print(f"  {rule:<24}{pct(c['r_t']):>10}{pct(c['r_others']):>9}"
              f"{100 * (c['r_t'] - c['r_others']):>+8.1f}pp{c['shortfall']:>10.2f}"
              f"{ranks[rule][target]:>4}/{n_centers}   {c['verdict']}")
    print(f"\n  thresholds: CONFLICT if shortfall >= {CONFLICT_THRESHOLD}, "
          f"WEAK if >= {WEAK_THRESHOLD}")

    print("\n" + "=" * 78)
    print(f"VERDICT: {overall}")
    print("=" * 78)
    print(f"  per-rule verdicts: {dict(zip(comparison, verdicts))}")

    # --- Markdown report ---
    class_names = cfg["dataset"]["class_names"]
    rare_desc = ", ".join(
        f"{over_rep[class_names[c]][f'centre_{target}']:.2f}x on {class_names[c]}"
        for c in rare_ids if class_names[c] in over_rep
    )
    header = ["rule"] + [f"centre {k}" for k in centres]
    size_row = ["*train images*"] + [f"{n:,}" for n in n_samples]

    md = [
        "# Byzantine-robust aggregation vs the rare-class specialist",
        "",
        f"**Verdict: {overall}**",
        "",
        f"Centre under test: **centre {target}**, the most over-represented centre for every "
        f"rare class relative to its size ({rare_desc}; from `results/rare_classes.yaml`).",
        "",
        "No client is malicious in this run, so every rejection is a rejection of an honest centre.",
        "",
        "## Setup",
        "",
        f"- Fed-ISIC2019 (HF mirror), 6 clients = the 6 real centres, train split only for training",
        f"- Model: 2-layer CNN on {IMAGE_SIZE}x{IMAGE_SIZE} images, plain cross-entropy",
        f"- {N_ROUNDS} rounds, all centres every round, {LOCAL_STEPS} local Adam steps per centre "
        f"per round (batch {BATCH_SIZE}, lr {LR}), seed {seed}",
        f"- Krum / multi-Krum: f = {F_BYZANTINE}; multi-Krum averages the n - f = "
        f"{n_centers - F_BYZANTINE} best-scored clients",
        f"- Trimmed mean: {TRIM} value trimmed from each end per coordinate; median: mean of the "
        f"middle two values (6 clients)",
        "",
        "**Rejected** means: not selected (Krum, multi-Krum), or effective weight below "
        f"{robust.NEAR_ZERO_FRACTION} x the uniform share 1/{n_centers} (trimmed mean, median - "
        "these drop coordinates, not clients, so effective weight is the fraction of the "
        "aggregate's coordinate values that came from that centre). FedAvg rejects no one.",
        "",
        "Verdict thresholds were fixed before the run. Per rule, *shortfall* = 1 - (centre's "
        "survival rate / other five centres' mean survival rate), survival = 1 - rejection rate. "
        f"CONFLICT if shortfall >= {CONFLICT_THRESHOLD}, WEAK if >= {WEAK_THRESHOLD}, else NO. "
        "Overall: CONFLICT CONFIRMED if at least 2 of the 4 robust rules give CONFLICT; WEAK "
        "CONFLICT if 1 does, or at least 2 give WEAK or stronger; otherwise NO CONFLICT.",
        "",
        "## Sanity check: did the models learn anything?",
        "",
        f"Final global model per rule on the pooled test split ({len(te_lab):,} images). "
        f"Majority-class accuracy is {np.bincount(te_lab).max() / len(te_lab):.3f}; "
        f"chance balanced accuracy is {1 / n_classes:.3f}.",
        "",
        md_table(["rule", "accuracy", "balanced accuracy", "rare-class macro-F1"],
                 [[r, f"{e['accuracy']:.3f}", f"{e['balanced_accuracy']:.3f}",
                   f"{e['rare_macro_f1']:.3f}"] for r, e in evals.items()]),
        "",
        "## 1. Fraction of rounds each centre was rejected",
        "",
        md_table(header, [size_row] + [[r] + [pct(rejection.at[r, k]) for k in centres]
                                       for r in RULES]),
        "",
        "Mean effective weight per round (uniform share = "
        f"{1 / n_centers:.3f}; FedAvg shows the sample-count share):",
        "",
        md_table(header, [[r] + [f"{mean_weight.at[r, k]:.3f}" for k in centres] for r in RULES]),
        "",
        f"## 2. Is centre {target} rejected more often than the average centre?",
        "",
        md_table(
            ["rule", f"centre {target}", "mean of other 5", "mean of all 6",
             "difference vs other 5", "ratio vs other 5", "shortfall", "verdict"],
            [[r, pct(c["r_t"]), pct(c["r_others"]), pct(c["r_all"]),
              f"{100 * (c['r_t'] - c['r_others']):+.1f} pp",
              f"{c['r_t'] / c['r_others']:.2f}x" if c["r_others"] > 0 else "n/a",
              f"{c['shortfall']:.2f}", c["verdict"]]
             for r, c in comparison.items()]),
        "",
        "## 3. Centres ranked by rejection rate",
        "",
        "1 = most rejected. Ties broken by lower mean effective weight.",
        "",
        md_table(["rule"] + [f"rank {i}" for i in range(1, n_centers + 1)],
                 [[r] + [f"centre {k} ({pct(rejection.at[r, k])})"
                         for k in sorted(centres, key=lambda k: ranks[r][k])]
                  for r in RULES]),
        "",
        "## Limitations",
        "",
        f"- One seed and {N_ROUNDS} rounds. Krum selects one client per round, so each centre's "
        f"Krum rejection rate rests on {N_ROUNDS} binary outcomes.",
        f"- A small CNN on {IMAGE_SIZE}px images, not DenseNet-121 at 224px. Update geometry, and "
        "therefore what the robust rules see, may differ for the real model.",
        "- Plain cross-entropy. The project config uses a class-balanced loss, which increases "
        "rare-class gradients and could change how distinctive the specialist's update is.",
        "- Fixed local steps per centre. Small centres cycle through their data several times "
        "per round and large centres see only part of theirs, which is itself a source of "
        "update differences unrelated to rare classes.",
        "- Rejection of a centre is not attributed to its rare-class content here; other "
        "differences between centres (size, other classes, imaging) can drive it too.",
        "",
        "Per-round selections and weights: `results/byzantine_conflict.csv`.",
        "",
    ]
    (out_dir / "byzantine_conflict.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n  wrote {out_dir / 'byzantine_conflict.csv'}")
    print(f"  wrote {out_dir / 'byzantine_conflict.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
