#!/usr/bin/env python3
"""
Pre-registered re-test of the Byzantine conflict hypothesis, with a model that learns.

scripts/03_byzantine_conflict.py returned NO CONFLICT, but its small CNN barely learned
(balanced accuracy 0.15-0.22 against chance 0.125), so the robust rules were comparing
near-noise updates. This is a single re-test with three changes from script 03, and only
these three:

  1. DenseNet-121 pretrained on ImageNet (src/models/densenet.py)
  2. Class-balanced loss (src/data/loader.py: class_balanced_weights), computed per centre
     from that centre's own label counts
  3. 40 rounds instead of 15, repeated for 3 seeds

Everything else is taken from script 03 itself (imported, not copied): the 6 real centres,
the five aggregation rules and their settings (f, trim), local steps, batch size, learning
rate, and the rejection definition.

Consequences of change 1 that are not additional experimental changes:
  - Images at the dataset's stored 224px. Script 03's 32px downsample was a CPU-speed
    measure; a pretrained DenseNet needs full-size input. Still no augmentation.
  - DenseNet has BatchNorm running statistics, which are buffers, not parameters. The
    aggregation rule decides on parameter updates exactly as in script 03; the clients'
    BatchNorm statistics are then combined using the per-client weights that rule produced.
    Every client starts each round from the global statistics.
  - Mixed precision on CUDA, for speed. Aggregation runs on CPU so the rules in
    src/federated/robust.py are used unchanged.
  - The CSV gains a leading `seed` column.

PRE-REGISTERED - fixed before any run. Do not change after seeing results.

  Gate (checked first): balanced accuracy of the final global model on the pooled test
  split, averaged across seeds, must be >= 0.35 for both krum and multi_krum (the rules the
  verdict uses). Otherwise print "GATE FAILED - model still not learning, result is
  uninformative" and report no verdict.

  Verdict: a centre's rejection rate is the fraction of rounds it was not selected. For krum
  and multi_krum, average across seeds (a) centre 2's rejection rate and (b) the mean
  rejection rate of the other five centres. Pool the two rules by averaging (a) across them
  and (b) across them, then ratio = (a) / (b).
      CONFLICT CONFIRMED   ratio >= 1.5
      WEAK CONFLICT        1.1 <= ratio < 1.5
      NO CONFLICT          ratio < 1.1
  If the other five are never rejected, the ratio is infinite when centre 2 is rejected at
  all and 1.0 when it is not. Per-rule ratios are also reported, but only the pooled ratio
  sets the verdict. Krum on its own cannot exceed 1.25: it rejects 5 of 6 centres every
  round, so even a centre rejected every round leaves the other five at 80%.

Outputs to --out-dir (default results/):
    byzantine_conflict_strong.csv    seed, round, rule, client_id, selected, weight
    byzantine_conflict_strong.md     gate, per-seed and mean rejection tables, verdict
    strong_checkpoints/              per-run state, so a disconnected session resumes

Needs a GPU and runs for many hours. On Google Colab (Runtime > Change runtime type > GPU):

    !git clone https://github.com/ananyac9820/fedrare.git
    %cd fedrare
    !pip install -q datasets
    from google.colab import drive; drive.mount("/content/drive")   # so results survive
    !python scripts/04_byzantine_conflict_strong.py --out-dir /content/drive/MyDrive/fedrare_results

Re-running the same command after a disconnect resumes from the last completed round.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from src.data.loader import (IMAGENET_MEAN, IMAGENET_STD, class_balanced_weights,
                             load_split_columns)
from src.federated import robust
from src.models.densenet import build_model
from src.utils.config import load_config
from src.utils.metrics import summarise
from src.utils.seed import get_device, set_seed


def _load_script(filename: str):
    spec = importlib.util.spec_from_file_location(Path(filename).stem, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Everything not listed as a change is read from the earlier scripts, so it cannot drift:
# rules, f, trim, local steps, batch size and learning rate from 03; the rare-class and
# specialist definitions from 02.
s02 = _load_script("02_explore_data.py")
s03 = _load_script("03_byzantine_conflict.py")

# --- Changes from script 03 ---
SEEDS = (42, 43, 44)
N_ROUNDS = 40

# --- Pre-registered hypothesis, gate and verdict. Fixed before any run. ---
TARGET_CENTRE = 2
VERDICT_RULES = ("krum", "multi_krum")
GATE_BALANCED_ACC = 0.35
CONFIRMED_RATIO = 1.5
WEAK_RATIO = 1.1

EVAL_BATCH = 128
_MEAN = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)


# --------------------------------------------------------------------------- data

def load_images(split: str, cfg: dict, cache_dir: Path):
    """uint8 images at the stored 224px, cached as .npy after the first decode (~3.5 GB)."""
    size = cfg["dataset"]["image_size"]
    centers, labels = load_split_columns(split=split)
    path = cache_dir / f"fed_isic2019_{split}_{size}px.npy"
    if path.exists():
        images = np.load(path)
        if len(images) == len(labels):
            return images, centers, labels

    from datasets import load_dataset
    from PIL import Image

    ds = load_dataset(cfg["dataset"]["hf_repo"], split=split)
    images = np.empty((len(ds), size, size, 3), dtype=np.uint8)
    t0 = time.time()
    for i, row in enumerate(ds):
        img = row["image"].convert("RGB")
        if img.size != (size, size):
            img = img.resize((size, size), Image.BILINEAR)
        images[i] = np.asarray(img)
        if (i + 1) % 2000 == 0 or i + 1 == len(ds):
            print(f"    decoded {i + 1:,}/{len(ds):,} {split} images ({time.time() - t0:.0f}s)")

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, images)
    os.replace(tmp, path)
    return images, centers, labels


def to_device_batch(images: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(images)).to(device, non_blocking=True)
    x = x.permute(0, 3, 1, 2).float().div_(255.0)
    return (x - _MEAN.to(device)) / _STD.to(device)


# ---------------------------------------------------------------------- training

def autocast(use_amp: bool):
    return torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()


def float_buffers(model: nn.Module) -> dict[str, torch.Tensor]:
    """BatchNorm running statistics (num_batches_tracked is an int counter, left alone)."""
    return {name: buf for name, buf in model.named_buffers() if buf.is_floating_point()}


def local_update(model: nn.Module, global_params: torch.Tensor, global_buffers: dict,
                 images: np.ndarray, idx: np.ndarray, y: torch.Tensor,
                 class_weights: torch.Tensor, rng: np.random.Generator,
                 device: torch.device, scaler, use_amp: bool):
    """Train one centre from the global state.

    Returns (parameter update, BatchNorm statistics, mean loss), tensors on CPU.
    """
    # Pass a copy: vector_to_parameters makes each parameter a view into the vector it is
    # given, so training would otherwise write straight into global_params.
    vector_to_parameters(global_params.clone(), model.parameters())
    with torch.no_grad():
        for name, buf in float_buffers(model).items():
            buf.copy_(global_buffers[name])

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=s03.LR)
    n = len(idx)
    perm, pos = rng.permutation(n), 0
    loss_sum = torch.zeros((), device=device)
    for _ in range(s03.LOCAL_STEPS):
        if pos + s03.BATCH_SIZE > n:
            perm, pos = rng.permutation(n), 0
        batch = perm[pos:pos + s03.BATCH_SIZE]
        pos += s03.BATCH_SIZE

        x = to_device_batch(images[idx[batch]], device)
        target = y[torch.from_numpy(batch).to(device)]
        opt.zero_grad(set_to_none=True)
        with autocast(use_amp):
            logits = model(x)
        loss = F.cross_entropy(logits.float(), target, weight=class_weights)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        loss_sum += loss.detach()

    update = (parameters_to_vector(model.parameters()).detach() - global_params).float().cpu()
    stats = {name: buf.detach().cpu().clone() for name, buf in float_buffers(model).items()}
    return update, stats, float(loss_sum.item()) / s03.LOCAL_STEPS


@torch.no_grad()
def predict(model: nn.Module, images: np.ndarray, device: torch.device, use_amp: bool) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(images), EVAL_BATCH):
        with autocast(use_amp):
            preds.append(model(to_device_batch(images[i:i + EVAL_BATCH], device)).argmax(1).cpu())
    return torch.cat(preds).numpy()


def save_checkpoint(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def run_one(seed: int, rule: str, data: dict, n_classes: int, rare_ids: list[int],
            device: torch.device, use_amp: bool, ckpt_dir: Path):
    """One federated run for one (seed, rule). Returns (log rows, test metrics)."""
    ckpt_path = ckpt_dir / f"seed{seed}_{rule}.pt"
    n_centers = len(data["client_idx"])

    set_seed(seed)
    model = build_model(n_classes, pretrained=True).to(device)
    global_params = parameters_to_vector(model.parameters()).detach().clone()
    global_buffers = {n: b.detach().clone() for n, b in float_buffers(model).items()}
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rows, start = [], 1

    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ck.get("done"):
            print(f"  already complete, loaded from {ckpt_path.name}")
            return ck["rows"], ck["eval"]
        global_params = ck["global_params"].to(device)
        global_buffers = {n: b.to(device) for n, b in ck["global_buffers"].items()}
        scaler.load_state_dict(ck["scaler"])
        rows, start = ck["rows"], ck["round"] + 1
        print(f"  resuming from round {start} ({ckpt_path.name})")

    run_t0 = time.time()
    for rnd in range(start, N_ROUNDS + 1):
        t0 = time.time()
        updates, stats, losses = [], [], []
        for k in range(n_centers):
            # Same batch order per (seed, round, centre) under every rule.
            u, s, loss = local_update(
                model, global_params, global_buffers, data["images"], data["client_idx"][k],
                data["client_y"][k], data["client_w"][k],
                np.random.default_rng([seed, rnd, k]), device, scaler, use_amp)
            updates.append(u)
            stats.append(s)
            losses.append(loss)

        res = s03.aggregate(rule, torch.stack(updates), data["n_samples"])
        global_params = global_params + res.aggregate.to(device)
        w = torch.as_tensor(res.weights, dtype=torch.float32)
        global_buffers = {
            name: sum(w[k] * stats[k][name] for k in range(n_centers)).to(device)
            for name in global_buffers
        }
        for k in range(n_centers):
            rows.append({"seed": seed, "round": rnd, "rule": rule, "client_id": k,
                         "selected": bool(res.selected[k]), "weight": float(res.weights[k])})

        save_checkpoint(ckpt_path, {
            "done": False, "round": rnd, "rows": rows, "scaler": scaler.state_dict(),
            "global_params": global_params.cpu(),
            "global_buffers": {n: b.cpu() for n, b in global_buffers.items()},
        })
        rejected = [k for k in range(n_centers) if not res.selected[k]]
        done_here = rnd - start + 1
        eta = (time.time() - run_t0) / done_here * (N_ROUNDS - rnd) / 60
        print(f"  seed {seed} | {rule:<22} | round {rnd:>2}/{N_ROUNDS} | "
              f"loss {np.mean(losses):.3f} | rejected {rejected} | "
              f"{time.time() - t0:4.0f}s | run ETA {eta:5.1f} min")

    vector_to_parameters(global_params.clone(), model.parameters())
    with torch.no_grad():
        for name, buf in float_buffers(model).items():
            buf.copy_(global_buffers[name])
    metrics = summarise(data["test_labels"], predict(model, data["test_images"], device, use_amp),
                        rare_ids, n_classes)
    save_checkpoint(ckpt_path, {"done": True, "rows": rows, "eval": metrics})
    return rows, metrics


# ---------------------------------------------------------------------- analysis

def rejection_ratio(r_target: float, r_others: float) -> float:
    if r_others == 0:
        return float("inf") if r_target > 0 else 1.0
    return r_target / r_others


def verdict_for(ratio: float) -> str:
    if ratio >= CONFIRMED_RATIO:
        return "CONFLICT CONFIRMED"
    if ratio >= WEAK_RATIO:
        return "WEAK CONFLICT"
    return "NO CONFLICT"


def fmt_ratio(ratio: float) -> str:
    return "inf" if ratio == float("inf") else f"{ratio:.2f}x"


def check_gate(evals: dict) -> dict:
    """evals: {(seed, rule): metrics}. Returns balanced-accuracy table and pass/fail."""
    rules = list(dict.fromkeys(r for _, r in evals))
    bal = pd.DataFrame({s: {r: evals[(s, r)]["balanced_accuracy"] for r in rules} for s in SEEDS})
    means = {r: float(bal.loc[r].mean()) for r in VERDICT_RULES}
    return {"table": bal, "means": means,
            "passed": all(v >= GATE_BALANCED_ACC for v in means.values())}


def analyse_rejections(log: pd.DataFrame, n_centers: int) -> dict:
    rules = list(dict.fromkeys(log["rule"]))
    rej = 1.0 - log.groupby(["seed", "rule", "client_id"])["selected"].mean()
    per_seed = {s: rej.loc[s].unstack().loc[rules] for s in SEEDS}
    mean_rej = sum(per_seed.values()) / len(SEEDS)
    mean_weight = log.groupby(["rule", "client_id"])["weight"].mean().unstack().loc[rules]

    others = [k for k in range(n_centers) if k != TARGET_CENTRE]
    by_rule = {}
    for rule in VERDICT_RULES:
        r_t = float(np.mean([per_seed[s].at[rule, TARGET_CENTRE] for s in SEEDS]))
        r_o = float(np.mean([per_seed[s].loc[rule, others].mean() for s in SEEDS]))
        by_rule[rule] = {"r_t": r_t, "r_o": r_o, "ratio": rejection_ratio(r_t, r_o)}

    pooled_t = float(np.mean([v["r_t"] for v in by_rule.values()]))
    pooled_o = float(np.mean([v["r_o"] for v in by_rule.values()]))
    pooled_ratio = rejection_ratio(pooled_t, pooled_o)
    return {"per_seed": per_seed, "mean": mean_rej, "mean_weight": mean_weight,
            "by_rule": by_rule, "pooled_t": pooled_t, "pooled_o": pooled_o,
            "ratio": pooled_ratio, "verdict": verdict_for(pooled_ratio)}


def fedavg_vs_rare(log: pd.DataFrame, client_counts: np.ndarray, rare_ids: list[int],
                   class_names: dict) -> dict:
    t = TARGET_CENTRE
    fedavg_w = float(log[(log["rule"] == "fedavg") & (log["client_id"] == t)]["weight"].mean())
    per_class = [(class_names[c], int(client_counts[t, c]), int(client_counts[:, c].sum()))
                 for c in rare_ids]
    rare_t = int(client_counts[t, rare_ids].sum())
    rare_all = int(client_counts[:, rare_ids].sum())
    return {"fedavg_weight": fedavg_w, "per_class": per_class,
            "rare_t": rare_t, "rare_all": rare_all, "rare_share": rare_t / rare_all,
            "image_share": client_counts[t].sum() / client_counts.sum()}


# ------------------------------------------------------------------------ report

def rejection_md(table: pd.DataFrame, n_samples: list[int] | None = None) -> str:
    centres = list(table.columns)
    rows = [["*train images*"] + [f"{n:,}" for n in n_samples]] if n_samples else []
    rows += [[r] + [s03.pct(table.at[r, k]) for k in centres] for r in table.index]
    return s03.md_table(["rule"] + [f"centre {k}" for k in centres], rows)


def write_report(path: Path, gate: dict, analysis: dict | None, rare: dict,
                 evals: dict, n_samples: list[int]) -> None:
    t = TARGET_CENTRE
    headline = (analysis["verdict"] if gate["passed"]
                else "GATE FAILED - model still not learning, result is uninformative")
    rules = list(gate["table"].index)
    md = [
        "# Byzantine conflict: pre-registered re-test (DenseNet-121)",
        "",
        f"**Result: {headline}**",
        "",
        f"Hypothesis: standard Byzantine-robust aggregation rejects centre {t}, the rare-class "
        "specialist, more often than the other centres. No client is malicious, so every "
        "rejection is a rejection of an honest centre.",
        "",
        "## Setup",
        "",
        "Changes from `scripts/03_byzantine_conflict.py`, and only these:",
        "",
        "1. DenseNet-121 pretrained on ImageNet",
        "2. Class-balanced loss (effective-number weights from each centre's own label counts)",
        f"3. {N_ROUNDS} rounds instead of 15, seeds {', '.join(map(str, SEEDS))}",
        "",
        "Taken unchanged from script 03: 6 clients = the 6 real centres; rules "
        f"{', '.join(s03.RULES)}; f = {s03.F_BYZANTINE}, trim = {s03.TRIM}; "
        f"{s03.LOCAL_STEPS} local Adam steps per centre per round (batch {s03.BATCH_SIZE}, "
        f"lr {s03.LR}); rejection = not selected (krum, multi_krum) or effective weight below "
        f"{robust.NEAR_ZERO_FRACTION} x 1/6 (trimmed mean, median).",
        "",
        "Consequences of using DenseNet, not further changes: images at the stored 224px "
        "(script 03 downsampled to 32px for CPU speed), no augmentation; BatchNorm running "
        "statistics combined with each rule's per-client weights; mixed precision on GPU with "
        "aggregation on CPU so `src/federated/robust.py` runs unchanged.",
        "",
        "## Pre-registered criteria (fixed before running)",
        "",
        f"- **Gate:** balanced accuracy on the pooled test split, averaged across seeds, "
        f">= {GATE_BALANCED_ACC} for both {' and '.join(VERDICT_RULES)}.",
        f"- **Verdict:** for {' and '.join(VERDICT_RULES)}, average across seeds centre {t}'s "
        "rejection rate and the mean rejection rate of the other five centres; average each "
        f"across the two rules; ratio = centre {t} / other five. CONFLICT CONFIRMED if ratio "
        f">= {CONFIRMED_RATIO}, WEAK CONFLICT if >= {WEAK_RATIO}, otherwise NO CONFLICT.",
        "- Krum alone cannot exceed 1.25x (it rejects 5 of 6 centres every round).",
        "",
        "## Gate",
        "",
        "Balanced accuracy of the final global model (chance = 0.125):",
        "",
        s03.md_table(["rule"] + [f"seed {s}" for s in SEEDS] + ["mean", "gate rule?"],
                     [[r] + [f"{gate['table'].at[r, s]:.3f}" for s in SEEDS]
                      + [f"{gate['table'].loc[r].mean():.3f}",
                         "yes" if r in VERDICT_RULES else "no"] for r in rules]),
        "",
        f"Gate {'PASSED' if gate['passed'] else 'FAILED'}: "
        + ", ".join(f"{r} mean {v:.3f}" for r, v in gate["means"].items())
        + f" (threshold {GATE_BALANCED_ACC}).",
        "",
    ]

    if gate["passed"]:
        md += ["## Rejection rate per centre, per seed", ""]
        for s in SEEDS:
            md += [f"### Seed {s}", "", rejection_md(analysis["per_seed"][s], n_samples), ""]
        md += [
            "## Mean across seeds", "",
            rejection_md(analysis["mean"], n_samples), "",
            "Mean effective weight per round, across seeds (uniform share = 0.167):", "",
            s03.md_table(["rule"] + [f"centre {k}" for k in analysis["mean_weight"].columns],
                         [[r] + [f"{analysis['mean_weight'].at[r, k]:.3f}"
                                 for k in analysis["mean_weight"].columns]
                          for r in analysis["mean_weight"].index]), "",
            "## Verdict", "",
            s03.md_table(
                ["", f"centre {t} rejection", "other five (mean)", "ratio", "role"],
                [[r, s03.pct(v["r_t"]), s03.pct(v["r_o"]), fmt_ratio(v["ratio"]),
                  "reported only"] for r, v in analysis["by_rule"].items()]
                + [["**pooled (krum + multi_krum)**", s03.pct(analysis["pooled_t"]),
                    s03.pct(analysis["pooled_o"]), f"**{fmt_ratio(analysis['ratio'])}**",
                    "**sets verdict**"]]), "",
            f"**{analysis['verdict']}**", "",
        ]
    else:
        md += ["Per the pre-registration, no rejection tables or verdict are reported. "
               "Raw per-round selections are still in `byzantine_conflict_strong.csv`.", ""]

    md += [
        f"## Separate from the verdict: centre {t}'s FedAvg weight vs its rare-class share",
        "",
        "Train split, which is what FedAvg's sample-count weights are based on. This does not "
        "depend on whether the model learned.",
        "",
        s03.md_table(["", f"centre {t}", "all centres", f"centre {t} share"],
                     [[name, n_t, n_all, s03.pct(n_t / n_all)] for name, n_t, n_all in rare["per_class"]]
                     + [["all rare classes", rare["rare_t"], rare["rare_all"],
                         s03.pct(rare["rare_share"])],
                        ["all images", n_samples[t], sum(n_samples), s03.pct(rare["image_share"])]]),
        "",
        f"Mean FedAvg aggregation weight of centre {t}: **{rare['fedavg_weight']:.3f}**, against "
        f"a **{rare['rare_share']:.3f}** share of rare-class training images "
        f"({rare['rare_share'] / rare['fedavg_weight']:.2f}x its weight).",
        "",
        "## Test metrics per run",
        "",
        s03.md_table(["seed", "rule", "accuracy", "balanced accuracy", "rare-class macro-F1"],
                     [[s, r, f"{e['accuracy']:.3f}", f"{e['balanced_accuracy']:.3f}",
                       f"{e['rare_macro_f1']:.3f}"] for (s, r), e in evals.items()]),
        "",
        "## Limitations",
        "",
        f"- Learning rate, local steps and batch size are script 03's, chosen for a small CNN, "
        "not re-tuned for DenseNet (re-tuning would have been a fourth change).",
        "- Krum's rejection ratio is capped at 1.25x by construction, so the pooled verdict "
        "relies more heavily on multi-Krum.",
        "- BatchNorm statistics follow the rule's parameter decisions rather than being "
        "robustly aggregated themselves.",
        "- A higher rejection rate for centre 2 would not by itself show that its rare-class "
        "content is the cause; size and other distribution differences are not controlled.",
        "",
    ]
    path.write_text("\n".join(md), encoding="utf-8")


# -------------------------------------------------------------------------- main

def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)  # progress shows promptly on Colab
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="where results and checkpoints go (default: results/)")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache",
                        help="where decoded images are cached")
    args = parser.parse_args()

    cfg = load_config(ROOT / "configs" / "default.yaml")
    n_classes, n_centers = cfg["dataset"]["n_classes"], cfg["dataset"]["n_centers"]
    out_dir = args.out_dir or ROOT / cfg.get("output_dir", "results")
    ckpt_dir = out_dir / "strong_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = get_device("cuda")
    use_amp = device.type == "cuda"
    print(f"Device: {torch.cuda.get_device_name(0) if use_amp else 'CPU - no GPU found, this will be very slow'}")
    print(f"Results: {out_dir}")

    print("\nMeasuring class distribution with script 02's definitions...")
    df = s02.collect(cfg)
    rare_ids, _ = s02.identify_rare(df, cfg["dataset"]["head_ratio_divisor"])
    _, specialists = s02.rare_over_representation(df, rare_ids)
    if not rare_ids or any(v != f"centre_{TARGET_CENTRE}" for v in specialists.values()):
        print(f"ABORT: the pre-registered target is centre {TARGET_CENTRE}, but script 02's "
              f"definitions give specialists {specialists} (rare ids {rare_ids}). The "
              "hypothesis no longer matches the data, so nothing was run.")
        return 1
    print(f"  rare classes {rare_ids}; specialist for all of them: centre {TARGET_CENTRE}")

    print("\nLoading images (first run decodes and caches them)...")
    tr_img, tr_ctr, tr_lab = load_images("train", cfg, args.cache_dir)
    te_img, _, te_lab = load_images("test", cfg, args.cache_dir)
    client_idx = [np.flatnonzero(tr_ctr == k) for k in range(n_centers)]
    client_counts = np.stack([np.bincount(tr_lab[i], minlength=n_classes) for i in client_idx])
    data = {
        "images": tr_img, "test_images": te_img, "test_labels": te_lab.astype(int),
        "client_idx": client_idx, "n_samples": [len(i) for i in client_idx],
        "client_y": [torch.from_numpy(tr_lab[i].astype(np.int64)).to(device) for i in client_idx],
        "client_w": [class_balanced_weights(c).to(device) for c in client_counts],
    }
    print(f"  {len(tr_lab):,} train / {len(te_lab):,} test images; "
          f"train per centre {data['n_samples']}")
    for k in range(n_centers):
        print(f"  centre {k} class-balanced loss weights: "
              f"{np.round(data['client_w'][k].cpu().numpy(), 2)}")

    runs = list(itertools.product(SEEDS, s03.RULES))
    rows, evals = [], {}
    for i, (seed, rule) in enumerate(runs, 1):
        print(f"\n=== run {i}/{len(runs)}: seed {seed}, rule {rule} ===")
        run_rows, metrics = run_one(seed, rule, data, n_classes, rare_ids, device, use_amp, ckpt_dir)
        rows += run_rows
        evals[(seed, rule)] = metrics
        print(f"  seed {seed} {rule}: test accuracy {metrics['accuracy']:.3f}, balanced accuracy "
              f"{metrics['balanced_accuracy']:.3f} (gate: {'/'.join(VERDICT_RULES)} mean across "
              f"seeds >= {GATE_BALANCED_ACC})")

    log = pd.DataFrame(rows)[["seed", "round", "rule", "client_id", "selected", "weight"]]
    log.to_csv(out_dir / "byzantine_conflict_strong.csv", index=False)

    # Gate first: no verdict from a model that did not learn.
    gate = check_gate(evals)
    rare = fedavg_vs_rare(log, client_counts, rare_ids, cfg["dataset"]["class_names"])
    report = out_dir / "byzantine_conflict_strong.md"

    print("\n" + "=" * 78)
    print("GATE: balanced accuracy of the final global model")
    print("=" * 78)
    print(f"  {'rule':<24}" + "".join(f"{f'seed {s}':>10}" for s in SEEDS) + f"{'mean':>10}")
    for r in gate["table"].index:
        print(f"  {r:<24}" + "".join(f"{gate['table'].at[r, s]:>10.3f}" for s in SEEDS)
              + f"{gate['table'].loc[r].mean():>10.3f}")

    if not gate["passed"]:
        print("\nGATE FAILED - model still not learning, result is uninformative")
        write_report(report, gate, None, rare, evals, data["n_samples"])
        print(f"  wrote {report}")
        return 2
    print(f"  gate passed ({', '.join(f'{r} {v:.3f}' for r, v in gate['means'].items())})")

    analysis = analyse_rejections(log, n_centers)
    print("\n" + "=" * 78)
    print("REJECTION RATE PER CENTRE, MEAN ACROSS SEEDS")
    print("=" * 78)
    table = analysis["mean"]
    print(f"  {'rule':<24}" + "".join(f"{f'centre {k}':>9}" for k in table.columns))
    for r in table.index:
        print(f"  {r:<24}" + "".join(f"{s03.pct(table.at[r, k]):>9}" for k in table.columns))

    print("\n" + "=" * 78)
    print(f"CENTRE {TARGET_CENTRE} VS OTHER FIVE (mean across seeds)")
    print("=" * 78)
    for r, v in analysis["by_rule"].items():
        print(f"  {r:<24} {s03.pct(v['r_t']):>7} vs {s03.pct(v['r_o']):>7}   "
              f"ratio {fmt_ratio(v['ratio'])}   (reported only)")
    print(f"  {'pooled':<24} {s03.pct(analysis['pooled_t']):>7} vs "
          f"{s03.pct(analysis['pooled_o']):>7}   ratio {fmt_ratio(analysis['ratio'])}   (sets verdict)")

    print("\n" + "=" * 78)
    print(f"VERDICT: {analysis['verdict']}")
    print("=" * 78)
    print(f"\n  Separately: centre {TARGET_CENTRE} FedAvg weight {rare['fedavg_weight']:.3f} vs "
          f"rare-class image share {rare['rare_share']:.3f}")

    write_report(report, gate, analysis, rare, evals, data["n_samples"])
    print(f"  wrote {out_dir / 'byzantine_conflict_strong.csv'}")
    print(f"  wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
