#!/usr/bin/env python3
"""
Gate G0b retry - unfreeze DenseNet-121's last dense block, re-extract features, re-measure G0b.
Pre-registered in docs/DEVIATIONS.md D3 (and the 100-round amendment D4) before its first run.

G0b failed on 20 Sep (Tier A FedAvg balanced accuracy 0.415 against 0.45). The design doc's
retry: "unfreeze the last dense block and re-extract features. One retry."

    1. Cache the frozen trunk's output (everything up to transition3: 512 x 7 x 7 per image)
       for train and test, once. data/features/trunk_transition3_<split>.npy (float16).
    2. Train denseblock4 + norm5 + a fresh classifier FEDERATED on S1 training images:
       FedAvg through src/federated/baselines.fedavg (BatchNorm running stats averaged),
       20 rounds x 50 local steps, batch 32, Adam (block 1e-4, classifier 5e-4),
       class-balanced loss, seed 42. The test split is never used for training.
    3. Re-extract 1024-d features with the trained block in eval mode:
       data/features/fed_isic2019_densenet121_ft4_{train,test}.npz (+ .json provenance).
    4. G0b, measured exactly as scripts/07: Tier A FedAvg on S1, TierAConfig defaults
       (40 rounds), seeds 42/43/44, final-round balanced accuracy, mean >= 0.45 passes.
       Reported alongside (D4, amendment): the same at 100 rounds, on both feature sets.

Output: results/gate_g0b_retry.json

    python scripts/08_g0b_retry.py          # ~10-15 min on Apple MPS; GPU similar; CPU slow
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.data.features import FEATURES_DIR, VARIANTS, load_features
from src.data.loader import build_transforms, class_balanced_weights
from src.federated import baselines
from src.federated.tier_a import TierAConfig, make_clients, run_federated
from src.utils.config import load_config
from src.utils.seed import get_device, set_seed

# --- Pre-registered (docs/DEVIATIONS.md D3, D4). Do not change after seeing results. ---
FT_ROUNDS = 20
FT_LOCAL_STEPS = 50
FT_BATCH = 32
FT_LR_BLOCK = 1e-4
FT_LR_HEAD = 5e-4
FT_SEED = 42
G0B_SEEDS = (42, 43, 44)
G0B_THRESHOLD = 0.45
AMENDED_ROUNDS = 100          # D4
RARE_IDS = [5, 6]             # scripts/02_explore_data.py, results/rare_classes.yaml


class Top(nn.Module):
    """DenseNet-121 after transition3: denseblock4 -> norm5 -> ReLU -> pool -> classifier."""

    def __init__(self, densenet: nn.Module, n_classes: int):
        super().__init__()
        self.denseblock4 = densenet.features.denseblock4
        self.norm5 = densenet.features.norm5
        self.classifier = nn.Linear(1024, n_classes)

    def features(self, x):
        out = F.relu(self.norm5(self.denseblock4(x)), inplace=True)
        return torch.flatten(F.adaptive_avg_pool2d(out, (1, 1)), 1)

    def forward(self, x):
        return self.classifier(self.features(x))


def trunk_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    f = model.features
    for name in ("conv0", "norm0", "relu0", "pool0", "denseblock1", "transition1",
                 "denseblock2", "transition2", "denseblock3", "transition3"):
        x = getattr(f, name)(x)
    return x


def cache_trunk(split: str, cfg: dict, model: nn.Module, device, batch_size: int = 64):
    path = FEATURES_DIR / f"trunk_transition3_{split}.npy"
    meta = FEATURES_DIR / f"trunk_transition3_{split}_meta.npz"
    if path.exists() and meta.exists():
        with np.load(meta) as z:
            return np.load(path, mmap_mode="r"), z["labels"], z["centers"]
    from datasets import load_dataset
    ds = load_dataset(cfg["dataset"]["hf_repo"], split=split)
    n = len(ds)
    transform = build_transforms(cfg["dataset"]["image_size"], train=False)
    out = np.lib.format.open_memmap(path.with_suffix(".tmp.npy"), mode="w+",
                                    dtype=np.float16, shape=(n, 512, 7, 7))
    t0 = time.time()
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        x = torch.stack([transform(img.convert("RGB")) for img in ds[start:stop]["image"]])
        with torch.inference_mode():
            out[start:stop] = trunk_forward(model, x.to(device)).half().cpu().numpy()
        if stop == n or (stop // batch_size) % 40 == 0:
            print(f"  trunk {split}: {stop:>6,}/{n:,} ({stop / (time.time() - t0):.0f} img/s)",
                  flush=True)
    out.flush()
    del out
    path.with_suffix(".tmp.npy").rename(path)
    labels = np.asarray(ds["label"], dtype=np.int64)
    centers = np.asarray(ds["center"], dtype=np.int64)
    np.savez(meta, labels=labels, centers=centers)
    return np.load(path, mmap_mode="r"), labels, centers


def cpu_state(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def local_train(top: Top, global_state: dict, x: torch.Tensor, y: torch.Tensor,
                rows: np.ndarray, rng: np.random.Generator, device) -> dict:
    """Local training on this client's rows of the full cached arrays (x, y)."""
    top.load_state_dict(global_state)
    top.train()
    opt = torch.optim.Adam([
        {"params": [p for n, p in top.named_parameters() if not n.startswith("classifier")],
         "lr": FT_LR_BLOCK},
        {"params": top.classifier.parameters(), "lr": FT_LR_HEAD}])
    weight = class_balanced_weights(np.bincount(y[rows].numpy(), minlength=8)).to(device)
    n = len(rows)
    perm, pos = rng.permutation(n), 0
    for _ in range(FT_LOCAL_STEPS):
        if pos + FT_BATCH > n:
            perm, pos = rng.permutation(n), 0
        idx = torch.from_numpy(rows[perm[pos:pos + FT_BATCH]])
        pos += FT_BATCH
        xb, yb = x[idx].to(device).float(), y[idx].to(device)
        opt.zero_grad()
        F.cross_entropy(top(xb), yb, weight=weight).backward()
        opt.step()
    return cpu_state(top)


@torch.no_grad()
def extract(top: Top, x: torch.Tensor, device, batch: int = 256) -> np.ndarray:
    top.eval()
    return np.concatenate([top.features(x[i:i + batch].to(device).float()).cpu().numpy()
                           for i in range(0, len(x), batch)]).astype(np.float32)


def g0b_runs(variant: str, rounds: int) -> dict:
    train, test = load_features("train", variant=variant), load_features("test", variant=variant)
    clients = make_clients(train.features, train.labels, train.centers, 6)
    cfg = replace(TierAConfig(), rounds=rounds)
    finals, curves = [], []
    for seed in G0B_SEEDS:
        res = run_federated(lambda: baselines.fedavg, clients, torch.from_numpy(test.features),
                            test.labels, cfg, seed, RARE_IDS)
        finals.append(res["final"])
        curves.append([m["balanced_accuracy"] for m in res["round_metrics"]])
    bal = np.array([f["balanced_accuracy"] for f in finals])
    return {"variant": variant, "rounds": rounds, "per_seed": dict(zip(map(str, G0B_SEEDS), bal.tolist())),
            "balanced_accuracy_mean": float(bal.mean()), "balanced_accuracy_std": float(bal.std()),
            "rare_macro_f1_mean": float(np.mean([f["rare_macro_f1"] for f in finals])),
            "per_class_f1_mean": {c: float(np.mean([f["per_class_f1"][c] for f in finals]))
                                  for c in range(8)},
            "curve_mean": np.mean(curves, axis=0).tolist()}


def main() -> int:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    device = get_device("cuda")
    out_dir = ROOT / cfg.get("output_dir", "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | torch {torch.__version__}")

    set_seed(FT_SEED)
    densenet = torchvision.models.densenet121(
        weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1).to(device).eval()

    # 1. frozen trunk, cached
    trunk = {}
    for split in ("train", "test"):
        print(f"\n[1] trunk features, {split}")
        arr, labels, centers = cache_trunk(split, cfg, densenet, device)
        trunk[split] = (torch.from_numpy(np.ascontiguousarray(arr)), torch.from_numpy(labels),
                        centers)

    # 2. federated fine-tuning of denseblock4 + norm5 (+ throwaway classifier), S1 train only
    print(f"\n[2] federated fine-tuning: {FT_ROUNDS} rounds x {FT_LOCAL_STEPS} steps, 6 centres")
    set_seed(FT_SEED)
    top = Top(densenet, cfg["dataset"]["n_classes"]).to(device)
    global_state = cpu_state(top)
    xtr, ytr, ctr = trunk["train"]
    idx = [np.flatnonzero(ctr == k) for k in range(6)]
    sizes = [len(i) for i in idx]
    t0 = time.time()
    for r in range(1, FT_ROUNDS + 1):
        states = []
        for k in range(6):
            states.append(local_train(top, global_state, xtr, ytr, idx[k],
                                      np.random.default_rng([FT_SEED, r, k]), device))
        global_state, _ = baselines.fedavg(states, sizes, global_state, r)
        print(f"  round {r:>2}/{FT_ROUNDS}  ({time.time() - t0:.0f}s)", flush=True)
    top.load_state_dict(global_state)
    ft_seconds = time.time() - t0

    # 3. re-extract features with the trained block
    print("\n[3] re-extracting features")
    meta = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "what": "DenseNet-121 (ImageNet) with denseblock4 + norm5 fine-tuned by FedAvg on "
                    "S1 training images; docs/DEVIATIONS.md D3",
            "fine_tuning": {"rounds": FT_ROUNDS, "local_steps": FT_LOCAL_STEPS,
                            "batch": FT_BATCH, "lr_block": FT_LR_BLOCK, "lr_head": FT_LR_HEAD,
                            "seed": FT_SEED, "seconds": round(ft_seconds, 1),
                            "client_sizes": sizes},
            "device": str(device), "versions": {"torch": torch.__version__,
                                                "torchvision": torchvision.__version__}}
    for split in ("train", "test"):
        x, y, c = trunk[split]
        feats = extract(top, x, device)
        if not np.isfinite(feats).all():
            raise RuntimeError(f"non-finite features in {split}")
        path = FEATURES_DIR / f"{VARIANTS['ft4']}_{split}.npz"
        np.savez(path, features=feats, labels=y.numpy(), centers=c)
        meta[split] = {"file": path.name, "n_images": int(len(y))}
        print(f"  wrote {path}")
    (FEATURES_DIR / f"{VARIANTS['ft4']}.json").write_text(json.dumps(meta, indent=2))

    # 4. G0b retry (decides) + D4 amendment (reported)
    print("\n[4] G0b")
    retry = g0b_runs("ft4", TierAConfig().rounds)
    amended = {v: g0b_runs(v, AMENDED_ROUNDS) for v in ("frozen", "ft4")}
    passed = retry["balanced_accuracy_mean"] >= G0B_THRESHOLD
    record = {"gate": "G0b-retry", "passed": passed, "threshold": G0B_THRESHOLD,
              "statistic": retry["balanced_accuracy_mean"],
              "definition": "D3: Tier A FedAvg on S1 over ft4 features, 40 rounds, final-round "
                            "balanced accuracy on the pooled test split, mean over seeds 42-44",
              "retry": retry, "amendment_D4_100_rounds": amended, "fine_tuning": meta}
    (out_dir / "gate_g0b_retry.json").write_text(json.dumps(record, indent=2))
    bar = "=" * 78
    print(f"\n{bar}\nG0b retry (D3, 40 rounds, ft4): {'PASS' if passed else 'FAIL'}  "
          f"balanced acc {retry['balanced_accuracy_mean']:.3f} ± {retry['balanced_accuracy_std']:.3f}"
          f"  (threshold {G0B_THRESHOLD})")
    for v, a in amended.items():
        print(f"  D4 amendment, {AMENDED_ROUNDS} rounds, {v:<6}: balanced acc "
              f"{a['balanced_accuracy_mean']:.3f} ± {a['balanced_accuracy_std']:.3f}, "
              f"rare macro-F1 {a['rare_macro_f1_mean']:.3f}")
    print(bar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
