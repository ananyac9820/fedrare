#!/usr/bin/env python3
"""
Tier B confirmation (docs/DEVIATIONS.md D8): does the sleeper finding hold with full DenseNet-121
fine-tuning instead of a head on saved features?

Full DenseNet-121 (ImageNet init) trained federated on the natural split S1, all six centres,
20 rounds x 50 local Adam steps (batch 32, lr 1e-4, class-balanced loss, train-time augmentation),
seed 42, through the same aggregation interface as Tier A. Three runs, fixed in D8:

    fedavg          no attack
    fedavg          A2 sleeper (centre 1 honest rounds 1-15, A1 from round 16)
    fedavg_clipped  A2 sleeper

The attack is the same as Tier A's (src/attacks/attacks.py): rare labels flipped to nevus, then the
two rare classifier rows rewritten to 3x the largest honest row norm along minus the attacker's mean
DenseNet feature. Test metrics after rounds 5, 10, 15, 20 on the pooled test split.

Each round's global state is checkpointed to data/tier_b/, so an interrupted run resumes.
About 2.1 h per run on an Apple M-series GPU (MPS); ~6.5 h for all three.

    python scripts/12_tier_b.py                 # all three runs
    python scripts/12_tier_b.py --smoke         # 1 round, 2 steps, 64 test images: checks the plumbing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from src.attacks.attacks import SLEEPER_HONEST_ROUNDS, craft_expert_rows, flip_labels
from src.data.loader import build_transforms, class_balanced_weights
from src.federated import baselines
from src.models.densenet import build_model
from src.utils.metrics import confusion, summarise
from src.utils.seed import get_device, set_seed

# --- Pre-registered (D8). Do not change after seeing results. ---
ROUNDS, LOCAL_STEPS, BATCH, LR, SEED = 20, 50, 32, 1e-4, 42
EVAL_ROUNDS = (5, 10, 15, 20)
RUNS = (("fedavg", False), ("fedavg", True), ("fedavg_clipped", True))   # (rule, sleeper attack?)
ATTACKER, RARE_IDS, N_CENTRES = 1, [5, 6], 6
OUT_DIR = ROOT / "data" / "tier_b"


def features(model, x):
    out = F.relu(model.features(x), inplace=True)
    return torch.flatten(F.adaptive_avg_pool2d(out, (1, 1)), 1)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class Data:
    def __init__(self, smoke: bool):
        from datasets import load_dataset
        self.train = load_dataset("flwrlabs/fed-isic2019", split="train")
        self.test = load_dataset("flwrlabs/fed-isic2019", split="test")
        self.labels = np.asarray(self.train["label"], dtype=np.int64)
        centres = np.asarray(self.train["center"], dtype=np.int64)
        self.rows = [np.flatnonzero(centres == k) for k in range(N_CENTRES)]
        self.test_n = 64 if smoke else len(self.test)
        self.tf_train, self.tf_eval = build_transforms(224, train=True), build_transforms(224, train=False)

    def batch(self, idx, labels=None):
        rows = self.train[[int(i) for i in idx]]
        x = torch.stack([self.tf_train(im.convert("RGB")) for im in rows["image"]])
        y = torch.as_tensor(self.labels[idx] if labels is None else labels, dtype=torch.long)
        return x, y


def local_train(model, global_state, data: Data, k, rnd, steps, device, attack: bool):
    model.load_state_dict(global_state)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rows = data.rows[k]
    labels = data.labels[rows]
    if attack:
        labels = flip_labels(torch.as_tensor(labels), RARE_IDS).numpy()
    weight = class_balanced_weights(np.bincount(labels, minlength=8)).to(device)
    rng = np.random.default_rng([SEED, rnd, k])
    for _ in range(steps):
        pick = rng.choice(len(rows), size=min(BATCH, len(rows)), replace=False)
        x, y = data.batch(rows[pick], labels[pick])
        opt.zero_grad()
        F.cross_entropy(model(x.to(device)), y.to(device), weight=weight).backward()
        opt.step()
    state = cpu_state(model)
    if attack:
        model.eval()
        with torch.no_grad():
            pick = rng.choice(len(rows), size=min(128, len(rows)), replace=False)
            x, _ = data.batch(rows[pick])
            feats = features(model, x.to(device)).cpu()
        state = craft_expert_rows(state, global_state, feats, RARE_IDS)
    return state


@torch.no_grad()
def evaluate(model, state, data: Data, device):
    model.load_state_dict(state)
    model.eval()
    preds, ys = [], []
    for s in range(0, data.test_n, 64):
        rows = data.test[s:min(s + 64, data.test_n)]
        x = torch.stack([data.tf_eval(im.convert("RGB")) for im in rows["image"]]).to(device)
        preds.append(model(x).argmax(1).cpu().numpy())
        ys.append(np.asarray(rows["label"]))
    y, p = np.concatenate(ys), np.concatenate(preds)
    out = summarise(y, p, RARE_IDS, 8)
    out["confusion"] = confusion(y, p, 8).tolist()
    return out


def run(rule, attack, data, device, rounds, steps, smoke):
    tag = f"{rule}_{'A2' if attack else 'none'}" + ("_smoke" if smoke else "")
    ckpt_dir = OUT_DIR / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    model = build_model(8, pretrained=True).to(device)
    global_state = cpu_state(model)
    sizes = [len(r) for r in data.rows]
    metrics, start = {}, 1
    done = sorted(ckpt_dir.glob("round_*.pt"))
    if done:
        ck = torch.load(done[-1])
        global_state, metrics, start = ck["state"], ck["metrics"], ck["round"] + 1
        print(f"  resuming {tag} from round {start}", flush=True)
    for r in range(start, rounds + 1):
        t0 = time.time()
        states = [local_train(model, global_state, data, k, r, steps, device,
                              attack and k == ATTACKER and r > SLEEPER_HONEST_ROUNDS)
                  for k in range(N_CENTRES)]
        global_state, _ = baselines.RULES[rule](states, sizes, global_state, r)
        if r in EVAL_ROUNDS or r == rounds:
            metrics[r] = evaluate(model, global_state, data, device)
        torch.save({"state": global_state, "metrics": metrics, "round": r}, ckpt_dir / f"round_{r:02d}.pt")
        for old in sorted(ckpt_dir.glob("round_*.pt"))[:-1]:
            old.unlink()
        m = metrics.get(r)
        print(f"  {tag} round {r:>2}/{rounds} ({time.time() - t0:.0f}s)"
              + (f" | balanced acc {m['balanced_accuracy']:.3f} | rare F1 {m['rare_macro_f1']:.3f}" if m else ""),
              flush=True)
    return {"rule": rule, "attack": "A2" if attack else "none",
            "metrics_by_round": {str(k): v for k, v in metrics.items()}, "final": metrics[rounds]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = get_device("cuda")
    rounds, steps = (1, 2) if args.smoke else (ROUNDS, LOCAL_STEPS)
    print(f"Device {device}; {rounds} rounds x {steps} steps; runs {RUNS}", flush=True)
    data = Data(args.smoke)
    results = [run(rule, attack, data, device, rounds, steps, args.smoke) for rule, attack in RUNS]
    record = {"entry": "D8", "tier": "B", "split": "s1", "seed": SEED, "rounds": rounds,
              "local_steps": steps, "lr": LR, "smoke": args.smoke, "runs": results}
    if not args.smoke:
        f1 = {f"{r['rule']}_{r['attack']}": r["final"]["rare_macro_f1"] for r in results}
        drop = f1["fedavg_none"] - f1["fedavg_A2"]
        recovered = f1["fedavg_clipped_A2"] - f1["fedavg_A2"]
        record["criterion"] = {"sleeper_drop": drop, "clipping_recovery": recovered,
                               "confirms": bool(drop >= 0.10 and recovered >= 0.5 * drop)}
        print(f"\nD8: sleeper drop {drop:.3f}, clipping recovers {recovered:.3f} -> "
              f"{'CONFIRMS' if record['criterion']['confirms'] else 'does not confirm'} the Tier A finding")
    name = "tier_b_smoke.json" if args.smoke else "tier_b.json"
    (ROOT / "results" / name).write_text(json.dumps(record, indent=2))
    print(f"wrote results/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
