#!/usr/bin/env python3
"""
Prepare everything the live demo needs: a saved model, and the sample images it serves.

The Tier A runs never saved a checkpoint, so this retrains the head under exactly the
configuration behind results/tier_a_s1_fedavg.csv - FedAvg, natural split S1, seed 42,
TierAConfig defaults - and writes it to models/. Same code path as the reported run
(src/federated/tier_a.run_federated), so the demo serves the model on the results page
rather than a second, differently-trained one. The script prints the final metrics next
to the CSV's last round so any drift is visible immediately.

It also picks the eight held-out test images the demo offers and writes them out as files.
The selection rule is fixed before the model is consulted: the first images of each chosen
class in Hugging Face test-split order. Nothing is picked for being predicted correctly.

    python demo/prepare_demo.py            # ~2-4 min on a laptop CPU
    python demo/prepare_demo.py --skip-train   # only re-cut the sample images

Outputs:
    models/tier_a_s1_fedavg_head.pt   head state_dict + provenance metadata
    demo/samples/<id>.png             eight test images (lossless: the demo can then be
                                      checked against the saved training features)
    demo/samples/samples.json         id, file, true label, split row index
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.federated import baselines
from src.federated.tier_a import TierAConfig, evaluate, make_clients, run_federated
from src.utils.config import load_config

FEATURES = ROOT / "data" / "features"
PREFIX = "fed_isic2019_densenet121"
MODEL_PATH = ROOT / "models" / "tier_a_s1_fedavg_head.pt"
SAMPLE_DIR = ROOT / "demo" / "samples"
SEED = 42

# Which classes the eight samples cover, in the order they appear on the page.
# Two Dermatofibroma and two Vascular lesion (the two rare classes this project is about),
# then four of the common classes for contrast.
SAMPLE_PLAN = [(5, 2), (6, 2), (0, 1), (1, 1), (2, 1), (4, 1)]


def load_split(split: str):
    path = FEATURES / f"{PREFIX}_{split}.npz"
    if not path.exists():
        raise SystemExit(
            f"missing {path}\nRun: python scripts/05_extract_features.py"
        )
    d = np.load(path)
    return d["features"], d["labels"], d["centers"]


def train(cfg_yaml: dict) -> dict:
    """Retrain the Tier A head with FedAvg on S1 and save it. Returns the saved payload."""
    n_classes = cfg_yaml["dataset"]["n_classes"]
    n_centers = cfg_yaml["dataset"]["n_centers"]
    rare_ids = [5, 6]

    train_x, train_y, train_c = load_split("train")
    test_x, test_y, _ = load_split("test")
    clients = make_clients(train_x, train_y, train_c, n_centers)
    cfg = TierAConfig()

    print(f"Training the Tier A head: FedAvg, S1, seed {SEED}, {cfg.rounds} rounds, "
          f"{len(clients)} centres ({[c.size for c in clients]} images)")
    result = run_federated(
        lambda: baselines.fedavg,
        clients,
        torch.from_numpy(np.ascontiguousarray(test_x)),
        test_y,
        cfg,
        seed=SEED,
        rare_ids=rare_ids,
        verbose=True,
    )
    final = result["final"]

    # Re-evaluate the exact state we are about to save, rather than trusting the loop's
    # last log line: if the two ever disagreed, the checkpoint would be the wrong thing.
    state = result["final_state"]
    metrics = evaluate(state, torch.from_numpy(np.ascontiguousarray(test_x)), test_y,
                       rare_ids, n_classes)
    payload = {
        "state_dict": {k: v.cpu() for k, v in state.items()},
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "what": "Tier A classifier head (single linear layer) for frozen DenseNet-121 "
                    "features. Drop-in replacement for densenet121.classifier.",
            "training": "FedAvg over the 6 real Fed-ISIC2019 centres, natural split S1, "
                        "class-balanced local loss",
            "config": result["config"],
            "seed": SEED,
            "rule": "fedavg",
            "feature_dim": cfg.feature_dim,
            "n_classes": n_classes,
            "class_names": [cfg_yaml["dataset"]["class_names"][i] for i in range(n_classes)],
            "preprocessing": "src/data/loader.build_transforms(224, train=False): "
                             "resize 224x224, ToTensor, ImageNet mean/std",
            "backbone": "torchvision densenet121, weights IMAGENET1K_V1, frozen, eval mode",
            "test_metrics": {k: v for k, v in metrics.items() if k != "confusion"},
            "versions": {"torch": torch.__version__, "numpy": np.__version__,
                         "python": platform.python_version()},
        },
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, MODEL_PATH)
    print(f"\nSaved {MODEL_PATH.relative_to(ROOT)} "
          f"({MODEL_PATH.stat().st_size / 1e3:.0f} kB)")
    print(f"  balanced accuracy {final['balanced_accuracy']:.4f} | "
          f"rare macro-F1 {final['rare_macro_f1']:.4f}")
    return payload


def compare_with_csv(payload: dict) -> None:
    """Print the saved model's metrics next to the reported run, so drift is visible."""
    csv = ROOT / "results" / "tier_a_s1_fedavg.csv"
    if not csv.exists():
        return
    rows = [r.split(",") for r in csv.read_text(encoding="utf-8").strip().splitlines()]
    head, last = rows[0], rows[-1]
    rec = dict(zip(head, last))
    got = payload["meta"]["test_metrics"]
    print("\n  against results/tier_a_s1_fedavg.csv (last round):")
    for key in ("balanced_accuracy", "rare_macro_f1", "accuracy", "macro_f1"):
        print(f"    {key:<20} saved {got[key]:.4f}   reported {float(rec[key]):.4f}")


def cut_samples(cfg_yaml: dict) -> list[dict]:
    """Write the eight sample images to demo/samples/ with their true labels."""
    from datasets import load_dataset

    names = cfg_yaml["dataset"]["class_names"]
    ds = load_dataset(cfg_yaml["dataset"]["hf_repo"], split="test")
    labels = np.asarray(ds["label"], dtype=int)
    centers = np.asarray(ds["center"], dtype=int)

    chosen: list[dict] = []
    for class_id, count in SAMPLE_PLAN:
        # Dataset order, but spread across hospitals where the class allows it: the project
        # is about six centres, so a demo drawing every image from centre 0 would be a poor
        # picture of the data. The rule uses only the class and centre columns - the model is
        # never consulted, so nothing here is chosen for being classified correctly.
        candidates = list(np.flatnonzero(labels == class_id))
        rows, used = [], set()
        for row in candidates:
            if len(rows) == count:
                break
            if centers[row] not in used:
                rows.append(row)
                used.add(centers[row])
        rows += [r for r in candidates if r not in rows][:count - len(rows)]
        for row in sorted(rows):
            chosen.append({
                "id": f"sample_{len(chosen) + 1:02d}",
                "row": int(row),
                "true_label": int(class_id),
                "true_label_name": names[class_id],
                "centre": int(centers[row]),
            })

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for old in list(SAMPLE_DIR.glob("*.png")) + list(SAMPLE_DIR.glob("*.jpg")):
        old.unlink()
    for s in chosen:
        image = ds[s["row"]]["image"].convert("RGB")
        # PNG, not JPEG: lossless means the file feeds the backbone the same pixels the
        # feature extraction saw, so demo/check_demo.py can verify the two agree exactly.
        path = SAMPLE_DIR / f"{s['id']}.png"
        image.save(path)
        s["file"] = path.name
        s["size"] = list(image.size)

    meta = {
        "note": "Held-out Fed-ISIC2019 test-split images. Selection rule fixed before the "
                "model was run: for each listed class, the earliest images in dataset order, "
                "preferring different hospitals. "
                "No image was chosen for being classified correctly.",
        "split": "test",
        "selection": [{"class": names[c], "count": n} for c, n in SAMPLE_PLAN],
        "samples": chosen,
    }
    (SAMPLE_DIR / "samples.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote {len(chosen)} sample images to {SAMPLE_DIR.relative_to(ROOT)}")
    for s in chosen:
        print(f"  {s['id']}  {s['true_label_name']:<22} centre {s['centre']}  row {s['row']}")
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the live demo's model and samples.")
    parser.add_argument("--skip-train", action="store_true",
                        help="keep the existing checkpoint, only re-cut the sample images")
    parser.add_argument("--skip-samples", action="store_true")
    args = parser.parse_args()

    cfg_yaml = load_config(ROOT / "configs" / "default.yaml")

    if not args.skip_train:
        payload = train(cfg_yaml)
        compare_with_csv(payload)
    if not args.skip_samples:
        cut_samples(cfg_yaml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
