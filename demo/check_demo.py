#!/usr/bin/env python3
"""
Check the demo before presenting it, and print what each sample image does.

Three things, in order:

  1. Preprocessing. Each sample image is pushed through the server's own pipeline and the
     resulting 1024-d feature is compared against the feature saved during training for
     that same dataset row. If the two agree, the demo is feeding the model exactly what
     training fed it - the failure mode that is otherwise invisible, because a mismatched
     normalisation still returns confident-looking predictions.
  2. Predictions. Every sample through the model, printed as a table with its true label.
     Wrong answers are marked, not hidden - a 41.7% balanced-accuracy model gets some wrong.
  3. The HTTP path (with --server URL): the same images through the running server, checked
     against the local numbers, so the page cannot be showing something different.

    python demo/check_demo.py
    python demo/check_demo.py --server http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from src.data.loader import build_transforms
from src.models.densenet import build_model, densenet_features

MODEL_PATH = ROOT / "models" / "tier_a_s1_fedavg_head.pt"
SAMPLE_DIR = ROOT / "demo" / "samples"
TEST_FEATURES = ROOT / "data" / "features" / "fed_isic2019_densenet121_test.npz"
TOLERANCE = 1e-3   # float32 through a 121-layer network; anything larger is a real mismatch


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the live demo end to end.")
    parser.add_argument("--server", default=None, help="also check a running server")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"no model at {MODEL_PATH} - run: python demo/prepare_demo.py")
    payload = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    classes = meta["class_names"]

    torch.set_grad_enabled(False)
    backbone = build_model(meta["n_classes"], pretrained=True)
    backbone.classifier = torch.nn.Identity()
    backbone.eval()
    head = torch.nn.Linear(meta["feature_dim"], meta["n_classes"])
    head.load_state_dict({"weight": payload["state_dict"]["classifier.weight"],
                          "bias": payload["state_dict"]["classifier.bias"]})
    head.eval()
    transform = build_transforms(224, train=False)

    samples = json.loads((SAMPLE_DIR / "samples.json").read_text(encoding="utf-8"))["samples"]
    saved = np.load(TEST_FEATURES) if TEST_FEATURES.exists() else None

    print(f"Model: {meta['training']}")
    print(f"  trained {meta['created_utc']}, {meta['config']['rounds']} rounds, seed "
          f"{meta['seed']}")
    m = meta["test_metrics"]
    print(f"  on the held-out test set: balanced accuracy {m['balanced_accuracy']:.3f}, "
          f"rare macro-F1 {m['rare_macro_f1']:.3f}, accuracy {m['accuracy']:.3f}")

    print("\n1. Preprocessing matches training")
    rows = []
    worst = 0.0
    for s in samples:
        x = transform(Image.open(SAMPLE_DIR / s["file"]).convert("RGB")).unsqueeze(0)
        feat = densenet_features(backbone, x)[0]
        probs = torch.softmax(head(feat.unsqueeze(0)), dim=1)[0]
        top = int(probs.argmax())
        diff = None
        if saved is not None:
            diff = float(np.abs(feat.numpy() - saved["features"][s["row"]]).max())
            worst = max(worst, diff)
            assert int(saved["labels"][s["row"]]) == s["true_label"], "row/label mismatch"
        rows.append({**s, "pred": top, "conf": float(probs[top]),
                     "true_conf": float(probs[s["true_label"]]), "diff": diff})

    if saved is None:
        print("   skipped: data/features/..._test.npz not present (extract features to check)")
    elif worst <= TOLERANCE:
        print(f"   OK - every sample's features match the ones training used "
              f"(largest difference {worst:.2e}, tolerance {TOLERANCE:g})")
    else:
        print(f"   MISMATCH - largest difference {worst:.2e} exceeds {TOLERANCE:g}. "
              f"The demo is not preprocessing images the way training did.")
        return 1

    print("\n2. What each sample image does\n")
    header = f"{'image':<12} {'hospital':<9} {'true label':<22} {'predicted':<22} {'conf':>6}  result"
    print(header)
    print("-" * len(header))
    for r in rows:
        ok = r["pred"] == r["true_label"]
        mark = "correct" if ok else f"wrong (true class got {r['true_conf'] * 100:4.1f}%)"
        print(f"{r['id']:<12} {'centre ' + str(r['centre']):<9} {r['true_label_name']:<22} "
              f"{classes[r['pred']]:<22} {r['conf'] * 100:5.1f}%  {mark}")
    n_ok = sum(r["pred"] == r["true_label"] for r in rows)
    rare = [r for r in rows if r["true_label"] in (5, 6)]
    print(f"\n   {n_ok}/{len(rows)} correct, of which rare classes "
          f"{sum(r['pred'] == r['true_label'] for r in rare)}/{len(rare)}")

    if args.server:
        print(f"\n3. The running server at {args.server}")
        import urllib.request

        try:
            with urllib.request.urlopen(f"{args.server}/health", timeout=10) as resp:
                health = json.loads(resp.read())
            print(f"   /health: ready, serving {health['model']['rule']} "
                  f"({health['model']['rounds']} rounds)")
        except Exception as exc:
            print(f"   could not reach the server: {exc}")
            print(f"   start it with: python demo/server.py")
            return 1

        agree = True
        for r in rows:
            body = json.dumps({"sample_id": r["id"]}).encode()
            req = urllib.request.Request(f"{args.server}/predict", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                got = json.loads(resp.read())
            same = (got["predicted"]["index"] == r["pred"]
                    and abs(got["predicted"]["confidence"] - r["conf"]) < 1e-4)
            agree &= same
            if not same:
                print(f"   {r['id']}: server says {got['predicted']['name']} "
                      f"{got['predicted']['confidence']:.3f}, local says "
                      f"{classes[r['pred']]} {r['conf']:.3f}")
        print("   /predict: identical to the local numbers for all eight samples" if agree
              else "   /predict: DISAGREES with the local numbers")
        return 0 if agree else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
