#!/usr/bin/env python3
"""
Live demo backend: the trained Tier A model behind three HTTP endpoints.

    python demo/prepare_demo.py     # once: trains the head, cuts the sample images
    python demo/server.py           # then: serves on http://127.0.0.1:8000

Endpoints
    GET  /health              is the model loaded, and which one
    GET  /samples             the eight held-out test images, with their true labels
    GET  /samples/<file>      one of those images
    POST /predict             multipart "image" file, or JSON {"sample_id": "sample_01"}
                              -> predicted class + probability for all eight classes

The model is the real one: frozen ImageNet DenseNet-121 as the feature extractor, and the
Tier A classifier head trained by FedAvg across the six centres (models/, written by
prepare_demo.py). No part of this file re-implements preprocessing - build_transforms()
and densenet_features() are imported from the same modules the training pipeline used, so
the demo cannot silently drift from training. demo/check_demo.py verifies that end to end
by comparing what this server computes against the features saved during training.

One deliberate difference: Fed-ISIC2019's images arrive from FLamby already colour-constancy
corrected, so a sample image must NOT be corrected twice. An uploaded photo has not been
corrected, so it gets the same Shades-of-Gray correction FLamby applies. Every response says
which path it took under "preprocessing".
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image

from src.data.loader import build_transforms
from src.models.densenet import build_model, densenet_features

MODEL_PATH = ROOT / "models" / "tier_a_s1_fedavg_head.pt"
SAMPLE_DIR = ROOT / "demo" / "samples"
IMAGE_SIZE = 224
MAX_UPLOAD_MB = 12

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
CORS(app)  # the Next.js dev server calls this from http://localhost:3000

STATE: dict = {"ready": False, "error": None}


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------

def load_model() -> None:
    """Load the backbone and the trained head once, at startup."""
    if not MODEL_PATH.exists():
        STATE["error"] = (f"no model at {MODEL_PATH.relative_to(ROOT)} - "
                          f"run: python demo/prepare_demo.py")
        return

    try:
        payload = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        meta = payload["meta"]

        backbone = build_model(meta["n_classes"], pretrained=True)
        backbone.classifier = torch.nn.Identity()  # we take the 1024-d features ourselves
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)

        head = torch.nn.Linear(meta["feature_dim"], meta["n_classes"])
        head.load_state_dict({"weight": payload["state_dict"]["classifier.weight"],
                              "bias": payload["state_dict"]["classifier.bias"]})
        head.eval()

        STATE.update({
            "ready": True,
            "backbone": backbone,
            "head": head,
            "meta": meta,
            "classes": meta["class_names"],
            "transform": build_transforms(IMAGE_SIZE, train=False),  # the training transform
        })
        print(f"Loaded {MODEL_PATH.relative_to(ROOT)}")
        print(f"  {meta['training']}")
        m = meta["test_metrics"]
        print(f"  held-out test: balanced accuracy {m['balanced_accuracy']:.3f}, "
              f"rare macro-F1 {m['rare_macro_f1']:.3f}")
    except Exception as exc:  # startup must explain itself rather than half-work
        STATE["error"] = f"could not load the model: {exc}"
        print(f"ERROR: {STATE['error']}", file=sys.stderr)


def colour_constancy(image: Image.Image, power: int = 6) -> Image.Image:
    """Shades-of-Gray colour constancy, as FLamby applies to Fed-ISIC2019.

    Dermoscopy images vary in illumination by hospital and camera. Without this a model can
    learn the camera instead of the disease - which is exactly the confound this project
    studies, so it is corrected rather than left in. Only uploads need it; the dataset's own
    images arrive already corrected.
    """
    arr = np.asarray(image, dtype=np.float32)
    rgb = np.power(np.mean(np.power(arr, power), axis=(0, 1)), 1.0 / power)
    rgb = rgb / np.sqrt(np.sum(rgb ** 2))
    arr = np.clip(arr * (1.0 / (rgb * np.sqrt(3.0))), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


@torch.inference_mode()
def predict(image: Image.Image, apply_cc: bool) -> dict:
    """Run one image through the exact training pipeline and return the full distribution."""
    t0 = time.time()
    image = image.convert("RGB")
    if apply_cc:
        image = colour_constancy(image)

    x = STATE["transform"](image).unsqueeze(0)
    features = densenet_features(STATE["backbone"], x)
    probs = torch.softmax(STATE["head"](features), dim=1)[0]

    order = torch.argsort(probs, descending=True)
    top = int(order[0])
    return {
        "predicted": {"index": top, "name": STATE["classes"][top],
                      "confidence": float(probs[top])},
        "runner_up": {"index": int(order[1]), "name": STATE["classes"][int(order[1])],
                      "confidence": float(probs[int(order[1])])},
        "probabilities": [{"index": i, "name": name, "probability": float(probs[i])}
                          for i, name in enumerate(STATE["classes"])],
        "preprocessing": {
            "resize": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
            "normalisation": "ImageNet mean/std",
            "colour_constancy": apply_cc,
            "note": ("Shades-of-Gray correction applied, as FLamby does for raw images"
                     if apply_cc else
                     "already colour-constancy corrected in the dataset; not applied twice"),
        },
        "inference_ms": round((time.time() - t0) * 1000, 1),
    }


# --------------------------------------------------------------------------------------
# samples
# --------------------------------------------------------------------------------------

def load_samples() -> dict:
    path = SAMPLE_DIR / "samples.json"
    if not path.exists():
        return {"samples": [], "note": "run: python demo/prepare_demo.py"}
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------------------

@app.get("/health")
def health():
    if not STATE["ready"]:
        return jsonify({"ready": False, "error": STATE["error"]}), 503
    meta = STATE["meta"]
    return jsonify({
        "ready": True,
        "model": {
            "backbone": meta["backbone"],
            "head": meta["what"],
            "training": meta["training"],
            "rounds": meta["config"]["rounds"],
            "seed": meta["seed"],
            "rule": meta["rule"],
            "trained_utc": meta["created_utc"],
            "preprocessing": meta["preprocessing"],
        },
        "test_metrics": meta["test_metrics"],
        "classes": STATE["classes"],
        "n_samples": len(load_samples().get("samples", [])),
    })


@app.get("/samples")
def samples():
    data = load_samples()
    return jsonify({
        "note": data.get("note"),
        "samples": [{
            "id": s["id"],
            "url": f"/samples/{s['file']}",
            "true_label": s["true_label"],
            "true_label_name": s["true_label_name"],
            "centre": s["centre"],
        } for s in data.get("samples", [])],
    })


@app.get("/samples/<path:filename>")
def sample_file(filename: str):
    return send_from_directory(SAMPLE_DIR, filename, max_age=3600)


@app.post("/predict")
def predict_route():
    if not STATE["ready"]:
        return jsonify({"error": STATE["error"] or "model not loaded"}), 503

    sample_id = (request.form.get("sample_id")
                 or (request.get_json(silent=True) or {}).get("sample_id"))

    if sample_id:
        match = next((s for s in load_samples().get("samples", [])
                      if s["id"] == sample_id), None)
        if match is None:
            return jsonify({"error": f"unknown sample_id {sample_id!r}"}), 404
        image = Image.open(SAMPLE_DIR / match["file"])
        # Dataset images are already corrected upstream, so correcting again would feed the
        # model something training never saw.
        out = predict(image, apply_cc=False)
        out["source"] = {
            "kind": "sample",
            "id": match["id"],
            "true_label": match["true_label"],
            "true_label_name": match["true_label_name"],
            "centre": match["centre"],
            "split": "held-out test",
        }
        out["correct"] = out["predicted"]["index"] == match["true_label"]
        return jsonify(out)

    upload = request.files.get("image") or request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "send a file as 'image', or a sample_id"}), 400
    try:
        image = Image.open(io.BytesIO(upload.read()))
        image.load()
    except Exception:
        return jsonify({"error": "that file could not be read as an image"}), 400

    out = predict(image, apply_cc=True)
    out["source"] = {"kind": "upload", "filename": upload.filename}
    return jsonify(out)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"image larger than {MAX_UPLOAD_MB} MB"}), 413


def main() -> int:
    parser = argparse.ArgumentParser(description="Live demo backend for the EARN project.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    load_model()
    if not STATE["ready"]:
        print(f"\nStarting anyway so the page can show a clear message.\n  {STATE['error']}\n",
              file=sys.stderr)
    print(f"\n  Demo API on http://{args.host}:{args.port}   (Ctrl+C to stop)\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
