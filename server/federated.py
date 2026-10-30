"""
The live system's federated engine.

Reuses the research pipeline unchanged - the frozen DenseNet-121 feature extractor, the Tier A
head, local_train, and EARN's trust mechanism - so what the console demonstrates is the same
machinery the study measured, not a second implementation that could drift from it.

Two things are deliberately separate from the research:

  * the model. The sandbox starts from models/tier_a_s1_fedavg_head.pt and from then on keeps
    its own checkpoint in server/sandbox/models/. The published checkpoint is read once and
    never written.
  * the numbers. Sandbox metrics live in the console database. Nothing here writes to results/
    or web/data/.

A hospital declares its class counts by uploading labelled images, and EARN audits that
declaration across rounds - which is the attested-evidence design from
docs/DESIGN_NOTE_attestation.md, running for real.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from src.data.loader import build_transforms
from src.federated.earn import EARN, EARNConfig
from src.federated.primitives import FlatLayout
from src.federated.tier_a import TierAConfig, TierAHead, local_train
from src.models.densenet import build_model, densenet_features
from src.utils.config import load_config

from . import store

RESEARCH_MODEL = ROOT / "models" / "tier_a_s1_fedavg_head.pt"
SANDBOX_MODEL = store.MODELS / "sandbox_head.pt"
EARN_STATE = store.MODELS / "earn_state.npz"
FEATURE_DIM = 1024
IMAGE_SIZE = 224
RARE_IDS = [5, 6]
HOLDER_TAU = 19.5          # the pre-registered absolute threshold (R8: do not use "mean")
_CACHE: dict = {}


def config() -> dict:
    if "cfg" not in _CACHE:
        _CACHE["cfg"] = load_config(ROOT / "configs" / "default.yaml")
    return _CACHE["cfg"]


def class_names() -> list[str]:
    cfg = config()
    return [cfg["dataset"]["class_names"][i] for i in range(cfg["dataset"]["n_classes"])]


def backbone():
    """The frozen feature extractor, loaded once per process."""
    if "backbone" not in _CACHE:
        model = build_model(len(class_names()), pretrained=True)
        model.classifier = torch.nn.Identity()
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _CACHE["backbone"] = model
        _CACHE["transform"] = build_transforms(IMAGE_SIZE, train=False)
    return _CACHE["backbone"], _CACHE["transform"]


def research_head() -> dict:
    """The published checkpoint. Read-only: the console never writes this file."""
    if "research" not in _CACHE:
        payload = torch.load(RESEARCH_MODEL, map_location="cpu", weights_only=False)
        _CACHE["research"] = {"state": {k: v.clone() for k, v in payload["state_dict"].items()},
                              "meta": payload["meta"]}
    return _CACHE["research"]


def sandbox_head() -> dict:
    """The sandbox model, seeded from the published one the first time it is asked for."""
    if SANDBOX_MODEL.exists():
        payload = torch.load(SANDBOX_MODEL, map_location="cpu", weights_only=False)
        return payload
    base = research_head()
    payload = {"state_dict": {k: v.clone() for k, v in base["state"].items()},
               "round": 0, "seeded_from": str(RESEARCH_MODEL.relative_to(ROOT)),
               "note": "sandbox model; starts as a copy of the published checkpoint"}
    torch.save(payload, SANDBOX_MODEL)
    return payload


def save_sandbox_head(state: dict, round_no: int) -> None:
    torch.save({"state_dict": {k: v.cpu().clone() for k, v in state.items()},
                "round": round_no, "seeded_from": str(RESEARCH_MODEL.relative_to(ROOT)),
                "note": "sandbox model; starts as a copy of the published checkpoint"},
               SANDBOX_MODEL)


def reset_sandbox() -> None:
    for path in (SANDBOX_MODEL, EARN_STATE):
        if path.exists():
            path.unlink()



# ---------------------------------------------------------------- EARN across requests

class ResumableEARN(EARN):
    """EARN that survives being rebuilt on every HTTP request.

    The research runs keep one aggregator alive for a whole experiment, so trust and the
    locked history simply accumulate in memory. A web server cannot: each round arrives in
    its own request. This subclass reloads both from disk, keyed by **hospital id** rather
    than position, so a hospital that joins later does not shift another one's record.

    The hash chain inside EARN is turned off here - it requires strictly sequential rounds
    within one live object. The console's audit trail is the rounds and trust tables written
    to SQLite, which is append-only and timestamped.
    """

    def __init__(self, *args, order: list[int], resume: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self._order = order
        self._resume = resume

    def _init(self, K: int, layout: FlatLayout) -> None:
        super()._init(K, layout)
        for i, hid in enumerate(self._order):
            saved = self._resume.get(hid)
            if not saved:
                continue
            if saved["hist_sum"].shape == self.hist_sum[i].shape:
                self.trust[i] = saved["trust"]
                self.hist_sum[i] = saved["hist_sum"]
                self.hist_n[i] = saved["hist_n"]


def load_earn_state() -> dict:
    """Per-hospital trust and locked history, as saved after the last round."""
    if not EARN_STATE.exists():
        return {}
    with np.load(EARN_STATE) as z:
        out: dict = {}
        for key in z.files:
            kind, hid = key.rsplit("_", 1)
            out.setdefault(int(hid), {})[kind] = z[key]
    return {hid: v for hid, v in out.items()
            if {"trust", "hist_sum", "hist_n"} <= set(v)}


def save_earn_state(order: list[int], aggregator: EARN) -> None:
    payload = {}
    for i, hid in enumerate(order):
        payload[f"trust_{hid}"] = aggregator.trust[i]
        payload[f"hist_sum_{hid}"] = aggregator.hist_sum[i]
        payload[f"hist_n_{hid}"] = aggregator.hist_n[i]
    np.savez(EARN_STATE, **payload)


# ---------------------------------------------------------------- features

def extract_features(paths: list[Path]) -> np.ndarray:
    """Images -> 1,024-d features, through the same transform the study used."""
    model, transform = backbone()
    out = np.empty((len(paths), FEATURE_DIM), dtype=np.float32)
    batch_size = 16
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            chunk = paths[start:start + batch_size]
            batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in chunk])
            out[start:start + len(chunk)] = densenet_features(model, batch).float().numpy()
    return out


def predict(path: Path, which: str = "sandbox") -> dict:
    """One image through the chosen model."""
    model, transform = backbone()
    state = (research_head()["state"] if which == "research"
             else sandbox_head()["state_dict"])
    w, b = state["classifier.weight"], state["classifier.bias"]
    t0 = time.time()
    with torch.inference_mode():
        x = transform(Image.open(path).convert("RGB")).unsqueeze(0)
        feat = densenet_features(model, x)
        probs = torch.softmax(feat @ w.T + b, dim=1)[0]
    order = torch.argsort(probs, descending=True)
    names = class_names()
    top = int(order[0])
    return {
        "model": which,
        "predicted": {"index": top, "name": names[top], "confidence": float(probs[top])},
        "runner_up": {"index": int(order[1]), "name": names[int(order[1])],
                      "confidence": float(probs[int(order[1])])},
        "probabilities": [{"index": i, "name": n, "probability": float(probs[i])}
                          for i, n in enumerate(names)],
        "inference_ms": round((time.time() - t0) * 1000, 1),
    }


# ---------------------------------------------------------------- a federated round

def _held_out():
    """The research test split, used to score the sandbox model. Read-only."""
    if "test" not in _CACHE:
        path = ROOT / "data" / "features" / "fed_isic2019_densenet121_test.npz"
        if not path.exists():
            _CACHE["test"] = None
        else:
            d = np.load(path)
            _CACHE["test"] = (torch.from_numpy(np.ascontiguousarray(d["features"])),
                              d["labels"].astype(np.int64))
    return _CACHE["test"]


def evaluate_state(state: dict) -> dict:
    """Balanced accuracy, rare-class recall and F1 on the held-out split."""
    held = _held_out()
    if held is None:
        return {"available": False}
    x, y = held
    from src.utils.metrics import summarise
    w, b = state["classifier.weight"], state["classifier.bias"]
    with torch.inference_mode():
        pred = (x @ w.T + b).argmax(1).numpy()
    m = summarise(y, pred, RARE_IDS, len(class_names()))
    cm = np.zeros((len(class_names()),) * 2, dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    recalls = [float(cm[c, c] / cm[c].sum()) if cm[c].sum() else 0.0 for c in RARE_IDS]
    return {
        "available": True,
        "balanced_accuracy": float(m["balanced_accuracy"]),
        "accuracy": float(m["accuracy"]),
        "rare_macro_f1": float(m["rare_macro_f1"]),
        "rare_recall": float(np.mean(recalls)),
        "per_class_f1": {class_names()[i]: float(v) for i, v in m["per_class_f1"].items()},
    }


def load_hospital_data(conn) -> list[dict]:
    """Every hospital that has uploaded anything, with its features and declared counts."""
    out = []
    for row in store.hospitals_with_data(conn):
        if not row["n_images"]:
            continue
        subs = store.hospital_submissions(conn, row["id"])
        feats, labels = [], []
        for s in subs:
            if not s["features_path"]:
                continue
            arr = np.load(s["features_path"])
            feats.append(arr["features"])
            labels.append(arr["labels"])
        if not feats:
            continue
        out.append({
            "hospital_id": row["id"], "name": row["name"],
            "x": torch.from_numpy(np.ascontiguousarray(np.concatenate(feats))),
            "y": torch.from_numpy(np.concatenate(labels)),
        })
    return out


def run_round(conn, started_by: int, rule: str = "earn") -> dict:
    """One federated round over whatever the hospitals have uploaded."""
    hospitals = load_hospital_data(conn)
    if len(hospitals) < 1:
        raise ValueError("no hospital has uploaded data yet")

    n_classes = len(class_names())
    cfg = TierAConfig(rounds=1, n_classes=n_classes)
    payload = sandbox_head()
    global_state = {k: v.clone() for k, v in payload["state_dict"].items()}
    prev_round = payload.get("round", 0)
    round_no = prev_round + 1

    counts = np.stack([np.bincount(h["y"].numpy(), minlength=n_classes) for h in hospitals])
    sizes = [len(h["y"]) for h in hospitals]
    t0 = time.time()

    states = []
    for i, h in enumerate(hospitals):
        rng = np.random.default_rng([round_no, i])
        states.append(local_train(global_state, h["x"], h["y"], cfg, rng))

    # EARN, with the declaration each hospital made by uploading labelled images as its
    # evidence - the attested-evidence design, running for real.
    def declared(updates, layout, round_num, _c=counts):
        return _c.astype(np.float64)

    if rule == "earn":
        order = [h["hospital_id"] for h in hospitals]
        aggregator = ResumableEARN(
            EARNConfig(tau=HOLDER_TAU, ledger=False), evidence_fn=declared, name="earn",
            order=order, resume=load_earn_state())
        # Round 1 as far as the aggregator is concerned: it was just rebuilt, and its trust
        # and history came from disk rather than from a previous call on this object.
        state, info = aggregator(states, sizes, global_state, 1,
                                 n_classes=n_classes, head_prefix="classifier")
        save_earn_state(order, aggregator)
    else:
        from src.federated import baselines
        state, info = baselines.fedavg(states, sizes, global_state, round_no,
                                       n_classes=n_classes, head_prefix="classifier")

    metrics = evaluate_state(state)
    seconds = round(time.time() - t0, 2)
    save_sandbox_head(state, round_no)

    round_id = store.add_round(conn, round_no, rule, started_by, metrics, len(hospitals), seconds)
    head_w = np.asarray(info.get("head_row_weights"), dtype=float)
    trust = np.asarray(info.get("trust", np.zeros_like(head_w)), dtype=float)
    total = float(sum(sizes))
    for i, h in enumerate(hospitals):
        store.add_contribution(
            conn, round_id, h["hospital_id"], sizes[i], sizes[i] / total,
            float(head_w[i, RARE_IDS].mean()),
            {class_names()[c]: float(trust[i, c]) for c in range(n_classes)},
            {class_names()[c]: int(counts[i, c]) for c in range(n_classes)})

    return {
        "round": round_no, "rule": rule, "seconds": seconds,
        "n_hospitals": len(hospitals), "metrics": metrics,
        "contributions": [{
            "hospital": h["name"], "hospital_id": h["hospital_id"], "images": sizes[i],
            "size_share": sizes[i] / total,
            "rare_row_weight": float(head_w[i, RARE_IDS].mean()),
            "trust": {class_names()[c]: float(trust[i, c]) for c in range(n_classes)},
            "declared": {class_names()[c]: int(counts[i, c]) for c in range(n_classes)},
        } for i, h in enumerate(hospitals)],
    }
