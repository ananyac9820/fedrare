#!/usr/bin/env python3
"""
The live system's API: hospitals submit data and train, clinics diagnose, admin sees all.

    python server/app.py            # http://127.0.0.1:8100

Roles
  hospital  upload labelled images, run a federated round, see its own trust and influence
  clinic    send an image, get a diagnosis from either model, see its own history
  admin     everything, plus the trust matrix, round history and a sandbox reset

What it will not do: write to results/ or web/data/. The published numbers are read-only
here; the sandbox keeps its own model, its own rounds and its own trust in server/sandbox/.

Security: this is a demonstration system. Passwords are hashed and sessions are random
tokens, but there is no rate limiting, no TLS, and the seeded accounts have known
passwords. It is not built to be exposed to the internet.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, g, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

import numpy as np

from server import federated, store

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024      # a batch of images
CORS(app, supports_credentials=True)

SEED_PASSWORD = "fedrare"        # demonstration accounts; printed at startup
ALLOWED = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------- auth plumbing

def seed_accounts() -> list[dict]:
    """Six hospitals, a clinic and an admin, created once."""
    made = []
    with store.db() as conn:
        for i in range(1, 7):
            name = f"Hospital {i}"
            hid = store.create_hospital(conn, name)
            store.create_user(conn, f"hospital{i}", generate_password_hash(SEED_PASSWORD),
                              "hospital", hid, name)
            made.append({"username": f"hospital{i}", "role": "hospital", "name": name})
        store.create_user(conn, "clinic", generate_password_hash(SEED_PASSWORD), "clinic",
                          None, "Riverside Clinic")
        store.create_user(conn, "admin", generate_password_hash(SEED_PASSWORD), "admin",
                          None, "System administrator")
    made += [{"username": "clinic", "role": "clinic"}, {"username": "admin", "role": "admin"}]
    return made


def current_user():
    auth = request.headers.get("Authorization", "")
    token = auth[7:] if auth.startswith("Bearer ") else request.args.get("token", "")
    if not token:
        return None
    with store.db() as conn:
        row = store.user_for_token(conn, token)
        return dict(row) if row else None


def require(*roles):
    """Decorator: a valid session, and one of these roles."""
    def wrap(fn):
        def inner(*args, **kwargs):
            user = current_user()
            if not user:
                return jsonify({"error": "not signed in"}), 401
            if roles and user["role"] not in roles:
                return jsonify({"error": f"this needs the {' or '.join(roles)} role"}), 403
            g.user = user
            return fn(*args, **kwargs)
        inner.__name__ = fn.__name__
        return inner
    return wrap


# ---------------------------------------------------------------- session

@app.post("/api/login")
def login():
    body = request.get_json(silent=True) or {}
    username, password = body.get("username", ""), body.get("password", "")
    with store.db() as conn:
        row = store.user_by_name(conn, username)
        if not row or not check_password_hash(row["password_hash"], password):
            return jsonify({"error": "wrong username or password"}), 401
        token = store.new_session(conn, row["id"])
        user = {"username": row["username"], "role": row["role"],
                "display_name": row["display_name"], "hospital_id": row["hospital_id"]}
    return jsonify({"token": token, "user": user})


@app.post("/api/logout")
@require()
def logout():
    auth = request.headers.get("Authorization", "")
    with store.db() as conn:
        store.end_session(conn, auth[7:])
    return jsonify({"ok": True})


@app.get("/api/me")
@require()
def me():
    u = g.user
    return jsonify({"username": u["username"], "role": u["role"],
                    "display_name": u["display_name"], "hospital_id": u["hospital_id"]})


@app.get("/api/meta")
def meta():
    """Public: what the system knows, for the login screen and label pickers."""
    research = federated.research_head()["meta"]
    return jsonify({
        "classes": federated.class_names(),
        "rare_classes": [federated.class_names()[c] for c in federated.RARE_IDS],
        "research_model": {
            "training": research["training"],
            "rounds": research["config"]["rounds"],
            "metrics": research["test_metrics"],
        },
        "demo_accounts": [{"username": f"hospital{i}", "role": "hospital"} for i in range(1, 7)]
                         + [{"username": "clinic", "role": "clinic"},
                            {"username": "admin", "role": "admin"}],
    })


# ---------------------------------------------------------------- hospital: submit + train

@app.post("/api/submissions")
@require("hospital", "admin")
def submit():
    """Upload labelled images. Labels come as label_<filename> fields, or a labels JSON map."""
    hospital_id = g.user["hospital_id"]
    if g.user["role"] == "admin":
        hospital_id = int(request.form.get("hospital_id", 0)) or None
    if not hospital_id:
        return jsonify({"error": "no hospital on this account"}), 400

    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "no images were sent"}), 400
    labels_map = json.loads(request.form.get("labels", "{}"))
    default_label = request.form.get("label")
    n_classes = len(federated.class_names())

    saved, counts, bad = [], {}, []
    with store.db() as conn:
        sub_id = store.add_submission(conn, hospital_id, {}, request.form.get("note"))
        folder = store.UPLOADS / f"hospital_{hospital_id}" / f"submission_{sub_id}"
        folder.mkdir(parents=True, exist_ok=True)
        for f in files:
            name = secure_filename(f.filename or "")
            if not name or Path(name).suffix.lower() not in ALLOWED:
                bad.append({"file": f.filename, "why": "not a jpg or png"})
                continue
            raw = labels_map.get(f.filename, labels_map.get(name, default_label))
            try:
                label = int(raw)
            except (TypeError, ValueError):
                bad.append({"file": f.filename, "why": "no label given"})
                continue
            if not 0 <= label < n_classes:
                bad.append({"file": f.filename, "why": f"label {label} is out of range"})
                continue
            path = folder / name
            f.save(path)
            store.add_image(conn, sub_id, name, str(path), label)
            saved.append((path, label))
            counts[str(label)] = counts.get(str(label), 0) + 1

        if not saved:
            conn.execute("DELETE FROM submissions WHERE id = ?", (sub_id,))
            return jsonify({"error": "nothing usable in that upload", "rejected": bad}), 400

        features = federated.extract_features([p for p, _ in saved])
        fpath = folder / "features.npz"
        np.savez(fpath, features=features,
                 labels=np.array([l for _, l in saved], dtype=np.int64))
        conn.execute("UPDATE submissions SET n_images = ?, counts = ?, features_path = ? "
                     "WHERE id = ?", (len(saved), json.dumps(counts), str(fpath), sub_id))

    names = federated.class_names()
    return jsonify({
        "submission_id": sub_id, "accepted": len(saved), "rejected": bad,
        "declared": {names[int(k)]: v for k, v in counts.items()},
        "note": "these counts are your declaration; trust is audited against them each round",
    })


@app.get("/api/submissions")
@require("hospital", "admin")
def list_submissions():
    hid = None if g.user["role"] == "admin" else g.user["hospital_id"]
    with store.db() as conn:
        rows = store.hospital_submissions(conn, hid)
    names = federated.class_names()
    return jsonify([{
        "id": r["id"], "hospital": r["hospital"], "created_at": r["created_at"],
        "n_images": r["n_images"], "note": r["note"],
        "counts": {names[int(k)]: v for k, v in json.loads(r["counts"]).items()},
    } for r in rows])


@app.post("/api/train")
@require("hospital", "admin")
def train():
    rule = (request.get_json(silent=True) or {}).get("rule", "earn")
    if rule not in ("earn", "fedavg"):
        return jsonify({"error": "rule must be earn or fedavg"}), 400
    try:
        with store.db() as conn:
            result = federated.run_round(conn, g.user["id"], rule)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


# ---------------------------------------------------------------- clinic: diagnose

@app.post("/api/diagnose")
@require("clinic", "admin", "hospital")
def diagnose():
    which = request.form.get("model", "sandbox")
    if which not in ("sandbox", "research"):
        return jsonify({"error": "model must be sandbox or research"}), 400
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify({"error": "no image was sent"}), 400
    name = secure_filename(upload.filename)
    if Path(name).suffix.lower() not in ALLOWED:
        return jsonify({"error": "that file is not a jpg or png"}), 400

    folder = store.UPLOADS / "diagnoses"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{g.user['id']}_{name}"
    upload.save(path)
    out = federated.predict(path, which)
    with store.db() as conn:
        out["id"] = store.add_prediction(
            conn, g.user["id"], which, name, out["predicted"]["index"],
            out["predicted"]["confidence"], out["probabilities"])
    out["filename"] = name
    return jsonify(out)


@app.get("/api/diagnoses")
@require("clinic", "admin", "hospital")
def diagnoses():
    uid = None if g.user["role"] == "admin" else g.user["id"]
    with store.db() as conn:
        rows = store.recent_predictions(conn, uid)
    names = federated.class_names()
    return jsonify([{
        "id": r["id"], "created_at": r["created_at"], "model": r["model"],
        "filename": r["filename"], "predicted": names[r["predicted"]],
        "confidence": r["confidence"],
        "who": r["who"] if "who" in r.keys() else None,
    } for r in rows])


# ---------------------------------------------------------------- shared reads

@app.get("/api/rounds")
@require()
def rounds():
    with store.db() as conn:
        rows = store.all_rounds(conn)
        out = []
        for r in rows:
            contribs = store.round_contributions(conn, r["id"])
            out.append({
                "round": r["round_no"], "rule": r["rule"], "created_at": r["created_at"],
                "seconds": r["seconds"], "n_hospitals": r["n_hospitals"],
                "metrics": json.loads(r["metrics"]),
                "contributions": [{
                    "hospital": c["hospital"], "hospital_id": c["hospital_id"],
                    "images": c["n_images"], "size_share": c["size_share"],
                    "rare_row_weight": c["rare_row_weight"],
                    "trust": json.loads(c["trust"]),
                    "declared": json.loads(c["declared_counts"]),
                } for c in contribs],
            })
    return jsonify(out)


@app.get("/api/status")
@require()
def status():
    """What the console shows on every dashboard: both models, side by side."""
    with store.db() as conn:
        latest = store.latest_round(conn)
        hospitals = [dict(h) for h in store.hospitals_with_data(conn)]
    payload = federated.sandbox_head()
    research = federated.research_head()["meta"]
    return jsonify({
        "sandbox": {
            "round": payload.get("round", 0),
            "metrics": json.loads(latest["metrics"]) if latest else None,
            "seeded_from": payload.get("seeded_from"),
        },
        "research": {"metrics": research["test_metrics"], "training": research["training"],
                     "read_only": True},
        "hospitals": hospitals,
        "classes": federated.class_names(),
        "rare_classes": [federated.class_names()[c] for c in federated.RARE_IDS],
    })


# ---------------------------------------------------------------- admin

@app.get("/api/admin/overview")
@require("admin")
def admin_overview():
    with store.db() as conn:
        hospitals = [dict(h) for h in store.hospitals_with_data(conn)]
        rounds_ = [dict(r) for r in store.all_rounds(conn, 10)]
        users = [dict(u) for u in conn.execute(
            "SELECT id, username, role, display_name, hospital_id, created_at FROM users "
            "ORDER BY role, id").fetchall()]
        preds = len(store.recent_predictions(conn, None, 1000))
        subs = len(store.hospital_submissions(conn))
    return jsonify({
        "hospitals": hospitals, "users": users,
        "rounds": [{"round": r["round_no"], "rule": r["rule"], "created_at": r["created_at"],
                    "metrics": json.loads(r["metrics"])} for r in rounds_],
        "totals": {"submissions": subs, "diagnoses": preds, "hospitals": len(hospitals)},
    })


@app.get("/api/admin/trust")
@require("admin")
def admin_trust():
    """The trust matrix: every hospital against every class, at the latest round."""
    with store.db() as conn:
        latest = store.latest_round(conn)
        if not latest:
            return jsonify({"round": 0, "classes": federated.class_names(), "rows": []})
        contribs = store.round_contributions(conn, latest["id"])
        rows = [{"hospital": c["hospital"], "hospital_id": c["hospital_id"],
                 "trust": json.loads(c["trust"]),
                 "declared": json.loads(c["declared_counts"]),
                 "rare_row_weight": c["rare_row_weight"], "size_share": c["size_share"]}
                for c in contribs]
    return jsonify({"round": latest["round_no"], "classes": federated.class_names(),
                    "rare_classes": [federated.class_names()[c] for c in federated.RARE_IDS],
                    "rows": rows})


@app.post("/api/admin/reset")
@require("admin")
def admin_reset():
    """Throw away the sandbox model and round history. Never touches the published results."""
    federated.reset_sandbox()
    with store.db() as conn:
        conn.execute("DELETE FROM contributions")
        conn.execute("DELETE FROM rounds")
    return jsonify({"ok": True, "note": "sandbox model and rounds cleared; uploads kept; "
                                        "published results untouched"})


@app.get("/api/health")
def health():
    ok = federated.RESEARCH_MODEL.exists()
    return jsonify({"ready": ok,
                    "error": None if ok else "models/tier_a_s1_fedavg_head.pt is missing - "
                                             "run python demo/prepare_demo.py"}), (200 if ok else 503)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "that upload is larger than 64 MB"}), 413


def main() -> int:
    store.init()
    accounts = seed_accounts()
    print("Live system API on http://127.0.0.1:8100")
    print(f"  seeded accounts (password '{SEED_PASSWORD}'):")
    for a in accounts:
        print(f"    {a['username']:<10} {a['role']}")
    print(f"  sandbox lives in {store.SANDBOX.relative_to(ROOT)}; "
          f"published results are read-only\n")
    app.run(host="127.0.0.1", port=8100, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
