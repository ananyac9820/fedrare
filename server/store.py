"""
SQLite store for the live system.

Everything the console writes lives here, in server/sandbox/ - uploads, the sandbox model,
round history, trust. The research results in results/ and web/data/ are never touched by
this module, which is what keeps the published numbers and the sandbox numbers apart.
"""

from __future__ import annotations

import json
import sqlite3
import secrets
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SANDBOX = ROOT / "server" / "sandbox"
DB_PATH = SANDBOX / "console.db"
UPLOADS = SANDBOX / "uploads"
MODELS = SANDBOX / "models"

SCHEMA = """
CREATE TABLE IF NOT EXISTS hospitals (
    id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'hospital', 'clinic')),
    hospital_id INTEGER REFERENCES hospitals(id), display_name TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL, last_seen TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY, hospital_id INTEGER NOT NULL REFERENCES hospitals(id),
    created_at TEXT NOT NULL, n_images INTEGER NOT NULL, counts TEXT NOT NULL,
    note TEXT, features_path TEXT
);
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY, submission_id INTEGER NOT NULL REFERENCES submissions(id),
    filename TEXT NOT NULL, path TEXT NOT NULL, label INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS rounds (
    id INTEGER PRIMARY KEY, round_no INTEGER NOT NULL, rule TEXT NOT NULL,
    created_at TEXT NOT NULL, started_by INTEGER REFERENCES users(id),
    metrics TEXT NOT NULL, n_hospitals INTEGER NOT NULL, seconds REAL
);
CREATE TABLE IF NOT EXISTS contributions (
    id INTEGER PRIMARY KEY, round_id INTEGER NOT NULL REFERENCES rounds(id),
    hospital_id INTEGER NOT NULL REFERENCES hospitals(id), n_images INTEGER NOT NULL,
    size_share REAL NOT NULL, rare_row_weight REAL NOT NULL, trust TEXT NOT NULL,
    declared_counts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL, model TEXT NOT NULL, filename TEXT,
    predicted INTEGER NOT NULL, confidence REAL NOT NULL, probabilities TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sub_hospital ON submissions(hospital_id);
CREATE INDEX IF NOT EXISTS idx_contrib_round ON contributions(round_id);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def db():
    SANDBOX.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init() -> None:
    UPLOADS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    with db() as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------- users and sessions

def create_hospital(conn, name: str) -> int:
    cur = conn.execute("INSERT OR IGNORE INTO hospitals (name, created_at) VALUES (?, ?)",
                       (name, now()))
    if cur.lastrowid:
        return cur.lastrowid
    return conn.execute("SELECT id FROM hospitals WHERE name = ?", (name,)).fetchone()["id"]


def create_user(conn, username, password_hash, role, hospital_id=None, display_name=None) -> int:
    cur = conn.execute(
        "INSERT OR IGNORE INTO users (username, password_hash, role, hospital_id, "
        "display_name, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (username, password_hash, role, hospital_id, display_name or username, now()))
    if cur.lastrowid:
        return cur.lastrowid
    return conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()["id"]


def user_by_name(conn, username):
    return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()


def new_session(conn, user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    conn.execute("INSERT INTO sessions (token, user_id, created_at, last_seen) "
                 "VALUES (?, ?, ?, ?)", (token, user_id, now(), now()))
    return token


def user_for_token(conn, token: str):
    row = conn.execute(
        "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token = ?",
        (token,)).fetchone()
    if row:
        conn.execute("UPDATE sessions SET last_seen = ? WHERE token = ?", (now(), token))
    return row


def end_session(conn, token: str) -> None:
    conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


# ---------------------------------------------------------------- submissions

def add_submission(conn, hospital_id: int, counts: dict, note: str | None) -> int:
    cur = conn.execute(
        "INSERT INTO submissions (hospital_id, created_at, n_images, counts, note) "
        "VALUES (?, ?, ?, ?, ?)",
        (hospital_id, now(), sum(counts.values()), json.dumps(counts), note))
    return cur.lastrowid


def add_image(conn, submission_id: int, filename: str, path: str, label: int) -> None:
    conn.execute("INSERT INTO images (submission_id, filename, path, label) VALUES (?, ?, ?, ?)",
                 (submission_id, filename, path, label))


def set_features_path(conn, submission_id: int, path: str) -> None:
    conn.execute("UPDATE submissions SET features_path = ? WHERE id = ?", (path, submission_id))


def hospital_submissions(conn, hospital_id: int | None = None):
    if hospital_id is None:
        return conn.execute(
            "SELECT s.*, h.name AS hospital FROM submissions s "
            "JOIN hospitals h ON h.id = s.hospital_id ORDER BY s.id DESC").fetchall()
    return conn.execute(
        "SELECT s.*, h.name AS hospital FROM submissions s JOIN hospitals h ON h.id = s.hospital_id "
        "WHERE s.hospital_id = ? ORDER BY s.id DESC", (hospital_id,)).fetchall()


def hospitals_with_data(conn):
    return conn.execute(
        "SELECT h.id, h.name, COUNT(s.id) AS n_submissions, "
        "COALESCE(SUM(s.n_images), 0) AS n_images FROM hospitals h "
        "LEFT JOIN submissions s ON s.hospital_id = h.id GROUP BY h.id ORDER BY h.id"
    ).fetchall()


# ---------------------------------------------------------------- rounds

def add_round(conn, round_no, rule, started_by, metrics, n_hospitals, seconds) -> int:
    cur = conn.execute(
        "INSERT INTO rounds (round_no, rule, created_at, started_by, metrics, n_hospitals, "
        "seconds) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (round_no, rule, now(), started_by, json.dumps(metrics), n_hospitals, seconds))
    return cur.lastrowid


def add_contribution(conn, round_id, hospital_id, n_images, size_share, rare_row_weight,
                     trust, declared_counts) -> None:
    conn.execute(
        "INSERT INTO contributions (round_id, hospital_id, n_images, size_share, "
        "rare_row_weight, trust, declared_counts) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (round_id, hospital_id, n_images, size_share, rare_row_weight,
         json.dumps(trust), json.dumps(declared_counts)))


def latest_round(conn):
    return conn.execute("SELECT * FROM rounds ORDER BY round_no DESC LIMIT 1").fetchone()


def all_rounds(conn, limit: int = 50):
    return conn.execute("SELECT * FROM rounds ORDER BY round_no DESC LIMIT ?",
                        (limit,)).fetchall()


def round_contributions(conn, round_id: int):
    return conn.execute(
        "SELECT c.*, h.name AS hospital FROM contributions c "
        "JOIN hospitals h ON h.id = c.hospital_id WHERE c.round_id = ? ORDER BY h.id",
        (round_id,)).fetchall()


def trust_history(conn, hospital_id: int):
    return conn.execute(
        "SELECT r.round_no, c.trust, c.rare_row_weight, c.size_share FROM contributions c "
        "JOIN rounds r ON r.id = c.round_id WHERE c.hospital_id = ? ORDER BY r.round_no",
        (hospital_id,)).fetchall()


# ---------------------------------------------------------------- predictions

def add_prediction(conn, user_id, model, filename, predicted, confidence, probabilities) -> int:
    cur = conn.execute(
        "INSERT INTO predictions (user_id, created_at, model, filename, predicted, "
        "confidence, probabilities) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (user_id, now(), model, filename, predicted, confidence, json.dumps(probabilities)))
    return cur.lastrowid


def recent_predictions(conn, user_id: int | None = None, limit: int = 25):
    if user_id is None:
        return conn.execute(
            "SELECT p.*, u.display_name AS who FROM predictions p JOIN users u ON u.id = p.user_id "
            "ORDER BY p.id DESC LIMIT ?", (limit,)).fetchall()
    return conn.execute("SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                        (user_id, limit)).fetchall()
