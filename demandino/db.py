"""Local workspace database (SQLite): accounts, sessions, saved data, scenarios, settings and logs.

The database lives in `.data/demandino.db` (override with DEMANDINO_DATA_DIR). Passwords are hashed
with scrypt, connector secrets are encrypted with a per-install key, and nothing is pickled.
"""
from __future__ import annotations

import gzip
import hashlib
import hmac
import io
import json
import os
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from cryptography.fernet import Fernet, InvalidToken

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DEMANDINO_DATA_DIR", ROOT / ".data"))
DB_PATH = DATA_DIR / "demandino.db"

SESSION_DAYS = 14
MAX_FAILED = 5
LOCK_MINUTES = 15
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
MIN_PASSWORD = 10
MAX_DATASET_MB = 30

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL COLLATE NOCASE, pw_hash BLOB NOT NULL, salt BLOB NOT NULL,
  created REAL NOT NULL, failed INTEGER NOT NULL DEFAULT 0, locked_until REAL NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS sessions (
  token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL, expires REAL NOT NULL,
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS datasets (
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, name TEXT NOT NULL, source TEXT NOT NULL, created REAL NOT NULL,
  data BLOB NOT NULL, rows INTEGER NOT NULL, products INTEGER NOT NULL,
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS scenarios (
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, name TEXT NOT NULL, created REAL NOT NULL, inputs TEXT NOT NULL,
  summary TEXT NOT NULL, FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS settings (
  user_id INTEGER NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY(user_id, key),
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS secrets (
  user_id INTEGER NOT NULL, key TEXT NOT NULL, value BLOB NOT NULL, PRIMARY KEY(user_id, key),
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS forecast_log (
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, data_key TEXT NOT NULL, product_id TEXT NOT NULL,
  product_name TEXT NOT NULL, created REAL NOT NULL, payload TEXT NOT NULL,
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS shares (
  token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL, created REAL NOT NULL, expires REAL NOT NULL,
  title TEXT NOT NULL, payload TEXT NOT NULL, FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS purchase_orders (
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, created REAL NOT NULL, status TEXT NOT NULL, note TEXT NOT NULL,
  lines TEXT NOT NULL, decided_at REAL, decided_by TEXT,
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE);
"""


class AuthError(ValueError):
    """Message is safe to show to the user."""


# -------------------------------------------------------------- plumbing
def _owner_only(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


@contextmanager
def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fresh = not DB_PATH.exists()
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    if fresh:
        _owner_only(DB_PATH)
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def _fernet() -> Fernet:
    env = os.environ.get("DEMANDINO_SECRET_KEY")
    if env:
        return Fernet(env.encode())
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / ".secret-key"
    if not path.exists():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(Fernet.generate_key())
    return Fernet(path.read_bytes())


def _hash_pw(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=64)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# --------------------------------------------------------------- accounts
def create_user(username: str, password: str) -> int:
    username = username.strip()
    if not USERNAME_RE.match(username):
        raise AuthError("Username must be 3–32 characters: letters, numbers, dot, dash or underscore.")
    if len(password) < MIN_PASSWORD:
        raise AuthError(f"Password must be at least {MIN_PASSWORD} characters.")
    if password.lower() in {"password123", "1234567890", "qwertyuiop"} or password.lower() == username.lower():
        raise AuthError("Please choose a less guessable password.")
    salt = secrets.token_bytes(16)
    with connect() as con:
        exists = con.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
        if not exists:
            cur = con.execute("INSERT INTO users(username, pw_hash, salt, created) VALUES (?,?,?,?)",
                              (username, _hash_pw(password, salt), salt, time.time()))
            return int(cur.lastrowid)
    raise AuthError("That username is taken.")


def authenticate(username: str, password: str) -> int:
    with connect() as con:
        row = con.execute("SELECT * FROM users WHERE username = ?", (username.strip(),)).fetchone()
    if row is None:
        _hash_pw(password, b"0" * 16)  # same work either way, so timing doesn't reveal usernames
        raise AuthError("Wrong username or password.")
    if row["locked_until"] > time.time():
        raise AuthError(f"Too many failed attempts. Try again in {LOCK_MINUTES} minutes.")
    ok = hmac.compare_digest(_hash_pw(password, row["salt"]), row["pw_hash"])
    # Record the outcome first (committed when the block exits), then raise — so failed attempts always count.
    with connect() as con:
        if ok:
            con.execute("UPDATE users SET failed = 0, locked_until = 0 WHERE id = ?", (row["id"],))
        else:
            failed = row["failed"] + 1
            locked = time.time() + LOCK_MINUTES * 60 if failed >= MAX_FAILED else 0
            con.execute("UPDATE users SET failed = ?, locked_until = ? WHERE id = ?",
                        (0 if locked else failed, locked, row["id"]))
    if not ok:
        raise AuthError("Wrong username or password.")
    return int(row["id"])


def change_password(user_id: int, old: str, new: str) -> None:
    with connect() as con:
        name = con.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()["username"]
    authenticate(name, old)
    if len(new) < MIN_PASSWORD:
        raise AuthError(f"Password must be at least {MIN_PASSWORD} characters.")
    salt = secrets.token_bytes(16)
    with connect() as con:
        con.execute("UPDATE users SET pw_hash = ?, salt = ? WHERE id = ?", (_hash_pw(new, salt), salt, user_id))
        con.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))


def delete_user(user_id: int) -> None:
    with connect() as con:
        con.execute("DELETE FROM users WHERE id = ?", (user_id,))


def username(user_id: int) -> str | None:
    with connect() as con:
        row = con.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()
    return row["username"] if row else None


def new_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with connect() as con:
        con.execute("DELETE FROM sessions WHERE expires < ?", (time.time(),))
        con.execute("INSERT INTO sessions VALUES (?,?,?)", (_token_hash(token), user_id,
                                                             time.time() + SESSION_DAYS * 86400))
    return token


def session_user(token: str | None) -> int | None:
    if not isinstance(token, str) or not token or len(token) > 100:
        return None
    with connect() as con:
        row = con.execute("SELECT user_id FROM sessions WHERE token_hash = ? AND expires > ?",
                          (_token_hash(token), time.time())).fetchone()
    return int(row["user_id"]) if row else None


def end_session(token: str | None) -> None:
    if token:
        with connect() as con:
            con.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))


# ---------------------------------------------------------------- datasets
def save_dataset(user_id: int, name: str, sales: pd.DataFrame, source: str) -> int:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(sales.to_csv(index=False, date_format="%Y-%m-%d").encode())
    blob = buf.getvalue()
    if len(blob) > MAX_DATASET_MB * 1024 * 1024:
        raise AuthError(f"Dataset is larger than {MAX_DATASET_MB} MB after compression.")
    with connect() as con:
        cur = con.execute("INSERT INTO datasets(user_id, name, source, created, data, rows, products) "
                          "VALUES (?,?,?,?,?,?,?)", (user_id, name[:80], source[:40], time.time(), blob, len(sales),
                                                     int(sales["product_id"].nunique())))
        return int(cur.lastrowid)


def list_datasets(user_id: int) -> pd.DataFrame:
    with connect() as con:
        rows = con.execute("SELECT id, name, source, created, rows, products FROM datasets WHERE user_id = ? "
                           "ORDER BY created DESC", (user_id,)).fetchall()
    return pd.DataFrame([dict(r) for r in rows], columns=["id", "name", "source", "created", "rows", "products"])


def load_dataset(user_id: int, dataset_id: int) -> pd.DataFrame | None:
    with connect() as con:
        row = con.execute("SELECT data FROM datasets WHERE id = ? AND user_id = ?", (dataset_id, user_id)).fetchone()
    if row is None:
        return None
    df = pd.read_csv(io.BytesIO(gzip.decompress(row["data"])), parse_dates=["date"])
    df["product_id"] = df["product_id"].astype(str)
    return df


def delete_dataset(user_id: int, dataset_id: int) -> None:
    with connect() as con:
        con.execute("DELETE FROM datasets WHERE id = ? AND user_id = ?", (dataset_id, user_id))


# ------------------------------------------------------ scenarios & settings
def save_scenario(user_id: int, name: str, inputs: dict, summary: dict) -> None:
    with connect() as con:
        con.execute("INSERT INTO scenarios(user_id, name, created, inputs, summary) VALUES (?,?,?,?,?)",
                    (user_id, name[:80], time.time(), json.dumps(inputs), json.dumps(summary)))


def list_scenarios(user_id: int) -> list[dict]:
    with connect() as con:
        rows = con.execute("SELECT id, name, created, inputs, summary FROM scenarios WHERE user_id = ? "
                           "ORDER BY created DESC", (user_id,)).fetchall()
    return [{**dict(r), "inputs": json.loads(r["inputs"]), "summary": json.loads(r["summary"])} for r in rows]


def delete_scenario(user_id: int, scenario_id: int) -> None:
    with connect() as con:
        con.execute("DELETE FROM scenarios WHERE id = ? AND user_id = ?", (scenario_id, user_id))


def set_setting(user_id: int, key: str, value) -> None:
    with connect() as con:
        con.execute("INSERT INTO settings VALUES (?,?,?) ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value",
                    (user_id, key, json.dumps(value)))


def get_setting(user_id: int, key: str, default=None):
    with connect() as con:
        row = con.execute("SELECT value FROM settings WHERE user_id = ? AND key = ?", (user_id, key)).fetchone()
    return json.loads(row["value"]) if row else default


def set_secret(user_id: int, key: str, value: str) -> None:
    token = _fernet().encrypt(value.encode())
    with connect() as con:
        con.execute("INSERT INTO secrets VALUES (?,?,?) ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value",
                    (user_id, key, token))


def get_secret(user_id: int, key: str) -> str | None:
    with connect() as con:
        row = con.execute("SELECT value FROM secrets WHERE user_id = ? AND key = ?", (user_id, key)).fetchone()
    if row is None:
        return None
    try:
        return _fernet().decrypt(row["value"]).decode()
    except InvalidToken:
        return None


def delete_secret(user_id: int, key: str) -> None:
    with connect() as con:
        con.execute("DELETE FROM secrets WHERE user_id = ? AND key = ?", (user_id, key))


# ----------------------------------------------------------- forecast log
def log_forecast(user_id: int, data_key: str, product_id: str, product_name: str, payload: dict) -> None:
    with connect() as con:
        con.execute("INSERT INTO forecast_log(user_id, data_key, product_id, product_name, created, payload) "
                    "VALUES (?,?,?,?,?,?)", (user_id, data_key, product_id, product_name[:80], time.time(),
                                             json.dumps(payload)))


def list_forecasts(user_id: int) -> list[dict]:
    with connect() as con:
        rows = con.execute("SELECT * FROM forecast_log WHERE user_id = ? ORDER BY created DESC LIMIT 500",
                           (user_id,)).fetchall()
    return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]


def delete_forecast(user_id: int, log_id: int) -> None:
    with connect() as con:
        con.execute("DELETE FROM forecast_log WHERE id = ? AND user_id = ?", (log_id, user_id))


# ----------------------------------------------------------------- shares
def create_share(user_id: int, title: str, payload: dict, days: int = 7) -> str:
    token = secrets.token_urlsafe(24)
    with connect() as con:
        con.execute("DELETE FROM shares WHERE expires < ?", (time.time(),))
        con.execute("INSERT INTO shares VALUES (?,?,?,?,?,?)", (_token_hash(token), user_id, time.time(),
                                                                time.time() + days * 86400, title[:120],
                                                                json.dumps(payload)))
    return token


def get_share(token: str | None) -> dict | None:
    if not isinstance(token, str) or not token or len(token) > 100:
        return None
    with connect() as con:
        row = con.execute("SELECT * FROM shares WHERE token_hash = ? AND expires > ?",
                          (_token_hash(token), time.time())).fetchone()
    return {**dict(row), "payload": json.loads(row["payload"])} if row else None


def list_shares(user_id: int) -> list[dict]:
    with connect() as con:
        rows = con.execute("SELECT token_hash, title, created, expires FROM shares WHERE user_id = ? AND expires > ? "
                           "ORDER BY created DESC", (user_id, time.time())).fetchall()
    return [dict(r) for r in rows]


def revoke_share(user_id: int, token_hash: str) -> None:
    with connect() as con:
        con.execute("DELETE FROM shares WHERE token_hash = ? AND user_id = ?", (token_hash, user_id))


# -------------------------------------------------------- purchase orders
def create_po(user_id: int, lines: list[dict], note: str = "") -> int:
    with connect() as con:
        cur = con.execute("INSERT INTO purchase_orders(user_id, created, status, note, lines) VALUES (?,?,?,?,?)",
                          (user_id, time.time(), "Draft", note[:300], json.dumps(lines, default=str)))
        return int(cur.lastrowid)


def list_pos(user_id: int) -> list[dict]:
    with connect() as con:
        rows = con.execute("SELECT * FROM purchase_orders WHERE user_id = ? ORDER BY created DESC",
                           (user_id,)).fetchall()
    return [{**dict(r), "lines": json.loads(r["lines"])} for r in rows]


def decide_po(user_id: int, po_id: int, status: str, by: str) -> None:
    if status not in ("Approved", "Rejected", "Ordered"):
        raise ValueError("Unknown status")
    with connect() as con:
        con.execute("UPDATE purchase_orders SET status = ?, decided_at = ?, decided_by = ? WHERE id = ? AND user_id = ?",
                    (status, time.time(), by[:40], po_id, user_id))
