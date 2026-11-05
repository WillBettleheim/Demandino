"""Safety measures: upload limits, signed cache files, export sanitisation and text cleaning."""
from __future__ import annotations

import hashlib
import hmac
import io
import os
import pickle
import re
import secrets
from pathlib import Path

import pandas as pd

MAX_UPLOAD_MB = 25          # also enforced by Streamlit (server.maxUploadSize)
MAX_ROWS = 2_000_000        # rows in a sales file
MAX_PRODUCTS = 300          # products the model will train on at once
MAX_YEARS = 10              # span of history
MAX_PRODUCT_DAYS = 400_000  # product-days after gaps are filled: what training's memory use depends on
MAX_VALUE = 1e9             # the most units or the highest price a row can hold
MAX_TEXT = 80               # product / scenario names

_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f<>]")  # angle brackets too: names must never read as HTML


class UploadError(ValueError):
    """A user file was rejected; the message is safe to show."""


# ----------------------------------------------------------------- uploads
def read_csv_upload(content: bytes, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    """Parse an uploaded CSV with size and row limits. Never evaluates anything in the file."""
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise UploadError(f"File is larger than {MAX_UPLOAD_MB} MB.")
    if b"\x00" in content[:4096]:
        raise UploadError("This does not look like a text CSV file.")
    try:
        df = pd.read_csv(io.BytesIO(content), nrows=max_rows + 1, low_memory=False, encoding_errors="replace")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError, MemoryError):
        raise UploadError("Could not read this as a CSV file. Check that it's plain text with one row per line.") from None
    if len(df) > max_rows:
        raise UploadError(f"Too many rows (limit {max_rows:,}). Aggregate to daily sales per product first.")
    if df.shape[1] > 50:
        raise UploadError("Too many columns (limit 50).")
    return df


def check_product_count(df: pd.DataFrame) -> None:
    products = df["product_id"].nunique()
    if products > MAX_PRODUCTS:
        raise UploadError(f"{products:,} products found; the limit is {MAX_PRODUCTS}. Upload your top products or a "
                          "category at a time.")


def check_history_size(df: pd.DataFrame) -> None:
    """Refuse a file before its gaps are filled in: two rows centuries apart would otherwise use all the memory."""
    check_product_count(df)
    span = df.groupby("product_id")["date"].agg(["min", "max"])
    longest = (span["max"] - span["min"]).dt.days
    if longest.max() > MAX_YEARS * 366:
        raise UploadError(f"History spans {int(longest.max()) // 365} years; the limit is {MAX_YEARS}. Check the dates.")
    if (longest + 1).sum() > MAX_PRODUCT_DAYS:
        raise UploadError("That's too much history for one forecast. Upload fewer products or a shorter period.")


def check_sales_limits(sales: pd.DataFrame) -> None:
    n = sales["product_id"].nunique()
    if n > MAX_PRODUCTS:
        raise UploadError(f"{n:,} products found; the limit is {MAX_PRODUCTS}. Upload your top products or a "
                          "category at a time.")
    span = (sales["date"].max() - sales["date"].min()).days
    if span > MAX_YEARS * 366:
        raise UploadError(f"History spans {span // 365} years; the limit is {MAX_YEARS}.")


def clean_text(value: str, max_len: int = MAX_TEXT) -> str:
    """Strip control characters and limit length of free text typed or uploaded by users."""
    return re.sub(r"!\[|\]\(", " ", _CONTROL.sub("", str(value))).strip()[:max_len]


# ----------------------------------------------------------------- exports
def _safe_cell(v):
    if isinstance(v, str) and v.lstrip(" \u00a0").startswith(_FORMULA_START):
        return "'" + v  # stops spreadsheet apps running it as a formula (CSV/formula injection)
    return v


def sanitize_export(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].map(_safe_cell)
    out.columns = [_safe_cell(str(c)) for c in out.columns]
    return out


def safe_csv(df: pd.DataFrame) -> str:
    return sanitize_export(df).to_csv(index=False)


# ------------------------------------------------------------ signed cache
def private_dir(path: Path) -> Path:
    """Create a folder only this account can read, tightening it if it already exists."""
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def _key(cache_dir: Path) -> bytes:
    """Per-installation secret used to sign cache files (created on first use, owner-only)."""
    private_dir(cache_dir)
    path = cache_dir / ".signing-key"
    if not path.exists():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(secrets.token_bytes(32))
    return path.read_bytes()


def dump_signed(obj: object, path: Path) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sig = hmac.new(_key(path.parent), data, hashlib.sha256).digest()
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(sig + data)
    tmp.replace(path)


def load_signed(path: Path) -> object | None:
    """Load a cache file only if its signature matches; tampered or foreign files are ignored."""
    raw = path.read_bytes()
    sig, data = raw[:32], raw[32:]
    good = hmac.new(_key(path.parent), data, hashlib.sha256).digest()
    if len(raw) < 33 or not hmac.compare_digest(sig, good):
        return None
    return pickle.loads(data)  # noqa: S301 - only files this installation signed
