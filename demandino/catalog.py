"""Edit or remove products without touching the sales data: changes are kept apart and laid over it, so anything can be
put back. Past sales keep their real prices; a new price applies from the latest day onwards."""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import streamlit as st

from . import account, db
from .model import MIN_CHECK_DAYS, MIN_TRAIN_DAYS
from .security import clean_text

MAX_NAME = 80
EMPTY = {"changes": {}, "removed": []}


def dataset_id(key: str) -> str:
    """A name for the dataset that stays the same from day to day (the sample's and real store's keys contain dates)."""
    return key.split("-")[0] if key.startswith(("sample-", "real-")) else key


def _setting(dataset: str) -> str:
    return f"catalog:{dataset}"


def load(dataset: str) -> dict:
    """The saved changes for a dataset."""
    got = st.session_state.get(_setting(dataset))
    if got is None:
        uid = account.current_user()
        got = (db.get_setting(uid, _setting(dataset)) if uid else None) or EMPTY
        st.session_state[_setting(dataset)] = got
    return got


def save(dataset: str, edits: dict) -> None:
    st.session_state[_setting(dataset)] = edits
    uid = account.current_user()
    if uid:
        db.set_setting(uid, _setting(dataset), edits)


def has_edits(edits: dict) -> bool:
    return bool(edits.get("changes") or edits.get("removed"))


def digest(edits: dict) -> str:
    return hashlib.md5(json.dumps(edits, sort_keys=True).encode()).hexdigest()[:10]


def apply(sales: pd.DataFrame, edits: dict) -> pd.DataFrame:
    """Sales with the changes laid over them. The data passed in is never altered."""
    if not has_edits(edits):
        return sales
    out = sales[~sales["product_id"].isin(edits.get("removed", []))].copy()
    last = out.groupby("product_id")["date"].transform("max") == out["date"]
    for pid, change in edits.get("changes", {}).items():
        rows = out["product_id"] == pid
        if "name" in change:
            out.loc[rows, "product_name"] = change["name"]
        if "category" in change:
            out.loc[rows, "category"] = change["category"]
        if "price" in change:
            latest = rows & last
            discount = (out.loc[latest, "price"] / out.loc[latest, "regular_price"]).fillna(1.0)
            out.loc[latest, "regular_price"] = change["price"]
            out.loc[latest, "price"] = change["price"] * discount
    return out.reset_index(drop=True)


def can_forecast(sales: pd.DataFrame) -> bool:
    """Whether what's left still has a product with enough history to train and test the model on."""
    return not sales.empty and sales.groupby("product_id")["date"].count().max() >= MIN_TRAIN_DAYS + MIN_CHECK_DAYS


def merge(edits: dict, original: pd.DataFrame, rows: pd.DataFrame, fx_rate: float = 1.0) -> tuple[dict, list[str]]:
    """Work out new edits from the edited table; returns them with what changed, in words.

    `original` is one row per product before any edits; `rows` has product_id, name, category, price (in the currency
    shown, which `fx_rate` converts from the data's) and remove."""
    changes = {k: dict(v) for k, v in edits.get("changes", {}).items()}
    removed = set(edits.get("removed", []))
    base = original.set_index("product_id")
    log = []
    for r in rows.itertuples():
        pid = str(r.product_id)
        name, category = clean_text(r.name or "", MAX_NAME), clean_text(r.category or "", MAX_NAME)
        if not name or not category:
            raise ValueError(f"{pid} needs a name and a category.")
        if pd.isna(r.price) or r.price <= 0:
            raise ValueError(f"{pid} needs a price above zero.")
        if r.remove:
            removed.add(pid)
            log.append(f"Removed product {pid}")
            continue
        was = base.loc[pid]
        change = {}
        if name != was["product_name"]:
            change["name"] = name
        if category != was["category"]:
            change["category"] = category
        shown = float(was["regular_price"]) * fx_rate
        if abs(float(r.price) - shown) >= 0.005:
            change["price"] = float(r.price) / fx_rate
        if change != changes.get(pid, {}):
            log.append(f"Edited product {pid}")
        if change:
            changes[pid] = change
        else:
            changes.pop(pid, None)
    return {"changes": changes, "removed": sorted(removed)}, log


def originals(sales: pd.DataFrame) -> pd.DataFrame:
    """One row per product: name, category and latest regular price."""
    return (sales.sort_values("date").groupby("product_id")[["product_name", "category", "regular_price"]].last()
            .reset_index())
