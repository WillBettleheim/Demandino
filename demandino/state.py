"""App-wide state: the active dataset, the trained model and cached evaluations.

Trained models are cached in memory (per server) and, for the built-in datasets only, on
disk in signed files (across restarts), so the app starts fast after the first run. Uploaded
data is never written to disk, and in-memory caches for uploads are bounded and expire.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st

from .analysis import elasticities
from .data import generate_sample_data, prepare_sales, template_csv
from .model import LEVEL_WINDOW, DemandForecaster, backtest
from .security import UploadError, check_sales_limits, dump_signed, load_signed, read_csv_upload

MODEL_VERSION = 6
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".cache"
REAL_DATA = ROOT / "data" / "uci_online_retail_daily.csv"


def _disk_cached(name: str, data_key: str, build: Callable[[], object]) -> object:
    if data_key.startswith("upload-"):
        return build()  # user data stays in memory only
    path = CACHE_DIR / f"{name}-{data_key}-v{MODEL_VERSION}.pkl"
    if path.exists():
        try:
            obj = load_signed(path)
            if obj is not None:
                return obj
        except Exception:  # noqa: BLE001 - stale or corrupt cache: rebuild
            pass
        path.unlink(missing_ok=True)
    obj = build()
    try:
        dump_signed(obj, path)
    except OSError:
        pass  # read-only deployments still work, just without the disk cache
    return obj


@st.cache_data(show_spinner=False)
def _load_sample(end: str) -> pd.DataFrame:
    return generate_sample_data(end_date=end)


@st.cache_data(show_spinner=False)
def _load_real(mtime: float) -> pd.DataFrame:
    return prepare_sales(pd.read_csv(REAL_DATA))


@st.cache_data(show_spinner=False, max_entries=4, ttl=3600)
def _load_upload(content: bytes) -> pd.DataFrame:
    sales = prepare_sales(read_csv_upload(content))
    check_sales_limits(sales)
    return sales


@st.cache_resource(show_spinner="Training the forecasting model on all products (first run only)…", max_entries=6,
                   ttl=6 * 3600)
def _model(data_key: str, _sales: pd.DataFrame) -> DemandForecaster:
    return _disk_cached("model", data_key, lambda: DemandForecaster(_sales).fit())


@st.cache_resource(show_spinner="Back-testing the model on the last 90 days (first run only)…", max_entries=6,
                   ttl=6 * 3600)
def _backtest(data_key: str, _sales: pd.DataFrame) -> dict:
    return _disk_cached("backtest", data_key, lambda: backtest(_sales, holdout_days=90))


@st.cache_data(show_spinner=False, max_entries=6, ttl=6 * 3600)
def _elasticities(data_key: str, _sales: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    return elasticities(_sales)


@dataclass
class AppState:
    sales: pd.DataFrame
    data_key: str
    source: str

    @property
    def fc(self) -> DemandForecaster:
        return _model(self.data_key, self.sales)

    @property
    def backtest(self) -> dict:
        return _backtest(self.data_key, self.sales)

    @property
    def elasticities(self) -> tuple[dict[str, float], dict[str, float]]:
        return _elasticities(self.data_key, self.sales)

    @property
    def level_uncertainty(self) -> float:
        """Level error calibrated on real held-out sales (sets the width of multi-day ranges)."""
        return self.fc.level_uncertainty

    @property
    def products(self) -> pd.DataFrame:
        return self.sales.groupby("product_id")[["product_name", "category"]].first().reset_index()

    @property
    def label_of(self) -> dict[str, str]:
        return {r.product_id: f"{r.product_name} · {r.product_id}" for r in self.products.itertuples()}

    @property
    def categories(self) -> list[str]:
        return sorted(self.sales["category"].unique())

    @property
    def last_date(self) -> pd.Timestamp:
        return self.sales["date"].max()

    def history(self, pid: str) -> pd.DataFrame:
        return self.sales[self.sales["product_id"] == pid].sort_values("date")


def sidebar_data_source() -> AppState | None:
    """Render the data-source controls in the sidebar and return the active dataset."""
    with st.sidebar:
        st.markdown("**Data**")
        options = ["Sample", "Real store", "My CSV"] if REAL_DATA.exists() else ["Sample", "My CSV"]
        source = st.segmented_control("Sales data", options, default="Sample", required=True,
                                      key="data_source", label_visibility="collapsed")
        sales, key = None, None
        if source == "My CSV":
            up = st.file_uploader("Daily sales CSV", type=["csv"], key="upload")
            st.download_button("Download CSV template", template_csv(), "sales_template.csv", "text/csv",
                               icon=":material/download:", width="stretch")
            st.caption("Required: `date, product_id, units_sold, price`. Optional: `product_name, category, "
                       "regular_price, promo`. One row per product per day.")
            if up is not None:
                content = up.getvalue()
                try:
                    sales = _load_upload(content)
                    key = "upload-" + hashlib.md5(content).hexdigest()[:16]
                except (UploadError, ValueError) as exc:
                    st.error(str(exc)[:300])
                except Exception:  # noqa: BLE001 - never show internals for a bad file
                    st.error("That file could not be processed. Check it matches the template.")
        elif source == "Real store":
            mtime = REAL_DATA.stat().st_mtime
            sales, key = _load_real(mtime), f"real-{int(mtime)}"
            st.caption("Real sales of a UK online gift wholesaler (Dec 2009 – Dec 2011), 40 best-selling products. "
                       "Source: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), "
                       "CC BY 4.0.")
        else:
            end = (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            sales, key = _load_sample(end), f"sample-{end}"
            st.caption("Simulated store: 30 products, 6 categories, 3 years of daily sales.")
        if sales is not None:
            st.caption(f"{sales['product_id'].nunique()} products · {sales['date'].min():%d %b %Y} → "
                       f"{sales['date'].max():%d %b %Y}")
        st.selectbox("Currency", ["$", "€", "£", "₹", "¥", "₦", "R", "A$"], key="currency")

    if sales is None:
        st.info("Upload a daily sales CSV in the sidebar, or switch to the sample dataset.", icon=":material/upload:")
        return None
    if sales.groupby("product_id")["date"].count().max() < LEVEL_WINDOW + 30:
        st.error(f"Need at least {LEVEL_WINDOW + 30} days of history for at least one product to train the model.")
        return None
    return AppState(sales=sales, data_key=key, source=source)


def app() -> AppState:
    """The active app state for a page script (set up by app.py before the page runs)."""
    state = st.session_state.get("_app")
    if state is None:
        st.stop()
    return state
