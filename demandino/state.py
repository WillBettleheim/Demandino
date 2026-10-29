"""App-wide state: the active dataset, the trained model and cached evaluations.

Trained models are cached in memory (per server) and, for the built-in datasets only, on
disk in signed files (across restarts), so the app starts fast after the first run. Uploaded
data is never written to disk, and in-memory caches for uploads are bounded and expire.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st

from .analysis import elasticities
from . import account, db
from .data import generate_sample_data, prepare_sales, template_csv
from .model import LEVEL_WINDOW, DemandForecaster, backtest
from .security import UploadError, check_sales_limits, clean_text, dump_signed, load_signed, read_csv_upload

MODEL_VERSION = 8
COUNTRIES = {"None": "No public holidays", "US": "United States", "GB": "United Kingdom", "IN": "India",
             "CA": "Canada", "AU": "Australia", "DE": "Germany", "FR": "France", "ES": "Spain", "IT": "Italy",
             "NL": "Netherlands", "IE": "Ireland", "BR": "Brazil", "MX": "Mexico", "JP": "Japan", "CN": "China",
             "SG": "Singapore", "AE": "United Arab Emirates", "SA": "Saudi Arabia", "ZA": "South Africa",
             "NG": "Nigeria", "KE": "Kenya", "NZ": "New Zealand", "PK": "Pakistan", "BD": "Bangladesh"}
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".cache"
REAL_DATA = ROOT / "data" / "uci_online_retail_daily.csv"


def _disk_cached(name: str, data_key: str, build: Callable[[], object]) -> object:
    if data_key.startswith(("upload-", "ws-")):
        return build()  # models of user data stay in memory only
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
def _model(data_key: str, _sales: pd.DataFrame, _country: str | None, _drivers: pd.DataFrame | None
           ) -> DemandForecaster:
    return _disk_cached("model", data_key,
                        lambda: DemandForecaster(_sales, country=_country, drivers=_drivers).fit())


@st.cache_resource(show_spinner="Back-testing the model on the last 90 days (first run only)…", max_entries=6,
                   ttl=6 * 3600)
def _backtest(data_key: str, _sales: pd.DataFrame, _country: str | None, _drivers: pd.DataFrame | None) -> dict:
    return _disk_cached("backtest", data_key,
                        lambda: backtest(_sales, holdout_days=90, country=_country, drivers=_drivers))


@st.cache_data(show_spinner=False, max_entries=6, ttl=6 * 3600)
def _elasticities(data_key: str, _sales: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    return elasticities(_sales)


@dataclass
class AppState:
    sales: pd.DataFrame
    data_key: str
    source: str
    country: str | None = None
    drivers: pd.DataFrame | None = None

    @property
    def fc(self) -> DemandForecaster:
        return _model(self.data_key, self.sales, self.country, self.drivers)

    @property
    def backtest(self) -> dict:
        return _backtest(self.data_key, self.sales, self.country, self.drivers)

    @property
    def user_id(self) -> int | None:
        return account.current_user()

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


def _read_drivers(content: bytes) -> pd.DataFrame:
    d = read_csv_upload(content, max_rows=10_000)
    d.columns = [str(c).strip().lower().replace(" ", "_") for c in d.columns]
    if "date" not in d:
        raise UploadError("The drivers file needs a `date` column plus 1–5 numeric columns.")
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
    d = d.dropna(subset=["date"]).groupby("date").mean(numeric_only=True)
    d = d.select_dtypes("number").iloc[:, :5]
    if d.empty:
        raise UploadError("No numeric driver columns found.")
    d.columns = [re.sub(r"[^a-z0-9_]", "", c)[:24] or f"driver{i}" for i, c in enumerate(d.columns)]
    return d


def sidebar_data_source() -> AppState | None:
    """Render the data-source controls in the sidebar and return the active dataset."""
    uid = account.current_user()
    with st.sidebar:
        st.markdown("**Data**")
        options = ["Sample", "Real store", "My CSV"] if REAL_DATA.exists() else ["Sample", "My CSV"]
        if uid is not None:
            options.append("Workspace")
        source = st.segmented_control("Sales data", options, default="Sample", required=True,
                                      key="data_source", label_visibility="collapsed")
        sales, key, default_country = None, None, "None"
        if source == "My CSV":
            up = st.file_uploader("Daily sales CSV", type=["csv"], key="upload")
            st.download_button("Download CSV template", template_csv(), "sales_template.csv", "text/csv",
                               icon=":material/download:", width="stretch")
            st.caption("Required: `date, product_id, units_sold, price`. Optional: `product_name, category, "
                       "regular_price, promo, on_hand` (stock on hand helps spot stock-outs).")
            if up is not None:
                content = up.getvalue()
                try:
                    sales = _load_upload(content)
                    key = "upload-" + hashlib.md5(content).hexdigest()[:16]
                except (UploadError, ValueError) as exc:
                    st.error(str(exc)[:300])
                except Exception:  # noqa: BLE001 - never show internals for a bad file
                    st.error("That file could not be processed. Check it matches the template.")
                if sales is not None and uid is not None:
                    name = st.text_input("Save as", up.name.rsplit(".", 1)[0][:60], max_chars=60, key="save_name")
                    if st.button("Save to my workspace", icon=":material/cloud_upload:", width="stretch"):
                        try:
                            db.save_dataset(uid, clean_text(name) or "My data", sales, "upload")
                            st.toast("Saved. Pick **Workspace** above to use it any time.", icon=":material/check:")
                        except db.AuthError as exc:
                            st.error(str(exc))
        elif source == "Workspace":
            saved = db.list_datasets(uid)
            if saved.empty:
                st.caption("No saved datasets yet. Upload a CSV (My CSV) or connect a store on the **Workspace** "
                           "page, then save it here.")
            else:
                ds = st.selectbox("Saved dataset", saved["id"].tolist(), key="dataset_id",
                                  format_func=lambda i: f"{saved.set_index('id').loc[i, 'name']} · "
                                                        f"{saved.set_index('id').loc[i, 'products']} products")
                sales = _load_saved(uid, int(ds), float(saved.set_index("id").loc[ds, "created"]))
                key = f"ws-{uid}-{ds}"
                default_country = db.get_setting(uid, f"country:{ds}", "None")
        elif source == "Real store":
            mtime = REAL_DATA.stat().st_mtime
            sales, key, default_country = _load_real(mtime), f"real-{int(mtime)}", "GB"
            st.caption("Real sales of a UK online gift wholesaler (Dec 2009 – Dec 2011), 40 best-selling products. "
                       "Source: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), "
                       "CC BY 4.0.")
        else:
            end = (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            sales, key, default_country = _load_sample(end), f"sample-{end}", "US"
            st.caption("Simulated store: 30 products, 6 categories, 3 years of daily sales.")
        if sales is not None:
            st.caption(f"{sales['product_id'].nunique()} products · {sales['date'].min():%d %b %Y} → "
                       f"{sales['date'].max():%d %b %Y}")

        codes = list(COUNTRIES)
        country = st.selectbox("Country (public holidays)", codes, index=codes.index(default_country),
                               format_func=lambda c: f"{COUNTRIES[c]}" + ("" if c == "None" else f" · {c}"),
                               key=f"country_{key}", help="Holidays of this country are used by the model.")
        if source == "Workspace" and sales is not None and country != default_country:
            db.set_setting(uid, f"country:{st.session_state['dataset_id']}", country)
        drivers = None
        with st.expander("Outside factors (optional)", icon=":material/thermostat:"):
            st.caption("Daily values that move demand — ad spend, temperature, search interest… CSV with `date` and "
                       "up to 5 numeric columns. Include future dates you already know (e.g. planned ad budget).")
            dfile = st.file_uploader("Drivers CSV", type=["csv"], key=f"drivers_{key}", label_visibility="collapsed")
            if dfile is not None:
                try:
                    drivers = _read_drivers(dfile.getvalue())
                    st.caption(f"Using {', '.join(drivers.columns)} ({len(drivers)} days).")
                except (UploadError, ValueError) as exc:
                    st.error(str(exc)[:300])
        c1, c2 = st.columns(2)
        cur = c1.selectbox("Show money in", ["$", "€", "£", "₹", "¥", "₦", "R", "A$"], key="currency")
        rate = c2.number_input("Exchange rate", min_value=0.0001, max_value=100000.0, value=1.0, format="%.4f",
                               key=f"fx_{key}", help="1 unit of your data's currency = this many of the shown currency. "
                                                     "Leave at 1 if your data is already in this currency.")
        if uid is not None:
            db.set_setting(uid, "currency", cur)

    if sales is None:
        st.info("Choose or upload sales data in the sidebar.", icon=":material/upload:")
        return None
    if sales.groupby("product_id")["date"].count().max() < LEVEL_WINDOW + 30:
        st.error(f"Need at least {LEVEL_WINDOW + 30} days of history for at least one product to train the model.")
        return None
    country = None if country == "None" else country
    full_key = f"{key}-{country or 'none'}"
    if abs(rate - 1.0) > 1e-9:
        sales = sales.assign(price=sales["price"] * rate, regular_price=sales["regular_price"] * rate)
        full_key += f"-fx{rate:.4f}"
    if drivers is not None:
        full_key += "-drv" + hashlib.md5(pd.util.hash_pandas_object(drivers).values.tobytes()).hexdigest()[:10]
    return AppState(sales=sales, data_key=full_key, source=source, country=country, drivers=drivers)


@st.cache_data(show_spinner=False, max_entries=4, ttl=3600)
def _load_saved(uid: int, dataset_id: int, created: float) -> pd.DataFrame | None:
    return db.load_dataset(uid, dataset_id)


def app() -> AppState:
    """The active app state for a page script (set up by app.py before the page runs)."""
    state = st.session_state.get("_app")
    if state is None:
        st.stop()
    return state
