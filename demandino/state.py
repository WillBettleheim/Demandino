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
from . import account, db, monitor, quality, weather
from .integrations import ConnectorError
from .data import generate_sample_data, prepare_sales, template_csv
from .model import MIN_CHECK_DAYS, MIN_TRAIN_DAYS, DemandForecaster, backtest
from .security import UploadError, check_sales_limits, clean_text, dump_signed, load_signed, read_csv_upload

MODEL_VERSION = 9
COUNTRIES = {"None": "No public holidays", "US": "United States", "GB": "United Kingdom", "IN": "India",
             "CA": "Canada", "AU": "Australia", "DE": "Germany", "FR": "France", "ES": "Spain", "IT": "Italy",
             "NL": "Netherlands", "IE": "Ireland", "BR": "Brazil", "MX": "Mexico", "JP": "Japan", "CN": "China",
             "SG": "Singapore", "AE": "United Arab Emirates", "SA": "Saudi Arabia", "ZA": "South Africa",
             "NG": "Nigeria", "KE": "Kenya", "NZ": "New Zealand", "PK": "Pakistan", "BD": "Bangladesh"}
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".cache"
REAL_DATA = ROOT / "data" / "uci_online_retail_daily.csv"


def _remove_stale_cache() -> None:
    """Delete cache files from older model versions, and any made with user data before that was disallowed."""
    for path in CACHE_DIR.glob("*.pkl"):
        if not path.stem.endswith(f"-v{MODEL_VERSION}") or "-drv" in path.stem:
            path.unlink(missing_ok=True)


def _disk_cached(name: str, data_key: str, build: Callable[[], object]) -> object:
    if data_key.startswith(("upload-", "ws-")) or "-drv" in data_key:
        return build()  # models of user data (including uploaded outside factors) stay in memory only
    _remove_stale_cache()
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
def _load_upload(content: bytes, choices: tuple = ()) -> tuple[pd.DataFrame, list]:
    """Read an upload, check it for problems, and apply the chosen fixes (see quality.py)."""
    raw = read_csv_upload(content)
    issues = quality.check(raw)
    sales = quality.apply_fixes(raw, dict(choices))
    check_sales_limits(sales)
    return sales, issues


@st.cache_resource(show_spinner="Training the forecasting model on all products (first run only)…", max_entries=6,
                   ttl=6 * 3600)
def _model(data_key: str, _sales: pd.DataFrame, _country: str | None, _drivers: pd.DataFrame | None,
           _bulk: tuple[frozenset, frozenset] = (frozenset(), frozenset())) -> DemandForecaster:
    return _disk_cached("model", data_key,
                        lambda: DemandForecaster(_sales, country=_country, drivers=_drivers, bulk_keep=_bulk[0],
                                                 bulk_add=_bulk[1]).fit())


@st.cache_resource(show_spinner="Back-testing the model on the last 90 days (first run only)…", max_entries=6,
                   ttl=6 * 3600)
def _backtest(data_key: str, _sales: pd.DataFrame, _country: str | None, _drivers: pd.DataFrame | None,
              _bulk: tuple[frozenset, frozenset] = (frozenset(), frozenset())) -> dict:
    return _disk_cached("backtest", data_key,
                        lambda: backtest(_sales, holdout_days=90, country=_country, drivers=_drivers,
                                         bulk_keep=_bulk[0], bulk_add=_bulk[1]))


@st.cache_data(show_spinner="Checking recent sales against the forecast…", max_entries=12, ttl=6 * 3600)
def _drift(data_key: str, weeks: int, _fc: DemandForecaster) -> tuple[pd.DataFrame, pd.Timestamp]:
    old, cutoff = monitor.holdout_model(_fc, weeks)
    return monitor.drift_report(old, cutoff, weeks), cutoff


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
    bulk: tuple[frozenset, frozenset] = (frozenset(), frozenset())  # (kept as real demand, marked one-off)
    base_key: str = ""  # data_key before bulk-order choices were applied

    @property
    def fc(self) -> DemandForecaster:
        return _model(self.data_key, self.sales, self.country, self.drivers, self.bulk)

    @property
    def backtest(self) -> dict:
        return _backtest(self.data_key, self.sales, self.country, self.drivers, self.bulk)

    @property
    def user_id(self) -> int | None:
        return account.current_user()

    def drift(self, weeks: int = monitor.DRIFT_WEEKS) -> tuple[pd.DataFrame, pd.Timestamp]:
        """Recent weeks' sales vs a model that never saw them (see monitor.py)."""
        return _drift(self.data_key, weeks, self.fc)

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


_KEPT = "_sidebar_kept"


def _kept_key(k: str) -> bool:
    return k in ("data_source", "currency") or k.startswith(("country_", "fx_"))


def sidebar_data_source() -> AppState | None:
    """Render the data-source controls in the sidebar and return the active dataset."""
    uid = account.current_user()
    # A page switch made from code (st.switch_page) drops sidebar widget values; put the last choices back.
    kept = st.session_state.setdefault(_KEPT, {})
    for k, v in kept.items():
        if k not in st.session_state:
            st.session_state[k] = v
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
            st.download_button("Try a messy example file", quality.messy_example(), "messy_sales_example.csv",
                               "text/csv", icon=":material/bug_report:", width="stretch",
                               help="A small file with one of every common data problem, to see the data check.")
            st.session_state.pop("_dq", None)
            if up is not None:
                content = up.getvalue()
                choices = tuple(sorted(st.session_state.get("dq_choices", {}).items()))
                try:
                    sales, issues = _load_upload(content, choices)
                    key = "upload-" + hashlib.md5(content + repr(choices).encode()).hexdigest()[:16]
                    st.session_state["_dq"] = {"name": up.name, "issues": issues, "products": sales["product_id"].nunique(),
                                               "days": len(sales), "start": sales["date"].min(),
                                               "end": sales["date"].max()}
                    todo = [i for i in issues if i.severity == "warn"]
                    if todo:
                        st.warning(f"Data check: {len(todo)} thing{'s' * (len(todo) > 1)} to review.",
                                   icon=":material/fact_check:")
                        st.page_link("views/datacheck.py", label="Review the data check", icon=":material/arrow_forward:")
                    else:
                        st.caption(":material/check_circle: Data check passed.")
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
            if sales is not None:
                wx = _weather_controls(key, sales, uid)
                if wx is not None:
                    drivers = wx if drivers is None else drivers.join(wx, how="outer")
        c1, c2 = st.columns(2)
        cur = c1.selectbox("Show money in", ["$", "€", "£", "₹", "¥", "₦", "R", "A$"], key="currency")
        rate = c2.number_input("Exchange rate", min_value=0.0001, max_value=100000.0, value=1.0, format="%.4f",
                               key=f"fx_{key}", help="1 unit of your data's currency = this many of the shown currency. "
                                                     "Leave at 1 if your data is already in this currency.")
        if uid is not None:
            db.set_setting(uid, "currency", cur)
    kept.update({k: v for k, v in st.session_state.items() if isinstance(k, str) and _kept_key(k)})

    if sales is None:
        st.info("Choose or upload sales data in the sidebar.", icon=":material/upload:")
        return None
    if sales.groupby("product_id")["date"].count().max() < MIN_TRAIN_DAYS + MIN_CHECK_DAYS:
        st.error(f"Need at least {MIN_TRAIN_DAYS + MIN_CHECK_DAYS} days of history for at least one product: "
                 f"{MIN_TRAIN_DAYS} to learn from and {MIN_CHECK_DAYS} to test the forecast on.")
        return None
    country = None if country == "None" else country
    full_key = f"{key}-{country or 'none'}"
    if abs(rate - 1.0) > 1e-9:
        sales = sales.assign(price=sales["price"] * rate, regular_price=sales["regular_price"] * rate)
        full_key += f"-fx{rate:.4f}"
    if drivers is not None:
        full_key += "-drv" + hashlib.md5(pd.util.hash_pandas_object(drivers).values.tobytes()).hexdigest()[:10]
    base_key = full_key
    keep, add = bulk_choices(base_key)
    if keep or add:
        full_key += "-blk" + hashlib.md5(repr((sorted(keep), sorted(add))).encode()).hexdigest()[:10]
    return AppState(sales=sales, data_key=full_key, source=source, country=country, drivers=drivers,
                    bulk=(keep, add), base_key=base_key)


@st.cache_data(show_spinner="Fetching weather…", max_entries=8, ttl=6 * 3600)
def _weather(lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return weather.daily_weather(lat, lon, start, end)


def _weather_controls(key: str, sales: pd.DataFrame, uid: int | None) -> pd.DataFrame | None:
    """Optional daily temperature and rain for a chosen place, as outside factors (sends only the place's
    coordinates and dates to Open-Meteo)."""
    setting = f"weather:{key}"
    place = st.session_state.get(setting)
    if place is None and uid is not None:
        place = db.get_setting(uid, setting)
    st.markdown("**Weather** · helps drinks, clothing, garden and other weather-sensitive products")
    if place is None:
        city = st.text_input("Town or city", key=f"wx_city_{key}", max_chars=80, placeholder="e.g. London")
        if st.button("Add weather", icon=":material/partly_cloudy_day:", disabled=not city, key=f"wx_add_{key}"):
            try:
                place = weather.find_place(city)
                st.session_state[setting] = place
                if uid is not None:
                    db.set_setting(uid, setting, place)
                st.rerun()
            except ConnectorError as exc:
                st.error(str(exc))
        st.caption("Only the place's location and dates are sent to Open-Meteo — never your sales.")
        return None
    try:
        wx = _weather(place["latitude"], place["longitude"], sales["date"].min(), sales["date"].max())
    except ConnectorError as exc:
        st.error(f"Weather unavailable: {exc}")
        return None
    st.caption(f":material/partly_cloudy_day: {place['name']}, {place['country']}: {weather.describe(wx)}")
    if st.button("Remove weather", icon=":material/close:", key=f"wx_remove_{key}"):
        st.session_state[setting] = None
        st.session_state.pop(setting)
        if uid is not None:
            db.set_setting(uid, setting, None)
        st.rerun()
    return wx


def _bulk_setting(base_key: str) -> str:
    return f"bulk:{base_key}"


def bulk_choices(base_key: str) -> tuple[frozenset, frozenset]:
    """Days the user said are real demand (keep) or one-off bulk orders (add), for this dataset."""
    got = st.session_state.get(_bulk_setting(base_key))
    if got is None:
        uid = account.current_user()
        got = db.get_setting(uid, _bulk_setting(base_key)) if uid else None
        got = got or {"keep": [], "add": []}
        st.session_state[_bulk_setting(base_key)] = got
    return (frozenset(tuple(x) for x in got["keep"]), frozenset(tuple(x) for x in got["add"]))


def set_bulk_choices(base_key: str, keep: set, add: set) -> None:
    got = {"keep": sorted(list(x) for x in keep), "add": sorted(list(x) for x in add)}
    st.session_state[_bulk_setting(base_key)] = got
    uid = account.current_user()
    if uid:
        db.set_setting(uid, _bulk_setting(base_key), got)


@st.cache_data(show_spinner=False, max_entries=4, ttl=3600)
def _load_saved(uid: int, dataset_id: int, created: float) -> pd.DataFrame | None:
    return db.load_dataset(uid, dataset_id)


def app() -> AppState:
    """The active app state for a page script (set up by app.py before the page runs)."""
    state = st.session_state.get("_app")
    if state is None:
        st.stop()
    return state
