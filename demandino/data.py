"""Sales data: a realistic synthetic generator and a loader for user CSV files.

Canonical daily schema (one row per product per day):
    date, product_id, product_name, category, price, regular_price, promo, units_sold
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .security import MAX_VALUE, check_history_size

REQUIRED_COLUMNS = ["date", "product_id", "units_sold", "price"]
OPTIONAL_COLUMNS = ["product_name", "category", "regular_price", "promo"]

# Monthly seasonality (Jan..Dec) and weekly pattern (Mon..Sun) per category.
CATEGORY_PROFILES = {
    "Electronics": dict(
        base=14, price=(45, 420), elasticity=(1.6, 2.6),
        months=[0.85, 0.80, 0.85, 0.90, 0.90, 0.95, 0.95, 1.05, 1.00, 1.00, 1.35, 1.60],
        week=[0.92, 0.88, 0.90, 0.96, 1.08, 1.22, 1.04], bf_boost=2.6,
        names=["Wireless Earbuds", "Smart Watch", "Bluetooth Speaker", "4K Webcam", "Power Bank 20k"],
    ),
    "Outdoor & Garden": dict(
        base=10, price=(15, 180), elasticity=(1.2, 2.0),
        months=[0.45, 0.55, 0.90, 1.25, 1.55, 1.65, 1.55, 1.30, 1.00, 0.75, 0.55, 0.50],
        week=[0.85, 0.82, 0.86, 0.92, 1.05, 1.35, 1.15], bf_boost=1.4,
        names=["Garden Hose 30m", "Camping Lantern", "Folding Chair", "Solar Path Lights", "Pruning Shears"],
    ),
    "Beverages": dict(
        base=45, price=(1.5, 12), elasticity=(1.4, 2.4),
        months=[0.80, 0.80, 0.90, 1.00, 1.15, 1.35, 1.45, 1.40, 1.10, 0.95, 0.85, 0.95],
        week=[0.88, 0.86, 0.90, 0.98, 1.12, 1.20, 1.06], bf_boost=1.2,
        names=["Sparkling Water 6pk", "Cold Brew Coffee", "Energy Drink", "Iced Tea Lemon", "Orange Juice 1L"],
    ),
    "Apparel": dict(
        base=12, price=(12, 140), elasticity=(1.5, 2.8),
        months=[0.80, 0.75, 1.00, 1.10, 1.05, 0.95, 0.90, 1.15, 1.20, 1.05, 1.15, 1.30],
        week=[0.86, 0.84, 0.88, 0.95, 1.10, 1.30, 1.07], bf_boost=2.0,
        names=["Running Shoes", "Hoodie Classic", "Denim Jacket", "Wool Beanie", "Sports Socks 3pk"],
    ),
    "Home & Kitchen": dict(
        base=16, price=(8, 160), elasticity=(1.1, 1.9),
        months=[0.95, 0.90, 0.95, 1.00, 1.00, 0.95, 0.95, 1.00, 1.00, 1.00, 1.15, 1.30],
        week=[0.94, 0.92, 0.94, 0.98, 1.04, 1.14, 1.04], bf_boost=1.8,
        names=["Chef Knife", "Air Fryer", "Coffee Grinder", "Nonstick Pan", "Glass Containers Set"],
    ),
    "Toys & Games": dict(
        base=9, price=(10, 90), elasticity=(1.3, 2.2),
        months=[0.60, 0.60, 0.70, 0.75, 0.80, 0.80, 0.85, 0.85, 0.90, 1.00, 1.60, 2.40],
        week=[0.85, 0.82, 0.86, 0.92, 1.08, 1.35, 1.12], bf_boost=2.4,
        names=["Building Blocks Set", "Board Game Classic", "RC Car", "Puzzle 1000pc", "Plush Bear"],
    ),
}


def black_friday(year: int) -> pd.Timestamp:
    """Day after the 4th Thursday of November."""
    nov1 = pd.Timestamp(year=year, month=11, day=1)
    first_thu = nov1 + pd.Timedelta(days=(3 - nov1.weekday()) % 7)
    return first_thu + pd.Timedelta(days=22)


def monthly_curve(months: list[float], dates: pd.DatetimeIndex) -> np.ndarray:
    """Smooth, periodic interpolation of 12 monthly factors onto daily dates."""
    mids = np.array([15 + 30.44 * i for i in range(12)])
    xs = np.concatenate([mids - 365.25, mids, mids + 365.25])
    ys = np.tile(np.asarray(months, dtype=float), 3)
    return np.interp(dates.day_of_year.to_numpy(), xs, ys)


def holiday_multiplier(dates: pd.DatetimeIndex, bf_boost: float) -> np.ndarray:
    mult = np.ones(len(dates))
    for year in sorted(set(dates.year)):
        bf = black_friday(year)
        for offset, strength in [(0, 1.0), (1, 0.7), (2, 0.5), (3, 0.85)]:  # Fri..Cyber Monday
            mult[dates == bf + pd.Timedelta(days=offset)] *= 1 + (bf_boost - 1) * strength
        xmas = pd.Timestamp(year=year, month=12, day=25)
        rush = (dates >= xmas - pd.Timedelta(days=14)) & (dates < xmas - pd.Timedelta(days=1))
        mult[rush] *= 1 + 0.12 * (bf_boost - 1)
        mult[dates == xmas] *= 0.25
        mult[dates == pd.Timestamp(year=year, month=1, day=1)] *= 0.55
    return mult


def generate_sample_data(end_date: str | pd.Timestamp | None = None, years: int = 3, seed: int = 42) -> pd.DataFrame:
    """Simulate daily sales for 30 products with trend, seasonality, promos, price changes and noise."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    dates = pd.date_range(end - pd.Timedelta(days=int(365.25 * years) - 1), end, freq="D")
    n = len(dates)
    t_years = np.arange(n) / 365.25
    frames = []
    pid = 0

    for category, prof in CATEGORY_PROFILES.items():
        season = monthly_curve(prof["months"], dates)
        weekly = np.asarray(prof["week"])[dates.day_of_week]
        holidays = holiday_multiplier(dates, prof["bf_boost"])
        lo, hi = prof["price"]
        cat_mid = np.sqrt(lo * hi)

        for name in prof["names"]:
            pid += 1
            anchor = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            anchor = round(anchor, 2)
            elasticity = rng.uniform(*prof["elasticity"])
            # Cheaper products in a category sell more units.
            base = prof["base"] * (anchor / cat_mid) ** -0.8 * rng.lognormal(0, 0.3)
            growth = rng.normal(0.06, 0.15)
            trend = np.exp(growth * t_years) * (1 + 0.08 * np.sin(2 * np.pi * t_years / rng.uniform(1.5, 3) + rng.uniform(0, 6)))

            # Regular price: mean-reverting random walk, changing every 60-120 days.
            regular = np.empty(n)
            level, i = anchor, 0
            while i < n:
                span = int(rng.integers(60, 121))
                regular[i:i + span] = round(level, 2)
                level = anchor * float(np.clip(level / anchor * np.exp(rng.normal(0, 0.09)) ** 0.7, 0.8, 1.25))
                i += span

            # Promotions: random 3-10 day events plus most Black Friday weekends.
            promo = np.zeros(n, dtype=int)
            discount = np.zeros(n)
            day = int(rng.integers(0, 30))
            while day < n:
                length = int(rng.integers(3, 11))
                promo[day:day + length] = 1
                discount[day:day + length] = rng.uniform(0.10, 0.35)
                day += length + int(rng.integers(25, 70))
            for year in sorted(set(dates.year)):
                if rng.random() < 0.8:
                    bf_mask = (dates >= black_friday(year)) & (dates <= black_friday(year) + pd.Timedelta(days=3))
                    promo[bf_mask] = 1
                    discount[bf_mask] = np.maximum(discount[bf_mask], rng.uniform(0.2, 0.4))

            price = np.round(regular * (1 - discount), 2)
            price_effect = (price / anchor) ** -elasticity
            promo_lift = np.where(promo == 1, 1.15, 1.0)  # display/visibility effect beyond price

            mu = base * trend * season * weekly * holidays * price_effect * promo_lift
            mu *= rng.lognormal(0, 0.05, n)  # day-level shocks (weather, local events)
            r = 6.0  # negative-binomial dispersion
            units = rng.negative_binomial(r, r / (r + mu))
            # Occasional stock-outs: nothing sells for a few days until the next delivery arrives.
            for _ in range(int(rng.integers(1, 4))):
                at, length = int(rng.integers(60, n - 20)), int(rng.integers(3, 11))
                units[at:at + length] = 0

            frames.append(pd.DataFrame({
                "date": dates,
                "product_id": f"P{pid:03d}",
                "product_name": name,
                "category": category,
                "price": price,
                "regular_price": regular,
                "promo": promo,
                "units_sold": units,
            }))

    return pd.concat(frames, ignore_index=True)


COLUMN_NAMES = {
    "date": ("order_date", "sale_date", "sales_date", "transaction_date", "invoice_date", "created_at", "day"),
    "product_id": ("sku", "item_id", "item", "item_code", "stockcode", "stock_code", "product_code", "code",
                   "variant_sku", "lineitem_sku"),
    "units_sold": ("quantity", "units", "sales", "qty", "qty_sold", "quantity_sold", "items_sold", "net_items_sold",
                   "net_quantity", "ordered_quantity", "lineitem_quantity"),
    "price": ("unit_price", "selling_price", "sale_price", "price_each", "avg_price", "lineitem_price"),
    "product_name": ("product", "product_title", "item_name", "title", "name", "description", "lineitem_name"),
    "category": ("product_type", "type", "department", "group"),
}


def standard_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """A copy with common export column names (Order Date, SKU, Qty…) changed to the ones used here."""
    df = raw.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    for standard, names in COLUMN_NAMES.items():
        if standard not in df.columns:
            found = next((n for n in names if n in df.columns), None)
            if found:
                df = df.rename(columns={found: standard})
    if "product_id" not in df.columns and "product_name" in df.columns:
        df["product_id"] = df["product_name"]
    return df


def parse_dates(values: pd.Series) -> pd.Series:
    """Dates as days. Numeric dates like 03/04/2025 are read day-first when the file shows that's its format (some first
    number is above 12); otherwise month-first. Guessing row by row would scramble the history."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return values.dt.normalize()
    text = values.astype("string").str.strip()
    parts = text.str.extract(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})")
    numeric = parts[0].notna()
    out = pd.to_datetime(text.where(~numeric), errors="coerce", format="mixed")
    if numeric.any():
        day_first = bool((pd.to_numeric(parts[0], errors="coerce") > 12).any())
        year = parts[2].where(parts[2].str.len() == 4, "20" + parts[2])
        day, month = (parts[0], parts[1]) if day_first else (parts[1], parts[0])
        got = pd.to_datetime({"year": pd.to_numeric(year), "month": pd.to_numeric(month), "day": pd.to_numeric(day)},
                             errors="coerce")
        out = out.where(~numeric, got)
    return out.dt.normalize()


def parse_numbers(values: pd.Series) -> pd.Series:
    """Numbers, accepting what spreadsheets export: $1,234.50, 1 234, 10,50 and similar."""
    if pd.api.types.is_numeric_dtype(values):
        return values
    text = values.astype("string").str.replace(r"[^\d,.\-]", "", regex=True)
    decimal_comma = text.str.fullmatch(r"-?\d+,\d{1,2}").fillna(False)
    text = text.where(~decimal_comma, text.str.replace(",", "."))
    got = pd.to_numeric(text.str.replace(",", ""), errors="coerce")
    return pd.Series(got.astype("Float64").to_numpy(dtype=float, na_value=np.nan), index=values.index)


def prepare_sales(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise a user dataset to the canonical daily schema.

    Missing days are filled with zero sales (price carried forward), optional columns
    are inferred, and duplicate rows per product/day are aggregated.
    """
    df = standard_columns(raw)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        words = {"date": "the date", "product_id": "the product", "units_sold": "how many sold", "price": "the price"}
        found = re.sub(r"[^\w ,.-]", "", ", ".join(str(c)[:30] for c in raw.columns[:12]))  # file text shown in a message
        raise ValueError(f"Couldn't find a column for {' or '.join(words[c] for c in missing)}. Your file has: {found}. "
                         "Rename the column (for example to date, product_id, units_sold or price) and upload it again.")

    df["date"] = parse_dates(df["date"])
    df["units_sold"] = parse_numbers(df["units_sold"])
    df["price"] = parse_numbers(df["price"])
    df = df.dropna(subset=["date", "units_sold", "price"])
    df = df[(df["units_sold"].abs() <= MAX_VALUE) & df["price"].between(0, MAX_VALUE)]  # also drops infinity
    if df.empty:
        raise ValueError("Couldn't read any rows: check that the date, units and price columns hold dates and numbers.")
    df["product_id"] = df["product_id"].astype(str)
    check_history_size(df)
    if "product_name" not in df.columns:
        df["product_name"] = df["product_id"]
    if "category" not in df.columns:
        df["category"] = "General"
    df["category"] = df["category"].fillna("General").astype(str)
    # Names come from user files: strip control characters and cap the length.
    for col in ("product_id", "product_name", "category"):
        df[col] = (df[col].astype(str).str.replace(r"[\x00-\x1f\x7f<>]", "", regex=True)
                   .str.replace(r"!\[|\]\(", " ", regex=True).str.strip().str[:80])  # never markup, images or links

    agg = {"units_sold": "sum", "price": "mean", "product_name": "first", "category": "first"}
    for col in ("regular_price", "promo"):
        if col in df.columns:
            agg[col] = "max"
    df = df.groupby(["product_id", "date"], as_index=False).agg(agg)

    out = []
    for pid, g in df.groupby("product_id"):
        g = g.set_index("date").sort_index()
        full = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(full)
        g["units_sold"] = g["units_sold"].fillna(0).clip(lower=0)
        g["price"] = g["price"].ffill()
        g[["product_name", "category"]] = g[["product_name", "category"]].ffill()
        if "regular_price" in g.columns:
            g["regular_price"] = g["regular_price"].ffill()
        else:
            # Regular price = typical recent ceiling of the selling price.
            g["regular_price"] = g["price"].rolling(56, min_periods=1).quantile(0.9)
        g["regular_price"] = np.maximum(g["regular_price"], g["price"])
        if "promo" in g.columns:
            g["promo"] = pd.to_numeric(g["promo"], errors="coerce").fillna(0).clip(0, 1).round().astype(int)
        else:
            g["promo"] = (g["price"] < 0.97 * g["regular_price"]).astype(int)
        g["product_id"] = pid
        out.append(g.rename_axis("date").reset_index())

    cols = ["date", "product_id", "product_name", "category", "price", "regular_price", "promo", "units_sold"]
    return pd.concat(out, ignore_index=True)[cols].sort_values(["product_id", "date"]).reset_index(drop=True)


def template_csv() -> str:
    """A small example of the upload format."""
    sample = generate_sample_data(end_date="2026-01-31", years=1, seed=1)
    sample = sample[sample["product_id"].isin(["P001", "P011"])].tail(10)
    sample["date"] = sample["date"].dt.strftime("%Y-%m-%d")
    return sample.to_csv(index=False)
