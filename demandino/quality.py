"""Data check for uploaded sales files: find problems before training, explain them, and offer fixes.

`check` looks at the raw file (before it is tidied), so nothing is fixed silently. `apply_fixes` applies the
chosen fixes and returns clean daily sales ready for the model. Each issue's first fix is the default: the
safest choice when you don't know more about your data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import generate_sample_data, parse_dates, parse_numbers, prepare_sales, standard_columns
from .model import MIN_TRAIN_DAYS

LONG_GAP = 7          # missing days in a row that look like "no data" (or no stock) rather than no sales
PRICE_FACTOR = 10     # a price this many times above/below the product's usual price is suspicious
MIN_DAYS = MIN_TRAIN_DAYS  # history a product needs to be forecast from its own sales


@dataclass
class Issue:
    code: str
    title: str
    count: int
    explain: str
    fixes: dict[str, str] = field(default_factory=dict)   # option -> what it does; first one is the default
    examples: pd.DataFrame | None = None
    severity: str = "warn"  # "warn" or "info"

    @property
    def default(self) -> str:
        return next(iter(self.fixes), "")


def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Column names and types as prepare_sales sees them, without dropping or merging anything."""
    df = standard_columns(raw)
    for col in ("date", "product_id", "units_sold", "price"):
        if col not in df.columns:
            return df  # prepare_sales reports the missing column clearly
    df["_date"] = parse_dates(df["date"])
    df["_units"] = parse_numbers(df["units_sold"])
    df["_price"] = parse_numbers(df["price"])
    df["product_id"] = df["product_id"].astype(str)
    return df


def check(raw: pd.DataFrame, today: pd.Timestamp | None = None) -> list[Issue]:
    df = _normalise(raw)
    if "_date" not in df:
        return []
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    show = ["date", "product_id", "units_sold", "price"]
    issues = []

    bad = df["_date"].isna() | df["_units"].isna() | df["_price"].isna()
    if bad.any():
        issues.append(Issue("unreadable", "Rows that can't be read", int(bad.sum()),
                            "The date, units or price on these rows isn't a valid value, so they can't be used.",
                            {"drop": "Leave them out"}, df.loc[bad, show].head(5)))
    ok = df[~bad]

    future = ok["_date"] > today
    if future.any():
        issues.append(Issue("future", "Dates in the future", int(future.sum()),
                            "Sales dated after today are usually typos (e.g. 2062 instead of 2026) or planned "
                            "orders, not real sales.", {"drop": "Leave them out", "keep": "Keep them"},
                            ok.loc[future, show].head(5)))

    dup = ok.duplicated(["product_id", "_date"], keep=False)
    if dup.any():
        issues.append(Issue("duplicates", "More than one row for the same product and day",
                            int(ok[dup].groupby(["product_id", "_date"]).ngroups),
                            "This happens with one row per order (fine — they get added up) or when the same "
                            "export was pasted twice (then adding them up double-counts).",
                            {"sum": "Add them up (one row per order)", "last": "Keep only the last row (repeated rows)"},
                            ok.loc[dup, show].sort_values(["product_id", "date"]).head(6)))

    neg = ok["_units"] < 0
    if neg.any():
        issues.append(Issue("negative", "Negative sales (returns or corrections)", int(neg.sum()),
                            "Returns are not lost demand: customers wanted the product when they bought it.",
                            {"zero": "Count those rows as 0", "net": "Subtract them from that day's sales"},
                            ok.loc[neg, show].head(5)))

    med = ok[ok["_price"] > 0].groupby("product_id")["_price"].median()
    typical = ok["product_id"].map(med)
    odd = (ok["_price"] <= 0) | (ok["_price"] > PRICE_FACTOR * typical) | (ok["_price"] < typical / PRICE_FACTOR)
    if odd.any():
        issues.append(Issue("prices", "Prices that look wrong", int(odd.sum()),
                            f"Zero or negative prices, or more than {PRICE_FACTOR}× away from the product's usual "
                            "price — often a missing decimal point or a free sample.",
                            {"usual": "Use the product's usual price", "keep": "Keep them as they are"},
                            ok.loc[odd, show].head(5)))

    gaps, longest, gap_rows = 0, 0, []
    for pid, g in ok[~future].groupby("product_id"):
        days = np.sort(g["_date"].unique())
        if len(days) < 2:
            continue
        step = np.diff(days).astype("timedelta64[D]").astype(int) - 1
        long = step >= LONG_GAP
        gaps += int(step[long].sum())
        longest = max(longest, int(step.max()))
        for i in np.flatnonzero(long)[:2]:
            gap_rows.append({"product_id": pid, "from": pd.Timestamp(days[i]) + pd.Timedelta(days=1),
                             "to": pd.Timestamp(days[i + 1]) - pd.Timedelta(days=1), "days": int(step[i])})
    if gaps:
        issues.append(Issue("gaps", f"Gaps of {LONG_GAP}+ days with no rows", gaps,
                            f"Missing days in a row (longest {longest}). Short gaps usually mean no sales; long ones "
                            "often mean the product was out of stock or the data is missing.",
                            {"zero": "Count them as days with no sales", "stockout": "Treat them as out of stock "
                             "(the model won't learn from them)"}, pd.DataFrame(gap_rows).head(6)))

    span = ok[~future].groupby("product_id")["_date"].agg(lambda d: (d.max() - d.min()).days + 1)
    short = span[span < MIN_DAYS]
    if len(short):
        issues.append(Issue("short", "Products with little history", len(short),
                            f"Under {MIN_DAYS} days of sales. They can still be forecast, using patterns from similar "
                            "products, but expect wider ranges.", {}, short.rename("days").reset_index().head(6),
                            severity="info"))
    return issues


def apply_fixes(raw: pd.DataFrame, choices: dict[str, str] | None = None, today: pd.Timestamp | None = None
                ) -> pd.DataFrame:
    """Clean daily sales with the chosen fix per issue code (defaults: the first option of each issue)."""
    choices = choices or {}
    df = _normalise(raw)
    if "_date" not in df:
        return prepare_sales(raw)  # raises the usual "missing column" message
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    df = df.dropna(subset=["_date", "_units", "_price"])
    if choices.get("future", "drop") == "drop":
        df = df[df["_date"] <= today]
    if choices.get("duplicates", "sum") == "last":
        df = df.drop_duplicates(["product_id", "_date"], keep="last")
    if choices.get("negative", "zero") == "zero":
        df = df.assign(_units=df["_units"].clip(lower=0))
    if choices.get("prices", "usual") == "usual":
        med = df[df["_price"] > 0].groupby("product_id")["_price"].median()
        typical = df["product_id"].map(med)
        odd = (df["_price"] <= 0) | (df["_price"] > PRICE_FACTOR * typical) | (df["_price"] < typical / PRICE_FACTOR)
        df = df.assign(_price=df["_price"].where(~odd, typical))
    seen = set(zip(df["product_id"], df["_date"]))
    df = df.assign(date=df["_date"], units_sold=df["_units"], price=df["_price"]).drop(columns=["_date", "_units", "_price"])
    sales = prepare_sales(df)
    if choices.get("gaps", "zero") == "stockout":
        filled = ~pd.Series([(p, d) in seen for p, d in zip(sales["product_id"], sales["date"])], index=sales.index)
        run = filled.groupby([sales["product_id"], (~filled).cumsum()]).transform("sum")
        sales["stockout"] = (filled & (run >= LONG_GAP)).astype(int)
    return sales


def messy_example() -> str:
    """A small CSV with one of each problem, to try the data check."""
    s = generate_sample_data(end_date="2026-06-30", years=1, seed=3)
    s = s[s["product_id"].isin(["P001", "P002", "P011"])].copy()
    s["date"] = s["date"].dt.strftime("%Y-%m-%d")
    s = s.drop(columns=["regular_price", "promo"])
    s = s[~((s["product_id"] == "P002") & s["date"].between("2026-03-01", "2026-03-20"))]   # a long gap
    extra = pd.concat([s.iloc[[10, 11]],                                                  # pasted twice
                       s.iloc[[40]].assign(units_sold=-3),                                # a return
                       s.iloc[[60]].assign(price=s.iloc[60]["price"] * 100),              # missing decimal point
                       s.iloc[[80]].assign(date="2062-05-01"),                            # typo in the year
                       s.iloc[[90]].assign(units_sold="n/a")])                            # unreadable
    new = pd.DataFrame({"date": pd.date_range("2026-05-15", "2026-06-30").strftime("%Y-%m-%d"),
                        "product_id": "P099", "product_name": "New Arrival Mug", "category": "Home & Kitchen",
                        "price": 14.0, "units_sold": 3})
    return pd.concat([s, extra, new], ignore_index=True).to_csv(index=False)
