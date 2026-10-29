"""One forecast scenario: user inputs in, a complete analysis out (no UI code here)."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .analysis import build_insights, inventory_plan, price_sensitivity, product_profile
from .data import black_friday
from .model import LEVEL_WINDOW, DemandForecaster, ForecastResult
from .security import read_csv_upload

PROMO_KINDS = ["None", "Recurring", "Specific dates", "Black Friday weekend"]
BASES = {"auto": "Category & price", "similar": "Similar product", "history": "My sales history"}


@dataclass
class ForecastInputs:
    new: bool
    pid: str | None
    name: str
    category: str
    price: float
    cost: float
    horizon: int
    start: pd.Timestamp
    basis: str = "auto"
    comparable: str | None = None
    expected: float | None = None
    ramp_days: int = 0
    history: pd.DataFrame | None = None
    promo_kind: str = "None"
    discount: float = 0.2
    every: int = 4
    plen: int = 5
    promo_dates: tuple | None = None
    events: tuple = ()  # (name, start date, end date, demand change %)
    stock: float = 500
    lead: int = 14
    review: int = 14
    service: float = 0.95


@dataclass
class ScenarioResult:
    inputs: ForecastInputs
    res: ForecastResult
    base: ForecastResult              # same scenario without promotions
    history: pd.DataFrame | None
    method: str
    elasticity: float
    profile: dict | None
    total: float
    low: float
    high: float
    revenue: float
    profit: float
    recent_daily: float | None
    inventory: dict
    promo_effect: dict | None
    sensitivity: pd.DataFrame
    drivers: list[tuple[str, float]]
    last_year: np.ndarray | None
    insights: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> str:
        width = (self.high - self.low) / max(self.total, 1e-9)
        return "High" if width < 0.6 else "Medium" if width < 1.1 else "Low"  # widths of a 95% range

    def frame(self) -> pd.DataFrame:
        f = self.res.frame()
        f["revenue"] = f["forecast"] * f["price"]
        if self.last_year is not None:
            f["last_year_actual"] = self.last_year
        return f


def promo_arrays(kind: str, horizon: int, start: pd.Timestamp, discount: float, every_weeks: int = 4,
                 length: int = 5, first_offset: int = 7, date_range: tuple | None = None) -> tuple[np.ndarray, np.ndarray]:
    promo = np.zeros(horizon, dtype=int)
    days = pd.date_range(start, periods=horizon, freq="D")
    if kind == "Recurring":
        for s in range(first_offset, horizon, every_weeks * 7):
            promo[s:s + length] = 1
    elif kind == "Specific dates" and date_range and len(date_range) == 2:
        promo[(days >= pd.Timestamp(date_range[0])) & (days <= pd.Timestamp(date_range[1]))] = 1
    elif kind == "Black Friday weekend":
        for y in set(days.year):
            bf = black_friday(y)
            promo[(days >= bf) & (days <= bf + pd.Timedelta(days=3))] = 1
    return promo, np.full(horizon, discount)


def event_multiplier(dates: pd.DatetimeIndex, events: tuple) -> np.ndarray:
    """Multiplicative demand adjustments for known events (campaigns, closures, competitor launches…)."""
    m = np.ones(len(dates))
    for _, s, e, change in events:
        if s is None or e is None or change is None or pd.isna(change):
            continue
        mask = (dates >= pd.Timestamp(s)) & (dates <= pd.Timestamp(e))
        m[mask] *= max(0.0, 1 + float(change) / 100)
    return m


def _scale(r: ForecastResult, k: np.ndarray | float, price: np.ndarray | None = None) -> ForecastResult:
    return r.with_values(r.mean * k, r.lower * k, r.upper * k, price)


def compute(fc: DemandForecaster, sales: pd.DataFrame, elasticities: tuple[dict, dict], level_unc: float,
            inp: ForecastInputs, detail: bool = True) -> ScenarioResult:
    """Full analysis of one scenario. With detail=False only the forecast, money and inventory are
    computed (no price sweep, drivers or findings) — fast enough to search many scenarios."""
    start, horizon, price = inp.start, inp.horizon, inp.price
    prod_e, cat_e = elasticities
    promo, disc = promo_arrays(inp.promo_kind, horizon, start, inp.discount, inp.every, inp.plen,
                               date_range=inp.promo_dates)
    profile, level_override, history = None, None, None

    if not inp.new:
        ctx = fc.context_for(inp.pid)
        history = sales[sales["product_id"] == inp.pid].sort_values("date")
        method, unc = "this product's own sales history", level_unc
        profile = product_profile(history)
        elasticity, ref_price = prod_e[inp.pid], float(ctx.regular[-1])
    else:
        use_hist = inp.basis == "history" and inp.history is not None
        ctx, level_override, method = fc.new_product_context(
            inp.category, price, start, history=inp.history if use_hist else None,
            comparable_pid=inp.comparable if inp.basis == "similar" else None, expected_daily=inp.expected)
        history = inp.history if use_hist else None
        if history is not None and len(history) >= LEVEL_WINDOW:
            unc = level_unc
        elif inp.expected:
            unc = 0.25  # user-supplied level: assume it is roughly right
        else:
            unc = max(fc.cold_start_residual_sd, level_unc)
        elasticity = cat_e.get(inp.category, float(np.median(list(cat_e.values()))))
        ref_price = price  # the demand level was estimated at this price
        if history is not None:
            ref_price = float(np.median(ctx.regular[-LEVEL_WINDOW:]))
            if len(history) >= 60:
                profile = product_profile(history.assign(regular_price=np.maximum(history["price"], price)))
    if profile is not None and not np.isnan(profile["elasticity"]):
        profile = {**profile, "elasticity": elasticity}

    dates = pd.date_range(start, periods=horizon, freq="D")
    events = event_multiplier(dates, inp.events)
    ramp = 1 - np.exp(-3 * (np.arange(horizon) + 1) / inp.ramp_days) if inp.ramp_days > 0 else np.ones(horizon)

    def run(promo_: np.ndarray, disc_: np.ndarray, reg: float = price, adjust: bool = True,
            with_events: bool = True) -> ForecastResult:
        """ML forecast at the reference price (seasonality, promos, level); the regular-price change from the
        reference is applied with the product's price elasticity, then events and launch ramp-up."""
        r = fc.forecast_context(ctx, start, horizon, ref_price * (1 - disc_ * promo_), np.full(horizon, ref_price),
                                promo_, level_override=level_override)
        r.level_uncertainty = unc
        k = (reg / ref_price) ** elasticity
        if adjust:
            k = k * ramp * (events if with_events else 1)
        return _scale(r, k, price=reg * (1 - disc_ * promo_))

    no_promo = np.zeros(horizon, int)
    res = run(promo, disc)
    base = run(no_promo, disc) if promo.any() else res
    total, low, high = res.total_interval()
    revenue = float((res.mean * res.price).sum())
    profit = revenue - inp.cost * total
    inv = inventory_plan(res, inp.stock, inp.lead, inp.review, inp.service)

    promo_effect = None
    if promo.any():
        base_rev = float((base.mean * base.price).sum())
        promo_effect = dict(promo_days=int(promo.sum()), extra_units=float((res.mean - base.mean).sum()),
                            lift=float(res.mean[promo == 1].sum() / max(base.mean[promo == 1].sum(), 1e-9) - 1),
                            extra_profit=profit - (base_rev - inp.cost * base.mean.sum()))
    sens = price_sensitivity(lambda reg: run(promo, disc, reg), price, inp.cost) if detail else pd.DataFrame()

    # Waterfall: how we get from the recent run-rate to the final forecast.
    recent = float(np.mean(ctx.units[-LEVEL_WINDOW:])) if len(ctx.units) else 0.0
    run_rate = (level_override if level_override is not None else recent) * horizon
    drivers, c, d = [], 0.0, 0.0
    if detail:
        a = run(no_promo, disc, ref_price, adjust=False).mean.sum()
        b = run(no_promo, disc, price, adjust=False).mean.sum()
        c = run(promo, disc, price, adjust=False).mean.sum()
        d = (run(promo, disc, price, adjust=False).mean * events).sum()
        drivers = [("Recent run-rate", run_rate), ("Season, trend & calendar", a - run_rate), ("Price change", b - a),
                   ("Promotions", c - b), ("Events", d - c), ("Launch ramp-up", total - d)]

    last_year = None
    if history is not None and len(history):
        ly = history.set_index("date")["units_sold"].reindex(dates - pd.Timedelta(days=364)).to_numpy(dtype=float)
        last_year = ly if np.isfinite(ly).sum() >= horizon * 0.8 else None

    recent_daily = recent if history is not None and len(history) else None
    result = ScenarioResult(inp, res, base, history, method, elasticity, profile, total, low, high, revenue, profit,
                            recent_daily, inv, promo_effect, sens, drivers, last_year)
    if not detail:
        return result
    result.insights = build_insights(res, inp.cost, recent_daily, promo_effect, inv, sens, price, profile, method)
    extra = []
    if last_year is not None:
        ly_total = float(np.nansum(last_year))
        if ly_total > 0:
            extra.append(f"**Year over year:** the same {horizon} days last year sold {ly_total:,.0f} units; "
                         f"this forecast is {(total / ly_total - 1) * 100:+.0f}% versus that.")
    if any(abs(c) > 0 for *_, c in inp.events if c is not None and not pd.isna(c)):
        ev = d - c
        extra.append(f"**Planned events** change demand by {ev:+,.0f} units over the horizon.")
    result.insights[1:1] = extra
    return result


def read_history_csv(content: bytes, price: float) -> pd.DataFrame:
    """Parse a single product's sales history (date, units_sold[, price, promo]) into daily rows."""
    h = read_csv_upload(content, max_rows=50_000)
    h.columns = [str(c).strip().lower().replace(" ", "_") for c in h.columns]
    h = h.rename(columns={c: "units_sold" for c in ("quantity", "units", "qty", "sales") if c in h.columns})
    if "date" not in h or "units_sold" not in h:
        raise ValueError("The file needs at least `date` and `units_sold` columns.")
    h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.normalize()
    h = h.dropna(subset=["date"])
    agg = {"units_sold": "sum"}
    if "price" in h:
        agg["price"] = "mean"
    if "promo" in h:
        agg["promo"] = "max"
    h = h.groupby("date", as_index=False).agg(agg).set_index("date").asfreq("D").reset_index()
    h["units_sold"] = pd.to_numeric(h["units_sold"], errors="coerce").fillna(0).clip(lower=0)
    h["price"] = h["price"].ffill().fillna(price) if "price" in h else price
    h["promo"] = h["promo"].fillna(0).astype(int) if "promo" in h else 0
    if len(h) < 7:
        raise ValueError("Need at least 7 days of history.")
    if len(h) > 3700:
        raise ValueError("History longer than 10 years; please upload the most recent years.")
    return h


def events_tuple(df: pd.DataFrame | None) -> tuple:
    if df is None or df.empty:
        return ()
    rows = []
    for r in df.itertuples(index=False):
        s, e, c = getattr(r, "start", None), getattr(r, "end", None), getattr(r, "change", None)
        if s is None or pd.isna(s) or c is None or pd.isna(c):
            continue
        e = s if e is None or pd.isna(e) else e
        rows.append((str(getattr(r, "event", "") or "Event"), pd.Timestamp(s).date(), pd.Timestamp(e).date(), float(c)))
    return tuple(rows)


def default_events() -> pd.DataFrame:
    return pd.DataFrame({"event": pd.Series([], dtype="object"), "start": pd.Series([], dtype="object"),
                         "end": pd.Series([], dtype="object"), "change": pd.Series([], dtype="float")})

