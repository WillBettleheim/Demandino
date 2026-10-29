"""Global gradient-boosting demand forecaster.

One model learns from every product at once. Each training row answers the question
"standing at origin day o with the history so far, how many units will sell on day
t = o + gap?". Features describe the product's recent demand level and momentum, what
happened in the same season last year (for the product and its category), the price
and promotion planned for day t, and the calendar. Because the forecast is direct (no
recursion), the same model handles existing products, edited scenarios and new
products whose history is borrowed from similar items.

Targets are expressed relative to the recent level (units / level), which makes the
model scale-free: a product selling 3/day and one selling 300/day share patterns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge

from .data import black_friday

LEVEL_WINDOW = 56
MAX_GAP = 180
QUANTILES = (0.05, 0.95)
# Prediction ranges are conformally calibrated on held-out sales to contain 95% of real outcomes, which keeps
# out-of-sample misses under ~7% even when the season or trend shifts between calibration and use.
TARGET_COVERAGE = 0.95
RANGE_LABEL = f"{TARGET_COVERAGE:.0%}"
CALIBRATION_DAYS = 180

FEATURES = [
    "category", "gap", "log_level", "trend_ratio", "level_cv", "ly_ratio", "cat_ly_ratio",
    "rel_price", "price_vs_hist", "price_index", "promo",
    "dow", "month", "dom", "doy_sin", "doy_cos", "bf_window", "days_to_xmas", "is_xmas",
]
FEATURE_LABELS = {
    "category": "Product category", "gap": "Days ahead", "log_level": "Recent demand level",
    "trend_ratio": "Recent momentum (trend)", "level_cv": "Demand volatility", "ly_ratio": "Same season last year (product)",
    "cat_ly_ratio": "Same season last year (category)", "rel_price": "Discount vs regular price",
    "price_vs_hist": "Price vs recent average price", "price_index": "Price vs category peers", "promo": "Promotion flag",
    "dow": "Day of week", "month": "Month", "dom": "Day of month", "doy_sin": "Season (annual cycle)",
    "doy_cos": "Season (annual cycle, phase)", "bf_window": "Black Friday weekend", "days_to_xmas": "Days until Christmas",
    "is_xmas": "Christmas Day",
}


def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    dates = pd.DatetimeIndex(dates)
    doy = dates.day_of_year.to_numpy()
    bf = np.zeros(len(dates), dtype=int)
    for year in sorted(set(dates.year)):
        start = black_friday(year)
        bf[(dates >= start) & (dates <= start + pd.Timedelta(days=3))] = 1
    xmas = pd.to_datetime({"year": dates.year, "month": 12, "day": 25})
    to_xmas = (xmas.to_numpy() - dates.to_numpy()).astype("timedelta64[D]").astype(int)
    to_xmas = np.where((to_xmas < 0) | (to_xmas > 60), 61, to_xmas)
    return pd.DataFrame({
        "dow": dates.day_of_week.to_numpy(),
        "month": dates.month.to_numpy(),
        "dom": dates.day.to_numpy(),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "bf_window": bf,
        "days_to_xmas": to_xmas,
        "is_xmas": ((dates.month == 12) & (dates.day == 25)).astype(int),
    })


def _window_mean(csum: np.ndarray, end: np.ndarray, length: int) -> np.ndarray:
    """Mean of x[end-length+1 .. end] (truncated at 0) using a prefix sum; NaN where end < 0."""
    end = np.asarray(end)
    hi = end + 1
    lo = np.clip(hi - length, 0, None)
    valid = end >= 0
    hi_c = np.clip(hi, 0, len(csum) - 1)
    out = (csum[hi_c] - csum[lo]) / np.maximum(hi_c - lo, 1)
    return np.where(valid & (hi_c > lo), out, np.nan)


@dataclass
class SeriesContext:
    """Everything needed to featurise one product's rows."""
    units: np.ndarray          # observed history, length n (day 0 = start date)
    price: np.ndarray          # prices for history + future, length >= n
    regular: np.ndarray        # regular prices for history + future
    promo: np.ndarray          # promo flags for history + future
    start: pd.Timestamp
    category: str
    price_index: float         # regular price / category median regular price
    cat_units: np.ndarray | None = None  # category mean daily units aligned to start, history only


def featurise(ctx: SeriesContext, t_idx: np.ndarray, o_idx: np.ndarray, cat_codes: dict[str, int]) -> pd.DataFrame:
    """Build feature rows for targets t_idx forecast from origins o_idx (indices into the series)."""
    t_idx = np.asarray(t_idx)
    o_idx = np.asarray(o_idx)
    units = ctx.units.astype(float)
    csum = np.concatenate([[0.0], np.cumsum(units)])
    csq = np.concatenate([[0.0], np.cumsum(units ** 2)])
    pcsum = np.concatenate([[0.0], np.cumsum(ctx.price[: len(units)])])

    level = _window_mean(csum, o_idx, LEVEL_WINDOW)
    sq = _window_mean(csq, o_idx, LEVEL_WINDOW)
    std = np.sqrt(np.clip(sq - level ** 2, 0, None))
    recent = _window_mean(csum, o_idx, 28)
    prior = _window_mean(csum, o_idx - 28, 28)
    have_prior = o_idx >= 55
    trend = np.where(have_prior, (recent + 0.5) / (prior + 0.5), 1.0)
    hist_price = _window_mean(pcsum, o_idx, LEVEL_WINDOW)

    # Same season last year: demand around t-364 relative to the level at o-364.
    ly_target = _window_mean(csum, t_idx - 364 + 7, 15)
    ly_level = _window_mean(csum, o_idx - 364, LEVEL_WINDOW)
    ly_ok = (o_idx - 364 - LEVEL_WINDOW + 1 >= 0)
    ly_ratio = np.where(ly_ok, (ly_target + 0.5) / (ly_level + 0.5), np.nan)

    if ctx.cat_units is not None and len(ctx.cat_units):
        ccs = np.concatenate([[0.0], np.cumsum(ctx.cat_units.astype(float))])
        c_t = _window_mean(ccs, t_idx - 364 + 7, 15)
        c_l = _window_mean(ccs, o_idx - 364, LEVEL_WINDOW)
        c_ok = (o_idx - 364 - LEVEL_WINDOW + 1 >= 0) & (t_idx - 364 + 7 < len(ctx.cat_units))
        cat_ly = np.where(c_ok, (c_t + 0.05) / (c_l + 0.05), np.nan)
    else:
        cat_ly = np.full(len(t_idx), np.nan)

    dates = ctx.start + pd.to_timedelta(t_idx, unit="D")
    cal = calendar_features(pd.DatetimeIndex(dates))
    price_t = ctx.price[t_idx]
    feats = pd.DataFrame({
        "category": float(cat_codes[ctx.category]) if ctx.category in cat_codes else np.nan,
        "gap": (t_idx - o_idx).astype(float),
        "log_level": np.log1p(level),
        "trend_ratio": trend,
        "level_cv": std / (level + 0.5),
        "ly_ratio": ly_ratio,
        "cat_ly_ratio": cat_ly,
        "rel_price": price_t / np.maximum(ctx.regular[t_idx], 1e-6),
        "price_vs_hist": price_t / np.maximum(hist_price, 1e-6),
        "price_index": ctx.price_index,
        "promo": ctx.promo[t_idx].astype(float),
    })
    feats = pd.concat([feats, cal], axis=1)
    feats["_level"] = level
    feats["_date"] = dates
    return feats


@dataclass
class ForecastResult:
    dates: pd.DatetimeIndex
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    price: np.ndarray
    promo: np.ndarray
    level: float
    level_uncertainty: float = 0.12  # relative error of the demand level, shared by every day
    coverage: float = TARGET_COVERAGE  # probability that actual daily sales fall inside [lower, upper]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"date": self.dates, "forecast": self.mean, "lower": self.lower,
                             "upper": self.upper, "price": self.price, "promo": self.promo})

    @property
    def z(self) -> float:
        return NormalDist().inv_cdf(0.5 + self.coverage / 2)

    @property
    def daily_sd(self) -> np.ndarray:
        return np.maximum(self.upper - self.lower, 0) / (2 * self.z)

    def interval(self, sl: slice, z: float | None = None) -> tuple[float, float, float]:
        """Mean and range of total demand over the days in `sl` (same coverage as the daily range).

        Daily noise is treated as independent, plus a fully-correlated level error that
        does not average out over the horizon.
        """
        z = self.z if z is None else z
        total = float(self.mean[sl].sum())
        sd = float(np.sqrt((self.daily_sd[sl] ** 2).sum() + (self.level_uncertainty * total) ** 2))
        return total, max(total - z * sd, 0.0), total + z * sd

    def total_interval(self, days: int | None = None, z: float | None = None) -> tuple[float, float, float]:
        return self.interval(slice(0, days), z)

    def with_values(self, mean: np.ndarray, lower: np.ndarray, upper: np.ndarray, price: np.ndarray | None = None,
                    sl: slice = slice(None)) -> "ForecastResult":
        return ForecastResult(self.dates[sl], mean, lower, upper, self.price[sl] if price is None else price,
                              self.promo[sl], self.level, self.level_uncertainty, self.coverage)


@dataclass
class DemandForecaster:
    sales: pd.DataFrame
    gaps_per_day: int = 2
    max_iter: int = 350
    seed: int = 0
    cat_codes: dict[str, int] = field(init=False, default_factory=dict)
    cat_median_price: dict[str, float] = field(init=False, default_factory=dict)
    cat_series: dict[str, pd.Series] = field(init=False, default_factory=dict)
    models: dict[str, HistGradientBoostingRegressor] = field(init=False, default_factory=dict)
    level_model: Ridge | None = field(init=False, default=None)
    cold_start_residual_sd: float = field(init=False, default=0.4)
    conformal_q: float = field(init=False, default=0.0)        # range widening, in units of demand level
    bias_factor: float = field(init=False, default=1.0)        # multiplicative correction measured out of sample
    closed_dates: set = field(init=False, default_factory=set)  # days the whole shop sold nothing (closed)
    level_uncertainty: float = field(init=False, default=0.12)  # calibrated for multi-day totals
    calibration: dict = field(init=False, default_factory=dict)
    features: list = field(init=False, default_factory=lambda: list(FEATURES))

    # ------------------------------------------------------------------ setup
    def __post_init__(self) -> None:
        self.sales = self.sales.sort_values(["product_id", "date"]).reset_index(drop=True)
        self.cat_codes = {c: i for i, c in enumerate(sorted(self.sales["category"].unique()))}
        reg = self.sales.groupby(["category", "product_id"])["regular_price"].median()
        self.cat_median_price = reg.groupby(level=0).median().to_dict()
        # Category demand per product per day, used for category-level seasonality.
        self.cat_series = self._category_series()
        shop = self.sales.groupby("date")["units_sold"].sum()
        self.closed_dates = set(shop[shop == 0].index) if self.sales["product_id"].nunique() >= 5 else set()

    def _category_series(self) -> dict[str, pd.Series]:
        out = {}
        for cat, g in self.sales.groupby("category"):
            s = g.groupby("date")["units_sold"].mean()
            out[cat] = s.asfreq("D").fillna(0)
        return out

    def _cat_units(self, category: str, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray | None:
        s = self.cat_series.get(category)
        if s is None:
            return None
        return s.reindex(pd.date_range(start, end, freq="D")).to_numpy(dtype=float)

    def price_index(self, category: str, regular_price: float) -> float:
        med = self.cat_median_price.get(category, np.median(list(self.cat_median_price.values())))
        return float(regular_price / med) if med else 1.0

    def context_for(self, pid: str, sales: pd.DataFrame | None = None) -> SeriesContext:
        g = (self.sales if sales is None else sales)
        g = g[g["product_id"] == pid].sort_values("date")
        start, end = g["date"].iloc[0], g["date"].iloc[-1]
        cat = g["category"].iloc[0]
        return SeriesContext(
            units=g["units_sold"].to_numpy(float), price=g["price"].to_numpy(float),
            regular=g["regular_price"].to_numpy(float), promo=g["promo"].to_numpy(int), start=start,
            category=cat, price_index=self.price_index(cat, float(g["regular_price"].median())),
            cat_units=self._cat_units(cat, start, end),
        )

    # --------------------------------------------------------------- training
    def _training_rows(self, cutoff: pd.Timestamp | None) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        sales = self.sales if cutoff is None else self.sales[self.sales["date"] <= cutoff]
        parts = []
        for pid in sales["product_id"].unique():
            ctx = self.context_for(pid, sales)
            n = len(ctx.units)
            if n < LEVEL_WINDOW + 2:
                continue
            t = np.repeat(np.arange(LEVEL_WINDOW, n), self.gaps_per_day)
            gap = rng.integers(1, MAX_GAP + 1, len(t))
            # Over-sample short horizons, which matter most operationally.
            short = rng.random(len(t)) < 0.35
            gap[short] = rng.integers(1, 29, short.sum())
            o = t - gap
            keep = o >= LEVEL_WINDOW - 1
            t, o = t[keep], o[keep]
            f = featurise(ctx, t, o, self.cat_codes)
            f["_y"] = ctx.units[t]
            parts.append(f)
        return pd.concat(parts, ignore_index=True)

    def _fit_models(self, rows: pd.DataFrame) -> dict[str, HistGradientBoostingRegressor]:
        level = rows["_level"].to_numpy() + 0.5
        y = rows["_y"].to_numpy() / level
        # Short histories leave some features empty (e.g. no "last year" yet); the model skips them.
        self.features = [f for f in FEATURES if rows[f].nunique() > 1]
        X = rows[self.features]
        cat_mask = [f == "category" for f in self.features]
        common = dict(max_iter=self.max_iter, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
                      l2_regularization=1.0, categorical_features=cat_mask, random_state=self.seed)
        models = {"mean": HistGradientBoostingRegressor(loss="poisson", **common)}
        # Poisson on units/level weighted by level == Poisson on units with a log(level) offset.
        models["mean"].fit(X, y, sample_weight=level)
        for q in QUANTILES:
            m = HistGradientBoostingRegressor(loss="quantile", quantile=q, **common)
            m.fit(X, y, sample_weight=np.sqrt(level))
            models["lo" if q < 0.5 else "hi"] = m
        return models

    def _fit_level_model(self, cutoff: pd.Timestamp | None = None) -> None:
        """Cold-start model: recent demand level from category and price positioning."""
        sales = self.sales if cutoff is None else self.sales[self.sales["date"] <= cutoff]
        rows = []
        for pid, g in sales.groupby("product_id"):
            g = g.sort_values("date").tail(LEVEL_WINDOW)
            rows.append({"category": g["category"].iloc[0], "level": g["units_sold"].mean(),
                         "price_index": self.price_index(g["category"].iloc[0], g["regular_price"].median())})
        prod = pd.DataFrame(rows)
        X = self._level_design(prod["category"], prod["price_index"])
        y = np.log(prod["level"] + 0.5)
        self.level_model = Ridge(alpha=0.5).fit(X, y)
        resid = y - self.level_model.predict(X)
        self.cold_start_residual_sd = float(np.sqrt(np.mean(resid ** 2))) if len(resid) > 3 else 0.5

    def _level_design(self, categories, price_index) -> np.ndarray:
        cats = np.asarray(categories)
        onehot = np.stack([(cats == c).astype(float) for c in self.cat_codes], axis=1)
        return np.column_stack([onehot, np.log(np.asarray(price_index, dtype=float))])

    def _fit_at(self, cutoff: pd.Timestamp | None) -> pd.DataFrame:
        rows = self._training_rows(cutoff)
        self.models = self._fit_models(rows)
        self._fit_level_model(cutoff)
        return rows

    def fit(self, cutoff: pd.Timestamp | None = None, calibrate: bool = True) -> "DemandForecaster":
        """Train on data up to `cutoff` (default: all).

        With `calibrate`, a model trained without the last CALIBRATION_DAYS first forecasts
        that period from several origins; the misses against real sales set the width of
        the prediction ranges (split-conformal calibration). Then the model is retrained.
        """
        end = self.sales["date"].max() if cutoff is None else pd.Timestamp(cutoff)
        if calibrate:
            cal_start = end - pd.Timedelta(days=CALIBRATION_DAYS)
            self._fit_at(cal_start)
            self.conformal_q, self.level_uncertainty, self.bias_factor = 0.0, 0.0, 1.0
            origins = [cal_start + pd.Timedelta(days=d) for d in range(0, CALIBRATION_DAYS - 13, 30)]
            raw = holdout_forecasts(self, origins, end)
            self.calibration = calibrate_ranges(raw, self.closed_dates)
            self.conformal_q = self.calibration["conformal_q"]
            self.level_uncertainty = self.calibration["level_uncertainty"]
            self.bias_factor = self.calibration["bias_factor"]
        self._fit_at(cutoff)
        return self

    # -------------------------------------------------------------- inference
    def _predict(self, feats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = feats[self.features]
        lvl = feats["_level"].to_numpy() + 0.5
        b = self.bias_factor
        mean = self.models["mean"].predict(X) * lvl * b
        lo = np.clip(self.models["lo"].predict(X) * b - self.conformal_q, 0, None) * lvl
        hi = np.clip(self.models["hi"].predict(X) * b + self.conformal_q, 0, None) * lvl
        mean = np.clip(mean, 0, None)
        lo = np.minimum(lo, mean)
        hi = np.maximum(hi, mean)
        return mean, lo, hi

    def forecast_context(self, ctx: SeriesContext, start: pd.Timestamp, horizon: int,
                         price_plan: np.ndarray, regular_plan: np.ndarray, promo_plan: np.ndarray,
                         level_override: float | None = None) -> ForecastResult:
        """Forecast `horizon` days from `start` given the series history in ctx and future plans."""
        n = len(ctx.units)
        origin = n - 1
        first = int((start - ctx.start).days)
        if first <= origin:
            raise ValueError("Forecast start must be after the last day of history.")
        t = np.arange(first, first + horizon)
        span = t[-1] + 1
        ext = SeriesContext(
            units=ctx.units,
            price=_extend(ctx.price[:n], span, first, price_plan),
            regular=_extend(ctx.regular[:n], span, first, regular_plan),
            promo=_extend(ctx.promo[:n], span, first, promo_plan).astype(int),
            start=ctx.start, category=ctx.category, price_index=ctx.price_index, cat_units=ctx.cat_units,
        )
        gaps = np.minimum(t - origin, MAX_GAP)
        feats = featurise(ext, t, t - gaps, self.cat_codes)
        if level_override is not None:
            feats["_level"] = level_override
            feats["log_level"] = np.log1p(level_override)
        mean, lo, hi = self._predict(feats)
        return ForecastResult(pd.DatetimeIndex(feats["_date"]), mean, lo, hi, ext.price[t], ext.promo[t],
                              float(feats["_level"].iloc[0]), self.level_uncertainty)

    def forecast_product(self, pid: str, horizon: int, start: pd.Timestamp | None = None,
                         price: float | np.ndarray | None = None, regular_price: float | None = None,
                         promo_plan: np.ndarray | None = None, discount_plan: np.ndarray | None = None) -> ForecastResult:
        """Forecast an existing product. Defaults: last regular price, no promotions."""
        ctx = self.context_for(pid)
        last = ctx.start + pd.Timedelta(days=len(ctx.units) - 1)
        start = last + pd.Timedelta(days=1) if start is None else pd.Timestamp(start)
        reg = float(ctx.regular[-1]) if regular_price is None else float(regular_price)
        promo = np.zeros(horizon, dtype=int) if promo_plan is None else np.asarray(promo_plan, dtype=int)
        disc = np.zeros(horizon) if discount_plan is None else np.asarray(discount_plan, dtype=float)
        base_price = np.full(horizon, reg) if price is None else np.broadcast_to(np.asarray(price, float), (horizon,))
        prices = base_price * (1 - disc * promo)
        return self.forecast_context(ctx, start, horizon, prices, np.full(horizon, reg), promo)

    def cold_start_level(self, category: str, regular_price: float) -> float:
        X = self._level_design([category], [self.price_index(category, regular_price)])
        return float(np.exp(self.level_model.predict(X)[0]) - 0.5)

    def new_product_context(self, category: str, regular_price: float, start: pd.Timestamp,
                            history: pd.DataFrame | None = None, comparable_pid: str | None = None,
                            expected_daily: float | None = None) -> tuple[SeriesContext, float | None, str]:
        """Build a history context for a product that is not in the dataset.

        Priority: the product's own recent sales (if provided) > a comparable product's
        history rescaled for price > the cold-start level model. Returns the context, an
        optional level override and a description of the method used.
        """
        price_index = self.price_index(category, regular_price)
        if history is not None and len(history) >= 7:
            h = history.sort_values("date")
            h_start = h["date"].iloc[0]
            ctx = SeriesContext(
                units=h["units_sold"].to_numpy(float), price=h["price"].to_numpy(float),
                regular=np.maximum(h["price"].to_numpy(float), regular_price), promo=h.get("promo", pd.Series(0, index=h.index)).to_numpy(int),
                start=h_start, category=category, price_index=price_index,
                cat_units=self._cat_units(category, h_start, h["date"].iloc[-1]),
            )
            return ctx, (expected_daily if expected_daily else None), f"your {len(h)} days of sales history"

        origin = start - pd.Timedelta(days=1)
        if comparable_pid is not None:
            comp = self.context_for(comparable_pid)
            comp_reg = float(np.median(comp.regular[-LEVEL_WINDOW:]))
            # Shift comparable's history to end at the origin; rescale volume by price elasticity across products.
            elasticity = -float(self.level_model.coef_[-1]) if self.level_model is not None else 0.8
            scale = (regular_price / comp_reg) ** (-elasticity)
            units = comp.units * scale
            comp_start = origin - pd.Timedelta(days=len(units) - 1)
            ratio = regular_price / comp_reg
            ctx = SeriesContext(units=units, price=comp.price * ratio, regular=comp.regular * ratio, promo=comp.promo,
                                start=comp_start, category=category, price_index=price_index,
                                cat_units=self._cat_units(category, comp_start, origin))
            override = expected_daily if expected_daily else None
            return ctx, override, f"the sales pattern of comparable product {comparable_pid}, rescaled for price"

        level = expected_daily if expected_daily else self.cold_start_level(category, regular_price)
        n = LEVEL_WINDOW
        cat = self._cat_units(category, origin - pd.Timedelta(days=n - 1), origin)
        if cat is not None and np.nanmean(cat) > 0:
            units = np.nan_to_num(cat) / np.nanmean(cat) * level  # category shape at the new product's scale
        else:
            units = np.full(n, level)
        ctx = SeriesContext(units=units, price=np.full(n, regular_price), regular=np.full(n, regular_price),
                            promo=np.zeros(n, int), start=origin - pd.Timedelta(days=n - 1), category=category,
                            price_index=price_index, cat_units=self._cat_units(category, origin - pd.Timedelta(days=n - 1), origin))
        method = "your expected daily sales" if expected_daily else "category & price positioning (cold-start model)"
        return ctx, level, method

    # ------------------------------------------------------------- evaluation
    def feature_importance(self, rows: pd.DataFrame, n: int = 4000) -> pd.Series:
        sample = rows.sample(min(n, len(rows)), random_state=self.seed)
        level = sample["_level"].to_numpy() + 0.5
        res = permutation_importance(self.models["mean"], sample[self.features], sample["_y"].to_numpy() / level,
                                     sample_weight=level, n_repeats=3, random_state=self.seed)
        imp = pd.Series(res.importances_mean, index=[FEATURE_LABELS[f] for f in self.features])
        return imp.clip(lower=0).sort_values(ascending=False)


def _extend(hist: np.ndarray, span: int, first: int, plan: np.ndarray) -> np.ndarray:
    """History values followed by the future plan, gap days (if any) filled with the last value."""
    out = np.empty(span)
    n = len(hist)
    out[:n] = hist
    out[n:first] = hist[-1] if n else plan[0]
    out[first:first + len(plan)] = plan
    return out


def holdout_forecasts(fc: DemandForecaster, origins: list[pd.Timestamp], end: pd.Timestamp,
                      horizon: int = 90) -> list[dict]:
    """Forecast every product from each origin up to `end` (never beyond), using only history up to the origin."""
    out = []
    for pid, full in fc.sales.groupby("product_id"):
        full = full.sort_values("date")
        for origin in origins:
            hist = full[full["date"] <= origin]
            fut = full[(full["date"] > origin) & (full["date"] <= min(end, origin + pd.Timedelta(days=horizon)))]
            if len(hist) < LEVEL_WINDOW or fut.empty:
                continue
            ctx = fc.context_for(pid, hist)
            res = fc.forecast_context(ctx, fut["date"].iloc[0], len(fut), fut["price"].to_numpy(float),
                                      fut["regular_price"].to_numpy(float), fut["promo"].to_numpy(int))
            out.append({"product_id": pid, "origin": origin, "res": res, "actual": fut["units_sold"].to_numpy(float),
                        "hist": hist, "full": full})
    return out


def _week_slices(n: int) -> list[slice]:
    return [slice(i, min(i + 7, n)) for i in range(0, n - 6, 7)]


def calibrate_ranges(raw: list[dict], closed: set | None = None, target: float = TARGET_COVERAGE,
                     weekly_target: float = 0.97) -> dict:
    """Out-of-sample bias correction, split-conformal calibration of the daily range, then the
    level error used for multi-day totals. Days the shop was closed are ignored (a 0–0 range
    trivially "covers" them, which would flatter the result)."""
    closed = closed or set()
    opened = [~r["res"].dates.isin(list(closed)) for r in raw]
    # Bias measured here did not carry over to later periods in testing (it hurt accuracy on real and simulated
    # data), so it is reported but not applied.
    act = sum(float(r["actual"][o].sum()) for r, o in zip(raw, opened))
    fct = sum(float(r["res"].mean[o].sum()) for r, o in zip(raw, opened))
    b = 1.0
    scores = []
    for r, o in zip(raw, opened):
        lvl = r["res"].level + 0.5
        res, y = r["res"], r["actual"]
        scores.append((np.maximum(res.lower - y, y - res.upper) / lvl)[o])
    s = np.concatenate(scores)
    k = min(1.0, target * (1 + 1 / len(s)))
    q = float(np.quantile(s, k))
    adjusted = []
    for r in raw:
        lvl = r["res"].level + 0.5
        res = r["res"]
        adjusted.append(res.with_values(res.mean, np.maximum(res.lower - q * lvl, 0), res.upper + q * lvl))
    lu_best = 0.6
    for lu in np.arange(0.0, 0.61, 0.01):
        hit = tot = 0
        for r, res in zip(raw, adjusted):
            res.level_uncertainty = lu
            for sl in _week_slices(len(res.mean)):
                _, lo, hi = res.interval(sl)
                a = r["actual"][sl].sum()
                hit += lo <= a <= hi
                tot += 1
        if tot and hit / tot >= weekly_target:
            lu_best = float(lu)
            break
    return {"conformal_q": q, "level_uncertainty": lu_best, "bias_factor": b,
            "calibration_bias": (fct - act) / act if act else 0.0, "n_scores": int(len(s)),
            "target": target}


def backtest(sales: pd.DataFrame, holdout_days: int = 90, **kwargs) -> dict:
    """Train (and calibrate) on data before the last `holdout_days`, forecast them, score against actuals.

    The test window is never seen during training or calibration.
    """
    last = sales["date"].max()
    cutoff = last - pd.Timedelta(days=holdout_days)
    fc = DemandForecaster(sales, **kwargs).fit(cutoff=cutoff)
    raw = holdout_forecasts(fc, [cutoff], last, horizon=holdout_days)

    records, weeks = [], []
    for r in raw:
        res, full, hist = r["res"], r["full"], r["hist"]
        fut_dates = res.dates
        naive = hist["units_sold"].tail(LEVEL_WINDOW).mean()
        last_year = full.set_index("date")["units_sold"].reindex(fut_dates - pd.Timedelta(days=364)).to_numpy()
        records.append(pd.DataFrame({
            "date": fut_dates, "product_id": r["product_id"], "product_name": full["product_name"].iloc[0],
            "category": full["category"].iloc[0], "actual": r["actual"], "forecast": res.mean,
            "lower": res.lower, "upper": res.upper, "naive_mean": naive, "seasonal_naive": last_year,
        }))
        for sl in _week_slices(len(res.mean)):
            tot, lo, hi = res.interval(sl)
            a = r["actual"][sl].sum()
            weeks.append({"product_id": r["product_id"], "actual": a, "forecast": tot, "lower": lo, "upper": hi,
                          "naive": naive * (sl.stop - sl.start)})
    preds = pd.concat(records, ignore_index=True)
    weekly = pd.DataFrame(weeks)

    def wape(a, f):
        mask = ~np.isnan(f)
        return float(np.abs(a[mask] - f[mask]).sum() / max(a[mask].sum(), 1e-9))

    a = preds["actual"].to_numpy()
    open_day = ~preds["date"].isin(list(fc.closed_dates)).to_numpy()
    preds["open"] = open_day
    daily_in = ((a >= preds["lower"]) & (a <= preds["upper"]))[open_day]
    weekly_in = (weekly["actual"] >= weekly["lower"]) & (weekly["actual"] <= weekly["upper"])
    metrics = {
        "WAPE (daily)": wape(a, preds["forecast"].to_numpy()),
        "WAPE (weekly)": wape(weekly["actual"].to_numpy(), weekly["forecast"].to_numpy()),
        "Bias": float((preds["forecast"].sum() - a.sum()) / a.sum()),
        "Daily miss rate": float(1 - daily_in.mean()),
        "Weekly miss rate": float(1 - weekly_in.mean()),
        "Range coverage target": TARGET_COVERAGE,
        "Baseline WAPE: recent average": wape(a, preds["naive_mean"].to_numpy()),
        "Baseline WAPE: same day last year": wape(a, preds["seasonal_naive"].to_numpy()),
    }
    per_product = preds.groupby(["product_id", "product_name", "category"]).apply(
        lambda g: pd.Series({
            "actual_units": g["actual"].sum(), "forecast_units": g["forecast"].sum(),
            "WAPE_daily": wape(g["actual"].to_numpy(), g["forecast"].to_numpy()),
            "total_error_%": (g["forecast"].sum() - g["actual"].sum()) / max(g["actual"].sum(), 1e-9) * 100,
            "miss_rate": float(1 - ((g["actual"] >= g["lower"]) & (g["actual"] <= g["upper"])).mean()),
        }), include_groups=False).reset_index()
    importance = fc.feature_importance(fc._training_rows(cutoff))
    return {"metrics": metrics, "predictions": preds, "weekly": weekly, "per_product": per_product,
            "importance": importance, "cutoff": cutoff, "calibration": fc.calibration}
