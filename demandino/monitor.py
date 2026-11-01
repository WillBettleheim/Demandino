"""Drift alerts: products whose recent sales keep landing outside what the model expected.

A copy of the model is trained only on sales up to `weeks` weeks ago, with the main model's calibrated range
widths, and forecasts those weeks. Real weekly sales are compared with the weekly ranges. A product is
flagged when sales sat outside the range on the same side for at least `min_weeks` weeks, or when the whole
period's total did. Those products have changed (a trend, a new competitor, a listing problem…) and their
forecasts and stock levels deserve a look.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import DemandForecaster, _week_slices, holdout_forecasts

DRIFT_WEEKS = 4
MIN_WEEKS = 2


def holdout_model(fc: DemandForecaster, weeks: int = DRIFT_WEEKS) -> tuple[DemandForecaster, pd.Timestamp]:
    """The main model's setup, retrained without the last `weeks` weeks (ranges copied from the main model)."""
    cutoff = fc.sales["date"].max() - pd.Timedelta(days=7 * weeks)
    sales = fc.sales.drop(columns=["stockout", "bulk", "demand"], errors="ignore")
    old = DemandForecaster(sales, country=fc.country, drivers=fc.drivers, max_iter=fc.max_iter,
                           stockout_correction=fc.stockout_correction, bulk_correction=fc.bulk_correction,
                           bulk_keep=fc.bulk_keep, bulk_add=fc.bulk_add).fit(cutoff=cutoff, calibrate=False)
    old.conformal_q, old.level_uncertainty = fc.conformal_q, fc.level_uncertainty
    return old, cutoff


def drift_report(old: DemandForecaster, cutoff: pd.Timestamp, weeks: int = DRIFT_WEEKS,
                 min_weeks: int = MIN_WEEKS) -> pd.DataFrame:
    """One row per product: weekly sales vs expected, how many weeks fell outside the range, and a flag."""
    last = old.sales["date"].max()
    rows = []
    for r in holdout_forecasts(old, [cutoff], last, horizon=7 * weeks):
        res, actual = r["res"], r["actual"]
        above = below = 0
        weekly = []
        for sl in _week_slices(len(actual)):
            exp, lo, hi = res.interval(sl)
            a = float(actual[sl].sum())
            above += a > hi
            below += a < lo
            weekly.append(a / exp - 1 if exp > 0 else 0.0)
        total, lo, hi = res.total_interval()
        act = float(actual.sum())
        direction = "Above" if above >= below else "Below"
        outside = max(above, below)
        flagged = outside >= min_weeks or act > hi or act < lo
        rows.append({"product_id": r["product_id"], "product_name": r["full"]["product_name"].iloc[0],
                     "category": r["full"]["category"].iloc[0], "expected": total, "low": lo, "high": hi,
                     "actual": act, "change": act / total - 1 if total > 0 else np.nan,
                     "weeks_outside": int(outside), "weeks": len(weekly), "weekly_change": weekly,
                     "direction": direction if flagged else "On track", "flagged": bool(flagged)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["severity"] = np.where(out["flagged"], out["change"].abs(), 0.0)
    return out.sort_values(["flagged", "severity"], ascending=False).reset_index(drop=True)


def advice(row) -> str:
    """What to do about a flagged product, in one sentence."""
    if row["direction"] == "Above":
        return ("Selling faster than expected: check stock now and consider ordering earlier. "
                "The current forecast already includes these recent weeks.")
    if row["direction"] == "Below":
        return ("Selling slower than expected: check the listing, price and competitors, and hold back on new "
                "orders until it recovers.")
    return "On track."


def drift_lines(report: pd.DataFrame, limit: int = 10) -> list[str]:
    """Plain-text alerts for the digest."""
    if report.empty:
        return []
    out = []
    for r in report[report["flagged"]].head(limit).itertuples():
        word = "ABOVE" if r.direction == "Above" else "BELOW"
        out.append(f"SALES {word} FORECAST · {r.product_name}: {r.actual:,.0f} sold in the last {r.weeks} weeks vs "
                   f"{r.expected:,.0f} expected ({r.change:+.0%}); outside the range {r.weeks_outside} of "
                   f"{r.weeks} weeks.")
    return out
