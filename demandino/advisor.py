"""Demand growth advisor: test the levers a seller controls with the trained model and rank them.

Every number here comes from re-running the forecast with one change (price, promotion depth,
frequency, timing, availability). Ideas the model cannot measure are listed separately as
general good practice, so measured and unmeasured advice are never mixed.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .analysis import detect_anomalies, promo_spillover
from .model import DemandForecaster
from .scenario import ForecastInputs, ScenarioResult, compute

GOOD_PRACTICE = [
    ("Listing quality", "Clear photos, a benefit-led title and complete specifications raise conversion on the same "
                        "traffic. Fix the weakest listing first."),
    ("Reviews & trust", "Ask recent buyers for reviews; products with more and fresher reviews convert better."),
    ("Bundles & cross-sell", "Pair the product with a frequently co-bought item, or offer a multi-pack at a small "
                             "per-unit discount."),
    ("New channels", "Test one additional sales channel (marketplace, wholesale, social shop) with a small batch."),
    ("Seasonal content", "Plan campaigns 3–4 weeks before the product's peak month so stock and ads are ready."),
]


def _fast(fc: DemandForecaster, sales: pd.DataFrame, el: tuple, unc: float, inp: ForecastInputs) -> ScenarioResult:
    return compute(fc, sales, el, unc, inp, detail=False)


def _row(lever: str, action: str, r: ScenarioResult, base: ScenarioResult, note: str = "") -> dict:
    return {"lever": lever, "action": action, "units": r.total, "revenue": r.revenue, "profit": r.profit,
            "units_delta": r.total - base.total, "profit_delta": r.profit - base.profit,
            "revenue_delta": r.revenue - base.revenue, "note": note}


def growth_plan(fc: DemandForecaster, sales: pd.DataFrame, el: tuple, unc: float, base_inp: ForecastInputs,
                target_units: float | None = None) -> dict:
    """Evaluate demand levers for one product and return ranked, model-measured recommendations."""
    plain = replace(base_inp, promo_kind="None", events=())
    base = _fast(fc, sales, el, unc, plain)
    recs: list[dict] = []

    # Cross-product effects: what this product's promotions do to its siblings' sales, per promo day.
    spill = promo_spillover(sales, base_inp.pid) if not base_inp.new and base_inp.pid else pd.DataFrame()
    margin = (base_inp.price - base_inp.cost) / base_inp.price if base_inp.price else 0.0
    cross_units = float((spill["effect"] * spill["daily_level"]).sum()) if len(spill) else 0.0
    cross_profit = float((spill["effect"] * spill["daily_level"] * spill["price"]).sum() * margin) if len(spill) else 0.0

    def net(r: ScenarioResult) -> tuple[float, float]:
        days = float(r.res.promo.sum())
        return cross_units * days, cross_profit * days

    # 1. Price: a full sweep (±30%) using the product's price elasticity.
    prices = []
    for f in np.linspace(0.7, 1.3, 13):
        if base_inp.cost and base_inp.price * f <= base_inp.cost:
            continue  # never suggest selling at or below cost
        r = _fast(fc, sales, el, unc, replace(plain, price=round(base_inp.price * f, 2)))
        prices.append({"price": base_inp.price * f, "change": f - 1, "units": r.total, "profit": r.profit,
                       "revenue": r.revenue, "result": r})
    price_df = pd.DataFrame(prices)
    if len(price_df):
        best = price_df.loc[price_df["profit"].idxmax()]
        if abs(best["change"]) > 0.01:
            recs.append(_row("Price", f"Set the regular price to {best['price']:,.2f} ({best['change']:+.0%})",
                             best["result"], base, "Profit-maximising price within ±30%."))
        cut = price_df[(price_df["change"] < 0) & (price_df["profit"] >= base.profit * 0.97)]
        if len(cut):
            c = cut.loc[cut["units"].idxmax()]
            recs.append(_row("Price", f"Cut the price to {c['price']:,.2f} ({c['change']:+.0%}) for volume",
                             c["result"], base, "Most units while keeping at least 97% of current profit."))

    # 2. Promotions: depth × frequency grid (5-day promos).
    grid = []
    for disc in (0.10, 0.15, 0.20, 0.25, 0.30):
        if base_inp.cost and base_inp.price * (1 - disc) <= base_inp.cost:
            continue
        for every in (2, 4, 6):
            r = _fast(fc, sales, el, unc, replace(plain, promo_kind="Recurring", discount=disc, every=every, plen=5))
            cu, cp = net(r)
            grid.append({"discount": disc, "every_weeks": every, "units_delta": r.total - base.total + cu,
                         "profit_delta": r.profit - base.profit + cp, "cross_units": cu, "result": r})
    grid_df = pd.DataFrame(grid)
    if len(grid_df):
        g = grid_df.loc[grid_df["profit_delta"].idxmax()]
        cross_note = (f" Net of {g['cross_units']:+,.0f} units on other products in the category."
                      if abs(g["cross_units"]) >= 1 else "")
        if g["profit_delta"] > 0:
            row = _row("Promotions", f"Run {g['discount']:.0%}-off promos for 5 days every {int(g['every_weeks'])} "
                                     "weeks", g["result"], base, "Best promotion plan for profit." + cross_note)
            row.update(units_delta=g["units_delta"], profit_delta=g["profit_delta"])
            recs.append(row)
        v = grid_df.loc[grid_df["units_delta"].idxmax()]
        row = _row("Promotions", f"For maximum volume: {v['discount']:.0%} off, 5 days every "
                                 f"{int(v['every_weeks'])} weeks", v["result"], base,
                   "Most extra units; check the profit impact.")
        row.update(units_delta=v["units_delta"], profit_delta=v["profit_delta"])
        recs.append(row)

    # 3. Timing: which week is the best moment for one 7-day, 20%-off promotion.
    timing = []
    disc = 0.2 if not base_inp.cost or base_inp.price * 0.8 > base_inp.cost else 0.1
    for wk in range(0, max(1, base_inp.horizon // 7)):
        s = base_inp.start + pd.Timedelta(days=7 * wk)
        r = _fast(fc, sales, el, unc, replace(plain, promo_kind="Specific dates", discount=disc,
                                              promo_dates=(s.date(), (s + pd.Timedelta(days=6)).date())))
        timing.append({"week_start": s, "units_delta": r.total - base.total, "profit_delta": r.profit - base.profit,
                       "result": r})
    timing_df = pd.DataFrame(timing)
    if len(timing_df):
        t = timing_df.loc[timing_df["units_delta"].idxmax()]
        recs.append(_row("Timing", f"If you run one promo, start it the week of {t['week_start']:%d %b} "
                                   f"({disc:.0%} off, 7 days)", t["result"], base,
                         "The week where a single promotion adds the most units."))

    # 4. Availability: sales at risk from running out of stock (no reorder) and past suspected stock-outs.
    inv = base.inventory
    lost_future = 0.0
    if inv["stockout_date"] is not None:
        after = base.res.dates >= inv["stockout_date"]
        lost_future = float(base.res.mean[after].sum())
    lost_past = 0.0
    if base.history is not None and len(base.history) > 60:
        drops = detect_anomalies(base.history)
        drops = drops[drops["type"] == "Drop"]
        lost_past = float((drops["expected"] - drops["units_sold"]).clip(lower=0).sum())
    unit_margin = base_inp.price - base_inp.cost
    if lost_future > 0:
        recs.append({"lever": "Availability",
                     "action": f"Reorder by {inv['reorder_by']:%d %b} — stock runs out {inv['stockout_date']:%d %b}",
                     "units": base.total, "revenue": base.revenue, "profit": base.profit,
                     "units_delta": lost_future, "revenue_delta": lost_future * base_inp.price,
                     "profit_delta": lost_future * unit_margin,
                     "note": "Forecast sales that would be lost after the stock-out if nothing is ordered."})

    # 5. Target: cheapest way (highest profit) to reach a unit goal.
    target = None
    if target_units:
        options = [(f"Price {p['price']:,.2f}", p["result"]) for p in prices]
        options += [(f"{g['discount']:.0%} off every {int(g['every_weeks'])} wks", g["result"]) for g in grid]
        for p in prices:  # price × promo combinations
            if p["change"] >= 0:
                continue
            for d in (0.10, 0.20):
                if base_inp.cost and p["price"] * (1 - d) <= base_inp.cost:
                    continue
                r = _fast(fc, sales, el, unc, replace(plain, price=round(p["price"], 2), promo_kind="Recurring",
                                                      discount=d, every=4, plen=5))
                options.append((f"Price {p['price']:,.2f} + {d:.0%} off every 4 wks", r))
        hits = [(n, r) for n, r in options if r.total >= target_units]
        if hits:
            name, r = max(hits, key=lambda x: x[1].profit)
            target = {"reached": True, "plan": name, "units": r.total, "profit": r.profit,
                      "profit_delta": r.profit - base.profit}
        else:
            name, r = max(options, key=lambda x: x[1].total) if options else ("Current plan", base)
            needed = target_units / max(r.total, 1e-9) - 1
            target = {"reached": False, "plan": name, "units": r.total, "profit": r.profit,
                      "profit_delta": r.profit - base.profit, "extra_uplift_needed": needed}

    rec_df = pd.DataFrame(recs)
    if len(rec_df):
        rec_df = rec_df.sort_values(["profit_delta", "units_delta"], ascending=False).reset_index(drop=True)
    return {"base": base, "recommendations": rec_df, "spillover": spill, "price": price_df.drop(columns="result", errors="ignore"),
            "promo_grid": grid_df.drop(columns="result", errors="ignore"),
            "timing": timing_df.drop(columns="result", errors="ignore"), "target": target,
            "lost_past": lost_past, "lost_future": lost_future, "good_practice": GOOD_PRACTICE}
