"""Did your adjustments help? Compares the model's own forecast with the one you adjusted, against real sales.

A tracked forecast stores both versions (the model's, and yours after your adjustments) plus the list of
adjustments. Once real sales arrive, each adjustment is scored only on the days it changed: if the absolute
error of your version is lower than the model's, it helped. This is the "forecast value added" check that
planning teams use to learn when their own judgement beats the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NO_DIFFERENCE = 0.02  # error changes smaller than this share count as "no real difference"


def _verdict(model_err: float, your_err: float) -> str:
    if model_err <= 0 and your_err <= 0:
        return "No difference"
    change = (your_err - model_err) / max(model_err, 1e-9)
    if abs(change) < NO_DIFFERENCE:
        return "No difference"
    return "Helped" if change < 0 else "Made it worse"


def score(logs: list[dict], sales: pd.DataFrame) -> pd.DataFrame:
    """One row per adjustment with real sales to judge it: days scored, model error, your error, verdict."""
    rows = []
    for lg in logs:
        p = lg["payload"]
        if not p.get("adjustments") or "base_mean" not in p:
            continue
        dates = pd.to_datetime(p["dates"])
        actual = (sales[sales["product_id"] == lg["product_id"]].set_index("date")["units_sold"]
                  .reindex(dates).to_numpy(float))
        model, yours = np.asarray(p["base_mean"], float), np.asarray(p["mean"], float)
        for name, start, end, change in p["adjustments"]:
            on = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)) & ~np.isnan(actual)
            if not on.any():
                continue
            a = actual[on]
            m_err, y_err = float(np.abs(a - model[on]).sum()), float(np.abs(a - yours[on]).sum())
            rows.append({"log_id": lg["id"], "product": lg["product_name"], "adjustment": str(name),
                         "change_%": float(change), "start": pd.Timestamp(start), "end": pd.Timestamp(end),
                         "days_scored": int(on.sum()), "actual": float(a.sum()), "model_forecast": float(model[on].sum()),
                         "your_forecast": float(yours[on].sum()), "model_error": m_err, "your_error": y_err,
                         "verdict": _verdict(m_err, y_err)})
    return pd.DataFrame(rows)


def summary(scored: pd.DataFrame) -> dict | None:
    """Headline: how much your adjustments changed the error on the days they applied."""
    if scored.empty:
        return None
    m, y = scored["model_error"].sum(), scored["your_error"].sum()
    return {"adjustments": len(scored), "helped": int((scored["verdict"] == "Helped").sum()),
            "hurt": int((scored["verdict"] == "Made it worse").sum()),
            "error_change": (y - m) / m if m else 0.0, "model_error": m, "your_error": y}
