"""Supplier delivery times learned from purchase orders that were marked as received.

Each received order gives one delivery time per supplier on it (arrival date − order date). Per supplier:
how long deliveries really take, how much that varies, how often they're later than the lead time you
planned with, and whether they're getting slower. Planning can then use a lead time that covers the
variation (the average plus a safety margin), so unreliable suppliers automatically get more safety stock.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

MIN_DELIVERIES = 3   # fewer deliveries than this: keep using the lead time typed in
DEFAULT_SUPPLIER = "Main supplier"


def deliveries(pos: list[dict]) -> pd.DataFrame:
    """One row per received purchase order × supplier: order date, arrival date, days taken, planned lead time."""
    rows = []
    for po in pos:
        if po.get("status") != "Received" or not po.get("received_at") or not po.get("ordered_at"):
            continue
        lines = pd.DataFrame(po["lines"])
        if lines.empty:
            continue
        if "supplier" not in lines:
            lines["supplier"] = DEFAULT_SUPPLIER
        lines["supplier"] = lines["supplier"].fillna(DEFAULT_SUPPLIER).astype(str).replace("", DEFAULT_SUPPLIER)
        ordered, received = pd.Timestamp(po["ordered_at"], unit="s"), pd.Timestamp(po["received_at"], unit="s")
        for supplier, g in lines.groupby("supplier"):
            planned = pd.to_numeric(g.get("lead_time"), errors="coerce").mean() if "lead_time" in g else np.nan
            rows.append({"po_id": po["id"], "supplier": supplier, "ordered": ordered.normalize(),
                         "received": received.normalize(), "days": (received.normalize() - ordered.normalize()).days,
                         "planned": float(planned) if pd.notna(planned) else np.nan, "lines": len(g)})
    return pd.DataFrame(rows, columns=["po_id", "supplier", "ordered", "received", "days", "planned", "lines"])


def stats(dl: pd.DataFrame, service: float = 0.95) -> pd.DataFrame:
    """Per supplier: deliveries, average and spread of days, share late, a planning lead time and a trend flag."""
    if dl.empty:
        return pd.DataFrame(columns=["supplier", "deliveries", "avg_days", "sd_days", "late_share", "planning_lead",
                                     "trend", "advice"])
    z = NormalDist().inv_cdf(service)
    rows = []
    for supplier, g in dl.sort_values("ordered").groupby("supplier"):
        d = g["days"].to_numpy(float)
        n = len(d)
        sd = float(d.std(ddof=1)) if n > 1 else 0.0
        late = float(np.mean(d > g["planned"].to_numpy(float))) if g["planned"].notna().any() else np.nan
        # The average plus a margin for variation: what to plan with so deliveries are rarely later.
        planning = int(np.ceil(d.mean() + z * sd)) if n >= MIN_DELIVERIES else None
        trend = "Not enough deliveries yet"
        if n >= 5:
            half = n // 2
            early, recent = d[:half].mean(), d[-half:].mean()
            trend = ("Getting slower" if recent > early * 1.2 and recent - early >= 2 else
                     "Getting faster" if recent < early * 0.8 and early - recent >= 2 else "Steady")
        rows.append({"supplier": supplier, "deliveries": n, "avg_days": float(d.mean()), "sd_days": sd,
                     "late_share": late, "planning_lead": planning, "trend": trend,
                     "advice": _advice(n, float(d.mean()), sd, late, trend)})
    return pd.DataFrame(rows).sort_values("avg_days", ascending=False).reset_index(drop=True)


def _advice(n: int, avg: float, sd: float, late: float, trend: str) -> str:
    if n < MIN_DELIVERIES:
        return f"Record {MIN_DELIVERIES - n} more deliveries before planning relies on them."
    if trend == "Getting slower":
        return "Deliveries are getting slower — ask why, and order earlier until it settles."
    if not np.isnan(late) and late >= 0.3:
        return f"Late {late:.0%} of the time against the lead time you planned with — plan with the learned time."
    if sd > avg * 0.3:
        return "Delivery times vary a lot — the extra safety stock covers it; a steadier supplier would free cash."
    return "Reliable — delivery times are steady."


def apply_learned_lead_times(stock: pd.DataFrame, st: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Replace each row's lead time with its supplier's planning lead time where enough deliveries exist."""
    out = stock.copy()
    if "supplier" not in out:
        out["supplier"] = DEFAULT_SUPPLIER
    learned = st.dropna(subset=["planning_lead"]).set_index("supplier")["planning_lead"].to_dict() if len(st) else {}
    notes = []
    for supplier, lead in learned.items():
        rows = out["supplier"].fillna(DEFAULT_SUPPLIER).astype(str) == supplier
        if rows.any():
            before = pd.to_numeric(out.loc[rows, "lead_time"], errors="coerce").mean()
            out.loc[rows, "lead_time"] = int(lead)
            if abs(before - lead) >= 1:
                notes.append(f"{supplier}: {before:.0f} → {int(lead)} days")
    return out, notes
