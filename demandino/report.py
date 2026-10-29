"""Excel export of a forecast scenario."""
from __future__ import annotations

import io
import re

import pandas as pd

from .model import RANGE_LABEL
from .scenario import ScenarioResult
from .security import sanitize_export


def _plain(markdown: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"\1", markdown)


def scenario_workbook(r: ScenarioResult, currency: str = "$") -> bytes:
    inp = r.inputs
    inv = r.inventory
    summary = [
        ("Product", inp.name), ("Category", inp.category), ("Forecast start", inp.start.date()),
        ("Horizon (days)", inp.horizon), ("Regular price", inp.price), ("Unit cost", inp.cost),
        ("Demand basis", r.method), ("Price elasticity used", round(r.elasticity, 2)),
        ("Promotion plan", inp.promo_kind if inp.promo_kind == "None" else f"{inp.promo_kind}, {inp.discount:.0%} off"),
        ("", ""),
        ("Expected units", round(r.total)), (f"Low ({RANGE_LABEL} range)", round(r.low)), (f"High ({RANGE_LABEL} range)", round(r.high)),
        ("Average per day", round(r.total / inp.horizon, 1)), (f"Revenue ({currency})", round(r.revenue)),
        (f"Gross profit ({currency})", round(r.profit)), ("Confidence", r.confidence), ("", ""),
        ("Stock on hand", inp.stock), ("Safety stock", round(inv["safety_stock"])),
        ("Reorder point", round(inv["reorder_point"])),
        ("Stock-out date", inv["stockout_date"].date() if inv["stockout_date"] is not None else "Not in horizon"),
        ("Order by", inv["reorder_by"].date() if inv["reorder_by"] is not None else "Not needed in horizon"),
        ("Order quantity", round(inv["order_qty"])),
    ]
    daily = r.frame()
    daily["date"] = daily["date"].dt.date
    f = r.frame().set_index("date")
    weekly = f.resample("W-SUN").agg(units=("forecast", "sum"), low=("lower", "sum"), high=("upper", "sum"),
                                      revenue=("revenue", "sum"), promo_days=("promo", "sum"))
    monthly = f.resample("MS").agg(units=("forecast", "sum"), revenue=("revenue", "sum"), promo_days=("promo", "sum"))
    for t in (weekly, monthly):
        t.index = t.index.date

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        sanitize_export(pd.DataFrame(summary, columns=["Item", "Value"]).astype({"Value": object})).to_excel(
            xl, sheet_name="Summary", index=False)
        sanitize_export(pd.DataFrame({"Key findings": [_plain(s) for s in r.insights]})).to_excel(
            xl, sheet_name="Findings", index=False)
        daily.round(2).to_excel(xl, sheet_name="Daily forecast", index=False)
        weekly.round(1).to_excel(xl, sheet_name="Weekly", index_label="week ending")
        monthly.round(1).to_excel(xl, sheet_name="Monthly", index_label="month")
        pd.DataFrame(r.drivers, columns=["Driver", "Units"]).round(1).to_excel(xl, sheet_name="Drivers", index=False)
        r.sensitivity.round(2).to_excel(xl, sheet_name="Price what-if", index=False)
        for ws in xl.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(12, width + 2), 110)
    return buf.getvalue()
