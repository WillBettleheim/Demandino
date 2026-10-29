"""One-page-plus PDF report of a forecast scenario (fpdf2, built-in fonts only)."""
from __future__ import annotations

import re

import pandas as pd
from fpdf import FPDF

from .model import RANGE_LABEL
from .scenario import ScenarioResult

_SUBS = {"–": "-", "—": "-", "≈": "~", "·": "-", "→": "->", "≥": ">=", "≤": "<=", "’": "'", "‘": "'", "“": '"',
         "”": '"', "…": "...", "€": "EUR ", "₹": "INR ", "₦": "NGN ", "¥": "JPY ", "×": "x", "±": "+/-", "✓": "ok"}


def _t(text: str) -> str:
    """Built-in PDF fonts are Latin-1 only; swap common symbols and drop the rest safely."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", str(text))
    for a, b in _SUBS.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


def scenario_pdf(r: ScenarioResult, currency: str = "$") -> bytes:
    cur = _t(currency)
    inp, inv = r.inputs, r.inventory
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_fill_color(13, 148, 136)
    pdf.rect(0, 0, 210, 28, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(12, 8)
    pdf.cell(0, 8, _t(f"Demand forecast - {inp.name}"))
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(12, 17)
    pdf.cell(0, 6, _t(f"{inp.category} - {inp.horizon} days from {inp.start:%d %b %Y} - Demandino"))
    pdf.set_text_color(11, 31, 42)
    pdf.set_xy(12, 34)

    stats = [("Expected units", f"{r.total:,.0f}"), (f"{RANGE_LABEL} range", f"{r.low:,.0f} - {r.high:,.0f}"),
             ("Revenue", f"{cur}{r.revenue:,.0f}"), ("Gross profit", f"{cur}{r.profit:,.0f}"),
             ("Confidence", r.confidence),
             ("Stock runs out", f"{inv['stockout_date']:%d %b %Y}" if inv["stockout_date"] is not None else "Covered")]
    w = 186 / 3
    for i, (k, v) in enumerate(stats):
        x, y = 12 + (i % 3) * w, 34 + (i // 3) * 18
        pdf.set_draw_color(211, 227, 231)
        pdf.rect(x, y, w - 3, 15)
        pdf.set_xy(x + 3, y + 2)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(w - 6, 4, _t(k.upper()))
        pdf.set_xy(x + 3, y + 7)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(w - 6, 6, _t(v))

    # Weekly bar chart drawn with rectangles.
    weekly = r.frame().set_index("date")["forecast"].resample("W").sum()
    top, height, left, width = 78, 45, 14, 182
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_xy(12, top - 8)
    pdf.cell(0, 6, "Weekly expected units")
    mx = max(float(weekly.max()), 1.0)
    bw = width / max(len(weekly), 1)
    pdf.set_fill_color(249, 115, 80)
    pdf.set_font("Helvetica", "", 6)
    for i, (d, v) in enumerate(weekly.items()):
        h = height * float(v) / mx
        pdf.rect(left + i * bw + 0.6, top + height - h, bw - 1.2, h, "F")
        if i % max(1, len(weekly) // 8) == 0:
            pdf.set_xy(left + i * bw, top + height + 1)
            pdf.cell(bw * 2, 3, d.strftime("%d %b"))

    pdf.set_xy(12, top + height + 8)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Key findings", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    for line in r.insights:
        pdf.set_x(12)
        pdf.multi_cell(186, 4.6, _t("- " + line), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(0.8)

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Inventory plan", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    for k, v in [("Stock on hand", f"{inp.stock:,.0f}"), ("Safety stock", f"{inv['safety_stock']:,.0f}"),
                 ("Reorder point", f"{inv['reorder_point']:,.0f}"),
                 ("Order by", f"{inv['reorder_by']:%d %b %Y}" if inv["reorder_by"] is not None else "Not needed"),
                 ("Order quantity", f"{inv['order_qty']:,.0f}")]:
        pdf.cell(50, 5, _t(k))
        pdf.cell(0, 5, _t(v), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(90, 100, 110)
    pdf.multi_cell(186, 3.5, _t(f"Forecasts are estimates, not guarantees; real sales can fall outside the {RANGE_LABEL} "
                                "range. Not financial advice. Generated " + pd.Timestamp.now().strftime("%d %b %Y %H:%M")
                                + "."))
    return bytes(pdf.output())
