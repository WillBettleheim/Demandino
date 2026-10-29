"""Which product details a forecast needs, which were provided, and what each one changes."""
from __future__ import annotations

from .scenario import ForecastInputs

DEFAULT_STOCK, DEFAULT_LEAD = 500, 14


def checklist(inp: ForecastInputs, default_price: float, default_cost: float) -> list[dict]:
    """Return rows of {detail, status: provided|assumed|missing|optional, why} for the current inputs."""
    rows = []

    def add(detail: str, status: str, why: str) -> None:
        rows.append({"detail": detail, "status": status, "why": why})

    if inp.new:
        add("Product name", "provided" if inp.name not in ("My new product", "New product") else "assumed",
            "Labels the forecast and reports.")
        add("Category", "provided", "Sets the seasonality, weekday pattern and typical demand level.")
        basis_status = {"history": "provided" if inp.history is not None else "missing",
                        "similar": "provided", "auto": "assumed"}[inp.basis]
        add("Sales evidence (history, similar product or pilot)",
            "provided" if inp.expected else basis_status,
            "The single biggest driver of accuracy for a new product. Without it the level is estimated from the "
            "category and price only, so the range is wide.")
        add("Launch ramp-up", "provided" if inp.ramp_days else "optional",
            "New products take weeks to reach full sales; ignoring this over-forecasts the first month.")
    else:
        add("Sales history", "provided", "Learned automatically from your data: level, trend, seasonality, weekdays.")
    add("Regular selling price", "provided" if abs(inp.price - default_price) > 0.005 or not inp.new else "assumed",
        "Price changes demand through the product's price elasticity.")
    add("Unit cost", "provided" if abs(inp.cost - default_cost) > 0.005 else "assumed",
        "Needed for profit, pricing advice and whether a promotion pays off (currently assumed at 55% of price)."
        if abs(inp.cost - default_cost) <= 0.005 else "Used for profit, pricing advice and promotion payback.")
    add("Promotion plan", "provided" if inp.promo_kind != "None" else "optional",
        "Planned discounts lift demand on promo days.")
    add("Known events (campaigns, closures)", "provided" if inp.events else "optional",
        "Things the history can't know about, e.g. a marketing push or a store closure.")
    add("Stock on hand & lead time",
        "provided" if (inp.stock != DEFAULT_STOCK or inp.lead != DEFAULT_LEAD) else "assumed",
        "Needed for the run-out date, reorder date and order quantity.")
    return rows


def score(rows: list[dict]) -> float:
    """Share of the essential (non-optional) details that were actually provided."""
    essential = [r for r in rows if r["status"] != "optional"]
    return sum(r["status"] == "provided" for r in essential) / max(len(essential), 1)
