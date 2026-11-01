"""First-run guidance and plain-language summaries, so people new to forecasting know what to do next."""
from __future__ import annotations

import html

import streamlit as st

from . import account, db, ui

WELCOME_KEY = "welcome_done"
WELCOME_COOKIE = "dm_welcome"
STEPS = [
    ("1", "Pick a product", "Choose one you already sell, or switch to <b>New product</b> and describe it."),
    ("2", "Set your plans", "Enter the price, any promotion and how much stock you have. Everything else is optional."),
    ("3", "Read the answer", "The top of the results says how many will sell, when to reorder and what it earns."),
]


def _welcome_done() -> bool:
    if st.session_state.get(WELCOME_KEY) or st.context.cookies.get(WELCOME_COOKIE) == "1":
        return True
    uid = account.current_user()
    return bool(uid and db.get_setting(uid, WELCOME_KEY, False))


def welcome() -> None:
    """A short three-step guide shown until the person dismisses it (remembered in a cookie and, when signed in,
    in their workspace)."""
    if st.session_state.pop("_welcome_cookie", False):
        st.html(f"<script>document.cookie = '{WELCOME_COOKIE}=1; Max-Age=31536000; Path=/; SameSite=Strict';</script>",
                unsafe_allow_javascript=True)
    if _welcome_done():
        return
    steps = "".join(f'<div class="dm-step"><span>{n}</span><div><b>{t}</b><p>{d}</p></div></div>' for n, t, d in STEPS)
    with st.container(key="dm_welcome"):
        st.html(f'<div class="dm-welcome-title">New here? Three steps to your first forecast</div>'
                f'<div class="dm-steps">{steps}</div>')
        with st.container(horizontal=True):
            if st.button("Got it", type="primary", icon=":material/check:", key="welcome_ok"):
                st.session_state[WELCOME_KEY] = True
                st.session_state["_welcome_cookie"] = True
                uid = account.current_user()
                if uid:
                    db.set_setting(uid, WELCOME_KEY, True)
                st.rerun()
            st.page_link("views/help.py", label="Open the full guide", icon=":material/menu_book:")


def bottom_line(r, horizon: int) -> None:
    """The forecast in two or three plain sentences, ending with what to do."""
    inv, inp = r.inventory, r.inputs
    per_day = r.total / horizon
    lines = [f"Expect to sell about <b>{ui.num(r.total)} units</b> in the next {horizon} days "
             f"(roughly {ui.num(per_day, 1 if per_day < 10 else 0)} a day). It will most likely land between "
             f"{ui.num(r.low)} and {ui.num(r.high)}."]
    if r.revenue:
        lines.append(f"That's about <b>{ui.money(r.revenue)}</b> in sales and <b>{ui.money(r.profit)}</b> profit "
                     f"at {ui.money(inp.price, 2)} each.")
    if inv["needs_order_now"]:
        action = f"Order about <b>{ui.num(inv['order_qty'])} units now</b> — stock is already at the reorder point."
        tone = "warn"
    elif inv["reorder_by"] is not None:
        runs_out = (f"Your {ui.num(inp.stock)} units run out around {inv['stockout_date']:%d %b}. "
                    if inv["stockout_date"] is not None else "")
        action = f"{runs_out}Order about <b>{ui.num(inv['order_qty'])} units by {inv['reorder_by']:%d %b}</b>."
        tone = "warn"
    else:
        action = f"Your {ui.num(inp.stock)} units in stock cover this period — <b>no order needed</b>."
        tone = "good"
    lines.append(action)
    st.html(f'<div class="dm-bottom {tone}"><div class="dm-bottom-label">In short</div>'
            + "".join(f"<p>{line}</p>" for line in lines) + "</div>")


GLOSSARY = {
    "Forecast": "Our best estimate of how many units will sell. It is an average expectation, not a promise.",
    "Range (95%)": "The span real sales should fall inside 95 days out of 100. Plan for the low end when cash is "
                   "tight and for the high end when running out is expensive.",
    "Horizon": "How many days ahead you want to forecast. Shorter horizons are more accurate.",
    "Lead time": "Days between placing an order with your supplier and the stock arriving on your shelf.",
    "Days between orders": "How often you usually place orders (for example every 14 days).",
    "Service level": "How hard to try never to run out. 95% means stock should last through 19 of every 20 "
                     "order cycles. Higher means more safety stock and more cash tied up.",
    "Safety stock": "Extra units kept on hand to cover days when demand is higher than expected.",
    "Reorder point": "When stock falls to this number, it's time to place the next order.",
    "Price elasticity": "How strongly demand reacts to price. −2 means a 10% price cut lifts units by about 20%.",
    "Promotion lift": "How much extra a discount sells on promotion days compared with normal days.",
    "Seasonality": "Patterns that repeat every year, such as a December peak or a summer dip.",
    "Demand events": "Things you know are coming that the past can't show: a campaign, a store closure, a "
                     "competitor launch. Enter them as a % change.",
    "Stock-out": "Days when a product had no stock. Sales were zero because nothing could be sold, not because "
                 "nobody wanted it, so the model ignores those days when learning.",
    "Miss rate": "How often real sales landed outside the predicted range in testing. Lower is better; the "
                 "target is under 7%.",
    "WAPE": "Average size of the forecast error as a share of real sales. Single days are noisy, so weekly "
            "WAPE is the more useful number for planning.",
    "Bias": "Whether forecasts run high (positive) or low (negative) on average.",
}


def glossary_cards(terms: dict[str, str] | None = None) -> None:
    terms = terms or GLOSSARY
    cards = "".join(f'<div class="dm-finding"><span class="dm-ft">{html.escape(k)}</span>{html.escape(v)}</div>'
                    for k, v in terms.items())
    st.html(f'<div class="dm-findings">{cards}</div>')
