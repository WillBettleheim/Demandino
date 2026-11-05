import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandino import budget, db, ui
from demandino.planning import default_stock
from demandino.security import safe_csv
from demandino.state import app

A = app()


@st.cache_data(show_spinner="Working out what each month needs to buy…", max_entries=8)
def needs_for(data_key: str, stock: pd.DataFrame, months: int) -> pd.DataFrame:
    return budget.needs(A.fc, stock, months)


ui.page_header("Buying budget", "How much each month needs to spend on new stock to meet the forecast, after "
               "using the stock you already have — against the budget you set.", eyebrow="Forecast")
saved = db.get_setting(A.user_id, f"stock:{A.settings_key}") if A.user_id else None
stock = st.session_state.get(f"pl_stock_{A.settings_key}")
stock = stock if stock is not None else (pd.DataFrame(saved) if saved else default_stock(A.sales))
n_months = st.segmented_control("Months ahead", [3, 6, 12], default=6, required=True, key="bb_months")
n = needs_for(A.data_key, stock, int(n_months))
typical = float(n.groupby("month")["buy"].sum().median())
amount = st.number_input(f"Monthly buying budget ({ui.currency()})", min_value=0.0,
                         value=float(round(typical, -3) or 1000.0), step=1000.0, key="bb_budget",
                         help="Starts at the typical month's need.")
m = budget.against_budget(n, amount)
over = int((m["status"] == "Over budget").sum())
ui.spotlight(f"Next {n_months} months", ui.money(m["buy"].sum()), "to spend on new stock",
             f"Budget <b>{ui.money(m['budget'].sum())}</b> · <b>{over}</b> month(s) over budget",
             [("All months fit", "good")] if not over else [(f"{over} over budget", "bad")])
ui.finding_cards(budget.summary(m, ui.currency()))
fig = go.Figure()
for cat, g in n.groupby("category"):
    fig.add_bar(x=g["month"], y=g["buy"], name=cat)
fig.add_scatter(x=m["month"], y=m["budget"], name="Budget", mode="lines", line=dict(color=ui.HIST, dash="dot"))
ui.show(ui.style(fig, f"Money needed for new stock, by category ({ui.currency()})", height=380, barmode="stack"))
st.dataframe(m, hide_index=True, width="stretch", column_config={
    "month": st.column_config.DateColumn("Month", format="MMM YYYY"),
    "units": st.column_config.NumberColumn("Units sold", format="%,.0f"),
    "cogs": st.column_config.NumberColumn("Cost of goods sold", format="%,.0f"),
    "buy": st.column_config.NumberColumn("Needs buying", format="%,.0f"),
    "budget": st.column_config.NumberColumn("Budget", format="%,.0f"),
    "left": st.column_config.NumberColumn("Left", format="%+,.0f"), "status": "Status"})
st.download_button("Download by month and category (CSV)", safe_csv(n), "buying_budget.csv", "text/csv",
                   icon=":material/download:")
st.caption("Uses the stock table from Demand planning (illustrative values until you save your own) and unit costs "
           "(55% of price where unknown). Buying is counted in the month the stock is needed; order lead times "
           "move the spending earlier.")
