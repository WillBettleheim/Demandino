"""Demandino — ML demand forecasting for products.

Run with:  streamlit run app.py
"""
import streamlit as st

from demandino import account, legal, ui
from demandino.state import sidebar_data_source

st.set_page_config(page_title="Demandino · Demand Forecasting", page_icon=":material/insights:", layout="wide")
ui.inject_css()

share = st.query_params.get("share")
if share:
    from demandino import shared_view

    shared_view.render(share)
    st.stop()

page = st.Page
pages = {
    "": [page("views/welcome.py", title="Home", icon=":material/home:", default=True),
         page("views/home.py", title="Dashboard", icon=":material/space_dashboard:", url_path="dashboard"),
         page("views/inbox.py", title="Inbox", icon=":material/inbox:")],
    "Forecast": [
        page("views/forecast.py", title="Forecast a product", icon=":material/online_prediction:", url_path="forecast"),
        page("views/changes.py", title="What changed", icon=":material/difference:"),
        page("views/alerts.py", title="Demand alerts", icon=":material/notifications_active:"),
        page("views/accuracy.py", title="Model accuracy", icon=":material/fact_check:"),
        page("views/tracking.py", title="Forecast tracking", icon=":material/track_changes:"),
    ],
    "Plan": [
        page("views/planning.py", title="Demand planning", icon=":material/inventory:"),
        page("views/budget.py", title="Buying budget", icon=":material/account_balance_wallet:"),
        page("views/orders.py", title="Purchase orders", icon=":material/receipt_long:"),
        page("views/clearance.py", title="Clear slow stock", icon=":material/sell:"),
    ],
    "Grow": [
        page("views/advisor.py", title="Grow demand", icon=":material/rocket_launch:"),
        page("views/targets.py", title="Sales targets", icon=":material/flag:"),
        page("views/whatif.py", title="Store what-if", icon=":material/science:"),
        page("views/together.py", title="Bought together", icon=":material/shopping_basket:"),
    ],
    "Analyze": [
        page("views/explore.py", title="Explore sales patterns", icon=":material/query_stats:"),
        page("views/overview.py", title="Portfolio overview", icon=":material/dashboard:"),
        page("views/hierarchy.py", title="Store & category totals", icon=":material/account_tree:"),
        page("views/lifecycle.py", title="Product life stages", icon=":material/timeline:"),
        page("views/lostsales.py", title="Lost sales", icon=":material/money_off:"),
        page("views/compare.py", title="Compare products", icon=":material/compare_arrows:"),
        page("views/assistant.py", title="Ask Demandino", icon=":material/auto_awesome:"),
    ],
    "Settings": [
        page("views/workspace.py", title="Workspace", icon=":material/folder_shared:"),
        page("views/datacheck.py", title="Data check", icon=":material/rule:"),
        page("views/connections.py", title="Connections & alerts", icon=":material/hub:"),
        page("views/security.py", title="Security", icon=":material/shield:"),
        page("views/history.py", title="Change history", icon=":material/history:"),
        page("views/help.py", title="Help & glossary", icon=":material/menu_book:"),
        page("views/terms.py", title="Terms & conditions", icon=":material/gavel:"),
    ],
}
nav = st.navigation(pages, position="top")

# The welcome page needs no data, sidebar or model, so it opens instantly and the terms come when someone starts.
if nav.url_path == "":
    legal.sync_cookie()
    account.sync_cookie()
    nav.run()
    st.stop()

with st.sidebar:
    st.html('<div class="dm-brand"><i></i><b>Demand<span>ino</span></b></div>'
            '<div class="dm-sub">Machine-learning demand forecasting</div>')
    ui.theme_toggle()
    if legal.accepted():
        account.sidebar_account()
legal.sync_cookie()
account.sync_cookie()

if not legal.gate():
    st.stop()

state = sidebar_data_source()
if state is None:
    st.stop()
st.session_state["_app"] = state
_ = state.fc  # train (or load) the model before any page renders
nav.run()
