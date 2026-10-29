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

with st.sidebar:
    st.html('<div class="dm-brand"><i></i><b>Demand<span>ino</span></b></div>'
            '<div class="dm-sub">Machine-learning demand forecasting</div>')
    ui.theme_toggle()
    if legal.accepted():
        account.sidebar_account()
legal.sync_cookie()
account.sync_cookie()

pages = {
    "Forecast": [
        st.Page("views/forecast.py", title="Forecast a product", icon=":material/online_prediction:", default=True),
        st.Page("views/planning.py", title="Demand planning", icon=":material/inventory:"),
        st.Page("views/advisor.py", title="Grow demand", icon=":material/rocket_launch:"),
    ],
    "Analyze": [
        st.Page("views/explore.py", title="Explore sales patterns", icon=":material/query_stats:"),
        st.Page("views/overview.py", title="Portfolio overview", icon=":material/dashboard:"),
        st.Page("views/hierarchy.py", title="Store & category totals", icon=":material/account_tree:"),
        st.Page("views/assistant.py", title="Ask Demandino", icon=":material/auto_awesome:"),
    ],
    "Workspace": [
        st.Page("views/workspace.py", title="Workspace", icon=":material/folder_shared:"),
        st.Page("views/connections.py", title="Connections & alerts", icon=":material/hub:"),
        st.Page("views/orders.py", title="Purchase orders", icon=":material/receipt_long:"),
    ],
    "Model": [st.Page("views/accuracy.py", title="Model accuracy", icon=":material/fact_check:"),
              st.Page("views/tracking.py", title="Forecast tracking", icon=":material/track_changes:")],
    "About": [st.Page("views/terms.py", title="Terms & conditions", icon=":material/gavel:")],
}
nav = st.navigation(pages)

if not legal.gate():
    st.stop()

state = sidebar_data_source()
if state is None:
    st.stop()
st.session_state["_app"] = state
_ = state.fc  # train (or load) the model before any page renders
nav.run()
