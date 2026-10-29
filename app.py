"""Demandino — ML demand forecasting for products.

Run with:  streamlit run app.py
"""
import streamlit as st

from demandino import legal, ui
from demandino.state import sidebar_data_source

st.set_page_config(page_title="Demandino · Demand Forecasting", page_icon=":material/insights:", layout="wide")
ui.inject_css()

with st.sidebar:
    st.html('<div class="dm-brand"><i></i><b>Demand<span>ino</span></b></div>'
            '<div class="dm-sub">Machine-learning demand forecasting</div>')
    ui.theme_toggle()
legal.sync_cookie()

pages = {
    "Forecast": [
        st.Page("views/forecast.py", title="Forecast a product", icon=":material/online_prediction:", default=True),
        st.Page("views/planning.py", title="Demand planning", icon=":material/inventory:"),
        st.Page("views/advisor.py", title="Grow demand", icon=":material/rocket_launch:"),
    ],
    "Analyze": [
        st.Page("views/explore.py", title="Explore sales patterns", icon=":material/query_stats:"),
        st.Page("views/overview.py", title="Portfolio overview", icon=":material/dashboard:"),
    ],
    "Model": [st.Page("views/accuracy.py", title="Model accuracy", icon=":material/fact_check:")],
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
