import streamlit as st

from demandino import landing

st.html(landing.CSS)
with st.container(key="dml_stage"):
    st.html(landing.hero_html() + landing.JS, unsafe_allow_javascript=True)  # our own static code
    with st.container(key="dml_cta", horizontal=True):
        if st.button("Open your dashboard", type="primary", icon=":material/arrow_outward:", key="dml_open"):
            st.switch_page("views/home.py")
        if st.button("Forecast a product", icon=":material/online_prediction:", key="dml_try"):
            st.switch_page("views/forecast.py")
st.html(landing.story_html())
with st.container(key="dml_cta2"):
    if st.button("Start with the sample shop", type="primary", icon=":material/arrow_forward:", key="dml_start"):
        st.switch_page("views/home.py")
st.html(landing.FOOT)
