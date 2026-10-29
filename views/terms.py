import streamlit as st

from demandino import ui
from demandino.legal import TERMS_VERSION, terms_markdown

ui.page_header("Terms & conditions", f"Version {TERMS_VERSION}. You accepted these terms to use Demandino.",
               eyebrow="Legal")
with ui.card("terms-page"):
    st.markdown(terms_markdown())
if st.button("Withdraw acceptance", icon=":material/undo:",
             help="You will need to accept the terms again to keep using the app."):
    st.session_state.pop("terms_accepted", None)
    st.session_state["_terms_cookie"] = "clear"
    st.session_state["terms_withdrawn"] = True
    st.rerun()
