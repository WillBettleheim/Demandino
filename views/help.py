import streamlit as st

from demandino import guide, ui

ui.page_header("Help & glossary",
               "Everything you need to get useful forecasts — no statistics background required.", eyebrow="Guide")

st.subheader("Your first forecast in three steps")
st.html('<div class="dm-steps">' + "".join(
    f'<div class="dm-step"><span>{n}</span><div><b>{t}</b><p>{d}</p></div></div>' for n, t, d in guide.STEPS)
    + "</div>")
st.page_link("views/forecast.py", label="Go to Forecast a product", icon=":material/arrow_forward:")

st.subheader("Which page do I need?")
PAGES = [
    ("views/forecast.py", "Forecast a product", ":material/online_prediction:",
     "How many of one product will sell, and when to reorder it."),
    ("views/planning.py", "Demand planning", ":material/inventory:",
     "Order quantities for every product at once, within a budget."),
    ("views/advisor.py", "Grow demand", ":material/rocket_launch:",
     "Concrete ideas to sell more: price, promotions, timing and stock."),
    ("views/alerts.py", "Demand alerts", ":material/notifications_active:",
     "Which products are suddenly selling faster or slower than the forecast said."),
    ("views/clearance.py", "Clear slow stock", ":material/sell:",
     "Products with too much stock, and the markdown that clears it."),
    ("views/tracking.py", "Forecast tracking", ":material/track_changes:",
     "Compare past forecasts with what sold — and whether your adjustments helped."),
    ("views/explore.py", "Explore sales patterns", ":material/query_stats:",
     "Look back at past sales: trends, best days, seasons, promotions."),
    ("views/overview.py", "Portfolio overview", ":material/dashboard:",
     "Which products and categories matter most, and which are growing."),
    ("views/hierarchy.py", "Store & category totals", ":material/account_tree:",
     "Forecasts added up for each category and the whole shop."),
    ("views/assistant.py", "Ask Demandino", ":material/auto_awesome:",
     "Ask questions about a forecast in plain language."),
    ("views/accuracy.py", "Model accuracy", ":material/fact_check:",
     "How reliable the forecasts were when tested on sales the model hadn't seen."),
]
cols = st.columns(2)
for i, (path, title, icon, what) in enumerate(PAGES):
    with cols[i % 2], ui.card(f"help-{i}"):
        st.page_link(path, label=title, icon=icon)
        st.caption(what)

st.subheader("Using your own data")
st.markdown(
    "1. In the sidebar under **Data**, choose **My CSV** and upload a file with one row per product per day.\n"
    "2. It needs at least `date`, `product_id` and `units_sold`. Price, category and promotion columns make the "
    "forecast smarter. Download the template there to see the format.\n"
    "3. Two years of history is ideal; a few months works, but seasonal patterns need at least one full year.\n"
    "4. Sign in to keep your data, scenarios and settings in your private workspace.")

st.subheader("Words you'll see")
guide.glossary_cards()

st.subheader("Common questions")
FAQ = {
    "How accurate is it?": "In testing on real shop data, actual daily sales landed inside the predicted range "
                           "about 95 days out of 100. Single days are hard to predict exactly; weekly and monthly "
                           "totals are much steadier. See **Model accuracy** for the details.",
    "Why is the range so wide?": "Some products sell in irregular bursts. A wide range is the model being honest "
                                 "about that. Plan with weekly totals, and keep a bit more safety stock.",
    "What if I don't know my unit cost?": "Leave the default (55% of the price). Units and reorder dates don't "
                                          "depend on it; only profit figures and pricing advice do.",
    "Can I forecast a product I haven't launched yet?": "Yes. Choose **New product**, give its category and price, "
                                                        "and the model borrows patterns from similar products.",
    "Is my data safe?": "Uploaded files are processed in memory. Workspace data is private to your account and "
                        "secrets are encrypted. See the Terms & conditions for details.",
}
for q, a in FAQ.items():
    with st.expander(q):
        st.markdown(a)
