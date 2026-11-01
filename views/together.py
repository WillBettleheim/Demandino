from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandino import baskets, lifecycle, ui
from demandino.security import UploadError
from demandino.state import app

A = app()
REAL_BASKETS = Path(__file__).resolve().parent.parent / "data" / "uci_online_retail_baskets.csv"


@st.cache_data(show_spinner=False)
def bundled_baskets(mtime: float) -> pd.DataFrame:
    return pd.read_csv(REAL_BASKETS, dtype=str)


@st.cache_data(show_spinner="Finding products bought together…", max_entries=8)
def pair_table(key: str, order_lines: pd.DataFrame, names: dict) -> pd.DataFrame:
    return baskets.pairs(order_lines, names)


ui.page_header("Bought together",
               "Which products end up in the same order more often than chance — ideas for bundles, "
               "“goes well with” suggestions and joint promotions.", eyebrow="Analyze")

names = A.products.set_index("product_id")["product_name"].to_dict()
lines, source = None, ""
if A.data_key.startswith("real-") and REAL_BASKETS.exists():
    lines, source = bundled_baskets(REAL_BASKETS.stat().st_mtime), "the real store's orders (UCI Online Retail II)"
with st.expander("Use your own order lines" if lines is not None else "Upload your order lines",
                 expanded=lines is None, icon=":material/receipt_long:"):
    st.caption("This needs one row per product per order, e.g. an export of order items. Daily totals can't show "
               "which products were in the same order. Required columns: an order number (`order_id`, `invoice`…) "
               "and a product (`product_id`, `sku`…).")
    up = st.file_uploader("Order lines CSV", type=["csv"], key="bt_upload", label_visibility="collapsed")
    st.download_button("Download a template", baskets.template(), "order_lines_template.csv", "text/csv",
                       icon=":material/download:")
    if up is not None:
        try:
            lines, source = baskets.read_order_lines(up.getvalue()), f"your file {up.name}"
        except UploadError as exc:
            st.error(str(exc))
if lines is None:
    st.info("Upload order lines above, or pick **Real store** in the sidebar to see it with real orders.",
            icon=":material/shopping_basket:")
    st.stop()

pt = pair_table(f"{A.data_key}:{source}", lines, names)
n_orders = lines["order_id"].nunique()
strong = pt[pt["lift"] >= 2]
ui.spotlight(f"From {source}", str(len(strong)), "strong pairs",
             f"<b>{n_orders:,}</b> orders · <b>{lines['product_id'].nunique()}</b> products · pairs bought together "
             f"at least {baskets.MIN_TOGETHER} times: <b>{len(pt)}</b>",
             [(f"Strongest: {pt.iloc[0]['lift']:.1f}× chance", "acc")] if len(pt) else [("No repeat pairs yet", "")])
if pt.empty:
    st.stop()

stage = lifecycle.stages(A.sales).set_index("product_id")["stage"].astype(str).to_dict()
ui.finding_cards(baskets.ideas(pt) + baskets.drop_risks(pt, stage))

c1, c2 = st.columns([1.25, 1])
top = pd.concat([pt["a"], pt["b"]]).value_counts().head(14).index.tolist()
idx = {p: k for k, p in enumerate(top)}
grid = np.full((len(top), len(top)), np.nan)
for r in pt[pt["a"].isin(top) & pt["b"].isin(top)].itertuples():
    grid[idx[r.a], idx[r.b]] = grid[idx[r.b], idx[r.a]] = r.lift
labels = [str(names.get(p, p))[:26] for p in top]
fig = go.Figure(go.Heatmap(z=grid, x=labels, y=labels, colorscale=ui.SEQ, hoverongaps=False,
                           hovertemplate="%{y} + %{x}: %{z:.1f}× chance<extra></extra>", colorbar=dict(title="×")))
ui.style(fig, "How much more often pairs are bought together than by chance", height=480, legend=False, hover="closest")
fig.update_xaxes(tickangle=-40)
ui.show(fig, c1)

with c2:
    pick = st.selectbox("Customers who bought…", sorted(set(pt["a"]) | set(pt["b"]), key=lambda p: names.get(p, p)),
                        format_func=lambda p: names.get(p, p), key="bt_pick")
    pa = baskets.partners(pt, pick)
    st.caption("…also bought (share of its orders):")
    st.dataframe(pa[["other_name", "chance", "lift", "together"]], hide_index=True, width="stretch", column_config={
        "other_name": "Product", "chance": st.column_config.ProgressColumn("In its orders", format="percent",
                                                                          min_value=0, max_value=1),
        "lift": st.column_config.NumberColumn("× chance", format="%.1f"),
        "together": st.column_config.NumberColumn("Orders", format="%,d")})

with st.expander(f"All pairs ({len(pt)})", icon=":material/table_rows:"):
    st.dataframe(pt.drop(columns=["a", "b"]), hide_index=True, width="stretch", column_config={
        "a_name": "Product A", "b_name": "Product B", "together": st.column_config.NumberColumn("Orders", format="%,d"),
        "share_of_orders": st.column_config.NumberColumn("Share of all orders", format="percent"),
        "a_to_b": st.column_config.NumberColumn("A's orders with B", format="percent"),
        "b_to_a": st.column_config.NumberColumn("B's orders with A", format="percent"),
        "lift": st.column_config.NumberColumn("× chance", format="%.1f")})
st.caption("“× chance” compares with customers picking products independently. Orders are counted among "
           "those containing at least one of these products.")
