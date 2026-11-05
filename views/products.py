import pandas as pd
import streamlit as st

from demandino import catalog, db, ui
from demandino.state import app, raw_sales

A = app()
ui.page_header("Products", "Fix a name, move a product to another category, set a new price or take a product out of "
               "your forecasts. Your sales data isn't changed, so anything can be put back.", eyebrow="Settings")

dataset = A.catalog_id
edits = catalog.load(dataset)
source = raw_sales()
original = catalog.originals(source)
current = catalog.originals(catalog.apply(source, edits))
if st.session_state.pop("_catalog_saved", None):
    st.success("Saved. Forecasts now use these details.", icon=":material/check_circle:")
if A.user_id is None:
    st.caption(":material/info: You aren't signed in, so these changes last until you close this tab. Sign in to keep them.")

table = pd.DataFrame({"product_id": current["product_id"], "name": current["product_name"],
                      "category": current["category"], "price": (current["regular_price"] * A.fx_rate).round(2),
                      "remove": False})
st.caption(f"{len(table)} products. Click a cell to edit it, tick **Remove** to take a product out, then save.")
edited = st.data_editor(table, key=f"catalog_{dataset}_{catalog.digest(edits)}", hide_index=True, width="stretch",
                        disabled=["product_id"], num_rows="fixed", column_config={
    "product_id": st.column_config.TextColumn("ID", width="small"),
    "name": st.column_config.TextColumn("Name", required=True, max_chars=catalog.MAX_NAME),
    "category": st.column_config.TextColumn("Category", required=True, max_chars=catalog.MAX_NAME,
                                            help="Type an existing category or a new one."),
    "price": st.column_config.NumberColumn(f"Price ({ui.currency()})", required=True, min_value=0.01, format="%.2f",
                                           help="Applies from now on. Past sales keep the prices they sold at."),
    "remove": st.column_config.CheckboxColumn("Remove", width="small")})

differs = edited.drop(columns="product_id").astype(str).ne(table.drop(columns="product_id").astype(str)).any(axis=1)
to_remove = int(edited["remove"].sum())
if differs.any():
    st.info(f"Unsaved: {int(differs.sum()) - to_remove} to edit, {to_remove} to remove. Saving rebuilds the forecast, "
            "which can take a minute.", icon=":material/edit_note:")
if st.button("Save changes", type="primary", icon=":material/save:", disabled=not differs.any(), key="catalog_save"):
    try:
        new, log = catalog.merge(edits, original, edited, A.fx_rate)
        if not catalog.can_forecast(catalog.apply(source, new)):
            raise ValueError("That would leave no product with enough sales history to forecast. Keep at least one.")
    except ValueError as exc:
        st.error(str(exc))
    else:
        catalog.save(dataset, new)
        if A.user_id:
            for line in log[:50]:
                db.log_activity(A.user_id, "Products", line)
        st.session_state["_catalog_saved"] = True
        st.rerun()

gone = original[original["product_id"].isin(edits.get("removed", []))]
if not gone.empty:
    st.subheader("Removed products")
    for r in gone.itertuples():
        a, b = st.columns([4, 1], vertical_alignment="center")
        a.markdown(f"{ui.md_safe(r.product_name)} · `{ui.md_safe(r.product_id)}` · {ui.md_safe(r.category)}")
        if b.button("Put back", key=f"catalog_back_{r.product_id}", icon=":material/undo:", width="stretch"):
            catalog.save(dataset, {**edits, "removed": [p for p in edits["removed"] if p != r.product_id]})
            st.session_state.pop(f"pl_stock_{A.settings_key}", None)  # so its row comes back in Demand planning
            if A.user_id:
                db.log_activity(A.user_id, "Products", f"Put back product {r.product_id}")
            st.rerun()
if catalog.has_edits(edits):
    st.divider()
    with st.expander("Undo all my changes", icon=":material/restart_alt:"):
        st.caption(f"Goes back to your original data: {len(edits.get('changes', {}))} edited, "
                   f"{len(edits.get('removed', []))} removed.")
        if st.button("Undo all changes", key="catalog_reset", icon=":material/restart_alt:"):
            catalog.save(dataset, dict(catalog.EMPTY))
            st.session_state.pop(f"pl_stock_{A.settings_key}", None)
            if A.user_id:
                db.log_activity(A.user_id, "Products", "Undid all product changes")
            st.rerun()
