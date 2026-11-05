"""Smoke tests: every page renders without errors."""
import pytest
from streamlit.testing.v1 import AppTest

from demandino.legal import TERMS_VERSION

PAGES = ["views/forecast.py", "views/planning.py", "views/advisor.py", "views/explore.py", "views/overview.py",
         "views/accuracy.py", "views/alerts.py", "views/clearance.py", "views/targets.py", "views/datacheck.py",
         "views/home.py", "views/welcome.py", "views/compare.py", "views/products.py", "views/lifecycle.py", "views/together.py", "views/changes.py", "views/lostsales.py",
         "views/whatif.py", "views/budget.py", "views/inbox.py", "views/help.py", "views/terms.py"]


def _new_app(accept: bool = True) -> AppTest:
    at = AppTest.from_file("../app.py", default_timeout=300)
    if accept:
        at.session_state["terms_accepted"] = TERMS_VERSION
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = _new_app()
    at.run()
    at.switch_page(page).run()
    assert not at.exception, at.exception


def _text(at) -> str:
    """All rendered markdown and HTML on the page."""
    return " ".join([m.value for m in at.markdown] + [h.proto.body for h in at.get("html")])


def _forecast_app():
    at = _new_app()
    at.run()
    at.switch_page("views/forecast.py").run()
    return at


def test_new_product_and_promo_and_scenarios():
    at = _forecast_app()
    at.segmented_control(key="fc_mode").set_value("New product").run()
    assert not at.exception, at.exception
    assert "cold-start" in _text(at)
    at.selectbox(key="fc_promo").set_value("Recurring").run()
    assert not at.exception, at.exception
    assert "Promotion impact" in _text(at)
    next(b for b in at.button if b.label == "Save").click().run()
    assert not at.exception, at.exception
    assert len(at.session_state["scenarios"]) == 1


def test_similar_product_basis():
    at = _forecast_app()
    at.segmented_control(key="fc_mode").set_value("New product").run()
    at.radio(key="fc_basis").set_value("similar").run()
    assert not at.exception, at.exception
    assert "comparable product" in _text(at)


@pytest.mark.parametrize("page", PAGES)
def test_real_store_pages_render(page):
    at = _new_app()
    at.run()
    at.switch_page("views/home.py").run()
    at.segmented_control(key="data_source").set_value("Real store").run()
    at.switch_page(page).run()
    assert not at.exception, at.exception


def test_terms_gate_blocks_until_accepted():
    at = _new_app(accept=False)
    at.run()
    at.switch_page("views/home.py").run()
    assert not at.exception
    assert any("Terms" in h.proto.body for h in at.get("html"))
    assert "home_pid" not in at.session_state  # the home page did not run
    continue_btn = next(b for b in at.button if b.label == "Continue to Demandino")
    assert continue_btn.disabled
    at.checkbox(key="terms_checkbox").check().run()
    next(b for b in at.button if b.label == "Continue to Demandino").click().run()
    assert not at.exception
    assert at.session_state["terms_accepted"] == TERMS_VERSION
    assert "home_pid" in at.session_state


def test_advisor_uses_forecast_page_product():
    at = _forecast_app()
    at.switch_page("views/advisor.py").run()
    assert not at.exception, at.exception
    assert "Growth advisor" in _text(at)


def test_theme_toggle_emits_switch_script():
    at = _forecast_app()
    at.segmented_control(key="dm_theme").set_value("light").run()
    assert not at.exception, at.exception
    assert any("stActiveTheme" in h.proto.body for h in at.get("html"))


SIGNED_IN_PAGES = ["views/workspace.py", "views/connections.py", "views/orders.py", "views/tracking.py",
                   "views/hierarchy.py", "views/assistant.py", "views/targets.py", "views/inbox.py",
                   "views/home.py", "views/history.py", "views/security.py"]


@pytest.mark.parametrize("page", SIGNED_IN_PAGES)
def test_workspace_pages_signed_in(page):
    from demandino import db
    import secrets as _s
    uid = db.create_user("u" + _s.token_hex(4), "a very long password")
    at = _new_app()
    at.session_state["user_id"] = uid
    at.run()
    at.switch_page(page).run()
    assert not at.exception, at.exception


def test_share_link_renders_without_terms():
    from demandino import db
    uid = db.create_user("sharer_test", "a very long password")
    summary = {"product": "Mug", "horizon_days": 30, "forecast_start": "2026-11-01", "expected_units": 120,
               "range_95": [80, 160], "revenue": 1200, "gross_profit": 500, "confidence": "High",
               "category": "Kitchen", "weekly_units": {"2026-11-02": 30.0, "2026-11-09": 31.0}, "findings": ["**A:** b"]}
    token = db.create_share(uid, "Mug forecast", {"summary": summary, "currency": "$"})
    at = AppTest.from_file("../app.py", default_timeout=60)
    at.query_params["share"] = token
    at.run()
    assert not at.exception, at.exception
    assert any("Mug forecast" in h.proto.body for h in at.get("html"))


def test_welcome_guide_summary_and_start_over():
    at = _forecast_app()
    text = _text(at)
    assert "Three steps to your first forecast" in text
    assert "In short" in text and "Expect to sell about" in text
    at.button(key="welcome_ok").click().run()
    assert "Three steps to your first forecast" not in _text(at)
    at.select_slider(key="fc_h").set_value(30).run()
    at.button(key="fc_reset").click().run()
    assert not at.exception, at.exception
    assert at.select_slider(key="fc_h").value == 90


def test_alert_opens_product_and_keeps_the_data_source():
    at = _forecast_app()
    at.segmented_control(key="data_source").set_value("Real store").run()
    at.switch_page("views/alerts.py").run()
    at.segmented_control(key="al_weeks").set_value(8).run()
    assert not at.exception, at.exception
    opens = [b for b in at.button if b.key and b.key.startswith("al_open_")]
    assert opens, "the real store data should have at least one flagged product"
    pid = opens[0].key.removeprefix("al_open_")
    opens[0].click().run()
    assert at.segmented_control(key="data_source").value == "Real store"
    assert at.selectbox(key="fc_pid").value == pid


def test_tracking_an_adjusted_forecast_keeps_the_model_version():
    import secrets as _s

    import numpy as np
    import pandas as pd

    from demandino import db
    from demandino.scenario import ForecastInputs
    uid = db.create_user("t" + _s.token_hex(4), "a very long password")
    at = _new_app()
    at.session_state["user_id"] = uid
    at.run()
    at.switch_page("views/home.py").run()
    start = pd.Timestamp(at.session_state["_app"].last_date) + pd.Timedelta(days=1)
    at.session_state["load_scenario"] = ForecastInputs(
        False, "P001", "Earbuds", "Electronics", 250.0, 140.0, 30, start,
        events=(("Campaign", (start + pd.Timedelta(days=3)).date(), (start + pd.Timedelta(days=9)).date(), 50.0),))
    at.switch_page("views/forecast.py").run()
    next(b for b in at.button if b.label == "Track this forecast").click().run()
    assert not at.exception, at.exception
    p = db.list_forecasts(uid)[0]["payload"]
    assert p["adjustments"][0][0] == "Campaign"
    ratio = np.asarray(p["mean"]) / np.maximum(np.asarray(p["base_mean"]), 1e-9)
    assert np.allclose(ratio[3:10], 1.5, atol=0.01) and np.allclose(ratio[:3], 1.0, atol=0.01)
    at.switch_page("views/tracking.py").run()
    assert not at.exception, at.exception


def test_orders_page_shows_supplier_delivery_times():
    import secrets as _s

    from demandino import db
    uid = db.create_user("s" + _s.token_hex(4), "a very long password")
    for k, days in enumerate([10, 12, 11]):
        po = db.create_po(uid, [{"product_id": "P001", "supplier": "Acme", "lead_time": 7, "order_cost": 100.0}])
        db.decide_po(uid, po, "Ordered", "buyer")
        db.receive_po(uid, po, 1_700_000_000.0 + k * 3e6, 1_700_000_000.0 + k * 3e6 + days * 86400, "buyer")
    at = _new_app()
    at.session_state["user_id"] = uid
    at.run()
    at.switch_page("views/orders.py").run()
    assert not at.exception, at.exception
    table = at.dataframe[0].value
    assert table.loc[0, "supplier"] == "Acme" and table.loc[0, "avg_days"] == 11
    at.switch_page("views/planning.py").run()
    assert not at.exception, at.exception


def test_planning_suggests_transfers_between_warehouses():
    import pandas as pd

    from demandino.planning import default_stock
    at = _forecast_app()
    state = at.session_state["_app"]
    one = default_stock(state.sales)
    two = pd.concat([one.assign(warehouse="North", demand_share=50.0, stock=0),
                     one.assign(warehouse="South", demand_share=50.0, stock=one["stock"] * 10)], ignore_index=True)
    at.session_state[f"pl_stock_{state.data_key}"] = two
    at.switch_page("views/planning.py").run()
    assert not at.exception, at.exception
    assert any("avoids about" in s.value for s in at.success)


@pytest.mark.parametrize("page", ["views/lifecycle.py", "views/changes.py", "views/lostsales.py", "views/whatif.py",
                                  "views/targets.py", "views/alerts.py", "views/clearance.py"])
def test_analysis_pages_offer_a_csv_download(page):
    at = _new_app()
    at.run()
    at.switch_page(page).run()
    assert not at.exception, at.exception
    assert any(b.proto.label.startswith("Download") for b in at.get("download_button"))


def test_welcome_page_opens_without_terms_data_or_sidebar():
    at = _new_app(accept=False)
    at.run()
    assert not at.exception, at.exception
    assert "before it sells" in _text(at) and "Terms &amp; conditions" not in _text(at)
    assert len(at.sidebar) == 0 and "_app" not in at.session_state


def test_welcome_buttons_lead_to_the_app():
    at = _new_app()
    at.run()
    at.button(key="dml_open").click().run()
    assert not at.exception, at.exception
    assert "Your shop at a glance" in _text(at)


def test_welcome_script_survives_the_html_sanitiser():
    import re

    from demandino import landing
    assert not re.search(r"<[/\\w!]", landing.JS_BODY)  # Streamlit removes a whole script if it holds anything like a tag
    assert "__ICONS__" not in landing.JS


def test_removed_and_renamed_products_flow_through_the_whole_app():
    import secrets as _s

    from demandino import db
    uid = db.create_user("p" + _s.token_hex(4), "a very long password")
    at = _new_app()
    at.session_state["user_id"] = uid
    at.run()
    at.switch_page("views/planning.py").run()  # leaves a stock table that still lists every product
    A = at.session_state["_app"]
    db.set_setting(uid, f"targets:{A.base_key}", [
        {"label": "Gone", "products": ["P002"], "start": "2020-01-01", "end": "2020-02-01", "metric": "units", "target": 5.0},
        {"label": "Part", "products": ["P001", "P002"], "start": "2020-01-01", "end": "2020-02-01", "metric": "units", "target": 5.0}])
    db.set_setting(uid, "catalog:sample", {"changes": {"P001": {"name": "Renamed Earbuds"}}, "removed": ["P002"]})
    del at.session_state["catalog:sample"]  # read the saved changes again
    for page in ("views/products.py", "views/home.py", "views/planning.py", "views/budget.py", "views/clearance.py",
                 "views/targets.py", "views/inbox.py", "views/lifecycle.py", "views/whatif.py", "views/forecast.py"):
        at.switch_page(page).run()
        assert not at.exception, (page, at.exception)
        assert "P002" not in set(at.session_state["_app"].sales["product_id"])
    assert "-cat" in at.session_state["_app"].data_key and "-cat" not in at.session_state["_app"].settings_key
    at.switch_page("views/products.py").run()
    assert "Removed products" in " ".join(s.value for s in at.subheader)
    next(b for b in at.button if b.key == "catalog_back_P002").click().run()
    assert "P002" in set(at.session_state["_app"].sales["product_id"])


def test_example_numbers_are_called_examples():
    at = _forecast_app()
    assert "An example stock of 500 units" in _text(at) and "a guess until you enter your unit cost" in _text(at)
    at.number_input(key="fc_stock").set_value(100_000).run()
    assert "Your 100,000 units would cover" in _text(at) and "An example stock" not in _text(at)
