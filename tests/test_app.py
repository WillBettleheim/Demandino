"""Smoke tests: every page renders without errors."""
import pytest
from streamlit.testing.v1 import AppTest

from demandino.legal import TERMS_VERSION

PAGES = ["views/forecast.py", "views/planning.py", "views/advisor.py", "views/explore.py", "views/overview.py",
         "views/accuracy.py", "views/alerts.py", "views/clearance.py", "views/targets.py", "views/datacheck.py",
         "views/lifecycle.py", "views/together.py", "views/changes.py", "views/lostsales.py",
         "views/whatif.py", "views/help.py", "views/terms.py"]


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
    at.segmented_control(key="data_source").set_value("Real store").run()
    at.switch_page(page).run()
    assert not at.exception, at.exception


def test_terms_gate_blocks_until_accepted():
    at = _new_app(accept=False)
    at.run()
    assert not at.exception
    assert any("Terms" in h.proto.body for h in at.get("html"))
    assert "fc_mode" not in at.session_state  # the forecast page did not run
    continue_btn = next(b for b in at.button if b.label == "Continue to Demandino")
    assert continue_btn.disabled
    at.checkbox(key="terms_checkbox").check().run()
    next(b for b in at.button if b.label == "Continue to Demandino").click().run()
    assert not at.exception
    assert at.session_state["terms_accepted"] == TERMS_VERSION
    assert "fc_mode" in at.session_state


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
                   "views/hierarchy.py", "views/assistant.py", "views/targets.py"]


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
    start = pd.Timestamp(at.session_state["_app"].last_date) + pd.Timedelta(days=1)
    at.session_state["load_scenario"] = ForecastInputs(
        False, "P001", "Earbuds", "Electronics", 250.0, 140.0, 30, start,
        events=(("Campaign", (start + pd.Timedelta(days=3)).date(), (start + pd.Timedelta(days=9)).date(), 50.0),))
    at.run()
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
