"""Smoke tests: every page renders without errors."""
import pytest
from streamlit.testing.v1 import AppTest

from demandino.legal import TERMS_VERSION

PAGES = ["views/forecast.py", "views/planning.py", "views/advisor.py", "views/explore.py", "views/overview.py",
         "views/accuracy.py", "views/terms.py"]


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
