"""Unit tests for accounts, connectors, ordering rules, model additions, exports and scenarios."""
import numpy as np
import pandas as pd
import pytest

from demandino import db, integrations as ig
from demandino.analysis import elasticities, promo_spillover
from demandino.data import generate_sample_data
from demandino.model import DemandForecaster, demand_class, detect_stockouts, holiday_dates
from demandino.pdf_report import scenario_pdf
from demandino.planning import apply_order_rules, cash_flow, default_stock, portfolio_plan
from demandino.scenario import ForecastInputs, compute, inputs_from_json, inputs_to_json


@pytest.fixture(scope="module")
def sales():
    return generate_sample_data(end_date="2026-10-28", years=2)


@pytest.fixture(scope="module")
def fc(sales):
    return DemandForecaster(sales, country="US", max_iter=60).fit(calibrate=False)


# --------------------------------------------------------------- accounts
def test_accounts_hash_lock_and_sessions():
    uid = db.create_user("alice_test", "correct horse battery")
    with pytest.raises(db.AuthError):
        db.create_user("alice_test", "another long password")
    with pytest.raises(db.AuthError):
        db.create_user("bob", "short")
    assert db.authenticate("alice_test", "correct horse battery") == uid
    for _ in range(db.MAX_FAILED):
        with pytest.raises(db.AuthError):
            db.authenticate("alice_test", "wrong password!!")
    with pytest.raises(db.AuthError, match="Too many"):
        db.authenticate("alice_test", "correct horse battery")  # locked even with the right password
    token = db.new_session(uid)
    assert db.session_user(token) == uid and db.session_user("nope") is None
    db.end_session(token)
    assert db.session_user(token) is None


def test_secrets_are_encrypted_at_rest():
    uid = db.create_user("carol_test", "a very long password")
    db.set_secret(uid, "slack_webhook", "https://hooks.slack.com/services/T/B/SECRET")
    with db.connect() as con:
        raw = con.execute("SELECT value FROM secrets WHERE user_id = ?", (uid,)).fetchone()["value"]
    assert b"SECRET" not in raw
    assert db.get_secret(uid, "slack_webhook").endswith("SECRET")


def test_datasets_are_private_per_user(sales):
    a = db.create_user("dave_test", "a very long password")
    b = db.create_user("erin_test", "a very long password")
    ds = db.save_dataset(a, "mine", sales.head(500), "upload")
    assert db.load_dataset(a, ds) is not None
    assert db.load_dataset(b, ds) is None


# ------------------------------------------------------------- connectors
@pytest.mark.parametrize("url", ["http://docs.google.com/spreadsheets/d/x", "https://evil.com/spreadsheets/d/x",
                                 "https://127.0.0.1/x", "https://user:pw@docs.google.com/x"])
def test_google_sheet_rejects_unsafe_urls(url):
    with pytest.raises(ig.ConnectorError):
        ig.google_sheet(url)


def test_shopify_and_slack_only_allow_their_hosts():
    with pytest.raises(ig.ConnectorError):
        ig.shopify("evil.com", "token", pd.Timestamp("2025-01-01"))
    with pytest.raises(ig.ConnectorError):
        ig.send_slack("https://example.com/hook", "hi")
    with pytest.raises(ig.ConnectorError):
        ig.send_email("smtp.example.com", 587, "u", "p", "not-an-email", "a@b.co", "s", "b")


def test_amazon_report_parsing():
    csv = ("purchase-date\tsku\tproduct-name\tquantity\titem-price\torder-status\n"
           + "".join(f"2026-0{1 + i // 28}-{1 + i % 28:02d}T10:00:00+00:00\tSKU1\tMug\t2\t20.00\tShipped\n"
                     for i in range(100))
           + "2026-01-02T10:00:00+00:00\tSKU1\tMug\t5\t50.00\tCancelled\n").encode()
    sales = ig.amazon_report(csv)
    assert sales["units_sold"].sum() == 200
    assert np.isclose(sales.loc[sales["units_sold"] > 0, "price"].mean(), 10.0)


# --------------------------------------------------------- ordering rules
def test_order_rules_round_to_cases_and_moq():
    assert apply_order_rules(0, 50, 12) == 0
    assert apply_order_rules(5, 0, 12) == 12
    assert apply_order_rules(30, 0, 12) == 36
    assert apply_order_rules(10, 50, 12) == 60


def test_budget_defers_and_cash_flow(fc, sales):
    stock = default_stock(sales).assign(stock=0, unit_cost=10.0)
    plan, daily = portfolio_plan(fc, 0.15, stock, 30, 14, 0.95, budget=500.0)
    kept = plan[plan["status"] != "Deferred (budget)"]
    assert kept["order_cost"].sum() <= 500.0
    assert (plan["status"] == "Deferred (budget)").any()
    cf = cash_flow(plan, daily)
    assert {"money_in", "money_out", "cumulative"} <= set(cf.columns)


# --------------------------------------------------------- model features
def test_stockout_detection_and_demand_class():
    units = np.r_[np.full(30, 10.0), np.zeros(5), np.full(30, 10.0)]
    flags, demand = detect_stockouts(units)
    assert flags[30:35].all() and not flags[:30].any()
    assert np.allclose(demand[30:35], 10.0, atol=1)
    assert demand_class(np.full(100, 5.0))[0] == "Smooth"
    assert demand_class(np.r_[np.zeros(80), np.full(20, 3.0)])[0] == "Intermittent"


def test_holidays_and_explainer(fc, sales):
    assert pd.Timestamp("2026-07-04") in holiday_dates("US", range(2026, 2027))
    assert "holiday" in fc.features or "days_to_holiday" in fc.features
    ctx = fc.context_for("P001")
    h = 30
    reg = float(ctx.regular[-1])
    target, avg, parts = fc.explain_day(ctx, fc.sales["date"].max() + pd.Timedelta(days=1), h, np.full(h, reg),
                                        np.full(h, reg), np.zeros(h, int), day=5)
    assert abs(avg + sum(v for _, v in parts) - target) < 1e-6


def test_drivers_become_features(sales):
    days = pd.date_range(sales["date"].min(), sales["date"].max() + pd.Timedelta(days=60))
    drivers = pd.DataFrame({"ad_spend": np.random.default_rng(0).uniform(0, 100, len(days))}, index=days)
    fc = DemandForecaster(sales, drivers=drivers, max_iter=30).fit(calibrate=False)
    assert "drv_ad_spend" in fc.features
    res = fc.forecast_product("P001", 14)
    assert np.isfinite(res.mean).all()


# ------------------------------------------------------ scenarios & exports
def test_scenario_json_roundtrip_pdf_and_spillover(fc, sales):
    inp = ForecastInputs(False, "P001", "Earbuds", "Electronics", 250.0, 140.0, 30,
                         fc.sales["date"].max() + pd.Timedelta(days=1), promo_kind="Recurring",
                         events=(("Push", pd.Timestamp("2026-11-05").date(), pd.Timestamp("2026-11-08").date(), 25.0),))
    again = inputs_from_json(inputs_to_json(inp))
    assert again == inp
    r = compute(fc, fc.sales, elasticities(sales), 0.15, inp)
    pdf = scenario_pdf(r, "€")
    assert pdf[:4] == b"%PDF" and len(pdf) > 2000
    sp = promo_spillover(sales, "P001")
    assert set(sp["product_id"]) and sp["effect"].abs().max() < 1
