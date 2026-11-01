"""Unit tests for accounts, connectors, ordering rules, model additions, exports and scenarios."""
import numpy as np
import pandas as pd
import pytest

from demandino import adjustments, clearance, db, integrations as ig, monitor, quality, targets
from demandino.analysis import elasticities, elasticity_effect, promo_spillover
from demandino.data import generate_sample_data, template_csv
from demandino.model import (DemandForecaster, backtest, demand_class, detect_bulk_orders, detect_stockouts,
                             holiday_dates)
from demandino.pdf_report import scenario_pdf
from demandino.planning import apply_order_rules, cash_flow, default_stock, portfolio_plan
from demandino.security import UploadError
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


# ------------------------------------------- adjustments, drift and clearance
def test_adjustment_scoring_rewards_the_better_forecast(sales):
    pid = "P001"
    dates = pd.date_range(sales["date"].max() - pd.Timedelta(days=13), periods=14)
    actual = sales[sales["product_id"] == pid].set_index("date")["units_sold"].reindex(dates).to_numpy()
    model = actual * 0.5  # the model badly under-forecast…
    yours = model.copy()
    yours[:7] = actual[:7]  # …your +100% adjustment fixed the first week only
    log = {"id": 1, "product_id": pid, "product_name": "Earbuds",
           "payload": {"dates": [d.strftime("%Y-%m-%d") for d in dates], "mean": yours.tolist(),
                       "base_mean": model.tolist(), "lower": [0] * 14, "upper": [0] * 14,
                       "adjustments": [["Campaign", str(dates[0].date()), str(dates[6].date()), 100.0],
                                       ["Ignored", str(dates[7].date()), str(dates[13].date()), 0.0]]}}
    scored = adjustments.score([log], sales)
    assert scored.set_index("adjustment").loc["Campaign", "verdict"] == "Helped"
    assert scored.set_index("adjustment").loc["Ignored", "verdict"] == "No difference"
    assert adjustments.summary(scored)["error_change"] < 0


def test_drift_flags_a_product_whose_sales_jumped(sales):
    jumped = sales.copy()
    recent = (jumped["product_id"] == "P001") & (jumped["date"] > jumped["date"].max() - pd.Timedelta(days=28))
    jumped.loc[recent, "units_sold"] *= 3
    fc = DemandForecaster(jumped, max_iter=60).fit(calibrate=False)
    fc.conformal_q, fc.level_uncertainty = 0.2, 0.15
    old, cutoff = monitor.holdout_model(fc)
    rep = monitor.drift_report(old, cutoff).set_index("product_id")
    assert rep.loc["P001", "flagged"] and rep.loc["P001", "direction"] == "Above"
    assert rep["flagged"].mean() < 0.5  # most untouched products stay on track
    assert any("P001" in line or rep.loc["P001", "product_name"] in line for line in monitor.drift_lines(rep.reset_index()))


def test_clearance_marks_down_only_overstocked_products(fc, sales):
    stock = default_stock(sales)
    stock["stock"] = 0
    stock.loc[stock["product_id"] == "P001", "stock"] = 100_000
    plan = clearance.clearance_plan(fc, stock, {"P001": -2.0}, window=60, target_days=90).set_index("product_id")
    assert plan.loc["P001", "slow"] and 0 < plan.loc["P001", "discount"] <= 0.5
    assert plan.loc["P001", "cash_freed"] > 0
    assert not plan.drop(index="P001")["slow"].any()
    assert "Mark down" in clearance.advice(plan.loc["P001"])


def test_bulk_orders_are_flagged_and_added_back_as_an_allowance(sales):
    units = np.r_[np.full(60, 10.0), [200.0], np.full(60, 10.0)]
    flags, normal = detect_bulk_orders(units)
    assert flags[60] and flags.sum() == 1 and normal[60] < 20
    promo = np.zeros(len(units), int)
    promo[60] = 1
    assert not detect_bulk_orders(units, promo=promo)[0].any()  # a promotion explains the spike

    spiked = sales.copy()
    day = spiked["date"].max() - pd.Timedelta(days=100)
    row = (spiked["product_id"] == "P001") & (spiked["date"] == day)
    spiked.loc[row, ["units_sold", "promo"]] = [5000, 0]
    fc = DemandForecaster(spiked, max_iter=40).fit(calibrate=False)
    marked = fc.sales.loc[row].iloc[0]
    assert marked["bulk"] and marked["demand"] < 200
    assert 4000 / 365 < fc.context_for("P001").bulk_allowance < 5000 / 365 + 1

    kept = DemandForecaster(spiked, max_iter=40, bulk_keep=frozenset({("P001", day.strftime("%Y-%m-%d"))}))
    assert not kept.sales.loc[row].iloc[0]["bulk"]


def test_elasticity_wording_stays_true_for_positive_values():
    assert "lifted units by about 23%" in elasticity_effect(-2.0)
    assert "has not lifted units" in elasticity_effect(0.36)


def test_target_progress_adds_actuals_and_forecast(fc):
    last = fc.sales["date"].max()
    start, end = last - pd.Timedelta(days=9), last + pd.Timedelta(days=20)
    p = targets.progress(fc, ["P001", "P002"], start, end, "units")
    sold = fc.sales[fc.sales["product_id"].isin(["P001", "P002"]) & (fc.sales["date"] >= start)]["units_sold"].sum()
    assert p["days_done"] == 10 and p["days_left"] == 20
    assert np.isclose(p["actual"], sold) and p["forecast_rest"] > 0
    assert np.isclose(p["projected"], p["actual"] + p["forecast_rest"])
    assert targets.chance(p, p["projected"] * 0.5) > 0.9 > 0.1 > targets.chance(p, p["projected"] * 2)
    covered = targets.levers(p, p["projected"] * 0.9, {}, 0.3)
    assert len(covered) == 1 and "to spare" in covered[0]
    gap = targets.levers(p, p["projected"] * 1.1, {"P001": -2.0, "P002": -2.0}, 0.5)
    assert any(line.startswith("**Price:**") for line in gap) and any("Promotions" in line for line in gap)


def test_default_period_is_the_month_or_quarter_ahead():
    assert targets.default_period(pd.Timestamp("2026-10-31"), "Month") == (pd.Timestamp("2026-11-01"),
                                                                         pd.Timestamp("2026-11-30"))
    assert targets.default_period(pd.Timestamp("2026-11-10"), "Quarter") == (pd.Timestamp("2026-10-01"),
                                                                           pd.Timestamp("2026-12-31"))


# ------------------------------------------------------------------- data check
def test_data_check_finds_each_problem_in_the_messy_example():
    import io
    raw = pd.read_csv(io.StringIO(quality.messy_example()))
    found = {i.code: i for i in quality.check(raw, today=pd.Timestamp("2026-07-01"))}
    assert {"unreadable", "future", "duplicates", "negative", "prices", "gaps", "short"} <= set(found)
    assert found["gaps"].count == 20 and found["short"].severity == "info"


def test_data_check_fixes_change_the_data_as_chosen():
    import io
    raw = pd.read_csv(io.StringIO(quality.messy_example()))
    today = pd.Timestamp("2026-07-01")
    default = quality.apply_fixes(raw, today=today)
    assert default["date"].max() <= today and default["units_sold"].min() >= 0
    assert "stockout" not in default
    assert default["price"].max() < 1000  # the ×100 price was replaced by the usual price
    chosen = quality.apply_fixes(raw, {"gaps": "stockout", "prices": "keep"}, today=today)
    assert chosen["stockout"].sum() == 20 and chosen["price"].max() > 1000
    assert quality.apply_fixes(raw, {"future": "keep"}, today=today)["date"].max().year == 2062
    summed = default.set_index(["product_id", "date"])["units_sold"]
    last = quality.apply_fixes(raw, {"duplicates": "last"}, today=today).set_index(["product_id", "date"])["units_sold"]
    assert summed.sum() > last.sum()  # adding up duplicated rows counts more than keeping one


def test_clean_file_has_no_issues():
    import io
    raw = pd.read_csv(io.StringIO(template_csv()))
    assert [i.code for i in quality.check(raw, today=pd.Timestamp("2026-07-01")) if i.severity == "warn"] == []


def test_short_history_trains_and_back_tests(sales):
    short = sales[sales["date"] > sales["date"].max() - pd.Timedelta(days=170)]
    fc = DemandForecaster(short, max_iter=40).fit()
    assert np.isfinite(fc.forecast_product("P001", 30).mean).all()
    bt = backtest(short, max_iter=40)
    assert 0 <= bt["metrics"]["Daily miss rate"] <= 1


# ------------------------------------------------------------ shelf life
def test_expiry_check_flags_waste_and_caps_orders():
    from demandino.planning import expiry_check
    daily = np.full(60, 10.0)
    assert expiry_check(daily, 500, 0, None, 5, 0)["order_cap"] is None  # doesn't expire
    out = expiry_check(daily, 500, 21, None, None, 0)
    assert out["expiring_units"] == 290  # 210 sell in 21 days
    assert expiry_check(daily, 500, 21, 7, None, 0)["expiring_units"] == 430  # current stock has 7 days left
    assert expiry_check(daily, 100, 21, None, 20, 0)["order_cap"] == 210


def test_plan_reorders_earlier_and_caps_orders_for_perishables(fc, sales):
    stock = default_stock(sales)
    stock = stock[stock["product_id"] == "P001"].assign(stock=3000, shelf_life=0, moq=0, case_pack=1)
    keep, _ = portfolio_plan(fc, 0.15, stock, 60, 14, 0.95)
    fresh, _ = portfolio_plan(fc, 0.15, stock.assign(shelf_life=14), 60, 14, 0.95)
    assert keep.loc[0, "expiring_units"] == 0 and fresh.loc[0, "expiring_units"] > 0
    assert fresh.loc[0, "days_of_cover"] < keep.loc[0, "days_of_cover"]
    forced, _ = portfolio_plan(fc, 0.15, stock.assign(stock=0, shelf_life=7, moq=5000), 60, 14, 0.95)
    assert forced.loc[0, "order_waste"] > 0
    assert any("EXPIRING" in line for line in ig.alert_lines(fresh))


# ------------------------------------------------------------- life stages
def _series(levels: list[float], days_each: int = 120) -> pd.DataFrame:
    units = np.concatenate([np.full(days_each, lv) for lv in levels])
    return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=len(units)), "units_sold": units})


def test_life_stages_follow_the_sales_curve():
    from demandino import lifecycle
    assert lifecycle.classify(_series([5.0], 60))["stage"] == "Launching"
    assert lifecycle.classify(_series([10, 10, 10, 10, 10, 10]))["stage"] == "Mature"
    assert lifecycle.classify(_series([10, 10, 10, 14, 14, 14]))["stage"] == "Growing"
    assert lifecycle.classify(_series([10, 10, 10, 6, 6, 6]))["stage"] == "Declining"
    assert lifecycle.classify(_series([10, 10, 10, 2, 2, 2]))["stage"] == "End of life"


def test_life_stages_for_every_product(sales):
    from demandino import lifecycle
    st_df = lifecycle.stages(sales)
    assert len(st_df) == sales["product_id"].nunique()
    assert set(st_df["stage"].astype(str)) <= set(lifecycle.STAGES)
    assert st_df["advice"].str.len().min() > 20


# ----------------------------------------------------------------- weather
def test_weather_combines_archive_forecast_and_climate_normal(monkeypatch):
    from demandino import weather

    def fake(url, params):
        if url == weather.GEOCODE:
            return {"results": [{"name": "London", "country": "United Kingdom", "latitude": 51.5, "longitude": -0.1}]}
        if url == weather.ARCHIVE:
            days = pd.date_range(params["start_date"], params["end_date"])
        else:
            days = pd.date_range("2026-10-25", periods=23)
        temp = 10 + 8 * np.sin(2 * np.pi * (days.day_of_year - 110) / 365.25)
        return {"daily": {"time": days.strftime("%Y-%m-%d").tolist(), "temperature_2m_mean": temp.tolist(),
                          "precipitation_sum": [1.0] * len(days)}}

    monkeypatch.setattr(weather, "_json", fake)
    assert weather.find_place("London")["country"] == "United Kingdom"
    w = weather.daily_weather(51.5, -0.1, pd.Timestamp("2024-01-01"), pd.Timestamp("2026-10-31"),
                              today=pd.Timestamp("2026-11-01"))
    assert w.index.min() == pd.Timestamp("2024-01-01")
    assert w.index.max() == pd.Timestamp("2026-10-31") + pd.Timedelta(days=weather.FUTURE_DAYS)
    assert not w.isna().any().any()
    spring = w.loc["2027-04-01":"2027-05-01", "temperature"].mean()
    winter = w.loc["2027-01-01":"2027-01-31", "temperature"].mean()
    assert spring > winter  # beyond the forecast, the climate normal keeps the seasons


def test_weather_only_talks_to_open_meteo():
    from demandino import weather
    with pytest.raises(ig.ConnectorError):
        ig._check_url("https://evil.example/v1/archive", hosts=weather.HOSTS)
    with pytest.raises(ig.ConnectorError):
        weather.find_place(" ")


# ---------------------------------------------------------- bought together
def _orders() -> pd.DataFrame:
    """300 orders: B is almost always bought with A (A also sells alone); C turns up now and then."""
    rows = []
    for k in range(300):
        rows.append((k, "A"))
        if k % 3:
            rows.append((k, "B"))
        if k % 5 == 0:
            rows.append((k, "C"))
    return pd.DataFrame(rows, columns=["order_id", "product_id"]).astype(str)


def test_pairs_find_products_bought_together():
    from demandino import baskets
    pt = baskets.pairs(_orders(), {"A": "Mug", "B": "Coaster"}, min_together=10)
    ab = pt[(pt["a"] == "A") & (pt["b"] == "B")].iloc[0]
    assert ab["a_name"] == "Mug" and ab["together"] > 100 and ab["b_to_a"] == 1.0
    partners = baskets.partners(pt, "B")
    assert partners.iloc[0]["other"] == "A"
    assert baskets.ideas(baskets.pairs(_orders(), min_together=10)) != []


def test_drop_risk_warns_about_declining_products_in_shared_orders():
    from demandino import baskets
    pt = pd.DataFrame({"a": ["A"], "a_name": ["Mug"], "b": ["B"], "b_name": ["Coaster"], "together": [100],
                       "share_of_orders": [0.3], "a_to_b": [0.5], "b_to_a": [0.9], "lift": [3.0]})
    warn = baskets.drop_risks(pt, {"A": "Declining", "B": "Mature"})
    assert len(warn) == 1 and "Careful before dropping:** Mug" in warn[0] and "90% of Coaster orders" in warn[0]
    assert baskets.drop_risks(pt, {"A": "Mature", "B": "Mature"}) == []


def test_order_lines_upload_accepts_common_column_names():
    from demandino import baskets
    csv = "Invoice,StockCode,Qty\n" + "".join(f"{k},{p},1\n" for k in range(60) for p in ("X1", "Y2"))
    lines = baskets.read_order_lines(csv.encode())
    assert set(lines.columns) == {"order_id", "product_id"} and lines["order_id"].nunique() == 60
    with pytest.raises(UploadError):
        baskets.read_order_lines(b"date,units\n2026-01-01,3\n")


# ------------------------------------------------------- supplier lead times
def _po(pid, supplier, ordered, days, planned=14):
    day = 86400.0
    return {"id": pid, "status": "Received", "ordered_at": ordered * day, "received_at": (ordered + days) * day,
            "lines": [{"product_id": "P001", "supplier": supplier, "lead_time": planned}]}


def test_supplier_stats_learn_delivery_times_and_trends():
    from demandino import suppliers
    pos = [_po(k, "Steady Co", 20000 + 30 * k, 10) for k in range(6)]
    pos += [_po(100 + k, "Slow Ltd", 20000 + 30 * k, d) for k, d in enumerate([9, 10, 9, 18, 20, 22])]
    pos.append({"id": 999, "status": "Ordered", "ordered_at": 1.0, "received_at": None, "lines": []})
    st_ = suppliers.stats(suppliers.deliveries(pos)).set_index("supplier")
    assert st_.loc["Steady Co", "avg_days"] == 10 and st_.loc["Steady Co", "planning_lead"] == 10
    assert st_.loc["Steady Co", "trend"] == "Steady" and st_.loc["Steady Co", "late_share"] == 0
    assert st_.loc["Slow Ltd", "trend"] == "Getting slower"
    assert st_.loc["Slow Ltd", "planning_lead"] > st_.loc["Slow Ltd", "avg_days"]  # margin for variation
    stock = pd.DataFrame({"product_id": ["P001", "P002"], "supplier": ["Slow Ltd", "Unknown"], "lead_time": [14, 7]})
    adjusted, notes = suppliers.apply_learned_lead_times(stock, st_.reset_index())
    assert adjusted.loc[0, "lead_time"] == st_.loc["Slow Ltd", "planning_lead"] and adjusted.loc[1, "lead_time"] == 7
    assert notes and "Slow Ltd" in notes[0]


def test_receiving_a_purchase_order_records_dates():
    uid = db.create_user("buyer_test", "a very long password")
    po = db.create_po(uid, [{"product_id": "P001", "supplier": "Acme", "lead_time": 14}])
    db.decide_po(uid, po, "Ordered", "buyer")
    assert db.list_pos(uid)[0]["ordered_at"] is not None
    with pytest.raises(ValueError):
        db.receive_po(uid, po, 2_000_000.0, 1_000_000.0, "buyer")
    db.receive_po(uid, po, 1_000_000.0, 1_000_000.0 + 12 * 86400, "buyer")
    got = db.list_pos(uid)[0]
    assert got["status"] == "Received" and round((got["received_at"] - got["ordered_at"]) / 86400) == 12


# -------------------------------------------------------------- transfers
def test_transfers_move_spare_stock_to_short_warehouses(fc, sales):
    from demandino.planning import suggest_transfers
    one = default_stock(sales)
    one = one[one["product_id"] == "P001"]
    short = one.assign(warehouse="North", demand_share=50.0, stock=0)
    full = one.assign(warehouse="South", demand_share=50.0, stock=5000)
    plan, _ = portfolio_plan(fc, 0.15, pd.concat([short, full], ignore_index=True), 60, 14, 0.95)
    moves = suggest_transfers(plan, 14)
    assert len(moves) == 1
    m = moves.iloc[0]
    assert m["from_warehouse"] == "South" and m["to_warehouse"] == "North"
    assert m["units"] == m["order_before"] and m["order_after"] == 0 and m["money_saved"] > 0
    south = plan[plan["warehouse"] == "South"].iloc[0]
    keep = south["forecast_daily"] * (south["lead_time"] + 14) + south["safety_stock"]
    assert m["units"] <= 5000 - keep  # the donor keeps enough for its own cycle


def test_no_transfers_with_one_warehouse_or_no_spare(fc, sales):
    from demandino.planning import suggest_transfers
    one = default_stock(sales)
    one = one[one["product_id"] == "P001"]
    plan, _ = portfolio_plan(fc, 0.15, one.assign(stock=0), 60, 14, 0.95)
    assert suggest_transfers(plan, 14).empty
    both_short = pd.concat([one.assign(warehouse=w, demand_share=50.0, stock=0) for w in ("A", "B")], ignore_index=True)
    plan, _ = portfolio_plan(fc, 0.15, both_short, 60, 14, 0.95)
    assert suggest_transfers(plan, 14).empty


# ------------------------------------------------------------ what changed
def test_change_parts_add_up_and_follow_last_weeks_sales(sales):
    from demandino import changes
    boosted = sales.copy()
    last_week = (boosted["product_id"] == "P001") & (boosted["date"] > boosted["date"].max() - pd.Timedelta(days=7))
    boosted.loc[last_week, "units_sold"] *= 3
    fc = DemandForecaster(boosted, max_iter=60).fit(calibrate=False)
    e = changes.explain(fc, "P001", 30)
    assert abs(e["before"] + sum(v for _, v in e["parts"]) - e["now"]) < 1e-6
    parts = dict(e["parts"])
    assert parts["Last week's sales"] > 0 and e["change"] > 0  # a strong week raises the forecast
    assert e["last_week_actual"] > e["last_week_expected"]
    text = changes.sentence(e)
    assert text.startswith("Up") and "last week's sales" in text and "more than" in text
    table = changes.all_products(fc, 14)
    assert len(table) == fc.sales["product_id"].nunique() and table["why"].str.len().min() > 20


# --------------------------------------------------------------- lost sales
def test_lost_sales_value_stockout_days():
    from demandino import lostsales
    days = pd.date_range("2026-01-01", periods=10)
    sales = pd.DataFrame({"date": days, "product_id": "A", "product_name": "Mug", "category": "Home", "price": 10.0,
                          "units_sold": [5, 5, 0, 0, 0, 5, 5, 0, 0, 5], "demand": [5, 5, 5, 5, 5, 5, 5, 4, 4, 5],
                          "stockout": [0, 0, 1, 1, 1, 0, 0, 1, 1, 0]})
    rows = lostsales.daily(sales, {"A": 6.0})
    assert rows["lost_units"].sum() == 23 and rows["lost_sales"].sum() == 230 and rows["lost_profit"].sum() == 92
    ep = lostsales.episodes(rows)
    assert len(ep) == 2 and ep.iloc[0]["days"] == 3 and ep.iloc[0]["lost_sales"] == 150
    prod = lostsales.by_product(rows)
    assert prod.loc[0, "episodes"] == 2 and prod.loc[0, "days"] == 5
    text = lostsales.ideas(prod, 300.0, "$")
    assert "Mug" in text[0] and "$230" in text[0] and "43.4%" in text[1]


def test_lost_sales_found_in_the_sample_shop(sales):
    from demandino import lostsales
    fc = DemandForecaster(sales)
    rows = lostsales.daily(fc.sales)
    assert len(rows) > 0 and (rows["lost_units"] > 0).all() and rows["lost_sales"].sum() > 0


# --------------------------------------------------------------- what-if
def test_store_what_if_adds_up_and_responds_to_price(fc, sales):
    from demandino import whatif
    el = elasticities(sales)
    pids = ["P001", "P002", "P011"]
    none = whatif.run(fc, el, 0.15, pids, whatif.Change(), 30)
    assert np.allclose(none["units_change"], 0) and np.allclose(none["profit_change"], 0)
    up = whatif.run(fc, el, 0.15, pids, whatif.Change(price_change=0.10), 30)
    assert (up["units_change"] < 0).all()  # dearer → fewer units
    t = whatif.totals(up)
    assert np.isclose(t["profit_change"], up["profit_change"].sum())
    assert np.isclose(whatif.by_category(up)["profit_change"].sum(), t["profit_change"])
    start = fc.sales["date"].max() + pd.Timedelta(days=5)
    sale = whatif.run(fc, el, 0.15, pids, whatif.Change(discount=0.3, promo_start=start,
                                                        promo_end=start + pd.Timedelta(days=6)), 30)
    assert whatif.totals(sale)["units_change"] > 0  # a sale sells more units
    assert whatif.verdict(whatif.totals(sale), sale, "$")[0].startswith("**Overall:**")


# ----------------------------------------------------------- buying budget
def test_buying_budget_uses_stock_first(fc, sales):
    from demandino import budget
    stock = default_stock(sales)
    empty = budget.needs(fc, stock.assign(stock=0), 3)
    full = budget.needs(fc, stock.assign(stock=10 ** 7), 3)
    assert empty["buy"].sum() > 0 and np.isclose(empty["buy"].sum(), empty["cogs"].sum())
    assert full["buy"].sum() == 0  # plenty of stock: nothing to buy
    m = budget.against_budget(empty, empty.groupby("month")["buy"].sum().max() / 2)
    assert (m["status"] == "Over budget").any() and "Over budget" in budget.summary(m, "$")[1]


def test_forecasts_beyond_180_days_work(fc):
    res = fc.forecast_product("P001", 240)
    assert len(res.mean) == 240 and np.isfinite(res.mean).all()
