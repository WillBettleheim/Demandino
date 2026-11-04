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
