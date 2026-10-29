"""Security measures: upload limits, export sanitising, signed cache files, HTML escaping."""
import pandas as pd
import pytest

from demandino import security
from demandino.security import UploadError


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
