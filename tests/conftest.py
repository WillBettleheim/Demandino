"""Test setup: full error messages, and a throwaway workspace database."""
import os
import tempfile

os.environ.setdefault("DEMANDINO_DATA_DIR", tempfile.mkdtemp(prefix="demandino-test-"))

from streamlit import config  # noqa: E402

config.set_option("client.showErrorDetails", "full")
