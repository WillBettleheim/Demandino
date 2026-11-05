"""Security check for this installation. Run before pushing or deploying; exits non-zero if anything fails.

Usage: python scripts/security_check.py [--history] [--deps]
Looks for keys in tracked files (and, with --history, in every past commit), checks that secret files are ignored
by git and private on disk, and that the app's protective settings are on. With --deps it asks the OSV database about
the installed packages, sending only their names and versions. It prints where a problem is, never the secret itself.
"""
import importlib.metadata
import json
import os
import re
import stat
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECRET_PATTERNS = {
    "Anthropic key": r"sk-ant-[A-Za-z0-9_-]{20,}",
    "OpenAI-style key": r"\bsk-[A-Za-z0-9]{32,}",
    "Shopify token": r"\bshp(?:at|ca|pa|ss)_[a-fA-F0-9]{20,}",
    "GitHub token": r"\b(?:ghp|gho|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}",
    "AWS key": r"\bAKIA[0-9A-Z]{16}\b",
    "Slack webhook": r"hooks\.slack\.com/services/T[A-Z0-9]{6,}/B[A-Z0-9]{6,}/[A-Za-z0-9]{16,}",
    "Slack token": r"\bxox[abprs]-[A-Za-z0-9-]{20,}",
    "Stripe live key": r"\b[sr]k_live_[A-Za-z0-9]{20,}",
    "WooCommerce key": r"\bc[ks]_[a-f0-9]{40}\b",
    "Private key": r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----",
}
MUST_BE_IGNORED = [".env", ".data/demandino.db", ".data/.secret-key", ".cache/.signing-key", ".streamlit/secrets.toml",
                   "data/raw/orders.csv", "server.pem"]
MUST_BE_PRIVATE = [".data", ".cache", ".data/demandino.db", ".data/.secret-key", ".cache/.signing-key"]
REQUIRED_SETTINGS = ['enableXsrfProtection = true', 'enableCORS = true', 'showErrorDetails = "type"',
                     'toolbarMode = "viewer"', 'maxUploadSize = 25', 'enableStaticServing = false',
                     'fileWatcherType = "none"', 'magicEnabled = false']


def find_secrets(text: str) -> list[str]:
    """Names of the kinds of secret found in some text."""
    return [name for name, pattern in SECRET_PATTERNS.items() if re.search(pattern, text)]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, errors="replace").stdout


def scan_tracked() -> list[str]:
    problems = []
    for name in _git("ls-files").splitlines():
        path = ROOT / name
        if path.is_file() and path.stat().st_size < 2_000_000:
            for kind in find_secrets(path.read_text(errors="replace")):
                problems.append(f"{kind} in {name}")
    return problems


def scan_history() -> list[str]:
    problems = []
    for commit in _git("rev-list", "--all").splitlines():
        for kind in find_secrets(_git("show", "--format=", commit)):
            problems.append(f"{kind} in commit {commit[:8]}")
    return problems


def check_ignored() -> list[str]:
    return [f"{p} is not ignored by git" for p in MUST_BE_IGNORED
            if subprocess.run(["git", "check-ignore", "-q", p], cwd=ROOT).returncode != 0]


def check_private() -> list[str]:
    problems = []
    for p in MUST_BE_PRIVATE:
        path = ROOT / p
        if path.exists() and stat.S_IMODE(path.stat().st_mode) & 0o077:
            problems.append(f"{p} can be read by other accounts on this machine")
    return problems


def check_settings() -> list[str]:
    config = (ROOT / ".streamlit" / "config.toml").read_text()
    return [f"setting missing from .streamlit/config.toml: {s}" for s in REQUIRED_SETTINGS if s not in config]


def check_dependencies() -> list[str]:
    """Installed packages with known vulnerabilities, according to osv.dev."""
    packages = sorted({(d.metadata["Name"].lower(), d.version) for d in importlib.metadata.distributions()
                       if d.metadata["Name"]})
    body = {"queries": [{"package": {"name": n, "ecosystem": "PyPI"}, "version": v} for n, v in packages]}
    request = urllib.request.Request("https://api.osv.dev/v1/querybatch", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "User-Agent": "demandino-security-check"})
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    try:
        with urllib.request.urlopen(request, context=context, timeout=60) as response:  # noqa: S310 - fixed https address
            results = json.load(response)["results"]
    except OSError as exc:
        return [f"could not reach osv.dev ({type(exc).__name__})"]
    return [f"{name} {version}: {', '.join(v['id'] for v in found['vulns'])}"
            for (name, version), found in zip(packages, results) if found.get("vulns")]


def advice() -> list[str]:
    tips = []
    if not os.environ.get("DEMANDINO_SECRET_KEY"):
        tips.append("DEMANDINO_SECRET_KEY isn't set here: on a server, set it so the encryption key isn't stored next "
                    "to the database it protects.")
    if (ROOT / ".data" / "demo-account.txt").exists():
        tips.append(".data/demo-account.txt holds a test password in plain text: delete it once you don't need it.")
    return tips


def main() -> int:
    checks = [("Keys in tracked files", scan_tracked), ("Secret files ignored by git", check_ignored),
              ("Secret files private on disk", check_private), ("Protective settings", check_settings)]
    if "--history" in sys.argv:
        checks.append(("Keys in git history", scan_history))
    if "--deps" in sys.argv:
        checks.append(("Known vulnerabilities in dependencies", check_dependencies))
    failed = False
    for name, check in checks:
        problems = check()
        print(f"{'FAIL' if problems else 'ok  '}  {name}")
        for p in problems:
            print(f"      - {p}")
        failed = failed or bool(problems)
    for tip in advice():
        print(f"note  {tip}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
