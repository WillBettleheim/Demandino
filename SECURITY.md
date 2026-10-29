# Security

Measures built into Demandino:

| Risk | Protection | Where |
|---|---|---|
| Huge or malformed uploads (denial of service) | 25 MB upload cap; at most 2,000,000 rows, 50 columns, 300 products and 10 years of history; binary files are rejected | `.streamlit/config.toml`, `demandino/security.py` |
| Memory growth from many uploads | In-memory caches for uploads are bounded (a few entries) and expire after 1–6 hours | `demandino/state.py` |
| User data left on disk | Uploaded data is never written to disk; only the built-in datasets use the disk cache | `demandino/state.py` |
| Tampered cache files (pickle code execution) | Cache files are signed with HMAC-SHA256 using a per-install secret (`.cache/.signing-key`, owner-only permissions); unsigned or modified files are ignored | `demandino/security.py` |
| Script injection in the page (XSS) | All text placed in custom HTML is escaped; uploaded names are stripped of control characters and capped at 80 characters | `demandino/ui.py`, `demandino/data.py` |
| Formula injection in CSV/Excel exports | Cells starting with `=`, `+`, `-`, `@`, tab or carriage return are prefixed with `'` | `demandino/security.py`, `demandino/report.py` |
| Leaking internals in error messages | Users see only the error type, never stack traces; file errors show a short, safe message | `.streamlit/config.toml`, `demandino/state.py` |
| Cross-site request forgery | Streamlit XSRF and CORS protection kept on explicitly | `.streamlit/config.toml` |
| Browser scripts | The only JavaScript is Demandino's own static code (theme detection, number animation, theme switch, terms cookie); no user content is ever placed in a script | `demandino/ui.py`, `demandino/legal.py` |
| Terms cookie | Stores only an acceptance token (`dm_terms`), `SameSite=Strict`, no personal data | `demandino/legal.py` |
| Supply chain | Dependencies are pinned in `requirements.txt` | |

Recommendations when you deploy:

- Serve it over HTTPS behind a reverse proxy, and add authentication if the data is private. Streamlit has no
  user accounts of its own.
- Keep `.cache/` and `data/raw/` out of version control (already in `.gitignore`).
- Update the pinned dependencies regularly, for example by checking them with `pip-audit`.

Report a vulnerability privately to the repository owner rather than in a public issue.
