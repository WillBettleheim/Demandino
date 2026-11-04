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
| Cookies | `dm_terms` (acceptance token) and `dm_session` (sign-in token); `SameSite=Strict`, no personal data | `demandino/legal.py` |
| Accounts | Passwords hashed with scrypt (n=2^14, r=8, p=1, per-user salt); constant-work checks for unknown usernames; 5 failed attempts lock the account for 15 minutes; sessions are random 256-bit tokens stored only as SHA-256 hashes, expire after 14 days and are revoked on sign-out or password change | `demandino/db.py`, `demandino/account.py` |
| Stored secrets | Connector keys, webhooks and SMTP passwords are encrypted with Fernet (key in `.data/.secret-key`, owner-only, or `DEMANDINO_SECRET_KEY`) | `demandino/db.py` |
| Data isolation | Every workspace query is scoped to the signed-in user's id; the database file is owner-only | `demandino/db.py` |
| Outbound requests (SSRF) | Connectors and alerts use HTTPS only, fixed host allow-lists (docs.google.com, *.myshopify.com, hooks.slack.com, the store's own host), refuse cross-host redirects, numeric/private hosts and embedded credentials, and cap response size and time; email requires TLS | `demandino/integrations.py` |
| Share links | Random tokens stored as hashes, read-only summaries only, expire after 7 days, revocable | `demandino/db.py`, `demandino/shared_view.py` |
| AI assistant | Sends only a forecast summary (no raw files); summary is marked as data, not instructions; question length and conversation length are capped | `demandino/assistant.py` |
| Supply chain | Dependencies are pinned in `requirements.txt` | |
| Markup from uploaded files | Product names and other file text are escaped before being shown in markdown, so a crafted CSV can't add links, images, HTML or icons | `demandino/ui.py` (`md_safe`) |
| Password guessing | Per-account lockout, plus a limit of 30 failures per 15 minutes from one address across all accounts; common and very simple passwords are refused | `demandino/db.py`, `demandino/account.py` |
| Stolen sessions | "Sign out everywhere" and a record of sign-ins and failed attempts on the Security page; the sign-in cookie is `Secure` over HTTPS | `views/security.py`, `demandino/account.py` |
| API keys | Encrypted at rest, never shown again after saving, and listed by name only. Keys never go in code: use `.env` (ignored by git; see `.env.example`), `DEMANDINO_SECRET_KEY` or the Connections page | `demandino/db.py`, `.gitignore` |
| Files on disk | Data and cache folders are private to the account running the app; cache files never include uploaded outside factors, and old ones are removed | `demandino/security.py`, `demandino/state.py` |
| Large order files | Order lines are held in a sparse matrix, so a big upload can't exhaust memory | `demandino/baskets.py` |
| Committed secrets | `scripts/security_check.py` scans tracked files (and git history with `--history`), checks ignored and private files and protective settings; the test suite runs it too | `scripts/security_check.py` |

Known limits:

- The sign-in cookie is set from the page, so it can't be `HttpOnly`: Streamlit gives apps no way to set response
  headers. Escaping all file text is what protects it. Sessions last 14 days and can be ended from the Security page.
- Saved datasets are compressed but not encrypted in `.data/demandino.db`, which is readable only by the account
  running the app. Use disk encryption on the server if they are sensitive.

Recommendations when you deploy:

- Serve it over HTTPS behind a reverse proxy, and add authentication if the data is private. Streamlit has no
  user accounts of its own.
- Keep `.cache/` and `data/raw/` out of version control (already in `.gitignore`).
- Update the pinned dependencies regularly, for example by checking them with `pip-audit`.
- Set `DEMANDINO_SECRET_KEY` on the server, so the encryption key isn't stored next to the database it protects.
- Have the reverse proxy add security headers (`Strict-Transport-Security`, `Content-Security-Policy`,
  `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`) and pass the real client address, so the
  per-address sign-in limit counts visitors rather than the proxy.
- Run `python scripts/security_check.py --history` before each deploy.

Report a vulnerability privately to the repository owner rather than in a public issue.
