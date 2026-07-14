# Operator Runbook — Cross Section Studio

## Deployment modes

| Mode | Entry | Audience |
|------|-------|----------|
| Streamlit dev | `streamlit run app.py` | Developers |
| Docker | `docker compose up --build` | Internal LAN / Cloud |
| Windows desktop | `CrossSectionStudio.exe` via PyInstaller | Field office |
| Streamlit Cloud | `python scripts/deploy_streamlit_cloud.py` | Pilot hosting |

## Environment variables

| Variable | Purpose |
|----------|---------|
| `CROSS_SECTION_AUTH_PASSWORD` | Shared password gate (Streamlit). **Recommended** for Docker / LAN exposure. |
| `CROSS_SECTION_AUTH_REQUIRED` | `1`/`true` — refuse to run if `CROSS_SECTION_AUTH_PASSWORD` is unset |
| `CROSS_SECTION_LOG_FORMAT` | `json` for structured logs; default text |
| `CROSS_SECTION_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING` (default `INFO`) |
| `CROSS_SECTION_AUDIT_LOG` | Path to JSON-lines audit file (must stay under the user data dir) |
| `CROSS_SECTION_DISABLE_LLM` | `1`/`true` — hide LLM assist controls (no third-party prompts) |
| `CROSS_SECTION_DEBUG_UI` | `1`/`true` — show full generate tracebacks in the UI (dev only) |
| `CROSS_SECTION_PORT` | Desktop launcher port override |
| `SENTRY_DSN` / `SENTRY_TRACES_RATE` / `SENTRY_ENVIRONMENT` | Optional APM (PII scrubbed; sample rate clamped 0–1) |
| `GROQ_API_KEY` | **Recommended free LLM** — [console.groq.com](https://console.groq.com); auto-enables Assist when set |
| `GEMINI_API_KEY` | Free LLM alternative — [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `OPENAI_API_KEY` | Optional paid LLM |

Provider preference when multiple keys are set: **Groq → Gemini → OpenAI**.

## Menus, shortcuts, and help

The Streamlit UI (including the Windows desktop build) shows a top **File / Edit / View / Help** bar.

| Shortcut | Action |
|----------|--------|
| `Ctrl+Shift+O` | Load sample project |
| `Ctrl+G` | Generate / regenerate cross-section |
| `Ctrl+Shift+C` | Clear section output |
| `Ctrl+H` | Toggle hatch patterns |
| `Ctrl+/ or F1` | Keyboard shortcuts help |
| `Ctrl+Shift+/` | Getting started help |

Help markdown lives in `docs/help/` (bundled for desktop; Docker keeps `docs/help/**` despite ignoring other `*.md`). Shortcuts are ignored while typing in text fields. In-app Help → Keyboard shortcuts is generated from `app_menubar._SHORTCUT_ROWS`.

## Release checklist

1. Run the full E2E gate (see [README](../README.md)).
2. Bump version tag if releasing a build artifact.
3. Regenerate GWM figures: `python scripts/plot_ecoventure_gwm.py --transect all`
4. Run parity check: `python scripts/compare_figure_parity.py --warn-only`
5. For Windows zip: `powershell -File scripts/build_windows.ps1`
6. For Docker: `docker build -t cross-section-studio .`

## Troubleshooting

| Symptom | Action |
|---------|--------|
| Generate blocked on overlaps | Review Configure preflight; disable **Block export on polygon overlaps** only after manual QA |
| Placeholder elevation error | Set site elevation in sidebar or switch to relative depth mode |
| Invalid SVG download | Regenerate; check overlap warnings and transect hole count |
| Docker unhealthy | Confirm port 8501 and `/_stcore/health` responds |
| Advantage ingest fails | Confirm `data/advantage_phase2_platform.xlsx` exists or run convert script with source workbook |

## Audit trail

When `CROSS_SECTION_AUDIT_LOG` is set (or the default path is used), upload and export events append JSON lines with timestamp, event type, and workbook name. Rotate or archive `data/audit.log` per your data-retention policy. Absolute overrides must stay under the user data directory.

## Security notes (Docker / shared hosts)

- Publish `8501` only behind a reverse proxy or set `CROSS_SECTION_AUTH_PASSWORD` (and prefer `CROSS_SECTION_AUTH_REQUIRED=1`).
- Do not bake `.streamlit/secrets.toml` into images (excluded via `.dockerignore`).
- Writable `data/` (audit log, lithology styles/aliases) is **single-tenant** on non-frozen deploys — one shared volume per instance.
- Failed sign-in attempts are rate-limited (temporary lockout after 5 failures; default ~30s).

## CI

GitHub Actions runs on push/PR to `main`/`master`:

- **E2E** (`.github/workflows/e2e.yml`) — pytest (+ `requirements-dev`), smoke scripts, per-module gate
- **Quality** (`.github/workflows/quality.yml`) — ruff, coverage threshold, pip-audit on `requirements.txt` (dev deps are install-only), figure parity (warn-only)

## Dev / agent verify

- Protocol: [`AGENTS.md`](../AGENTS.md)
- One-shot: `python scripts/agent_supervisor.py verify --report`
- Batch: `powershell -File scripts/run_verify_batch.ps1`
- Auth: after sign-in, use **Sign out** in the sidebar when `CROSS_SECTION_AUTH_PASSWORD` is set
