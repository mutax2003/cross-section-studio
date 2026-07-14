# Cross Section Studio

Engineering-grade **borehole cross-section generation** for environmental and geotechnical projects. Upload a workbook, configure a transect, and export fence diagrams (SVG after Generate; PNG/PDF after Prepare).

## Quick start

```powershell
# Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. Use **Try sample project** on the welcome screen for a demo workbook.

The top **File / Edit / View / Help** menu bar mirrors Windows-style menus (with Ctrl shortcuts). Help topics live in [`docs/help/`](docs/help/).

Optional free LLM assist (Validate / Configure polish): set `GROQ_API_KEY` ([Groq](https://console.groq.com)) or `GEMINI_API_KEY` ([Gemini](https://aistudio.google.com/apikey)). The sidebar **AI Assist** section auto-enables when a free key is present.

## Windows desktop

```powershell
pip install -r requirements.txt pyinstaller
powershell -File scripts/build_windows.ps1
```

Run `dist\CrossSectionStudio\CrossSectionStudio.exe` (launches Streamlit on localhost).

## Docker

```bash
docker compose up --build
```

Optional auth (shared deployments):

```bash
export CROSS_SECTION_AUTH_PASSWORD=your-secret
docker compose up --build
```

## Quality gate (run before release)

From the repo root:

```powershell
python -m pytest -q
python scripts/e2e_smoke_direct.py
python scripts/smoke_test.py
```

Or: `python scripts/agent_supervisor.py verify --report`

Optional per-file diagnostic: `python scripts/run_e2e_tests.py` (writes `e2e_test_results.txt`; not part of the verify gate).

## Regenerate reference figures

```bash
python scripts/plot_ecoventure_gwm.py --transect all
python scripts/compare_figure_parity.py --warn-only
```

## Documentation

- [Workbook format](docs/workbook-format.md)
- [Operator runbook](docs/operator-runbook.md)
- [Agent orchestration](AGENTS.md)

## Interpretation disclaimer

Inter-hole geology is a **rule-based 2D fence diagram** (linear contacts, midpoint pinch-outs). It is not geostatistical surface interpolation. Review correlation preflight and overlap warnings before client delivery.
