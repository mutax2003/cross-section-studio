# Cross Section Studio

Engineering-grade **borehole cross-section generation** for environmental and geotechnical projects. Upload an Excel workbook, configure a transect, and export fence diagrams:

- **Generate** → SVG preview immediately  
- **Prepare deliverables** → PNG + PDF in one draw, then Word / clipboard / report ZIP / multi-transect package  

Inter-hole geology is a **rule-based 2D fence diagram** (linear contacts, midpoint pinch-outs) — **not** geostatistical or 3D surface interpolation. Review Configure preflight and overlap warnings before client delivery.

Requires **Python 3.12+**.

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

Open [http://localhost:8501](http://localhost:8501).

1. **Try sample project** on the welcome card (or `Ctrl+Shift+O`), or download the **data entry template** and upload your `.xlsx`.
2. Walk **Upload → Validate → Configure → Generate**.
3. After Generate, click **Prepare deliverables** for PNG/PDF and the drafter package.

The top **File / Edit / View / Help** bar mirrors Windows-style menus. In-app help topics live in [`docs/help/`](docs/help/).

### Optional LLM assist

Validate / Configure guidance only (never changes geometry). Set `GROQ_API_KEY` ([Groq](https://console.groq.com)) or `GEMINI_API_KEY` ([Gemini](https://aistudio.google.com/apikey)). Sidebar **AI Assist** auto-enables when a free key is present. Prefer Groq when both are set.

## Operator workflow (summary)

| Step | What you do |
|------|-------------|
| **Upload** | Excel workbook (native Collars/Lithology, field export, or filled template) |
| **Validate** | Parse warnings, lithology mapping, water/screens/environmental summaries |
| **Configure** | Hole order, transect, layout preset, overlays; correlation preflight; optional multi-transect batch lines |
| **Generate** | SVG fence diagram |
| **Prepare** | PNG · PDF · Word · clipboard · report ZIP (Generate-step action, not a fifth workflow step) |

### Workbook essentials

| Sheet | Role |
|-------|------|
| **Collars** + **Lithology** | Required |
| **Water** / **Screens** / **Gradients** / **Environmental** | Optional overlays |
| **Sections** | Named transects → Configure **Multi-transect batch ZIP** |
| **Field Data** | Field-export `OVA` / `EC` → environmental readings (not stratigraphy) |

Full schemas: [`docs/workbook-format.md`](docs/workbook-format.md). Short in-app guide: Help → **Workbook & data entry**.

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

Shared / LAN deployments — set a password gate:

```bash
export CROSS_SECTION_AUTH_PASSWORD=your-secret
export CROSS_SECTION_AUTH_REQUIRED=1   # refuse to start if password unset
docker compose up --build
```

See [`docs/operator-runbook.md`](docs/operator-runbook.md) for env vars, audit log, and security notes.

## Quality gate (before release)

```powershell
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
python scripts/e2e_smoke_direct.py
python scripts/smoke_test.py
```

Or: `python scripts/agent_supervisor.py verify --report`

Optional diagnostic (not part of the gate): `python scripts/run_e2e_tests.py`.

### Figure parity (quality CI, not E2E)

```bash
python scripts/plot_ecoventure_gwm.py --transect all
python scripts/compare_figure_parity.py --suite all
```

Suite MSE ceilings hard-fail in CI; use `--warn-only` for local soft runs. Default comparison letterboxes aspect ratio.

## Documentation

| Doc | Audience |
|-----|----------|
| [`docs/help/`](docs/help/) | In-app Help menu (operators) |
| [`docs/workbook-format.md`](docs/workbook-format.md) | Sheet schemas & templates |
| [`docs/operator-runbook.md`](docs/operator-runbook.md) | Deploy, auth, env, release checklist |
| [`AGENTS.md`](AGENTS.md) | Agent / CI orchestration for contributors |

## Interpretation disclaimer

Fence fills between boreholes use deterministic correlation rules (including midpoint pinch-outs when enabled). They are schematic engineering drawings, not interpolated geological models. Always review **Configure** correlation health, **Block export on polygon overlaps**, and Generate overlap warnings before issuing figures to clients.
