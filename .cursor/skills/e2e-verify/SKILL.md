---
name: e2e-verify
description: >-
  Runs the Cross Section Studio E2E quality gate (VERIFY_COMMANDS): pytest -q,
  e2e_smoke_direct.py, smoke_test.py. Use when the user asks to test end to end,
  run E2E, verify, quality gate, or VERIFY_COMMANDS; after Implement before COMPLETE;
  or when checking merge readiness.
---

# Cross Section Studio — E2E Verify

## When to run

| Mode | Commands |
|------|----------|
| **COMPLETE / merge / “test end to end”** | Full three-step `VERIFY_COMMANDS` below |
| **Fast iteration** | Routing-table Verify-focus pytest file(s) from `AGENTS.md` + one smoke |

Do **not** edit source or fixtures while verifying. Stop on first failure.

## Full gate (required)

Run from repo root. Prefer `python -m pytest` (not bare `pytest`).

### Windows (PowerShell)

```powershell
python -m pytest -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/e2e_smoke_direct.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/smoke_test.py
```

### Linux / macOS / CI

```bash
python -m pytest -q && \
python scripts/e2e_smoke_direct.py && \
python scripts/smoke_test.py
```

**Equivalent:** `python scripts/agent_supervisor.py verify`. Batch wrapper (10 sequential verify cycles by default, not a single gate run): `powershell -File scripts/run_verify_batch.ps1` (writes under `orchestration_reports/`).

Canonical list: `VERIFY_COMMANDS` in `scripts/agent_supervisor.py`. Mirror CI: `.github/workflows/e2e.yml`.

## Not part of the gate

- `scripts/run_e2e_tests.py` — optional diagnostic (curated file list → `e2e_test_results.txt`)
- `scripts/compare_figure_parity.py --suite all --warn-only` — Quality CI hygiene, not VERIFY
- Streamlit UI steps (Upload → Validate → Configure → Generate) — unrelated naming

## Overlap warnings

Smoke scripts may print `Overlapping polygons detected...` on stderr. That is expected QA logging; exit code 0 still means PASS.

## Report format

```markdown
### Result
PASS | FAIL (step: <name>)

### Summary
- pytest: N passed, N failed, N skipped
- e2e_smoke_direct: 8/8 or failure detail
- smoke_test: OK (holes/polygons) or error

### Failure excerpt
(last 50 lines of the failing step only)
```

On FAIL: return the excerpt for a focused Implementer; do not start a broad refactor.

## Deps

```bash
pip install -r requirements.txt -r requirements-dev.txt
```
