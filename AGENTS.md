# Cross Section Studio — Agent Orchestration

Hierarchical dev orchestration for this repo. The **supervisor** (parent agent or `scripts/agent_supervisor.py`) delegates scoped work to specialist subagents, then gates completion on the deterministic E2E test spine.

**Not in scope:** in-product geology LLM agents. Keep `ai_assistant.py` UI-only; geometry stays in `pipeline.build_cross_section()`.

**Boundaries (always on):** `.cursor/rules/borehole-platform-architecture.mdc`. **Full module table, workbook sheets, entry points:** `.cursor/rules/borehole-module-catalog.mdc` (globs on engine/UI roots). This file is protocol and routing only.

## Supervisor protocol

1. **Decompose** every non-trivial task into: Scout → Implement → Verify (skip Scout for single-file needle queries).
2. **Parallelize** independent Scouts only; serialize Implement and Verify.
3. **One module boundary per Implement pass** — never edit `app.py` + `pipeline.py` + `stratigraphy.py` in the same subagent run. `agent_supervisor.py` prints **advisory** `BOUNDARY WARNING`s when `--modules` spans risky sets (does not hard-fail unless you stop and split).
4. **Verify before done** — run the full E2E gate (see below).
5. **On failure** — route to CI triage or a focused Implementer with the failing log excerpt only.

## Specialist roles

| Role | `subagent_type` | Scope | Hard rules |
|------|-----------------|-------|------------|
| **Scout** | `explore` | Read-only discovery | Never edit files; return paths, function names, risks |
| **Implementer** | `generalPurpose` | One module boundary | No Streamlit widgets in engine modules; use `build_cross_section()` for render validation |
| **Verifier** | `shell` | Tests only | Run E2E gate; return pass/fail + last 50 lines on failure. **No source edits.** IDE: `verify.md`. SDK: local subprocess gate |
| **Reviewer** | `bugbot` | Diff review | Module-boundary + geology integrity |
| **CI triage** | `ci-investigator` | One failed check | Root cause + smallest fix path |

### Extended personas (IDE feature path)

For vague / multi-module features, use `.cursor/rules/multi-agent-personas.mdc`:

**Feature path (IDE):** PM → Architect → Implementer(s) → Reviewer → Verifier  

**Short path (default):** Scout → Implementer → Verifier  

**SDK path** (`agent_supervisor.py`): Scout → Implement → Verify → optional Review/Summary only.  
PM/Architect prompts (`pm.md`, `architect.md`) are **IDE-only** — the SDK does not run those phases.

## Task routing

| Intent | Scout scope | Implement scope | Verify focus |
|--------|-------------|-----------------|--------------|
| Optimize renderer | `renderer*.py`, `render_*.py` | One mixin or `renderer_common.py` | `tests/test_renderer_styles.py` |
| PDF / report export | `report_export.py`, `pipeline.py` | `report_export.py` | `tests/test_pipeline.py`, `tests/test_renderer_styles.py` |
| Fix ingest | `ingestion.py`, `parsing.py` | Same boundary | `tests/test_ingestion.py` |
| Workbook template | `workbook_template.py` | `workbook_template.py` | `tests/test_workbook_template.py` |
| Stratigraphy bug | `stratigraphy.py` | `stratigraphy.py` only | `tests/test_stratigraphy.py`, `tests/test_e2e_edge_cases.py` |
| Projection math | `projection.py` | `projection.py` | `tests/test_projection.py` |
| Transect planner | `transect_planner.py` | `transect_planner.py` | `tests/test_ai_assistant.py`, `tests/test_e2e_edge_cases.py` |
| Lithology palette | `constants.py` | `constants.py` | `tests/test_renderer_styles.py`, `tests/test_advantage_p2_reference.py` |
| Lithology codes helper | `lithology_codes.py` | `lithology_codes.py` | `tests/test_pipeline.py` |
| Water / GW QA & UI | `ai_quality.py`, `app_validate.py` (scout may read both) | One of `ai_quality.py` or one `app_*.py` per pass — do not mix in `--modules` | `tests/test_water_quality.py`, `tests/test_streamlit_app.py` |
| AI assistant (UI) | `ai_assistant.py` | `ai_assistant.py` only | `tests/test_ai_assistant.py` |
| SVG-first / Generate cache | `app_services.py`, `section_build_request.py`, `pipeline.py` | Serialize preferred — one per pass (`section_build_request` → `pipeline` → `app_services`); kwargs/cache glue may need `app_services` after DTO/pipeline changes | `tests/test_pipeline.py`, `tests/test_section_build_request.py`, `tests/test_import_smoke.py` |
| Pipeline / export | `pipeline.py`, `section_build_request.py`, `app_services.py` | Serialize preferred — one per pass; SVG-first triad allowlisted for cache/DTO glue (see `module_boundary_warnings`) | `tests/test_pipeline.py`, `tests/test_section_build_request.py` |
| UI / sidebar / presets | `app_sidebar.py`, `ui_helpers.py`, `ui_output_presets.py` | One UI module per pass | `tests/test_app_helpers.py`, `tests/test_ui_helpers.py`, `tests/test_ui_output_presets.py`, `tests/test_streamlit_app.py` |
| Menubar / help | `app_menubar.py`, `docs/help/getting-started.md`, `paths.py` | `app_menubar.py` | `tests/test_menubar.py`, `tests/test_paths.py` |
| Ops / auth / audit | `ops_*.py`, `paths.py` | One `ops_*.py` per pass | `tests/test_ops.py` |
| Packaging / Docker | `Dockerfile`, `launcher.py`, `paths.py` | One packaging file | `tests/test_paths.py`, `tests/test_launcher_port.py` |
| Reference parity | `gwm_reference/`, `advantage_p2_reference/` | Same package only | `tests/test_ecoventure_gwm_figures.py`, `tests/test_advantage_p2_reference.py` |
| Supervisor / prompts | `scripts/agent_supervisor.py` | Same | `tests/test_agent_supervisor.py`, `tests/test_agent_supervisor_edge_cases.py` |
| Full refactor | Parallel scouts | One implementer per boundary, serialized | Full E2E gate |

## E2E quality gate

Canonical command list is also `VERIFY_COMMANDS` in `scripts/agent_supervisor.py`. Exit non-zero if any step fails.

**Windows (PowerShell)** — stop on first failure:

```powershell
python -m pytest -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/e2e_smoke_direct.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/smoke_test.py
```

**Linux / macOS / CI:**

```bash
python -m pytest -q && \
python scripts/e2e_smoke_direct.py && \
python scripts/smoke_test.py
```

Batch wrapper: `powershell -File scripts/run_verify_batch.ps1` (writes under `orchestration_reports/`).

**Optional diagnostic:** `scripts/run_e2e_tests.py` re-runs a curated pytest file list and writes per-file results to `e2e_test_results.txt`. It is **not** part of `VERIFY_COMMANDS` (pytest already covers the suite).

**Deps:** CI and local verify need `pip install -r requirements.txt -r requirements-dev.txt` (pytest + `cursor-sdk` for supervisor unit tests).

**CI hygiene (not E2E):** `.github/workflows/quality.yml` runs scoped ruff, engine coverage, and pip-audit on PRs — not a substitute for `VERIFY_COMMANDS`.

### IDE iteration verify

While iterating: run the routing-table **Verify focus** pytest file(s) for touched modules; optionally one smoke (`e2e_smoke_direct` or `smoke_test`).

Before **COMPLETE** / SDK `verify` / merge: run the full three-step `VERIFY_COMMANDS` gate above.

## SDK supervisor

```text
python scripts/agent_supervisor.py run \
  --task "Describe the change" \
  --modules renderer_common.py,renderer.py \
  --runtime local
```

Phases: Scout → Implement → **Verify (local subprocess)** → optional `--review` / `--summary-agent`.

Requires `CURSOR_API_KEY` for agent phases. Weekly automation notes: [`docs/cursor-automation-supervisor.md`](docs/cursor-automation-supervisor.md).

## SDK flags

| Flag | Effect |
|------|--------|
| `verify` / `run --verify-only` | Local E2E gate only (no API key) |
| `--skip-scout` | Implement uses placeholder scout context |
| `--skip-implement` | Scout (optional) then verify only |
| `--review` | Diff review agent after verify |
| `--report [PATH]` | Markdown report (`orchestration_reports/latest_run.md` default) |
| `--summary-agent` | Executive summary agent on report |
| `--runtime cloud` | Needs `git remote origin`; current branch as ref |
| `--model` | Agent model (default `composer-2.5`) |
| `--api-key` | Override `CURSOR_API_KEY` |

## Prompt inventory

| Prompt | Runtime |
|--------|---------|
| `scout.md`, `implement.md`, `review.md`, `summary.md` | SDK + IDE |
| `verify.md` | **IDE shell only** (SDK uses `run_verify_local`) |
| `pm.md`, `architect.md` | **IDE feature path only** |
| `security.md`, `ci_triage.md` | IDE when using those roles |

Replace `{task}`, `{modules}`, `{pm_spec}`, etc. before pasting into Task tool.

## IDE run report

When asked what agents did: task, phases, verify pass/fail, `git diff --stat HEAD`, verdict (`COMPLETE` / `COMPLETE_WITH_NOTES` / `NEEDS_ATTENTION`). Offer `orchestration_reports/` save.

## Related rules

See `.cursor/rules/` — especially `borehole-platform-architecture.mdc`, `borehole-module-catalog.mdc`, `multi-agent-personas.mdc`, `pipeline-orchestration.mdc`, `stratigraphy-engine.mdc`, `projection-engine.mdc`, `renderer-visualization.mdc`, `streamlit-app.mdc`, `models-ingestion.mdc`, `ai-quality-assistant.mdc`, `packaging-deployment.mdc`, `agent-orchestration.mdc`.
