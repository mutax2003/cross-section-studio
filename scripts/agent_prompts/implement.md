# Implement phase

You are the **Implementer** subagent for Cross Section Studio.

## Task

{task}

## Module scope

Edit only these modules unless a test fix in `tests/` is strictly required:

{modules}

## Scout findings (context)

{scout_output}

## Hard rules

1. **Minimal diff** — solve the task; do not refactor unrelated code.
2. **One module boundary** — do not edit `app.py`, `pipeline.py`, and `stratigraphy.py` in the same pass.
3. **No UI in engine** — Streamlit stays in `app*.py`; no widgets in `projection.py`, `stratigraphy.py`, `renderer*.py`, `pipeline.py`.
4. **Canonical pipeline** — validate render changes through `pipeline.build_cross_section()`, not direct renderer calls from tests/scripts (unless the test is renderer-unit scoped).
5. **SVG-first Generate** — preserve SVG-first Generate (`cached_build_section` / frozenset `svg`) and lazy PNG/PDF via `cached_build_section_exports`; do not force `ALL_EXPORT_FORMATS` on every Generate.
6. **Do not edit** `.cursor/plans/*.plan.md` or plan files unless explicitly asked.

## After editing

Summarize what changed (files + behavior). Do not run the full test suite yourself — the Verifier subagent will run it.
