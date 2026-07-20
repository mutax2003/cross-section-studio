# Cursor Automation — Weekly E2E Supervisor

Canonical local/SDK protocol: **`AGENTS.md`**. Use this doc when creating a **Cursor Automation** in the Automations editor (Agents Window).

## Purpose

Scheduled health check: run the deterministic E2E gate; optionally invoke `agent_supervisor.py` with cloud runtime. Configure PR creation in the **Automations / Cloud Agent UI** if desired — the CLI does not set `auto_create_pr`.

## Recommended trigger

| Field | Value |
|-------|-------|
| Trigger | Cron schedule |
| Schedule | `0 6 * * 1` (Mondays 06:00 UTC) |

## Tools

- Shell / terminal (run pytest and smoke scripts)
- Git (read repo state)

## Instructions (paste into Automation prompt)

```
Repository: Cross Section Studio (borehole cross-section platform).

On each run:

1. Run the E2E quality gate from repo root (see AGENTS.md / VERIFY_COMMANDS):
   python -m pytest -q
   python scripts/e2e_smoke_direct.py
   python scripts/smoke_test.py
   (scripts/run_e2e_tests.py is optional diagnostic logging only — not in VERIFY_COMMANDS)

2. If any step fails:
   - Summarize which step failed and the last 50 lines of output.
   - Propose the smallest fix scoped to one module boundary (see AGENTS.md).
   - If this automation's cloud agent step is configured to open PRs in the Cursor UI, open a PR with the fix.

3. If all steps pass, post a one-line success summary only.

Constraints:
- Do not replace pipeline.build_cross_section() with LLM logic.
- One module boundary per fix pass.
- Reference AGENTS.md and .cursor/rules/.
```

## Optional: SDK supervisor (cloud)

Requires `CURSOR_API_KEY` as an automation secret.

```bash
pip install -r requirements.txt -r requirements-dev.txt
python scripts/agent_supervisor.py run \
  --task "Weekly E2E health and dead-code scan" \
  --modules pipeline.py \
  --runtime cloud \
  --review \
  --summary-agent \
  --report
```

Prefer a **single** module boundary in `--modules` (serialize Implement passes). The SVG-first triad (`pipeline.py`, `section_build_request.py`, `app_services.py`) is allowlisted for cache/DTO glue — see `module_boundary_warnings` in `scripts/agent_supervisor.py` and AGENTS.md.

Reports land in `orchestration_reports/latest_run.md` (gitignored). Cloud runtime needs `git remote origin` and uses the current branch as `starting_ref`. Enable PR creation in the Automations editor if you want auto-PRs.

## To finish in editor

- Confirm repo and branch for git/cron trigger
- Add `CURSOR_API_KEY` secret if using cloud supervisor
- Select shell tool permissions
- Test with manual trigger before enabling weekly cron

## Related files

- [AGENTS.md](../AGENTS.md) — supervisor protocol
- [scripts/agent_supervisor.py](../scripts/agent_supervisor.py) — SDK CLI
- [scripts/run_verify_batch.ps1](../scripts/run_verify_batch.ps1) — local batch verify
- [.github/workflows/e2e.yml](../.github/workflows/e2e.yml) — PR CI gate
