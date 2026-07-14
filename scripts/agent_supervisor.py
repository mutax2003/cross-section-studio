"""Hierarchical Cursor agent supervisor for Cross Section Studio.

Phases: Scout (readonly discovery) → Implement (scoped edit) → Verify (local E2E gate)
→ optional Review (diff review via agent).

Requires CURSOR_API_KEY and: pip install -r requirements-dev.txt

Example:
    python scripts/agent_supervisor.py run \\
        --task "Optimize renderer lithology rects" \\
        --modules renderer_common.py,renderer.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = Path(__file__).resolve().parent / "agent_prompts"
DEFAULT_MODEL = "composer-2.5"
DEFAULT_REPORT_PATH = ROOT / "orchestration_reports" / "latest_run.md"

VERIFY_COMMANDS: tuple[tuple[str, list[str]], ...] = (
    ("pytest", [sys.executable, "-m", "pytest", "-q"]),
    ("e2e_smoke_direct", [sys.executable, str(ROOT / "scripts" / "e2e_smoke_direct.py")]),
    ("smoke_test", [sys.executable, str(ROOT / "scripts" / "smoke_test.py")]),
)


@dataclass
class PhaseRecord:
    phase: str
    status: str  # skipped | ok | failed | not_run
    agent_id: str = ""
    run_id: str = ""
    body: str = ""


@dataclass
class RunReport:
    task: str
    modules: list[str]
    runtime: str
    model: str
    phases: list[PhaseRecord] = field(default_factory=list)
    verify_exit_code: int = 0
    verify_log: str = ""
    git_diff_stat: str = ""
    summary_body: str = ""

    @property
    def verify_status(self) -> str:
        return "PASS" if self.verify_exit_code == 0 else f"FAIL (exit {self.verify_exit_code})"


def _git_diff_stat() -> str:
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    stat = (result.stdout or "").strip()
    if stat:
        return stat
    staged = subprocess.run(
        ["git", "diff", "--stat", "--cached"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return (staged.stdout or "").strip() or "(no git diff stat — clean tree or not a git repo)"


def run_verify_with_log() -> tuple[int, str]:
    """Run E2E gate; return exit code and markdown log lines."""
    lines: list[str] = []
    for label, command in VERIFY_COMMANDS:
        print(f"\n=== verify:{label} ===", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        status = "PASS" if completed.returncode == 0 else f"FAIL (exit {completed.returncode})"
        lines.append(f"- **{label}**: {status}")
        if completed.returncode != 0:
            print(f"VERIFY FAILED at step: {label} (exit {completed.returncode})", flush=True)
            return completed.returncode, "\n".join(lines)
    lines.append("- **overall**: PASS")
    print("\nVERIFY PASS — all three E2E steps succeeded.", flush=True)
    return 0, "\n".join(lines)


def build_run_report_markdown(report: RunReport) -> str:
    """Assemble a consolidated markdown report from phase records."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    modules_text = _format_modules(report.modules)
    lines = [
        "# Cross Section Studio — Orchestration Run Report",
        "",
        f"- **When**: {timestamp}",
        f"- **Task**: {report.task}",
        f"- **Runtime**: {report.runtime}",
        f"- **Model**: {report.model}",
        f"- **Verify**: {report.verify_status}",
        "",
        "## Module scope",
        "",
        modules_text,
        "",
        "## Phases",
        "",
    ]
    for phase in report.phases:
        lines.append(f"### {phase.phase} ({phase.status})")
        if phase.agent_id:
            lines.append(f"- agent_id: `{phase.agent_id}`")
        if phase.run_id:
            lines.append(f"- run_id: `{phase.run_id}`")
        if phase.body.strip():
            lines.append("")
            lines.append(phase.body.strip())
        lines.append("")

    lines.extend(
        [
            "## Verify log",
            "",
            report.verify_log or "(verify not run)",
            "",
            "## Git diff stat",
            "",
            "```",
            report.git_diff_stat or "(none)",
            "```",
            "",
        ]
    )
    if report.summary_body.strip():
        lines.extend(["## Summary agent", "", report.summary_body.strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def write_run_report(path: Path, report: RunReport) -> Path:
    resolved = path if path.is_absolute() else ROOT / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(build_run_report_markdown(report), encoding="utf-8")
    print(f"Wrote orchestration report: {resolved}", flush=True)
    return resolved


def _load_prompt(name: str, **values: str) -> str:
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    text = path.read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text


def _format_modules(modules: Sequence[str]) -> str:
    if not modules:
        return "(infer from task — see AGENTS.md routing table)"
    return "\n".join(f"- `{module}`" for module in modules)


def _parse_modules(modules_arg: str) -> list[str]:
    return [item.strip() for item in modules_arg.split(",") if item.strip()]


def module_boundary_warnings(modules: Sequence[str]) -> list[str]:
    """Return human-readable warnings when --modules spans risky boundaries."""
    basenames = {Path(module).name for module in modules}
    warnings: list[str] = []

    forbidden_core = {"app.py", "pipeline.py", "stratigraphy.py"}
    if forbidden_core.issubset(basenames):
        warnings.append(
            "modules span app.py + pipeline.py + stratigraphy.py — split into separate implement passes"
        )

    has_stratigraphy = "stratigraphy.py" in basenames
    has_renderer = any(name.startswith("renderer") for name in basenames)
    if has_stratigraphy and has_renderer:
        warnings.append(
            "modules span stratigraphy.py and renderer*.py — correlate in stratigraphy first, then renderer"
        )

    def _is_ui_module(name: str) -> bool:
        return (
            name in {"ui_helpers.py", "ui_output_presets.py", "ai_assistant.py"}
            or (name.startswith("app") and name.endswith(".py"))
        )

    engine_core = {
        "pipeline.py",
        "stratigraphy.py",
        "projection.py",
        "ingestion.py",
        "parsing.py",
        "parse_ops.py",
        "models.py",
        "constants.py",
        "lithology_codes.py",
        "report_export.py",
        "render_theme.py",
        "render_profiles.py",
        "renderer.py",
        "renderer_common.py",
        "renderer_chart.py",
        "renderer_consulting.py",
        "renderer_section_sheet.py",
        "workbook_template.py",
        "section_build_request.py",
        "transect_planner.py",
        "ai_quality.py",
    }
    svg_first_triad = {"app_services.py", "pipeline.py", "section_build_request.py"}
    has_ui = any(_is_ui_module(name) for name in basenames)
    engine_overlap = basenames & engine_core
    if has_ui and engine_overlap and not basenames.issubset(svg_first_triad):
        warnings.append("modules mix UI (app*.py) and engine core — keep one boundary per implement pass")

    has_ops = any(name.startswith("ops_") and name.endswith(".py") for name in basenames)
    if has_ops and basenames & engine_core:
        warnings.append("modules mix ops_*.py and engine core — keep one boundary per implement pass")

    has_projection = "projection.py" in basenames
    if has_projection and has_renderer:
        warnings.append(
            "modules span projection.py and renderer*.py — keep projection math out of renderer"
        )

    return warnings


def _run_result_failed(result) -> bool:
    status = getattr(result, "status", "")
    if status == "error":
        return True
    status_text = str(status)
    if status_text.endswith(".error"):
        return True
    return status_text == "error"


def _git_diff_summary() -> str:
    result = subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    diff = (result.stdout or "").strip()
    if not diff:
        staged = subprocess.run(
            ["git", "diff", "--cached"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        diff = (staged.stdout or "").strip()
    if not diff:
        return "(no git diff — working tree clean or not a git repo)"
    max_chars = 12000
    if len(diff) > max_chars:
        return diff[:max_chars] + f"\n\n... truncated ({len(diff) - max_chars} chars omitted)"
    return diff


def run_verify_local() -> int:
    """Run the deterministic three-step E2E gate. Returns exit code."""
    exit_code, _ = run_verify_with_log()
    return exit_code


def _resolve_api_key(explicit: str | None) -> str:
    key = (explicit or os.environ.get("CURSOR_API_KEY") or "").strip()
    if not key:
        raise SystemExit(
            "CURSOR_API_KEY is required for agent phases. "
            "Set the environment variable or pass --api-key."
        )
    return key


def _build_agent_options(
    *,
    api_key: str,
    runtime: str,
    model: str,
    name: str,
):
    from cursor_sdk import AgentOptions, CloudAgentOptions, CloudRepository, LocalAgentOptions

    if runtime == "local":
        return AgentOptions(
            api_key=api_key,
            model=model,
            name=name,
            local=LocalAgentOptions(cwd=str(ROOT)),
        )
    if runtime == "cloud":
        remote = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        repo_url = (remote.stdout or "").strip()
        if not remote.returncode == 0 or not repo_url:
            raise SystemExit("Cloud runtime requires a git origin remote URL.")
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        ref = (branch.stdout or "main").strip() or "main"
        return AgentOptions(
            api_key=api_key,
            model=model,
            name=name,
            cloud=CloudAgentOptions(
                repos=[CloudRepository(url=repo_url, starting_ref=ref)],
            ),
        )
    raise SystemExit(f"Unknown runtime: {runtime!r} (use local or cloud)")


async def _run_agent_phase(
    client,
    *,
    api_key: str,
    runtime: str,
    model: str,
    phase_name: str,
    prompt: str,
) -> tuple[str, str, str]:
    """Spawn one agent phase; return (agent_id, run_id, result_text)."""
    from cursor_sdk import CursorAgentError

    options = _build_agent_options(
        api_key=api_key,
        runtime=runtime,
        model=model,
        name=f"cross-section-{phase_name}",
    )
    agent = await client.create_agent(options)
    try:
        run = await agent.send(prompt)
        print(f"phase={phase_name} agent_id={agent.agent_id} run_id={run.id}", flush=True)
        try:
            result = await run.wait()
        except CursorAgentError as exc:
            raise SystemExit(f"phase={phase_name} startup failed: {exc}") from exc
        if _run_result_failed(result):
            raise SystemExit(
                f"phase={phase_name} run failed: agent_id={result.agent_id} run_id={result.id}"
            )
        text = (result.result or "").strip()
        print(f"\n--- {phase_name} output ---\n{text}\n--- end {phase_name} ---\n", flush=True)
        return agent.agent_id, run.id, text
    finally:
        await agent.close()


async def _orchestrate_async(args: argparse.Namespace) -> tuple[int, RunReport]:
    from cursor_sdk import AsyncClient, CursorAgentError

    api_key = _resolve_api_key(args.api_key)
    modules = _parse_modules(args.modules)
    modules_text = _format_modules(modules)
    report = RunReport(
        task=args.task,
        modules=modules,
        runtime=args.runtime,
        model=args.model,
        git_diff_stat=_git_diff_stat(),
    )

    for warning in module_boundary_warnings(modules):
        print(f"BOUNDARY WARNING: {warning}", flush=True)

    try:
        client = await AsyncClient.launch_bridge(workspace=str(ROOT))
    except CursorAgentError as exc:
        raise SystemExit(f"Failed to launch SDK bridge: {exc}") from exc

    scout_output = "(scout skipped)"
    implement_output = "(implement skipped)"
    review_output = "(review skipped)"

    async with client:
        if args.skip_scout:
            report.phases.append(PhaseRecord(phase="scout", status="skipped", body=scout_output))
        else:
            scout_prompt = _load_prompt("scout", task=args.task, modules=modules_text)
            agent_id, run_id, scout_output = await _run_agent_phase(
                client,
                api_key=api_key,
                runtime=args.runtime,
                model=args.model,
                phase_name="scout",
                prompt=scout_prompt,
            )
            report.phases.append(
                PhaseRecord(
                    phase="scout",
                    status="ok",
                    agent_id=agent_id,
                    run_id=run_id,
                    body=scout_output,
                )
            )

        if args.skip_implement:
            report.phases.append(PhaseRecord(phase="implement", status="skipped", body=implement_output))
        else:
            implement_prompt = _load_prompt(
                "implement",
                task=args.task,
                modules=modules_text,
                scout_output=scout_output,
            )
            agent_id, run_id, implement_output = await _run_agent_phase(
                client,
                api_key=api_key,
                runtime=args.runtime,
                model=args.model,
                phase_name="implement",
                prompt=implement_prompt,
            )
            report.phases.append(
                PhaseRecord(
                    phase="implement",
                    status="ok",
                    agent_id=agent_id,
                    run_id=run_id,
                    body=implement_output,
                )
            )

        report.verify_exit_code, report.verify_log = run_verify_with_log()
        report.git_diff_stat = _git_diff_stat()
        if report.verify_exit_code != 0:
            report.phases.append(PhaseRecord(phase="verify", status="failed", body=report.verify_log))
            return report.verify_exit_code, report

        report.phases.append(PhaseRecord(phase="verify", status="ok", body=report.verify_log))

        if args.review:
            review_prompt = _load_prompt(
                "review",
                task=args.task,
                diff_summary=_git_diff_summary(),
            )
            agent_id, run_id, review_output = await _run_agent_phase(
                client,
                api_key=api_key,
                runtime=args.runtime,
                model=args.model,
                phase_name="review",
                prompt=review_prompt,
            )
            report.phases.append(
                PhaseRecord(
                    phase="review",
                    status="ok",
                    agent_id=agent_id,
                    run_id=run_id,
                    body=review_output,
                )
            )
        else:
            report.phases.append(PhaseRecord(phase="review", status="skipped", body=review_output))

        if args.summary_agent:
            summary_prompt = _load_prompt(
                "summary",
                task=args.task,
                modules=modules_text,
                scout_output=scout_output,
                implement_output=implement_output,
                verify_log=report.verify_log,
                review_output=review_output,
                diff_stat=report.git_diff_stat,
            )
            agent_id, run_id, summary_body = await _run_agent_phase(
                client,
                api_key=api_key,
                runtime=args.runtime,
                model=args.model,
                phase_name="summary",
                prompt=summary_prompt,
            )
            report.summary_body = summary_body
            report.phases.append(
                PhaseRecord(
                    phase="summary",
                    status="ok",
                    agent_id=agent_id,
                    run_id=run_id,
                    body=summary_body,
                )
            )

    return 0, report


def _resolve_report_path(report_arg: str | None) -> Path | None:
    if report_arg is None:
        return None
    if report_arg == "":
        return DEFAULT_REPORT_PATH
    return Path(report_arg)


def _cmd_run(args: argparse.Namespace) -> int:
    report_path = _resolve_report_path(getattr(args, "report", None))
    if args.verify_only:
        verify_code, verify_log = run_verify_with_log()
        if report_path is not None:
            write_run_report(
                report_path,
                RunReport(
                    task=args.task,
                    modules=_parse_modules(args.modules),
                    runtime=args.runtime,
                    model=args.model,
                    verify_exit_code=verify_code,
                    verify_log=verify_log,
                    git_diff_stat=_git_diff_stat(),
                    phases=[PhaseRecord(phase="verify", status="ok" if verify_code == 0 else "failed", body=verify_log)],
                ),
            )
        return verify_code

    exit_code, report = asyncio.run(_orchestrate_async(args))
    if report_path is not None:
        write_run_report(report_path, report)
    return exit_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hierarchical Cursor agent supervisor for Cross Section Studio.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Scout → implement → verify (→ optional review)")
    run.add_argument("--task", required=True, help="Task description for agents")
    run.add_argument(
        "--modules",
        default="",
        help="Comma-separated module paths (e.g. renderer_common.py,renderer.py)",
    )
    run.add_argument(
        "--runtime",
        choices=("local", "cloud"),
        default="local",
        help="Agent runtime (default: local)",
    )
    run.add_argument("--model", default=DEFAULT_MODEL, help=f"Model id (default: {DEFAULT_MODEL})")
    run.add_argument("--api-key", default=None, help="Cursor API key (else CURSOR_API_KEY env)")
    run.add_argument("--review", action="store_true", help="Run bugbot-style review after verify")
    run.add_argument("--skip-scout", action="store_true", help="Skip scout phase")
    run.add_argument("--skip-implement", action="store_true", help="Skip implement phase")
    run.add_argument(
        "--verify-only",
        action="store_true",
        help="Run local E2E gate only (no API key required)",
    )
    run.add_argument(
        "--report",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="Write consolidated markdown report (default: orchestration_reports/latest_run.md)",
    )
    run.add_argument(
        "--summary-agent",
        action="store_true",
        help="After verify, spawn summary agent for executive report section",
    )
    run.set_defaults(func=_cmd_run)

    verify = sub.add_parser("verify", help="Run local E2E gate only (no API key)")
    verify.add_argument(
        "--report",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="Write verify-only markdown report",
    )
    verify.add_argument("--task", default="verify-only", help="Task label for report header")

    def _cmd_verify(args: argparse.Namespace) -> int:
        verify_code, verify_log = run_verify_with_log()
        report_path = _resolve_report_path(args.report)
        if report_path is not None:
            write_run_report(
                report_path,
                RunReport(
                    task=args.task,
                    modules=[],
                    runtime="local",
                    model=DEFAULT_MODEL,
                    verify_exit_code=verify_code,
                    verify_log=verify_log,
                    git_diff_stat=_git_diff_stat(),
                    phases=[
                        PhaseRecord(
                            phase="verify",
                            status="ok" if verify_code == 0 else "failed",
                            body=verify_log,
                        )
                    ],
                ),
            )
        return verify_code

    verify.set_defaults(func=_cmd_verify)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
