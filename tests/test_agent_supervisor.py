"""Tests for scripts/agent_supervisor.py (no API key required)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts" / "agent_supervisor.py"


def test_load_prompt_scout_substitution() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import agent_supervisor as sup

    text = sup._load_prompt("scout", task="test task", modules="- `foo.py`")
    assert "test task" in text
    assert "`foo.py`" in text
    assert "{task}" not in text


def test_run_help() -> None:
    result = subprocess.run(
        [sys.executable, str(SUPERVISOR), "run", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--task" in result.stdout
    assert "--verify-only" in result.stdout
    assert "--report" in result.stdout


def test_verify_subcommand_delegates_to_run_verify_with_log() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import agent_supervisor as sup

    with patch.object(sup, "run_verify_with_log", return_value=(0, "- **overall**: PASS")) as mock_verify:
        with patch.object(sys, "argv", ["agent_supervisor.py", "verify"]):
            exit_code = sup.main()

    assert exit_code == 0
    mock_verify.assert_called_once()


def test_verify_subcommand_propagates_failure_code() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import agent_supervisor as sup

    with patch.object(sup, "run_verify_with_log", return_value=(3, "- **pytest**: FAIL (exit 3)")):
        with patch.object(sys, "argv", ["agent_supervisor.py", "verify"]):
            exit_code = sup.main()

    assert exit_code == 3


def test_build_run_report_markdown() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import agent_supervisor as sup

    report = sup.RunReport(
        task="optimize renderer",
        modules=["renderer.py"],
        runtime="local",
        model="composer-2.5",
        verify_exit_code=0,
        verify_log="- **overall**: PASS",
        git_diff_stat=" renderer.py | 2 +-",
        phases=[
            sup.PhaseRecord(phase="scout", status="ok", body="found hot path"),
            sup.PhaseRecord(phase="verify", status="ok", body="- **overall**: PASS"),
        ],
    )
    text = sup.build_run_report_markdown(report)
    assert "optimize renderer" in text
    assert "found hot path" in text
    assert "renderer.py" in text


def test_write_run_report_creates_file(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import agent_supervisor as sup

    target = tmp_path / "reports" / "run.md"
    report = sup.RunReport(
        task="t",
        modules=[],
        runtime="local",
        model="composer-2.5",
    )
    written = sup.write_run_report(target, report)
    assert written.exists()
    assert "Orchestration Run Report" in written.read_text(encoding="utf-8")


def test_verify_subcommand_writes_report(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import agent_supervisor as sup

    report_path = tmp_path / "verify_run.md"
    with patch.object(sup, "run_verify_with_log", return_value=(0, "- **overall**: PASS")):
        with patch.object(sup, "_git_diff_stat", return_value=" pipeline.py | 1 +"):
            with patch.object(
                sys,
                "argv",
                ["agent_supervisor.py", "verify", "--report", str(report_path)],
            ):
                exit_code = sup.main()

    assert exit_code == 0
    text = report_path.read_text(encoding="utf-8")
    assert "verify-only" in text
    assert "pipeline.py" in text


@pytest.mark.slow
def test_verify_subcommand_full_gate() -> None:
    """Full four-step E2E gate via CLI (slow; run with: pytest -m slow)."""
    result = subprocess.run(
        [sys.executable, str(SUPERVISOR), "verify"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
