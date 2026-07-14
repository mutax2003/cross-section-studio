"""Optional per-file pytest diagnostic — not part of VERIFY_COMMANDS.

Re-runs a curated test file list and writes per-file pass/fail lines to
``e2e_test_results.txt``. The canonical E2E gate is ``python -m pytest -q``
plus smoke scripts (see ``AGENTS.md`` / ``agent_supervisor.VERIFY_COMMANDS``).
Use this script when you need a persisted per-file log, not for routine verify.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
    from subprocess import TimeoutExpired
except ImportError:  # pragma: no cover
    TimeoutExpired = subprocess.TimeoutExpired  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "e2e_test_results.txt"
SAMPLE = ROOT / "data" / "sample_boreholes.xlsx"

TEST_FILES = [
    "tests/test_projection.py",
    "tests/test_stratigraphy.py",
    "tests/test_pipeline.py",
    "tests/test_renderer_styles.py",
    "tests/test_app_helpers.py",
    "tests/test_ingestion.py",
    "tests/test_e2e_edge_cases.py",
    "tests/test_ecoventure_gwm_figures.py",
    "tests/test_consulting_reference_parity.py",
    "tests/test_advantage_conversion.py",
    "tests/test_ai_quality.py",
    "tests/test_ai_assistant.py",
    "tests/test_e2e_consulting_ai.py",
    "tests/test_section_build_request.py",
    "tests/test_paths.py",
    "tests/test_geology_qc.py",
    "tests/test_launcher_port.py",
    "tests/test_ui_helpers.py",
    "tests/test_import_smoke.py",
    "tests/test_unit_order_auto.py",
    "tests/test_agent_supervisor.py",
    "tests/test_agent_supervisor_edge_cases.py",
    "tests/test_figure_improvements.py",
    "tests/test_ui_output_presets.py",
    "tests/test_streamlit_app.py",
    "tests/test_menubar.py",
    "tests/test_ops.py",
    "tests/test_water_quality.py",
    "tests/test_workbook_template.py",
    "tests/test_advantage_p2_reference.py",
]


def main() -> int:
    lines: list[str] = []

    def log(message: str) -> None:
        lines.append(message)
        print(message, flush=True)
        LOG.write_text("\n".join(lines), encoding="utf-8")

    log("=== Cross Section Studio E2E test run ===")

    if not SAMPLE.exists():
        log("Generating sample_boreholes.xlsx ...")
        gen = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_sample_data.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        log(gen.stdout.strip() or "(no stdout)")
        if gen.returncode != 0:
            log(gen.stderr.strip() or "(no stderr)")
            return gen.returncode

    total_failed = 0
    for test_file in TEST_FILES:
        path = ROOT / test_file
        if not path.exists():
            log(f"SKIP missing {test_file}")
            continue
        log(f"RUN {test_file} ...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_file, "-q", "--tb=line"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except TimeoutExpired:
            log(f"TIMEOUT on {test_file}")
            return 124
        summary = (result.stdout or "").strip().splitlines()[-1:] or ["(no output)"]
        log(f"  {summary[0]}")
        if result.returncode != 0:
            total_failed += 1
            if result.stdout:
                log(result.stdout[-2000:])
            if result.stderr:
                log(result.stderr[-2000:])

    log(f"done failed_files={total_failed}")
    return total_failed


if __name__ == "__main__":
    raise SystemExit(main())
