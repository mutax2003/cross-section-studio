"""Run end-to-end pytest suite and write results to a log file."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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
    "tests/test_advantage_conversion.py",
    "tests/test_ai_quality.py",
    "tests/test_ai_assistant.py",
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
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-q", "--tb=line"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        summary = (result.stdout or "").strip().splitlines()[-1:] or ["(no output)"]
        log(f"  {summary[0]}")
        if result.returncode != 0:
            total_failed += 1
            if result.stdout:
                log(result.stdout[-2000:])
            if result.stderr:
                log(result.stderr[-2000:])
        if result.returncode == 124:
            log(f"TIMEOUT on {test_file}")
            return 124

    log(f"done failed_files={total_failed}")
    return total_failed


if __name__ == "__main__":
    raise SystemExit(main())
