"""Regenerate data/bh_log_lithology_legend.json from BH Log Lithology Legend.xlsx."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from constants import parse_bh_log_legend_xlsx
from paths import bh_log_lithology_legend_path, bh_log_lithology_legend_xlsx_path


def convert_legend(source: Path | None = None, output: Path | None = None) -> Path:
    source = source or bh_log_lithology_legend_xlsx_path()
    output = output or bh_log_lithology_legend_path()
    payload = parse_bh_log_legend_xlsx(source)
    if not payload:
        raise SystemExit(f"No lithology colours parsed from {source}")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> None:
    path = convert_legend()
    print(f"Wrote {path} ({len(json.loads(path.read_text(encoding='utf-8')))} codes)")


if __name__ == "__main__":
    main()
