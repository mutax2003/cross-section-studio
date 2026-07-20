"""Build multi-tab data/Cross_Section_Input_Template.xlsx."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workbook_template import build_input_template


def main() -> None:
    path = build_input_template()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
