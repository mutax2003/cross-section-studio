"""Generate sample borehole workbook for smoke testing."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "data" / "sample_boreholes.xlsx"


def main() -> None:
    collars = pd.DataFrame(
        [
            {"hole_id": "BH-01", "easting": 500000.0, "northing": 4500000.0, "elevation": 102.5, "total_depth": 25.0},
            {"hole_id": "BH-02", "easting": 500050.0, "northing": 4500000.0, "elevation": 101.8, "total_depth": 28.0},
            {"hole_id": "BH-03", "easting": 500100.0, "northing": 4500000.0, "elevation": 100.9, "total_depth": 30.0},
            {"hole_id": "BH-04", "easting": 500150.0, "northing": 4500000.0, "elevation": 100.2, "total_depth": 32.0},
        ]
    )

    lithology = pd.DataFrame(
        [
            {"hole_id": "BH-01", "from_depth": 0.0, "to_depth": 3.0, "lithology_code": "Sandstone"},
            {"hole_id": "BH-01", "from_depth": 3.0, "to_depth": 10.0, "lithology_code": "Clay"},
            {"hole_id": "BH-01", "from_depth": 10.0, "to_depth": 25.0, "lithology_code": "Silt"},
            {"hole_id": "BH-02", "from_depth": 0.0, "to_depth": 4.0, "lithology_code": "Sandstone"},
            {"hole_id": "BH-02", "from_depth": 4.0, "to_depth": 12.0, "lithology_code": "Clay"},
            {"hole_id": "BH-02", "from_depth": 12.0, "to_depth": 28.0, "lithology_code": "Bedrock"},
            {"hole_id": "BH-03", "from_depth": 0.0, "to_depth": 5.0, "lithology_code": "Sandstone"},
            {"hole_id": "BH-03", "from_depth": 5.0, "to_depth": 18.0, "lithology_code": "Silt"},
            {"hole_id": "BH-03", "from_depth": 18.0, "to_depth": 30.0, "lithology_code": "Bedrock"},
            {"hole_id": "BH-04", "from_depth": 0.0, "to_depth": 6.0, "lithology_code": "Gravel"},
            {"hole_id": "BH-04", "from_depth": 6.0, "to_depth": 16.0, "lithology_code": "Silt"},
            {"hole_id": "BH-04", "from_depth": 16.0, "to_depth": 32.0, "lithology_code": "Bedrock"},
        ]
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        collars.to_excel(writer, sheet_name="Collars", index=False)
        lithology.to_excel(writer, sheet_name="Lithology", index=False)

    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
