"""Build a filled multi-tab workbook for manual / UI testing.

Writes ``data/test_workbook.xlsx`` with Collars, Lithology, Water, Screens,
Gradients, Environmental, and Project sheets — suitable for Upload → Validate →
Configure → Generate (SVG-first) in Streamlit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "data" / "test_workbook.xlsx"


def main() -> Path:
    project = pd.DataFrame(
        [
            {"field": "client_name", "value": "Cross Section Studio QA"},
            {"field": "prepared_by", "value": "Test Lab"},
            {"field": "project_number", "value": "TEST-2026-001"},
            {"field": "section_title", "value": "Test Section A–A'"},
            {"field": "report_date", "value": "14/07/26"},
            {"field": "drawn_by", "value": "AI"},
            {"field": "coordinate_reference", "value": "EPSG:32611"},
            {"field": "transect_start", "value": "A / WEST"},
            {"field": "transect_end", "value": "A' / EAST"},
            {"field": "vertical_exaggeration", "value": "5"},
            {"field": "notes", "value": "Synthetic test workbook — safe to regenerate."},
        ]
    )

    collars = pd.DataFrame(
        [
            {"hole_id": "MW-01", "easting": 500000.0, "northing": 4500000.0, "elevation": 102.5, "total_depth": 25.0},
            {"hole_id": "MW-02", "easting": 500040.0, "northing": 4500005.0, "elevation": 101.8, "total_depth": 28.0},
            {"hole_id": "MW-03", "easting": 500080.0, "northing": 4500010.0, "elevation": 100.9, "total_depth": 30.0},
            {"hole_id": "MW-04", "easting": 500120.0, "northing": 4500012.0, "elevation": 100.2, "total_depth": 32.0},
            # Off-transect collar (~45 m north) — useful for offset-threshold tests
            {"hole_id": "MW-05", "easting": 500060.0, "northing": 4500045.0, "elevation": 101.0, "total_depth": 20.0},
        ]
    )

    lithology = pd.DataFrame(
        [
            {"hole_id": "MW-01", "from_depth": 0.0, "to_depth": 3.0, "lithology_code": "Sand", "unit_order": 1},
            {"hole_id": "MW-01", "from_depth": 3.0, "to_depth": 10.0, "lithology_code": "Clay", "unit_order": 2},
            {"hole_id": "MW-01", "from_depth": 10.0, "to_depth": 25.0, "lithology_code": "Till", "unit_order": 3},
            {"hole_id": "MW-02", "from_depth": 0.0, "to_depth": 4.0, "lithology_code": "Sand", "unit_order": 1},
            {"hole_id": "MW-02", "from_depth": 4.0, "to_depth": 12.0, "lithology_code": "Clay", "unit_order": 2},
            {"hole_id": "MW-02", "from_depth": 12.0, "to_depth": 28.0, "lithology_code": "Bedrock", "unit_order": 3},
            {"hole_id": "MW-03", "from_depth": 0.0, "to_depth": 5.0, "lithology_code": "Sand", "unit_order": 1},
            {"hole_id": "MW-03", "from_depth": 5.0, "to_depth": 18.0, "lithology_code": "Silt", "unit_order": 2},
            {"hole_id": "MW-03", "from_depth": 18.0, "to_depth": 30.0, "lithology_code": "Bedrock", "unit_order": 3},
            {"hole_id": "MW-04", "from_depth": 0.0, "to_depth": 6.0, "lithology_code": "Gravel", "unit_order": 1},
            {"hole_id": "MW-04", "from_depth": 6.0, "to_depth": 16.0, "lithology_code": "Silt", "unit_order": 2},
            {"hole_id": "MW-04", "from_depth": 16.0, "to_depth": 32.0, "lithology_code": "Bedrock", "unit_order": 3},
            {"hole_id": "MW-05", "from_depth": 0.0, "to_depth": 8.0, "lithology_code": "Clay", "unit_order": 1},
            {"hole_id": "MW-05", "from_depth": 8.0, "to_depth": 20.0, "lithology_code": "Till", "unit_order": 2},
        ]
    )

    # Sheet XOR: each row uses depth OR elevation_masl (not both)
    water = pd.DataFrame(
        [
            {"hole_id": "MW-01", "depth": 4.5, "elevation_masl": None, "series_id": "2026-05", "series_label": "May 2026"},
            {"hole_id": "MW-02", "depth": 5.0, "elevation_masl": None, "series_id": "2026-05", "series_label": "May 2026"},
            {"hole_id": "MW-03", "depth": None, "elevation_masl": 96.5, "series_id": "2026-05", "series_label": "May 2026"},
            {"hole_id": "MW-04", "depth": 6.2, "elevation_masl": None, "series_id": "2026-05", "series_label": "May 2026"},
            {"hole_id": "MW-01", "depth": 5.1, "elevation_masl": None, "series_id": "2026-07", "series_label": "Jul 2026"},
            {"hole_id": "MW-02", "depth": 5.4, "elevation_masl": None, "series_id": "2026-07", "series_label": "Jul 2026"},
            {"hole_id": "MW-03", "depth": 5.8, "elevation_masl": None, "series_id": "2026-07", "series_label": "Jul 2026"},
        ]
    )

    screens = pd.DataFrame(
        [
            {"hole_id": "MW-01", "from_depth": 8.0, "to_depth": 12.0},
            {"hole_id": "MW-02", "from_depth": 14.0, "to_depth": 18.0},
            {"hole_id": "MW-03", "from_depth": 16.0, "to_depth": 22.0},
            {"hole_id": "MW-04", "from_depth": 18.0, "to_depth": 24.0},
        ]
    )

    gradients = pd.DataFrame(
        [
            {"hole_id": "MW-01", "direction": "down"},
            {"hole_id": "MW-02", "direction": "up"},
            {"hole_id": "MW-03", "direction": "down"},
        ]
    )

    environmental = pd.DataFrame(
        [
            {"hole_id": "MW-01", "parameter": "Chloride", "value": 120.0, "depth": 10.0, "unit": "mg/L", "value_label": "120 mg/L"},
            {"hole_id": "MW-02", "parameter": "Chloride", "value": 85.0, "depth": 15.0, "unit": "mg/L", "value_label": "85 mg/L"},
            {"hole_id": "MW-03", "parameter": "Chloride", "value": 210.0, "depth": 18.0, "unit": "mg/L", "value_label": "210 mg/L"},
            {"hole_id": "MW-04", "parameter": "Chloride", "value": 45.0, "depth": 20.0, "unit": "mg/L", "value_label": "45 mg/L"},
            {"hole_id": "MW-02", "parameter": "pH", "value": 7.2, "depth": 15.0, "unit": "-", "value_label": "pH 7.2"},
            {"hole_id": "MW-03", "parameter": "pH", "value": 6.8, "depth": 18.0, "unit": "-", "value_label": "pH 6.8"},
        ]
    )

    how_to = pd.DataFrame(
        {
            "step": [1, 2, 3, 4, 5],
            "action": [
                "In Streamlit sidebar, upload this file (data/test_workbook.xlsx).",
                "Validate: check Data Health, groundwater series, chloride coverage.",
                "Configure: choose MW-01 → MW-04 (omit MW-05 to avoid off-transect noise, or include it to test offsets).",
                "Generate: SVG should appear immediately; use Prepare PNG / Prepare PDF for deliverables.",
                "Regenerate anytime: python scripts/build_test_workbook.py",
            ],
        }
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        how_to.to_excel(writer, sheet_name="HowTo", index=False)
        project.to_excel(writer, sheet_name="Project", index=False)
        collars.to_excel(writer, sheet_name="Collars", index=False)
        lithology.to_excel(writer, sheet_name="Lithology", index=False)
        water.to_excel(writer, sheet_name="Water", index=False)
        screens.to_excel(writer, sheet_name="Screens", index=False)
        gradients.to_excel(writer, sheet_name="Gradients", index=False)
        environmental.to_excel(writer, sheet_name="Environmental", index=False)

    print(f"Wrote {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    main()
