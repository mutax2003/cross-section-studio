"""End-to-end pipeline smoke test without Streamlit UI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import DataParser
from pipeline import build_cross_section

SAMPLE = ROOT / "data" / "sample_boreholes.xlsx"
OUTPUT = ROOT / "data" / "smoke_test_output.svg"


def main() -> None:
    parse_result = DataParser().parse_file(SAMPLE)
    assert not parse_result.errors, parse_result.errors

    hole_ids = [collar.hole_id for collar in parse_result.collars]
    collar_lookup = {collar.hole_id: collar for collar in parse_result.collars}
    transect_points = [
        (collar_lookup[hole_id].easting, collar_lookup[hole_id].northing) for hole_id in hole_ids
    ]

    _, polygons, svg_bytes, _, _, lithology_codes, overlap_warnings = build_cross_section(
        parse_result.collars,
        parse_result.lithologies,
        transect_points,
        vertical_exaggeration=5.0,
    )
    OUTPUT.write_bytes(svg_bytes)

    print(f"Holes: {len(hole_ids)}")
    print(f"Polygons: {len(polygons)}")
    print(f"Lithology codes: {len(lithology_codes)}")
    print(f"Overlap warnings: {len(overlap_warnings)}")
    print(f"Wrote {OUTPUT} ({len(svg_bytes)} bytes)")


if __name__ == "__main__":
    main()
