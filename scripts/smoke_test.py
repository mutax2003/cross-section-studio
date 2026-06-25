"""End-to-end pipeline smoke test without Streamlit UI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import DataParser, Transect
from projection import project_boreholes
from renderer import CrossSectionRenderer
from stratigraphy import build_stratigraphy

SAMPLE = ROOT / "data" / "sample_boreholes.xlsx"
OUTPUT = ROOT / "data" / "smoke_test_output.svg"


def main() -> None:
    parse_result = DataParser().parse_file(SAMPLE)
    assert not parse_result.errors, parse_result.errors

    hole_ids = [collar.hole_id for collar in parse_result.collars]
    collar_lookup = {collar.hole_id: collar for collar in parse_result.collars}
    transect = Transect(
        points=[(collar_lookup[hole_id].easting, collar_lookup[hole_id].northing) for hole_id in hole_ids]
    )

    projected = project_boreholes(parse_result.collars, parse_result.lithologies, transect)
    polygons = build_stratigraphy(projected)
    collar_depths = {collar.hole_id: collar.total_depth for collar in parse_result.collars}

    renderer = CrossSectionRenderer(vertical_exaggeration=5.0)
    figure = renderer.render(polygons, projected, collar_depths=collar_depths)
    svg_bytes = renderer.to_svg_bytes(figure)
    OUTPUT.write_bytes(svg_bytes)

    print(f"Holes: {len(hole_ids)}")
    print(f"Projected intervals: {len(projected)}")
    print(f"Polygons: {len(polygons)}")
    print(f"Wrote {OUTPUT} ({len(svg_bytes)} bytes)")


if __name__ == "__main__":
    main()
