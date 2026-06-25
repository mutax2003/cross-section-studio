"""Direct end-to-end smoke checks without pytest (fast sanity path)."""

from __future__ import annotations

import sys
import traceback
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

from ingestion import ingest_workbook
from models import Collar, DataParser, Lithology, ParseResult, Transect, WaterLevel, subset_parse_result
from pipeline import build_cross_section, validate_interpretation_mode
from projection import project_boreholes, project_collar_to_transect
from stratigraphy import build_stratigraphy, detect_polygon_overlaps
from tests.conftest import assert_valid_svg, make_field_export_bytes, run_pipeline

LOG = ROOT / "e2e_smoke_results.txt"


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"PASS {name}", flush=True)
    except Exception as exc:
        print(f"FAIL {name}: {exc}", flush=True)
        traceback.print_exc()
        raise


def main() -> int:
    lines: list[str] = []

    def run(name: str, fn) -> None:
        try:
            fn()
            line = f"PASS {name}"
        except Exception as exc:
            line = f"FAIL {name}: {exc}"
            lines.append(line)
            lines.append(traceback.format_exc())
            LOG.write_text("\n".join(lines), encoding="utf-8")
            print(line, flush=True)
            raise
        lines.append(line)
        print(line, flush=True)
        LOG.write_text("\n".join(lines), encoding="utf-8")

    sample = ROOT / "data" / "sample_boreholes.xlsx"
    if not sample.exists():
        import subprocess

        subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_sample_data.py")], check=True)

    run("validate_interpretation_mode", lambda: validate_interpretation_mode("interpolated"))
    run(
        "project_collar_to_transect",
        lambda: (
            project_collar_to_transect(25.0, 0.0, Transect(points=[(0.0, 0.0), (100.0, 0.0)]))[0] == 25.0
        ) or (_ for _ in ()).throw(AssertionError("bad projection")),
    )
    run(
        "two_hole_pipeline",
        lambda: (
            len(run_pipeline(
                [
                    Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
                    Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
                ],
                [
                    Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
                    Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
                ],
                [(0.0, 0.0), (50.0, 0.0)],
            )[2])
            > 100
        ),
    )
    run(
        "field_export_ingest_render",
        lambda: _field_export_smoke(),
    )
    run(
        "sample_workbook_pipeline",
        lambda: _sample_workbook_smoke(sample),
    )
    run(
        "borehole_only_mode",
        lambda: _borehole_only_smoke(),
    )
    run(
        "overlap_detection",
        lambda: _overlap_smoke(),
    )
    run(
        "subset_and_water",
        lambda: _water_smoke(),
    )

    summary = f"ALL {len(lines)} CHECKS PASSED"
    lines.append(summary)
    LOG.write_text("\n".join(lines), encoding="utf-8")
    print(summary, flush=True)
    return 0


def _field_export_smoke() -> None:
    raw = make_field_export_bytes(
        [
            {"hole_id": "BH-01", "depth": "0.00-2.00m", "lithology": "clay", "lat": 58.57, "long": -119.19},
            {"hole_id": "BH-02", "depth": "0-3m", "lithology": "silt", "lat": 58.571, "long": -119.189},
        ]
    )
    parse_result, report = ingest_workbook(BytesIO(raw), profile_id="field_export_v1")
    assert report.hole_count == 2
    lookup = {c.hole_id: c for c in parse_result.collars}
    _, polygons, svg = run_pipeline(
        parse_result.collars,
        parse_result.lithologies,
        [
            (lookup["BH-01"].easting, lookup["BH-01"].northing),
            (lookup["BH-02"].easting, lookup["BH-02"].northing),
        ],
    )
    assert len(polygons) >= 1
    assert_valid_svg(svg)


def _sample_workbook_smoke(sample: Path) -> None:
    parse_result = DataParser().parse_file(sample)
    assert len(parse_result.collars) == 4
    lookup = {c.hole_id: c for c in parse_result.collars}
    hole_ids = sorted(lookup, key=lambda h: lookup[h].easting)
    points = [(lookup[h].easting, lookup[h].northing) for h in hole_ids]
    projected, polygons, svg = run_pipeline(
        parse_result.collars,
        parse_result.lithologies,
        points,
        vertical_exaggeration=5.0,
    )
    assert len(projected) == len(parse_result.lithologies)
    assert len(polygons) >= 6
    assert_valid_svg(svg)


def _borehole_only_smoke() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Silt"),
    ]
    _, polygons, svg = run_pipeline(
        collars,
        lithologies,
        [(0.0, 0.0), (50.0, 0.0)],
        interpretation_mode="borehole_only",
    )
    assert polygons == []
    assert_valid_svg(svg)


def _overlap_smoke() -> None:
    from shapely.geometry import Polygon as ShapelyPolygon

    from stratigraphy import GeologicalPolygon

    pair = ("BH-01", "BH-02")
    left = GeologicalPolygon("Clay", ShapelyPolygon([(0, 10), (25, 10), (25, 5), (0, 5)]), pair)
    right = GeologicalPolygon("Silt", ShapelyPolygon([(0, 8), (25, 8), (25, 3), (0, 3)]), pair)
    overlaps = detect_polygon_overlaps([left, right])
    assert len(overlaps) == 1


def _water_smoke() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
    ]
    _, _, svg, _, _ = build_cross_section(
        collars,
        lithologies,
        [(0.0, 0.0), (50.0, 0.0)],
        water_levels=[WaterLevel(hole_id="BH-01", depth=3.0), WaterLevel(hole_id="BH-02", depth=4.0)],
    )
    assert_valid_svg(svg)
    subset = subset_parse_result(
        ParseResult(
            collars=tuple(collars),
            lithologies=tuple(lithologies),
            errors=(),
            water_levels=(WaterLevel(hole_id="BH-01", depth=3.0),),
        ),
        ("BH-01",),
    )
    assert len(subset.water_levels) == 1


if __name__ == "__main__":
    raise SystemExit(main())
