"""End-to-end pipeline tests with geological and data-quality edge cases."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from shapely.geometry import Point

from ai_quality import analyze_parsed_data, analyze_workbook, load_lithology_aliases, read_mapped_sheets
from constants import USGS_LITHOLOGY_COLORS
from ingestion import (
    NATIVE_PROFILE_ID,
    FormatDetector,
    export_platform_workbook,
    ingest_workbook,
    parse_depth_interval,
)
from models import (
    Collar,
    DataParser,
    Lithology,
    ParseResult,
    Transect,
    WaterLevel,
    parse_result_to_json_bundle,
    subset_json_bundle,
    subset_parse_result,
)
from pipeline import build_cross_section
from projection import off_transect_warnings, project_boreholes, suggest_offset_threshold_m
from stratigraphy import build_stratigraphy
from tests.conftest import (
    assert_valid_svg,
    make_field_export_bytes,
    make_workbook_bytes,
    run_pipeline,
)
from transect_planner import recommend_transects
from ui_helpers import holes_missing_lithology

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_WORKBOOK = ROOT / "data" / "sample_boreholes.xlsx"


# --- Happy path ---


@pytest.mark.skipif(not SAMPLE_WORKBOOK.exists(), reason="Run scripts/generate_sample_data.py first")
def test_e2e_sample_workbook_full_pipeline() -> None:
    parse_result = DataParser().parse_file(SAMPLE_WORKBOOK)
    assert len(parse_result.collars) == 4
    assert len(parse_result.lithologies) >= 8

    collar_lookup = {collar.hole_id: collar for collar in parse_result.collars}
    hole_ids = sorted(collar_lookup, key=lambda hole_id: collar_lookup[hole_id].easting)
    transect_points = [(collar_lookup[h].easting, collar_lookup[h].northing) for h in hole_ids]

    projected, polygons, svg_bytes = run_pipeline(
        parse_result.collars,
        parse_result.lithologies,
        transect_points,
        vertical_exaggeration=5.0,
    )
    assert len(projected) == len(parse_result.lithologies)
    assert len(polygons) >= 6
    assert_valid_svg(svg_bytes)


def test_e2e_two_hole_minimal(axis_collars, axis_lithologies) -> None:
    projected, polygons, svg_bytes = run_pipeline(
        axis_collars,
        axis_lithologies,
        [(0.0, 0.0), (100.0, 0.0)],
    )
    assert len(projected) == 4
    assert len(polygons) == 2
    assert projected["x_profile"].nunique() == 2
    assert_valid_svg(svg_bytes)


# --- Ingestion E2E (field export → canonical → render) ---


def test_e2e_field_export_ingest_to_render() -> None:
    raw = make_field_export_bytes(
        [
            {
                "hole_id": "BH-01",
                "depth": "0.00-2.00m",
                "lithology": "silty clay",
                "lat": 58.57,
                "long": -119.19,
            },
            {
                "hole_id": "BH-01",
                "depth": "2.00-4.00m",
                "lithology": "sand",
                "lat": 58.57,
                "long": -119.19,
            },
            {
                "hole_id": "BH-02",
                "depth": "0-3m",
                "lithology": "clay",
                "lat": 58.571,
                "long": -119.189,
            },
        ]
    )
    detection = FormatDetector().detect(BytesIO(raw))
    assert detection.profile_id == "field_export_v1"

    parse_result, report = ingest_workbook(BytesIO(raw), profile_id="field_export_v1")
    assert report.hole_count == 2
    assert report.lithology_interval_count == 3
    assert "Silty Clay" in {lit.lithology_code for lit in parse_result.lithologies}

    collar_lookup = {c.hole_id: c for c in parse_result.collars}
    transect_points = [
        (collar_lookup["BH-01"].easting, collar_lookup["BH-01"].northing),
        (collar_lookup["BH-02"].easting, collar_lookup["BH-02"].northing),
    ]
    _, polygons, svg_bytes = run_pipeline(
        parse_result.collars,
        parse_result.lithologies,
        transect_points,
    )
    assert len(polygons) >= 1
    assert_valid_svg(svg_bytes)


@pytest.mark.skipif(not SAMPLE_WORKBOOK.exists(), reason="Run scripts/generate_sample_data.py first")
def test_e2e_native_ingest_workbook() -> None:
    parse_result, report = ingest_workbook(SAMPLE_WORKBOOK)
    assert report.profile_id == NATIVE_PROFILE_ID
    assert len(parse_result.collars) == 4
    assert report.mapping_proposal is not None


def test_e2e_export_roundtrip_workbook(tmp_path: Path) -> None:
    raw = make_field_export_bytes(
        [
            {
                "hole_id": "BH-A",
                "depth": "0.00-1.00m",
                "lithology": "clay",
                "lat": 58.57,
                "long": -119.19,
            },
            {
                "hole_id": "BH-B",
                "depth": "0.00-2.00m",
                "lithology": "silt",
                "lat": 58.571,
                "long": -119.189,
            },
        ]
    )
    output = tmp_path / "platform.xlsx"
    export_platform_workbook(BytesIO(raw), output, profile_id="field_export_v1")

    parse_result = DataParser().parse_file(output)
    assert len(parse_result.collars) == 2
    collar_lookup = {c.hole_id: c for c in parse_result.collars}
    points = [
        (collar_lookup["BH-A"].easting, collar_lookup["BH-A"].northing),
        (collar_lookup["BH-B"].easting, collar_lookup["BH-B"].northing),
    ]
    _, _, svg_bytes = run_pipeline(parse_result.collars, parse_result.lithologies, points)
    assert_valid_svg(svg_bytes)


def test_e2e_field_export_with_field_data_sheet() -> None:
    raw = make_field_export_bytes(
        [
            {
                "hole_id": "BH-01",
                "depth": "0.00-2.00m",
                "lithology": "clay",
                "lat": 58.57,
                "long": -119.19,
            },
            {
                "hole_id": "BH-02",
                "depth": "0.00-2.00m",
                "lithology": "silt",
                "lat": 58.571,
                "long": -119.189,
            },
        ],
        extra_sheets={"Field Data": pd.DataFrame([{"OVA": 1.5, "Sample": "S-01"}])},
    )
    _, report = ingest_workbook(BytesIO(raw), profile_id="field_export_v1")
    assert "Field Data" in report.optional_sheets_detected
    assert any("Field Data" in warning for warning in report.warnings)


# --- Excel ingestion edge cases ---


def test_e2e_fuzzy_headers_workbook() -> None:
    raw = make_workbook_bytes(
        collars=[
            {"BH": "BH-01", "East": 0.0, "North": 0.0, "RL": 100.0, "TD": 15.0},
            {"BH": "BH-02", "East": 50.0, "North": 0.0, "RL": 101.0, "TD": 15.0},
        ],
        lithology=[
            {"BH": "BH-01", "From": 0.0, "To": 5.0, "Lith": "snd"},
            {"BH": "BH-01", "From": 5.0, "To": 15.0, "Lith": "CLY"},
            {"BH": "BH-02", "From": 0.0, "To": 15.0, "Lith": "Silt"},
        ],
        collars_sheet="Collar",
        lithology_sheet="Lith",
    )
    proposal = analyze_workbook(BytesIO(raw))
    collars_df, lithology_df = read_mapped_sheets(BytesIO(raw), proposal)
    result = DataParser().parse_file(
        BytesIO(raw),
        collars_df=collars_df,
        lithology_df=lithology_df,
        lithology_aliases=load_lithology_aliases(),
    )
    assert len(result.collars) == 2
    codes = {lit.lithology_code for lit in result.lithologies}
    assert "Sand" in codes
    assert "Clay" in codes

    projected, polygons, svg_bytes = run_pipeline(
        result.collars,
        result.lithologies,
        [(0.0, 0.0), (50.0, 0.0)],
    )
    assert len(polygons) >= 1
    assert_valid_svg(svg_bytes)


def test_e2e_parser_skips_invalid_rows_continues() -> None:
    raw = make_workbook_bytes(
        collars=[
            {"hole_id": "BH-01", "easting": 0.0, "northing": 0.0, "elevation": 100.0, "total_depth": 10.0},
            {"hole_id": "BH-01", "easting": 1.0, "northing": 0.0, "elevation": 100.0, "total_depth": 10.0},
            {"hole_id": "BH-02", "easting": 50.0, "northing": 0.0, "elevation": 100.0, "total_depth": 10.0},
        ],
        lithology=[
            {"hole_id": "BH-01", "from_depth": 0.0, "to_depth": 5.0, "lithology_code": "Clay"},
            {"hole_id": "BH-02", "from_depth": 0.0, "to_depth": 5.0, "lithology_code": "Clay"},
        ],
    )
    result = DataParser().parse_file(BytesIO(raw))
    assert len(result.collars) == 2
    assert {collar.hole_id for collar in result.collars} == {"BH-01", "BH-02"}
    assert any("duplicate" in err.lower() for err in result.errors)


def test_e2e_missing_collars_sheet_raises() -> None:
    raw = make_workbook_bytes(
        collars=[{"hole_id": "BH-01", "easting": 0.0, "northing": 0.0, "elevation": 100.0, "total_depth": 5.0}],
        lithology=[{"hole_id": "BH-01", "from_depth": 0.0, "to_depth": 5.0, "lithology_code": "Clay"}],
        collars_sheet="OnlyLithology",
        lithology_sheet="Lithology",
    )
    with pytest.raises(ValueError, match="Could not detect"):
        FormatDetector().detect(BytesIO(raw))


def test_e2e_touching_intervals_no_gap() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Sandstone"),
        Lithology(hole_id="BH-01", from_depth=5.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=5.0, lithology_code="Sandstone"),
        Lithology(hole_id="BH-02", from_depth=5.0, to_depth=10.0, lithology_code="Clay"),
    ]
    report = analyze_parsed_data(collars, lithologies)
    assert not any(issue.code == "depth_gap" for issue in report.issues)
    _, polygons, svg_bytes = run_pipeline(collars, lithologies, [(0.0, 0.0), (50.0, 0.0)])
    assert len(polygons) == 2
    assert_valid_svg(svg_bytes)


# --- Stratigraphy / projection edge cases ---


def test_e2e_pinch_out_wedge_between_holes() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=20.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Sandstone"),
        Lithology(hole_id="BH-01", from_depth=5.0, to_depth=15.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Sandstone"),
        Lithology(hole_id="BH-02", from_depth=10.0, to_depth=20.0, lithology_code="Silt"),
    ]
    projected, polygons, svg_bytes = run_pipeline(collars, lithologies, [(0.0, 0.0), (50.0, 0.0)])
    clay = next(p for p in polygons if p.lithology_code == "Clay")
    assert clay.polygon.area > 0
    apex = Point(25.0, 90.0)
    assert clay.polygon.covers(apex)
    assert clay.polygon.area == pytest.approx(125.0)
    assert_valid_svg(svg_bytes)


def test_e2e_unit_absent_on_one_hole_pinches_out() -> None:
    """Clay only on BH-01 should form a wedge pinching toward BH-02."""
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=40.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Sandstone"),
        Lithology(hole_id="BH-01", from_depth=5.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Sandstone"),
    ]
    _, polygons, svg_bytes = run_pipeline(collars, lithologies, [(0.0, 0.0), (40.0, 0.0)])
    clay_polys = [p for p in polygons if p.lithology_code == "Clay"]
    assert len(clay_polys) == 1
    assert clay_polys[0].polygon.area > 0
    assert_valid_svg(svg_bytes)


def test_e2e_sloping_ground_surface() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=105.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=95.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id=h, from_depth=0.0, to_depth=10.0, lithology_code="Silt")
        for h in ("BH-01", "BH-02")
    ]
    projected, polygons, svg_bytes = run_pipeline(collars, lithologies, [(0.0, 0.0), (50.0, 0.0)])
    tops = projected.groupby("hole_id")["collar_elevation"].first()
    assert tops["BH-01"] == pytest.approx(105.0)
    assert tops["BH-02"] == pytest.approx(95.0)
    assert len(polygons) == 1
    assert_valid_svg(svg_bytes)


def test_e2e_l_shaped_transect_corner_projection() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=10.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=100.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-03", easting=100.0, northing=40.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id=h, from_depth=0.0, to_depth=10.0, lithology_code="Sandstone")
        for h in ("BH-01", "BH-02", "BH-03")
    ]
    projected, polygons, _ = run_pipeline(
        collars,
        lithologies,
        [(0.0, 0.0), (100.0, 0.0), (100.0, 50.0)],
    )
    x_positions = (
        projected.groupby("hole_id")["x_profile"].first().sort_values().tolist()
    )
    assert x_positions == pytest.approx([10.0, 100.0, 140.0])
    assert len(polygons) == 2


def test_e2e_single_hole_produces_no_polygons() -> None:
    collars = [Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0)]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
    ]
    projected, polygons, svg_bytes = run_pipeline(collars, lithologies, [(0.0, 0.0), (100.0, 0.0)])
    assert build_stratigraphy(projected) == []
    assert polygons == []
    assert_valid_svg(svg_bytes)


def test_e2e_borehole_far_from_transect_still_projects() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=80.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
    ]
    projected, _, _ = run_pipeline(collars, lithologies, [(0.0, 0.0), (50.0, 0.0)])
    assert projected[projected["hole_id"] == "BH-02"]["x_profile"].iloc[0] == 50.0


def test_e2e_off_transect_warnings_helper() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=100.0, elevation=100.0, total_depth=10.0),
    ]
    transect = Transect(points=[(0.0, 0.0), (50.0, 0.0)])
    warnings = off_transect_warnings(collars, transect, offset_threshold_m=50.0)
    assert len(warnings) == 1
    assert "BH-02" in warnings[0]


def test_e2e_unknown_lithology_code_renders() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="MysteryRock"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="MysteryRock"),
    ]
    _, polygons, svg_bytes = run_pipeline(collars, lithologies, [(0.0, 0.0), (50.0, 0.0)])
    assert len(polygons) == 1
    assert polygons[0].lithology_code == "MysteryRock"
    assert_valid_svg(svg_bytes)


def test_e2e_all_canonical_lithologies_render() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=float(len(USGS_LITHOLOGY_COLORS))),
        Collar(hole_id="BH-02", easting=30.0, northing=0.0, elevation=100.0, total_depth=float(len(USGS_LITHOLOGY_COLORS))),
    ]
    lithologies: list[Lithology] = []
    for index, code in enumerate(USGS_LITHOLOGY_COLORS, start=1):
        lithologies.append(
            Lithology(
                hole_id="BH-01",
                from_depth=float(index - 1),
                to_depth=float(index),
                lithology_code=code,
            )
        )
        lithologies.append(
            Lithology(
                hole_id="BH-02",
                from_depth=float(index - 1),
                to_depth=float(index),
                lithology_code=code,
            )
        )
    _, polygons, svg_bytes = run_pipeline(
        collars,
        lithologies,
        [(0.0, 0.0), (30.0, 0.0)],
        show_hatches=True,
        show_legend=True,
    )
    rendered_codes = {polygon.lithology_code for polygon in polygons}
    assert rendered_codes == set(USGS_LITHOLOGY_COLORS)
    assert_valid_svg(svg_bytes)


def test_e2e_high_vertical_exaggeration() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=110.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id=h, from_depth=0.0, to_depth=10.0, lithology_code="Sandstone")
        for h in ("BH-01", "BH-02")
    ]
    _, _, svg_bytes = run_pipeline(
        collars, lithologies, [(0.0, 0.0), (50.0, 0.0)], vertical_exaggeration=20.0
    )
    assert_valid_svg(svg_bytes)


def test_e2e_renderer_without_hatches_or_legend() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=5.0),
        Collar(hole_id="BH-02", easting=40.0, northing=0.0, elevation=100.0, total_depth=5.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Gravel"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=5.0, lithology_code="Bedrock"),
    ]
    _, _, svg_bytes = run_pipeline(
        collars,
        lithologies,
        [(0.0, 0.0), (40.0, 0.0)],
        show_hatches=False,
        show_legend=False,
    )
    assert_valid_svg(svg_bytes)


def test_e2e_three_hole_transect_many_units() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=15.0),
        Collar(hole_id="BH-02", easting=30.0, northing=0.0, elevation=100.0, total_depth=15.0),
        Collar(hole_id="BH-03", easting=60.0, northing=0.0, elevation=100.0, total_depth=15.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Sandstone"),
        Lithology(hole_id="BH-01", from_depth=5.0, to_depth=15.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=5.0, lithology_code="Silt"),
        Lithology(hole_id="BH-02", from_depth=5.0, to_depth=15.0, lithology_code="Gravel"),
        Lithology(hole_id="BH-03", from_depth=0.0, to_depth=5.0, lithology_code="Organics"),
        Lithology(hole_id="BH-03", from_depth=5.0, to_depth=15.0, lithology_code="Bedrock"),
    ]
    _, polygons, svg_bytes = run_pipeline(
        collars,
        lithologies,
        [(0.0, 0.0), (30.0, 0.0), (60.0, 0.0)],
    )
    assert len(polygons) >= 4
    assert_valid_svg(svg_bytes)


# --- Depth parser edge cases (field export path) ---


@pytest.mark.parametrize(
    "value,expected",
    [
        ("0.00-2.00m", (0.0, 2.0)),
        ("2.50-4.00m", (2.5, 4.0)),
        ("0-2", (0.0, 2.0)),
        ("0.00 - 2.00 m", (0.0, 2.0)),
    ],
)
def test_e2e_depth_interval_strings_in_pipeline(value: str, expected: tuple[float, float]) -> None:
    assert parse_depth_interval(value) == expected


# --- QA edge cases (pipeline gating logic) ---


def test_e2e_qa_blocking_overlap_prevents_clean_report() -> None:
    collars = [Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0)]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=12.0, lithology_code="Sandstone"),
        Lithology(hole_id="BH-01", from_depth=10.0, to_depth=18.0, lithology_code="Clay"),
    ]
    report = analyze_parsed_data(collars, lithologies)
    assert report.has_blocking_errors
    assert any(issue.code == "depth_overlap" for issue in report.issues)


def test_e2e_qa_depth_gap_warning_not_blocking() -> None:
    collars = [Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0)]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Sandstone"),
        Lithology(hole_id="BH-01", from_depth=8.0, to_depth=15.0, lithology_code="Clay"),
    ]
    report = analyze_parsed_data(collars, lithologies)
    assert not report.has_blocking_errors
    assert report.warning_count >= 1
    assert any(issue.code == "depth_gap" for issue in report.issues)


def test_e2e_qa_off_transect_warning() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=100.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id=h, from_depth=0.0, to_depth=10.0, lithology_code="Clay") for h in ("BH-01", "BH-02")
    ]
    transect = Transect(points=[(0.0, 0.0), (50.0, 0.0)])
    report = analyze_parsed_data(collars, lithologies, transect=transect, offset_threshold_m=50.0)
    assert any(issue.code == "off_transect" and issue.hole_id == "BH-02" for issue in report.issues)


def test_e2e_qa_flat_collar_grid_warning() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=10.0, northing=10.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=10.0, northing=10.0, elevation=101.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id=h, from_depth=0.0, to_depth=10.0, lithology_code="Clay") for h in ("BH-01", "BH-02")
    ]
    report = analyze_parsed_data(collars, lithologies)
    assert any(issue.code == "flat_collar_grid" for issue in report.issues)


def test_e2e_qa_orphan_lithology_error() -> None:
    collars = [Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0)]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Clay"),
        Lithology(hole_id="BH-99", from_depth=0.0, to_depth=5.0, lithology_code="Clay"),
    ]
    report = analyze_parsed_data(collars, lithologies)
    assert report.has_blocking_errors
    assert any(issue.code == "orphan_lithology" for issue in report.issues)


def test_e2e_qa_below_total_depth_error() -> None:
    collars = [Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0)]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=12.0, lithology_code="Clay"),
    ]
    report = analyze_parsed_data(collars, lithologies)
    assert any(issue.code == "below_td" for issue in report.issues)


# --- Transect planner edge cases ---


def test_e2e_transect_recommender_minimum_two_holes() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id=h, from_depth=0.0, to_depth=10.0, lithology_code="Sandstone")
        for h in ("BH-01", "BH-02")
    ]
    candidates = recommend_transects(collars, lithologies, top_n=3)
    assert len(candidates) == 1
    assert candidates[0].hole_ids == ("BH-01", "BH-02")


def test_e2e_transect_recommender_empty_with_one_hole() -> None:
    collars = [Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0)]
    lithologies = [Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay")]
    assert recommend_transects(collars, lithologies) == []


def test_e2e_transect_recommender_large_project_sliding_window() -> None:
    """More than 9 holes triggers sliding-window path instead of full combinations."""
    collars = [
        Collar(
            hole_id=f"BH-{index:02d}",
            easting=float(index * 25),
            northing=0.0,
            elevation=100.0,
            total_depth=10.0,
        )
        for index in range(15)
    ]
    lithologies = [
        Lithology(hole_id=collar.hole_id, from_depth=0.0, to_depth=10.0, lithology_code="Silt")
        for collar in collars
    ]
    candidates = recommend_transects(collars, lithologies, top_n=3, max_holes=4)
    assert len(candidates) == 3
    for candidate in candidates:
        assert 2 <= len(candidate.hole_ids) <= 4


# --- Degenerate / empty inputs ---


def test_e2e_empty_lithology_dataframe() -> None:
    assert build_stratigraphy(pd.DataFrame()) == []


def test_e2e_lithology_for_unknown_hole_skipped_in_projection() -> None:
    collars = [Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0)]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-ghost", from_depth=0.0, to_depth=5.0, lithology_code="Clay"),
    ]
    projected, _, _ = run_pipeline(
        collars,
        lithologies,
        [(0.0, 0.0), (50.0, 0.0)],
    )
    assert set(projected["hole_id"]) == {"BH-01"}


def test_e2e_empty_projection_returns_no_polygons() -> None:
    collars = [Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0)]
    transect = Transect(points=[(0.0, 0.0), (50.0, 0.0)])
    projected = project_boreholes(collars, [], transect)
    assert projected.empty
    assert build_stratigraphy(projected) == []


def test_e2e_transect_subset_excludes_off_line_holes() -> None:
    """Only selected transect holes are projected; others must not trigger offset warnings."""
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-03", easting=25.0, northing=80.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id=h, from_depth=0.0, to_depth=10.0, lithology_code="Clay")
        for h in ("BH-01", "BH-02", "BH-03")
    ]
    parse_result = ParseResult(collars=tuple(collars), lithologies=tuple(lithologies), errors=())
    subset = subset_parse_result(parse_result, ("BH-01", "BH-02"))
    transect = Transect(points=[(0.0, 0.0), (50.0, 0.0)])
    warnings = off_transect_warnings(subset.collars, transect, offset_threshold_m=50.0)
    assert warnings == []

    projected, polygons, svg_bytes = run_pipeline(
        subset.collars,
        subset.lithologies,
        [(0.0, 0.0), (50.0, 0.0)],
    )
    assert set(projected["hole_id"]) == {"BH-01", "BH-02"}
    assert len(polygons) == 1
    assert_valid_svg(svg_bytes)


def test_suggest_offset_threshold_scales_with_spread() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=5.0),
        Collar(hole_id="BH-02", easting=400.0, northing=0.0, elevation=100.0, total_depth=5.0),
    ]
    assert suggest_offset_threshold_m(collars) >= 50.0


def test_subset_parse_result_preserves_only_requested_holes() -> None:
    parse_result = ParseResult(
        collars=(
            Collar(hole_id="A", easting=0.0, northing=0.0, elevation=1.0, total_depth=5.0),
            Collar(hole_id="B", easting=1.0, northing=0.0, elevation=1.0, total_depth=5.0),
        ),
        lithologies=(
            Lithology(hole_id="A", from_depth=0.0, to_depth=5.0, lithology_code="Clay"),
            Lithology(hole_id="B", from_depth=0.0, to_depth=5.0, lithology_code="Silt"),
        ),
        errors=(),
    )
    subset = subset_parse_result(parse_result, ("A",))
    assert len(subset.collars) == 1
    assert subset.collars[0].hole_id == "A"
    assert len(subset.lithologies) == 1


def test_lithology_unit_order_accepts_excel_floats() -> None:
    lit = Lithology(
        hole_id="A",
        from_depth=0.0,
        to_depth=5.0,
        lithology_code="Clay",
        unit_order=1.0,
    )
    assert lit.unit_order == 1
    assert (
        Lithology(
            hole_id="A",
            from_depth=0.0,
            to_depth=5.0,
            lithology_code="Clay",
            unit_order="2.0",
        ).unit_order
        == 2
    )


def test_subset_parse_result_accepts_dict_lithology_index() -> None:
    parse_result = ParseResult(
        collars=(
            Collar(hole_id="A", easting=0.0, northing=0.0, elevation=1.0, total_depth=5.0),
            Collar(hole_id="B", easting=1.0, northing=0.0, elevation=1.0, total_depth=5.0),
        ),
        lithologies=(
            Lithology(hole_id="A", from_depth=0.0, to_depth=5.0, lithology_code="Clay", unit_order=1.0),
            Lithology(hole_id="B", from_depth=0.0, to_depth=5.0, lithology_code="Silt", unit_order=1.0),
        ),
        errors=(),
    )
    index = {
        "A": (parse_result.lithologies[0].model_dump(),),
        "B": (parse_result.lithologies[1].model_dump(),),
    }
    subset = subset_parse_result(parse_result, ("A", "B"), lithology_index=index)
    assert len(subset.lithologies) == 2
    assert subset.lithologies[0].unit_order == 1


def test_subset_parse_result_preserves_hole_order() -> None:
    parse_result = ParseResult(
        collars=(
            Collar(hole_id="A", easting=0.0, northing=0.0, elevation=1.0, total_depth=5.0),
            Collar(hole_id="B", easting=1.0, northing=0.0, elevation=1.0, total_depth=5.0),
            Collar(hole_id="C", easting=2.0, northing=0.0, elevation=1.0, total_depth=5.0),
        ),
        lithologies=(),
        errors=(),
    )
    subset = subset_parse_result(parse_result, ("C", "A"))
    assert [collar.hole_id for collar in subset.collars] == ["C", "A"]


def test_subset_json_bundle_matches_parse_result_subset() -> None:
    parse_result = ParseResult(
        collars=(
            Collar(hole_id="A", easting=0.0, northing=0.0, elevation=1.0, total_depth=5.0),
            Collar(hole_id="B", easting=1.0, northing=0.0, elevation=1.0, total_depth=5.0),
        ),
        lithologies=(
            Lithology(hole_id="A", from_depth=0.0, to_depth=5.0, lithology_code="Clay"),
            Lithology(hole_id="B", from_depth=0.0, to_depth=5.0, lithology_code="Silt"),
        ),
        errors=(),
        water_levels=(WaterLevel(hole_id="A", depth=1.0),),
    )
    bundle = parse_result_to_json_bundle(parse_result)
    filtered = subset_json_bundle(*bundle, ("B", "A"))
    pydantic_subset = parse_result_to_json_bundle(subset_parse_result(parse_result, ("B", "A")))
    assert filtered == pydantic_subset


def test_subset_parse_result_filters_water_levels() -> None:
    parse_result = ParseResult(
        collars=(
            Collar(hole_id="A", easting=0.0, northing=0.0, elevation=1.0, total_depth=5.0),
            Collar(hole_id="B", easting=1.0, northing=0.0, elevation=1.0, total_depth=5.0),
        ),
        lithologies=(),
        errors=(),
        water_levels=(
            WaterLevel(hole_id="A", depth=1.0),
            WaterLevel(hole_id="B", depth=2.0),
        ),
    )
    subset = subset_parse_result(parse_result, ("A",))
    assert len(subset.water_levels) == 1
    assert subset.water_levels[0].hole_id == "A"


def test_e2e_generate_requires_lithology_on_all_selected_holes() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
    ]
    subset = subset_parse_result(
        ParseResult(collars=tuple(collars), lithologies=tuple(lithologies), errors=()),
        ("BH-01", "BH-02"),
    )
    assert len(subset.collars) == 2
    assert holes_missing_lithology(subset.lithologies, ("BH-01", "BH-02")) == ("BH-02",)


def test_e2e_transect_requires_two_points() -> None:
    with pytest.raises(ValueError):
        Transect(points=[(0.0, 0.0)])


def test_e2e_borehole_only_vs_interpolated(axis_collars, axis_lithologies) -> None:
    _, interp_polygons, interp_svg = run_pipeline(
        axis_collars,
        axis_lithologies,
        [(0.0, 0.0), (100.0, 0.0)],
        interpretation_mode="interpolated",
    )
    _, observed_polygons, observed_svg = run_pipeline(
        axis_collars,
        axis_lithologies,
        [(0.0, 0.0), (100.0, 0.0)],
        interpretation_mode="borehole_only",
    )
    assert len(interp_polygons) >= 1
    assert observed_polygons == []
    assert_valid_svg(interp_svg)
    assert_valid_svg(observed_svg)


def test_e2e_native_workbook_unit_order_and_water_sheet(tmp_path: Path) -> None:
    workbook = tmp_path / "extended.xlsx"
    collars = [
        {
            "hole_id": "BH-01",
            "easting": 0.0,
            "northing": 0.0,
            "elevation": 100.0,
            "total_depth": 10.0,
        },
        {
            "hole_id": "BH-02",
            "easting": 50.0,
            "northing": 0.0,
            "elevation": 100.0,
            "total_depth": 10.0,
        },
    ]
    lithology = [
        {
            "hole_id": "BH-01",
            "from_depth": 0.0,
            "to_depth": 10.0,
            "lithology_code": "Clay",
            "unit_order": 1,
        },
        {
            "hole_id": "BH-02",
            "from_depth": 0.0,
            "to_depth": 10.0,
            "lithology_code": "Clay",
            "unit_order": 1,
        },
    ]
    water = [{"hole_id": "BH-01", "depth": 3.0}, {"hole_id": "BH-02", "depth": 4.0}]
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame(collars).to_excel(writer, sheet_name="Collars", index=False)
        pd.DataFrame(lithology).to_excel(writer, sheet_name="Lithology", index=False)
        pd.DataFrame(water).to_excel(writer, sheet_name="Water", index=False)

    parse_result = DataParser().parse_file(workbook)
    assert parse_result.lithologies[0].unit_order == 1
    assert len(parse_result.water_levels) == 2
    _, polygons, svg_bytes, _, _, _, _ = build_cross_section(
        parse_result.collars,
        parse_result.lithologies,
        [(0.0, 0.0), (50.0, 0.0)],
        water_levels=parse_result.water_levels,
    )
    assert len(polygons) == 1
    assert_valid_svg(svg_bytes)
