"""Tests for unified cross-section pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from constants import BOREHOLE_ONLY_DISCLAIMER, INTERPOLATED_DISCLAIMER
from models import Collar, Lithology, WaterLevel, EnvironmentalReading
from pipeline import (
    auto_scale_bar_m,
    build_cross_section,
    compute_section_geometry,
    validate_interpretation_mode,
)
from tests.conftest import assert_valid_svg


def test_auto_scale_bar_picks_nearest_candidate() -> None:
    assert auto_scale_bar_m(100.0) == 20.0
    assert auto_scale_bar_m(5.0) == 1.0


def test_compute_section_geometry_returns_projected_and_polygons() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
    ]
    geometry = compute_section_geometry(collars, lithologies, [(0.0, 0.0), (50.0, 0.0)])
    assert not geometry.projected.empty
    assert len(geometry.polygons) > 0
    assert geometry.lithology_codes == ["Clay"]
    assert geometry.projected_hole_ids == frozenset({"BH-01", "BH-02"})
    assert geometry.collar_depths == {"BH-01": 10.0, "BH-02": 10.0}
    assert geometry.x_span > 0.0

    result = build_cross_section(collars, lithologies, [(0.0, 0.0), (50.0, 0.0)])
    assert len(result.projected) == len(geometry.projected)
    assert len(result.polygons) == len(geometry.polygons)
    assert_valid_svg(result.svg_bytes)
    projected, polygons, svg_bytes, _, _, codes, _ = result
    assert len(projected) == 2
    assert len(polygons) == 1
    assert codes == ["Clay"]
    assert_valid_svg(svg_bytes)


def test_build_cross_section_returns_svg() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
    ]
    result = build_cross_section(collars, lithologies, [(0.0, 0.0), (50.0, 0.0)])
    assert len(result.projected) == 2
    assert len(result.polygons) == 1
    assert result.lithology_codes == ["Clay"]
    assert_valid_svg(result.svg_bytes)
    assert INTERPOLATED_DISCLAIMER.encode() in result.svg_bytes

    projected, polygons, svg_bytes, _, _, codes, _ = result
    assert len(projected) == 2
    assert len(polygons) == 1
    assert codes == ["Clay"]
    assert_valid_svg(svg_bytes)


def test_borehole_only_mode_skips_polygons() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Silt"),
    ]
    _, polygons, svg_bytes, _, _, codes, _ = build_cross_section(
        collars,
        lithologies,
        [(0.0, 0.0), (50.0, 0.0)],
        interpretation_mode="borehole_only",
    )
    assert polygons == []
    assert set(codes) == {"Clay", "Silt"}
    assert_valid_svg(svg_bytes)
    assert BOREHOLE_ONLY_DISCLAIMER.encode() in svg_bytes


def test_allow_pinch_outs_false_reduces_polygons() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Sandstone"),
        Lithology(hole_id="BH-01", from_depth=5.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Sandstone"),
    ]
    _, with_pinch, _, _, _, _, _ = build_cross_section(
        collars, lithologies, [(0.0, 0.0), (50.0, 0.0)], allow_pinch_outs=True
    )
    _, without_pinch, _, _, _, _, _ = build_cross_section(
        collars, lithologies, [(0.0, 0.0), (50.0, 0.0)], allow_pinch_outs=False
    )
    assert len(with_pinch) > len(without_pinch)
    assert any(polygon.is_pinch_out for polygon in with_pinch)


def test_overlap_warnings_returned_from_pipeline() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Sandstone"),
        Lithology(hole_id="BH-01", from_depth=5.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Sandstone"),
    ]
    _, _, svg_bytes, _, _, _, overlap_warnings = build_cross_section(
        collars, lithologies, [(0.0, 0.0), (50.0, 0.0)]
    )
    assert_valid_svg(svg_bytes)
    assert isinstance(overlap_warnings, tuple)


def test_water_levels_render_in_svg() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
    ]
    _, _, svg_bytes, _, _, _, _ = build_cross_section(
        collars,
        lithologies,
        [(0.0, 0.0), (50.0, 0.0)],
        water_levels=[
            WaterLevel(hole_id="BH-01", depth=3.0),
            WaterLevel(hole_id="BH-02", depth=4.0),
        ],
    )
    assert_valid_svg(svg_bytes)


def test_environmental_readings_render_labels_in_svg() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
    ]
    readings = [
        EnvironmentalReading(hole_id="BH-01", parameter="Chloride", value=120.0, depth=3.5, unit="mg/L"),
        EnvironmentalReading(hole_id="BH-02", parameter="Chloride", value=85.0, depth=3.5, unit="mg/L"),
    ]
    _, _, svg_bytes, _, _, _, _ = build_cross_section(
        collars,
        lithologies,
        [(0.0, 0.0), (50.0, 0.0)],
        environmental_readings=readings,
        environmental_parameters=("Chloride",),
        show_parameter_labels=True,
    )
    assert_valid_svg(svg_bytes)
    text = svg_bytes.decode("utf-8", errors="ignore")
    assert "120 mg/L" in text
    assert "85 mg/L" in text


def test_validate_interpretation_mode_accepts_correlation_lines() -> None:
    assert validate_interpretation_mode("correlation_lines") == "correlation_lines"


def test_validate_interpretation_mode_rejects_unknown() -> None:
    import pytest

    with pytest.raises(ValueError, match="interpretation_mode"):
        validate_interpretation_mode("fence_diagram")


def test_build_cross_section_rejects_non_positive_vertical_exaggeration() -> None:
    import pytest

    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=5.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=5.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=5.0, lithology_code="Clay"),
    ]
    with pytest.raises(ValueError, match="vertical_exaggeration"):
        build_cross_section(
            collars,
            lithologies,
            [(0.0, 0.0), (50.0, 0.0)],
            vertical_exaggeration=0.0,
        )


def test_normalize_export_formats_drops_unknown() -> None:
    from pipeline import _normalize_export_formats

    assert _normalize_export_formats(None) == frozenset({"svg"})
    assert _normalize_export_formats(frozenset({"SVG", "jpeg", "png"})) == frozenset({"svg", "png"})
    assert _normalize_export_formats(frozenset({"jpeg"})) == frozenset({"svg"})


def test_build_cross_section_requires_two_transect_points() -> None:
    import pytest

    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=5.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Clay"),
    ]
    with pytest.raises(ValueError, match="At least two transect points"):
        build_cross_section(collars, lithologies, [(0.0, 0.0)])


def test_single_water_level_renders_marker() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
    ]
    _, _, svg_bytes, _, _, _, _ = build_cross_section(
        collars,
        lithologies,
        [(0.0, 0.0), (50.0, 0.0)],
        water_levels=[WaterLevel(hole_id="BH-01", depth=3.0)],
    )
    assert_valid_svg(svg_bytes)
    assert b"path" in svg_bytes.lower()


def test_max_offset_for_interpolation_excludes_far_holes() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-03", easting=25.0, northing=80.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id=hole, from_depth=0.0, to_depth=10.0, lithology_code="Clay")
        for hole in ("BH-01", "BH-02", "BH-03")
    ]
    result = build_cross_section(
        collars,
        lithologies,
        [(0.0, 0.0), (50.0, 0.0)],
        max_offset_for_interpolation_m=10.0,
    )
    assert len(result.polygons) > 0
    assert "BH-03" not in {polygon.hole_pair[0] for polygon in result.polygons}


def test_max_offset_counts_unique_holes_not_intervals() -> None:
    """One hole with many near-transect intervals must not satisfy the two-hole gate."""
    import pytest

    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0),
        Collar(hole_id="BH-FAR", easting=0.0, northing=100.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Sand"),
        Lithology(hole_id="BH-01", from_depth=5.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-01", from_depth=10.0, to_depth=15.0, lithology_code="Silt"),
        Lithology(hole_id="BH-FAR", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
    ]
    with pytest.raises(ValueError, match="Fewer than two boreholes"):
        build_cross_section(
            collars,
            lithologies,
            [(0.0, 0.0), (50.0, 0.0)],
            max_offset_for_interpolation_m=10.0,
        )


def test_warn_on_correlation_gaps_adds_warning_text() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Sandstone"),
        Lithology(hole_id="BH-01", from_depth=5.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Sandstone"),
    ]
    _, _, _, _, _, _, warnings = build_cross_section(
        collars,
        lithologies,
        [(0.0, 0.0), (50.0, 0.0)],
        allow_pinch_outs=True,
        warn_on_correlation_gaps=True,
    )
    assert any("Correlation gap" in warning for warning in warnings)

