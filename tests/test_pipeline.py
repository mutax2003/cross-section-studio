"""Tests for unified cross-section pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from constants import BOREHOLE_ONLY_DISCLAIMER, INTERPOLATED_DISCLAIMER  # noqa: E402
from models import Collar, EnvironmentalReading, Lithology, WaterLevel  # noqa: E402
from pipeline import (  # noqa: E402
    auto_scale_bar_m,
    build_cross_section,
    compute_section_geometry,
    render_cross_section_from_geometry,
    validate_interpretation_mode,
)
from tests.conftest import assert_valid_svg  # noqa: E402


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
    assert len(geometry.correlation_summaries) == 1
    assert geometry.correlation_summaries[0].left_hole_id == "BH-01"
    assert geometry.correlation_summaries[0].right_hole_id == "BH-02"

    result = build_cross_section(collars, lithologies, [(0.0, 0.0), (50.0, 0.0)])
    assert len(result.projected) == len(geometry.projected)
    assert len(result.polygons) == len(geometry.polygons)
    assert_valid_svg(result.svg_bytes)
    projected, polygons, svg_bytes, _, _, codes, _ = result
    assert len(projected) == 2
    assert len(polygons) == 1
    assert codes == ["Clay"]
    assert_valid_svg(svg_bytes)


def test_render_cross_section_from_geometry_reuses_polygons() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
    ]
    transect = [(0.0, 0.0), (50.0, 0.0)]
    geometry = compute_section_geometry(collars, lithologies, transect)
    svg_result = render_cross_section_from_geometry(
        geometry, transect, export_formats=frozenset({"svg"})
    )
    png_result = render_cross_section_from_geometry(
        geometry, transect, export_formats=frozenset({"png"})
    )
    assert_valid_svg(svg_result.svg_bytes)
    assert png_result.png_bytes
    assert len(png_result.polygons) == len(geometry.polygons)
    assert png_result.lithology_codes == geometry.lithology_codes


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


def test_water_overlay_disclaimer_appended() -> None:
    from constants import WATER_TABLE_OVERLAY_DISCLAIMER
    from models import WaterLevel

    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
    ]
    water = (
        WaterLevel(hole_id="BH-01", depth=2.0),
        WaterLevel(hole_id="BH-02", depth=3.0),
    )
    result = build_cross_section(
        collars,
        lithologies,
        [(0.0, 0.0), (50.0, 0.0)],
        water_levels=water,
        interpolate_water_table=True,
    )
    assert WATER_TABLE_OVERLAY_DISCLAIMER.encode() in result.svg_bytes


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


def test_export_framing_can_suppress_water_table() -> None:
    from export_framing import ExportFramingConfig

    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
    ]
    water = [
        WaterLevel(hole_id="BH-01", depth=3.0),
        WaterLevel(hole_id="BH-02", depth=4.0),
    ]
    _, _, with_water, _, _, _, _ = build_cross_section(
        collars,
        lithologies,
        [(0.0, 0.0), (50.0, 0.0)],
        water_levels=water,
        interpolate_water_table=True,
    )
    _, _, without_water, _, _, _, _ = build_cross_section(
        collars,
        lithologies,
        [(0.0, 0.0), (50.0, 0.0)],
        water_levels=water,
        interpolate_water_table=True,
        export_framing=ExportFramingConfig(include_water_table=False),
    )
    assert_valid_svg(with_water)
    assert_valid_svg(without_water)
    assert len(without_water) < len(with_water)


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
    assert "120" in text
    assert "85" in text
    assert "120 mg/L" not in text
    assert "85 mg/L" not in text


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



def test_borehole_only_draws_logs_beyond_the_interpolation_offset() -> None:
    """The sidebar sets max_offset_for_interpolation_m = offset_warning_m; a
    borehole-only section must not fail 'fewer than two boreholes' because
    one log is past that limit (no interpolation is requested)."""
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=80.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Silt"),
    ]
    projected, polygons, svg_bytes, _, _, _, _ = build_cross_section(
        collars,
        lithologies,
        [(0.0, 0.0), (50.0, 0.0)],
        interpretation_mode="borehole_only",
        max_offset_for_interpolation_m=50.0,
    )
    assert polygons == []
    assert set(projected["hole_id"]) == {"BH-01", "BH-02"}
    assert_valid_svg(svg_bytes)
    # A single hole is likewise fine in borehole-only mode.
    _, polygons, svg_bytes, _, _, _, _ = build_cross_section(
        collars[:1],
        lithologies[:1],
        [(0.0, 0.0), (50.0, 0.0)],
        interpretation_mode="borehole_only",
        max_offset_for_interpolation_m=50.0,
    )
    assert polygons == [] and assert_valid_svg(svg_bytes) is None


def test_unknown_render_layout_is_rejected() -> None:
    """An unknown layout used to fall back silently to the section sheet."""
    import pytest

    from models import Collar, Lithology
    from pipeline import build_cross_section

    collars = [
        Collar(hole_id=h, easting=10.0 * i, northing=0.0, elevation=100.0, total_depth=5.0)
        for i, h in enumerate(("BH-1", "BH-2"))
    ]
    liths = [Lithology(hole_id=c.hole_id, from_depth=0.0, to_depth=5.0, lithology_code="Clay") for c in collars]
    with pytest.raises(ValueError, match="render_layout"):
        build_cross_section(collars, liths, [(0.0, 0.0), (10.0, 0.0)], render_layout="consulting")


def test_layers_logged_in_opposite_order_are_reported() -> None:
    from models import Collar, Lithology
    from pipeline import compute_section_geometry

    collars = [
        Collar(hole_id="A", easting=0, northing=0, elevation=100, total_depth=10),
        Collar(hole_id="B", easting=50, northing=0, elevation=100, total_depth=10),
    ]
    rows = [("A", 0, 5, "Sand"), ("A", 5, 10, "Clay"), ("B", 0, 5, "Clay"), ("B", 5, 10, "Sand")]
    lithology = [Lithology(hole_id=h, from_depth=a, to_depth=b, lithology_code=c) for h, a, b, c in rows]
    geometry = compute_section_geometry(collars, lithology, [(0, 0), (50, 0)])
    assert any(message.startswith("Crossing correlation A–B") for message in geometry.overlap_warnings)


def test_crossing_note_wording_follows_pinch_out_setting() -> None:
    from models import Collar, Lithology
    from pipeline import compute_section_geometry

    collars = [
        Collar(hole_id="A", easting=0, northing=0, elevation=100, total_depth=10),
        Collar(hole_id="B", easting=50, northing=0, elevation=100, total_depth=10),
    ]
    rows = [("A", 0, 5, "Sand"), ("A", 5, 10, "Clay"), ("B", 0, 5, "Clay"), ("B", 5, 10, "Sand")]
    lithology = [Lithology(hole_id=h, from_depth=a, to_depth=b, lithology_code=c) for h, a, b, c in rows]
    off = compute_section_geometry(collars, lithology, [(0, 0), (50, 0)], allow_pinch_outs=False)
    notes = [m for m in off.overlap_warnings if m.startswith("Crossing correlation")]
    assert notes and all("not drawn between these holes" in m for m in notes)


def test_wedge_only_pairs_do_not_report_correlation_conflicts() -> None:
    from models import Collar, Lithology
    from pipeline import compute_section_geometry

    collars = [
        Collar(hole_id="A", easting=0, northing=0, elevation=100, total_depth=10),
        Collar(hole_id="B", easting=30, northing=0, elevation=96, total_depth=12),
    ]
    rows = [
        ("A", 0, 4, "Sand"), ("A", 4, 10, "Clay"),
        ("B", 0, 3, "Silt"), ("B", 3, 7, "Gravel"), ("B", 7, 12, "Till"),
    ]
    lithology = [Lithology(hole_id=h, from_depth=a, to_depth=b, lithology_code=c) for h, a, b, c in rows]
    geometry = compute_section_geometry(collars, lithology, [(0, 0), (30, 0)])
    assert not [m for m in geometry.overlap_warnings if m.startswith("Fence clipped")]
