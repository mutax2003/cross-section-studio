"""Tests for unified cross-section pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from constants import BOREHOLE_ONLY_DISCLAIMER, INTERPOLATED_DISCLAIMER
from models import Collar, Lithology, WaterLevel
from pipeline import (
    auto_scale_bar_m,
    build_cross_section,
    validate_interpretation_mode,
)
from tests.conftest import assert_valid_svg


def test_auto_scale_bar_picks_nearest_candidate() -> None:
    assert auto_scale_bar_m(100.0) == 20.0
    assert auto_scale_bar_m(5.0) == 1.0


def test_build_cross_section_returns_svg() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
    ]
    projected, polygons, svg_bytes, codes, _ = build_cross_section(
        collars, lithologies, [(0.0, 0.0), (50.0, 0.0)]
    )
    assert len(projected) == 2
    assert len(polygons) == 1
    assert codes == ["Clay"]
    assert_valid_svg(svg_bytes)
    assert INTERPOLATED_DISCLAIMER.encode() in svg_bytes


def test_borehole_only_mode_skips_polygons() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Silt"),
    ]
    _, polygons, svg_bytes, codes, _ = build_cross_section(
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
    _, with_pinch, _, _, _ = build_cross_section(
        collars, lithologies, [(0.0, 0.0), (50.0, 0.0)], allow_pinch_outs=True
    )
    _, without_pinch, _, _, _ = build_cross_section(
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
    _, _, svg_bytes, _, overlap_warnings = build_cross_section(
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
    _, _, svg_bytes, _, _ = build_cross_section(
        collars,
        lithologies,
        [(0.0, 0.0), (50.0, 0.0)],
        water_levels=[
            WaterLevel(hole_id="BH-01", depth=3.0),
            WaterLevel(hole_id="BH-02", depth=4.0),
        ],
    )
    assert_valid_svg(svg_bytes)


def test_validate_interpretation_mode_rejects_unknown() -> None:
    import pytest

    with pytest.raises(ValueError, match="interpretation_mode"):
        validate_interpretation_mode("fence_diagram")


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
    _, _, svg_bytes, _, _ = build_cross_section(
        collars,
        lithologies,
        [(0.0, 0.0), (50.0, 0.0)],
        water_levels=[WaterLevel(hole_id="BH-01", depth=3.0)],
    )
    assert_valid_svg(svg_bytes)
    assert b"path" in svg_bytes.lower()
