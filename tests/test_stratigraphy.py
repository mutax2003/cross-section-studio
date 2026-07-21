"""Tests for stratigraphy.py polygon construction."""

from __future__ import annotations

import pandas as pd
import pytest
from shapely.geometry import Point, Polygon as ShapelyPolygon

from models import CorrelationOverride
from stratigraphy import (
    GeologicalPolygon,
    _resolve_overlaps_in_pair,
    build_stratigraphy,
    detect_polygon_overlaps,
    preview_correlation_health,
)


def _projected_pair_continuous() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hole_id": "BH-01",
                "x_profile": 0.0,
                "collar_elevation": 100.0,
                "top_elevation": 95.0,
                "bottom_elevation": 85.0,
                "lithology_code": "Clay",
            },
            {
                "hole_id": "BH-02",
                "x_profile": 50.0,
                "collar_elevation": 100.0,
                "top_elevation": 94.0,
                "bottom_elevation": 80.0,
                "lithology_code": "Clay",
            },
        ]
    )


def _projected_pair_pinch_out() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hole_id": "BH-01",
                "x_profile": 0.0,
                "collar_elevation": 100.0,
                "top_elevation": 100.0,
                "bottom_elevation": 95.0,
                "lithology_code": "Sandstone",
            },
            {
                "hole_id": "BH-01",
                "x_profile": 0.0,
                "collar_elevation": 100.0,
                "top_elevation": 95.0,
                "bottom_elevation": 85.0,
                "lithology_code": "Clay",
            },
            {
                "hole_id": "BH-02",
                "x_profile": 50.0,
                "collar_elevation": 100.0,
                "top_elevation": 100.0,
                "bottom_elevation": 90.0,
                "lithology_code": "Sandstone",
            },
            {
                "hole_id": "BH-02",
                "x_profile": 50.0,
                "collar_elevation": 100.0,
                "top_elevation": 90.0,
                "bottom_elevation": 80.0,
                "lithology_code": "Silt",
            },
        ]
    )


def test_continuous_layer_quadrilateral_area() -> None:
    polygons = build_stratigraphy(_projected_pair_continuous())
    clay = next(item for item in polygons if item.lithology_code == "Clay")
    expected = 0.5 * (10.0 + 14.0) * 50.0
    assert clay.polygon.area == pytest.approx(expected)


def test_pinch_out_triangle_apex() -> None:
    polygons = build_stratigraphy(_projected_pair_pinch_out())
    clay = next(item for item in polygons if item.lithology_code == "Clay")
    assert clay.is_pinch_out
    assert clay.polygon.geom_type in {"Polygon", "MultiPolygon"}
    apex = Point(25.0, 90.0)
    assert clay.polygon.buffer(0.01).contains(apex)

    expected_area = 0.5 * (95.0 - 85.0) * 25.0
    assert clay.polygon.area == pytest.approx(expected_area)


def test_pinch_out_uses_elevation_neighbors_when_collars_differ() -> None:
    """Pinch apex must use elevation contacts, not hole-local depths across unequal RLs."""
    projected = pd.DataFrame(
        [
            {
                "hole_id": "BH-01",
                "x_profile": 0.0,
                "collar_elevation": 100.0,
                "top_elevation": 100.0,
                "bottom_elevation": 95.0,
                "lithology_code": "Sand",
            },
            {
                "hole_id": "BH-01",
                "x_profile": 0.0,
                "collar_elevation": 100.0,
                "top_elevation": 95.0,
                "bottom_elevation": 85.0,
                "lithology_code": "Clay",
            },
            {
                "hole_id": "BH-02",
                "x_profile": 50.0,
                "collar_elevation": 120.0,
                "top_elevation": 120.0,
                "bottom_elevation": 115.0,
                "lithology_code": "Gravel",
            },
            {
                "hole_id": "BH-02",
                "x_profile": 50.0,
                "collar_elevation": 120.0,
                "top_elevation": 100.0,
                "bottom_elevation": 90.0,
                "lithology_code": "Silt",
            },
        ]
    )
    polygons = build_stratigraphy(projected, allow_pinch_outs=True)
    clay = next(item for item in polygons if item.lithology_code == "Clay" and item.is_pinch_out)
    # Elevation neighbor above = Gravel bottom 115 → apex at mid-x, z=115
    apex = Point(25.0, 115.0)
    assert clay.polygon.buffer(0.05).contains(apex)
    # Depth-based neighbors would average 115 and 100 → 107.5 (must not be used)
    wrong_apex = Point(25.0, 107.5)
    assert not clay.polygon.buffer(0.05).contains(wrong_apex)


def test_unit_order_correlates_duplicate_lithology_codes() -> None:
    projected = pd.DataFrame(
        [
            {
                "hole_id": "BH-01",
                "x_profile": 0.0,
                "collar_elevation": 100.0,
                "top_elevation": 100.0,
                "bottom_elevation": 90.0,
                "lithology_code": "Clay",
                "offset_distance": 0.0,
                "unit_order": 1,
            },
            {
                "hole_id": "BH-01",
                "x_profile": 0.0,
                "collar_elevation": 100.0,
                "top_elevation": 90.0,
                "bottom_elevation": 80.0,
                "lithology_code": "Sand",
                "offset_distance": 0.0,
                "unit_order": 2,
            },
            {
                "hole_id": "BH-02",
                "x_profile": 50.0,
                "collar_elevation": 100.0,
                "top_elevation": 100.0,
                "bottom_elevation": 88.0,
                "lithology_code": "Clay",
                "offset_distance": 0.0,
                "unit_order": 1,
            },
            {
                "hole_id": "BH-02",
                "x_profile": 50.0,
                "collar_elevation": 100.0,
                "top_elevation": 88.0,
                "bottom_elevation": 78.0,
                "lithology_code": "Sand",
                "offset_distance": 0.0,
                "unit_order": 2,
            },
        ]
    )
    polygons = build_stratigraphy(projected)
    assert {polygon.lithology_code for polygon in polygons} == {"Clay", "Sand"}
    assert all(not polygon.is_pinch_out for polygon in polygons)


def test_detect_polygon_overlaps_finds_intersection_centroid() -> None:
    polygons = [
        GeologicalPolygon(
            lithology_code="Clay",
            polygon=ShapelyPolygon([(0.0, 90.0), (25.0, 90.0), (25.0, 80.0), (0.0, 80.0)]),
            hole_pair=("BH-01", "BH-02"),
        ),
        GeologicalPolygon(
            lithology_code="Silt",
            polygon=ShapelyPolygon([(10.0, 88.0), (30.0, 88.0), (30.0, 78.0), (10.0, 78.0)]),
            hole_pair=("BH-01", "BH-02"),
        ),
    ]
    overlaps = detect_polygon_overlaps(polygons)
    assert len(overlaps) == 1
    assert overlaps[0].left_lithology_code == "Clay"
    assert overlaps[0].right_lithology_code == "Silt"
    assert "Clay / Silt" in overlaps[0].message()


def test_detect_polygon_overlaps_finds_nested_containment() -> None:
    """Containment is not shapely 'overlaps'; intersects + area filter must catch it."""
    outer = ShapelyPolygon([(0.0, 100.0), (40.0, 100.0), (40.0, 60.0), (0.0, 60.0)])
    inner = ShapelyPolygon([(10.0, 90.0), (20.0, 90.0), (20.0, 80.0), (10.0, 80.0)])
    assert outer.contains(inner)
    assert not outer.overlaps(inner)
    polygons = [
        GeologicalPolygon(
            lithology_code="Clay",
            polygon=outer,
            hole_pair=("BH-01", "BH-02"),
        ),
        GeologicalPolygon(
            lithology_code="Sand",
            polygon=inner,
            hole_pair=("BH-01", "BH-02"),
        ),
    ]
    overlaps = detect_polygon_overlaps(polygons)
    assert len(overlaps) == 1
    assert {overlaps[0].left_lithology_code, overlaps[0].right_lithology_code} == {
        "Clay",
        "Sand",
    }


def test_correlation_override_increases_matched_units() -> None:
    projected = pd.DataFrame(
        [
            {
                "hole_id": "BH-01",
                "x_profile": 0.0,
                "collar_elevation": 100.0,
                "top_elevation": 100.0,
                "bottom_elevation": 90.0,
                "lithology_code": "Clay",
                "unit_order": 1,
            },
            {
                "hole_id": "BH-01",
                "x_profile": 0.0,
                "collar_elevation": 100.0,
                "top_elevation": 90.0,
                "bottom_elevation": 80.0,
                "lithology_code": "Sand",
                "unit_order": 2,
            },
            {
                "hole_id": "BH-02",
                "x_profile": 50.0,
                "collar_elevation": 100.0,
                "top_elevation": 100.0,
                "bottom_elevation": 88.0,
                "lithology_code": "Silt",
                "unit_order": 1,
            },
            {
                "hole_id": "BH-02",
                "x_profile": 50.0,
                "collar_elevation": 100.0,
                "top_elevation": 88.0,
                "bottom_elevation": 78.0,
                "lithology_code": "Sand",
                "unit_order": 2,
            },
        ]
    )
    without = build_stratigraphy(projected, allow_pinch_outs=True)
    with_override = build_stratigraphy(
        projected,
        allow_pinch_outs=True,
        correlation_overrides=[
            CorrelationOverride(
                left_hole_id="BH-01",
                right_hole_id="BH-02",
                left_unit_order=1,
                right_unit_order=1,
            )
        ],
    )
    assert any(
        polygon.lithology_code == "Clay" and not polygon.is_pinch_out
        for polygon in with_override
    )
    assert any(polygon.is_pinch_out for polygon in without)
    # Remapped Clay/Silt pair must not also leave an orphan pinch-out for the same intervals.
    clay_polygons = [p for p in with_override if p.lithology_code == "Clay"]
    assert len(clay_polygons) == 1
    assert not clay_polygons[0].is_pinch_out


def test_correlation_override_accepts_reversed_hole_order() -> None:
    projected = pd.DataFrame(
        [
            {
                "hole_id": "BH-01",
                "x_profile": 0.0,
                "collar_elevation": 100.0,
                "top_elevation": 100.0,
                "bottom_elevation": 90.0,
                "lithology_code": "Clay",
                "unit_order": 1,
            },
            {
                "hole_id": "BH-01",
                "x_profile": 0.0,
                "collar_elevation": 100.0,
                "top_elevation": 90.0,
                "bottom_elevation": 80.0,
                "lithology_code": "Sand",
                "unit_order": 2,
            },
            {
                "hole_id": "BH-02",
                "x_profile": 50.0,
                "collar_elevation": 100.0,
                "top_elevation": 100.0,
                "bottom_elevation": 88.0,
                "lithology_code": "Silt",
                "unit_order": 1,
            },
            {
                "hole_id": "BH-02",
                "x_profile": 50.0,
                "collar_elevation": 100.0,
                "top_elevation": 88.0,
                "bottom_elevation": 78.0,
                "lithology_code": "Sand",
                "unit_order": 2,
            },
        ]
    )
    with_override = build_stratigraphy(
        projected,
        allow_pinch_outs=True,
        correlation_overrides=[
            CorrelationOverride(
                left_hole_id="BH-02",
                right_hole_id="BH-01",
                left_unit_order=1,
                right_unit_order=1,
            )
        ],
    )
    clay_polygons = [p for p in with_override if p.lithology_code == "Clay"]
    assert len(clay_polygons) == 1
    assert not clay_polygons[0].is_pinch_out


def test_make_polygon_returns_single_polygon_geom() -> None:
    from stratigraphy import _make_polygon

    geo = _make_polygon(
        [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (0.0, 0.0)],
        "Clay",
        ("BH-01", "BH-02"),
    )
    if geo is not None:
        assert geo.polygon.geom_type == "Polygon"


def test_resolve_overlaps_in_pair_keeps_largest_fragment() -> None:
    shallow = GeologicalPolygon(
        lithology_code="Sand",
        polygon=ShapelyPolygon([(0.0, 95.0), (50.0, 94.0), (50.0, 85.0), (0.0, 90.0)]),
        hole_pair=("BH-01", "BH-02"),
    )
    deep = GeologicalPolygon(
        lithology_code="Clay",
        polygon=ShapelyPolygon([(0.0, 90.0), (50.0, 88.0), (50.0, 80.0), (0.0, 82.0)]),
        hole_pair=("BH-01", "BH-02"),
    )
    resolved = _resolve_overlaps_in_pair([shallow, deep])
    assert len(resolved) == 2
    assert all(polygon.polygon.area > 0 for polygon in resolved)


def test_resolve_overlaps_warns_when_clip_discards_significant_area(caplog) -> None:
    """Heavy overlap after a flushed batch: kept area can fall below 85% of original.

    _resolve_overlaps_in_pair logs logger.warning with lithology codes in that case;
    geometry algorithm is otherwise unchanged.
    """
    import logging

    # Four non-overlapping sands fill the unary_union batch (limit 4), then a
    # pinch-out clay (sorted later) is clipped against the occupied union.
    sands = [
        GeologicalPolygon(
            lithology_code="Sand",
            polygon=ShapelyPolygon([(0.0, top), (50.0, top), (50.0, bot), (0.0, bot)]),
            hole_pair=("BH-01", "BH-02"),
        )
        for top, bot in ((100.0, 98.0), (98.0, 96.0), (96.0, 94.0), (94.0, 92.0))
    ]
    clay = GeologicalPolygon(
        lithology_code="Clay",
        polygon=ShapelyPolygon([(0.0, 99.0), (50.0, 99.0), (50.0, 88.0), (0.0, 88.0)]),
        hole_pair=("BH-01", "BH-02"),
        is_pinch_out=True,
    )
    with caplog.at_level(logging.WARNING, logger="stratigraphy"):
        resolved = _resolve_overlaps_in_pair([*sands, clay])
    assert len(resolved) >= 1
    assert any("Overlap clip discarded fragments" in record.message for record in caplog.records)
    assert any("Clay" in record.message for record in caplog.records)


def test_preview_correlation_health_reports_unmatched_keys() -> None:
    summaries = preview_correlation_health(_projected_pair_pinch_out(), allow_pinch_outs=True)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.unmatched_keys_count >= 1
    assert summary.pinch_out_candidates >= 1


def test_pinch_out_uses_unit_order_neighbor_contacts() -> None:
    projected = pd.DataFrame(
        [
            {
                "hole_id": "BH-01",
                "x_profile": 0.0,
                "collar_elevation": 100.0,
                "top_elevation": 100.0,
                "bottom_elevation": 95.0,
                "lithology_code": "Sand",
                "unit_order": 1,
            },
            {
                "hole_id": "BH-01",
                "x_profile": 0.0,
                "collar_elevation": 100.0,
                "top_elevation": 95.0,
                "bottom_elevation": 85.0,
                "lithology_code": "Clay",
                "unit_order": 2,
            },
            {
                "hole_id": "BH-02",
                "x_profile": 50.0,
                "collar_elevation": 100.0,
                "top_elevation": 100.0,
                "bottom_elevation": 92.0,
                "lithology_code": "Sand",
                "unit_order": 1,
            },
            {
                "hole_id": "BH-02",
                "x_profile": 50.0,
                "collar_elevation": 100.0,
                "top_elevation": 92.0,
                "bottom_elevation": 82.0,
                "lithology_code": "Silt",
                "unit_order": 2,
            },
        ]
    )
    polygons = build_stratigraphy(projected, allow_pinch_outs=True)
    clay = next(item for item in polygons if item.lithology_code == "Clay")
    assert clay.is_pinch_out
    apex = Point(25.0, 92.0)
    assert clay.polygon.buffer(0.01).contains(apex)
