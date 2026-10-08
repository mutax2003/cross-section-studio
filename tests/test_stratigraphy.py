"""Tests for stratigraphy.py polygon construction."""

from __future__ import annotations

import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon as ShapelyPolygon

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
    # Tip sits on the matched Sandstone base interpolated mid-way (95 -> 90).
    apex = Point(25.0, 92.5)
    assert clay.polygon.buffer(0.01).contains(apex)

    # Clay (left only) and Silt (right only) are both base units under the matched
    # Sandstone: they meet along a facies change at mid-span and fill down to the
    # base line between the hole bottoms (85 -> 80) instead of tapering to a tip.
    assert clay.polygon.area == pytest.approx(10.0 * 25.0)
    silt = next(item for item in polygons if item.lithology_code == "Silt")
    assert silt.is_pinch_out and silt.polygon.area == pytest.approx(10.0 * 25.0)
    assert clay.polygon.intersection(silt.polygon).length == pytest.approx(10.0)  # 92.5 -> 82.5


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
                # Logged (not a gap): an open logging gap would be closed for the fence.
                "hole_id": "BH-02",
                "x_profile": 50.0,
                "collar_elevation": 120.0,
                "top_elevation": 115.0,
                "bottom_elevation": 100.0,
                "lithology_code": "Fill",
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
    # Elevation neighbor above = Fill bottom 100 (nothing below 85) → apex at mid-x, z=100
    apex = Point(25.0, 100.0)
    assert clay.polygon.buffer(0.05).contains(apex)
    # Depth-based neighbors (Gravel base 5 m → 115, Silt top 20 m → 100) would
    # average to 107.5 (must not be used)
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
    # Matched Sand (unit_order 1) base interpolated mid-way between holes (95 -> 92).
    apex = Point(25.0, 93.5)
    assert clay.polygon.buffer(0.01).contains(apex)


def _shifted_unit_order_pair() -> pd.DataFrame:
    """GWM fig 3 shape: a Silt lens in BH-02 shifts the underlying Clay's unit_order."""
    rows = [
        ("BH-01", 0.0, 632.0, 629.0, "Sand", 1),
        ("BH-01", 0.0, 629.0, 623.0, "Sand and Clay", 2),
        ("BH-01", 0.0, 623.0, 602.0, "Clay", 3),
        ("BH-02", 32.0, 631.5, 629.5, "Sand", 1),
        ("BH-02", 32.0, 629.5, 624.0, "Sand and Clay", 2),
        ("BH-02", 32.0, 624.0, 619.5, "Silt", 3),
        ("BH-02", 32.0, 619.5, 603.5, "Clay", 4),
    ]
    return pd.DataFrame(
        [
            {
                "hole_id": hole_id,
                "x_profile": x,
                "collar_elevation": 632.0 if hole_id == "BH-01" else 631.5,
                "top_elevation": top,
                "bottom_elevation": bottom,
                "lithology_code": code,
                "unit_order": order,
            }
            for hole_id, x, top, bottom, code, order in rows
        ]
    )


def test_pinch_out_wedge_attaches_to_source_hole_at_logged_depths() -> None:
    """Regression: GWM fig 3 wedges floated away from the hole where the unit was logged."""
    polygons = build_stratigraphy(_shifted_unit_order_pair(), allow_pinch_outs=True)
    silt = [p for p in polygons if p.lithology_code == "Silt"]
    assert len(silt) == 1 and silt[0].is_pinch_out
    wedge = silt[0].polygon
    # Vertical edge on the source hole (x=32) spans the logged interval 624.0-619.5.
    assert wedge.bounds[2] == pytest.approx(32.0)
    for z in (624.0, 619.5, 621.75):
        assert wedge.boundary.distance(Point(32.0, z)) < 1e-9
    # Tapers toward the neighbour, reaching mid-way (x=16) and no further.
    assert wedge.bounds[0] == pytest.approx(16.0)
    # The same Clay unit correlates as one continuous fill (no crossing pinch-out wedges).
    clay = [p for p in polygons if p.lithology_code == "Clay"]
    assert len(clay) == 1 and not clay[0].is_pinch_out
    assert detect_polygon_overlaps(polygons) == []


def test_resolve_overlaps_keeps_pinch_fragment_attached_to_hole() -> None:
    """A clip that splits a wedge keeps the hole-attached part, not the largest sliver."""
    blocker = GeologicalPolygon(
        lithology_code="Clay",
        polygon=ShapelyPolygon([(0.0, 99.0), (50.0, 99.0), (50.0, 97.0), (0.0, 97.0)]),
        hole_pair=("BH-01", "BH-02"),
    )
    wedge = GeologicalPolygon(
        lithology_code="Silt",
        polygon=ShapelyPolygon([(0.0, 100.0), (0.0, 98.0), (25.0, 80.0)]),
        hole_pair=("BH-01", "BH-02"),
        is_pinch_out=True,
    )
    # Deep fills flush the first clip batch so the wedge is clipped against the blocker.
    fillers = [
        GeologicalPolygon(
            lithology_code="Gravel",
            polygon=ShapelyPolygon([(0.0, top), (50.0, top), (50.0, top - 2.0), (0.0, top - 2.0)]),
            hole_pair=("BH-01", "BH-02"),
        )
        for top in (60.0, 55.0, 50.0)
    ]
    resolved = _resolve_overlaps_in_pair([blocker, *fillers, wedge])
    silt = next(p for p in resolved if p.lithology_code == "Silt")
    assert silt.polygon.bounds[0] == pytest.approx(0.0)
    assert silt.polygon.boundary.distance(Point(0.0, 99.5)) < 1e-9


def test_resolve_overlaps_clips_within_unflushed_batch() -> None:
    """Polygons kept earlier in the same (not yet unioned) batch must also clip later ones."""
    upper = GeologicalPolygon(
        lithology_code="Sand",
        polygon=ShapelyPolygon([(0.0, 100.0), (50.0, 100.0), (50.0, 90.0), (0.0, 90.0)]),
        hole_pair=("BH-01", "BH-02"),
    )
    wedge = GeologicalPolygon(
        lithology_code="Silt",
        polygon=ShapelyPolygon([(0.0, 92.0), (0.0, 85.0), (25.0, 95.0)]),
        hole_pair=("BH-01", "BH-02"),
        is_pinch_out=True,
    )
    resolved = _resolve_overlaps_in_pair([upper, wedge])
    assert detect_polygon_overlaps(resolved) == []


def _rows_df(rows: list[tuple[str, float, float, float, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hole_id": hole_id,
                "x_profile": x,
                "collar_elevation": 100.0,
                "top_elevation": top,
                "bottom_elevation": bottom,
                "lithology_code": code,
            }
            for hole_id, x, top, bottom, code in rows
        ]
    )


def _assert_pinch_outs_attached(polygons: list[GeologicalPolygon], hole_x: dict[str, float]) -> None:
    """Every pinch-out keeps a vertical edge of non-zero length on one of its holes."""
    for polygon in polygons:
        if not polygon.is_pinch_out:
            continue
        contact = max(
            polygon.polygon.boundary.intersection(LineString([(x, -1e4), (x, 1e4)])).length
            for x in (hole_x[hole] for hole in polygon.hole_pair)
        )
        assert contact > 0.0, f"{polygon.lithology_code} wedge is detached from its hole"


def test_repeated_unit_correlates_without_sliver_pinch_outs() -> None:
    """Clay above and below a Gravel lens: the Clays correlate top-down instead of
    becoming crossing pinch-out slivers, and nothing overlaps."""
    projected = _rows_df(
        [
            ("A", 0.0, 100.0, 95.0, "Clay"),
            ("A", 0.0, 95.0, 90.0, "Gravel"),
            ("A", 0.0, 90.0, 80.0, "Clay"),
            ("B", 50.0, 100.0, 92.0, "Sand"),
            ("B", 50.0, 92.0, 88.0, "Clay"),
            ("B", 50.0, 88.0, 84.0, "Gravel"),
            ("B", 50.0, 84.0, 75.0, "Clay"),
        ]
    )
    polygons = build_stratigraphy(projected, allow_pinch_outs=True)
    clays = [p for p in polygons if p.lithology_code == "Clay"]
    assert len(clays) == 2 and not any(p.is_pinch_out for p in clays)
    assert [p.lithology_code for p in polygons if p.is_pinch_out] == ["Sand"]
    assert detect_polygon_overlaps(polygons) == []
    _assert_pinch_outs_attached(polygons, {"A": 0.0, "B": 50.0})


def test_repeated_unit_against_single_unit_keeps_top_correlation() -> None:
    projected = _rows_df(
        [
            ("A", 0.0, 100.0, 95.0, "Clay"),
            ("A", 0.0, 95.0, 90.0, "Gravel"),
            ("A", 0.0, 90.0, 80.0, "Clay"),
            ("B", 50.0, 100.0, 80.0, "Clay"),
        ]
    )
    polygons = build_stratigraphy(projected, allow_pinch_outs=True)
    continuous = [p for p in polygons if not p.is_pinch_out]
    assert [p.lithology_code for p in continuous] == ["Clay"]
    assert continuous[0].polygon.bounds[3] == pytest.approx(100.0)
    assert detect_polygon_overlaps(polygons) == []
    _assert_pinch_outs_attached(polygons, {"A": 0.0, "B": 50.0})


@pytest.mark.parametrize(("bedrock_bottom", "right_bottom"), [(85.0, 80.0), (70.0, 92.0)])
def test_unique_base_unit_pinch_out_has_no_overlap(bedrock_bottom: float, right_bottom: float) -> None:
    projected = _rows_df(
        [
            ("A", 0.0, 100.0, 90.0, "Clay"),
            ("A", 0.0, 90.0, bedrock_bottom, "Bedrock"),
            ("B", 50.0, 100.0, right_bottom, "Clay"),
        ]
    )
    polygons = build_stratigraphy(projected, allow_pinch_outs=True)
    bedrock = [p for p in polygons if p.lithology_code == "Bedrock"]
    assert len(bedrock) == 1 and bedrock[0].is_pinch_out
    assert detect_polygon_overlaps(polygons) == []
    _assert_pinch_outs_attached(polygons, {"A": 0.0, "B": 50.0})


def _fence_gap_area(polygons: list[GeologicalPolygon], x0: float, x1: float, z_top, z_bot) -> float:
    """Uncovered area of the fence region between two holes bounded by contact lines."""
    from shapely.ops import unary_union

    region = ShapelyPolygon([(x0, z_top[0]), (x1, z_top[1]), (x1, z_bot[1]), (x0, z_bot[0])])
    covered = unary_union([p.polygon for p in polygons])
    return region.difference(covered).area


def _pairwise_overlap_area(polygons: list[GeologicalPolygon]) -> float:
    total = 0.0
    for i, first in enumerate(polygons):
        for second in polygons[i + 1 :]:
            total += first.polygon.intersection(second.polygon).area
    return total


def test_pinch_out_bends_bounding_units_so_fence_has_no_gap() -> None:
    """Regression (GWM fig 3 MW18-06B/MW18-16): Silt wedge between Sand-and-Clay and
    Clay left a white triangle from its tip to the neighbour hole."""
    polygons = build_stratigraphy(_shifted_unit_order_pair(), allow_pinch_outs=True)
    # Fence between the top of Sand-and-Clay and the base of Clay at both holes.
    gap = _fence_gap_area(polygons, 0.0, 32.0, (629.0, 629.5), (602.0, 603.5))
    assert gap == pytest.approx(0.0, abs=1e-6)
    assert _pairwise_overlap_area(polygons) == pytest.approx(0.0, abs=1e-6)
    assert detect_polygon_overlaps(polygons) == []
    _assert_pinch_outs_attached(polygons, {"BH-01": 0.0, "BH-02": 32.0})
    silt = next(p for p in polygons if p.lithology_code == "Silt").polygon
    # Tip mid-way, on the mean of the bracketing contacts: ((623+623)/2 + (624+619.5)/2) / 2.
    assert silt.boundary.distance(Point(16.0, 622.375)) < 1e-9
    sand_clay = next(p for p in polygons if p.lithology_code == "Sand and Clay").polygon
    clay = next(p for p in polygons if p.lithology_code == "Clay").polygon
    assert sand_clay.boundary.distance(Point(16.0, 622.375)) < 1e-9
    assert clay.boundary.distance(Point(16.0, 622.375)) < 1e-9


@pytest.mark.parametrize("source_left", [True, False])
def test_stacked_pinch_outs_fan_to_shared_tip_without_gaps(source_left: bool) -> None:
    """Two stacked lenses in one hole between matched Sand and Clay, either direction."""
    lensed = [
        (100.0, 95.0, "Sand"),
        (95.0, 92.0, "Silt"),
        (92.0, 88.0, "Gravel"),
        (88.0, 70.0, "Clay"),
    ]
    plain = [(99.0, 90.0, "Sand"), (90.0, 72.0, "Clay")]
    left, right = (lensed, plain) if source_left else (plain, lensed)
    rows = [("A", 0.0, *row) for row in left] + [("B", 40.0, *row) for row in right]
    polygons = build_stratigraphy(_rows_df(rows), allow_pinch_outs=True)
    wedges = [p for p in polygons if p.is_pinch_out]
    assert sorted(p.lithology_code for p in wedges) == ["Gravel", "Silt"]
    z_top = (left[0][0], right[0][0])
    z_bot = (left[-1][1], right[-1][1])
    assert _fence_gap_area(polygons, 0.0, 40.0, z_top, z_bot) == pytest.approx(0.0, abs=1e-6)
    assert _pairwise_overlap_area(polygons) == pytest.approx(0.0, abs=1e-6)
    _assert_pinch_outs_attached(polygons, {"A": 0.0, "B": 40.0})
    for wedge in wedges:
        assert wedge.polygon.bounds[0 if source_left else 2] == pytest.approx(
            0.0 if source_left else 40.0
        )


def test_opposing_pinch_outs_in_one_gap_meet_at_common_tip() -> None:
    """Different lenses logged in each hole between the same matched units tile the gap."""
    rows = [
        ("A", 0.0, 100.0, 95.0, "Sand"),
        ("A", 0.0, 95.0, 90.0, "Silt"),
        ("A", 0.0, 90.0, 80.0, "Clay"),
        ("B", 50.0, 100.0, 96.0, "Sand"),
        ("B", 50.0, 96.0, 88.0, "Gravel"),
        ("B", 50.0, 88.0, 78.0, "Clay"),
    ]
    polygons = build_stratigraphy(_rows_df(rows), allow_pinch_outs=True)
    assert sorted(p.lithology_code for p in polygons if p.is_pinch_out) == ["Gravel", "Silt"]
    assert _fence_gap_area(polygons, 0.0, 50.0, (100.0, 100.0), (80.0, 78.0)) == pytest.approx(
        0.0, abs=1e-6
    )
    assert _pairwise_overlap_area(polygons) == pytest.approx(0.0, abs=1e-6)
    _assert_pinch_outs_attached(polygons, {"A": 0.0, "B": 50.0})


@pytest.mark.parametrize("at_top", [True, False])
def test_single_bounded_pinch_out_tip_on_matched_contact(at_top: bool) -> None:
    """A wedge at the top or base of a hole has one bounding unit: the tip lies on its
    straight contact, so nothing overlaps and the fill below/above is unchanged."""
    if at_top:
        rows = [
            ("A", 0.0, 100.0, 97.0, "Fill"),
            ("A", 0.0, 97.0, 85.0, "Clay"),
            ("B", 50.0, 99.0, 80.0, "Clay"),
        ]
    else:
        rows = [
            ("A", 0.0, 100.0, 90.0, "Clay"),
            ("A", 0.0, 90.0, 82.0, "Bedrock"),
            ("B", 50.0, 100.0, 86.0, "Clay"),
        ]
    polygons = build_stratigraphy(_rows_df(rows), allow_pinch_outs=True)
    clay = next(p for p in polygons if p.lithology_code == "Clay")
    assert len(clay.polygon.exterior.coords) == 5  # straight quadrilateral, no bend
    wedge = next(p for p in polygons if p.is_pinch_out)
    tip_z = (97.0 + 99.0) / 2 if at_top else (90.0 + 86.0) / 2
    assert wedge.polygon.boundary.distance(Point(25.0, tip_z)) < 1e-9
    assert _pairwise_overlap_area(polygons) == pytest.approx(0.0, abs=1e-6)
    _assert_pinch_outs_attached(polygons, {"A": 0.0, "B": 50.0})


def test_pinch_out_tiling_skipped_without_pinch_outs() -> None:
    """allow_pinch_outs=False keeps hole-to-hole quadrilaterals (no mid-span bends)."""
    polygons = build_stratigraphy(_shifted_unit_order_pair(), allow_pinch_outs=False)
    assert not any(p.is_pinch_out for p in polygons)
    for polygon in polygons:
        assert len(polygon.polygon.exterior.coords) == 5


# ---------------------------------------------------------------------------
# Logging gaps (not-logged / no-recovery intervals) must not leave white wedges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gap_on_left", [True, False])
@pytest.mark.parametrize("allow_pinch_outs", [True, False])
def test_mid_log_gap_does_not_leave_white_wedge_in_fence(
    gap_on_left: bool, allow_pinch_outs: bool
) -> None:
    """Regression (synthetic site B-B', BH-09): a 0.6 m not-logged interval at the
    Sand/Clay contact left an unfilled wedge across the whole hole pair."""
    gapped = [(100.0, 95.0, "Silt"), (95.0, 90.6, "Sand"), (90.0, 80.0, "Clay")]
    plain = [(100.0, 96.0, "Silt"), (96.0, 91.0, "Sand"), (91.0, 82.0, "Clay")]
    left, right = (gapped, plain) if gap_on_left else (plain, gapped)
    rows = [("A", 0.0, *row) for row in left] + [("B", 50.0, *row) for row in right]
    polygons = build_stratigraphy(_rows_df(rows), allow_pinch_outs=allow_pinch_outs)
    assert not any(p.is_pinch_out for p in polygons)
    z_bot = (left[-1][1], right[-1][1])
    assert _fence_gap_area(polygons, 0.0, 50.0, (100.0, 100.0), z_bot) == pytest.approx(
        0.0, abs=1e-6
    )
    assert _pairwise_overlap_area(polygons) == pytest.approx(0.0, abs=1e-6)
    # Sand base and Clay top meet at the gap midpoint on the gapped hole.
    x_gap = 0.0 if gap_on_left else 50.0
    sand = next(p for p in polygons if p.lithology_code == "Sand").polygon
    clay = next(p for p in polygons if p.lithology_code == "Clay").polygon
    assert sand.boundary.distance(Point(x_gap, 90.3)) < 1e-9
    assert clay.boundary.distance(Point(x_gap, 90.3)) < 1e-9


def test_mid_log_gap_under_pinch_out_wedge_tiles() -> None:
    """Gap under a lens that pinches out toward the neighbour (white sliver beside BH-09)."""
    rows = [
        ("A", 0.0, 100.0, 95.0, "Silt"),
        ("A", 0.0, 95.0, 92.0, "Sand"),
        ("A", 0.0, 91.4, 80.0, "Clay"),  # 0.6 m not logged at the Sand/Clay contact
        ("B", 50.0, 100.0, 93.0, "Silt"),
        ("B", 50.0, 93.0, 82.0, "Clay"),
    ]
    polygons = build_stratigraphy(_rows_df(rows), allow_pinch_outs=True)
    assert [p.lithology_code for p in polygons if p.is_pinch_out] == ["Sand"]
    gap = _fence_gap_area(polygons, 0.0, 50.0, (100.0, 100.0), (80.0, 82.0))
    assert gap == pytest.approx(0.0, abs=1e-6)
    assert _pairwise_overlap_area(polygons) == pytest.approx(0.0, abs=1e-6)
    _assert_pinch_outs_attached(polygons, {"A": 0.0, "B": 50.0})


@pytest.mark.parametrize("where", ["top", "base"])
def test_top_or_base_gap_fence_follows_logged_extent(where: str) -> None:
    """Gaps above the first / below the last logged interval are not closed: the fence
    tapers to the logged extent, still without holes or overlaps inside it."""
    if where == "top":
        left = [(98.0, 90.0, "Sand"), (90.0, 80.0, "Clay")]  # collar 100, top 2 m unlogged
    else:
        left = [(100.0, 90.0, "Sand"), (90.0, 84.0, "Clay")]  # TD below 84 unlogged
    right = [(100.0, 91.0, "Sand"), (91.0, 80.0, "Clay")]
    rows = [("A", 0.0, *row) for row in left] + [("B", 50.0, *row) for row in right]
    polygons = build_stratigraphy(_rows_df(rows), allow_pinch_outs=True)
    z_top = (left[0][0], right[0][0])
    z_bot = (left[-1][1], right[-1][1])
    assert _fence_gap_area(polygons, 0.0, 50.0, z_top, z_bot) == pytest.approx(0.0, abs=1e-6)
    assert _pairwise_overlap_area(polygons) == pytest.approx(0.0, abs=1e-6)
    sand = next(p for p in polygons if p.lithology_code == "Sand").polygon
    assert sand.bounds[3] == pytest.approx(100.0)
    assert sand.boundary.distance(Point(0.0, left[0][0])) < 1e-9


def test_logging_gap_keeps_hole_column_data_untouched() -> None:
    """Gap closing is fence-only: the projected intervals the column renders from are unchanged."""
    projected = _rows_df(
        [
            ("A", 0.0, 100.0, 95.0, "Sand"),
            ("A", 0.0, 94.0, 80.0, "Clay"),
            ("B", 50.0, 100.0, 95.0, "Sand"),
            ("B", 50.0, 95.0, 80.0, "Clay"),
        ]
    )
    before = projected.copy()
    build_stratigraphy(projected, allow_pinch_outs=True)
    pd.testing.assert_frame_equal(projected, before)


def test_nested_interval_is_not_mistaken_for_a_logging_gap() -> None:
    """An interval logged inside a longer one (overlapping rows) does not open a 'gap'."""
    from stratigraphy import _close_logging_gaps, _intervals_for_hole

    hole = _rows_df(
        [
            ("A", 0.0, 100.0, 90.0, "Clay"),
            ("A", 0.0, 98.0, 97.0, "Sand"),
            ("A", 0.0, 90.0, 80.0, "Gravel"),
        ]
    )
    intervals = _intervals_for_hole(hole)
    assert _close_logging_gaps(intervals) is intervals


def _crossing_sand_clay(reverse: bool) -> pd.DataFrame:
    left = [("Clay", 100.0, 99.0, 1), ("Sand", 99.0, 90.0, 2)]
    right = [("Sand", 100.0, 91.0, 1), ("Clay", 91.0, 90.0, 2)]
    holes = [("L", 0.0, left), ("R", 50.0, right)]
    if reverse:
        holes = [("L", 50.0, left), ("R", 0.0, right)]
    return pd.DataFrame(
        [
            {
                "hole_id": hole,
                "x_profile": x,
                "collar_elevation": 100.0,
                "top_elevation": top,
                "bottom_elevation": bottom,
                "lithology_code": code,
                "unit_order": order,
            }
            for hole, x, units in holes
            for code, top, bottom, order in units
        ]
    )


def _mirrored_signature(polygons: list[GeologicalPolygon], mirror: bool) -> list[tuple]:
    signature = []
    for polygon in polygons:
        coords = sorted(
            (round(50.0 - x if mirror else x, 6), round(z, 6))
            for x, z in polygon.polygon.exterior.coords[:-1]
        )
        signature.append(
            (polygon.lithology_code, polygon.is_pinch_out, round(polygon.polygon.area, 6), coords)
        )
    return sorted(signature)


def test_shifted_rematch_is_symmetric_under_transect_reversal() -> None:
    """Same-code re-matching ranks candidate pairs globally (elevation overlap first),
    so reversing the transect mirrors the section instead of swapping which unit
    is filled across and which pinches out."""
    forward = build_stratigraphy(_crossing_sand_clay(reverse=False), allow_pinch_outs=True)
    backward = build_stratigraphy(_crossing_sand_clay(reverse=True), allow_pinch_outs=True)
    assert _mirrored_signature(forward, False) == _mirrored_signature(backward, True)
    # The thick Sand (9 m in both holes) correlates; the thin Clays pinch out.
    sand = [p for p in forward if p.lithology_code == "Sand"]
    assert len(sand) == 1 and not sand[0].is_pinch_out
    assert sand[0].polygon.area == pytest.approx(450.0)
    assert all(p.is_pinch_out for p in forward if p.lithology_code == "Clay")


def test_crossing_override_does_not_bend_fill_inside_out() -> None:
    """A crossing override (top unit in H0 → deep unit in H1) must not bend that fill's
    bottom above its top, which made buffer(0) repair silently drop area."""
    from stratigraphy import _polygons_for_pair, _sorted_hole_profiles

    rows = [
        ("H0", 0.0, 97.22, 94.22, "Gravel", 1),
        ("H0", 0.0, 94.22, 92.22, "Silt", 2),
        ("H0", 0.0, 92.22, 87.22, "Gravel", 3),
        ("H0", 0.0, 87.22, 82.22, "Gravel", 4),
        ("H0", 0.0, 82.22, 80.22, "Silt", 5),
        ("H0", 0.0, 80.22, 79.22, "Clay", 6),
        ("H1", 50.0, 97.09, 95.09, "Gravel", 1),
        ("H1", 50.0, 95.09, 92.09, "Silt", 2),
        ("H1", 50.0, 91.09, 88.09, "Gravel", 3),
        ("H1", 50.0, 88.09, 86.09, "Sand", 4),
        ("H1", 50.0, 86.09, 85.59, "Gravel", 5),
        ("H1", 50.0, 85.59, 83.59, "Sand", 6),
    ]
    projected = pd.DataFrame(
        [
            {
                "hole_id": hole,
                "x_profile": x,
                "collar_elevation": 100.0,
                "top_elevation": top,
                "bottom_elevation": bottom,
                "lithology_code": code,
                "unit_order": order,
            }
            for hole, x, top, bottom, code, order in rows
        ]
    )
    override = CorrelationOverride(
        left_hole_id="H0", right_hole_id="H1", left_unit_order=1, right_unit_order=5
    )
    (x0, _l, left_intervals, _ll), (x1, _r, right_intervals, _rl) = _sorted_hole_profiles(
        projected
    )
    polygons = _polygons_for_pair(
        "H0", "H1", x0, x1, left_intervals, right_intervals, correlation_overrides=(override,)
    )
    for polygon in polygons:
        if polygon.is_pinch_out:
            continue
        # Every fill's ring is simple as built (no buffer(0) repair needed).
        assert polygon.polygon.is_valid
    overridden = next(
        p
        for p in polygons
        if not p.is_pinch_out
        and p.polygon.boundary.distance(Point(0.0, 97.22)) < 1e-9
        and p.polygon.boundary.distance(Point(50.0, 85.59)) < 1e-9
    )
    # Vertical edges keep their full logged thickness at both holes (3 m and 0.5 m).
    assert overridden.polygon.boundary.distance(Point(0.0, 94.22)) < 1e-9
    assert overridden.polygon.boundary.distance(Point(50.0, 86.09)) < 1e-9
    assert overridden.polygon.area >= 50.0 * (3.0 + 0.5) / 2.0 - 1e-6


def test_inverted_layer_order_unmatches_crossing_correlations() -> None:
    """Sand over Clay in A, Clay over Sand in B (code-only keys): the crossing fills
    used to be clipped to a 25% sliver, silently dropping B's Sand from the fence.
    Both crossing units now pinch out, anchored at their own holes, and the pair
    summary reports the conflict."""
    projected = _rows_df(
        [
            ("A", 0.0, 100.0, 95.0, "Sand"),
            ("A", 0.0, 95.0, 90.0, "Clay"),
            ("B", 50.0, 100.0, 95.0, "Clay"),
            ("B", 50.0, 95.0, 90.0, "Sand"),
        ]
    )
    summaries: list = []
    clip_warnings: list[str] = []
    polygons = build_stratigraphy(
        projected, allow_pinch_outs=True, pair_summaries=summaries, clip_warnings=clip_warnings
    )
    assert len(polygons) == 4 and all(p.is_pinch_out for p in polygons)
    _assert_pinch_outs_attached(polygons, {"A": 0.0, "B": 50.0})
    for polygon in polygons:
        assert polygon.polygon.area == pytest.approx(0.5 * 5.0 * 25.0)
    assert _pairwise_overlap_area(polygons) == pytest.approx(0.0, abs=1e-6)
    assert summaries[0].crossing_codes == ("Clay", "Sand")
    assert summaries[0].matched_count == 0
    assert clip_warnings == []
    health = preview_correlation_health(projected)
    assert health[0].crossing_codes == ("Clay", "Sand")


def test_crossing_correlation_keeps_best_ranked_match() -> None:
    """When only one of two crossing matches lines up in elevation, that one stays."""
    projected = _rows_df(
        [
            ("A", 0.0, 100.0, 99.0, "Clay"),
            ("A", 0.0, 99.0, 90.0, "Sand"),
            ("B", 50.0, 100.0, 91.0, "Sand"),
            ("B", 50.0, 91.0, 90.0, "Clay"),
        ]
    )
    summaries: list = []
    polygons = build_stratigraphy(projected, allow_pinch_outs=True, pair_summaries=summaries)
    sand = [p for p in polygons if p.lithology_code == "Sand"]
    assert len(sand) == 1 and not sand[0].is_pinch_out
    assert all(p.is_pinch_out for p in polygons if p.lithology_code == "Clay")
    assert summaries[0].crossing_codes == ("Clay",)


def test_explicit_crossing_override_is_kept() -> None:
    projected = pd.DataFrame(
        [
            {
                "hole_id": hole,
                "x_profile": x,
                "collar_elevation": 100.0,
                "top_elevation": top,
                "bottom_elevation": bottom,
                "lithology_code": code,
                "unit_order": order,
            }
            for hole, x, top, bottom, code, order in [
                ("A", 0.0, 100.0, 95.0, "Sand", 1),
                ("A", 0.0, 95.0, 90.0, "Clay", 2),
                ("B", 50.0, 100.0, 95.0, "Clay", 1),
                ("B", 50.0, 95.0, 90.0, "Sand", 2),
            ]
        ]
    )
    override = CorrelationOverride(
        left_hole_id="A", right_hole_id="B", left_unit_order=1, right_unit_order=2
    )
    polygons = build_stratigraphy(
        projected, allow_pinch_outs=True, correlation_overrides=(override,)
    )
    sand = [p for p in polygons if p.lithology_code == "Sand" and not p.is_pinch_out]
    assert len(sand) == 1


def test_clip_warnings_report_substantial_fence_loss() -> None:
    first = GeologicalPolygon(
        "Clay", ShapelyPolygon([(0, 0), (10, 0), (10, 10), (0, 10)]), ("A", "B")
    )
    second = GeologicalPolygon(
        "Sand", ShapelyPolygon([(0, -2), (10, -2), (10, 8), (0, 8)]), ("A", "B")
    )
    warnings: list[str] = []
    _resolve_overlaps_in_pair([first, second], clip_warnings=warnings)
    assert len(warnings) == 1
    assert warnings[0].startswith("Fence clipped: Sand between A–B kept 20%")


def test_no_false_overlap_from_near_coincident_tip_vertices() -> None:
    """GEOS precision: a clipped wedge with two tip vertices ~5e-15 apart reported a
    Silt/Clay overlap in one intersection direction only."""
    from models import Collar, Lithology
    from pipeline import compute_section_geometry

    transect = [
        (50.0, -19.15486194756213),
        (50.0, 19.52559208826952),
        (50.0, 0.18536507035369532),
    ]
    collars = [
        Collar(
            hole_id="H0",
            easting=50.0,
            northing=9.047282356731813,
            elevation=166.187008501202,
            total_depth=37.8,
        ),
        Collar(
            hole_id="H1",
            easting=50.0,
            northing=12.17329004604181,
            elevation=100.0,
            total_depth=400,
        ),
    ]
    lithologies = [
        Lithology(
            hole_id="H0", from_depth=1.49572, to_depth=6.34054, lithology_code="Silt", unit_order=2
        ),
        Lithology(
            hole_id="H1", from_depth=0, to_depth=244.87773, lithology_code="Clay", unit_order=1
        ),
        Lithology(
            hole_id="H1",
            from_depth=244.87773,
            to_depth=330.65682,
            lithology_code="Silt",
            unit_order=2,
        ),
    ]
    geometry = compute_section_geometry(collars, lithologies, transect)
    assert geometry.overlap_warnings == ()
    for polygon in geometry.polygons:
        coords = list(polygon.polygon.exterior.coords)[:-1]
        for (x0, z0), (x1, z1) in zip(coords, coords[1:] + coords[:1]):
            assert abs(x0 - x1) + abs(z0 - z1) > 1e-9


def test_dedupe_vertices_drops_float_noise() -> None:
    from stratigraphy import _dedupe_vertices

    assert _dedupe_vertices([(0.0, 0.0), (1.0, 1.0), (1.0, 1.0 + 5e-15), (0.0, 1.0)]) == [
        (0.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ]


# ---------------------------------------------------------------------------
# Single-bounded pinch-outs (base / top units) tile against the fence base / top
# ---------------------------------------------------------------------------


def _fence_envelope_gap_and_overlap(
    polygons: list[GeologicalPolygon], rows: list[tuple[str, float, float, float, str]]
) -> tuple[float, float]:
    """Uncovered and doubly-covered area per adjacent pair, inside the envelope bounded
    by the line joining hole tops and the base line joining hole bottoms."""
    holes: dict[str, tuple[float, float, float]] = {}
    for hole, x, top, bottom, _code in rows:
        _x, hole_top, hole_bottom = holes.get(hole, (x, top, bottom))
        holes[hole] = (x, max(hole_top, top), min(hole_bottom, bottom))
    ordered = sorted(holes.items(), key=lambda item: item[1][0])
    gap = overlap = 0.0
    for (left, (x0, top0, bot0)), (right, (x1, top1, bot1)) in zip(ordered, ordered[1:]):
        in_pair = [p for p in polygons if p.hole_pair == (left, right)]
        gap += _fence_gap_area(in_pair, x0, x1, (top0, top1), (bot0, bot1))
        overlap += _pairwise_overlap_area(in_pair)
    return gap, overlap


def _mirror_rows(
    rows: list[tuple[str, float, float, float, str]], span: float
) -> list[tuple[str, float, float, float, str]]:
    return [(hole, span - x, top, bottom, code) for hole, x, top, bottom, code in rows]


@pytest.mark.parametrize("reverse", [False, True])
def test_opposing_base_units_meet_at_facies_change_without_gap(reverse: bool) -> None:
    """Regression (client B-B' BH23-03 / 2017-BH12): the bottom unit changes lithology
    between holes under a shared contact. The two one-sided wedges used to taper to
    the midpoint and leave a white triangle down to the base line."""
    rows = [
        ("A", 0.0, 100.0, 95.0, "Clay Loam"),
        ("A", 0.0, 95.0, 85.0, "Sandy Clay Loam"),
        ("B", 50.0, 100.0, 92.0, "Clay Loam"),
        ("B", 50.0, 92.0, 88.0, "Sandy Clay"),
    ]
    if reverse:
        rows = _mirror_rows(rows, 50.0)
    polygons = build_stratigraphy(_rows_df(rows), allow_pinch_outs=True)
    gap, overlap = _fence_envelope_gap_and_overlap(polygons, rows)
    assert gap == pytest.approx(0.0, abs=1e-6)
    assert overlap == pytest.approx(0.0, abs=1e-6)
    assert detect_polygon_overlaps(polygons) == []
    _assert_pinch_outs_attached(polygons, {"A": rows[0][1], "B": rows[2][1]})
    left = next(p for p in polygons if p.lithology_code == "Sandy Clay Loam").polygon
    right = next(p for p in polygons if p.lithology_code == "Sandy Clay").polygon
    # Vertical facies boundary at mid-span from the contact (93.5) to the base (86.5).
    shared = left.intersection(right)
    assert shared.length == pytest.approx(7.0)
    assert shared.bounds[0] == pytest.approx(25.0) and shared.bounds[2] == pytest.approx(25.0)
    # The matched Clay Loam above stays a straight quadrilateral.
    clay_loam = next(p for p in polygons if p.lithology_code == "Clay Loam")
    assert not clay_loam.is_pinch_out and len(clay_loam.polygon.exterior.coords) == 5


def test_opposing_base_units_stack_proportionally() -> None:
    """Several unmatched base units per hole keep their thickness ratios across the
    region and still tile with the opposite hole's units."""
    rows = [
        ("A", 0.0, 100.0, 90.0, "Clay"),
        ("A", 0.0, 90.0, 86.0, "Silt"),
        ("A", 0.0, 86.0, 80.0, "Gravel"),
        ("B", 40.0, 100.0, 94.0, "Clay"),
        ("B", 40.0, 94.0, 92.0, "Sand"),
        ("B", 40.0, 92.0, 84.0, "Till"),
        ("B", 40.0, 84.0, 83.0, "Bedrock"),
    ]
    polygons = build_stratigraphy(_rows_df(rows), allow_pinch_outs=True)
    gap, overlap = _fence_envelope_gap_and_overlap(polygons, rows)
    assert gap == pytest.approx(0.0, abs=1e-6)
    assert overlap == pytest.approx(0.0, abs=1e-6)
    _assert_pinch_outs_attached(polygons, {"A": 0.0, "B": 40.0})
    silt = next(p for p in polygons if p.lithology_code == "Silt").polygon
    # Silt is 40% of A's 10 m base column; at mid-span the column is 92 -> 81.5 thick.
    assert silt.bounds[2] == pytest.approx(20.0)
    assert silt.intersection(LineString([(20.0, 0.0), (20.0, 200.0)])).length == pytest.approx(
        0.4 * 10.5
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_opposing_top_units_meet_at_facies_change_without_gap(reverse: bool) -> None:
    """Top units differ (Fill vs Topsoil) above a shared Clay: they tile the region
    between the ground line (hole tops) and the Clay top."""
    rows = [
        ("A", 0.0, 101.0, 98.0, "Fill"),
        ("A", 0.0, 98.0, 85.0, "Clay"),
        ("B", 30.0, 99.5, 99.0, "Topsoil"),
        ("B", 30.0, 99.0, 82.0, "Clay"),
    ]
    if reverse:
        rows = _mirror_rows(rows, 30.0)
    polygons = build_stratigraphy(_rows_df(rows), allow_pinch_outs=True)
    gap, overlap = _fence_envelope_gap_and_overlap(polygons, rows)
    assert gap == pytest.approx(0.0, abs=1e-6)
    assert overlap == pytest.approx(0.0, abs=1e-6)
    _assert_pinch_outs_attached(polygons, {"A": rows[0][1], "B": rows[2][1]})
    fill = next(p for p in polygons if p.lithology_code == "Fill").polygon
    topsoil = next(p for p in polygons if p.lithology_code == "Topsoil").polygon
    assert fill.intersection(topsoil).length == pytest.approx((101.0 + 99.5) / 2 - 98.5)


def test_opposing_base_units_mirror_under_transect_reversal() -> None:
    rows = [
        ("A", 0.0, 100.0, 95.0, "Clay"),
        ("A", 0.0, 95.0, 82.0, "Sand"),
        ("B", 50.0, 100.0, 91.0, "Clay"),
        ("B", 50.0, 91.0, 87.0, "Gravel"),
    ]
    forward = build_stratigraphy(_rows_df(rows), allow_pinch_outs=True)
    backward = build_stratigraphy(_rows_df(_mirror_rows(rows, 50.0)), allow_pinch_outs=True)
    assert _mirrored_signature(forward, False) == _mirrored_signature(backward, True)


@pytest.mark.parametrize("at_top", [False, True])
def test_lone_single_bounded_units_run_to_neighbour_hole_end(at_top: bool) -> None:
    """Nothing logged opposite (the neighbour stops at the contact): the unit thins to
    zero at the neighbour's hole end, so its edge follows the fence base (or ground)
    line instead of leaving a triangle under the tip."""
    if at_top:
        rows = [
            ("A", 0.0, 102.0, 101.0, "Topsoil"),
            ("A", 0.0, 101.0, 99.0, "Fill"),
            ("A", 0.0, 99.0, 85.0, "Clay"),
            ("B", 50.0, 97.0, 80.0, "Clay"),
        ]
    else:
        rows = [
            ("A", 0.0, 100.0, 90.0, "Clay"),
            ("A", 0.0, 90.0, 86.0, "Sand"),
            ("A", 0.0, 86.0, 80.0, "Gravel"),
            ("B", 50.0, 100.0, 88.0, "Clay"),
        ]
    polygons = build_stratigraphy(_rows_df(rows), allow_pinch_outs=True)
    gap, overlap = _fence_envelope_gap_and_overlap(polygons, rows)
    assert gap == pytest.approx(0.0, abs=1e-6)
    assert overlap == pytest.approx(0.0, abs=1e-6)
    assert detect_polygon_overlaps(polygons) == []
    _assert_pinch_outs_attached(polygons, {"A": 0.0, "B": 50.0})
    hole_end = Point(50.0, 97.0 if at_top else 88.0)
    for wedge in (p for p in polygons if p.is_pinch_out):
        assert wedge.polygon.boundary.distance(hole_end) < 1e-9
        assert wedge.polygon.bounds[0] == pytest.approx(0.0)


def test_client_bb_base_facies_changes_leave_no_white_fence() -> None:
    """Client B-B' east end: base units change between every hole (Sandy Clay Loam,
    Sandy Clay, Silty Clay, Sandy Clay) and BH23-03 vs 2017-BH12 swap Sand for
    Loamy Sand mid-log."""
    logs = {
        "BH23-03": (15.0, [(0.0, 0.3, "Fill"), (0.3, 2.0, "Clay"), (2.0, 4.0, "Sand"),
                           (4.0, 7.0, "Clay Loam"), (7.0, 10.0, "Sandy Clay Loam")]),
        "2017-BH12": (20.0, [(0.0, 0.4, "Fill"), (0.4, 1.8, "Clay"), (1.8, 3.5, "Loamy Sand"),
                             (3.5, 6.5, "Clay Loam"), (6.5, 9.0, "Sandy Clay")]),
        "BH24-08": (26.0, [(0.0, 0.3, "Fill"), (0.3, 1.5, "Clay"), (1.5, 3.5, "Sand"),
                           (3.5, 6.0, "Clay Loam"), (6.0, 8.0, "Silty Clay")]),
        "BH23-01": (32.0, [(0.0, 0.3, "Fill"), (0.3, 1.2, "Clay"), (1.2, 2.5, "Sand"),
                           (2.5, 4.5, "Clay Loam"), (4.5, 6.0, "Sandy Clay")]),
    }  # fmt: skip
    rows = [
        (hole, x, 635.0 - top, 635.0 - bottom, code)
        for hole, (x, units) in logs.items()
        for top, bottom, code in units
    ]
    polygons = build_stratigraphy(_rows_df(rows), allow_pinch_outs=True)
    gap, overlap = _fence_envelope_gap_and_overlap(polygons, rows)
    assert gap == pytest.approx(0.0, abs=1e-6)
    assert overlap == pytest.approx(0.0, abs=1e-6)
    assert detect_polygon_overlaps(polygons) == []
    _assert_pinch_outs_attached(polygons, {hole: x for hole, (x, _units) in logs.items()})


def test_pinch_anchor_edge_prefers_hole_edge_over_facies_boundary() -> None:
    from stratigraphy import _pinch_anchor_edge

    # Facies-change tile: 2 m on the hole (x=0), 20 m on the mid-span boundary (x=25).
    tile = ShapelyPolygon([(0.0, 10.0), (0.0, 8.0), (25.0, 0.0), (25.0, 20.0)])
    assert _pinch_anchor_edge(tile).bounds[0] == pytest.approx(25.0)
    assert _pinch_anchor_edge(tile, (0.0, 50.0)).bounds[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Pinch-outs off: bracketing fills close the space of undrawn unmatched units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mirror", [False, True])
def test_pinch_outs_off_bracketing_contacts_converge_at_source_hole(mirror: bool) -> None:
    """Regression (QA c_nopinch / sA_nopinch): with pinch-outs off the unmatched Silt
    left a white wedge. The upper unit's base and lower unit's top now meet at the
    Silt's mid-elevation on the hole that logged it, so the fence tiles."""
    projected = _shifted_unit_order_pair()
    if mirror:
        projected = projected.assign(x_profile=32.0 - projected["x_profile"])
    polygons = build_stratigraphy(projected, allow_pinch_outs=False)
    assert not any(p.is_pinch_out for p in polygons)
    assert "Silt" not in {p.lithology_code for p in polygons}
    z_top = (629.0, 629.5) if not mirror else (629.5, 629.0)
    z_bot = (602.0, 603.5) if not mirror else (603.5, 602.0)
    assert _fence_gap_area(polygons, 0.0, 32.0, z_top, z_bot) == pytest.approx(0.0, abs=1e-6)
    assert _pairwise_overlap_area(polygons) == pytest.approx(0.0, abs=1e-6)
    assert detect_polygon_overlaps(polygons) == []
    meet = Point(0.0 if mirror else 32.0, (624.0 + 619.5) / 2.0)
    for code in ("Sand and Clay", "Clay"):
        polygon = next(p for p in polygons if p.lithology_code == code).polygon
        assert polygon.boundary.distance(meet) < 1e-9


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("at_top", [False, True])
@pytest.mark.parametrize("opposing", [False, True])
def test_pinch_outs_off_single_bounded_units_leave_no_gap(
    reverse: bool, at_top: bool, opposing: bool
) -> None:
    """Base (or top) units logged in one hole only: the bracketing matched unit takes
    their space, running to the hole end, so the fence envelope stays fully tiled."""
    rows = [
        ("A", 0.0, 100.0, 95.0, "Clay Loam"),
        ("A", 0.0, 95.0, 85.0, "Sandy Clay Loam"),
        ("B", 50.0, 100.0, 92.0, "Clay Loam"),
    ]
    if opposing:
        rows.append(("B", 50.0, 92.0, 88.0, "Sandy Clay"))
    if at_top:
        rows = [(hole, x, -bottom, -top, code) for hole, x, top, bottom, code in rows]
    if reverse:
        rows = _mirror_rows(rows, 50.0)
    projected = _rows_df(rows)
    before = projected.copy()
    polygons = build_stratigraphy(projected, allow_pinch_outs=False)
    pd.testing.assert_frame_equal(projected, before)  # hole data untouched
    assert [p.lithology_code for p in polygons] == ["Clay Loam"]
    gap, overlap = _fence_envelope_gap_and_overlap(polygons, rows)
    assert gap == pytest.approx(0.0, abs=1e-6)
    assert overlap == pytest.approx(0.0, abs=1e-6)


def test_pinch_outs_off_synthetic_site_fence_has_no_gaps() -> None:
    """Regression (QA sA_nopinch slivers at BH-02/03 and BH-04/05)."""
    from io import BytesIO

    from batch_export import transect_points_from_collars
    from ingestion import ingest_workbook
    from models import subset_parse_result
    from pipeline import compute_section_geometry
    from tests.test_multi_section_workbook import build_site_workbook_bytes

    parse_result, _report = ingest_workbook(BytesIO(build_site_workbook_bytes(site_unit_order=True)))
    hole_ids = parse_result.section_specs[0].hole_ids
    subset = subset_parse_result(parse_result, hole_ids)
    geometry = compute_section_geometry(
        subset.collars,
        subset.lithologies,
        transect_points_from_collars(parse_result.collars, hole_ids),
        allow_pinch_outs=False,
    )
    holes = (
        geometry.projected.groupby("hole_id")
        .agg(
            x=("x_profile", "first"),
            top=("top_elevation", "max"),
            bottom=("bottom_elevation", "min"),
        )
        .sort_values("x")
    )
    rows = list(holes.itertuples())
    gap = overlap = 0.0
    for left, right in zip(rows, rows[1:]):
        in_pair = [p for p in geometry.polygons if p.hole_pair == (left.Index, right.Index)]
        assert in_pair
        gap += _fence_gap_area(
            in_pair, left.x, right.x, (left.top, right.top), (left.bottom, right.bottom)
        )
        overlap += _pairwise_overlap_area(in_pair)
    assert gap == pytest.approx(0.0, abs=1e-3)
    assert overlap == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Depth mode (depth below collar) must tile exactly like elevation mode
# ---------------------------------------------------------------------------


def _depth_mode_polygons(
    polygons: list[GeologicalPolygon],
    collars: dict[str, float],
    hole_xs: dict[str, float],
) -> list[GeologicalPolygon]:
    """Fence polygons as the renderer plots them on a depth-below-collar axis."""
    from types import SimpleNamespace

    import numpy as np

    from renderer import CrossSectionRenderer

    fake = SimpleNamespace(profile=SimpleNamespace(y_axis_mode="depth_below_collar"))
    out = []
    for polygon in polygons:
        verts = CrossSectionRenderer._fence_plot_coords(
            fake,
            np.asarray(polygon.polygon.exterior.coords, dtype=float),
            1.0,
            collars,
            polygon.hole_pair,
            hole_xs,
        )
        out.append(
            GeologicalPolygon(
                polygon.lithology_code,
                ShapelyPolygon(verts),
                polygon.hole_pair,
                polygon.is_pinch_out,
            )
        )
    return out


def test_depth_mode_fence_tiles_like_elevation_mode() -> None:
    """Regression (QA sA_depth_*): with unequal collars the depth transform used a
    left/right collar step per polygon, tearing pinch-out wedges off the bent fills
    (white triangle above, overlap below). Interpolating the collar along the pair is
    affine, so the depth fence is the elevation fence mirrored: same areas, no gaps,
    no overlaps, hole vertices at exact per-hole depths."""
    collars = {"BH-01": 632.0, "BH-02": 629.0}
    projected = _shifted_unit_order_pair()
    shift = projected["hole_id"].map({"BH-01": 0.0, "BH-02": -2.5})
    projected = projected.assign(
        collar_elevation=projected["hole_id"].map(collars),
        top_elevation=projected["top_elevation"] + shift,
        bottom_elevation=projected["bottom_elevation"] + shift,
    )
    polygons = build_stratigraphy(projected, allow_pinch_outs=True)
    assert any(p.is_pinch_out for p in polygons)
    depth = _depth_mode_polygons(polygons, collars, {"BH-01": 0.0, "BH-02": 32.0})
    for elevation_poly, depth_poly in zip(polygons, depth):
        assert depth_poly.polygon.is_valid
        assert depth_poly.polygon.area == pytest.approx(elevation_poly.polygon.area, rel=1e-9)
    assert _pairwise_overlap_area(depth) == pytest.approx(0.0, abs=1e-6)
    # Envelope from the top of Sand-and-Clay to the base of Clay, as depths.
    gap = _fence_gap_area(depth, 0.0, 32.0, (3.0, 2.0), (30.0, 28.0))
    assert gap == pytest.approx(0.0, abs=1e-6)
    silt = next(p for p in depth if p.lithology_code == "Silt").polygon
    assert silt.boundary.distance(Point(32.0, 629.0 - 621.5)) < 1e-9
    assert silt.boundary.distance(Point(32.0, 629.0 - 617.0)) < 1e-9
