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
    """allow_pinch_outs=False keeps straight hole-to-hole fills (no bends, gap left)."""
    polygons = build_stratigraphy(_shifted_unit_order_pair(), allow_pinch_outs=False)
    assert not any(p.is_pinch_out for p in polygons)
    for polygon in polygons:
        assert len(polygon.polygon.exterior.coords) == 5
