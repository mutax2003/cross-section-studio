"""Tests for stratigraphy.py polygon construction."""

from __future__ import annotations

import pandas as pd
import pytest
from shapely.geometry import Point, Polygon as ShapelyPolygon

from stratigraphy import GeologicalPolygon, build_stratigraphy, detect_polygon_overlaps


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
