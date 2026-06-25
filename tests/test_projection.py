"""Tests for projection.py spatial math."""

from __future__ import annotations

import numpy as np
import pytest

from models import Collar, Lithology, Transect
from projection import (
    _TransectGeometry,
    _cumulative_distance,
    _find_closest_segment,
    project_boreholes,
    project_collar_to_transect,
)


def test_project_point_on_first_segment(axis_transect: Transect) -> None:
    x_profile, offset = project_collar_to_transect(25.0, 0.0, axis_transect)
    assert x_profile == 25.0
    assert offset == 0.0


def test_project_point_on_second_segment(axis_transect: Transect) -> None:
    x_profile, offset = project_collar_to_transect(100.0, 20.0, axis_transect)
    assert x_profile == 120.0
    assert offset == 0.0


def test_closest_segment_at_corner() -> None:
    transect = [(0.0, 0.0), (100.0, 0.0), (100.0, 50.0)]
    b = np.array([100.0, 10.0])
    segment_index, t, projected, distance = _find_closest_segment(b, transect)
    assert segment_index == 1
    assert t == pytest.approx(0.2)
    assert projected[0] == pytest.approx(100.0)
    assert projected[1] == pytest.approx(10.0)
    assert distance == pytest.approx(0.0)


def test_cumulative_distance_matches_segment_sum() -> None:
    transect = [(0.0, 0.0), (100.0, 0.0), (100.0, 50.0)]
    distance = _cumulative_distance(transect, segment_index=1, t=0.4)
    assert distance == pytest.approx(120.0)


def test_project_boreholes_elevations(axis_collars, axis_lithologies, axis_transect) -> None:
    projected = project_boreholes(axis_collars, axis_lithologies, axis_transect)
    bh01 = projected[projected["hole_id"] == "BH-01"].iloc[0]
    assert bh01["top_elevation"] == 100.0
    assert bh01["bottom_elevation"] == 95.0
    assert projected["x_profile"].is_monotonic_increasing


def test_project_boreholes_perpendicular_offset() -> None:
    collars = [Collar(hole_id="BH-01", easting=0.0, northing=10.0, elevation=50.0, total_depth=10.0)]
    lithologies = [Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Clay")]
    transect = Transect(points=[(0.0, 0.0), (100.0, 0.0)])
    _, offset = project_collar_to_transect(0.0, 10.0, transect)
    assert offset == 10.0

    projected = project_boreholes(collars, lithologies, transect)
    assert projected.iloc[0]["x_profile"] == 0.0


def test_project_many_matches_single_project(axis_transect: Transect) -> None:
    geometry = _TransectGeometry.from_transect(axis_transect)
    eastings = np.array([0.0, 25.0, 100.0, 100.0])
    northings = np.array([0.0, 0.0, 0.0, 20.0])
    x_profiles, offsets = geometry.project_many(eastings, northings)
    for index, (easting, northing) in enumerate(zip(eastings, northings)):
        x_single, offset_single = geometry.project(float(easting), float(northing))
        assert x_profiles[index] == pytest.approx(x_single)
        assert offsets[index] == pytest.approx(offset_single)
    assert x_profiles.tolist() == pytest.approx([0.0, 25.0, 100.0, 120.0])
    assert offsets.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
