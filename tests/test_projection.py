"""Tests for projection.py spatial math."""

from __future__ import annotations

import numpy as np
import pytest

from models import Collar, DeviationReading, Lithology, Transect
from projection import (
    _TransectGeometry,
    _cumulative_distance,
    _find_closest_segment,
    build_minimum_curvature_survey,
    interpolate_survey_at_depth,
    project_boreholes,
    project_collar_to_transect,
    select_and_order_holes_near_transect,
    select_holes_near_transect,
    order_holes_along_transect,
    transect_length_m,
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


def test_transect_length_m_sums_segments() -> None:
    transect = Transect(points=[(0.0, 0.0), (100.0, 0.0), (100.0, 50.0)])
    assert transect_length_m(transect) == pytest.approx(150.0)


def test_select_holes_near_transect_excludes_far_collar() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-03", easting=25.0, northing=80.0, elevation=100.0, total_depth=10.0),
    ]
    transect = Transect(points=[(0.0, 0.0), (50.0, 0.0)])
    near = select_holes_near_transect(collars, transect, offset_threshold_m=50.0)
    assert near == ("BH-01", "BH-02")


def test_order_holes_along_transect_sorts_by_profile_distance() -> None:
    collars = [
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    transect = Transect(points=[(0.0, 0.0), (50.0, 0.0)])
    ordered = order_holes_along_transect(collars, transect, ("BH-02", "BH-01"))
    assert ordered == ("BH-01", "BH-02")


def test_select_and_order_holes_near_transect_single_pass() -> None:
    collars = [
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-03", easting=25.0, northing=80.0, elevation=100.0, total_depth=10.0),
    ]
    transect = Transect(points=[(0.0, 0.0), (50.0, 0.0)])
    ordered = select_and_order_holes_near_transect(collars, transect, offset_threshold_m=50.0)
    assert ordered == ("BH-01", "BH-02")


def test_minimum_curvature_shifts_deviated_hole_east() -> None:
    collar = Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0)
    readings = [
        DeviationReading(hole_id="BH-01", depth=10.0, inclination_deg=45.0, azimuth_deg=90.0),
    ]
    stations = build_minimum_curvature_survey(collar, readings)
    easting, northing, tvd = interpolate_survey_at_depth(stations, 10.0)
    assert easting > 0.0
    assert northing == pytest.approx(0.0, abs=0.01)
    assert tvd < 10.0


def test_project_boreholes_uses_deviation_survey() -> None:
    collar = Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0)
    lithologies = [Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Sand")]
    readings = [
        DeviationReading(hole_id="BH-01", depth=10.0, inclination_deg=45.0, azimuth_deg=90.0),
    ]
    transect = Transect(points=[(0.0, 0.0), (100.0, 0.0)])
    vertical = project_boreholes([collar], lithologies, transect)
    deviated = project_boreholes([collar], lithologies, transect, deviation_readings=readings)
    assert deviated.iloc[0]["x_profile"] > vertical.iloc[0]["x_profile"]
    assert deviated.iloc[0]["bottom_elevation"] > vertical.iloc[0]["bottom_elevation"]
