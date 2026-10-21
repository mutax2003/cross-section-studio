"""QA fixes for borehole survey math, checked against an independent reference.

* a Deviations row at depth 0 sets the collar orientation;
* a Collars-sheet inclination with no Deviations rows is a straight inclined hole
  (lithology, water heads and the MD→TVD tables agree);
* TVD between survey stations follows the minimum-curvature arc (< 1 cm);
* vertical holes stay on the legacy ``collar − depth`` path;
* the dry-well legend key reads cleanly.
"""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest

from hydro_metrics import head_points_by_group
from models import Collar, DeviationReading, Lithology, Transect, WaterLevel
from pipeline import compute_section_geometry
from projection import (
    build_along_hole_tvd,
    build_minimum_curvature_survey,
    interpolate_survey_many,
    project_boreholes,
)
from renderer_common import water_status_legend_label

# --- independent minimum-curvature reference --------------------------------------


def _unit(inc_deg: float, azi_deg: float) -> np.ndarray:
    inc, azi = math.radians(inc_deg), math.radians(azi_deg)
    return np.array([math.sin(inc) * math.cos(azi), math.sin(inc) * math.sin(azi), math.cos(inc)])


def _reference_point(stations: list[tuple[float, float, float]], md: float) -> np.ndarray:
    """(north, east, tvd) at ``md`` on the minimum-curvature path through ``stations``.

    ``stations`` are (md, inc°, azi°) starting at md 0. Each segment is a circular
    arc: integrate the slerped tangent numerically (Simpson, fine steps) so the
    reference shares no closed-form code with ``projection``.
    """
    position = np.zeros(3)
    for (md_a, inc_a, azi_a), (md_b, inc_b, azi_b) in zip(stations[:-1], stations[1:], strict=True):
        if md <= md_a:
            break
        t_a, t_b = _unit(inc_a, azi_a), _unit(inc_b, azi_b)
        dogleg = math.acos(max(-1.0, min(1.0, float(t_a @ t_b))))
        end = min(md, md_b)
        length = md_b - md_a

        def tangent(s: float, t_a=t_a, t_b=t_b, dogleg=dogleg, md_a=md_a, length=length) -> np.ndarray:
            f = (s - md_a) / length
            if dogleg < 1e-12:
                return t_a
            return (math.sin((1 - f) * dogleg) * t_a + math.sin(f * dogleg) * t_b) / math.sin(dogleg)

        n = 2000
        grid = np.linspace(md_a, end, 2 * n + 1)
        values = np.array([tangent(s) for s in grid])
        h = (end - md_a) / (2 * n)
        position = position + h / 3 * (
            values[0] + values[-1] + 4 * values[1:-1:2].sum(axis=0) + 2 * values[2:-1:2].sum(axis=0)
        )
    return position


# --- 1. depth-0 survey row sets the collar orientation ----------------------------


def test_depth_zero_row_sets_collar_orientation() -> None:
    collar = Collar(hole_id="C", easting=0, northing=0, elevation=200, total_depth=100)
    dev = [DeviationReading(hole_id="C", depth=d, inclination_deg=30, azimuth_deg=45) for d in (0, 50, 100)]
    table = build_along_hole_tvd([collar], dev)["C"]
    cos30 = math.cos(math.radians(30))
    assert table.tvd_at(50) == pytest.approx(50 * cos30, abs=1e-9)  # 43.30, not 47.75
    assert table.tvd_at(100) == pytest.approx(100 * cos30, abs=1e-9)

    stations = build_minimum_curvature_survey(collar, dev)
    easting, northing, _tvd = interpolate_survey_many(stations, [100.0])
    horizontal = 100 * math.sin(math.radians(30))
    assert easting[0] == pytest.approx(horizontal * math.sin(math.radians(45)))
    assert northing[0] == pytest.approx(horizontal * math.cos(math.radians(45)))


def test_depth_zero_row_wins_over_collar_angles() -> None:
    collar = Collar(
        hole_id="C", easting=0, northing=0, elevation=200, total_depth=60,
        inclination_deg=60, azimuth_deg=180,
    )
    dev = [DeviationReading(hole_id="C", depth=d, inclination_deg=20, azimuth_deg=90) for d in (0, 60)]
    table = build_along_hole_tvd([collar], dev)["C"]
    assert table.tvd_at(30) == pytest.approx(30 * math.cos(math.radians(20)), abs=1e-9)


def test_collar_angles_used_when_no_depth_zero_row() -> None:
    collar = Collar(
        hole_id="C", easting=0, northing=0, elevation=200, total_depth=60,
        inclination_deg=20, azimuth_deg=90,
    )
    dev = [DeviationReading(hole_id="C", depth=60, inclination_deg=20, azimuth_deg=90)]
    table = build_along_hole_tvd([collar], dev)["C"]
    assert table.tvd_at(30) == pytest.approx(30 * math.cos(math.radians(20)), abs=1e-9)


# --- 2. collar-only inclined hole -------------------------------------------------


def _inclined_collars() -> list[Collar]:
    return [
        Collar(hole_id="V", easting=0, northing=0, elevation=100, total_depth=40),
        Collar(
            hole_id="I", easting=50, northing=0, elevation=110, total_depth=40,
            inclination_deg=45, azimuth_deg=90,
        ),
    ]


def _inclined_lithologies() -> list[Lithology]:
    return [
        Lithology(hole_id=h, from_depth=a, to_depth=b, lithology_code=c)
        for h in ("V", "I")
        for a, b, c in ((0, 15, "CLAY"), (15, 40, "SAND"))
    ]


def test_collar_inclination_without_deviations_is_straight_inclined_hole() -> None:
    cos45 = math.cos(math.radians(45))
    points = [(-20.0, 0.0), (120.0, 0.0)]
    projected = project_boreholes(
        _inclined_collars(), _inclined_lithologies(), Transect(points=points)
    )
    rows_i = projected[projected["hole_id"] == "I"]
    assert float(rows_i["bottom_elevation"].min()) == pytest.approx(110 - 40 * cos45)
    # Hole runs east along the transect: interval midpoints move with depth.
    mid_sand = 50 + 27.5 * math.sin(math.radians(45)) + 20
    assert float(rows_i["x_profile"].max()) == pytest.approx(mid_sand)
    rows_v = projected[projected["hole_id"] == "V"]
    assert rows_v["bottom_elevation"].tolist() == [85.0, 60.0]

    tables = build_along_hole_tvd(_inclined_collars(), None)
    assert set(tables) == {"I"}
    assert tables["I"].tvd_at(40) == pytest.approx(40 * cos45)
    assert pickle.loads(pickle.dumps(tables["I"])) == tables["I"]

    geometry = compute_section_geometry(_inclined_collars(), _inclined_lithologies(), points)
    assert set(geometry.along_hole_tvd) == {"I"}

    water = [WaterLevel(hole_id="I", depth=20.0), WaterLevel(hole_id="V", depth=20.0)]
    ((_label, points),) = head_points_by_group(_inclined_collars(), water).values()
    heads = {point.hole_id: point.head_masl for point in points}
    assert heads["I"] == pytest.approx(110 - 20 * cos45)
    assert heads["V"] == pytest.approx(80.0)


def test_zero_inclination_collar_stays_vertical_fast_path() -> None:
    collars = [
        Collar(hole_id="Z", easting=0, northing=0, elevation=100, total_depth=40, inclination_deg=0.0),
        Collar(hole_id="N", easting=10, northing=0, elevation=100, total_depth=40),
    ]
    liths = [Lithology(hole_id=h, from_depth=0, to_depth=40, lithology_code="SAND") for h in ("Z", "N")]
    assert build_along_hole_tvd(collars, None) == {}
    projected = project_boreholes(collars, liths, Transect(points=[(-5.0, 0.0), (20.0, 0.0)]))
    assert projected["bottom_elevation"].tolist() == [60.0, 60.0]
    assert projected["x_profile"].tolist() == [5.0, 15.0]


# --- 3. curved survey: arc-accurate interpolation ----------------------------------


@pytest.mark.parametrize(
    "survey",
    [
        [(0.0, 0.0, 0.0), (50.0, 30.0, 45.0), (100.0, 60.0, 120.0)],
        [(0.0, 10.0, 0.0), (30.0, 40.0, 90.0), (90.0, 5.0, 270.0), (150.0, 70.0, 10.0)],
    ],
)
def test_curved_survey_tvd_follows_minimum_curvature_arc(survey) -> None:
    total_depth = survey[-1][0]
    collar = Collar(hole_id="H", easting=0, northing=0, elevation=0, total_depth=total_depth)
    dev = [DeviationReading(hole_id="H", depth=md, inclination_deg=i, azimuth_deg=a) for md, i, a in survey]
    table = build_along_hole_tvd([collar], dev)["H"]
    stations = build_minimum_curvature_survey(collar, dev)
    from projection import densify_survey

    dense = densify_survey(stations)
    for md in np.linspace(0.0, total_depth, 61):
        north, east, tvd = _reference_point(survey, float(md))
        assert table.tvd_at(float(md)) == pytest.approx(tvd, abs=0.01)
        e, n, t = interpolate_survey_many(dense, [float(md)])
        assert t[0] == pytest.approx(tvd, abs=0.01)
        assert e[0] == pytest.approx(east, abs=0.01)
        assert n[0] == pytest.approx(north, abs=0.01)
    # Station TVDs are still the closed-form minimum-curvature values.
    for station, (md, *_rest) in zip(stations, survey, strict=True):
        assert station.true_vertical_depth == pytest.approx(_reference_point(survey, md)[2], abs=1e-6)
    assert len(table.measured_depths) < 400  # stays cheap


def test_curved_survey_lithology_matches_tvd_table() -> None:
    survey = [(0.0, 0.0, 90.0), (50.0, 30.0, 90.0), (100.0, 60.0, 90.0)]
    collar = Collar(hole_id="H", easting=0, northing=0, elevation=300, total_depth=100)
    dev = [DeviationReading(hole_id="H", depth=md, inclination_deg=i, azimuth_deg=a) for md, i, a in survey]
    liths = [Lithology(hole_id="H", from_depth=0, to_depth=25, lithology_code="A"),
             Lithology(hole_id="H", from_depth=25, to_depth=100, lithology_code="B")]
    projected = project_boreholes([collar], liths, Transect(points=[(-10.0, 0.0), (200.0, 0.0)]),
                                  deviation_readings=dev)
    reference_tvd_25 = _reference_point(survey, 25.0)[2]
    top_b = float(projected.loc[projected["lithology_code"] == "B", "top_elevation"].iloc[0])
    assert top_b == pytest.approx(300 - reference_tvd_25, abs=0.01)


# --- 4. legend text ---------------------------------------------------------------


def test_dry_well_legend_label_reads_cleanly() -> None:
    assert water_status_legend_label(["DRY"]) == "DRY — dry well"
    assert water_status_legend_label(["NM"]) == "NM — not measured"
    assert water_status_legend_label(["NM", "DRY"]) == "NM / DRY — not measured / dry well"
    assert water_status_legend_label(["DRY"], upper=True) == "DRY — DRY WELL"


def test_mining_style_negative_collar_dip_is_converted() -> None:
    """−90 (mining dip, straight down) stays vertical; −60 is 30° off vertical."""
    from projection import build_along_hole_tvd

    straight_down = Collar(
        hole_id="M", easting=0.0, northing=0.0, elevation=100.0, total_depth=50.0,
        inclination_deg=-90.0, azimuth_deg=0.0,
    )
    assert "M" not in build_along_hole_tvd([straight_down], [])

    dipping = Collar(
        hole_id="N", easting=0.0, northing=0.0, elevation=100.0, total_depth=50.0,
        inclination_deg=-60.0, azimuth_deg=90.0,
    )
    tvd = build_along_hole_tvd([dipping], [])["N"]
    assert abs(tvd.tvd_at(40.0) - 40.0 * math.cos(math.radians(30.0))) < 1e-6
