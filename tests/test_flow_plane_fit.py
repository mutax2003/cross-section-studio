"""Plan-view groundwater flow from a planar head fit (hydro_metrics.fit_head_plane).

Conventions under test:
* flow azimuth = direction of DECREASING head (−∇h), degrees clockwise from grid north;
* 3 wells → exact three-point solution; ≥4 → least squares with R² and residuals;
* near-collinear wells are flagged and give no direction;
* section relation folds the undirected line to 0–90° and reports |∇h|·cos θ.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from hydro_metrics import (
    HeadPoint,
    compass_point,
    fit_head_plane,
    fit_head_planes,
    format_flow_direction,
    groundwater_flow_notes,
    head_plane_fit_warnings,
    section_flow_relation,
)
from models import Collar, DeviationReading, Lithology, WaterLevel
from section_build_request import SectionBuildRequest

WELLS = [(0.0, 0.0), (200.0, 0.0), (0.0, 200.0), (200.0, 200.0), (90.0, 130.0)]


def _plane_points(flow_azimuth_deg: float, gradient: float, *, wells=WELLS, origin=(0.0, 0.0)):
    """Heads on a plane whose flow (−∇h) points toward ``flow_azimuth_deg``."""
    az = math.radians(flow_azimuth_deg)
    # −∇h = gradient · (sin az, cos az)  ⇒  ∇h = −gradient · (sin az, cos az)
    dh_de = -gradient * math.sin(az)
    dh_dn = -gradient * math.cos(az)
    return [
        HeadPoint(f"W{i}", origin[0] + e, origin[1] + n, 50.0 + dh_de * e + dh_dn * n)
        for i, (e, n) in enumerate(wells)
    ]


@pytest.mark.parametrize("azimuth", [0.0, 90.0, 225.0, 128.0, 315.0, 180.0])
def test_synthetic_plane_recovers_azimuth_and_gradient(azimuth: float) -> None:
    fit = fit_head_plane(_plane_points(azimuth, 0.004))
    assert fit is not None and fit.usable
    assert fit.method == "planar fit"
    assert fit.gradient == pytest.approx(0.004, rel=1e-9)
    assert fit.flow_azimuth_deg is not None
    delta = (fit.flow_azimuth_deg - azimuth + 180.0) % 360.0 - 180.0
    assert abs(delta) < 1e-6
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.max_abs_residual_m < 1e-9


def test_utm_scale_coordinates_are_stable() -> None:
    fit = fit_head_plane(_plane_points(60.0, 0.002, origin=(512_345.0, 6_543_210.0)))
    assert fit is not None and fit.usable
    assert fit.flow_azimuth_deg == pytest.approx(60.0, abs=1e-6)
    assert fit.gradient == pytest.approx(0.002, rel=1e-8)


def test_flow_points_toward_decreasing_head() -> None:
    # Head falls to the east: flow toward 90° (E), not 270°.
    points = [
        HeadPoint("A", 0.0, 0.0, 10.0),
        HeadPoint("B", 100.0, 0.0, 9.0),
        HeadPoint("C", 0.0, 100.0, 10.0),
    ]
    fit = fit_head_plane(points)
    assert fit is not None
    assert fit.flow_azimuth_deg == pytest.approx(90.0)
    assert format_flow_direction(fit) == "toward 90° (E)"
    assert compass_point(128.0) == "SE"
    assert compass_point(359.0) == "N"
    assert compass_point(225.0) == "SW"


def test_three_point_exact_solution() -> None:
    fit = fit_head_plane(_plane_points(30.0, 0.01, wells=WELLS[:3]))
    assert fit is not None and fit.usable
    assert fit.method == "3-point"
    assert fit.n_wells == 3
    assert fit.r_squared is None  # exact: R² not meaningful
    assert fit.max_abs_residual_m == 0.0
    assert fit.flow_azimuth_deg == pytest.approx(30.0, abs=1e-9)
    assert fit.gradient == pytest.approx(0.01, rel=1e-12)


def test_fewer_than_three_wells_returns_none() -> None:
    assert fit_head_plane(_plane_points(0.0, 0.01, wells=WELLS[:2])) is None


def test_collinear_wells_rejected() -> None:
    line = [(0.0, 0.0), (100.0, 100.0), (250.0, 250.2), (400.0, 400.0)]
    fit = fit_head_plane(_plane_points(45.0, 0.01, wells=line))
    assert fit is not None
    assert fit.collinear and not fit.usable
    assert fit.flow_azimuth_deg is None
    assert section_flow_relation(fit, 45.0) is None
    warnings = head_plane_fit_warnings(fit)
    assert warnings and "collinear" in warnings[0]


def test_thin_triangle_rejected_by_ratio() -> None:
    # 1 km baseline, 20 m off-line: ratio ≈ 0.02 < 0.05.
    fit = fit_head_plane(_plane_points(10.0, 0.01, wells=[(0, 0), (1000, 0), (500, 20)]))
    assert fit is not None and fit.collinear


def test_noisy_fit_reports_r_squared_and_residuals() -> None:
    rng = np.random.default_rng(7)
    grid = [(float(e), float(n)) for e in range(0, 1001, 250) for n in range(0, 1001, 250)]
    points = _plane_points(200.0, 0.005, wells=grid)
    noisy = [
        HeadPoint(p.hole_id, p.easting, p.northing, p.head_masl + float(rng.normal(0, 0.05)))
        for p in points
    ]
    fit = fit_head_plane(noisy)
    assert fit is not None and fit.usable
    assert 0.95 < fit.r_squared < 1.0
    assert fit.flow_azimuth_deg == pytest.approx(200.0, abs=3.0)
    assert fit.max_abs_residual_m == pytest.approx(
        max(abs(r) for _h, r in fit.residuals_m)
    )
    assert not fit.poor_fit


def test_mounded_heads_flagged_not_planar() -> None:
    points = [
        HeadPoint("A", 0, 0, 10.0),
        HeadPoint("B", 200, 0, 10.0),
        HeadPoint("C", 0, 200, 10.0),
        HeadPoint("D", 200, 200, 10.0),
        HeadPoint("M", 100, 100, 12.0),  # mound in the middle
        HeadPoint("E", 100, 0, 10.2),
    ]
    fit = fit_head_plane(points)
    assert fit is not None and fit.poor_fit
    warnings = head_plane_fit_warnings(fit)
    assert any("not planar" in w and "mounding" in w for w in warnings)


@pytest.mark.parametrize(
    ("flow", "section", "angle", "cls"),
    [
        (90.0, 90.0, 0.0, "along section"),
        (90.0, 270.0, 0.0, "along section"),  # undirected line
        (128.0, 86.0, 42.0, "oblique"),
        (0.0, 80.0, 80.0, "nearly perpendicular"),
        (10.0, 345.0, 25.0, "oblique"),
        (225.0, 45.0, 0.0, "along section"),
    ],
)
def test_section_relation_angles(flow: float, section: float, angle: float, cls: str) -> None:
    fit = fit_head_plane(_plane_points(flow, 0.004))
    relation = section_flow_relation(fit, section)
    assert relation is not None
    assert relation.angle_deg == pytest.approx(angle, abs=1e-6)
    assert relation.classification == cls
    assert relation.along_section_gradient == pytest.approx(
        0.004 * math.cos(math.radians(angle)), rel=1e-6
    )


def test_perpendicular_section_warns_apparent() -> None:
    fit = fit_head_plane(_plane_points(0.0, 0.004))
    warnings = head_plane_fit_warnings(fit, section_flow_relation(fit, 90.0))
    assert any("apparent gradients" in w for w in warnings)


def _workbook(heads: dict[str, float], *, extra_collars=()):
    collars = [
        Collar(hole_id=hid, easting=e, northing=n, elevation=100.0, total_depth=40.0)
        for hid, (e, n) in zip(heads, WELLS, strict=False)
    ] + list(extra_collars)
    levels = [WaterLevel(hole_id=hid, depth=100.0 - h) for hid, h in heads.items()]
    return collars, levels


def test_btoc_and_inclined_heads_flow_through_water_head_masl() -> None:
    # Truth: flow due east, i = 0.01 → heads 60 at E=0, 58 at E=200.
    collars = [
        Collar(hole_id="A", easting=0, northing=0, elevation=100.0, total_depth=60.0, stick_up_m=1.0),
        Collar(hole_id="B", easting=200, northing=0, elevation=100.0, total_depth=60.0),
        Collar(
            hole_id="C",
            easting=0,
            northing=200,
            elevation=100.0,
            total_depth=80.0,
            inclination_deg=60.0,
            azimuth_deg=0.0,
        ),
    ]
    deviations = [
        DeviationReading(hole_id="C", depth=0.0, inclination_deg=60.0, azimuth_deg=0.0),
        DeviationReading(hole_id="C", depth=80.0, inclination_deg=60.0, azimuth_deg=0.0),
    ]
    # A: btoc 41 m with 1 m stick-up → 40 m bgs → head 60.
    # C: 60° from vertical → TVD = MD·cos60 = MD/2; head 60 → TVD 40 → MD 80 along hole.
    levels = [
        WaterLevel(hole_id="A", depth=41.0, depth_reference="btoc"),
        WaterLevel(hole_id="B", depth=42.0),
        WaterLevel(hole_id="C", depth=80.0),
    ]
    (fit,) = fit_head_planes(collars, levels, deviation_readings=deviations)
    assert fit.usable
    assert fit.flow_azimuth_deg == pytest.approx(90.0, abs=1e-3)
    assert fit.gradient == pytest.approx(0.01, rel=1e-3)
    # The collar angle alone (no Deviations rows) is a straight inclined hole.
    (collar_only,) = fit_head_planes(collars, levels)
    assert collar_only.flow_azimuth_deg == pytest.approx(90.0, abs=1e-3)
    # Treated as vertical, C would read collar − 80 = 20 m: wrong direction.
    vertical = [*collars[:2], collars[2].model_copy(update={"inclination_deg": None})]
    (naive,) = fit_head_planes(vertical, levels)
    assert naive.flow_azimuth_deg == pytest.approx(
        math.degrees(math.atan2(0.01, 0.2)), abs=0.5
    )


def test_grouping_dry_excluded_last_wins_and_nests() -> None:
    collars, _ = _workbook({"A": 0, "B": 0, "C": 0, "D": 0})
    levels = [
        WaterLevel(hole_id="A", depth=40.0),
        WaterLevel(hole_id="B", depth=99.0),  # superseded (last wins)
        WaterLevel(hole_id="B", depth=42.0),
        WaterLevel(hole_id="C", depth=40.0),
        WaterLevel(hole_id="D", depth=0.0, status="dry"),
        WaterLevel(hole_id="A", depth=30.0, connect_group="deep"),
        WaterLevel(hole_id="B", depth=30.0, connect_group="deep"),
    ]
    fits = fit_head_planes(collars, levels)
    shallow = next(f for f in fits if f.connect_group == "")
    assert shallow.hole_ids == ("A", "B", "C")
    assert shallow.flow_azimuth_deg == pytest.approx(90.0)
    # Deep nest has only 2 wells → no fit.
    assert all(f.connect_group != "deep" for f in fits)


def test_flow_note_text() -> None:
    collars, levels = _workbook({"A": 60.0, "B": 58.0, "C": 60.0, "D": 58.0, "E": 59.1})
    notes = groundwater_flow_notes(collars, levels, 45.0)
    assert len(notes) == 1
    assert notes[0].startswith("Groundwater flow (planar fit, 5 wells): toward 90° (E), i = 0.01")
    assert "section is oblique (45°) — gradients shown are apparent." in notes[0]


def _section_inputs():
    collars, levels = _workbook({"A": 60.0, "B": 58.0, "C": 60.0, "D": 58.0, "E": 59.1})
    lithologies = [
        Lithology(hole_id=c.hole_id, from_depth=0.0, to_depth=40.0, lithology_code="SAND")
        for c in collars
    ]
    return collars, lithologies, levels


def test_pipeline_note_present_and_absent() -> None:
    from pipeline import build_cross_section

    collars, lithologies, levels = _section_inputs()
    points = [(-10.0, 100.0), (210.0, 100.0)]
    fmt = frozenset({"svg"})
    with_water = build_cross_section(
        collars, lithologies, points, water_levels=levels, export_formats=fmt
    )
    flow = [n for n in with_water.qa_notes if n.startswith("Groundwater flow")]
    assert len(flow) == 1 and "along flow (0°)" in flow[0]
    assert flow[0] in with_water.overlap_warnings

    without_water = build_cross_section(collars, lithologies, points, export_formats=fmt)
    assert not any(n.startswith("Groundwater flow") for n in without_water.qa_notes)

    # Only two measured wells: no note.
    two = build_cross_section(
        collars, lithologies, points, water_levels=levels[:2], export_formats=fmt
    )
    assert not any(n.startswith("Groundwater flow") for n in two.qa_notes)

    suppressed = build_cross_section(
        collars,
        lithologies,
        points,
        water_levels=levels,
        export_formats=fmt,
        groundwater_flow_notes=(),
    )
    assert not any(n.startswith("Groundwater flow") for n in suppressed.qa_notes)


def test_flow_notes_do_not_touch_geometry_cache_payload() -> None:
    base = SectionBuildRequest(transect_points=((0.0, 0.0), (1.0, 0.0)))
    noted = base.model_copy(update={"groundwater_flow_notes": ("Groundwater flow (…)",)})
    assert base.geometry_cache_payload() == noted.geometry_cache_payload()
    assert "groundwater_flow_notes" not in noted.geometry_cache_payload()
    assert base.cache_key(("A",)) != noted.cache_key(("A",))


def test_validate_flow_panel_apptest() -> None:
    from streamlit.testing.v1 import AppTest

    def _panel() -> None:
        from types import SimpleNamespace

        from app_validate import _render_groundwater_flow
        from models import Collar, WaterLevel

        wells = {"A": (0, 0, 60.0), "B": (200, 0, 58.0), "C": (0, 200, 60.0)}
        result = SimpleNamespace(
            collars=[
                Collar(hole_id=h, easting=e, northing=n, elevation=100.0, total_depth=50.0)
                for h, (e, n, _hd) in wells.items()
            ],
            water_levels=[WaterLevel(hole_id=h, depth=100.0 - hd) for h, (_e, _n, hd) in wells.items()],
            deviation_readings=(),
        )
        _render_groundwater_flow(result, ((0.0, -50.0), (0.0, 250.0)))

    at = AppTest.from_function(_panel, default_timeout=60)
    at.run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "toward 90° (E)" in text and "A, B, C" in text
    assert any("nearly perpendicular" in w.value for w in at.warning)
