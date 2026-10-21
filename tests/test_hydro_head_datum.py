"""Groundwater head / screen / chemistry elevations follow the hole survey and depth datum.

* Inclined (surveyed) holes: water levels, screens, chemistry samples, the
  well-column bottom and unlogged TD gaps sit at true vertical depth — the
  same minimum-curvature TVD the lithology uses.
* ``depth_reference = btoc``: head = collar + stick-up − depth; a btoc reading
  on a collar without stick-up is taken as 0 m stick-up and warned on Validate.
* Labels: RL in elevation mode, vertical depth below ground in depth mode.
* Gradients (renderer + QA) use the same corrected heads.
* Vertical holes keep the legacy collar − depth arithmetic exactly.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ai_quality import summarize_water_levels
from hydro_metrics import (
    horizontal_gradients_along_profile,
    water_depth_below_ground_m,
    water_head_masl,
)
from models import (
    Collar,
    DeviationReading,
    EnvironmentalReading,
    Lithology,
    ScreenInterval,
    WaterLevel,
)
from pipeline import compute_section_geometry, render_cross_section_from_geometry
from projection import AlongHoleTVD, build_along_hole_tvd
from render_theme import SCREEN_INTERVAL_HATCH
from renderer_common import unlogged_intervals

COS30 = math.cos(math.radians(30.0))
POINTS = [(-20.0, 0.0), (120.0, 0.0)]


def _collars(*, stick_up_b: float | None = None, inclined_b: bool = True) -> list[Collar]:
    # A non-zero collar inclination makes B a straight inclined hole even
    # without Deviations rows; ``inclined_b=False`` gives a vertical B.
    angle = 30.0 if inclined_b else None
    return [
        Collar(hole_id="A", easting=0.0, northing=0.0, elevation=100.0, total_depth=30.0),
        Collar(
            hole_id="B",
            easting=60.0,
            northing=0.0,
            elevation=102.0,
            total_depth=40.0,
            inclination_deg=angle,
            azimuth_deg=0.0,
            stick_up_m=stick_up_b,
        ),
    ]


def _lithologies(b_logged_to: float = 40.0) -> list[Lithology]:
    return [
        Lithology(hole_id="A", from_depth=0.0, to_depth=12.0, lithology_code="CLAY"),
        Lithology(hole_id="A", from_depth=12.0, to_depth=30.0, lithology_code="SAND"),
        Lithology(hole_id="B", from_depth=0.0, to_depth=12.0, lithology_code="CLAY"),
        Lithology(hole_id="B", from_depth=12.0, to_depth=b_logged_to, lithology_code="SAND"),
    ]


def _deviations() -> list[DeviationReading]:
    # Constant 30° from vertical: TVD = MD × cos 30° exactly (zero dogleg).
    return [
        DeviationReading(hole_id="B", depth=depth, inclination_deg=30.0, azimuth_deg=0.0)
        for depth in (10.0, 20.0, 40.0)
    ]


def _render(
    *,
    collars=None,
    lithologies=None,
    deviations=None,
    water=(),
    screens=(),
    chemistry=(),
    elevation_mode: str = "absolute",
):
    geometry = compute_section_geometry(
        collars or _collars(),
        lithologies or _lithologies(),
        POINTS,
        deviation_readings=_deviations() if deviations is None else deviations,
        interpretation_mode="borehole_only",
    )
    result = render_cross_section_from_geometry(
        geometry,
        POINTS,
        vertical_exaggeration=None,  # data stored at true scale (factor 1)
        interpretation_mode="borehole_only",
        water_levels=list(water),
        screen_intervals=list(screens),
        environmental_readings=list(chemistry),
        environmental_parameters=("Chloride",) if chemistry else None,
        parameter_draw_markers=True if chemistry else None,
        show_water_elevation_labels=True,
        render_layout="section_sheet",
        elevation_mode=elevation_mode,
        export_formats=frozenset({"svg"}),
        close_figure=False,
    )
    bundle = result.retained_export_bundle
    return geometry, bundle["renderer"], bundle["figure"]


def _marker_offsets(figure, kind: str) -> np.ndarray:
    rows = [
        np.asarray(collection._css_base_offsets, dtype=float)
        for ax in figure.axes
        for collection in ax.__dict__.get("_css_marker_sets", [])
        if collection._css_marker_kind == kind
    ]
    assert rows, f"no {kind} markers drawn"
    return np.vstack(rows)


def _screen_y_extents(figure) -> list[tuple[float, float]]:
    extents = []
    for ax in figure.axes:
        for collection in ax.collections:
            hatch = getattr(collection, "get_hatch", lambda: None)()
            if hatch != SCREEN_INTERVAL_HATCH:
                continue
            for path in collection.get_paths():
                ys = path.vertices[:, 1]
                extents.append((float(ys.min()), float(ys.max())))
    return extents


# --- projection table -------------------------------------------------------------


def test_along_hole_tvd_matches_survey_and_is_picklable() -> None:
    import pickle

    tables = build_along_hole_tvd(_collars(), _deviations())
    assert set(tables) == {"B"}
    table = tables["B"]
    assert isinstance(table, AlongHoleTVD)
    for md in (0.0, 5.0, 17.3, 40.0):
        assert table.tvd_at(md) == pytest.approx(md * COS30, abs=1e-9)
    # Past the last station: continue along the last segment.
    assert table.tvd_at(45.0) == pytest.approx(45.0 * COS30, abs=1e-9)
    assert pickle.loads(pickle.dumps(table)) == table


def test_geometry_carries_tvd_tables_only_for_surveyed_holes() -> None:
    geometry = compute_section_geometry(
        _collars(), _lithologies(), POINTS, deviation_readings=_deviations()
    )
    assert set(geometry.along_hole_tvd) == {"B"}
    vertical = compute_section_geometry(_collars(inclined_b=False), _lithologies(), POINTS)
    assert vertical.along_hole_tvd == {}
    # Collar inclination alone (no Deviations rows) is a straight inclined hole.
    collar_only = compute_section_geometry(_collars(), _lithologies(), POINTS)
    assert set(collar_only.along_hole_tvd) == {"B"}


# --- inclined hole: everything at survey TVD --------------------------------------


def test_inclined_hole_water_screen_and_chemistry_sit_at_survey_tvd() -> None:
    water = [
        WaterLevel(hole_id="A", depth=5.0),
        WaterLevel(hole_id="B", depth=10.0),
    ]
    screens = [ScreenInterval(hole_id="B", from_depth=20.0, to_depth=30.0)]
    chemistry = [
        EnvironmentalReading(hole_id="B", parameter="Chloride", value=50.0, depth=15.0),
    ]
    geometry, renderer, figure = _render(water=water, screens=screens, chemistry=chemistry)
    try:
        # Lithology already used TVD: SAND base at MD 40 is 102 − 40 cos30.
        b_rows = geometry.projected[geometry.projected["hole_id"] == "B"]
        assert float(b_rows["bottom_elevation"].min()) == pytest.approx(102.0 - 40.0 * COS30)

        water_y = sorted(_marker_offsets(figure, "water")[:, 1].tolist())
        assert water_y == pytest.approx(sorted([95.0, 102.0 - 10.0 * COS30]))

        (screen,) = _screen_y_extents(figure)
        assert screen == pytest.approx((102.0 - 30.0 * COS30, 102.0 - 20.0 * COS30))

        chem_y = _marker_offsets(figure, "chem")[:, 1]
        assert chem_y.tolist() == pytest.approx([102.0 - 15.0 * COS30])

        # Well column bottom at the true vertical depth of TD.
        hole_summary = (
            geometry.projected.groupby("hole_id", as_index=False)
            .agg(
                x_profile=("x_profile", "first"),
                collar_elevation=("collar_elevation", "first"),
                bottom_elevation=("bottom_elevation", "min"),
            )
        )
        _x, _top, bottom = renderer._well_extents(hole_summary, geometry.collar_depths, {})
        bottoms = dict(zip(hole_summary["hole_id"], bottom, strict=True))
        assert bottoms["A"] == 100.0 - 30.0
        assert bottoms["B"] == pytest.approx(102.0 - 40.0 * COS30)
    finally:
        plt.close(figure)


def test_unlogged_td_gap_uses_tvd_on_inclined_hole() -> None:
    geometry = compute_section_geometry(
        _collars(), _lithologies(b_logged_to=30.0), POINTS, deviation_readings=_deviations()
    )
    gaps = unlogged_intervals(
        geometry.projected, geometry.collar_depths, None, geometry.along_hole_tvd
    )
    b_gap = gaps[gaps["hole_id"] == "B"].iloc[0]
    assert float(b_gap["top_elevation"]) == pytest.approx(102.0 - 30.0 * COS30)
    assert float(b_gap["bottom_elevation"]) == pytest.approx(102.0 - 40.0 * COS30)


# --- depth datum ------------------------------------------------------------------


def test_btoc_reading_adds_stick_up() -> None:
    level = WaterLevel(hole_id="B", depth=6.0, depth_reference="mbtoc")
    assert level.depth_reference == "btoc"
    assert water_head_masl(level, 102.0, stick_up_m=0.8) == pytest.approx(96.8)
    assert water_depth_below_ground_m(level, 102.0, stick_up_m=0.8) == pytest.approx(5.2)
    # Inclined: casing stick-up first, the rest along the hole at TVD.
    table = build_along_hole_tvd(_collars(), _deviations())["B"]
    head = water_head_masl(
        level, 102.0, stick_up_m=0.8, md_to_elevation=lambda md: table.elevation_at(102.0, md)
    )
    assert head == pytest.approx(102.0 - 5.2 * COS30)


def test_btoc_reading_renders_with_stick_up() -> None:
    water = [WaterLevel(hole_id="B", depth=6.0, depth_reference="top of casing")]
    _geometry, _renderer, figure = _render(
        collars=_collars(stick_up_b=0.8, inclined_b=False), deviations=[], water=water
    )
    try:
        assert _marker_offsets(figure, "water")[:, 1].tolist() == pytest.approx([102.0 + 0.8 - 6.0])
    finally:
        plt.close(figure)


@pytest.mark.parametrize("alias", ["bgs", "mbgs", "BGL", "ground", "", None])
def test_depth_reference_bgs_aliases(alias) -> None:
    assert WaterLevel(hole_id="A", depth=1.0, depth_reference=alias).depth_reference == "bgs"


def test_depth_reference_rejects_unknown_datum() -> None:
    with pytest.raises(ValueError, match="depth_reference"):
        WaterLevel(hole_id="A", depth=1.0, depth_reference="sea level")


def test_btoc_without_stick_up_warns_and_uses_zero() -> None:
    level = WaterLevel(hole_id="B", depth=6.0, depth_reference="btoc")
    assert water_head_masl(level, 102.0, stick_up_m=None) == pytest.approx(96.0)
    summary = summarize_water_levels(_collars(), [level], ("A", "B"))
    assert any("stick_up_m is blank" in warning and "B" in warning for warning in summary.warnings)
    with_stick_up = summarize_water_levels(_collars(stick_up_b=0.5), [level], ("A", "B"))
    assert not any("stick_up_m is blank" in warning for warning in with_stick_up.warnings)


def test_td_check_uses_depth_below_ground_for_btoc() -> None:
    # 40.5 m below casing with 0.8 m stick-up is 39.7 m below ground: inside TD 40.
    level = WaterLevel(hole_id="B", depth=40.5, depth_reference="btoc")
    summary = summarize_water_levels(_collars(stick_up_b=0.8), [level], ("A", "B"))
    assert not any("exceeds total depth" in warning for warning in summary.warnings)
    deep = WaterLevel(hole_id="B", depth=41.5, depth_reference="btoc")
    summary = summarize_water_levels(_collars(stick_up_b=0.8), [deep], ("A", "B"))
    assert any("exceeds total depth" in warning for warning in summary.warnings)


# --- labels -----------------------------------------------------------------------


def _rl_label_values(renderer) -> list[float]:
    return sorted(
        float(annotation._water_value)
        for kind, annotation, _c in renderer._water_labels
        if kind == "rl"
    )


def test_labels_in_elevation_and_depth_modes() -> None:
    water = [
        WaterLevel(hole_id="A", depth=5.0),
        WaterLevel(hole_id="B", depth=10.0),
    ]
    _g, renderer, figure = _render(water=water)
    try:
        assert _rl_label_values(renderer) == pytest.approx(sorted([95.0, 102.0 - 10.0 * COS30]))
    finally:
        plt.close(figure)
    _g, renderer, figure = _render(water=water, elevation_mode="relative")
    try:
        # Depth mode: vertical depth below ground (TVD on the inclined hole).
        assert _rl_label_values(renderer) == pytest.approx(sorted([5.0, 10.0 * COS30]))
        suffixes = {
            annotation._water_suffix
            for kind, annotation, _c in renderer._water_labels
            if kind == "rl"
        }
        assert suffixes == {"mbgs"}
    finally:
        plt.close(figure)


def test_depth_mode_label_converts_btoc_to_below_ground() -> None:
    water = [WaterLevel(hole_id="B", depth=6.0, depth_reference="btoc")]
    _g, renderer, figure = _render(
        collars=_collars(stick_up_b=0.8, inclined_b=False),
        deviations=[],
        water=water,
        elevation_mode="relative",
    )
    try:
        assert _rl_label_values(renderer) == pytest.approx([5.2])
    finally:
        plt.close(figure)


# --- gradients --------------------------------------------------------------------


def test_gradients_use_corrected_heads() -> None:
    levels = [
        WaterLevel(hole_id="A", depth=5.0),
        WaterLevel(hole_id="B", depth=10.0),
    ]
    table = build_along_hole_tvd(_collars(), _deviations())["B"]
    (segment,) = horizontal_gradients_along_profile(
        levels,
        hole_order=("A", "B"),
        x_by_hole={"A": 0.0, "B": 60.0},
        collar_rl_by_hole={"A": 100.0, "B": 102.0},
        md_to_elevation_by_hole={"B": lambda md: table.elevation_at(102.0, md)},
    )
    assert segment.head_right_masl == pytest.approx(102.0 - 10.0 * COS30)
    assert segment.gradient == pytest.approx((102.0 - 10.0 * COS30 - 95.0) / 60.0)

    # btoc + stick-up feeds the gradient too.
    btoc = [
        WaterLevel(hole_id="A", depth=5.0),
        WaterLevel(hole_id="B", depth=10.0, depth_reference="btoc"),
    ]
    (segment,) = horizontal_gradients_along_profile(
        btoc,
        hole_order=("A", "B"),
        x_by_hole={"A": 0.0, "B": 60.0},
        collar_rl_by_hole={"A": 100.0, "B": 102.0},
        stick_up_by_hole={"B": 1.0},
    )
    assert segment.head_right_masl == pytest.approx(93.0)


def test_validate_gradient_qa_uses_survey_heads() -> None:
    collars = [
        Collar(hole_id="A", easting=0.0, northing=0.0, elevation=100.0, total_depth=60.0),
        Collar(hole_id="B", easting=20.0, northing=0.0, elevation=100.0, total_depth=60.0),
    ]
    deviations = [
        DeviationReading(hole_id="B", depth=depth, inclination_deg=60.0, azimuth_deg=0.0)
        for depth in (10.0, 60.0)
    ]
    levels = [WaterLevel(hole_id="A", depth=5.0), WaterLevel(hole_id="B", depth=10.0)]
    # Vertical: heads 95 vs 90 → |i| = 0.25 (warn). Survey: 100 − 10·cos60 = 95 → i = 0.
    vertical = summarize_water_levels(collars, levels, ("A", "B"))
    collars[1] = collars[1].model_copy(update={"inclination_deg": 60.0, "azimuth_deg": 0.0})
    # Collar angle alone already puts B on its survey path.
    collar_only = summarize_water_levels(collars, levels, ("A", "B"))
    assert not any("Horizontal hydraulic gradient" in w for w in collar_only.warnings)
    assert any("Horizontal hydraulic gradient" in w for w in vertical.warnings)
    surveyed = summarize_water_levels(collars, levels, ("A", "B"), deviation_readings=deviations)
    assert not any("Horizontal hydraulic gradient" in w for w in surveyed.warnings)


# --- vertical regression ----------------------------------------------------------


def test_vertical_holes_keep_legacy_arithmetic() -> None:
    level = WaterLevel(hole_id="A", depth=3.3)
    collar_rl = 101.7
    assert water_head_masl(level, collar_rl) == collar_rl - 3.3
    assert water_depth_below_ground_m(level, collar_rl) == 3.3
    masl = WaterLevel(hole_id="A", depth=collar_rl - 98.1, elevation_masl=98.1)
    assert water_head_masl(masl, collar_rl) == 98.1

    water = [WaterLevel(hole_id="A", depth=5.3), WaterLevel(hole_id="B", depth=7.1)]
    screens = [ScreenInterval(hole_id="B", from_depth=20.0, to_depth=30.0)]
    chemistry = [
        EnvironmentalReading(hole_id="A", parameter="Chloride", value=5.0, depth=12.7),
    ]
    geometry, renderer, figure = _render(
        collars=_collars(inclined_b=False),
        deviations=[],
        water=water,
        screens=screens,
        chemistry=chemistry,
    )
    try:
        assert renderer.along_hole_tvd == {}
        assert sorted(_marker_offsets(figure, "water")[:, 1].tolist()) == sorted(
            [100.0 - 5.3, 102.0 - 7.1]
        )
        assert _screen_y_extents(figure) == [(102.0 - 30.0, 102.0 - 20.0)]
        assert _marker_offsets(figure, "chem")[:, 1].tolist() == [100.0 - 12.7]
        b_rows = geometry.projected[geometry.projected["hole_id"] == "B"]
        assert float(b_rows["bottom_elevation"].min()) == 102.0 - 40.0
    finally:
        plt.close(figure)


def test_workbook_depth_reference_column_parses() -> None:
    from io import BytesIO

    import pandas as pd

    from parsing import DataParser

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                {
                    "hole_id": "MW-1",
                    "easting": 0,
                    "northing": 0,
                    "elevation": 50,
                    "total_depth": 20,
                    "stick_up_m": 0.9,
                },
            ]
        ).to_excel(writer, sheet_name="Collars", index=False)
        pd.DataFrame(
            [{"hole_id": "MW-1", "from_depth": 0, "to_depth": 20, "lithology_code": "SAND"}]
        ).to_excel(writer, sheet_name="Lithology", index=False)
        pd.DataFrame(
            [
                {"hole_id": "MW-1", "depth": 4.0, "depth_reference": "TOC", "series_id": "a"},
                {"hole_id": "MW-1", "depth": 3.0, "depth_reference": "", "series_id": "b"},
            ]
        ).to_excel(writer, sheet_name="Water", index=False)
    buffer.seek(0)
    result = DataParser().parse_file(buffer)
    refs = {level.series_id: level.depth_reference for level in result.water_levels}
    assert refs == {"a": "btoc", "b": "bgs"}
