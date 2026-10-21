"""LNAPL (free product) corrected groundwater elevations.

Convention: ``WaterLevel.depth`` (or ``elevation_masl``) is the measured depth
to WATER — the oil/water interface — and ``product_thickness_m`` the apparent
product thickness above it. Corrected head = interface elevation + vertical
thickness × SG (``product_density``, default 0.75). ``water_head_masl`` applies
it, so labels, gradients, the plan-view flow fit, nested vertical gradients and
Validate all use corrected heads. Wells without product are unchanged.
"""

from __future__ import annotations

import io
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ai_quality import summarize_water_levels
from hydro_metrics import (
    fit_head_planes,
    horizontal_gradients_along_profile,
    lnapl_correction_note,
    lnapl_interval_masl,
    nested_vertical_gradients,
    water_depth_below_ground_m,
    water_head_masl,
)
from models import (
    DEFAULT_LNAPL_SPECIFIC_GRAVITY,
    Collar,
    DeviationReading,
    Lithology,
    ParseResult,
    ScreenInterval,
    WaterLevel,
)
from parsing import DataParser
from pipeline import compute_section_geometry, render_cross_section_from_geometry
from projection import build_along_hole_tvd
from render_theme import LNAPL_FILL_COLOR, LNAPL_HATCH, LNAPL_LEGEND_LABEL
from workbook_template import WATER_COLUMNS, export_cleaned_workbook_bytes

COS30 = math.cos(math.radians(30.0))
POINTS = [(-20.0, 0.0), (220.0, 0.0)]


# ---------------------------------------------------------------- model / math


def test_density_defaults_when_thickness_given() -> None:
    level = WaterLevel(hole_id="MW-3", depth=5.0, product_thickness_m=0.32)
    assert level.product_density == DEFAULT_LNAPL_SPECIFIC_GRAVITY == 0.75
    plain = WaterLevel(hole_id="MW-1", depth=5.0)
    assert plain.product_thickness_m is None and plain.product_density is None


@pytest.mark.parametrize("density", [1.0, 1.2])
def test_dense_product_rejected_with_dnapl_hint(density: float) -> None:
    with pytest.raises(ValidationError, match="DNAPL"):
        WaterLevel(hole_id="MW-3", depth=5.0, product_thickness_m=0.3, product_density=density)


@pytest.mark.parametrize(
    ("field", "value"),
    [("product_density", 0.3), ("product_thickness_m", -0.1), ("product_thickness_m", math.inf)],
)
def test_out_of_range_product_values_rejected(field: str, value: float) -> None:
    payload = {"hole_id": "MW-3", "depth": 5.0, "product_thickness_m": 0.3, field: value}
    with pytest.raises(ValidationError):
        WaterLevel(**payload)


def test_bgs_correction_vertical_hole() -> None:
    level = WaterLevel(hole_id="MW-3", depth=5.0, product_thickness_m=0.32, product_density=0.8)
    assert water_head_masl(level, 100.0) == pytest.approx(95.0 + 0.32 * 0.8)
    assert water_head_masl(level, 100.0, lnapl_corrected=False) == pytest.approx(95.0)
    assert water_depth_below_ground_m(level, 100.0) == pytest.approx(5.0 - 0.32 * 0.8)
    assert lnapl_interval_masl(level, 100.0) == pytest.approx((95.32, 95.0))


def test_btoc_correction_uses_stick_up() -> None:
    level = WaterLevel(
        hole_id="MW-3", depth=5.8, depth_reference="btoc", product_thickness_m=0.4
    )
    head = water_head_masl(level, 100.0, stick_up_m=0.8)
    assert head == pytest.approx(100.0 + 0.8 - 5.8 + 0.4 * 0.75)
    assert water_depth_below_ground_m(level, 100.0, stick_up_m=0.8) == pytest.approx(
        5.0 - 0.3
    )


def test_explicit_elevation_is_measured_interface_and_still_corrected() -> None:
    level = WaterLevel(hole_id="MW-3", depth=0.0, elevation_masl=96.0, product_thickness_m=0.2)
    assert water_head_masl(level, 100.0) == pytest.approx(96.15)
    assert lnapl_interval_masl(level, 100.0) == pytest.approx((96.2, 96.0))


def test_inclined_hole_uses_vertical_product_thickness() -> None:
    collar = Collar(
        hole_id="B",
        easting=0.0,
        northing=0.0,
        elevation=100.0,
        total_depth=40.0,
        inclination_deg=30.0,
        azimuth_deg=0.0,
    )
    readings = [
        DeviationReading(hole_id="B", depth=d, inclination_deg=30.0, azimuth_deg=0.0)
        for d in (10.0, 20.0, 40.0)
    ]
    table = build_along_hole_tvd([collar], readings, {"B"})["B"]

    def md_fn(md: float) -> float:
        return table.elevation_at(100.0, md)

    level = WaterLevel(hole_id="B", depth=10.0, product_thickness_m=1.0, product_density=0.8)
    interface = 100.0 - 10.0 * COS30
    head = water_head_masl(level, 100.0, md_to_elevation=md_fn)
    assert head == pytest.approx(interface + 1.0 * COS30 * 0.8, abs=1e-6)
    top, bottom = lnapl_interval_masl(level, 100.0, md_to_elevation=md_fn)
    assert bottom == pytest.approx(interface, abs=1e-6)
    assert top == pytest.approx(100.0 - 9.0 * COS30, abs=1e-6)


def test_zero_thickness_is_no_product() -> None:
    level = WaterLevel(hole_id="MW-3", depth=5.0, product_thickness_m=0.0)
    assert water_head_masl(level, 100.0) == 95.0
    assert lnapl_interval_masl(level, 100.0) is None
    assert lnapl_correction_note([level]) is None


def test_no_product_heads_unchanged() -> None:
    level = WaterLevel(hole_id="MW-1", depth=4.37)
    assert water_head_masl(level, 101.23) == 101.23 - 4.37
    assert water_depth_below_ground_m(level, 101.23) == 4.37


def test_correction_note_text() -> None:
    levels = [
        WaterLevel(hole_id="MW-3", depth=5.0, product_thickness_m=0.32),
        WaterLevel(hole_id="MW-3", depth=5.0, product_thickness_m=0.32, series_id="b"),
        WaterLevel(hole_id="MW-4", depth=5.0, product_thickness_m=0.5, status="dry"),
    ]
    assert lnapl_correction_note(levels) == (
        "Groundwater elevations corrected for LNAPL at MW-3 (0.32 m, SG 0.75)"
    )


# --------------------------------------------------------------------- parsing


def _parse_water(rows: list[dict[str, object]]):
    collars = [
        Collar(hole_id="MW-3", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0),
        Collar(hole_id="MW-4", easting=50.0, northing=0.0, elevation=100.0, total_depth=20.0),
    ]
    return DataParser()._parse_water_levels(pd.DataFrame(rows), collars)


def test_parse_product_columns() -> None:
    levels, errors = _parse_water(
        [
            {"hole_id": "MW-3", "depth": 5.0, "product_thickness_m": 0.32, "product_density": 0.8},
            {"hole_id": "MW-4", "depth": 6.0, "product_thickness_m": None, "product_density": None},
        ]
    )
    assert errors == []
    assert levels[0].product_thickness_m == pytest.approx(0.32)
    assert levels[0].product_density == pytest.approx(0.8)
    assert levels[1].product_thickness_m is None and levels[1].product_density is None


def test_parse_depth_to_product_alternative() -> None:
    levels, errors = _parse_water(
        [{"hole_id": "MW-3", "depth": 5.0, "depth_to_product": 4.68}]
    )
    assert errors == []
    assert levels[0].product_thickness_m == pytest.approx(0.32)
    assert levels[0].product_density == 0.75


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"hole_id": "MW-3", "depth": 5.0, "depth_to_product": 5.5}, "below the water depth"),
        ({"hole_id": "MW-3", "elevation_masl": 95.0, "depth_to_product": 4.5}, "elevation_masl"),
        (
            {"hole_id": "MW-3", "depth": 5.0, "depth_to_product": 4.5, "product_thickness_m": 0.2},
            "disagrees",
        ),
        ({"hole_id": "MW-3", "depth": 5.0, "product_thickness_m": 0.2, "product_density": 1.05}, "DNAPL"),
    ],
)
def test_parse_product_errors(row: dict[str, object], message: str) -> None:
    levels, errors = _parse_water([row])
    assert levels == []
    assert len(errors) == 1 and message in errors[0]


def test_water_columns_append_product_fields_at_end() -> None:
    assert WATER_COLUMNS[-3:] == ("product_thickness_m", "product_density", "depth_to_product")
    assert WATER_COLUMNS.index("depth_reference") < WATER_COLUMNS.index("product_thickness_m")


# ------------------------------------------------- gradients / flow fit / nests


def test_horizontal_gradient_uses_corrected_heads() -> None:
    levels = [
        WaterLevel(hole_id="A", depth=5.0),
        WaterLevel(hole_id="B", depth=5.0, product_thickness_m=0.4),
    ]
    (segment,) = horizontal_gradients_along_profile(
        levels,
        hole_order=["A", "B"],
        x_by_hole={"A": 0.0, "B": 100.0},
        collar_rl_by_hole={"A": 100.0, "B": 100.0},
    )
    assert segment.head_right_masl == pytest.approx(95.3)
    assert segment.gradient == pytest.approx(0.003)


def test_flow_plane_fit_uses_corrected_heads() -> None:
    collars = [
        Collar(hole_id=hole, easting=e, northing=n, elevation=100.0, total_depth=20.0)
        for hole, e, n in (("A", 0.0, 0.0), ("B", 100.0, 0.0), ("C", 0.0, 100.0))
    ]
    levels = [
        WaterLevel(hole_id="A", depth=5.0),
        WaterLevel(hole_id="B", depth=5.0, product_thickness_m=0.4),
        WaterLevel(hole_id="C", depth=5.0),
    ]
    (fit,) = fit_head_planes(collars, levels)
    # Head 0.3 m higher at B (+E): dh/dE = 0.003, flow toward west.
    assert fit.dh_de == pytest.approx(0.003, abs=1e-9)
    assert fit.dh_dn == pytest.approx(0.0, abs=1e-9)
    assert fit.flow_azimuth_deg == pytest.approx(270.0)


def test_nested_vertical_gradient_uses_corrected_heads() -> None:
    levels = [
        WaterLevel(hole_id="S", depth=5.0, product_thickness_m=0.4),
        WaterLevel(hole_id="D", depth=5.0),
    ]
    screens = [
        ScreenInterval(hole_id="S", from_depth=4.0, to_depth=8.0),
        ScreenInterval(hole_id="D", from_depth=14.0, to_depth=18.0),
    ]
    analysis = nested_vertical_gradients(
        levels,
        screens,
        collar_rl_by_hole={"S": 100.0, "D": 100.0},
        plan_xy_by_hole={"S": (0.0, 0.0), "D": (1.0, 0.0)},
    )
    (gradient,) = analysis.gradients
    # Shallow head 95.3 vs deep 95.0 → downward (positive) gradient.
    assert gradient.head_shallow_masl == pytest.approx(95.3)
    assert gradient.gradient > 0


# --------------------------------------------------------------------- Validate


def _validate(levels):
    collars = [
        Collar(hole_id="MW-1", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0),
        Collar(hole_id="MW-3", easting=50.0, northing=0.0, elevation=100.0, total_depth=20.0),
    ]
    return summarize_water_levels(collars, levels, ["MW-1", "MW-3"])


def test_validate_lists_corrected_wells_and_warns_on_thick_product() -> None:
    summary = _validate(
        [
            WaterLevel(hole_id="MW-1", depth=5.0),
            WaterLevel(hole_id="MW-3", depth=5.0, product_thickness_m=1.4),
        ]
    )
    assert any("corrected for LNAPL at MW-3 (1.40 m, SG 0.75)" in note for note in summary.notes)
    assert any("apparent thickness in wells overstates formation thickness" in w for w in summary.warnings)


def test_validate_warns_on_product_with_dry_or_nm_status() -> None:
    summary = _validate(
        [
            WaterLevel(hole_id="MW-1", depth=5.0),
            WaterLevel(hole_id="MW-3", depth=5.0, product_thickness_m=0.2, status="dry"),
        ]
    )
    assert any("MW-3" in w and "dry" in w and "LNAPL" in w for w in summary.warnings)


def test_validate_no_product_no_lnapl_messages() -> None:
    summary = _validate(
        [WaterLevel(hole_id="MW-1", depth=5.0), WaterLevel(hole_id="MW-3", depth=5.5)]
    )
    assert not any("LNAPL" in text for text in (*summary.warnings, *summary.notes))


# ---------------------------------------------------------------------- figure


def _collars() -> list[Collar]:
    return [
        Collar(hole_id="MW-1", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0),
        Collar(hole_id="MW-3", easting=100.0, northing=0.0, elevation=100.5, total_depth=20.0),
        Collar(hole_id="MW-4", easting=200.0, northing=0.0, elevation=101.0, total_depth=20.0),
    ]


def _lithologies() -> list[Lithology]:
    return [
        Lithology(hole_id=hole, from_depth=0.0, to_depth=20.0, lithology_code="SAND")
        for hole in ("MW-1", "MW-3", "MW-4")
    ]


def _water(*, product: bool) -> list[WaterLevel]:
    return [
        WaterLevel(hole_id="MW-1", depth=4.0),
        WaterLevel(
            hole_id="MW-3",
            depth=5.0,
            product_thickness_m=0.32 if product else None,
        ),
        WaterLevel(hole_id="MW-4", depth=5.5),
    ]


def _render(*, product: bool, layout: str = "section_sheet", elevation_mode: str = "absolute"):
    geometry = compute_section_geometry(
        _collars(), _lithologies(), POINTS, interpretation_mode="borehole_only"
    )
    return render_cross_section_from_geometry(
        geometry,
        POINTS,
        vertical_exaggeration=None,
        interpretation_mode="borehole_only",
        water_levels=_water(product=product),
        show_water_elevation_labels=True,
        render_layout=layout,
        elevation_mode=elevation_mode,
        export_formats=frozenset({"svg"}),
        close_figure=False,
    )


def _lnapl_collections(figure):
    return [
        collection
        for ax in figure.axes
        for collection in ax.collections
        if collection.get_hatch() == LNAPL_HATCH
    ]


def _all_texts(figure) -> list[str]:
    texts = [text.get_text() for text in figure.texts]
    for ax in figure.axes:
        texts.extend(text.get_text() for text in ax.texts)
        for legend in [ax.get_legend(), *[a for a in ax.artists if hasattr(a, "get_texts")]]:
            if legend is not None:
                texts.extend(t.get_text() for t in legend.get_texts())
    for legend in figure.legends:
        texts.extend(t.get_text() for t in legend.get_texts())
    return texts


@pytest.mark.parametrize("elevation_mode", ["absolute", "relative"])
def test_section_sheet_draws_product_band_label_and_note(elevation_mode: str) -> None:
    result = _render(product=True, elevation_mode=elevation_mode)
    bundle = result.retained_export_bundle
    figure, renderer = bundle["figure"], bundle["renderer"]
    try:
        bands = _lnapl_collections(figure)
        assert len(bands) == 1
        assert len(bands[0].get_paths()) == 1
        face = bands[0].get_facecolor()[0]
        assert matplotlib.colors.to_hex(face) == LNAPL_FILL_COLOR.lower()
        assert renderer.lnapl_drawn
        texts = _all_texts(figure)
        assert any("LNAPL 0.32 m" in text for text in texts)
        assert "Groundwater elevations corrected for LNAPL at MW-3 (0.32 m, SG 0.75)" in (
            result.qa_notes
        )
        corrected = 100.5 - 5.0 + 0.32 * 0.75
        if elevation_mode == "absolute":
            assert any(text.startswith(f"{corrected:.2f}") for text in texts)
        else:
            assert any(text.startswith(f"{100.5 - corrected:.2f}") for text in texts)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("layout", ["consulting_section", "chart"])
def test_other_layouts_draw_band_and_legend_entry(layout: str) -> None:
    result = _render(product=True, layout=layout)
    figure = result.retained_export_bundle["figure"]
    try:
        assert len(_lnapl_collections(figure)) >= 1
        svg = result.svg_bytes.decode("utf-8")
        label = LNAPL_LEGEND_LABEL.upper() if layout == "consulting_section" else LNAPL_LEGEND_LABEL
        assert label in svg
    finally:
        plt.close(figure)


@pytest.mark.parametrize("layout", ["section_sheet", "consulting_section", "chart"])
def test_no_product_no_band_legend_or_note(layout: str) -> None:
    result = _render(product=False, layout=layout)
    figure = result.retained_export_bundle["figure"]
    try:
        assert _lnapl_collections(figure) == []
        svg = result.svg_bytes.decode("utf-8")
        assert "LNAPL" not in svg
        assert not any("LNAPL" in note for note in result.qa_notes)
    finally:
        plt.close(figure)


def test_zero_thickness_figure_matches_no_product_figure() -> None:
    """A recorded 0 m thickness draws exactly what a blank cell draws."""

    def svg(thickness):
        geometry = compute_section_geometry(
            _collars(), _lithologies(), POINTS, interpretation_mode="borehole_only"
        )
        water = [
            WaterLevel(hole_id="MW-1", depth=4.0),
            WaterLevel(hole_id="MW-3", depth=5.0, product_thickness_m=thickness),
            WaterLevel(hole_id="MW-4", depth=5.5),
        ]
        return render_cross_section_from_geometry(
            geometry,
            POINTS,
            vertical_exaggeration=None,
            interpretation_mode="borehole_only",
            water_levels=water,
            show_water_elevation_labels=True,
            export_formats=frozenset({"svg"}),
        ).svg_bytes

    assert svg(0.0) == svg(None)


# ------------------------------------------------------------ workbook round-trip


def test_workbook_round_trip_keeps_product_fields() -> None:
    parse_result = ParseResult(
        collars=tuple(_collars()),
        lithologies=tuple(_lithologies()),
        errors=(),
        water_levels=(
            WaterLevel(hole_id="MW-1", depth=4.0),
            WaterLevel(hole_id="MW-3", depth=5.0, product_thickness_m=0.32, product_density=0.82),
        ),
    )
    payload = export_cleaned_workbook_bytes(parse_result)
    reparsed = DataParser().parse_file(io.BytesIO(payload))
    by_hole = {level.hole_id: level for level in reparsed.water_levels}
    assert by_hole["MW-3"].product_thickness_m == pytest.approx(0.32)
    assert by_hole["MW-3"].product_density == pytest.approx(0.82)
    assert by_hole["MW-1"].product_thickness_m is None
    assert by_hole["MW-1"].product_density is None


def test_workbook_round_trip_without_product_has_no_product_columns() -> None:
    parse_result = ParseResult(
        collars=tuple(_collars()),
        lithologies=tuple(_lithologies()),
        errors=(),
        water_levels=(WaterLevel(hole_id="MW-1", depth=4.0),),
    )
    payload = export_cleaned_workbook_bytes(parse_result)
    frame = pd.read_excel(io.BytesIO(payload), sheet_name="Water")
    assert "product_thickness_m" not in frame.columns
    assert np.isclose(float(frame.loc[0, "depth"]), 4.0)
