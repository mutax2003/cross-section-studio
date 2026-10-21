"""Nested / co-located well vertical gradients and groundwater-vs-screen QA.

Convention: iv = (h_shallow − h_deep) / (z_mid_shallow − z_mid_deep) with z the
screen mid-point elevation — positive = downward flow (recharge), negative =
upward flow (discharge). User ``VerticalGradient`` arrows are cross-checked.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest

from ai_quality import summarize_water_levels
from hydro_metrics import (
    DEFAULT_NEST_DISTANCE_M,
    assign_screens_to_connect_groups,
    connect_group_depth_rank,
    format_vertical_gradient_note,
    nested_vertical_gradients,
    vertical_gradient_arrow_conflicts,
)
from models import Collar, Lithology, ScreenInterval, VerticalGradient, WaterLevel
from pipeline import build_cross_section

SHALLOW = ScreenInterval(hole_id="MW-1S", from_depth=5.0, to_depth=8.0)  # mid 93.5 masl
DEEP = ScreenInterval(hole_id="MW-1D", from_depth=20.0, to_depth=23.0)  # mid 78.5 masl


def _collars(deep_offset_m: float = 3.0) -> list[Collar]:
    return [
        Collar(hole_id="MW-1S", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="MW-1D", easting=deep_offset_m, northing=0.0, elevation=100.0, total_depth=25.0),
        Collar(hole_id="MW-2", easting=60.0, northing=0.0, elevation=101.0, total_depth=25.0),
    ]


def _maps(collars):
    return (
        {c.hole_id: float(c.elevation) for c in collars},
        {c.hole_id: (float(c.easting), float(c.northing)) for c in collars},
    )


def _levels(h_shallow: float, h_deep: float, **kwargs) -> list[WaterLevel]:
    return [
        WaterLevel(hole_id="MW-1S", depth=0.0, elevation_masl=h_shallow, connect_group="S", **kwargs),
        WaterLevel(hole_id="MW-1D", depth=0.0, elevation_masl=h_deep, connect_group="D", **kwargs),
    ]


def _analyse(levels, screens=(SHALLOW, DEEP), collars=None, **kwargs):
    collars = collars or _collars()
    rl, xy = _maps(collars)
    return nested_vertical_gradients(levels, screens, collar_rl_by_hole=rl, plan_xy_by_hole=xy, **kwargs)


def test_downward_gradient_is_positive_recharge() -> None:
    (gradient,) = _analyse(_levels(95.0, 94.7)).gradients
    assert gradient.shallow_hole_id == "MW-1S" and gradient.deep_hole_id == "MW-1D"
    assert gradient.z_mid_shallow_masl == pytest.approx(93.5)
    assert gradient.z_mid_deep_masl == pytest.approx(78.5)
    assert gradient.gradient == pytest.approx(0.3 / 15.0)
    assert gradient.direction == "down"
    assert "downward (recharge)" in format_vertical_gradient_note(gradient)
    assert "iv=+0.020" in format_vertical_gradient_note(gradient)


def test_upward_gradient_is_negative_discharge() -> None:
    (gradient,) = _analyse(_levels(94.7, 95.0)).gradients
    assert gradient.gradient == pytest.approx(-0.3 / 15.0)
    assert gradient.direction == "up"
    assert "upward (discharge)" in format_vertical_gradient_note(gradient)


def test_pairing_respects_plan_distance() -> None:
    assert len(_analyse(_levels(95.0, 94.7), collars=_collars(4.9)).gradients) == 1
    assert _analyse(_levels(95.0, 94.7), collars=_collars(6.0)).gradients == ()
    assert DEFAULT_NEST_DISTANCE_M == 5.0
    widened = _analyse(_levels(95.0, 94.7), collars=_collars(6.0), nest_distance_m=10.0)
    assert len(widened.gradients) == 1
    assert widened.gradients[0].plan_distance_m == pytest.approx(6.0)


def test_overlapping_screens_are_skipped() -> None:
    overlapping = ScreenInterval(hole_id="MW-1D", from_depth=7.0, to_depth=12.0)
    assert _analyse(_levels(95.0, 94.7), screens=(SHALLOW, overlapping)).gradients == ()


def test_both_wells_need_a_screen() -> None:
    assert _analyse(_levels(95.0, 94.7), screens=(SHALLOW,)).gradients == ()


def test_different_series_do_not_pair() -> None:
    levels = [
        WaterLevel(hole_id="MW-1S", depth=0.0, elevation_masl=95.0, series_id="2024-06"),
        WaterLevel(hole_id="MW-1D", depth=0.0, elevation_masl=94.7, series_id="2024-09"),
    ]
    assert _analyse(levels).gradients == ()


def test_same_hole_multiple_screens_by_connect_group() -> None:
    collars = [Collar(hole_id="MW-5", easting=0.0, northing=0.0, elevation=100.0, total_depth=30.0)]
    screens = (
        ScreenInterval(hole_id="MW-5", from_depth=20.0, to_depth=22.0),
        ScreenInterval(hole_id="MW-5", from_depth=4.0, to_depth=6.0),
    )
    levels = [
        WaterLevel(hole_id="MW-5", depth=3.0, connect_group="shallow"),
        WaterLevel(hole_id="MW-5", depth=3.5, connect_group="deep"),
    ]
    (gradient,) = _analyse(levels, screens=screens, collars=collars).gradients
    assert gradient.shallow_connect_group == "shallow"
    assert gradient.z_mid_shallow_masl == pytest.approx(95.0)
    assert gradient.z_mid_deep_masl == pytest.approx(79.0)
    assert gradient.gradient == pytest.approx(0.5 / 16.0)
    assert gradient.nest_label == "MW-5 [shallow]/MW-5 [deep]"


def test_same_hole_ambiguous_groups_are_noted_not_guessed() -> None:
    collars = [Collar(hole_id="MW-5", easting=0.0, northing=0.0, elevation=100.0, total_depth=30.0)]
    screens = (
        ScreenInterval(hole_id="MW-5", from_depth=4.0, to_depth=6.0),
        ScreenInterval(hole_id="MW-5", from_depth=20.0, to_depth=22.0),
    )
    levels = [
        WaterLevel(hole_id="MW-5", depth=3.0, connect_group="A"),
        WaterLevel(hole_id="MW-5", depth=3.5, connect_group="B"),
    ]
    analysis = _analyse(levels, screens=screens, collars=collars)
    assert analysis.gradients == ()
    assert any("cannot tell which screen" in note for note in analysis.notes)


def test_connect_group_depth_rank_names() -> None:
    assert connect_group_depth_rank("S") < connect_group_depth_rank("I") < connect_group_depth_rank("D")
    assert connect_group_depth_rank("MW-1 shallow") == 0.0
    assert connect_group_depth_rank("1") < connect_group_depth_rank("2")
    assert connect_group_depth_rank("A") is None
    mapping, note = assign_screens_to_connect_groups("H", [SHALLOW], ["x", "y"])
    assert note is None and set(mapping) == {"x", "y"}


def test_inclined_hole_uses_md_to_elevation_for_screen_midpoint() -> None:
    # Deep well inclined: TVD = 0.5 × MD, so its screen mid sits at 100 − 10.75.
    rl, xy = _maps(_collars())
    analysis = nested_vertical_gradients(
        _levels(95.0, 94.7),
        (SHALLOW, DEEP),
        collar_rl_by_hole=rl,
        plan_xy_by_hole=xy,
        md_to_elevation_by_hole={"MW-1D": lambda md: 100.0 - 0.5 * md},
    )
    (gradient,) = analysis.gradients
    assert gradient.z_mid_deep_masl == pytest.approx(89.25)
    assert gradient.gradient == pytest.approx(0.3 / (93.5 - 89.25))


def test_arrow_contradiction_warns_only_beyond_tolerance() -> None:
    gradients = _analyse(_levels(95.0, 94.7)).gradients  # downward
    up_arrow = [VerticalGradient(hole_id="MW-1S", direction="up")]
    warnings = vertical_gradient_arrow_conflicts(up_arrow, gradients)
    assert len(warnings) == 1 and "arrow points up" in warnings[0] and "downward flow" in warnings[0]
    assert vertical_gradient_arrow_conflicts([VerticalGradient(hole_id="MW-1D", direction="down")], gradients) == []
    small = _analyse(_levels(95.0, 94.97)).gradients
    assert small[0].direction == "none"
    assert vertical_gradient_arrow_conflicts(up_arrow, small) == []
    # An arrow on a hole with no computed gradient is left alone.
    assert vertical_gradient_arrow_conflicts([VerticalGradient(hole_id="MW-2", direction="up")], gradients) == []


# --- Validate summary (ai_quality) -------------------------------------------------


def _summary(levels, screens=(SHALLOW, DEEP), **kwargs):
    return summarize_water_levels(
        _collars(), levels, ("MW-1S", "MW-1D", "MW-2"), screen_intervals=screens, **kwargs
    )


def test_summary_reports_nested_gradient_and_arrow_conflict() -> None:
    summary = _summary(
        _levels(95.0, 94.7),
        vertical_gradients=[VerticalGradient(hole_id="MW-1D", direction="up")],
    )
    assert len(summary.vertical_gradients) == 1
    assert any(note.startswith("Vertical gradient MW-1S [S]/MW-1D [D]") for note in summary.notes)
    assert any(w.startswith("Info: Vertical gradient") for w in summary.warnings)
    assert any("arrow points up" in w for w in summary.warnings)


def test_summary_without_screens_is_unchanged() -> None:
    summary = summarize_water_levels(_collars(), _levels(95.0, 94.7), ("MW-1S", "MW-1D", "MW-2"))
    assert summary.vertical_gradients == ()
    assert not any("Vertical gradient" in w for w in summary.warnings)


def test_flowing_artesian_is_info_not_error() -> None:
    levels = [WaterLevel(hole_id="MW-2", depth=0.0, elevation_masl=101.4)]
    summary = summarize_water_levels(_collars(), levels, ("MW-1S", "MW-1D", "MW-2"))
    assert "flowing artesian at MW-2 (head +0.40 m above ground) — plotted above the ground line" in summary.notes
    assert any(w.startswith("Info: flowing artesian at MW-2") for w in summary.warnings)


def test_artesian_within_stick_up_is_not_flowing() -> None:
    collars = [
        Collar(hole_id="MW-2", easting=0.0, northing=0.0, elevation=101.0, total_depth=25.0, stick_up_m=0.9)
    ]
    levels = [WaterLevel(hole_id="MW-2", depth=0.5, depth_reference="btoc")]
    summary = summarize_water_levels(collars, levels, ("MW-2",))
    assert any(note.startswith("artesian at MW-2 (head +0.40 m") for note in summary.notes)
    assert not any("flowing" in note for note in summary.notes)


def test_reading_below_screen_warns() -> None:
    levels = [WaterLevel(hole_id="MW-1S", depth=9.0)]  # screen 5–8 m
    summary = _summary(levels)
    assert any("reading below screen — well may be dry or reading unreliable" in w for w in summary.warnings)
    inside = _summary([WaterLevel(hole_id="MW-1S", depth=6.0)])
    assert not any("reading below screen" in w for w in inside.warnings)


def test_dry_upper_screen_is_info() -> None:
    collars = [Collar(hole_id="MW-5", easting=0.0, northing=0.0, elevation=100.0, total_depth=30.0)]
    screens = (
        ScreenInterval(hole_id="MW-5", from_depth=2.0, to_depth=4.0),
        ScreenInterval(hole_id="MW-5", from_depth=20.0, to_depth=22.0),
    )
    summary = summarize_water_levels(
        collars, [WaterLevel(hole_id="MW-5", depth=10.0)], ("MW-5",), screen_intervals=screens
    )
    assert any("2.00–4.00 m is entirely above the water level" in note for note in summary.notes)
    assert not any("reading below screen" in w for w in summary.warnings)


def _two_well_line(lith_b: str, *, unit_a=None, unit_b=None, groups=("", "")):
    collars = [
        Collar(hole_id="A", easting=0.0, northing=0.0, elevation=100.0, total_depth=30.0),
        Collar(hole_id="B", easting=80.0, northing=0.0, elevation=100.0, total_depth=30.0),
    ]
    lithologies = [
        Lithology(hole_id="A", from_depth=0.0, to_depth=30.0, lithology_code="SAND", unit_order=unit_a),
        Lithology(hole_id="B", from_depth=0.0, to_depth=30.0, lithology_code=lith_b, unit_order=unit_b),
    ]
    screens = [
        ScreenInterval(hole_id="A", from_depth=10.0, to_depth=12.0),
        ScreenInterval(hole_id="B", from_depth=10.0, to_depth=12.0),
    ]
    levels = [
        WaterLevel(hole_id="A", depth=5.0, connect_group=groups[0]),
        WaterLevel(hole_id="B", depth=5.5, connect_group=groups[1]),
    ]
    return summarize_water_levels(
        collars, levels, ("A", "B"), screen_intervals=screens, lithologies=lithologies
    )


def test_water_line_across_different_units_warns() -> None:
    summary = _two_well_line("GRAVEL")
    assert any(
        "water line joins wells screened in different units (SAND vs GRAVEL)" in w
        and "set connect_group to separate them" in w
        for w in summary.warnings
    )


def test_water_line_same_unit_or_separate_groups_is_quiet() -> None:
    assert not any("different units" in w for w in _two_well_line("SAND").warnings)
    assert not any("different units" in w for w in _two_well_line("GRAVEL", groups=("x", "y")).warnings)


def test_unit_order_distinguishes_same_lithology() -> None:
    summary = _two_well_line("SAND", unit_a=2, unit_b=4)
    assert any("SAND (unit 2) vs SAND (unit 4)" in w for w in summary.warnings)


def test_mixed_unit_check_needs_logged_units() -> None:
    summary = summarize_water_levels(
        [
            Collar(hole_id="A", easting=0.0, northing=0.0, elevation=100.0, total_depth=30.0),
            Collar(hole_id="B", easting=80.0, northing=0.0, elevation=100.0, total_depth=30.0),
        ],
        [WaterLevel(hole_id="A", depth=5.0), WaterLevel(hole_id="B", depth=5.5)],
        ("A", "B"),
        screen_intervals=[ScreenInterval(hole_id="A", from_depth=10.0, to_depth=12.0)],
        lithologies=[Lithology(hole_id="A", from_depth=0.0, to_depth=30.0, lithology_code="SAND")],
    )
    assert not any("different units" in w for w in summary.warnings)


# --- Figure QA note (pipeline) -----------------------------------------------------


def test_pipeline_qa_notes_list_vertical_gradient_and_arrow_conflict() -> None:
    collars = _collars()
    lithologies = [
        Lithology(hole_id=c.hole_id, from_depth=0.0, to_depth=c.total_depth, lithology_code="SAND")
        for c in collars
    ]
    result = build_cross_section(
        collars,
        lithologies,
        [(-10.0, 0.0), (70.0, 0.0)],
        water_levels=_levels(95.0, 94.7),
        screen_intervals=[SHALLOW, DEEP],
        vertical_gradients=[VerticalGradient(hole_id="MW-1S", direction="up")],
        export_formats=frozenset({"svg"}),
    )
    assert any(
        note.startswith("Vertical gradient MW-1S [S]/MW-1D [D]: iv=+0.020 downward")
        for note in result.qa_notes
    )
    assert any("arrow points up" in note for note in result.qa_notes)
    assert any("Vertical gradient" in w for w in result.overlap_warnings)
