"""Wave B: chemistry + GW combined, connect_group, stick-up, series filter."""

from __future__ import annotations

from models import Collar, EnvironmentalReading, Lithology, WaterLevel
from pipeline import build_cross_section
from render_theme import (
    CONSULTING_GW_BLUE_SHADES,
    consulting_gw_series_style,
    filter_water_levels_for_plot,
)
from ui_output_presets import FIGURE_PRESET_IDS, normalize_figure_preset, resolve_output_preset
from tests.conftest import assert_valid_svg


def test_chemistry_gw_preset_enables_both() -> None:
    config = resolve_output_preset("chemistry_gw")
    assert config.prefer_chemistry is True
    assert config.interpolate_water_table is True
    assert config.show_water_legend is True
    assert config.water_line_solid is False
    assert config.sample_figure_profile is True
    assert "chemistry_gw" in FIGURE_PRESET_IDS
    assert normalize_figure_preset("p2_gw") == "chemistry_gw"


def test_filter_water_levels_caps_at_four() -> None:
    levels = [
        WaterLevel(hole_id="MW-01", depth=1.0, series_id=f"s{i}")
        for i in range(6)
    ]
    filtered = filter_water_levels_for_plot(levels, max_series=4)
    series = {level.series_id for level in filtered}
    assert series == {"s0", "s1", "s2", "s3"}


def test_filter_water_levels_respects_selection() -> None:
    levels = [
        WaterLevel(hole_id="MW-01", depth=1.0, series_id="shallow"),
        WaterLevel(hole_id="MW-01", depth=2.0, series_id="deep"),
        WaterLevel(hole_id="MW-02", depth=1.5, series_id="shallow"),
    ]
    filtered = filter_water_levels_for_plot(levels, ("shallow",))
    assert all(level.series_id == "shallow" for level in filtered)
    assert len(filtered) == 2


def test_consulting_gw_unknown_series_uses_blue_triangle() -> None:
    color, marker, label = consulting_gw_series_style("event-a", series_index=1)
    assert marker == "v"
    assert color == CONSULTING_GW_BLUE_SHADES[1]
    assert label == "event-a"


def test_connect_group_keeps_nests_separate_in_svg() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0, stick_up_m=0.8),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=20.0, stick_up_m=0.8),
        Collar(hole_id="BH-03", easting=100.0, northing=0.0, elevation=100.0, total_depth=20.0),
    ]
    lithologies = [
        Lithology(hole_id=hole, from_depth=0.0, to_depth=20.0, lithology_code="Clay")
        for hole in ("BH-01", "BH-02", "BH-03")
    ]
    water = [
        WaterLevel(hole_id="BH-01", depth=3.0, series_id="2025-06", connect_group="shallow"),
        WaterLevel(hole_id="BH-02", depth=3.5, series_id="2025-06", connect_group="shallow"),
        WaterLevel(hole_id="BH-01", depth=12.0, series_id="2025-06", connect_group="deep"),
        WaterLevel(hole_id="BH-03", depth=11.0, series_id="2025-06", connect_group="deep"),
    ]
    readings = [
        EnvironmentalReading(hole_id="BH-01", parameter="Chloride", value=120.0, depth=4.0, unit="mg/L"),
        EnvironmentalReading(hole_id="BH-02", parameter="Chloride", value=85.0, depth=4.0, unit="mg/L"),
    ]
    result = build_cross_section(
        collars,
        lithologies,
        [(0.0, 0.0), (100.0, 0.0)],
        water_levels=water,
        environmental_readings=readings,
        environmental_parameters=("Chloride",),
        render_layout="consulting_section",
        interpretation_mode="borehole_only",
        interpolate_water_table=True,
        show_water_legend=True,
        water_line_solid=False,
        show_parameter_labels=True,
    )
    assert_valid_svg(result.svg_bytes)
    text = result.svg_bytes.decode("utf-8", errors="ignore")
    assert "120" in text
    assert "85" in text


def test_stick_up_is_carried_on_collar_parse() -> None:
    collar = Collar(
        hole_id="MW-01",
        easting=0.0,
        northing=0.0,
        elevation=100.0,
        total_depth=10.0,
        stick_up_m=0.75,
    )
    assert collar.stick_up_m == 0.75
