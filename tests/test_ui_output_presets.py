"""Tests for output preset mapping."""

from __future__ import annotations

from app_build import effective_render_options
from ui_output_presets import (
    OUTPUT_PRESETS,
    FIGURE_PRESET_IDS,
    normalize_figure_preset,
    resolve_output_preset,
)


def test_consulting_report_preset() -> None:
    config = resolve_output_preset("consulting_report")
    assert config.render_layout == "consulting_section"
    assert config.interpolate_water_table is True
    assert config.show_legend is False
    assert config.sample_figure_profile is False


def test_unknown_preset_falls_back_to_section_sheet() -> None:
    config = resolve_output_preset("not_a_preset")
    assert config == OUTPUT_PRESETS["section_sheet"]


def test_gwm_fence_preset_matches_sample_defaults() -> None:
    config = resolve_output_preset("gwm_fence")
    assert config.render_layout == "consulting_section"
    assert config.interpretation_mode == "interpolated"
    assert config.elevation_mode == "absolute"
    assert config.vertical_exaggeration == 5.0
    assert config.interpolate_water_table is True
    assert config.show_water_elevation_labels is True
    assert config.show_dry_well_nm is True
    assert config.prefer_chemistry is False
    assert config.sample_figure_profile is True
    assert "gwm_fence" in FIGURE_PRESET_IDS


def test_p2_chemistry_sticks_preset_matches_sample_defaults() -> None:
    config = resolve_output_preset("p2_chemistry_sticks")
    assert config.render_layout == "consulting_section"
    assert config.interpretation_mode == "borehole_only"
    assert config.elevation_mode == "relative"
    assert config.vertical_exaggeration == 1.0
    assert config.interpolate_water_table is False
    assert config.show_water_legend is False
    assert config.prefer_chemistry is True
    assert config.parameter_interpolate_segments is False
    assert config.parameter_draw_markers is False
    assert config.sample_figure_profile is True


def test_p2_effective_render_keeps_water_off() -> None:
    preset = resolve_output_preset("p2_chemistry_sticks")
    effective = effective_render_options(
        report_preset=False,
        render_layout=preset.render_layout,
        show_ground_surface=True,
        track_width_m=3.0,
        show_legend=False,
        interpolate_water_table=preset.interpolate_water_table,
        allow_pinch_outs=preset.allow_pinch_outs,
        consulting_title_block=None,
        sample_figure_profile=preset.sample_figure_profile,
    )
    assert effective.layout == "consulting_section"
    assert effective.interpolate_water_table is False
    assert effective.allow_pinch_outs is False


def test_generic_consulting_still_forces_water_on() -> None:
    effective = effective_render_options(
        report_preset=False,
        render_layout="consulting_section",
        show_ground_surface=False,
        track_width_m=4.0,
        show_legend=True,
        interpolate_water_table=False,
        allow_pinch_outs=True,
        consulting_title_block=None,
        sample_figure_profile=False,
    )
    assert effective.interpolate_water_table is True
    assert effective.allow_pinch_outs is False
    assert effective.show_legend is False


def test_normalize_figure_preset_aliases() -> None:
    assert normalize_figure_preset("gwm_fence") == "gwm_fence"
    assert normalize_figure_preset("P2") == "p2_chemistry_sticks"
    assert normalize_figure_preset("advantage_p2") == "p2_chemistry_sticks"
    assert normalize_figure_preset("section_style") is None
    assert normalize_figure_preset("") is None
