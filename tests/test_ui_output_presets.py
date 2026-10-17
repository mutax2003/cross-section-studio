"""Tests for output preset mapping."""

from __future__ import annotations

from app_build import effective_render_options
from ui_output_presets import (
    FIGURE_PRESET_IDS,
    OUTPUT_PRESETS,
    VE_AUTO,
    locked_figure_summary,
    normalize_figure_preset,
    normalize_ve_choice,
    resolve_output_preset,
    sidebar_visibility,
    ve_choice_label,
    ve_choice_options,
    ve_choice_to_request,
    ve_short_text,
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
    # GWM-style sheets fit the page; the band prints the measured VE.
    assert config.vertical_exaggeration == VE_AUTO
    assert sidebar_visibility("gwm_fence").vertical_exaggeration_editable
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
    # Client P2 figures: NO VERTICAL EXAGGERATION, drawn at exactly 1x (locked).
    assert config.vertical_exaggeration == 1.0
    assert not sidebar_visibility("p2_chemistry_sticks").vertical_exaggeration_editable
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


def test_report_preset_preserves_track_width() -> None:
    effective = effective_render_options(
        report_preset=True,
        render_layout="section_sheet",
        show_ground_surface=False,
        track_width_m=5.5,
        auto_fit_track_width=False,
        show_legend=True,
        interpolate_water_table=False,
        allow_pinch_outs=False,
        consulting_title_block=None,
    )
    assert effective.layout == "section_sheet"
    assert effective.track_width_m == 5.5
    assert effective.auto_fit_track_width is False
    assert effective.show_ground_surface is True


def test_normalize_figure_preset_aliases() -> None:
    assert normalize_figure_preset("gwm_fence") == "gwm_fence"
    assert normalize_figure_preset("P2") == "p2_chemistry_sticks"
    assert normalize_figure_preset("advantage_p2") == "p2_chemistry_sticks"
    assert normalize_figure_preset("section_style") is None
    assert normalize_figure_preset("") is None


def test_consulting_report_seeds_the_pinch_out_choice_the_build_uses() -> None:
    """The preset seeded the toggle on while the build forced pinch-outs off,
    so Consulting report and GWM fence rendered identically under an
    "on" toggle."""
    preset = resolve_output_preset("consulting_report")
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
    assert preset.allow_pinch_outs is effective.allow_pinch_outs is False


def test_sidebar_visibility_hides_exactly_what_the_build_overrides() -> None:
    """A control is shown only when effective_render_options honours it."""
    from ui_output_presets import sidebar_visibility

    for preset_id, config in OUTPUT_PRESETS.items():
        vis = sidebar_visibility(preset_id)

        def build(**overrides):
            kwargs = dict(
                report_preset=config.report_preset,
                render_layout=config.render_layout,
                show_ground_surface=True,
                track_width_m=3.0,
                show_legend=True,
                interpolate_water_table=True,
                allow_pinch_outs=True,
                consulting_title_block=None,
                sample_figure_profile=config.sample_figure_profile,
            )
            kwargs.update(overrides)
            return effective_render_options(**kwargs)

        ground_honoured = build(show_ground_surface=False).show_ground_surface is False
        legend_honoured = build(show_legend=False).show_legend != build().show_legend
        assert vis.ground_surface_editable == ground_honoured, preset_id
        assert vis.chart_legend_editable == legend_honoured, preset_id
        if not vis.pinch_outs_editable and not config.sample_figure_profile:
            assert build().allow_pinch_outs is False, preset_id
        assert vis.title_block_shown == (config.render_layout == "consulting_section")


def test_output_style_names_are_plain() -> None:
    from ui_output_presets import OUTPUT_PRESET_LABELS, locked_groundwater_summary

    assert OUTPUT_PRESET_LABELS["gwm_fence"] == "Groundwater fence (elevation + water levels)"
    assert OUTPUT_PRESET_LABELS["p2_chemistry_sticks"] == "Chemistry columns (depth + lab values)"
    assert locked_groundwater_summary("section_sheet") is None
    summary = locked_groundwater_summary("consulting_report")
    assert summary and summary.startswith("Set by Consulting report:")


def test_ve_choice_helpers() -> None:
    assert normalize_ve_choice(None) == VE_AUTO
    assert normalize_ve_choice("auto") == VE_AUTO
    assert normalize_ve_choice("5") == 5.0
    assert normalize_ve_choice("5x") == 5.0
    assert normalize_ve_choice("-2") == VE_AUTO
    assert ve_choice_to_request(VE_AUTO) is None
    assert ve_choice_to_request(2.0) == 2.0
    assert ve_choice_label(VE_AUTO) == "Auto (fit page)"
    assert ve_choice_label(1.0) == "1× (true scale)"
    assert ve_choice_label(5.0) == "5×"
    options = ve_choice_options(3.0)
    assert options[0] == VE_AUTO and 3.0 in options and 5.0 in options
    assert ve_short_text(None) == "auto (fit page)"
    assert ve_short_text(5.0) == "5×"
    assert "exactly 1×" in (locked_figure_summary("p2_chemistry_sticks") or "")
