"""Tests for output preset mapping."""

from __future__ import annotations

from ui_output_presets import OUTPUT_PRESETS, resolve_output_preset


def test_consulting_report_preset() -> None:
    config = resolve_output_preset("consulting_report")
    assert config.render_layout == "consulting_section"
    assert config.interpolate_water_table is True
    assert config.show_legend is False


def test_unknown_preset_falls_back_to_section_sheet() -> None:
    config = resolve_output_preset("not_a_preset")
    assert config == OUTPUT_PRESETS["section_sheet"]
