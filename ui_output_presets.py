"""Map user-facing output presets to render configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OutputPresetConfig:
    render_layout: str
    report_preset: bool
    allow_pinch_outs: bool
    show_ground_surface: bool
    interpolate_water_table: bool
    show_legend: bool


OUTPUT_PRESET_LABELS: dict[str, str] = {
    "section_sheet": "Section sheet (Strater-style)",
    "consulting_report": "Consulting report (title block)",
    "quick_preview": "Quick preview (chart)",
}

OUTPUT_PRESETS: dict[str, OutputPresetConfig] = {
    "section_sheet": OutputPresetConfig(
        render_layout="section_sheet",
        report_preset=True,
        allow_pinch_outs=False,
        show_ground_surface=True,
        interpolate_water_table=False,
        show_legend=True,
    ),
    "consulting_report": OutputPresetConfig(
        render_layout="consulting_section",
        report_preset=False,
        allow_pinch_outs=True,
        show_ground_surface=True,
        interpolate_water_table=True,
        show_legend=False,
    ),
    "quick_preview": OutputPresetConfig(
        render_layout="chart",
        report_preset=False,
        allow_pinch_outs=True,
        show_ground_surface=True,
        interpolate_water_table=False,
        show_legend=True,
    ),
}


def resolve_output_preset(preset: str) -> OutputPresetConfig:
    return OUTPUT_PRESETS.get(preset, OUTPUT_PRESETS["section_sheet"])
