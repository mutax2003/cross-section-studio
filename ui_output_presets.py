"""Map user-facing output presets to render configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

InterpretationMode = Literal["interpolated", "correlation_lines", "borehole_only"]
ElevationMode = Literal["absolute", "relative"]


@dataclass(frozen=True)
class OutputPresetConfig:
    render_layout: str
    report_preset: bool
    allow_pinch_outs: bool
    show_ground_surface: bool
    interpolate_water_table: bool
    show_legend: bool
    # Sample-figure profiles (GWM fence / P2 sticks). None = leave sidebar free.
    interpretation_mode: InterpretationMode | None = None
    elevation_mode: ElevationMode | None = None
    vertical_exaggeration: float | None = None
    show_water_elevation_labels: bool | None = None
    show_water_legend: bool | None = None
    show_dry_well_nm: bool | None = None
    water_interpolate_across_gaps: bool | None = None
    # When True, consulting layout does not force water interpolation / pinch-outs off.
    sample_figure_profile: bool = False
    # Chemistry (Configure step defaults when preset is active).
    prefer_chemistry: bool = False
    show_parameter_labels: bool | None = None
    parameter_interpolate_segments: bool | None = None
    parameter_draw_markers: bool | None = None
    # Wave A drafting chrome (None = layout profile default).
    show_scale_bar: bool | None = None
    show_ve_annotation: bool | None = None
    show_parameter_legend_text: bool | None = None
    # Wave B: dashed CAD-style water connectors when False.
    water_line_solid: bool | None = None


OUTPUT_PRESET_LABELS: dict[str, str] = {
    "section_sheet": "Section sheet (Strater-style)",
    "consulting_report": "Consulting report (title block)",
    "gwm_fence": "GWM fence (MASL + groundwater)",
    "p2_chemistry_sticks": "P2 chemistry sticks (mbgs + chlorides)",
    "chemistry_gw": "Chemistry + groundwater (combined)",
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
        show_scale_bar=False,
        show_ve_annotation=False,
        show_parameter_legend_text=False,
    ),
    "consulting_report": OutputPresetConfig(
        render_layout="consulting_section",
        report_preset=False,
        allow_pinch_outs=True,
        show_ground_surface=True,
        interpolate_water_table=True,
        show_legend=False,
        show_scale_bar=True,
    ),
    "gwm_fence": OutputPresetConfig(
        render_layout="consulting_section",
        report_preset=False,
        allow_pinch_outs=False,
        show_ground_surface=True,
        interpolate_water_table=True,
        show_legend=False,
        interpretation_mode="interpolated",
        elevation_mode="absolute",
        vertical_exaggeration=5.0,
        show_water_elevation_labels=True,
        show_water_legend=True,
        show_dry_well_nm=True,
        water_interpolate_across_gaps=False,
        sample_figure_profile=True,
        prefer_chemistry=False,
        show_scale_bar=True,
        water_line_solid=True,
    ),
    "p2_chemistry_sticks": OutputPresetConfig(
        render_layout="consulting_section",
        report_preset=False,
        allow_pinch_outs=False,
        show_ground_surface=True,
        interpolate_water_table=False,
        show_legend=False,
        interpretation_mode="borehole_only",
        elevation_mode="relative",
        vertical_exaggeration=1.0,
        show_water_elevation_labels=False,
        show_water_legend=False,
        show_dry_well_nm=False,
        water_interpolate_across_gaps=False,
        sample_figure_profile=True,
        prefer_chemistry=True,
        show_parameter_labels=True,
        parameter_interpolate_segments=False,
        parameter_draw_markers=False,
        show_scale_bar=True,
    ),
    "chemistry_gw": OutputPresetConfig(
        render_layout="consulting_section",
        report_preset=False,
        allow_pinch_outs=False,
        show_ground_surface=True,
        interpolate_water_table=True,
        show_legend=False,
        interpretation_mode="borehole_only",
        elevation_mode="absolute",
        vertical_exaggeration=5.0,
        show_water_elevation_labels=True,
        show_water_legend=True,
        show_dry_well_nm=True,
        water_interpolate_across_gaps=False,
        sample_figure_profile=True,
        prefer_chemistry=True,
        show_parameter_labels=True,
        parameter_interpolate_segments=False,
        parameter_draw_markers=False,
        show_scale_bar=True,
        water_line_solid=False,
    ),
    "quick_preview": OutputPresetConfig(
        render_layout="chart",
        report_preset=False,
        allow_pinch_outs=True,
        show_ground_surface=True,
        interpolate_water_table=False,
        show_legend=True,
        show_scale_bar=True,
    ),
}

FIGURE_PRESET_IDS: frozenset[str] = frozenset(
    {"gwm_fence", "p2_chemistry_sticks", "chemistry_gw"}
)


def resolve_output_preset(preset: str) -> OutputPresetConfig:
    return OUTPUT_PRESETS.get(preset, OUTPUT_PRESETS["section_sheet"])


def normalize_figure_preset(raw: str | None) -> str | None:
    """Map Project-sheet figure_preset / section_style to a known output preset id."""
    if raw is None:
        return None
    key = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "gwm": "gwm_fence",
        "gwm_fence": "gwm_fence",
        "ecoventure_gwm": "gwm_fence",
        "p2": "p2_chemistry_sticks",
        "p2_chemistry_sticks": "p2_chemistry_sticks",
        "p2_sticks": "p2_chemistry_sticks",
        "advantage_p2": "p2_chemistry_sticks",
        "chemistry_sticks": "p2_chemistry_sticks",
        "chemistry_gw": "chemistry_gw",
        "p2_gw": "chemistry_gw",
        "chemistry_and_groundwater": "chemistry_gw",
    }
    resolved = aliases.get(key, key)
    return resolved if resolved in OUTPUT_PRESETS else None
