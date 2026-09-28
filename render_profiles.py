"""Cross-section render layout profiles (chart vs Strater-like section sheet)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

LayoutMode = Literal["chart", "section_sheet", "consulting_section"]
YAxisMode = Literal["elevation_rl", "depth_below_collar"]
WaterSymbol = Literal["circle", "triangle", "diamond"]
ScaleBarPosition = Literal["bottom_left", "bottom_right"]
ColumnHeaderDetail = Literal["id_only", "id_rl_td"]
ChemistryColorMode = Literal["black", "threshold"]


class CrossSectionRenderProfile(BaseModel, frozen=True):
    layout: LayoutMode = "section_sheet"
    track_width_m: float = Field(
        default=3.0,
        gt=0.0,
        description="Schematic borehole column width on the profile X axis (metres), not casing diameter.",
    )
    auto_fit_track_width: bool = Field(
        default=True,
        description="Narrow columns when adjacent holes are closer than track_width_m to avoid overlap.",
    )
    show_grid: bool = False
    show_ground_surface: bool = True
    show_sky_fill: bool = True
    show_contact_ticks: bool = True
    show_eol_bar: bool = True
    show_track_border: bool = True
    show_centerline: bool = False
    show_column_headers: bool = True
    column_header_detail: ColumnHeaderDetail = "id_only"
    show_track_lithology: bool = True
    show_dual_y_axes: bool = False
    show_report_grid: bool = False
    legend_in_title_block: bool = False
    interpolate_water_table_default: bool = False
    water_interpolate_segments: bool = True
    water_interpolate_across_gaps: bool = False
    show_water_elevation_labels: bool = False
    show_water_legend: bool = False
    show_dry_well_nm: bool = False
    y_axis_mode: YAxisMode = "elevation_rl"
    water_symbol: WaterSymbol = "triangle"
    title_block: bool = True
    show_ve_annotation: bool = False
    show_scale_bar: bool = False
    show_parameter_legend_text: bool = False
    scale_bar_position: ScaleBarPosition = "bottom_left"
    fence_alpha: float = Field(default=0.58, ge=0.0, le=1.0)
    show_overlap_markers: bool = False
    show_overlap_footer: bool = False
    use_consulting_palette: bool = False
    show_pinch_out_legend: bool = True
    compact_water_legend: bool = False
    water_line_solid: bool = False
    consulting_axis_from_zero: bool = False
    show_parameter_markers: bool = False
    show_parameter_labels: bool = True
    parameter_interpolate_segments: bool = False
    parameter_interpolate_across_gaps: bool = False
    parameter_draw_markers: bool = True
    parameter_marker: str = "o"
    parameter_marker_size: float = Field(default=16.0, gt=0.0)
    parameter_draw_leaders: bool = False
    parameter_label_include_units: bool = False
    export_font_family: str = "Arial"
    export_font_size: float = Field(default=8.0, gt=0.0)
    legend_ncol: int = Field(default=1, ge=1, le=3)
    chemistry_color_mode: ChemistryColorMode = "black"
    chemistry_threshold_green_max: float | None = None
    chemistry_threshold_yellow_max: float | None = None
    x_major_grid_m: float = 10.0
    y_axis_label: str = ""


CHART_PROFILE = CrossSectionRenderProfile(
    layout="chart",
    track_width_m=1.6,
    show_grid=True,
    show_ground_surface=False,
    show_sky_fill=False,
    show_contact_ticks=False,
    show_eol_bar=False,
    show_track_border=False,
    show_centerline=True,
    show_column_headers=False,
    y_axis_mode="elevation_rl",
    water_symbol="circle",
    title_block=False,
    show_ve_annotation=False,
    show_scale_bar=True,
    fence_alpha=0.92,
    show_overlap_markers=True,
    show_overlap_footer=True,
)

SECTION_SHEET_PROFILE = CrossSectionRenderProfile(
    layout="section_sheet",
    track_width_m=3.0,
    show_grid=False,
    show_ground_surface=True,
    show_sky_fill=True,
    show_contact_ticks=True,
    show_eol_bar=True,
    show_track_border=True,
    show_centerline=False,
    show_column_headers=True,
    column_header_detail="id_only",
    y_axis_mode="elevation_rl",
    water_symbol="triangle",
    title_block=True,
    show_ve_annotation=False,
    show_scale_bar=False,
    show_parameter_legend_text=False,
    fence_alpha=0.58,
    show_parameter_markers=True,
    parameter_interpolate_segments=False,
    parameter_marker="o",
    parameter_draw_leaders=False,
    parameter_label_include_units=False,
    legend_ncol=2,
    chemistry_color_mode="black",
)

CONSULTING_SECTION_PROFILE = CrossSectionRenderProfile(
    layout="consulting_section",
    track_width_m=1.2,
    show_grid=False,
    show_ground_surface=True,
    show_sky_fill=False,
    show_contact_ticks=False,
    show_eol_bar=False,
    show_track_border=False,
    show_centerline=True,
    show_column_headers=False,
    show_track_lithology=False,
    show_dual_y_axes=True,
    show_report_grid=True,
    legend_in_title_block=True,
    interpolate_water_table_default=True,
    water_interpolate_segments=True,
    water_interpolate_across_gaps=False,
    show_water_elevation_labels=True,
    show_water_legend=True,
    show_dry_well_nm=True,
    y_axis_mode="elevation_rl",
    water_symbol="triangle",
    title_block=True,
    show_ve_annotation=False,
    # Consulting subtitle band includes scale / VE (report chrome).
    show_scale_bar=True,
    show_parameter_legend_text=False,
    fence_alpha=1.0,
    show_overlap_markers=True,
    show_overlap_footer=True,
    use_consulting_palette=True,
    show_pinch_out_legend=False,
    compact_water_legend=True,
    water_line_solid=True,
    consulting_axis_from_zero=True,
    x_major_grid_m=10.0,
    y_axis_label="ELEVATION ABOVE SEA LEVEL (MASL)",
    show_parameter_markers=True,
    show_parameter_labels=True,
    parameter_interpolate_segments=False,
    parameter_marker="o",
    parameter_draw_leaders=False,
    parameter_label_include_units=False,
    legend_ncol=2,
    chemistry_color_mode="black",
)


def profile_for_layout(layout: LayoutMode) -> CrossSectionRenderProfile:
    if layout == "chart":
        return CHART_PROFILE
    if layout == "consulting_section":
        return CONSULTING_SECTION_PROFILE
    return SECTION_SHEET_PROFILE


def profile_with_elevation_mode(
    profile: CrossSectionRenderProfile,
    elevation_mode: str,
) -> CrossSectionRenderProfile:
    y_mode: YAxisMode = (
        "depth_below_collar" if elevation_mode == "relative" else "elevation_rl"
    )
    return profile.model_copy(update={"y_axis_mode": y_mode})
