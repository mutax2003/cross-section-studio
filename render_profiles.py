"""Cross-section render layout profiles (chart vs Strater-like section sheet)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

LayoutMode = Literal["chart", "section_sheet", "consulting_section"]
YAxisMode = Literal["elevation_rl", "depth_below_collar"]
WaterSymbol = Literal["circle", "triangle", "diamond"]
ScaleBarPosition = Literal["bottom_left", "bottom_right"]


class CrossSectionRenderProfile(BaseModel, frozen=True):
    layout: LayoutMode = "section_sheet"
    track_width_m: float = Field(default=3.0, gt=0.0)
    show_grid: bool = False
    show_ground_surface: bool = True
    show_sky_fill: bool = True
    show_contact_ticks: bool = True
    show_eol_bar: bool = True
    show_track_border: bool = True
    show_centerline: bool = False
    show_column_headers: bool = True
    show_track_lithology: bool = True
    show_dual_y_axes: bool = False
    show_report_grid: bool = False
    legend_in_title_block: bool = False
    interpolate_water_table_default: bool = False
    y_axis_mode: YAxisMode = "elevation_rl"
    water_symbol: WaterSymbol = "triangle"
    title_block: bool = True
    show_ve_annotation: bool = True
    scale_bar_position: ScaleBarPosition = "bottom_left"
    fence_alpha: float = Field(default=0.58, ge=0.0, le=1.0)
    show_overlap_markers: bool = False
    show_overlap_footer: bool = False
    use_consulting_palette: bool = False
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
    y_axis_mode="elevation_rl",
    water_symbol="triangle",
    title_block=True,
    show_ve_annotation=True,
    fence_alpha=0.58,
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
    y_axis_mode="elevation_rl",
    water_symbol="triangle",
    title_block=True,
    show_ve_annotation=False,
    fence_alpha=1.0,
    show_overlap_markers=False,
    show_overlap_footer=False,
    use_consulting_palette=True,
    x_major_grid_m=10.0,
    y_axis_label="ELEVATION ABOVE SEA LEVEL (MASL)",
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
