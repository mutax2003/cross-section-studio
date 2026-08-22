"""Render configuration for cross-section builds and cache keys."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from models import (
    ConsultingTitleBlock,
    CorrelationOverride,
    DeviationReading,
    EnvironmentalReading,
    Fault,
    InterpretationMode,
    RasterLogStrip,
    ScreenInterval,
    SectionFigureMetadata,
    Unconformity,
    VerticalGradient,
    WaterLevel,
)
from render_profiles import ChemistryColorMode, ColumnHeaderDetail, LayoutMode

ElevationMode = Literal["absolute", "relative"]


class SectionBuildRequest(BaseModel, frozen=True):
    transect_points: tuple[tuple[float, float], ...] = Field(min_length=2)
    vertical_exaggeration: float = 5.0
    show_hatches: bool = False
    show_legend: bool = True
    section_title: str = "Borehole Cross-Section"
    interpretation_mode: InterpretationMode = "interpolated"
    allow_pinch_outs: bool = False
    fail_on_overlaps: bool = False
    offset_warning_m: float = 50.0
    uncertainty_spacing_m: float = 80.0
    uncertainty_offset_m: float = 50.0
    max_offset_for_interpolation_m: float | None = None
    show_ground_surface: bool = True
    interpolate_water_table: bool = False
    warn_on_correlation_gaps: bool = False
    show_water_elevation_labels: bool | None = None
    show_water_legend: bool | None = None
    show_dry_well_nm: bool | None = None
    water_interpolate_across_gaps: bool | None = None
    environmental_parameters: tuple[str, ...] = ()
    show_parameter_labels: bool | None = None
    parameter_interpolate_segments: bool | None = None
    parameter_interpolate_across_gaps: bool | None = None
    parameter_draw_markers: bool | None = None
    parameter_marker_size: float | None = None
    parameter_draw_leaders: bool | None = None
    parameter_label_include_units: bool | None = None
    column_header_detail: ColumnHeaderDetail | None = None
    show_scale_bar: bool | None = None
    show_ve_annotation: bool | None = None
    show_parameter_legend_text: bool | None = None
    export_font_family: str | None = None
    export_font_size: float | None = None
    selected_water_series_ids: tuple[str, ...] = ()
    water_line_solid: bool | None = None
    legend_ncol: int | None = None
    chemistry_color_mode: ChemistryColorMode | None = None
    chemistry_threshold_green_max: float | None = None
    chemistry_threshold_yellow_max: float | None = None
    render_layout: LayoutMode = "section_sheet"
    track_width_m: float = 3.0
    raster_log_strips: tuple[RasterLogStrip, ...] = ()
    coordinate_reference: str = ""
    uses_placeholder_elevation: bool = False
    elevation_mode: ElevationMode = "absolute"
    figure_metadata: SectionFigureMetadata | None = None
    water_levels: tuple[WaterLevel, ...] = ()
    correlation_overrides: tuple[CorrelationOverride, ...] = ()
    faults: tuple[Fault, ...] = ()
    unconformities: tuple[Unconformity, ...] = ()
    environmental_readings: tuple[EnvironmentalReading, ...] = ()
    deviation_readings: tuple[DeviationReading, ...] = ()
    consulting_title_block: ConsultingTitleBlock | None = None
    screen_intervals: tuple[ScreenInterval, ...] = ()
    vertical_gradients: tuple[VerticalGradient, ...] = ()

    def cache_key(self, hole_ids: tuple[str, ...]) -> str:
        return json.dumps(
            {
                "holes": hole_ids,
                "request": self.model_dump(mode="json"),
            },
            sort_keys=True,
        )
