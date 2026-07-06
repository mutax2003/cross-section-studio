"""Cached ingest and render services."""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any

import streamlit as st

from ingestion import ingest_workbook
from models import CorrelationOverride, ParseResult, SectionFigureMetadata, Transect
from pipeline import PDF_EXPORT_FORMATS, SVG_PNG_EXPORT_FORMATS, build_cross_section, validate_interpretation_mode
from projection import project_boreholes
from section_build_request import SectionBuildRequest
from stratigraphy import CorrelationPairSummary, preview_correlation_health
from transect_planner import recommend_transects


@st.cache_data(show_spinner="Parsing workbook...")
def cached_ingest_workbook(
    file_bytes: bytes,
    profile_id: str | None,
    override_id: str | None,
    elevation_m: float | None,
    target_crs: str | None,
    aliases_json: str,
    auto_assign_unit_order: bool,
) -> tuple[ParseResult, Any]:
    aliases = json.loads(aliases_json)
    return ingest_workbook(
        BytesIO(file_bytes),
        profile_id=profile_id,
        override_id=override_id,
        elevation_m=elevation_m,
        target_crs=target_crs,
        lithology_aliases=aliases,
        auto_assign_unit_order=auto_assign_unit_order,
    )


@st.cache_data(show_spinner=False)
def cached_recommend_transects(
    collars: tuple[Any, ...],
    lithologies: tuple[Any, ...],
    top_n: int,
) -> list:
    return recommend_transects(list(collars), list(lithologies), top_n=top_n)


def _build_section_kwargs(
    subset: ParseResult,
    request: SectionBuildRequest,
) -> tuple[SectionFigureMetadata, str, tuple[CorrelationOverride, ...]]:
    elevation_datum = next(
        (collar.elevation_datum for collar in subset.collars if collar.elevation_datum),
        "Collar RL",
    )
    if request.elevation_mode == "relative":
        elevation_datum = "Depth below collar (relative)"
    figure_metadata = request.figure_metadata or SectionFigureMetadata(
        coordinate_reference=request.coordinate_reference,
        elevation_datum=elevation_datum,
        vertical_exaggeration=request.vertical_exaggeration,
        hole_count=len(subset.collars),
        uses_placeholder_elevation=request.uses_placeholder_elevation
        and request.elevation_mode == "absolute",
    )
    mode = validate_interpretation_mode(request.interpretation_mode)
    overrides = tuple(request.correlation_overrides) + tuple(subset.correlation_overrides)
    return figure_metadata, mode, overrides


def _run_build_cross_section(
    subset: ParseResult,
    request: SectionBuildRequest,
    *,
    export_formats: frozenset[str],
) -> tuple[bytes, bytes, bytes, int, tuple[str, ...], tuple[str, ...]]:
    figure_metadata, mode, overrides = _build_section_kwargs(subset, request)
    result = build_cross_section(
        subset.collars,
        subset.lithologies,
        request.transect_points,
        vertical_exaggeration=request.vertical_exaggeration,
        show_hatches=request.show_hatches,
        show_legend=request.show_legend,
        title=request.section_title,
        interpretation_mode=mode,
        allow_pinch_outs=request.allow_pinch_outs,
        water_levels=request.water_levels or subset.water_levels or None,
        offset_warning_m=request.offset_warning_m,
        uncertainty_spacing_m=request.uncertainty_spacing_m,
        uncertainty_offset_m=request.uncertainty_offset_m,
        max_offset_for_interpolation_m=request.max_offset_for_interpolation_m,
        correlation_overrides=overrides,
        faults=request.faults or subset.faults,
        unconformities=request.unconformities or subset.unconformities,
        environmental_readings=request.environmental_readings or subset.environmental_readings,
        deviation_readings=request.deviation_readings or subset.deviation_readings,
        figure_metadata=figure_metadata,
        show_ground_surface=request.show_ground_surface,
        interpolate_water_table=request.interpolate_water_table,
        render_layout=request.render_layout,
        track_width_m=request.track_width_m,
        elevation_mode=request.elevation_mode,
        raster_log_strips=request.raster_log_strips,
        export_formats=export_formats,
        consulting_title_block=request.consulting_title_block,
        screen_intervals=request.screen_intervals or subset.screen_intervals,
        vertical_gradients=request.vertical_gradients or subset.vertical_gradients,
    )
    return (
        result.svg_bytes,
        result.png_bytes,
        result.pdf_bytes,
        len(result.polygons),
        result.lithology_codes,
        result.overlap_warnings,
    )


@st.cache_data(show_spinner=False)
def cached_build_section(
    subset_json: str,
    request_json: str,
) -> tuple[bytes, bytes, bytes, int, tuple[str, ...], tuple[str, ...]]:
    subset = ParseResult.model_validate_json(subset_json)
    request = SectionBuildRequest.model_validate_json(request_json)  # type: ignore[attr-defined]
    return _run_build_cross_section(subset, request, export_formats=SVG_PNG_EXPORT_FORMATS)


@st.cache_data(show_spinner="Preparing PDF report...")
def cached_build_section_pdf(
    subset_json: str,
    request_json: str,
) -> bytes:
    subset = ParseResult.model_validate_json(subset_json)
    request = SectionBuildRequest.model_validate_json(request_json)  # type: ignore[attr-defined]
    return _run_build_cross_section(
        subset,
        request,
        export_formats=PDF_EXPORT_FORMATS,
    )[2]


def preflight_correlation_health(
    subset: ParseResult,
    transect_points: tuple[tuple[float, float], ...],
    *,
    interpretation_mode: str,
    allow_pinch_outs: bool,
    correlation_overrides: tuple[CorrelationOverride, ...],
    offset_warning_m: float = 50.0,
) -> list[CorrelationPairSummary]:
    """Project transect subset and summarize correlation match rates (UI preflight only)."""
    if interpretation_mode == "borehole_only":
        return []
    projected = project_boreholes(
        subset.collars,
        subset.lithologies,
        Transect(points=list(transect_points)),
        offset_warning_m=offset_warning_m,
        deviation_readings=subset.deviation_readings or None,
    )
    if projected.empty:
        return []
    return preview_correlation_health(
        projected,
        allow_pinch_outs=allow_pinch_outs,
        correlation_overrides=correlation_overrides,
    )
