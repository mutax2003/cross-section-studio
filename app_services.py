"""Cached ingest and render services."""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any

import streamlit as st

from ingestion import ingest_workbook
from models import CorrelationOverride, ParseResult, SectionFigureMetadata, Transect
from pipeline import (
    build_cross_section,
    filter_projected_for_interpolation,
    validate_interpretation_mode,
)
from projection import off_transect_warnings, project_boreholes
from section_build_request import SectionBuildRequest
from stratigraphy import (
    CorrelationPairSummary,
    build_stratigraphy,
    detect_polygon_overlaps,
    preview_correlation_health,
)
from transect_planner import recommend_transects


@st.cache_data(show_spinner="Parsing workbook...", ttl=3600, max_entries=8)
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


@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def cached_recommend_transects(
    collars: tuple[Any, ...],
    lithologies: tuple[Any, ...],
    top_n: int,
) -> list:
    return recommend_transects(collars, lithologies, top_n=top_n)


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
        warn_on_correlation_gaps=request.warn_on_correlation_gaps,
        show_water_elevation_labels=request.show_water_elevation_labels,
        show_water_legend=request.show_water_legend,
        show_dry_well_nm=request.show_dry_well_nm,
        water_interpolate_across_gaps=request.water_interpolate_across_gaps,
        environmental_parameters=request.environmental_parameters,
        show_parameter_labels=request.show_parameter_labels,
        parameter_interpolate_segments=request.parameter_interpolate_segments,
        parameter_interpolate_across_gaps=request.parameter_interpolate_across_gaps,
        render_layout=request.render_layout,
        track_width_m=request.track_width_m,
        elevation_mode=request.elevation_mode,
        raster_log_strips=request.raster_log_strips,
        export_formats=export_formats,
        consulting_title_block=request.consulting_title_block,
        screen_intervals=request.screen_intervals or subset.screen_intervals,
        vertical_gradients=request.vertical_gradients or subset.vertical_gradients,
        fail_on_overlaps=request.fail_on_overlaps,
    )
    return (
        result.svg_bytes,
        result.png_bytes,
        result.pdf_bytes,
        len(result.polygons),
        result.lithology_codes,
        result.overlap_warnings,
    )


@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def cached_build_section(
    subset_json: str,
    request_json: str,
) -> tuple[bytes, bytes, bytes, int, tuple[str, ...], tuple[str, ...]]:
    """Fast path: SVG (+ empty PNG/PDF placeholders). Use ``cached_build_section_exports`` for deliverables."""
    subset = ParseResult.model_validate_json(subset_json)
    request = SectionBuildRequest.model_validate_json(request_json)  # type: ignore[attr-defined]
    svg, _png, _pdf, count, codes, warnings = _run_build_cross_section(
        subset, request, export_formats=frozenset({"svg"})
    )
    return svg, b"", b"", count, codes, warnings


@st.cache_data(show_spinner="Preparing PNG export...", ttl=3600, max_entries=8)
def cached_build_section_png(
    subset_json: str,
    request_json: str,
) -> bytes:
    """Lazy PNG-only deliverable (avoids PDF work when only PNG is needed)."""
    subset = ParseResult.model_validate_json(subset_json)
    request = SectionBuildRequest.model_validate_json(request_json)  # type: ignore[attr-defined]
    _svg, png, _pdf, _count, _codes, _warnings = _run_build_cross_section(
        subset, request, export_formats=frozenset({"png"})
    )
    return png


@st.cache_data(show_spinner="Preparing PDF export...", ttl=3600, max_entries=8)
def cached_build_section_pdf(
    subset_json: str,
    request_json: str,
) -> bytes:
    """Lazy PDF-only deliverable (avoids PNG work when only PDF is needed)."""
    subset = ParseResult.model_validate_json(subset_json)
    request = SectionBuildRequest.model_validate_json(request_json)  # type: ignore[attr-defined]
    _svg, _png, pdf, _count, _codes, _warnings = _run_build_cross_section(
        subset, request, export_formats=frozenset({"pdf"})
    )
    return pdf


@st.cache_data(show_spinner="Preparing PNG/PDF exports...", ttl=3600, max_entries=8)
def cached_build_section_exports(
    subset_json: str,
    request_json: str,
) -> tuple[bytes, bytes]:
    """Build PNG+PDF in one pipeline pass (Prepare both)."""
    subset = ParseResult.model_validate_json(subset_json)
    request = SectionBuildRequest.model_validate_json(request_json)  # type: ignore[attr-defined]
    _svg, png, pdf, _count, _codes, _warnings = _run_build_cross_section(
        subset, request, export_formats=frozenset({"png", "pdf"})
    )
    return png, pdf



def preflight_correlation_health(
    subset: ParseResult,
    transect_points: tuple[tuple[float, float], ...],
    *,
    interpretation_mode: str,
    allow_pinch_outs: bool,
    correlation_overrides: tuple[CorrelationOverride, ...],
    offset_warning_m: float = 50.0,
    max_offset_for_interpolation_m: float | None = None,
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
    try:
        projected = filter_projected_for_interpolation(
            projected, max_offset_for_interpolation_m
        )
    except ValueError:
        return []
    return preview_correlation_health(
        projected,
        allow_pinch_outs=allow_pinch_outs,
        correlation_overrides=correlation_overrides,
    )


def preflight_polygon_overlap_warnings(
    subset: ParseResult,
    transect_points: tuple[tuple[float, float], ...],
    *,
    interpretation_mode: str,
    allow_pinch_outs: bool,
    correlation_overrides: tuple[CorrelationOverride, ...],
    offset_warning_m: float = 50.0,
    max_offset_for_interpolation_m: float | None = None,
) -> tuple[str, ...]:
    """Return configure-step warnings when fence polygons overlap (engine-only)."""
    if interpretation_mode == "borehole_only":
        return ()
    projected = project_boreholes(
        subset.collars,
        subset.lithologies,
        Transect(points=list(transect_points)),
        offset_warning_m=offset_warning_m,
        deviation_readings=subset.deviation_readings or None,
    )
    if projected.empty or len(projected["hole_id"].unique()) < 2:
        return ()
    try:
        projected = filter_projected_for_interpolation(
            projected, max_offset_for_interpolation_m
        )
    except ValueError as exc:
        return (str(exc),)
    if len(projected["hole_id"].unique()) < 2:
        return ()
    polygons = build_stratigraphy(
        projected,
        allow_pinch_outs=allow_pinch_outs,
        correlation_overrides=correlation_overrides,
    )
    overlaps = detect_polygon_overlaps(polygons)
    if not overlaps:
        return ()
    return (
        (
            f"Polygon overlap: {len(overlaps)} inter-hole contact conflict(s) detected — "
            "review correlation before export."
        ),
    )


@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def cached_configure_preflight(
    subset_json: str,
    transect_points_json: str,
    interpretation_mode: str,
    allow_pinch_outs: bool,
    correlation_overrides_json: str,
    offset_warning_m: float,
    max_offset_for_interpolation_m: float = 50.0,
    check_overlaps: bool = True,
) -> tuple[tuple[str, ...], tuple[CorrelationPairSummary, ...]]:
    """Cached Configure-step preflight: project once, then correlation + overlap checks."""
    subset = ParseResult.model_validate_json(subset_json)
    transect_points: tuple[tuple[float, float], ...] = tuple(
        tuple(point) for point in json.loads(transect_points_json)
    )
    overrides = tuple(
        CorrelationOverride.model_validate(item)
        for item in json.loads(correlation_overrides_json)
    )
    warnings = tuple(
        off_transect_warnings(
            subset.collars,
            Transect(points=list(transect_points)),
            offset_warning_m,
        )
    )
    if interpretation_mode == "borehole_only":
        return warnings, ()

    projected = project_boreholes(
        subset.collars,
        subset.lithologies,
        Transect(points=list(transect_points)),
        offset_warning_m=offset_warning_m,
        deviation_readings=subset.deviation_readings or None,
    )
    if projected.empty:
        return warnings, ()
    try:
        projected = filter_projected_for_interpolation(
            projected, max_offset_for_interpolation_m
        )
    except ValueError as exc:
        return warnings + (str(exc),), ()

    summaries = tuple(
        preview_correlation_health(
            projected,
            allow_pinch_outs=allow_pinch_outs,
            correlation_overrides=overrides,
        )
    )
    overlap_extra: tuple[str, ...] = ()
    if check_overlaps and len(projected["hole_id"].unique()) >= 2:
        polygons = build_stratigraphy(
            projected,
            allow_pinch_outs=allow_pinch_outs,
            correlation_overrides=overrides,
        )
        overlaps = detect_polygon_overlaps(polygons)
        if overlaps:
            overlap_extra = (
                (
                    f"Polygon overlap: {len(overlaps)} inter-hole contact conflict(s) detected — "
                    "review correlation before export."
                ),
            )
    return warnings + overlap_extra, summaries
