"""Cached ingest and render services."""

from __future__ import annotations

import json
from dataclasses import replace
from io import BytesIO
from typing import Any

import streamlit as st

from ingestion import ingest_workbook
from models import CorrelationOverride, ParseResult, SectionFigureMetadata, Transect
from pipeline import (
    ALL_EXPORT_FORMATS,
    SectionGeometry,
    compute_section_geometry,
    render_cross_section_from_geometry,
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


def _apply_section_geometry_qa(
    geometry: SectionGeometry,
    request: SectionBuildRequest,
) -> SectionGeometry:
    """Apply request QA flags after geometry cache hit (polygons unchanged)."""
    if request.fail_on_overlaps and geometry.overlap_pairs:
        raise ValueError(
            f"Polygon overlap detected ({len(geometry.overlap_pairs)} pair(s)); "
            "resolve correlation or set fail_on_overlaps=False to export."
        )
    if not request.warn_on_correlation_gaps:
        filtered_warnings = tuple(
            message
            for message in geometry.overlap_warnings
            if not message.startswith("Correlation gap ")
        )
        if filtered_warnings != geometry.overlap_warnings:
            geometry = replace(geometry, overlap_warnings=filtered_warnings)
    return geometry
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


@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def cached_parse_subset(subset_json: str) -> ParseResult:
    """Parse ``ParseResult`` JSON once; Generate/Prepare/geometry share this cache."""
    return ParseResult.model_validate_json(subset_json)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def cached_parse_request(request_json: str) -> SectionBuildRequest:
    """Parse ``SectionBuildRequest`` JSON once; Generate/Prepare/geometry share this cache."""
    return SectionBuildRequest.model_validate_json(request_json)  # type: ignore[attr-defined]


def _cached_section_inputs(
    subset_json: str, request_json: str
) -> tuple[ParseResult, SectionBuildRequest]:
    return cached_parse_subset(subset_json), cached_parse_request(request_json)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def cached_compute_section_geometry(
    subset_json: str, geometry_request_json: str
) -> SectionGeometry:
    """``SectionGeometry`` shared by Generate SVG and Prepare deliverables (PNG/PDF).

    Streamlit ``@st.cache_data`` serializes the returned object. Cache key is
    geometry-scoped JSON from ``SectionBuildRequest.geometry_cache_payload()``
    (``geometry_request_json``), so cosmetic / render-only fields do not bust
    this cache.
    """
    subset = cached_parse_subset(subset_json)
    request = SectionBuildRequest.model_validate(json.loads(geometry_request_json))
    _figure_metadata, mode, overrides = _build_section_kwargs(subset, request)
    return compute_section_geometry(
        subset.collars,
        subset.lithologies,
        request.transect_points,
        offset_warning_m=request.offset_warning_m,
        interpretation_mode=mode,
        allow_pinch_outs=request.allow_pinch_outs,
        max_offset_for_interpolation_m=request.max_offset_for_interpolation_m,
        correlation_overrides=overrides,
        deviation_readings=request.deviation_readings or subset.deviation_readings,
        warn_on_correlation_gaps=True,
        fail_on_overlaps=False,
    )


def _run_build_cross_section(
    subset: ParseResult,
    request: SectionBuildRequest,
    *,
    export_formats: frozenset[str],
    subset_json: str,
) -> tuple[bytes, bytes, bytes, int, tuple[str, ...], tuple[str, ...]]:
    figure_metadata, mode, _overrides = _build_section_kwargs(subset, request)
    geometry_json = json.dumps(request.geometry_cache_payload(), sort_keys=True)
    geometry = _apply_section_geometry_qa(
        cached_compute_section_geometry(subset_json, geometry_json),
        request,
    )
    result = render_cross_section_from_geometry(
        geometry,
        request.transect_points,
        vertical_exaggeration=request.vertical_exaggeration,
        show_hatches=request.show_hatches,
        show_legend=request.show_legend,
        title=request.section_title,
        interpretation_mode=mode,
        water_levels=request.water_levels or subset.water_levels or None,
        uncertainty_spacing_m=request.uncertainty_spacing_m,
        uncertainty_offset_m=request.uncertainty_offset_m,
        faults=request.faults or subset.faults,
        unconformities=request.unconformities or subset.unconformities,
        environmental_readings=request.environmental_readings or subset.environmental_readings,
        figure_metadata=figure_metadata,
        show_ground_surface=request.show_ground_surface,
        interpolate_water_table=request.interpolate_water_table,
        show_water_elevation_labels=request.show_water_elevation_labels,
        show_water_legend=request.show_water_legend,
        show_dry_well_nm=request.show_dry_well_nm,
        water_interpolate_across_gaps=request.water_interpolate_across_gaps,
        environmental_parameters=request.environmental_parameters,
        show_parameter_labels=request.show_parameter_labels,
        parameter_interpolate_segments=request.parameter_interpolate_segments,
        parameter_interpolate_across_gaps=request.parameter_interpolate_across_gaps,
        parameter_draw_markers=request.parameter_draw_markers,
        parameter_marker_size=request.parameter_marker_size,
        parameter_draw_leaders=request.parameter_draw_leaders,
        parameter_label_include_units=request.parameter_label_include_units,
        column_header_detail=request.column_header_detail,
        show_scale_bar=request.show_scale_bar,
        show_ve_annotation=request.show_ve_annotation,
        show_parameter_legend_text=request.show_parameter_legend_text,
        export_font_family=request.export_font_family,
        export_font_size=request.export_font_size,
        selected_water_series_ids=request.selected_water_series_ids or None,
        water_line_solid=request.water_line_solid,
        legend_ncol=request.legend_ncol,
        chemistry_color_mode=request.chemistry_color_mode,
        chemistry_threshold_green_max=request.chemistry_threshold_green_max,
        chemistry_threshold_yellow_max=request.chemistry_threshold_yellow_max,
        render_layout=request.render_layout,
        track_width_m=request.track_width_m,
        auto_fit_track_width=request.auto_fit_track_width,
        elevation_mode=request.elevation_mode,
        raster_log_strips=request.raster_log_strips,
        export_formats=export_formats,
        consulting_title_block=request.consulting_title_block,
        screen_intervals=request.screen_intervals or subset.screen_intervals,
        vertical_gradients=request.vertical_gradients or subset.vertical_gradients,
        export_framing=request.export_framing,
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
def cached_build_section_bundle(
    subset_json: str,
    request_json: str,
) -> tuple[bytes, bytes, bytes, int, tuple[str, ...], tuple[str, ...]]:
    """One-shot SVG+PNG+PDF (scripts / full export). Prefer SVG-first Generate + Prepare exports."""
    subset, request = _cached_section_inputs(subset_json, request_json)
    return _run_build_cross_section(
        subset,
        request,
        export_formats=ALL_EXPORT_FORMATS,
        subset_json=subset_json,
    )


@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def cached_build_section(
    subset_json: str,
    request_json: str,
) -> tuple[bytes, bytes, bytes, int, tuple[str, ...], tuple[str, ...]]:
    """Generate path: SVG only. Geometry is cached for Prepare reuse."""
    subset, request = _cached_section_inputs(subset_json, request_json)
    svg, _png, _pdf, count, codes, warnings = _run_build_cross_section(
        subset,
        request,
        export_formats=frozenset({"svg"}),
        subset_json=subset_json,
    )
    return svg, b"", b"", count, codes, warnings


def cached_build_section_png(
    subset_json: str,
    request_json: str,
) -> bytes:
    """PNG-only Prepare; delegates to ``cached_build_section_exports`` (shared draw cache)."""
    png, _pdf = cached_build_section_exports(subset_json, request_json)
    return png


def cached_build_section_pdf(
    subset_json: str,
    request_json: str,
) -> bytes:
    """PDF-only Prepare; delegates to ``cached_build_section_exports`` (shared draw cache)."""
    _png, pdf = cached_build_section_exports(subset_json, request_json)
    return pdf


@st.cache_data(show_spinner="Preparing PNG/PDF exports...", ttl=3600, max_entries=8)
def cached_build_section_exports(
    subset_json: str,
    request_json: str,
) -> tuple[bytes, bytes]:
    """Prepare both: one matplotlib draw for PNG+PDF; reuses Generate geometry cache."""
    subset, request = _cached_section_inputs(subset_json, request_json)
    _svg, png, pdf, _count, _codes, _warnings = _run_build_cross_section(
        subset,
        request,
        export_formats=frozenset({"png", "pdf"}),
        subset_json=subset_json,
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
    """Cached Configure-step preflight; warms Generate geometry cache when possible."""
    subset = cached_parse_subset(subset_json)
    transect_points: tuple[tuple[float, float], ...] = tuple(
        tuple(point) for point in json.loads(transect_points_json)
    )
    overrides = tuple(
        CorrelationOverride.model_validate(item)
        for item in json.loads(correlation_overrides_json)
    )
    transect = Transect(points=list(transect_points))
    warnings = tuple(
        off_transect_warnings(
            subset.collars,
            transect,
            offset_warning_m,
        )
    )
    if interpretation_mode == "borehole_only":
        return warnings, ()

    preflight_request = SectionBuildRequest(
        transect_points=transect_points,
        interpretation_mode=interpretation_mode,  # type: ignore[arg-type]
        allow_pinch_outs=allow_pinch_outs,
        correlation_overrides=overrides,
        offset_warning_m=offset_warning_m,
        max_offset_for_interpolation_m=max_offset_for_interpolation_m,
    )
    geometry_json = json.dumps(
        preflight_request.geometry_cache_payload(),
        sort_keys=True,
    )
    try:
        geometry = cached_compute_section_geometry(subset_json, geometry_json)
    except ValueError as exc:
        message = str(exc)
        if message.startswith("No lithology intervals were projected"):
            return warnings, ()
        return warnings + (message,), ()

    overlap_extra: tuple[str, ...] = ()
    if check_overlaps and geometry.overlap_pairs:
        overlap_extra = (
            (
                f"Polygon overlap: {len(geometry.overlap_pairs)} inter-hole contact conflict(s) detected — "
                "review correlation before export."
            ),
        )

    try:
        filtered = filter_projected_for_interpolation(
            geometry.projected,
            max_offset_for_interpolation_m,
        )
    except ValueError:
        return warnings + overlap_extra, ()

    summaries = tuple(
        preview_correlation_health(
            filtered,
            allow_pinch_outs=allow_pinch_outs,
            correlation_overrides=overrides,
        )
    )
    return warnings + overlap_extra, summaries
