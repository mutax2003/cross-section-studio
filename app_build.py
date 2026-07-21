"""Section build request assembly and cache-key resolution."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from app_common import _active_transect_selection, _session_correlation_overrides
from app_services import cached_build_section
from models import (
    ConsultingTitleBlock,
    Lithology,
    ParseResult,
    Transect,
    subset_parse_result,
)
from projection import off_transect_warnings
from section_build_request import SectionBuildRequest
from ui_helpers import dedupe_messages, holes_missing_lithology, screen_interval_warnings


@dataclass(frozen=True)
class EffectiveRenderOptions:
    layout: str
    show_ground_surface: bool
    track_width_m: float
    show_legend: bool
    interpolate_water_table: bool
    allow_pinch_outs: bool
    consulting_title_block: ConsultingTitleBlock | None


def effective_render_options(
    *,
    report_preset: bool,
    render_layout: str,
    show_ground_surface: bool,
    track_width_m: float,
    show_legend: bool,
    interpolate_water_table: bool,
    allow_pinch_outs: bool,
    consulting_title_block: ConsultingTitleBlock | None,
) -> EffectiveRenderOptions:
    layout = "section_sheet" if report_preset else render_layout
    is_consulting = layout == "consulting_section"
    return EffectiveRenderOptions(
        layout=layout,
        show_ground_surface=True if report_preset or is_consulting else show_ground_surface,
        track_width_m=3.0 if report_preset else track_width_m,
        show_legend=False if is_consulting else show_legend,
        interpolate_water_table=True if is_consulting else interpolate_water_table,
        allow_pinch_outs=False if is_consulting else allow_pinch_outs,
        consulting_title_block=consulting_title_block if is_consulting else None,
    )


def build_section_request(
    *,
    transect_points: tuple[tuple[float, float], ...],
    vertical_exaggeration: float,
    show_hatches: bool,
    show_legend: bool,
    section_title: str,
    interpretation_mode: str,
    allow_pinch_outs: bool,
    offset_warning_m: float,
    uncertainty_spacing_m: float,
    uncertainty_offset_m: float,
    max_offset_for_interpolation_m: float | None,
    show_ground_surface: bool,
    interpolate_water_table: bool,
    warn_on_correlation_gaps: bool = False,
    show_water_elevation_labels: bool | None = None,
    show_water_legend: bool | None = None,
    show_dry_well_nm: bool | None = None,
    water_interpolate_across_gaps: bool | None = None,
    environmental_parameters: tuple[str, ...] = (),
    show_parameter_labels: bool | None = None,
    parameter_interpolate_segments: bool | None = None,
    parameter_interpolate_across_gaps: bool | None = None,
    render_layout: str,
    track_width_m: float,
    coordinate_reference: str,
    uses_placeholder_elevation: bool,
    elevation_mode: str,
    consulting_title_block: ConsultingTitleBlock | None = None,
    fail_on_overlaps: bool = False,
) -> SectionBuildRequest:
    return SectionBuildRequest(
        transect_points=transect_points,
        vertical_exaggeration=vertical_exaggeration,
        show_hatches=show_hatches,
        show_legend=show_legend,
        section_title=section_title,
        interpretation_mode=interpretation_mode,  # type: ignore[arg-type]
        allow_pinch_outs=allow_pinch_outs,
        offset_warning_m=offset_warning_m,
        uncertainty_spacing_m=uncertainty_spacing_m,
        uncertainty_offset_m=uncertainty_offset_m,
        max_offset_for_interpolation_m=max_offset_for_interpolation_m,
        show_ground_surface=show_ground_surface,
        interpolate_water_table=interpolate_water_table,
        warn_on_correlation_gaps=warn_on_correlation_gaps,
        show_water_elevation_labels=show_water_elevation_labels,
        show_water_legend=show_water_legend,
        show_dry_well_nm=show_dry_well_nm,
        water_interpolate_across_gaps=water_interpolate_across_gaps,
        environmental_parameters=environmental_parameters,
        show_parameter_labels=show_parameter_labels,
        parameter_interpolate_segments=parameter_interpolate_segments,
        parameter_interpolate_across_gaps=parameter_interpolate_across_gaps,
        render_layout=render_layout,  # type: ignore[arg-type]
        track_width_m=track_width_m,
        coordinate_reference=coordinate_reference,
        uses_placeholder_elevation=uses_placeholder_elevation,
        elevation_mode=elevation_mode,  # type: ignore[arg-type]
        correlation_overrides=_session_correlation_overrides(),
        consulting_title_block=consulting_title_block,
        fail_on_overlaps=fail_on_overlaps,
    )


def collect_section_build_request(
    parse_result: ParseResult,
    *,
    transect_mode: str,
    selected_holes: list[str],
    coordinate_text: str,
    offset_warning_m: float,
    vertical_exaggeration: float,
    show_hatches: bool,
    show_legend: bool,
    section_title: str,
    interpretation_mode: str,
    allow_pinch_outs: bool,
    uncertainty_spacing_m: float,
    uncertainty_offset_m: float,
    max_offset_for_interpolation_m: float | None,
    show_ground_surface: bool,
    interpolate_water_table: bool,
    warn_on_correlation_gaps: bool = False,
    show_water_elevation_labels: bool | None = None,
    show_water_legend: bool | None = None,
    show_dry_well_nm: bool | None = None,
    water_interpolate_across_gaps: bool | None = None,
    environmental_parameters: tuple[str, ...] = (),
    show_parameter_labels: bool | None = None,
    parameter_interpolate_segments: bool | None = None,
    parameter_interpolate_across_gaps: bool | None = None,
    render_layout: str,
    track_width_m: float,
    coordinate_reference: str,
    uses_placeholder_elevation: bool,
    elevation_mode: str,
    report_preset: bool,
    consulting_title_block: ConsultingTitleBlock | None = None,
    selection: tuple[tuple[str, ...], tuple[tuple[float, float], ...]] | None = None,
    fail_on_overlaps: bool = False,
) -> tuple[SectionBuildRequest | None, str | None]:
    """Single collector for generate click and staleness checks."""
    if selection is None:
        selection = _active_transect_selection(
            parse_result,
            transect_mode,
            selected_holes,
            coordinate_text,
            offset_warning_m,
        )
    if selection is None:
        return None, None
    active_hole_ids, transect_points = selection
    effective = effective_render_options(
        report_preset=report_preset,
        render_layout=render_layout,
        show_ground_surface=show_ground_surface,
        track_width_m=track_width_m,
        show_legend=show_legend,
        interpolate_water_table=interpolate_water_table,
        allow_pinch_outs=allow_pinch_outs,
        consulting_title_block=consulting_title_block,
    )
    request = build_section_request(
        transect_points=transect_points,
        vertical_exaggeration=vertical_exaggeration,
        show_hatches=show_hatches,
        show_legend=effective.show_legend,
        section_title=section_title,
        interpretation_mode=interpretation_mode,
        allow_pinch_outs=effective.allow_pinch_outs,
        offset_warning_m=offset_warning_m,
        uncertainty_spacing_m=uncertainty_spacing_m,
        uncertainty_offset_m=uncertainty_offset_m,
        max_offset_for_interpolation_m=max_offset_for_interpolation_m,
        show_ground_surface=effective.show_ground_surface,
        interpolate_water_table=effective.interpolate_water_table,
        warn_on_correlation_gaps=warn_on_correlation_gaps,
        show_water_elevation_labels=show_water_elevation_labels,
        show_water_legend=show_water_legend,
        show_dry_well_nm=show_dry_well_nm,
        water_interpolate_across_gaps=water_interpolate_across_gaps,
        environmental_parameters=environmental_parameters,
        show_parameter_labels=show_parameter_labels,
        parameter_interpolate_segments=parameter_interpolate_segments,
        parameter_interpolate_across_gaps=parameter_interpolate_across_gaps,
        render_layout=effective.layout,
        track_width_m=effective.track_width_m,
        coordinate_reference=coordinate_reference,
        uses_placeholder_elevation=uses_placeholder_elevation,
        elevation_mode=elevation_mode,
        consulting_title_block=effective.consulting_title_block,
        fail_on_overlaps=fail_on_overlaps,
    )
    return request, request.cache_key(active_hole_ids)


def sidebar_render_cache_key(
    parse_result: ParseResult,
    transect_mode: str,
    selected_holes: list[str],
    coordinate_text: str,
    request: SectionBuildRequest,
    offset_warning_m: float,
    selection: tuple[tuple[str, ...], tuple[tuple[float, float], ...]] | None = None,
) -> str | None:
    if selection is None:
        selection = _active_transect_selection(
            parse_result,
            transect_mode,
            selected_holes,
            coordinate_text,
            offset_warning_m,
        )
    if selection is None:
        return None
    active_hole_ids, _ = selection
    return request.cache_key(active_hole_ids)


def generate_cross_section(
    parse_result: ParseResult,
    transect_points: list[tuple[float, float]],
    hole_ids: tuple[str, ...],
    request: SectionBuildRequest,
    offset_warning_m: float,
    *,
    lithology_index: dict[str, tuple[Lithology, ...]] | None = None,
) -> tuple[bytes, bytes, bytes, int, list[str], list[str]]:
    subset = subset_parse_result(
        parse_result,
        hole_ids,
        lithology_index=lithology_index,
    )
    if len(subset.collars) < 2:
        raise ValueError("Select at least two boreholes with collar and lithology data")
    if not subset.lithologies:
        raise ValueError("Selected boreholes have no lithology intervals")
    missing_lithology = holes_missing_lithology(subset.lithologies, hole_ids)
    if missing_lithology:
        raise ValueError(
            "Missing lithology for hole(s): " + ", ".join(missing_lithology)
        )

    transect = Transect(points=transect_points)
    warnings = off_transect_warnings(subset.collars, transect, offset_warning_m)
    if request.render_layout in {"consulting_section", "section_sheet"}:
        warnings = list(warnings) + screen_interval_warnings(hole_ids, subset.screen_intervals)
    if warnings:
        for message in dedupe_messages(warnings):
            st.warning(message)

    build_request = request.model_copy(
        update={
            "transect_points": tuple(transect_points),
            "correlation_overrides": _session_correlation_overrides()
            + tuple(subset.correlation_overrides),
            "water_levels": subset.water_levels,
            "screen_intervals": subset.screen_intervals,
            "vertical_gradients": subset.vertical_gradients,
            "faults": subset.faults,
            "unconformities": subset.unconformities,
            "environmental_readings": subset.environmental_readings,
            "deviation_readings": subset.deviation_readings,
        }
    )
    subset_json = subset.model_dump_json()
    request_json = build_request.model_dump_json()
    st.session_state.section_build_subset_json = subset_json
    st.session_state.section_build_request_json = request_json
    svg_bytes, png_bytes, pdf_bytes, polygon_count, lithology_codes, overlap_warnings = cached_build_section(
        subset_json,
        request_json,
    )
    return svg_bytes, png_bytes, pdf_bytes, polygon_count, list(lithology_codes), list(dedupe_messages(overlap_warnings))
