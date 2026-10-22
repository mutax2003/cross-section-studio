"""Cross-section build pipeline: projection → stratigraphy → SVG."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import get_args

import numpy as np
import pandas as pd

from ai_quality import (
    chemistry_threshold_inversion_warnings,
    implausible_environmental_reading,
    implausible_screen_interval,
    implausible_water_reading,
)
from constants import (
    BOREHOLE_ONLY_DISCLAIMER,
    CHEMISTRY_OVERLAY_DISCLAIMER,
    CORRELATION_LINES_DISCLAIMER,
    INTERPOLATED_DISCLAIMER,
    WATER_TABLE_OVERLAY_DISCLAIMER,
)
from export_framing import ExportFramingConfig, merge_framing_into_profile_updates
from hydro_metrics import (
    format_vertical_gradient_note,
    latest_environmental_readings,
    nested_vertical_gradients,
    vertical_gradient_arrow_conflicts,
)
from hydro_metrics import groundwater_flow_notes as _groundwater_flow_notes
from lithology_codes import collect_lithology_codes
from models import (
    _XML_ILLEGAL,
    MAX_WATER_SERIES,
    Collar,
    ConsultingTitleBlock,
    CorrelationOverride,
    DeviationReading,
    EnvironmentalReading,
    Fault,
    InterpretationMode,
    Lithology,
    RasterLogStrip,
    ScreenInterval,
    SectionFigureMetadata,
    Transect,
    Unconformity,
    VerticalGradient,
    WaterLevel,
)
from projection import (
    DEFAULT_OFFSET_WARNING_M,
    AlongHoleTVD,
    build_along_hole_tvd,
    project_boreholes,
    transect_azimuth_deg,
)
from render_lock import RENDER_LOCK
from render_profiles import (
    ChemistryLabelStyle,
    LayoutMode,
    normalise_chemistry_thresholds,
    profile_for_layout,
    profile_with_elevation_mode,
    resolved_chemistry_thresholds,
)
from render_theme import filter_water_levels_for_plot
from renderer import CrossSectionRenderer
from stratigraphy import (
    CorrelationPairSummary,
    GeologicalPolygon,
    PolygonOverlap,
    build_stratigraphy,
    detect_polygon_overlaps,
    log_polygon_overlaps,
)

logger = logging.getLogger(__name__)

# Matplotlib keeps process-wide state; every render → export → close (and the
# later export of a retained figure) runs under this lock (see render_lock.py).
_RENDER_LOCK = RENDER_LOCK

CROSSING_CORRELATION_PREFIX = "Crossing correlation"
FENCE_NOTE_PREFIXES = (CROSSING_CORRELATION_PREFIX, "Fence clipped")


def _merge_optional_profile_updates(
    updates: dict[str, object],
    *pairs: tuple[str, object | None],
) -> None:
    """Apply ``(profile_field, value)`` pairs when value is not None."""
    for key, value in pairs:
        if value is not None:
            updates[key] = value


@dataclass(frozen=True)
class SectionGeometry:
    """Projection + stratigraphy outputs shared by render and future Prepare reuse."""

    projected: pd.DataFrame
    polygons: list[GeologicalPolygon]
    lithology_codes: list[str]
    overlap_warnings: tuple[str, ...]
    collar_depths: dict[str, float]
    overlap_pairs: tuple[PolygonOverlap, ...]
    x_span: float
    max_offset: float
    projected_hole_ids: frozenset[str]
    stick_up_by_hole: dict[str, float] = field(default_factory=dict)
    correlation_summaries: tuple[CorrelationPairSummary, ...] = ()
    # Surveyed (deviated) holes only: sampled MD→TVD tables so water heads,
    # screens, chemistry samples and well bottoms use the same true vertical
    # depth as the lithology. Absent holes are vertical (collar − depth).
    along_hole_tvd: dict[str, AlongHoleTVD] = field(default_factory=dict)
    # Collar plan coordinates (easting, northing) of projected holes: pairs
    # co-located / nested wells for the vertical-gradient QA note.
    collar_plan_xy: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class CrossSectionResult:
    """Canonical return value from ``build_cross_section``."""

    projected: pd.DataFrame
    polygons: list[GeologicalPolygon]
    svg_bytes: bytes
    png_bytes: bytes
    pdf_bytes: bytes
    lithology_codes: list[str]
    overlap_warnings: tuple[str, ...]
    retained_export_bundle: dict[str, object] | None = field(default=None, repr=False, compare=False)
    # Layout QA for the exported page (e.g. a title-block map scale that does
    # not fit and printed "AS SHOWN"); not part of legacy tuple unpacking.
    qa_notes: tuple[str, ...] = ()

    def __iter__(self) -> Iterator[object]:
        """Allow legacy tuple unpacking: ``proj, polys, svg, png, pdf, codes, warns = result``."""
        yield self.projected
        yield self.polygons
        yield self.svg_bytes
        yield self.png_bytes
        yield self.pdf_bytes
        yield self.lithology_codes
        yield self.overlap_warnings


_VALID_INTERPRETATION_MODES = frozenset({"borehole_only", "interpolated", "correlation_lines"})

_SCALE_BAR_CANDIDATES = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
_DEFAULT_UNCERTAINTY_SPACING_M = 80.0

DEFAULT_UNCERTAINTY_SPACING_M = _DEFAULT_UNCERTAINTY_SPACING_M
_DEFAULT_EXPORT_FORMATS = frozenset({"svg"})
SVG_PNG_EXPORT_FORMATS = frozenset({"svg", "png"})
ALL_EXPORT_FORMATS = frozenset({"svg", "png", "pdf"})
_SUPPORTED_EXPORT_FORMATS = ALL_EXPORT_FORMATS


def _normalize_export_formats(export_formats: frozenset[str] | None) -> frozenset[str]:
    """Lowercase and drop unknown formats; fall back to SVG-only when empty."""
    if not export_formats:
        return _DEFAULT_EXPORT_FORMATS
    normalized = frozenset(str(item).strip().lower() for item in export_formats if str(item).strip())
    unknown = normalized - _SUPPORTED_EXPORT_FORMATS
    if unknown:
        logger.warning("Ignoring unsupported export format(s): %s", ", ".join(sorted(unknown)))
    kept = normalized & _SUPPORTED_EXPORT_FORMATS
    return kept or _DEFAULT_EXPORT_FORMATS


PDF_EXPORT_FORMATS = frozenset({"pdf"})


def take_retained_figure():
    """Deprecated no-op: figures are returned on ``CrossSectionResult.retained_export_bundle``."""
    return None


def take_retained_export_bundle() -> dict[str, object] | None:
    """Deprecated no-op: use ``CrossSectionResult.retained_export_bundle`` instead."""
    return None


_DISCLAIMER_BY_MODE = {
    "borehole_only": BOREHOLE_ONLY_DISCLAIMER,
    "correlation_lines": CORRELATION_LINES_DISCLAIMER,
    "interpolated": INTERPOLATED_DISCLAIMER,
}


def validate_interpretation_mode(mode: str) -> InterpretationMode:
    if mode not in _VALID_INTERPRETATION_MODES:
        allowed = ", ".join(sorted(_VALID_INTERPRETATION_MODES))
        raise ValueError(f"interpretation_mode must be one of: {allowed} (got {mode!r})")
    return mode  # type: ignore[return-value]


def auto_scale_bar_m(x_span: float) -> float:
    target = max(x_span / 5.0, 1.0)
    return min(_SCALE_BAR_CANDIDATES, key=lambda value: abs(value - target))


def _filter_projected_for_interpolation(
    projected: pd.DataFrame,
    max_offset_m: float | None,
) -> pd.DataFrame:
    """Drop whole holes whose max offset exceeds ``max_offset_m`` before fence build."""
    if max_offset_m is None or projected.empty:
        return projected
    hole_offsets = (
        projected.groupby("hole_id", sort=False)["offset_distance"]
        .max()
        .astype(float)
    )
    keep_holes = hole_offsets[hole_offsets <= max_offset_m].index
    if len(keep_holes) < 2:
        raise ValueError(
            f"Fewer than two boreholes within {max_offset_m:.0f} m of the transect for interpolation"
        )
    return projected.loc[projected["hole_id"].isin(keep_holes)]


def filter_projected_for_interpolation(
    projected: pd.DataFrame,
    max_offset_m: float | None,
) -> pd.DataFrame:
    """Drop holes farther than ``max_offset_m`` from the transect before fence build."""
    return _filter_projected_for_interpolation(projected, max_offset_m)


def compute_section_geometry(
    collars: Sequence[Collar],
    lithologies: Sequence[Lithology],
    transect_points: Sequence[tuple[float, float]],
    *,
    offset_warning_m: float = DEFAULT_OFFSET_WARNING_M,
    interpretation_mode: InterpretationMode = "interpolated",
    allow_pinch_outs: bool = True,
    detect_overlaps: bool = True,
    fail_on_overlaps: bool = False,
    max_offset_for_interpolation_m: float | None = None,
    correlation_overrides: Sequence[CorrelationOverride] = (),
    deviation_readings: Sequence[DeviationReading] = (),
    warn_on_correlation_gaps: bool = False,
) -> SectionGeometry:
    """Project boreholes, build stratigraphy, and collect overlap/correlation QA."""
    interpretation_mode = validate_interpretation_mode(interpretation_mode)
    if not math.isfinite(offset_warning_m) or offset_warning_m <= 0:
        raise ValueError("offset_warning_m must be a positive finite number")
    if len(transect_points) < 2:
        raise ValueError("At least two transect points are required")
    transect = Transect(points=list(transect_points))
    projected = project_boreholes(
        collars,
        lithologies,
        transect,
        offset_warning_m=offset_warning_m,
        deviation_readings=deviation_readings,
    )
    if projected.empty:
        raise ValueError("No lithology intervals were projected for the selected transect")

    # Borehole-only sections draw every projected log: the interpolation offset
    # limit must not reject them (it used to fail with "fewer than two holes").
    interpolation_df = (
        projected
        if interpretation_mode == "borehole_only"
        else _filter_projected_for_interpolation(projected, max_offset_for_interpolation_m)
    )

    correlation_summaries: list[CorrelationPairSummary] = []
    clip_warnings: list[str] = []
    if interpretation_mode == "borehole_only":
        polygons: list[GeologicalPolygon] = []
        overlap_pairs: tuple[PolygonOverlap, ...] = ()
    else:
        polygons = build_stratigraphy(
            interpolation_df,
            allow_pinch_outs=allow_pinch_outs,
            correlation_overrides=correlation_overrides,
            pair_summaries=correlation_summaries,
            clip_warnings=clip_warnings,
        )
        overlap_pairs = (
            tuple(detect_polygon_overlaps(polygons))
            if detect_overlaps and len(polygons) >= 2
            else ()
        )
    if overlap_pairs and logger.isEnabledFor(logging.WARNING):
        log_polygon_overlaps(overlap_pairs)
    overlap_warnings = (
        tuple(overlap.message() for overlap in overlap_pairs) if overlap_pairs else ()
    )
    correlation_warnings: list[str] = []
    if warn_on_correlation_gaps and interpretation_mode != "borehole_only":
        for summary in correlation_summaries:
            if summary.unmatched_keys_count > 0 or summary.pinch_out_candidates > 0:
                correlation_warnings.append(
                    f"Correlation gap {summary.left_hole_id}–{summary.right_hole_id}: "
                    f"{summary.matched_count} matched, "
                    f"{summary.unmatched_keys_count} unmatched, "
                    f"{summary.pinch_out_candidates} pinch-out(s)"
                )
    if correlation_warnings and logger.isEnabledFor(logging.WARNING):
        for message in correlation_warnings:
            logger.warning(message)
    # Fence-level notes the user must see (not gated on correlation-gap
    # warnings): layers logged in opposite order in neighbouring holes are
    # drawn as pinch-outs, and heavy overlap clipping is reported.
    fence_notes = [
        f"{CROSSING_CORRELATION_PREFIX} {summary.left_hole_id}–{summary.right_hole_id}: "
        f"{', '.join(summary.crossing_codes)} logged in opposite order — "
        + (
            "drawn as pinch-outs"
            if allow_pinch_outs
            else "drawn as short pinch-outs at their own holes, not joined across"
        )
        for summary in correlation_summaries
        if summary.crossing_codes
    ] + clip_warnings
    for message in fence_notes:
        logger.warning(message)
    overlap_warnings = overlap_warnings + tuple(correlation_warnings) + tuple(fence_notes)
    if fail_on_overlaps and overlap_pairs:
        raise ValueError(
            f"Polygon overlap detected ({len(overlap_pairs)} pair(s)); "
            "resolve correlation or set fail_on_overlaps=False to export."
        )

    lithology_codes = collect_lithology_codes(projected, polygons)
    projected_hole_ids = frozenset(projected["hole_id"].unique())
    collar_depths = {
        collar.hole_id: collar.total_depth
        for collar in collars
        if collar.hole_id in projected_hole_ids
    }
    stick_up_by_hole = {
        collar.hole_id: float(collar.stick_up_m)
        for collar in collars
        if collar.hole_id in projected_hole_ids and collar.stick_up_m
    }
    along_hole_tvd = build_along_hole_tvd(collars, deviation_readings, projected_hole_ids)
    x_span = float(projected["x_profile"].max() - projected["x_profile"].min()) if not projected.empty else 1.0
    max_offset = float(projected["offset_distance"].max()) if not projected.empty else 0.0
    return SectionGeometry(
        projected=projected,
        polygons=polygons,
        lithology_codes=lithology_codes,
        overlap_warnings=overlap_warnings,
        collar_depths=collar_depths,
        overlap_pairs=overlap_pairs,
        x_span=x_span,
        max_offset=max_offset,
        projected_hole_ids=projected_hole_ids,
        stick_up_by_hole=stick_up_by_hole,
        correlation_summaries=tuple(correlation_summaries),
        along_hole_tvd=along_hole_tvd,
        collar_plan_xy={
            collar.hole_id: (float(collar.easting), float(collar.northing))
            for collar in collars
            if collar.hole_id in projected_hole_ids
        },
    )



def _check_vertical_exaggeration(vertical_exaggeration: float | None) -> None:
    """``None`` = auto (fit page, caption prints the measured VE); else exact VE."""
    if vertical_exaggeration is None:
        return
    if not math.isfinite(vertical_exaggeration) or vertical_exaggeration <= 0:
        raise ValueError("vertical_exaggeration must be a positive finite number (or None for auto)")


def _check_render_layout(render_layout: str) -> None:
    """Reject unknown layouts; profile_for_layout would silently fall back."""
    if render_layout not in get_args(LayoutMode):
        raise ValueError(
            f"render_layout must be one of {get_args(LayoutMode)} (got {render_layout!r})"
        )


def section_groundwater_flow_notes(
    collars: Sequence[Collar],
    water_levels: Sequence[WaterLevel] | None,
    transect_points: Sequence[tuple[float, float]],
    *,
    deviation_readings: Sequence[DeviationReading] | None = None,
    selected_water_series_ids: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Plan-view flow QA lines (planar head fit) for the plotted water series.

    Pass every collar / water reading in the workbook — the fit uses all
    measured wells, not only those on the section. Section bearing is the
    overall start→end azimuth of the line (polylines included). Cosmetic /
    QA only: never part of ``geometry_cache_payload``.
    """
    if not water_levels:
        return ()
    plotted = filter_water_levels_for_plot(
        water_levels,
        tuple(selected_water_series_ids) if selected_water_series_ids else None,
        max_series=MAX_WATER_SERIES,
    )
    # Readings far outside their hole are left off the figure; keep them out
    # of the head-plane fit too (one 3200 m depth would swing the flow arrow).
    collar_by_id = {collar.hole_id: collar for collar in collars}
    plotted = tuple(
        level
        for level in plotted
        if level.hole_id not in collar_by_id
        or implausible_water_reading(
            level,
            collar_rl=collar_by_id[level.hole_id].elevation,
            total_depth=collar_by_id[level.hole_id].total_depth,
            stick_up_m=collar_by_id[level.hole_id].stick_up_m,
        )
        is None
    )
    if not plotted:
        return ()
    try:
        return _groundwater_flow_notes(
            collars,
            plotted,
            transect_azimuth_deg(transect_points),
            deviation_readings=deviation_readings,
        )
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        logger.warning("Groundwater flow fit skipped: %s", exc)
        return ()


def build_cross_section(
    collars: Sequence[Collar],
    lithologies: Sequence[Lithology],
    transect_points: Sequence[tuple[float, float]],
    *,
    vertical_exaggeration: float | None = None,
    show_hatches: bool = True,
    show_legend: bool = True,
    title: str = "Borehole Cross-Section",
    offset_warning_m: float = DEFAULT_OFFSET_WARNING_M,
    interpretation_mode: InterpretationMode = "interpolated",
    allow_pinch_outs: bool = True,
    detect_overlaps: bool = True,
    fail_on_overlaps: bool = False,
    water_levels: Sequence[WaterLevel] | None = None,
    uncertainty_spacing_m: float = _DEFAULT_UNCERTAINTY_SPACING_M,
    uncertainty_offset_m: float = DEFAULT_OFFSET_WARNING_M,
    max_offset_for_interpolation_m: float | None = None,
    correlation_overrides: Sequence[CorrelationOverride] = (),
    faults: Sequence[Fault] = (),
    unconformities: Sequence[Unconformity] = (),
    environmental_readings: Sequence[EnvironmentalReading] = (),
    deviation_readings: Sequence[DeviationReading] = (),
    figure_metadata: SectionFigureMetadata | None = None,
    show_ground_surface: bool = True,
    interpolate_water_table: bool = False,
    warn_on_correlation_gaps: bool = False,
    show_water_elevation_labels: bool | None = None,
    show_water_legend: bool | None = None,
    show_dry_well_nm: bool | None = None,
    water_interpolate_across_gaps: bool | None = None,
    environmental_parameters: Sequence[str] | None = None,
    show_parameter_labels: bool | None = None,
    parameter_interpolate_segments: bool | None = None,
    parameter_interpolate_across_gaps: bool | None = None,
    parameter_draw_markers: bool | None = None,
    parameter_marker_size: float | None = None,
    parameter_draw_leaders: bool | None = None,
    parameter_label_include_units: bool | None = None,
    column_header_detail: str | None = None,
    show_scale_bar: bool | None = None,
    show_ve_annotation: bool | None = None,
    show_parameter_legend_text: bool | None = None,
    export_font_family: str | None = None,
    export_font_size: float | None = None,
    selected_water_series_ids: Sequence[str] | None = None,
    water_line_solid: bool | None = None,
    legend_ncol: int | None = None,
    chemistry_color_mode: str | None = None,
    chemistry_threshold_green_max: float | None = None,
    chemistry_threshold_yellow_max: float | None = None,
    chemistry_thresholds: Mapping[str, tuple[float | None, float | None]] | None = None,
    chemistry_label_style: ChemistryLabelStyle | None = None,
    render_layout: str = "section_sheet",
    track_width_m: float = 3.0,
    auto_fit_track_width: bool = True,
    elevation_mode: str = "absolute",
    raster_log_strips: Sequence[RasterLogStrip] = (),
    export_formats: frozenset[str] | None = None,
    consulting_title_block: ConsultingTitleBlock | None = None,
    screen_intervals: Sequence[ScreenInterval] | None = None,
    vertical_gradients: Sequence[VerticalGradient] | None = None,
    export_framing: ExportFramingConfig | None = None,
    groundwater_flow_notes: Sequence[str] | None = None,
) -> CrossSectionResult:
    """Project, build stratigraphy, render. Returns ``CrossSectionResult`` (also unpackable as a 7-tuple).

    ``vertical_exaggeration``: ``None`` (default) = auto — the section fills the
    page and every VE caption prints the true VE measured after final layout;
    a number draws exactly that VE (plot box locked, centred in its frame).

    ``groundwater_flow_notes``: ``None`` (default) fits a head plane through
    every measured well in ``collars`` / ``water_levels`` (≥3 non-collinear
    wells per series) and adds the plan-view flow line to the QA notes; pass
    ``()`` to suppress or precomputed lines (e.g. from the full workbook).
    """
    export_formats = _normalize_export_formats(export_formats)
    interpretation_mode = validate_interpretation_mode(interpretation_mode)
    _check_vertical_exaggeration(vertical_exaggeration)
    if not math.isfinite(uncertainty_spacing_m) or uncertainty_spacing_m <= 0:
        raise ValueError("uncertainty_spacing_m must be a positive finite number")
    if not math.isfinite(uncertainty_offset_m) or uncertainty_offset_m <= 0:
        raise ValueError("uncertainty_offset_m must be a positive finite number")
    if not math.isfinite(track_width_m) or track_width_m <= 0:
        raise ValueError("track_width_m must be a positive finite number")
    _check_render_layout(render_layout)
    if groundwater_flow_notes is None:
        groundwater_flow_notes = section_groundwater_flow_notes(
            collars,
            water_levels,
            transect_points,
            deviation_readings=deviation_readings,
            selected_water_series_ids=selected_water_series_ids,
        )
    geometry = compute_section_geometry(
        collars,
        lithologies,
        transect_points,
        offset_warning_m=offset_warning_m,
        interpretation_mode=interpretation_mode,
        allow_pinch_outs=allow_pinch_outs,
        detect_overlaps=detect_overlaps,
        fail_on_overlaps=fail_on_overlaps,
        max_offset_for_interpolation_m=max_offset_for_interpolation_m,
        correlation_overrides=correlation_overrides,
        deviation_readings=deviation_readings,
        warn_on_correlation_gaps=warn_on_correlation_gaps,
    )
    return render_cross_section_from_geometry(
        geometry,
        transect_points,
        vertical_exaggeration=vertical_exaggeration,
        show_hatches=show_hatches,
        show_legend=show_legend,
        title=title,
        interpretation_mode=interpretation_mode,
        water_levels=water_levels,
        uncertainty_spacing_m=uncertainty_spacing_m,
        uncertainty_offset_m=uncertainty_offset_m,
        faults=faults,
        unconformities=unconformities,
        environmental_readings=environmental_readings,
        figure_metadata=figure_metadata,
        show_ground_surface=show_ground_surface,
        interpolate_water_table=interpolate_water_table,
        show_water_elevation_labels=show_water_elevation_labels,
        show_water_legend=show_water_legend,
        show_dry_well_nm=show_dry_well_nm,
        water_interpolate_across_gaps=water_interpolate_across_gaps,
        environmental_parameters=environmental_parameters,
        show_parameter_labels=show_parameter_labels,
        parameter_interpolate_segments=parameter_interpolate_segments,
        parameter_interpolate_across_gaps=parameter_interpolate_across_gaps,
        parameter_draw_markers=parameter_draw_markers,
        parameter_marker_size=parameter_marker_size,
        parameter_draw_leaders=parameter_draw_leaders,
        parameter_label_include_units=parameter_label_include_units,
        column_header_detail=column_header_detail,
        show_scale_bar=show_scale_bar,
        show_ve_annotation=show_ve_annotation,
        show_parameter_legend_text=show_parameter_legend_text,
        export_font_family=export_font_family,
        export_font_size=export_font_size,
        selected_water_series_ids=selected_water_series_ids,
        water_line_solid=water_line_solid,
        legend_ncol=legend_ncol,
        chemistry_color_mode=chemistry_color_mode,
        chemistry_threshold_green_max=chemistry_threshold_green_max,
        chemistry_threshold_yellow_max=chemistry_threshold_yellow_max,
        chemistry_thresholds=chemistry_thresholds,
        chemistry_label_style=chemistry_label_style,
        render_layout=render_layout,
        track_width_m=track_width_m,
        auto_fit_track_width=auto_fit_track_width,
        elevation_mode=elevation_mode,
        raster_log_strips=raster_log_strips,
        export_formats=export_formats,
        consulting_title_block=consulting_title_block,
        screen_intervals=screen_intervals,
        vertical_gradients=vertical_gradients,
        export_framing=export_framing,
        groundwater_flow_notes=groundwater_flow_notes,
    )


@dataclass(frozen=True)
class OverlayExclusions:
    """Overlay readings far outside their hole, left off the figure (with why).

    Keyed by ``id()`` of the reading objects passed in, so callers can drop
    exactly those objects from any list derived from the same inputs.
    """

    water: dict[int, str] = field(default_factory=dict)
    screens: dict[int, str] = field(default_factory=dict)
    environmental: dict[int, str] = field(default_factory=dict)


def _section_hole_extents(geometry: SectionGeometry) -> dict[str, tuple[float, float]]:
    """``hole_id -> (collar RL, total depth)`` for the projected holes."""
    projected = geometry.projected
    rl_by_hole: dict[str, float] = {}
    if not projected.empty and "collar_elevation" in projected.columns:
        for hole_id, rl in zip(projected["hole_id"], projected["collar_elevation"], strict=True):
            rl_by_hole.setdefault(str(hole_id), float(rl))
    return {
        hole_id: (rl_by_hole[hole_id], float(td))
        for hole_id, td in geometry.collar_depths.items()
        if hole_id in rl_by_hole
    }


def find_implausible_overlays(
    geometry: SectionGeometry,
    *,
    water_levels: Sequence[WaterLevel] = (),
    screen_intervals: Sequence[ScreenInterval] = (),
    environmental_readings: Sequence[EnvironmentalReading] = (),
) -> OverlayExclusions:
    """Water / screen / chemistry readings lying far outside their hole.

    One mis-typed depth (3200 for 3.2 m) used to stretch the depth axis until
    the section was a sliver, with no QA note. Limits (``ai_quality``): below
    ground past TD + max(5 m, 25 % of TD), or a head more than 30 m above
    ground. Holes not on the section are not checked (nothing is drawn).
    """
    extents = _section_hole_extents(geometry)
    exclusions = OverlayExclusions()
    for level in water_levels:
        extent = extents.get(level.hole_id)
        if extent is None:
            continue
        reason = implausible_water_reading(
            level,
            collar_rl=extent[0],
            total_depth=extent[1],
            stick_up_m=geometry.stick_up_by_hole.get(level.hole_id),
        )
        if reason is not None:
            exclusions.water[id(level)] = reason
    for screen in screen_intervals:
        extent = extents.get(screen.hole_id)
        if extent is None:
            continue
        reason = implausible_screen_interval(screen, total_depth=extent[1])
        if reason is not None:
            exclusions.screens[id(screen)] = reason
    for reading in environmental_readings:
        extent = extents.get(reading.hole_id)
        if extent is None:
            continue
        reason = implausible_environmental_reading(reading, total_depth=extent[1])
        if reason is not None:
            exclusions.environmental[id(reading)] = reason
    return exclusions


# The per-sample "N sampling dates, latest … shown" notes: a large dated lab
# table produced hundreds of figure QA lines. Beyond this many, one summary.
MAX_CHEMISTRY_DATE_NOTES = 10


def _cap_notes(notes: Sequence[str], limit: int, summary: str) -> tuple[str, ...]:
    """First ``limit`` notes, then ``summary`` formatted with the remainder count."""
    if len(notes) <= limit:
        return tuple(notes)
    return tuple(notes[:limit]) + (summary.format(count=len(notes) - limit),)


def _latest_chemistry_samples(
    readings: Sequence[EnvironmentalReading],
    hole_ids: Sequence[str],
    parameters: Sequence[str] | None,
) -> tuple[tuple[EnvironmentalReading, ...], tuple[str, ...]]:
    """Latest dated sample per hole/parameter/depth, plus QA notes for the drawn ones.

    Notes are limited to holes on the section and the selected parameters,
    and capped at ``MAX_CHEMISTRY_DATE_NOTES`` lines plus an "and N more"
    summary; without dates (or with undated duplicates only) readings pass
    unchanged.
    """
    if not readings:
        return tuple(readings), ()
    kept, _all_notes = latest_environmental_readings(readings)
    # Case-insensitive, like the sample grouping (environmental_sample_key).
    active = {name.strip().casefold() for name in (parameters or ()) if name.strip()}
    if not active:
        return kept, ()
    on_section = set(hole_ids)
    _drawn, notes = latest_environmental_readings(
        [
            r
            for r in readings
            if r.hole_id in on_section and r.parameter.strip().casefold() in active
        ]
    )
    notes = _cap_notes(
        notes,
        MAX_CHEMISTRY_DATE_NOTES,
        "…and {count} more sample points with several sampling dates (latest date shown "
        "for each; see Validate → Environmental / Lab)",
    )
    return kept, notes


def vertical_gradient_qa_notes(
    geometry: SectionGeometry,
    water_levels: Sequence[WaterLevel],
    screen_intervals: Sequence[ScreenInterval],
    vertical_gradients: Sequence[VerticalGradient] = (),
) -> tuple[str, ...]:
    """QA lines: computed iv per well nest + arrows contradicting the heads.

    Uses the section's collars, stick-ups and hole surveys so the heads and
    screen mid-points match the figure (``hydro_metrics`` conventions:
    iv > 0 downward / recharge, iv < 0 upward / discharge).
    """
    if not water_levels or not screen_intervals:
        return ()
    hole_ids = geometry.projected_hole_ids
    projected = geometry.projected
    collar_rl = {
        str(hole_id): float(rl)
        for hole_id, rl in zip(projected["hole_id"], projected["collar_elevation"], strict=True)
    }
    md_maps = {
        hole_id: (lambda md, _t=table, _rl=collar_rl[hole_id]: _t.elevation_at(_rl, md))
        for hole_id, table in geometry.along_hole_tvd.items()
        if hole_id in collar_rl
    }
    analysis = nested_vertical_gradients(
        [level for level in water_levels if level.hole_id in hole_ids],
        [screen for screen in screen_intervals if screen.hole_id in hole_ids],
        collar_rl_by_hole=collar_rl,
        plan_xy_by_hole=geometry.collar_plan_xy,
        stick_up_by_hole=geometry.stick_up_by_hole,
        md_to_elevation_by_hole=md_maps,
    )
    notes = [format_vertical_gradient_note(gradient) for gradient in analysis.gradients]
    notes.extend(
        vertical_gradient_arrow_conflicts(
            [arrow for arrow in vertical_gradients if arrow.hole_id in hole_ids],
            analysis.gradients,
        )
    )
    return tuple(notes)


def render_cross_section_from_geometry(
    geometry: SectionGeometry,
    transect_points: Sequence[tuple[float, float]],
    *,
    vertical_exaggeration: float | None = None,
    show_hatches: bool = True,
    show_legend: bool = True,
    title: str = "Borehole Cross-Section",
    interpretation_mode: InterpretationMode = "interpolated",
    water_levels: Sequence[WaterLevel] | None = None,
    uncertainty_spacing_m: float = _DEFAULT_UNCERTAINTY_SPACING_M,
    uncertainty_offset_m: float = DEFAULT_OFFSET_WARNING_M,
    faults: Sequence[Fault] = (),
    unconformities: Sequence[Unconformity] = (),
    environmental_readings: Sequence[EnvironmentalReading] = (),
    figure_metadata: SectionFigureMetadata | None = None,
    show_ground_surface: bool = True,
    interpolate_water_table: bool = False,
    show_water_elevation_labels: bool | None = None,
    show_water_legend: bool | None = None,
    show_dry_well_nm: bool | None = None,
    water_interpolate_across_gaps: bool | None = None,
    environmental_parameters: Sequence[str] | None = None,
    show_parameter_labels: bool | None = None,
    parameter_interpolate_segments: bool | None = None,
    parameter_interpolate_across_gaps: bool | None = None,
    parameter_draw_markers: bool | None = None,
    parameter_marker_size: float | None = None,
    parameter_draw_leaders: bool | None = None,
    parameter_label_include_units: bool | None = None,
    column_header_detail: str | None = None,
    show_scale_bar: bool | None = None,
    show_ve_annotation: bool | None = None,
    show_parameter_legend_text: bool | None = None,
    export_font_family: str | None = None,
    export_font_size: float | None = None,
    selected_water_series_ids: Sequence[str] | None = None,
    water_line_solid: bool | None = None,
    legend_ncol: int | None = None,
    chemistry_color_mode: str | None = None,
    chemistry_threshold_green_max: float | None = None,
    chemistry_threshold_yellow_max: float | None = None,
    chemistry_thresholds: Mapping[str, tuple[float | None, float | None]] | None = None,
    chemistry_label_style: ChemistryLabelStyle | None = None,
    render_layout: str = "section_sheet",
    track_width_m: float = 3.0,
    auto_fit_track_width: bool = True,
    elevation_mode: str = "absolute",
    raster_log_strips: Sequence[RasterLogStrip] = (),
    export_formats: frozenset[str] | None = None,
    consulting_title_block: ConsultingTitleBlock | None = None,
    screen_intervals: Sequence[ScreenInterval] | None = None,
    vertical_gradients: Sequence[VerticalGradient] | None = None,
    export_framing: ExportFramingConfig | None = None,
    close_figure: bool = True,
    groundwater_flow_notes: Sequence[str] = (),
) -> CrossSectionResult:
    """Render/export from precomputed ``SectionGeometry`` (Prepare can reuse Generate geometry).

    When ``close_figure`` is False, the matplotlib figure is left open for a follow-up
    encode pass (caller must ``plt.close``). Used by Streamlit Generate→Prepare one-draw.
    """
    export_formats = _normalize_export_formats(export_formats)
    interpretation_mode = validate_interpretation_mode(interpretation_mode)
    _check_vertical_exaggeration(vertical_exaggeration)
    if not math.isfinite(uncertainty_spacing_m) or uncertainty_spacing_m <= 0:
        raise ValueError("uncertainty_spacing_m must be a positive finite number")
    if not math.isfinite(uncertainty_offset_m) or uncertainty_offset_m <= 0:
        raise ValueError("uncertainty_offset_m must be a positive finite number")
    if not math.isfinite(track_width_m) or track_width_m <= 0:
        raise ValueError("track_width_m must be a positive finite number")

    projected = geometry.projected
    polygons = geometry.polygons
    lithology_codes = geometry.lithology_codes
    overlap_warnings = geometry.overlap_warnings
    collar_depths = geometry.collar_depths
    overlap_pairs = geometry.overlap_pairs
    x_span = geometry.x_span
    max_offset = geometry.max_offset
    projected_hole_ids = geometry.projected_hole_ids
    stick_up_by_hole = geometry.stick_up_by_hole
    disclaimer = _DISCLAIMER_BY_MODE[interpretation_mode]

    # Overlay readings far outside their hole (mm typed as m ...) are left off
    # the figure, before anything sizes the axes, and named in the QA notes.
    exclusions = find_implausible_overlays(
        geometry,
        water_levels=water_levels or (),
        screen_intervals=screen_intervals or (),
        environmental_readings=environmental_readings or (),
    )
    selected_water = filter_water_levels_for_plot(
        water_levels or (),
        tuple(selected_water_series_ids) if selected_water_series_ids else None,
        max_series=MAX_WATER_SERIES,
    )
    excluded_water_notes = tuple(
        exclusions.water[id(level)] for level in selected_water if id(level) in exclusions.water
    )
    plotted_water = tuple(level for level in selected_water if id(level) not in exclusions.water)
    if water_levels and exclusions.water:
        water_levels = [level for level in water_levels if id(level) not in exclusions.water]
    excluded_screen_notes = tuple(
        exclusions.screens[id(screen)]
        for screen in screen_intervals or ()
        if id(screen) in exclusions.screens
    )
    if screen_intervals and exclusions.screens:
        screen_intervals = [s for s in screen_intervals if id(s) not in exclusions.screens]
    active_parameters = {
        name.strip().casefold() for name in (environmental_parameters or ()) if name.strip()
    }
    excluded_chemistry_notes = tuple(
        exclusions.environmental[id(reading)]
        for reading in environmental_readings or ()
        if id(reading) in exclusions.environmental
        and reading.parameter.strip().casefold() in active_parameters
    )
    if environmental_readings and exclusions.environmental:
        environmental_readings = [
            r for r in environmental_readings if id(r) not in exclusions.environmental
        ]
    environmental_readings, chemistry_date_notes = _latest_chemistry_samples(
        environmental_readings or (), projected_hole_ids, environmental_parameters
    )

    metadata = figure_metadata or SectionFigureMetadata(
        vertical_exaggeration=vertical_exaggeration,
        transect_azimuth_deg=transect_azimuth_deg(transect_points),
        hole_count=len(projected_hole_ids),
        max_offset_m=max_offset,
    )

    _check_render_layout(render_layout)
    base_profile = profile_for_layout(render_layout)  # type: ignore[arg-type]
    profile_updates: dict[str, object] = {
        "show_ground_surface": show_ground_surface,
        "track_width_m": track_width_m,
        "auto_fit_track_width": auto_fit_track_width,
    }
    _merge_optional_profile_updates(
        profile_updates,
        ("show_water_elevation_labels", show_water_elevation_labels),
        ("show_water_legend", show_water_legend),
        ("show_dry_well_nm", show_dry_well_nm),
        ("water_interpolate_across_gaps", water_interpolate_across_gaps),
        ("show_parameter_labels", show_parameter_labels),
        ("parameter_interpolate_segments", parameter_interpolate_segments),
        ("parameter_interpolate_across_gaps", parameter_interpolate_across_gaps),
        ("parameter_draw_markers", parameter_draw_markers),
        ("parameter_marker_size", parameter_marker_size),
        ("parameter_draw_leaders", parameter_draw_leaders),
        ("parameter_label_include_units", parameter_label_include_units),
        ("column_header_detail", column_header_detail),
        ("show_scale_bar", show_scale_bar),
        ("show_ve_annotation", show_ve_annotation),
        ("show_parameter_legend_text", show_parameter_legend_text),
        ("export_font_family", export_font_family),
        ("export_font_size", export_font_size),
        ("water_line_solid", water_line_solid),
        ("legend_ncol", legend_ncol),
        ("chemistry_color_mode", chemistry_color_mode),
        ("chemistry_threshold_green_max", chemistry_threshold_green_max),
        ("chemistry_threshold_yellow_max", chemistry_threshold_yellow_max),
        ("chemistry_label_style", chemistry_label_style),
    )
    if chemistry_thresholds:
        # Per-parameter limits (parameter -> (green_max, orange_max)); the
        # sheet-wide pair above fills any limit a parameter leaves blank.
        profile_updates["chemistry_thresholds"] = normalise_chemistry_thresholds(
            chemistry_thresholds
        )
    if show_water_legend is False:
        # The sidebar "Show groundwater legend" switch was turned off: no
        # per-series water key either, even with several series.
        profile_updates["water_series_key_min_series"] = 0
    if chemistry_label_style is not None and chemistry_label_style not in get_args(ChemistryLabelStyle):
        raise ValueError(
            f"chemistry_label_style must be one of {get_args(ChemistryLabelStyle)} "
            f"(got {chemistry_label_style!r})"
        )
    if environmental_parameters:
        profile_updates["show_parameter_markers"] = True
    if render_layout == "consulting_section" and interpretation_mode == "borehole_only":
        profile_updates["show_track_lithology"] = True
        if parameter_draw_markers is None:
            profile_updates["parameter_draw_markers"] = False
        if show_dry_well_nm is None and not plotted_water:
            profile_updates["show_dry_well_nm"] = False
    profile_updates = merge_framing_into_profile_updates(export_framing, profile_updates)
    render_profile = profile_with_elevation_mode(base_profile, elevation_mode).model_copy(
        update=profile_updates
    )

    if render_profile.chemistry_color_mode == "threshold" and (
        render_profile.chemistry_threshold_green_max is None
        or render_profile.chemistry_threshold_yellow_max is None
    ):
        # Threshold mode without limits used to print black values beside
        # series-red dots; apply the Configure step's default bands instead.
        green_max, yellow_max = resolved_chemistry_thresholds(
            render_profile.chemistry_threshold_green_max,
            render_profile.chemistry_threshold_yellow_max,
        )
        if not render_profile.chemistry_thresholds:
            # With per-parameter limits the sheet-wide pair is only a fallback,
            # so missing values there are expected, not a misconfiguration.
            if math.isnan(green_max) or math.isnan(yellow_max):
                logger.info(
                    "chemistry_color_mode='threshold' without limits: values print black "
                    "(no limit set)."
                )
            else:
                logger.warning(
                    "chemistry_color_mode='threshold' without both limits: using green <= %g, "
                    "orange <= %g, red above.",
                    green_max,
                    yellow_max,
                )
        render_profile = render_profile.model_copy(
            update={
                "chemistry_threshold_green_max": green_max,
                "chemistry_threshold_yellow_max": yellow_max,
            }
        )

    effective_show_legend = show_legend
    effective_interpolate_wt = interpolate_water_table
    if render_profile.legend_in_title_block:
        effective_show_legend = False
    if export_framing is not None and not export_framing.include_legend:
        effective_show_legend = False
    if render_profile.interpolate_water_table_default:
        effective_interpolate_wt = True
    if export_framing is not None and not export_framing.include_water_table:
        effective_interpolate_wt = False
        plotted_water = ()
    # Heads as plotted (selected series), before framing drops the water layer:
    # the vertical-gradient QA note describes the data, not the drawing.
    water_levels_for_gradients = tuple(plotted_water)
    overlay_notes: list[str] = []
    if plotted_water:
        overlay_notes.append(WATER_TABLE_OVERLAY_DISCLAIMER)
    chemistry_plotted = bool(environmental_parameters) and bool(
        environmental_readings or ()
    )
    if chemistry_plotted and (
        render_profile.parameter_interpolate_segments
        or render_profile.parameter_interpolate_across_gaps
    ):
        overlay_notes.append(CHEMISTRY_OVERLAY_DISCLAIMER)
    if overlay_notes:
        disclaimer = f"{disclaimer} {' '.join(overlay_notes)}"
    effective_consulting_block = consulting_title_block
    if export_framing is not None and (
        export_framing.fence_only or not export_framing.include_title_block
    ):
        effective_consulting_block = None
    # Plan-view flow (planar head fit) only matters when water is drawn; the
    # lines are QA text, never geometry (post-cache, outside the cache key).
    flow_notes = tuple(groundwater_flow_notes or ()) if plotted_water else ()
    if not chemistry_plotted:
        chemistry_date_notes = ()
    # Nested-well vertical gradients (screen mid-points) and arrow cross-check:
    # data QA, so they reach the PDF QA lines as well as the app warnings.
    gradient_notes = vertical_gradient_qa_notes(
        geometry,
        water_levels_for_gradients,
        screen_intervals or (),
        vertical_gradients or (),
    )
    if export_framing is not None and not export_framing.include_water_table:
        excluded_water_notes = ()
    if export_framing is not None and not export_framing.include_screens:
        excluded_screen_notes = ()
    threshold_notes: tuple[str, ...] = ()
    if chemistry_plotted and render_profile.chemistry_color_mode == "threshold":
        # Configure / workbook limits entered upside down: the bands are clamped
        # (no orange band), so say so on the figure's QA lines.
        threshold_notes = tuple(
            chemistry_threshold_inversion_warnings(
                {
                    name: pair
                    for name, pair in (render_profile.chemistry_thresholds or {}).items()
                    if str(name).strip().casefold() in active_parameters
                },
                green_max=chemistry_threshold_green_max,
                orange_max=chemistry_threshold_yellow_max,
            )
        )
    data_notes = (
        excluded_water_notes + excluded_screen_notes + excluded_chemistry_notes + threshold_notes
    )
    qa_lines = (
        tuple(overlap_warnings) + data_notes + gradient_notes + flow_notes + chemistry_date_notes
    )
    if export_framing is not None and not export_framing.include_qa_footer:
        qa_lines = ()
    # Scrub XML-1.0-illegal control characters so the SVG backend cannot emit
    # unparseable documents (hole_ids are scrubbed at model validation).
    title = _XML_ILLEGAL.sub("", title)
    disclaimer = _XML_ILLEGAL.sub("", disclaimer)
    renderer = CrossSectionRenderer(
        vertical_exaggeration=vertical_exaggeration,
        scale_bar_length_m=auto_scale_bar_m(x_span),
        show_hatches=show_hatches,
        show_legend=effective_show_legend,
        title=title,
        disclaimer=disclaimer,
        interpretation_mode=interpretation_mode,
        uncertainty_spacing_m=uncertainty_spacing_m,
        uncertainty_offset_m=uncertainty_offset_m,
        overlap_pairs=overlap_pairs,
        show_ground_surface=show_ground_surface,
        interpolate_water_table=effective_interpolate_wt,
        figure_metadata=metadata,
        faults=faults,
        unconformities=unconformities,
        environmental_readings=environmental_readings,
        environmental_parameters=tuple(environmental_parameters or ()),
        render_profile=render_profile,
        raster_log_strips=raster_log_strips,
        consulting_title_block=effective_consulting_block,
        screen_intervals=(
            ()
            if export_framing is not None and not export_framing.include_screens
            else (screen_intervals or ())
        ),
        vertical_gradients=vertical_gradients or (),
        stick_up_by_hole=stick_up_by_hole,
        along_hole_tvd=geometry.along_hole_tvd,
        export_framing=export_framing,
    )
    with _RENDER_LOCK:
        figure = renderer.render(
            polygons,
            projected,
            collar_depths=collar_depths,
            water_levels=plotted_water,
            lithology_codes=lithology_codes,
        )
        # Chemistry values the label placement could not fit beside their hole are
        # dropped (markers stay) and a set map scale may not fit the page; both are
        # only final after export framing, so the export appends them to the PDF
        # QA lines and they are read back below for the app warnings.
        include_render_notes = not (export_framing is not None and not export_framing.include_qa_footer)
        try:
            svg_bytes, png_bytes, pdf_bytes = renderer.export_figure_bytes(
                figure,
                export_formats,
                polygons=polygons,
                projected_df=projected,
                collar_depths=collar_depths,
                water_levels=water_levels,
                lithology_codes=lithology_codes,
                qa_lines=qa_lines,
                append_render_notes=include_render_notes,
            )
        except Exception:
            from matplotlib import pyplot as plt

            plt.close(figure)
            raise
        # Export re-places labels for the page size: report the final drops.
        overlap_warnings = tuple(overlap_warnings) + tuple(
            getattr(renderer, "chemistry_label_notes", None) or ()
        )
        # Read after export framing: whether the map scale fits depends on the page;
        # raster notes say when PNG resolution was lowered to stay within the size cap.
        # Readings left off the figure (far outside their hole) and inverted
        # threshold bands lead: they change what the figure shows.
        qa_notes = data_notes + tuple(renderer.map_scale_notes(figure)) + tuple(
            getattr(renderer, "raster_notes", None) or ()
        )
        # Non-detects whose detection limit is above the guideline (threshold
        # colour mode): printed grey on the figure, named here.
        qa_notes = tuple(getattr(renderer, "chemistry_rdl_notes", None) or ()) + qa_notes
        # Hole-ID headers that still overprint after every header fallback.
        qa_notes = qa_notes + tuple(renderer.header_overlap_notes(figure))
        # Water lines clipped to the ground / left above it in an artesian zone.
        qa_notes = qa_notes + tuple(getattr(renderer, "water_line_notes", None) or ())
        # Nested-well vertical gradients (screen mid-points) and arrow cross-check.
        qa_notes = qa_notes + gradient_notes
        qa_notes = qa_notes + flow_notes
        # Several dated samples at one hole/parameter/depth: the latest is drawn.
        qa_notes = qa_notes + chemistry_date_notes
        # Surface them with the other section notes the app shows after Generate.
        overlap_warnings = tuple(overlap_warnings) + qa_notes
        retained: dict[str, object] | None = None
        if close_figure:
            from matplotlib import pyplot as plt

            plt.close(figure)
        else:
            retained = {
                "figure": figure,
                "renderer": renderer,
                "polygons": polygons,
                "projected": projected,
                "collar_depths": collar_depths,
                "water_levels": water_levels,
                "lithology_codes": lithology_codes,
                "qa_lines": qa_lines,
                "append_render_notes": include_render_notes,
            }
    return CrossSectionResult(
        projected=projected,
        polygons=polygons,
        svg_bytes=svg_bytes,
        png_bytes=png_bytes,
        pdf_bytes=pdf_bytes,
        lithology_codes=lithology_codes,
        overlap_warnings=overlap_warnings,
        retained_export_bundle=retained,
        qa_notes=qa_notes,
    )
