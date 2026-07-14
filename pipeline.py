"""Cross-section build pipeline: projection → stratigraphy → SVG."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Sequence

import pandas as pd

from constants import (
    BOREHOLE_ONLY_DISCLAIMER,
    CORRELATION_LINES_DISCLAIMER,
    INTERPOLATED_DISCLAIMER,
)
from lithology_codes import collect_lithology_codes
from models import (
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
from projection import DEFAULT_OFFSET_WARNING_M, project_boreholes, transect_azimuth_deg
from render_profiles import profile_for_layout, profile_with_elevation_mode
from renderer import CrossSectionRenderer
from stratigraphy import (
    GeologicalPolygon,
    PolygonOverlap,
    build_stratigraphy,
    detect_polygon_overlaps,
    log_polygon_overlaps,
    preview_correlation_health,
)

logger = logging.getLogger(__name__)


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


def build_cross_section(
    collars: Sequence[Collar],
    lithologies: Sequence[Lithology],
    transect_points: Sequence[tuple[float, float]],
    *,
    vertical_exaggeration: float = 1.0,
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
    render_layout: str = "section_sheet",
    track_width_m: float = 3.0,
    elevation_mode: str = "absolute",
    raster_log_strips: Sequence[RasterLogStrip] = (),
    export_formats: frozenset[str] | None = None,
    consulting_title_block: ConsultingTitleBlock | None = None,
    screen_intervals: Sequence[ScreenInterval] | None = None,
    vertical_gradients: Sequence[VerticalGradient] | None = None,
) -> CrossSectionResult:
    """Project, build stratigraphy, render. Returns ``CrossSectionResult`` (also unpackable as a 7-tuple)."""
    export_formats = _normalize_export_formats(export_formats)
    interpretation_mode = validate_interpretation_mode(interpretation_mode)
    if vertical_exaggeration <= 0:
        raise ValueError("vertical_exaggeration must be positive")
    if offset_warning_m <= 0:
        raise ValueError("offset_warning_m must be positive")
    if uncertainty_spacing_m <= 0:
        raise ValueError("uncertainty_spacing_m must be positive")
    if uncertainty_offset_m <= 0:
        raise ValueError("uncertainty_offset_m must be positive")
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

    interpolation_df = _filter_projected_for_interpolation(
        projected,
        max_offset_for_interpolation_m,
    )

    if interpretation_mode == "borehole_only":
        polygons: list[GeologicalPolygon] = []
        overlap_pairs: tuple[PolygonOverlap, ...] = ()
    else:
        polygons = build_stratigraphy(
            interpolation_df,
            allow_pinch_outs=allow_pinch_outs,
            correlation_overrides=correlation_overrides,
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
    if (
        warn_on_correlation_gaps
        and interpretation_mode != "borehole_only"
        and not interpolation_df.empty
    ):
        for summary in preview_correlation_health(
            interpolation_df,
            allow_pinch_outs=allow_pinch_outs,
            correlation_overrides=correlation_overrides,
        ):
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
    overlap_warnings = overlap_warnings + tuple(correlation_warnings)
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
    x_span = float(projected["x_profile"].max() - projected["x_profile"].min()) if not projected.empty else 1.0
    max_offset = float(projected["offset_distance"].max()) if not projected.empty else 0.0
    disclaimer = _DISCLAIMER_BY_MODE[interpretation_mode]

    metadata = figure_metadata or SectionFigureMetadata(
        vertical_exaggeration=vertical_exaggeration,
        transect_azimuth_deg=transect_azimuth_deg(transect_points),
        hole_count=len(projected_hole_ids),
        max_offset_m=max_offset,
    )

    base_profile = profile_for_layout(render_layout)  # type: ignore[arg-type]
    profile_updates: dict[str, object] = {"show_ground_surface": show_ground_surface}
    if render_layout == "section_sheet":
        profile_updates["track_width_m"] = track_width_m
    if show_water_elevation_labels is not None:
        profile_updates["show_water_elevation_labels"] = show_water_elevation_labels
    if show_water_legend is not None:
        profile_updates["show_water_legend"] = show_water_legend
    if show_dry_well_nm is not None:
        profile_updates["show_dry_well_nm"] = show_dry_well_nm
    if water_interpolate_across_gaps is not None:
        profile_updates["water_interpolate_across_gaps"] = water_interpolate_across_gaps
    if environmental_parameters:
        profile_updates["show_parameter_markers"] = True
    if show_parameter_labels is not None:
        profile_updates["show_parameter_labels"] = show_parameter_labels
    if parameter_interpolate_segments is not None:
        profile_updates["parameter_interpolate_segments"] = parameter_interpolate_segments
    if parameter_interpolate_across_gaps is not None:
        profile_updates["parameter_interpolate_across_gaps"] = parameter_interpolate_across_gaps
    render_profile = profile_with_elevation_mode(base_profile, elevation_mode).model_copy(
        update=profile_updates
    )

    effective_show_legend = show_legend
    effective_interpolate_wt = interpolate_water_table
    if render_profile.legend_in_title_block:
        effective_show_legend = False
    if render_profile.interpolate_water_table_default:
        effective_interpolate_wt = True
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
        consulting_title_block=consulting_title_block,
        screen_intervals=screen_intervals or (),
        vertical_gradients=vertical_gradients or (),
    )
    figure = renderer.render(
        polygons,
        projected,
        collar_depths=collar_depths,
        water_levels=water_levels,
        lithology_codes=lithology_codes,
    )
    try:
        svg_bytes, png_bytes, pdf_bytes = renderer.export_figure_bytes(
            figure,
            export_formats,
            polygons=polygons,
            projected_df=projected,
            collar_depths=collar_depths,
            water_levels=water_levels,
            lithology_codes=lithology_codes,
            qa_lines=overlap_warnings,
        )
    finally:
        from matplotlib import pyplot as plt

        plt.close(figure)
    return CrossSectionResult(
        projected=projected,
        polygons=polygons,
        svg_bytes=svg_bytes,
        png_bytes=png_bytes,
        pdf_bytes=pdf_bytes,
        lithology_codes=lithology_codes,
        overlap_warnings=overlap_warnings,
    )
