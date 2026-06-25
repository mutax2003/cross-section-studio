"""Cross-section build pipeline: projection → stratigraphy → SVG."""

from __future__ import annotations

import logging
from typing import Literal, Sequence

import pandas as pd

from constants import BOREHOLE_ONLY_DISCLAIMER, INTERPOLATED_DISCLAIMER
from lithology_codes import collect_lithology_codes
from models import Collar, Lithology, Transect, WaterLevel
from projection import DEFAULT_OFFSET_WARNING_M, project_boreholes
from renderer import CrossSectionRenderer
from stratigraphy import (
    GeologicalPolygon,
    PolygonOverlap,
    build_stratigraphy,
    detect_polygon_overlaps,
    log_polygon_overlaps,
)

logger = logging.getLogger(__name__)

InterpretationMode = Literal["borehole_only", "interpolated"]
_VALID_INTERPRETATION_MODES = frozenset({"borehole_only", "interpolated"})

_SCALE_BAR_CANDIDATES = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
_DEFAULT_UNCERTAINTY_SPACING_M = 80.0

DEFAULT_UNCERTAINTY_SPACING_M = _DEFAULT_UNCERTAINTY_SPACING_M


def validate_interpretation_mode(mode: str) -> InterpretationMode:
    if mode not in _VALID_INTERPRETATION_MODES:
        allowed = ", ".join(sorted(_VALID_INTERPRETATION_MODES))
        raise ValueError(f"interpretation_mode must be one of: {allowed} (got {mode!r})")
    return mode  # type: ignore[return-value]


def auto_scale_bar_m(x_span: float) -> float:
    target = max(x_span / 5.0, 1.0)
    return min(_SCALE_BAR_CANDIDATES, key=lambda value: abs(value - target))


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
    water_levels: Sequence[WaterLevel] | None = None,
    uncertainty_spacing_m: float = _DEFAULT_UNCERTAINTY_SPACING_M,
    uncertainty_offset_m: float = DEFAULT_OFFSET_WARNING_M,
) -> tuple[pd.DataFrame, list[GeologicalPolygon], bytes, list[str], tuple[str, ...]]:
    """Project, build stratigraphy, render SVG. Returns projected df, polygons, svg, codes, overlap warnings."""
    interpretation_mode = validate_interpretation_mode(interpretation_mode)
    if len(transect_points) < 2:
        raise ValueError("At least two transect points are required")
    transect = Transect(points=list(transect_points))
    projected = project_boreholes(
        collars,
        lithologies,
        transect,
        offset_warning_m=offset_warning_m,
    )
    if projected.empty:
        raise ValueError("No lithology intervals were projected for the selected transect")

    if interpretation_mode == "borehole_only":
        polygons: list[GeologicalPolygon] = []
    else:
        polygons = build_stratigraphy(projected, allow_pinch_outs=allow_pinch_outs)

    if len(polygons) < 2:
        overlap_pairs: tuple[PolygonOverlap, ...] = ()
    else:
        overlap_pairs = tuple(detect_polygon_overlaps(polygons))
    if overlap_pairs and logger.isEnabledFor(logging.WARNING):
        log_polygon_overlaps(overlap_pairs)
    overlap_warnings = tuple(overlap.message() for overlap in overlap_pairs)

    lithology_codes = collect_lithology_codes(projected, polygons)
    collar_depths = {collar.hole_id: collar.total_depth for collar in collars}
    x_span = float(projected["x_profile"].max() - projected["x_profile"].min())
    disclaimer = (
        BOREHOLE_ONLY_DISCLAIMER
        if interpretation_mode == "borehole_only"
        else INTERPOLATED_DISCLAIMER
    )
    renderer = CrossSectionRenderer(
        vertical_exaggeration=vertical_exaggeration,
        scale_bar_length_m=auto_scale_bar_m(x_span),
        show_hatches=show_hatches,
        show_legend=show_legend,
        title=title,
        disclaimer=disclaimer,
        interpretation_mode=interpretation_mode,
        uncertainty_spacing_m=uncertainty_spacing_m,
        uncertainty_offset_m=uncertainty_offset_m,
        overlap_pairs=overlap_pairs,
    )
    svg_bytes = renderer.render_to_svg(
        polygons,
        projected,
        collar_depths=collar_depths,
        water_levels=water_levels,
        lithology_codes=lithology_codes,
    )
    return projected, polygons, svg_bytes, lithology_codes, overlap_warnings
