"""Vector rendering for borehole cross-section profiles."""

from __future__ import annotations

import io
import logging
from bisect import bisect_left
from dataclasses import dataclass
from typing import Literal, Sequence, TypedDict

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PatchCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.image as mpimg

from constants import HATCH_LINE_COLOR, POLYGON_EDGE_COLOR, get_lithology_style
from lithology_codes import collect_lithology_codes
from models import (
    ConsultingTitleBlock,
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
from render_profiles import (
    CHART_PROFILE,
    CrossSectionRenderProfile,
    SECTION_SHEET_PROFILE,
)
from render_theme import (
    AXES_BG,
    CONSULTING_FIGURE_BG,
    CONSULTING_NM_COLOR,
    CONSULTING_SCALE_BAR_M,
    CONSULTING_SURFACE_COLOR,
    CONSULTING_WATER_COLOR,
    consulting_gw_series_style,
    consulting_section_title,
    CONTACT_TICK_COLOR,
    CONTACT_TICK_WIDTH,
    DEFAULT_CONSULTING_NOTES,
    EOL_BAR_COLOR,
    FIGURE_BG,
    GRID_COLOR,
    LABEL_COLOR,
    PARAMETER_PALETTE,
    PARAMETER_READING_COLOR,
    PINCH_OUT_ALPHA,
    REPORT_GRID_ALPHA,
    REPORT_GRID_COLOR,
    SKY_FILL_COLOR,
    STICK_COLOR,
    SURFACE_COLOR,
    TRACK_BORDER_COLOR,
    TRACK_FILL_COLOR,
    UNCERTAINTY_COLOR,
    WATER_COLOR,
)
from stratigraphy import GeologicalPolygon, PolygonOverlap
from renderer_chart import ChartLayoutMixin
from renderer_common import RendererGeometryMixin
from renderer_consulting import ConsultingLayoutMixin
from renderer_section_sheet import SectionSheetLayoutMixin

logger = logging.getLogger(__name__)

_GW_MARKER_MAP = {
    "circle": "o",
    "triangle": "v",
    "diamond": "D",
    "plus": "P",
    "x": "x",
}


class WaterSeriesLegendEntry(TypedDict):
    series_id: str
    color: str
    marker: str
    level_label: str
    elevation_label: str


class ParameterLegendEntry(TypedDict):
    parameter: str
    color: str
    marker: str
    label: str


def _group_water_levels(
    water_levels: Sequence[WaterLevel],
    profile_lookup: dict[str, tuple[float, float]],
) -> dict[str, list[WaterLevel]]:
    groups: dict[str, list[WaterLevel]] = {}
    for level in water_levels:
        if level.hole_id not in profile_lookup:
            continue
        series_id = level.series_id or "default"
        groups.setdefault(series_id, []).append(level)
    return groups


_PARAMETER_LABEL_MIN_GAP_PTS = 26.0
_PARAMETER_LABEL_DX = 8.0
_PARAMETER_LABEL_BASE_DY = 0.0
_PARAMETER_LABEL_LEADER_EPS_PTS = 2.5
_PARAMETER_LABEL_FONTSIZE = 6.5
_PARAMETER_LABEL_FONTSIZE_CONSULTING = 7.25
_PARAMETER_LABEL_BBOX = {
    "boxstyle": "square,pad=0.12",
    "facecolor": "white",
    "edgecolor": "none",
    "alpha": 0.88,
}


def _nearest_unused_by_depth(
    depths: Sequence[float],
    used: Sequence[bool],
    target_depth: float,
) -> int | None:
    """Return index of unused depth nearest to ``target_depth`` (``depths`` ascending)."""
    count = len(depths)
    if count == 0:
        return None
    pos = bisect_left(depths, target_depth)
    best_idx: int | None = None
    best_delta: float | None = None
    for index in range(pos, count):
        if used[index]:
            continue
        delta = abs(depths[index] - target_depth)
        if best_delta is not None and depths[index] - target_depth > best_delta:
            break
        if best_delta is None or delta < best_delta:
            best_idx = index
            best_delta = delta
    for index in range(pos - 1, -1, -1):
        if used[index]:
            continue
        delta = abs(depths[index] - target_depth)
        if best_delta is not None and target_depth - depths[index] > best_delta:
            break
        if best_delta is None or delta < best_delta:
            best_idx = index
            best_delta = delta
    return best_idx


def _cluster_parameter_bands_by_depth(
    measured_holes: Sequence[str],
    readings_by_hole: dict[str, list[EnvironmentalReading]],
    *,
    depth_tol: float = 1.5,
) -> list[list[tuple[str, EnvironmentalReading]]]:
    """Group multi-depth samples into bands (same seed-tolerance semantics as before).

    Seeds are created in transect hole order; a reading joins the first existing band
    whose seed depth is within ``depth_tol``.
    """
    bands: list[list[tuple[str, EnvironmentalReading]]] = []
    seed_depths: list[float] = []
    for hole_id in measured_holes:
        for reading in readings_by_hole[hole_id]:
            depth = reading.sample_depth
            placed = False
            for band_index, seed_depth in enumerate(seed_depths):
                if abs(seed_depth - depth) <= depth_tol:
                    bands[band_index].append((hole_id, reading))
                    placed = True
                    break
            if not placed:
                bands.append([(hole_id, reading)])
                seed_depths.append(depth)
    return bands


def _resolve_parameter_label_offsets(
    ax,
    marker_labels: list[tuple[float, float, str]],
    *,
    min_gap_pts: float = _PARAMETER_LABEL_MIN_GAP_PTS,
) -> list[tuple[float, float, bool]]:
    """Return ``(dx, dy, draw_leader)`` offset-points for each parameter label.

    Dense stacks on one hole share the same X. Labels stay in one column to the
    right of the stick and are nudged downward to keep a minimum vertical gap.
    """
    if not marker_labels:
        return []

    groups: dict[float, list[int]] = {}
    for index, (x_profile, _y, _text) in enumerate(marker_labels):
        groups.setdefault(round(float(x_profile), 4), []).append(index)

    y0, y1 = ax.get_ylim()
    data_span = abs(float(y1) - float(y0)) or 1.0
    pos = ax.get_position()
    height_pts = float(ax.figure.get_figheight()) * float(pos.height) * 72.0
    pts_per_data = height_pts / data_span if height_pts > 0 else 1.0

    offsets: list[tuple[float, float, bool]] = [
        (_PARAMETER_LABEL_DX, _PARAMETER_LABEL_BASE_DY, False)
        for _ in marker_labels
    ]

    for indices in groups.values():
        indices_sorted = sorted(indices, key=lambda i: marker_labels[i][1], reverse=True)
        last_text_y: float | None = None
        for label_index in indices_sorted:
            _x, y, _text = marker_labels[label_index]
            marker_y_pts = float(y) * pts_per_data
            dy = _PARAMETER_LABEL_BASE_DY
            text_y = marker_y_pts + dy
            if last_text_y is not None and text_y > last_text_y - min_gap_pts:
                text_y = last_text_y - min_gap_pts
                dy = text_y - marker_y_pts
            # Keep labels inside axes (inverted Y still has y0/y1 span).
            y_lo_pts = min(float(y0), float(y1)) * pts_per_data
            y_hi_pts = max(float(y0), float(y1)) * pts_per_data
            text_y = min(max(text_y, y_lo_pts + min_gap_pts * 0.25), y_hi_pts - min_gap_pts * 0.25)
            dy = text_y - marker_y_pts
            draw_leader = abs(dy - _PARAMETER_LABEL_BASE_DY) > _PARAMETER_LABEL_LEADER_EPS_PTS
            offsets[label_index] = (_PARAMETER_LABEL_DX, dy, draw_leader)
            last_text_y = text_y
    return offsets


@dataclass(frozen=True)
class _HoleContext:
    summary: pd.DataFrame
    collar_lookup: dict[str, float]
    x_by_hole: dict[str, float]
    profile_lookup: dict[str, tuple[float, float]]
    x_span: float
    track_half: float


class CrossSectionRenderer(
    ConsultingLayoutMixin,
    SectionSheetLayoutMixin,
    ChartLayoutMixin,
    RendererGeometryMixin,
):
    """Render geological polygons and borehole tracks to a matplotlib figure."""

    def __init__(
        self,
        vertical_exaggeration: float = 1.0,
        scale_bar_length_m: float = 10.0,
        *,
        show_hatches: bool = True,
        show_legend: bool = True,
        title: str | None = None,
        disclaimer: str | None = None,
        interpretation_mode: InterpretationMode = "interpolated",
        uncertainty_spacing_m: float = 80.0,
        uncertainty_offset_m: float = 50.0,
        show_stick_logs: bool = True,
        overlap_pairs: Sequence[PolygonOverlap] | None = None,
        show_ground_surface: bool | None = None,
        interpolate_water_table: bool = False,
        figure_metadata: SectionFigureMetadata | None = None,
        faults: Sequence[Fault] = (),
        unconformities: Sequence[Unconformity] = (),
        environmental_readings: Sequence[EnvironmentalReading] = (),
        environmental_parameters: Sequence[str] = (),
        render_profile: CrossSectionRenderProfile | None = None,
        raster_log_strips: Sequence[RasterLogStrip] = (),
        consulting_title_block: ConsultingTitleBlock | None = None,
        screen_intervals: Sequence[ScreenInterval] = (),
        vertical_gradients: Sequence[VerticalGradient] = (),
    ) -> None:
        self.vertical_exaggeration = vertical_exaggeration
        self.scale_bar_length_m = scale_bar_length_m
        self.show_hatches = show_hatches
        self.show_legend = show_legend
        self.title = title or "Borehole Cross-Section"
        self.disclaimer = disclaimer
        self.interpretation_mode = interpretation_mode
        self.uncertainty_spacing_m = uncertainty_spacing_m
        self.uncertainty_offset_m = uncertainty_offset_m
        self.show_stick_logs = show_stick_logs
        self.overlap_pairs = tuple(overlap_pairs or ())
        self.interpolate_water_table = interpolate_water_table
        self.figure_metadata = figure_metadata
        self.faults = tuple(faults)
        self.unconformities = tuple(unconformities)
        self.environmental_readings = tuple(environmental_readings)
        self.environmental_parameters = tuple(environmental_parameters)
        self.profile = render_profile or SECTION_SHEET_PROFILE
        self.raster_log_strips = tuple(raster_log_strips)
        self.consulting_title_block = consulting_title_block
        self.screen_intervals = tuple(screen_intervals)
        self.vertical_gradients = tuple(vertical_gradients)
        self.water_series_legend: list[WaterSeriesLegendEntry] = []
        self.parameter_series_legend: list[ParameterLegendEntry] = []
        self._has_pinch_out = False
        if show_ground_surface is not None:
            self.profile = self.profile.model_copy(update={"show_ground_surface": show_ground_surface})

    def render(
        self,
        polygons: list[GeologicalPolygon],
        projected_df: pd.DataFrame,
        collar_depths: dict[str, float] | None = None,
        water_levels: Sequence[WaterLevel] | None = None,
        *,
        lithology_codes: Sequence[str] | None = None,
    ) -> Figure:
        hatch_context = {"hatch.color": HATCH_LINE_COLOR, "hatch.linewidth": 0.65}
        with mpl.rc_context(hatch_context):
            return self._render_figure(
                polygons,
                projected_df,
                collar_depths,
                water_levels=water_levels,
                lithology_codes=lithology_codes,
            )

    def render_to_svg(
        self,
        polygons: list[GeologicalPolygon],
        projected_df: pd.DataFrame,
        collar_depths: dict[str, float] | None = None,
        water_levels: Sequence[WaterLevel] | None = None,
        *,
        lithology_codes: Sequence[str] | None = None,
    ) -> bytes:
        figure = self.render(
            polygons,
            projected_df,
            collar_depths=collar_depths,
            water_levels=water_levels,
            lithology_codes=lithology_codes,
        )
        try:
            return self.to_svg_bytes(figure)
        finally:
            plt.close(figure)

    def render_to_png(
        self,
        polygons: list[GeologicalPolygon],
        projected_df: pd.DataFrame,
        collar_depths: dict[str, float] | None = None,
        water_levels: Sequence[WaterLevel] | None = None,
        *,
        lithology_codes: Sequence[str] | None = None,
        dpi: int = 300,
    ) -> bytes:
        figure = self.render(
            polygons,
            projected_df,
            collar_depths=collar_depths,
            water_levels=water_levels,
            lithology_codes=lithology_codes,
        )
        try:
            return self.to_png_bytes(figure, dpi=dpi)
        finally:
            plt.close(figure)

    def _render_figure(
        self,
        polygons: list[GeologicalPolygon],
        projected_df: pd.DataFrame,
        collar_depths: dict[str, float] | None,
        *,
        water_levels: Sequence[WaterLevel] | None = None,
        lithology_codes: Sequence[str] | None = None,
    ) -> Figure:
        self.water_series_legend = []
        if self.profile.layout == "section_sheet":
            return self._render_section_sheet(
                polygons,
                projected_df,
                collar_depths,
                water_levels=water_levels,
                lithology_codes=lithology_codes,
            )
        if self.profile.layout == "consulting_section":
            return self._render_consulting_section(
                polygons,
                projected_df,
                collar_depths,
                water_levels=water_levels,
                lithology_codes=lithology_codes,
            )
        return self._render_chart_layout(
            polygons,
            projected_df,
            collar_depths,
            water_levels=water_levels,
            lithology_codes=lithology_codes,
        )

    def _hole_context(self, projected_df: pd.DataFrame) -> _HoleContext:
        summary = projected_df.groupby("hole_id", as_index=False, sort=False).agg(
            x_profile=("x_profile", "first"),
            collar_elevation=("collar_elevation", "first"),
            bottom_elevation=("bottom_elevation", "min"),
            offset_distance=("offset_distance", "first"),
        )
        if not projected_df["x_profile"].is_monotonic_increasing:
            summary = summary.sort_values("x_profile")
        hole_ids = summary["hole_id"].astype(str).to_numpy()
        collar_values = summary["collar_elevation"].to_numpy(dtype=float)
        x_values = summary["x_profile"].to_numpy(dtype=float)
        profile_lookup = {
            str(hole_id): (float(x_profile), float(collar))
            for hole_id, x_profile, collar in zip(hole_ids, x_values, collar_values, strict=True)
        }
        collar_lookup = {hole_id: collar for hole_id, (_, collar) in profile_lookup.items()}
        x_by_hole = {hole_id: x_profile for hole_id, (x_profile, _) in profile_lookup.items()}
        x_span = (
            float(summary["x_profile"].max() - summary["x_profile"].min())
            if len(summary) >= 2
            else 10.0
        )
        return _HoleContext(
            summary=summary,
            collar_lookup=collar_lookup,
            x_by_hole=x_by_hole,
            profile_lookup=profile_lookup,
            x_span=x_span,
            track_half=self._track_half_width(x_span),
        )

    def _uncertainty_y_bounds(self, hole_summary: pd.DataFrame) -> tuple[float, float]:
        ve = self.vertical_exaggeration
        collars = hole_summary["collar_elevation"].to_numpy(dtype=float)
        bottoms = hole_summary["bottom_elevation"].to_numpy(dtype=float)
        if self.profile.y_axis_mode == "depth_below_collar":
            depths_top = (collars - collars) * ve
            depths_bottom = (collars - bottoms) * ve
            return float(depths_bottom.min()), float(depths_top.max())
        return float((bottoms * ve).min()), float((collars * ve).max())

    def _plot_y(self, elevation: float, collar_rl: float) -> float:
        ve = self.vertical_exaggeration
        if self.profile.y_axis_mode == "depth_below_collar":
            return (collar_rl - elevation) * ve
        return elevation * ve

    def _plot_y_values(self, elevations: np.ndarray, collar_rls: np.ndarray) -> np.ndarray:
        ve = self.vertical_exaggeration
        if self.profile.y_axis_mode == "depth_below_collar":
            return (collar_rls - elevations) * ve
        return elevations * ve

    def _track_half_width(self, x_span: float) -> float:
        if self.profile.layout in {"section_sheet", "consulting_section"}:
            return self.profile.track_width_m / 2.0
        return max(x_span * 0.015, 0.8)

    def _transform_coords(self, coords: list[tuple[float, float]], ve: float | None = None) -> np.ndarray:
        multiplier = self.vertical_exaggeration if ve is None else ve
        array = np.asarray(coords, dtype=float)
        array[:, 1] *= multiplier
        return array

    def _fence_plot_coords(
        self,
        coords: np.ndarray,
        ve: float,
        collar_lookup: dict[str, float],
        hole_pair: tuple[str, str] | None,
    ) -> np.ndarray:
        """Map fence polygon (x, elevation) verts into plot Y (RL or depth-below-collar)."""
        if self.profile.y_axis_mode != "depth_below_collar":
            return self._transform_coords(coords.tolist(), ve)
        # Prefer left collar for left half of the polygon, right for right half.
        left_rl = collar_lookup.get(hole_pair[0]) if hole_pair else None
        right_rl = collar_lookup.get(hole_pair[1]) if hole_pair else None
        if left_rl is None and right_rl is None:
            return self._transform_coords(coords.tolist(), ve)
        mid_x = float(np.mean(coords[:, 0])) if len(coords) else 0.0
        out = np.asarray(coords, dtype=float).copy()
        for index, (x_val, elev) in enumerate(out):
            if left_rl is not None and right_rl is not None:
                collar_rl = left_rl if float(x_val) <= mid_x + 1e-9 else right_rl
            else:
                collar_rl = left_rl if left_rl is not None else right_rl
            out[index, 1] = (float(collar_rl) - float(elev)) * ve
        return out

    def _draw_fence_polygons(
        self,
        ax,
        polygons: list[GeologicalPolygon],
        style_cache: dict,
        ve: float,
        *,
        alpha: float,
        collar_lookup: dict[str, float] | None = None,
    ) -> None:
        if self.interpretation_mode not in {"interpolated", "correlation_lines"}:
            self._has_pinch_out = False
            return
        self._has_pinch_out = any(polygon.is_pinch_out for polygon in polygons)
        collars = collar_lookup or {}
        if self.interpretation_mode == "correlation_lines":
            line_groups: dict[tuple[str, str], list[np.ndarray]] = {}
            for geo_polygon in polygons:
                style = self._resolve_style(geo_polygon.lithology_code, style_cache)
                coords = np.asarray(geo_polygon.polygon.exterior.coords, dtype=float)
                line_style = "--" if geo_polygon.is_pinch_out else "-"
                style_key = (style.edge_color, line_style)
                if len(coords) >= 4:
                    top_coords = self._fence_plot_coords(
                        coords[[0, 1]], ve, collars, geo_polygon.hole_pair
                    )
                    bottom_coords = self._fence_plot_coords(
                        coords[[3, 2]], ve, collars, geo_polygon.hole_pair
                    )
                    line_groups.setdefault(style_key, []).extend([top_coords, bottom_coords])
                elif len(coords) == 3:
                    top_coords = self._fence_plot_coords(
                        coords[[0, 2]], ve, collars, geo_polygon.hole_pair
                    )
                    bottom_coords = self._fence_plot_coords(
                        coords[[1, 2]], ve, collars, geo_polygon.hole_pair
                    )
                    line_groups.setdefault(style_key, []).extend([top_coords, bottom_coords])
            for (edge_color, line_style), segments in line_groups.items():
                collection = LineCollection(
                    segments,
                    colors=edge_color,
                    linewidths=1.0,
                    linestyles=line_style,
                    zorder=3,
                )
                ax.add_collection(collection)
            return
        polygon_groups: dict[tuple[str, str, str, str, float], list[np.ndarray]] = {}
        for geo_polygon in polygons:
            style = self._resolve_style(geo_polygon.lithology_code, style_cache)
            coords = np.asarray(geo_polygon.polygon.exterior.coords, dtype=float)
            linestyle = "--" if geo_polygon.is_pinch_out else "-"
            patch_alpha = PINCH_OUT_ALPHA if geo_polygon.is_pinch_out else alpha
            verts = self._fence_plot_coords(coords, ve, collars, geo_polygon.hole_pair)
            style_key = (style.color, style.hatch or "", style.edge_color, linestyle, patch_alpha)
            polygon_groups.setdefault(style_key, []).append(verts)
        for style_key, verts_list in polygon_groups.items():
            color, hatch, edge_color, linestyle, patch_alpha = style_key
            collection = PolyCollection(
                verts_list,
                facecolors=color,
                edgecolors=edge_color,
                linewidths=0.75,
                linestyles=linestyle,
                hatch=hatch or None,
                alpha=patch_alpha,
            )
            collection.set_zorder(2)
            ax.add_collection(collection)

    def _draw_sky_and_surface(
        self,
        ax,
        hole_summary: pd.DataFrame,
        collar_lookup: dict[str, float],
    ) -> None:
        if len(hole_summary) < 2:
            return
        surface_x = hole_summary["x_profile"].to_numpy(dtype=float)
        ve = self.vertical_exaggeration
        if self.profile.y_axis_mode == "depth_below_collar":
            surface_y = np.zeros(len(hole_summary), dtype=float)
        else:
            surface_y = hole_summary["collar_elevation"].to_numpy(dtype=float) * ve
        sample_count = max(20, min(100, len(surface_x) * 20))
        x_dense = np.linspace(surface_x.min(), surface_x.max(), sample_count)
        y_dense = np.interp(x_dense, surface_x, surface_y)
        # Depth mode inverts later: positive Y is into geology, so skip sky fill there.
        if self.profile.show_sky_fill and self.profile.y_axis_mode != "depth_below_collar":
            y_top = y_dense.max() + 0.12 * max(y_dense.max() - y_dense.min(), 1.0)
            ax.fill_between(x_dense, y_dense, y_top, facecolor=SKY_FILL_COLOR, edgecolor="none", alpha=0.85, zorder=1)
        if self.profile.show_ground_surface:
            ax.plot(x_dense, y_dense, color=SURFACE_COLOR, linewidth=3.0, solid_capstyle="round", zorder=6)

    def _draw_track_lithology(
        self,
        ax,
        projected_df: pd.DataFrame,
        style_cache: dict,
        track_half_width: float,
        collar_lookup: dict[str, float],
        *,
        collar_arr: np.ndarray | None = None,
    ) -> None:
        self._draw_lithology_interval_rects(
            ax,
            projected_df,
            style_cache,
            track_half_width,
            collar_lookup,
            collar_arr=collar_arr,
        )

    def _draw_track_borders(
        self,
        ax,
        hole_summary: pd.DataFrame,
        collar_depths: dict[str, float],
        collar_lookup: dict[str, float],
        track_half_width: float,
    ) -> None:
        geometry = self._well_rect_geometry(
            hole_summary, collar_depths, collar_lookup, track_half_width
        )
        if geometry is None:
            return
        self._add_rect_collection(
            ax,
            geometry,
            facecolors="none",
            edgecolors=TRACK_BORDER_COLOR,
            linewidths=1.4,
            zorder=6,
        )

    def _draw_contact_ticks(
        self,
        ax,
        projected_df: pd.DataFrame,
        track_half_width: float,
        collar_lookup: dict[str, float],
        *,
        collar_arr: np.ndarray | None = None,
    ) -> None:
        if projected_df.empty:
            return
        collars = (
            collar_arr
            if collar_arr is not None
            else self._collar_values(
                projected_df["hole_id"],
                projected_df["collar_elevation"],
                collar_lookup,
            )
        )
        x_profiles = projected_df["x_profile"].to_numpy(dtype=float)
        tops = projected_df["top_elevation"].to_numpy(dtype=float)
        bottoms = projected_df["bottom_elevation"].to_numpy(dtype=float)
        hole_ids = projected_df["hole_id"].astype(str).to_numpy()
        contact_x = np.concatenate([x_profiles, x_profiles])
        contact_elev = np.concatenate([tops, bottoms])
        contact_collar = np.concatenate([collars, collars])
        contact_hole = np.concatenate([hole_ids, hole_ids])
        order = np.lexsort((contact_x, contact_elev, contact_hole))
        contact_x = contact_x[order]
        contact_elev = contact_elev[order]
        contact_collar = contact_collar[order]
        contact_hole = contact_hole[order]
        keep = np.ones(len(contact_x), dtype=bool)
        keep[1:] = (contact_hole[1:] != contact_hole[:-1]) | (contact_elev[1:] != contact_elev[:-1])
        contact_x = contact_x[keep]
        contact_elev = contact_elev[keep]
        contact_collar = contact_collar[keep]
        ys = self._plot_y_values(contact_elev, contact_collar)
        x_left = contact_x - track_half_width
        x_right = contact_x + track_half_width
        segments = np.stack(
            [
                np.column_stack([x_left, ys]),
                np.column_stack([x_right, ys]),
            ],
            axis=1,
        )
        if len(segments):
            collection = LineCollection(
                segments,
                colors=CONTACT_TICK_COLOR,
                linewidths=CONTACT_TICK_WIDTH,
                zorder=7,
            )
            ax.add_collection(collection)

    def _draw_eol_bars(
        self,
        ax,
        hole_summary: pd.DataFrame,
        collar_depths: dict[str, float],
        collar_lookup: dict[str, float],
        track_half_width: float,
    ) -> None:
        if hole_summary.empty or not collar_depths:
            return
        hole_ids = hole_summary["hole_id"].astype(str)
        td_values = hole_ids.map(collar_depths)
        mask = td_values.notna().to_numpy()
        if not mask.any():
            return
        x_values = hole_summary.loc[mask, "x_profile"].to_numpy(dtype=float)
        collars = self._collar_values(
            hole_summary.loc[mask, "hole_id"],
            hole_summary.loc[mask, "collar_elevation"],
            collar_lookup,
        )
        td_numeric = td_values[mask].to_numpy(dtype=float)
        ys = self._plot_y_values(collars - td_numeric, collars)
        width = track_half_width * 1.15
        segments = np.stack(
            [
                np.column_stack([x_values - width, ys]),
                np.column_stack([x_values + width, ys]),
            ],
            axis=1,
        )
        if len(segments):
            collection = LineCollection(
                segments,
                colors=EOL_BAR_COLOR,
                linewidths=2.5,
                zorder=8,
            )
            ax.add_collection(collection)

    def _draw_column_headers(
        self,
        ax,
        hole_summary: pd.DataFrame,
        collar_depths: dict[str, float],
        collar_lookup: dict[str, float],
    ) -> None:
        header_transform = ax.get_xaxis_transform()
        for row in hole_summary.itertuples(index=False):
            hole_id = str(row.hole_id)
            collar = float(collar_lookup.get(hole_id, row.collar_elevation))
            td = collar_depths.get(hole_id)
            td_text = f"TD {td:.1f} m" if td is not None else ""
            rl_text = (
                f"RL {collar:.1f}"
                if self.profile.y_axis_mode == "elevation_rl"
                else "Depth section"
            )
            if self.profile.y_axis_mode == "elevation_rl":
                y_pos = -0.05
                va = "top"
            else:
                y_pos = 1.05
                va = "bottom"
            ax.text(
                float(row.x_profile),
                y_pos,
                f"{hole_id}\n{rl_text}\n{td_text}".strip(),
                transform=header_transform,
                ha="center",
                va=va,
                fontsize=8,
                fontweight="bold",
                color=LABEL_COLOR,
                linespacing=1.15,
                clip_on=False,
                zorder=10,
            )

    def _draw_deviated_centerlines(
        self,
        ax,
        projected_df: pd.DataFrame,
        collar_lookup: dict[str, float],
    ) -> None:
        lines: list[list[tuple[float, float]]] = []
        for hole_id, group in projected_df.groupby("hole_id", sort=False):
            xs = group["x_profile"].to_numpy(dtype=float)
            if len(xs) < 2 or np.ptp(xs) < 0.05:
                continue
            collar = float(collar_lookup.get(str(hole_id), group["collar_elevation"].iloc[0]))
            mids = (
                group["top_elevation"].to_numpy(dtype=float)
                + group["bottom_elevation"].to_numpy(dtype=float)
            ) / 2.0
            collar_arr = np.full(len(mids), collar, dtype=float)
            ys = self._plot_y_values(mids, collar_arr)
            lines.append(np.column_stack((xs, ys)))
        if lines:
            collection = LineCollection(
                lines,
                colors=STICK_COLOR,
                linestyles=":",
                linewidths=1.0,
                alpha=0.7,
                zorder=4,
            )
            ax.add_collection(collection)

    def _draw_raster_strips(
        self,
        ax,
        x_by_hole: dict[str, float],
        collar_lookup: dict[str, float],
        track_half_width: float,
    ) -> None:
        if not self.raster_log_strips:
            return
        for strip in self.raster_log_strips:
            x = x_by_hole.get(strip.hole_id)
            if x is None:
                continue
            collar = float(collar_lookup.get(strip.hole_id, 0.0))
            top = self._plot_y(collar - strip.depth_top, collar)
            bottom = self._plot_y(collar - strip.depth_bottom, collar)
            y0, height = (top, bottom - top) if top <= bottom else (bottom, top - top)
            ax.add_patch(
                Rectangle(
                    (float(x) - track_half_width, y0),
                    2.0 * track_half_width,
                    height,
                    facecolor="#E2E8F0",
                    edgecolor="#64748B",
                    linewidth=1.0,
                    hatch="///",
                    alpha=0.5,
                )
            )
            ax.text(float(x), y0 + height / 2.0, strip.label, ha="center", va="center", fontsize=6, rotation=90, color="#475569")

    def _draw_water_table(
        self,
        ax,
        hole_summary: pd.DataFrame,
        water_levels: Sequence[WaterLevel],
        collar_lookup: dict[str, float],
        *,
        label_elevations: bool = False,
        label_dry_wells: bool = False,
        label_series_gaps: bool = False,
        water_color: str | None = None,
        profile_lookup: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if hole_summary.empty:
            return
        if profile_lookup is None:
            profile_lookup = self._profile_lookup(hole_summary, collar_lookup)
        series_groups = _group_water_levels(water_levels, profile_lookup)
        holes_with_any_water = {
            level.hole_id for levels in series_groups.values() for level in levels
        }
        fully_dry_nm_drawn: set[str] = set()
        if label_dry_wells:
            dry_lookup = {
                hole_id: profile
                for hole_id, profile in profile_lookup.items()
                if hole_id not in holes_with_any_water
            }
            if dry_lookup:
                dry_x = np.fromiter((p[0] for p in dry_lookup.values()), dtype=float, count=len(dry_lookup))
                dry_collars = np.fromiter((p[1] for p in dry_lookup.values()), dtype=float, count=len(dry_lookup))
                dry_y = self._plot_y_values(dry_collars - 1.0, dry_collars)
                for hole_id, x_profile, y in zip(dry_lookup.keys(), dry_x, dry_y, strict=True):
                    fully_dry_nm_drawn.add(str(hole_id))
                    ax.annotate(
                        "NM",
                        xy=(float(x_profile), float(y)),
                        xytext=(4, 0),
                        textcoords="offset points",
                        fontsize=8,
                        color=CONSULTING_NM_COLOR,
                        zorder=8,
                    )
        if not series_groups:
            return
        self.water_series_legend = []
        interpolate = self.interpolate_water_table or self.profile.interpolate_water_table_default
        use_segments = self.profile.water_interpolate_segments
        across_gaps = self.profile.water_interpolate_across_gaps
        default_water_color = (
            CONSULTING_WATER_COLOR
            if self.profile.layout == "consulting_section"
            else WATER_COLOR
        )
        elev_suffix = " masl" if self.profile.layout == "consulting_section" else " m"
        profile_marker = _GW_MARKER_MAP.get(self.profile.water_symbol, self.profile.water_symbol)
        transect_hole_ids = hole_summary.sort_values("x_profile")["hole_id"].astype(str).tolist()
        transect_x = {
            str(row.hole_id): float(row.x_profile)
            for row in hole_summary.itertuples(index=False)
        }
        for series_id, levels in sorted(series_groups.items()):
            first = levels[0]
            default_color, default_marker, default_label = consulting_gw_series_style(
                series_id,
                first.series_label,
            )
            color = first.color or water_color or default_water_color or default_color
            marker_key = (first.marker or default_marker or profile_marker).lower()
            marker = _GW_MARKER_MAP.get(marker_key, marker_key)
            label = first.series_label or default_label or series_id
            level_by_hole = {level.hole_id: level for level in levels}
            if label_series_gaps:
                # Fully dry holes: one NM only (skip if label_dry_wells already drew them,
                # or draw once across series when dry-well labeling is off).
                for hole_id in transect_hole_ids:
                    if hole_id in level_by_hole:
                        continue
                    if hole_id not in holes_with_any_water:
                        if label_dry_wells or hole_id in fully_dry_nm_drawn:
                            continue
                        fully_dry_nm_drawn.add(hole_id)
                    profile = profile_lookup.get(hole_id)
                    if profile is None:
                        continue
                    x_profile, collar_rl = profile
                    y = self._plot_y(collar_rl - 1.0, collar_rl)
                    ax.annotate(
                        "NM",
                        xy=(float(x_profile), float(y)),
                        xytext=(4, 0),
                        textcoords="offset points",
                        fontsize=8,
                        color=CONSULTING_NM_COLOR,
                        zorder=8,
                    )
            xs: list[float] = []
            water_rls: list[float] = []
            collars: list[float] = []
            for hole_id in transect_hole_ids:
                level = level_by_hole.get(hole_id)
                if level is None:
                    continue
                profile = profile_lookup.get(hole_id)
                if profile is None:
                    continue
                x_profile, collar_rl = profile
                xs.append(x_profile)
                water_rls.append(collar_rl - level.depth)
                collars.append(collar_rl)
            if not xs:
                continue
            xs_arr = np.asarray(xs, dtype=float)
            water_arr = np.asarray(water_rls, dtype=float)
            collar_arr = np.asarray(collars, dtype=float)
            ys = self._plot_y_values(water_arr, collar_arr)
            ax.scatter(xs_arr, ys, marker=marker, c=color, s=49, zorder=7)
            if label_elevations:
                for x_profile, water_rl, y in zip(xs_arr, water_arr, ys, strict=True):
                    ax.annotate(
                        f"{water_rl:.3f}{elev_suffix}",
                        xy=(float(x_profile), float(y)),
                        xytext=(4, -8),
                        textcoords="offset points",
                        fontsize=7,
                        color=color,
                        zorder=8,
                    )
            if len(xs_arr) >= 2 and interpolate:
                gw_linestyle = "-" if getattr(self.profile, "water_line_solid", False) else "--"
                if across_gaps:
                    # Dense interp only when explicitly allowed across dry/missing holes.
                    x_dense = np.linspace(float(xs_arr.min()), float(xs_arr.max()), 100)
                    y_dense = np.interp(x_dense, xs_arr, ys)
                    ax.plot(
                        x_dense, y_dense, color=color, linewidth=2.0, linestyle=gw_linestyle, zorder=6
                    )
                elif use_segments:
                    for left_id, right_id in zip(transect_hole_ids, transect_hole_ids[1:], strict=False):
                        if left_id not in level_by_hole or right_id not in level_by_hole:
                            continue
                        x0 = transect_x[left_id]
                        x1 = transect_x[right_id]
                        y0 = float(
                            self._plot_y(
                                profile_lookup[left_id][1] - level_by_hole[left_id].depth,
                                profile_lookup[left_id][1],
                            )
                        )
                        y1 = float(
                            self._plot_y(
                                profile_lookup[right_id][1] - level_by_hole[right_id].depth,
                                profile_lookup[right_id][1],
                            )
                        )
                        ax.plot(
                            [x0, x1], [y0, y1], color=color, linewidth=2.0, linestyle=gw_linestyle, zorder=6
                        )
                else:
                    # Measured holes only — do not invent water across dry gaps.
                    ax.plot(
                        xs_arr, ys, color=color, linewidth=2.0, linestyle=gw_linestyle, zorder=6
                    )
            if series_id == "default" and not label:
                level_label_text = "GROUNDWATER LEVEL (masl)"
                elevation_label_text = "GROUNDWATER ELEVATION (masl)"
            else:
                display_label = (label or default_label or series_id).upper()
                level_label_text = f"GROUNDWATER LEVEL ({display_label})"
                elevation_label_text = f"GROUNDWATER ELEVATION masl ({display_label})"
            self.water_series_legend.append(
                {
                    "series_id": series_id,
                    "color": color,
                    "marker": marker,
                    "level_label": level_label_text,
                    "elevation_label": elevation_label_text,
                }
            )

    def _draw_compact_water_legend(self, ax) -> None:
        if not self.water_series_legend:
            return
        lines = []
        for entry in self.water_series_legend:
            lines.append(entry["level_label"])
        ax.text(
            0.01,
            0.01,
            " | ".join(lines),
            transform=ax.transAxes,
            fontsize=7,
            color=LABEL_COLOR,
            va="bottom",
            ha="left",
            zorder=20,
        )

    def _collar_rl_for_overlay(
        self,
        x_profile: float,
        collar_lookup: dict[str, float],
        *,
        hole_pair: tuple[str, str] | None = None,
        hole_summary: pd.DataFrame | None = None,
    ) -> float:
        """Collar RL for depth-mode overlays at a section X (pair average or surface interp)."""
        if hole_pair is not None:
            left = collar_lookup.get(hole_pair[0])
            right = collar_lookup.get(hole_pair[1])
            if left is not None and right is not None:
                return (float(left) + float(right)) / 2.0
            if left is not None:
                return float(left)
            if right is not None:
                return float(right)
        if hole_summary is not None and not hole_summary.empty:
            xs = hole_summary["x_profile"].to_numpy(dtype=float)
            rls = hole_summary["collar_elevation"].to_numpy(dtype=float)
            if len(xs) == 1:
                return float(rls[0])
            order = np.argsort(xs)
            return float(np.interp(x_profile, xs[order], rls[order]))
        if collar_lookup:
            return float(np.mean(list(collar_lookup.values())))
        return 0.0

    def _draw_overlap_markers(
        self,
        ax,
        collar_lookup: dict[str, float],
        *,
        hole_summary: pd.DataFrame | None = None,
    ) -> None:
        if not self.overlap_pairs:
            return
        overlaps = self.overlap_pairs[:12]
        xs = np.fromiter((overlap.centroid_x for overlap in overlaps), dtype=float, count=len(overlaps))
        centroid_ys = np.fromiter((overlap.centroid_y for overlap in overlaps), dtype=float, count=len(overlaps))
        collar_rls = np.fromiter(
            (
                self._collar_rl_for_overlay(
                    float(overlap.centroid_x),
                    collar_lookup,
                    hole_pair=overlap.hole_pair,
                    hole_summary=hole_summary,
                )
                for overlap in overlaps
            ),
            dtype=float,
            count=len(overlaps),
        )
        ys = self._plot_y_values(centroid_ys, collar_rls)
        ax.scatter(
            xs,
            ys,
            marker="x",
            c=OVERLAP_MARKER_COLOR,
            s=64,
            linewidths=1.5,
            zorder=9,
        )

    def _draw_uncertainty_zones(
        self,
        ax,
        hole_summary: pd.DataFrame,
        y_min: float,
        y_max: float,
        collar_lookup: dict[str, float],
    ) -> None:
        if len(hole_summary) < 2:
            return
        x_values = hole_summary["x_profile"].to_numpy(dtype=float)
        offset_values = hole_summary["offset_distance"].to_numpy(dtype=float)
        spacing = np.abs(np.diff(x_values))
        max_offset = np.maximum(offset_values[:-1], offset_values[1:])
        uncertain = (spacing > self.uncertainty_spacing_m) | (max_offset > self.uncertainty_offset_m)
        uncertainty_patches = [
            Rectangle(
                (min(float(x_values[i]), float(x_values[i + 1])), y_min),
                abs(float(x_values[i + 1]) - float(x_values[i])),
                y_max - y_min,
                facecolor=UNCERTAINTY_COLOR,
                edgecolor="none",
                alpha=0.28,
            )
            for i in np.flatnonzero(uncertain)
        ]
        if uncertainty_patches:
            collection = PatchCollection(uncertainty_patches, match_original=True)
            collection.set_zorder(0)
            ax.add_collection(collection)

    def _draw_faults(
        self,
        ax,
        collar_lookup: dict[str, float],
        *,
        hole_summary: pd.DataFrame | None = None,
    ) -> None:
        if not self.faults:
            return
        segments: list[np.ndarray] = []
        for fault in self.faults:
            if len(fault.trace_points) < 2:
                continue
            xs, ys = zip(*fault.trace_points)
            collar_rls = np.fromiter(
                (
                    self._collar_rl_for_overlay(float(x), collar_lookup, hole_summary=hole_summary)
                    for x in xs
                ),
                dtype=float,
                count=len(xs),
            )
            plot_ys = self._plot_y_values(np.asarray(ys, dtype=float), collar_rls)
            segments.append(np.column_stack([xs, plot_ys]))
        if segments:
            collection = LineCollection(
                segments,
                colors="#B91C1C",
                linewidths=2.0,
                linestyles="-.",
                zorder=8,
            )
            ax.add_collection(collection)

    def _draw_unconformities(
        self,
        ax,
        collar_lookup: dict[str, float],
        *,
        hole_summary: pd.DataFrame | None = None,
    ) -> None:
        if not self.unconformities:
            return
        segments: list[np.ndarray] = []
        for surface in self.unconformities:
            if len(surface.elevation_profile) < 2:
                continue
            xs, ys = zip(*surface.elevation_profile)
            collar_rls = np.fromiter(
                (
                    self._collar_rl_for_overlay(float(x), collar_lookup, hole_summary=hole_summary)
                    for x in xs
                ),
                dtype=float,
                count=len(xs),
            )
            plot_ys = self._plot_y_values(np.asarray(ys, dtype=float), collar_rls)
            segments.append(np.column_stack([xs, plot_ys]))
        if segments:
            collection = LineCollection(
                segments,
                colors="#7C3AED",
                linewidths=1.8,
                linestyles=":",
                zorder=7,
            )
            ax.add_collection(collection)

    def _draw_parameter_readings(
        self,
        ax,
        hole_summary: pd.DataFrame,
        collar_lookup: dict[str, float],
        *,
        profile_lookup: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if hole_summary.empty or not self.environmental_readings:
            return
        if not self.profile.show_parameter_markers or not self.environmental_parameters:
            return
        if profile_lookup is None:
            profile_lookup = self._profile_lookup(hole_summary, collar_lookup)
        active_parameters = {name.strip() for name in self.environmental_parameters if name.strip()}
        if not active_parameters:
            return

        use_segments = self.profile.parameter_interpolate_segments
        across_gaps = self.profile.parameter_interpolate_across_gaps
        label_values = self.profile.show_parameter_labels
        marker_key = self.profile.parameter_marker
        marker = _GW_MARKER_MAP.get(marker_key, marker_key)
        transect_hole_ids = hole_summary.sort_values("x_profile")["hole_id"].astype(str).tolist()
        transect_x = {
            str(row.hole_id): float(row.x_profile)
            for row in hole_summary.itertuples(index=False)
        }

        by_parameter: dict[str, list[EnvironmentalReading]] = {}
        profile_holes = set(profile_lookup)
        for reading in self.environmental_readings:
            if reading.parameter not in active_parameters or reading.hole_id not in profile_holes:
                continue
            by_parameter.setdefault(reading.parameter, []).append(reading)

        self.parameter_series_legend = []
        for index, (parameter, readings) in enumerate(sorted(by_parameter.items())):
            color = PARAMETER_PALETTE[index % len(PARAMETER_PALETTE)]
            readings_by_hole: dict[str, list[EnvironmentalReading]] = {}
            for reading in readings:
                readings_by_hole.setdefault(reading.hole_id, []).append(reading)
            for hole_readings in readings_by_hole.values():
                hole_readings.sort(key=lambda item: item.sample_depth)

            marker_xs: list[float] = []
            marker_ys: list[float] = []
            marker_labels: list[tuple[float, float, str]] = []
            y_cache: dict[tuple[str, float], float] = {}

            def plot_depth_y(hole_id: str, sample_depth: float) -> float:
                key = (hole_id, sample_depth)
                cached = y_cache.get(key)
                if cached is not None:
                    return cached
                collar_rl = profile_lookup[hole_id][1]
                y = float(self._plot_y(collar_rl - sample_depth, collar_rl))
                y_cache[key] = y
                return y

            for hole_id in transect_hole_ids:
                hole_readings = readings_by_hole.get(hole_id)
                if not hole_readings:
                    continue
                x_profile = float(profile_lookup[hole_id][0])
                for reading in hole_readings:
                    y = plot_depth_y(hole_id, reading.sample_depth)
                    marker_xs.append(x_profile)
                    marker_ys.append(y)
                    if (
                        reading.from_depth is not None
                        and reading.to_depth is not None
                        and abs(reading.to_depth - reading.from_depth) > 1e-9
                    ):
                        y_top = plot_depth_y(hole_id, reading.from_depth)
                        y_bottom = plot_depth_y(hole_id, reading.to_depth)
                        ax.plot(
                            [x_profile, x_profile],
                            [y_top, y_bottom],
                            color=color,
                            linewidth=2.0,
                            solid_capstyle="round",
                            zorder=7,
                        )
                    if label_values:
                        marker_labels.append((x_profile, y, reading.display_label))

            if not marker_xs:
                continue
            ax.scatter(marker_xs, marker_ys, marker=marker, c=color, s=49, zorder=8)
            label_offsets = _resolve_parameter_label_offsets(ax, marker_labels)
            font_size = (
                _PARAMETER_LABEL_FONTSIZE_CONSULTING
                if getattr(self.profile, "layout", "") == "consulting_section"
                else _PARAMETER_LABEL_FONTSIZE
            )
            label_base_kwargs: dict[str, object] = {
                "textcoords": "offset points",
                "fontsize": font_size,
                "color": color,
                "zorder": 9,
                "clip_on": False,
                "ha": "left",
                "va": "center",
                "bbox": _PARAMETER_LABEL_BBOX,
            }
            leader_props = {
                "arrowstyle": "-",
                "color": color,
                "lw": 0.55,
                "linestyle": "--",
                "shrinkA": 2,
                "shrinkB": 1,
                "alpha": 0.7,
            }
            for (x_profile, y, label_text), (dx, dy, draw_leader) in zip(
                marker_labels, label_offsets, strict=True
            ):
                annotate_kwargs = {
                    **label_base_kwargs,
                    "xy": (x_profile, y),
                    "xytext": (dx, dy),
                }
                if draw_leader:
                    annotate_kwargs["arrowprops"] = leader_props
                ax.annotate(label_text, **annotate_kwargs)

            measured_holes = [hole_id for hole_id in transect_hole_ids if readings_by_hole.get(hole_id)]
            if use_segments and not across_gaps:
                for left_id, right_id in zip(transect_hole_ids, transect_hole_ids[1:], strict=False):
                    left_items = readings_by_hole.get(left_id, [])
                    right_items = readings_by_hole.get(right_id, [])
                    if not left_items or not right_items:
                        continue
                    x0 = transect_x[left_id]
                    x1 = transect_x[right_id]
                    used_right = [False] * len(right_items)
                    right_depths = [item.sample_depth for item in right_items]
                    for left_reading in left_items:
                        best_idx = _nearest_unused_by_depth(
                            right_depths, used_right, left_reading.sample_depth
                        )
                        if best_idx is None:
                            continue
                        used_right[best_idx] = True
                        right_reading = right_items[best_idx]
                        y0 = plot_depth_y(left_id, left_reading.sample_depth)
                        y1 = plot_depth_y(right_id, right_reading.sample_depth)
                        ax.plot([x0, x1], [y0, y1], color=color, linewidth=1.5, linestyle="--", zorder=7)
            elif len(measured_holes) >= 2:
                # Single sample per hole: classic fence (dense when across_gaps).
                # Multi-depth: band by nearest sample depth so deep contacts are not dropped.
                if all(len(readings_by_hole[hole_id]) == 1 for hole_id in measured_holes):
                    bands = [
                        [
                            (hole_id, readings_by_hole[hole_id][0])
                            for hole_id in measured_holes
                        ]
                    ]
                else:
                    bands = _cluster_parameter_bands_by_depth(
                        measured_holes, readings_by_hole, depth_tol=1.5
                    )
                for band in bands:
                    if len(band) < 2:
                        continue
                    band.sort(key=lambda item: float(profile_lookup[item[0]][0]))
                    fence_xs = [float(profile_lookup[hole_id][0]) for hole_id, _reading in band]
                    fence_ys = [
                        plot_depth_y(hole_id, reading.sample_depth) for hole_id, reading in band
                    ]
                    xs_arr = np.asarray(fence_xs, dtype=float)
                    ys_arr = np.asarray(fence_ys, dtype=float)
                    if across_gaps and len(xs_arr) >= 2:
                        x_dense = np.linspace(float(xs_arr.min()), float(xs_arr.max()), 100)
                        y_dense = np.interp(x_dense, xs_arr, ys_arr)
                        ax.plot(x_dense, y_dense, color=color, linewidth=1.5, linestyle="--", zorder=7)
                    else:
                        ax.plot(xs_arr, ys_arr, color=color, linewidth=1.5, linestyle="--", zorder=7)
            self.parameter_series_legend.append(
                {
                    "parameter": parameter,
                    "color": color,
                    "marker": marker,
                    "label": parameter.upper(),
                }
            )

    def _draw_compact_parameter_legend(self, ax) -> None:
        if not self.parameter_series_legend:
            return
        lines = []
        for entry in self.parameter_series_legend:
            lines.append(f"{entry['label']} ({entry['marker']})")
        ax.text(
            0.01,
            0.02,
            "Parameters: " + "; ".join(lines),
            transform=ax.transAxes,
            fontsize=7,
            color=LABEL_COLOR,
            va="bottom",
        )

    def _draw_legend(
        self,
        ax,
        style_cache: dict,
        lithology_codes: list[str],
        polygons: list[GeologicalPolygon],
    ) -> None:
        legend_handles = []
        for code in lithology_codes:
            style = self._resolve_style(code, style_cache)
            legend_handles.append(
                Patch(
                    facecolor=style.color,
                    edgecolor=POLYGON_EDGE_COLOR,
                    hatch=style.hatch or None,
                    linewidth=0.75,
                    label=code,
                )
            )
        if self._has_pinch_out:
            legend_handles.append(
                Patch(
                    facecolor="none",
                    edgecolor=POLYGON_EDGE_COLOR,
                    linestyle="--",
                    linewidth=1.0,
                    label="Inferred pinch-out",
                )
            )
        ax.legend(
            handles=legend_handles,
            title="Lithology",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            framealpha=0.95,
            edgecolor="#CBD5E1",
            fontsize=8,
            title_fontsize=9,
        )

    def _draw_ve_annotation(self, ax) -> None:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        ax.text(
            x_max - 0.02 * (x_max - x_min),
            y_min + 0.04 * (y_max - y_min),
            f"V.E. {self.vertical_exaggeration:.0f}×",
            ha="right",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=LABEL_COLOR,
            zorder=10,
        )

    def _draw_footers(self, fig: Figure) -> None:
        if self.disclaimer:
            fig.text(0.5, 0.01, self.disclaimer, ha="center", va="bottom", fontsize=8, color="#64748B", style="italic")
        footer_y = 0.045 if self.disclaimer else 0.01
        metadata_lines = self._metadata_footer_lines()
        if metadata_lines and self.profile.title_block:
            fig.text(0.5, footer_y, " | ".join(metadata_lines), ha="center", va="bottom", fontsize=7, color="#475569")
            footer_y += 0.02
        if self.overlap_pairs and self.profile.show_overlap_footer:
            fig.text(
                0.5,
                footer_y,
                f"Polygon overlap markers ({len(self.overlap_pairs)}): review layer correlation.",
                ha="center",
                va="bottom",
                fontsize=7,
                color=OVERLAP_MARKER_COLOR,
            )

    def _metadata_footer_lines(self) -> list[str]:
        if self.figure_metadata is None:
            return []
        meta = self.figure_metadata
        lines: list[str] = []
        if meta.coordinate_reference:
            lines.append(f"CRS: {meta.coordinate_reference}")
        if meta.elevation_datum:
            lines.append(f"Datum: {meta.elevation_datum}")
        lines.append(f"VE: {meta.vertical_exaggeration:.1f}×")
        if meta.transect_azimuth_deg is not None:
            lines.append(f"Azimuth: {meta.transect_azimuth_deg:.0f}°")
        lines.append(f"Holes: {meta.hole_count}")
        if meta.uses_placeholder_elevation:
            lines.append("WARNING: placeholder collar elevation")
        return lines

    def _savefig_kwargs(self) -> dict[str, object]:
        """Consulting uses fixed letter page; other layouts keep tight crop."""
        if getattr(self.profile, "layout", "") == "consulting_section":
            return {"bbox_inches": None, "pad_inches": 0.05}
        return {"bbox_inches": "tight"}

    def to_svg_bytes(self, fig: Figure) -> bytes:
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="svg",
            facecolor=fig.get_facecolor(),
            metadata={"Creator": "Cross Section Studio"},
            **self._savefig_kwargs(),
        )
        buffer.seek(0)
        return buffer.getvalue()

    def to_png_bytes(self, fig: Figure, *, dpi: int = 300) -> bytes:
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=dpi,
            facecolor=fig.get_facecolor(),
            **self._savefig_kwargs(),
        )
        buffer.seek(0)
        return buffer.getvalue()

    def to_pdf_bytes(self, fig: Figure) -> bytes:
        """Single-page vector PDF of the rendered figure."""
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="pdf",
            facecolor=fig.get_facecolor(),
            **self._savefig_kwargs(),
        )
        buffer.seek(0)
        return buffer.getvalue()

    def export_figure_bytes(
        self,
        figure: Figure,
        export_formats: frozenset[str],
        *,
        polygons: list[GeologicalPolygon],
        projected_df: pd.DataFrame,
        collar_depths: dict[str, float] | None = None,
        water_levels: Sequence[WaterLevel] | None = None,
        lithology_codes: Sequence[str] | None = None,
        qa_lines: Sequence[str] = (),
    ) -> tuple[bytes, bytes, bytes]:
        # savefig() draws on demand; an explicit canvas.draw() here doubled render cost.
        # Prefer cheaper formats first so a later failure still leaves cheaper bytes usable.
        ordered = [fmt for fmt in ("svg", "png", "pdf") if fmt in export_formats]
        svg_bytes = b""
        png_bytes = b""
        pdf_bytes = b""
        save_kwargs = self._savefig_kwargs()
        face = figure.get_facecolor()
        for fmt in ordered:
            if fmt == "svg":
                buffer = io.BytesIO()
                figure.savefig(
                    buffer,
                    format="svg",
                    facecolor=face,
                    metadata={"Creator": "Cross Section Studio"},
                    **save_kwargs,
                )
                buffer.seek(0)
                svg_bytes = buffer.getvalue()
            elif fmt == "png":
                buffer = io.BytesIO()
                figure.savefig(
                    buffer,
                    format="png",
                    dpi=300,
                    facecolor=face,
                    **save_kwargs,
                )
                buffer.seek(0)
                png_bytes = buffer.getvalue()
            elif fmt == "pdf":
                if self.profile.layout == "consulting_section":
                    buffer = io.BytesIO()
                    figure.savefig(
                        buffer,
                        format="pdf",
                        facecolor=face,
                        **save_kwargs,
                    )
                    buffer.seek(0)
                    pdf_bytes = buffer.getvalue()
                else:
                    # Lazy import: PDF backend/font tools are heavy and slow startup for SVG/PNG-only runs.
                    from report_export import export_section_pdf

                    pdf_bytes = export_section_pdf(
                        self,
                        polygons,
                        projected_df,
                        collar_depths=collar_depths,
                        water_levels=water_levels,
                        lithology_codes=lithology_codes,
                        qa_lines=qa_lines,
                        section_figure=figure,
                    )
        return svg_bytes, png_bytes, pdf_bytes

    def _transform_y(self, value: float) -> float:
        return value * self.vertical_exaggeration

    def _transform_ys(self, values: list[float] | np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.vertical_exaggeration

    def _draw_scale_bar(self, ax) -> None:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        bar_length = self.scale_bar_length_m
        if bar_length <= 0:
            return
        if self.profile.scale_bar_position == "bottom_right":
            x_start = x_max - 0.04 * (x_max - x_min) - bar_length
        else:
            x_start = x_min + 0.04 * (x_max - x_min)
        y_pos = y_min + 0.06 * (y_max - y_min)
        tick = 0.01 * (y_max - y_min)
        ax.plot([x_start, x_start + bar_length], [y_pos, y_pos], color=STICK_COLOR, linewidth=4, solid_capstyle="butt", zorder=9)
        ax.plot([x_start, x_start], [y_pos - tick, y_pos + tick], color=STICK_COLOR, linewidth=1.5, zorder=9)
        ax.plot([x_start + bar_length, x_start + bar_length], [y_pos - tick, y_pos + tick], color=STICK_COLOR, linewidth=1.5, zorder=9)
        ax.text(x_start + bar_length / 2.0, y_pos + 0.025 * (y_max - y_min), f"{bar_length:g} m", ha="center", va="bottom", fontsize=8, fontweight="bold", color=LABEL_COLOR, zorder=9)
