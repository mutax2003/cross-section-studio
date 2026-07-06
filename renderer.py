"""Vector rendering for borehole cross-section profiles."""

from __future__ import annotations

import io
import logging
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
    OVERLAP_MARKER_COLOR,
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
        self.profile = render_profile or SECTION_SHEET_PROFILE
        self.raster_log_strips = tuple(raster_log_strips)
        self.consulting_title_block = consulting_title_block
        self.screen_intervals = tuple(screen_intervals)
        self.vertical_gradients = tuple(vertical_gradients)
        self.water_series_legend: list[WaterSeriesLegendEntry] = []
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

    def _draw_fence_polygons(
        self,
        ax,
        polygons: list[GeologicalPolygon],
        style_cache: dict,
        ve: float,
        *,
        alpha: float,
    ) -> None:
        if self.interpretation_mode not in {"interpolated", "correlation_lines"}:
            return
        if self.interpretation_mode == "correlation_lines":
            line_groups: dict[tuple[str, str], list[np.ndarray]] = {}
            for geo_polygon in polygons:
                style = self._resolve_style(geo_polygon.lithology_code, style_cache)
                coords = np.asarray(geo_polygon.polygon.exterior.coords, dtype=float)
                line_style = "--" if geo_polygon.is_pinch_out else "-"
                top_coords = self._transform_coords(
                    [(coords[0, 0], coords[0, 1]), (coords[1, 0], coords[1, 1])], ve
                )
                bottom_coords = self._transform_coords(
                    [(coords[3, 0], coords[3, 1]), (coords[2, 0], coords[2, 1])], ve
                )
                style_key = (style.edge_color, line_style)
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
            verts = self._transform_coords(coords, ve)
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
        y_top = y_dense.max() + 0.12 * max(y_dense.max() - y_dense.min(), 1.0)
        if self.profile.show_sky_fill:
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
    ) -> None:
        self._draw_lithology_interval_rects(
            ax,
            projected_df,
            style_cache,
            track_half_width,
            collar_lookup,
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
    ) -> None:
        if projected_df.empty:
            return
        collars = self._collar_values(
            projected_df["hole_id"],
            projected_df["collar_elevation"],
            collar_lookup,
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
        water_color: str | None = None,
        profile_lookup: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if hole_summary.empty:
            return
        if profile_lookup is None:
            profile_lookup = self._profile_lookup(hole_summary, collar_lookup)
        series_groups = _group_water_levels(water_levels, profile_lookup)
        holes_with_water = {level.hole_id for levels in series_groups.values() for level in levels}
        if label_dry_wells:
            dry_lookup = {
                hole_id: profile
                for hole_id, profile in profile_lookup.items()
                if hole_id not in holes_with_water
            }
            if dry_lookup:
                dry_x = np.fromiter((p[0] for p in dry_lookup.values()), dtype=float, count=len(dry_lookup))
                dry_collars = np.fromiter((p[1] for p in dry_lookup.values()), dtype=float, count=len(dry_lookup))
                dry_y = self._plot_y_values(dry_collars - 1.0, dry_collars)
                for x_profile, y in zip(dry_x, dry_y, strict=True):
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
        elev_suffix = " masl" if self.profile.layout == "consulting_section" else ""
        for series_id, levels in sorted(series_groups.items()):
            first = levels[0]
            default_color, default_marker, default_label = consulting_gw_series_style(
                series_id,
                first.series_label,
            )
            color = first.color or water_color or default_color
            marker_key = (first.marker or default_marker).lower()
            marker = _GW_MARKER_MAP.get(marker_key, marker_key)
            label = first.series_label or default_label or series_id
            xs: list[float] = []
            water_rls: list[float] = []
            collars: list[float] = []
            for level in levels:
                profile = profile_lookup.get(level.hole_id)
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
            order = np.argsort(xs_arr)
            xs = xs_arr[order]
            water_rls = water_arr[order]
            collars = collar_arr[order]
            ys = self._plot_y_values(water_rls, collars)
            ax.scatter(xs, ys, marker=marker, c=color, s=49, zorder=7)
            if label_elevations:
                suffix = elev_suffix
                for x_profile, water_rl, y in zip(xs, water_rls, ys, strict=True):
                    ax.annotate(
                        f"{water_rl:.3f}{suffix}",
                        xy=(float(x_profile), float(y)),
                        xytext=(4, -8),
                        textcoords="offset points",
                        fontsize=7,
                        color=color,
                        zorder=8,
                    )
            if len(xs) >= 2 and interpolate:
                x_dense = np.linspace(float(xs.min()), float(xs.max()), 100)
                y_dense = np.interp(x_dense, xs, ys)
                ax.plot(x_dense, y_dense, color=color, linewidth=2.0, linestyle="--", zorder=6)
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

    def _draw_overlap_markers(self, ax, collar_lookup: dict[str, float]) -> None:
        if not self.overlap_pairs:
            return
        ref_collar = next(iter(collar_lookup.values()), 0.0)
        overlaps = self.overlap_pairs[:12]
        xs = np.fromiter((overlap.centroid_x for overlap in overlaps), dtype=float, count=len(overlaps))
        centroid_ys = np.fromiter((overlap.centroid_y for overlap in overlaps), dtype=float, count=len(overlaps))
        ys = self._plot_y_values(centroid_ys, np.full(len(overlaps), ref_collar))
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

    def _draw_faults(self, ax, collar_lookup: dict[str, float]) -> None:
        ref_collar = next(iter(collar_lookup.values()), 0.0)
        for fault in self.faults:
            if len(fault.trace_points) < 2:
                continue
            xs, ys = zip(*fault.trace_points)
            plot_ys = self._plot_y_values(np.asarray(ys, dtype=float), np.full(len(ys), ref_collar))
            ax.plot(xs, plot_ys, color="#B91C1C", linewidth=2.0, linestyle="-.", zorder=8)

    def _draw_unconformities(self, ax, collar_lookup: dict[str, float]) -> None:
        ref_collar = next(iter(collar_lookup.values()), 0.0)
        for surface in self.unconformities:
            if len(surface.elevation_profile) < 2:
                continue
            xs, ys = zip(*surface.elevation_profile)
            plot_ys = self._plot_y_values(np.asarray(ys, dtype=float), np.full(len(ys), ref_collar))
            ax.plot(xs, plot_ys, color="#7C3AED", linewidth=1.8, linestyle=":", zorder=7)

    def _draw_environmental_markers(
        self,
        ax,
        x_by_hole: dict[str, float],
        collar_lookup: dict[str, float],
    ) -> None:
        if not self.environmental_readings:
            return
        for reading in self.environmental_readings:
            x_profile = x_by_hole.get(reading.hole_id)
            collar = collar_lookup.get(reading.hole_id)
            if x_profile is None or collar is None:
                continue
            mid_depth = (reading.from_depth + reading.to_depth) / 2.0
            ax.plot(
                x_profile,
                self._plot_y(collar - mid_depth, collar),
                marker="D",
                color="#EA580C",
                markersize=5,
                linestyle="None",
                zorder=8,
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
        if any(polygon.is_pinch_out for polygon in polygons):
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

    def to_svg_bytes(self, fig: Figure) -> bytes:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor(), metadata={"Creator": "Cross Section Studio"})
        buffer.seek(0)
        return buffer.getvalue()

    def to_png_bytes(self, fig: Figure, *, dpi: int = 300) -> bytes:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
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
        from report_export import export_section_pdf

        svg_bytes = self.to_svg_bytes(figure) if "svg" in export_formats else b""
        png_bytes = self.to_png_bytes(figure, dpi=300) if "png" in export_formats else b""
        pdf_bytes = (
            export_section_pdf(
                self,
                polygons,
                projected_df,
                collar_depths=collar_depths,
                water_levels=water_levels,
                lithology_codes=lithology_codes,
                qa_lines=qa_lines,
                section_figure=figure,
            )
            if "pdf" in export_formats
            else b""
        )
        return svg_bytes, png_bytes, pdf_bytes

    def _transform_y(self, value: float) -> float:
        return value * self.vertical_exaggeration

    def _transform_ys(self, values: list[float] | np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.vertical_exaggeration

    def _transform_coords(self, coords: list[tuple[float, float]], ve: float | None = None) -> np.ndarray:
        multiplier = self.vertical_exaggeration if ve is None else ve
        array = np.asarray(coords, dtype=float)
        array[:, 1] *= multiplier
        return array

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
