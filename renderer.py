"""Vector rendering for borehole cross-section profiles."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Literal, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from matplotlib.patches import Polygon as MplPolygon

from constants import (
    HATCH_LINE_COLOR,
    POLYGON_EDGE_COLOR,
    get_lithology_style,
)
from lithology_codes import collect_lithology_codes
from models import WaterLevel
from stratigraphy import GeologicalPolygon, PolygonOverlap

logger = logging.getLogger(__name__)

InterpretationMode = Literal["borehole_only", "interpolated"]

_SURFACE_COLOR = "#3D8B5F"
_STICK_COLOR = "#1F2937"
_LABEL_COLOR = "#111827"
_GRID_COLOR = "#D1D5DB"
_WATER_COLOR = "#2563EB"
_UNCERTAINTY_COLOR = "#FDE68A"
_PINCH_OUT_ALPHA = 0.72
_OVERLAP_MARKER_COLOR = "#DC2626"


@dataclass
class _LabelSpec:
    x: float
    y: float
    text: str
    dx: float = 0.0
    dy: float = 0.0
    draw_leader: bool = False


class CrossSectionRenderer:
    """Render geological polygons and borehole sticks to a matplotlib figure."""

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

    def render(
        self,
        polygons: list[GeologicalPolygon],
        projected_df: pd.DataFrame,
        collar_depths: dict[str, float] | None = None,
        water_levels: Sequence[WaterLevel] | None = None,
        *,
        lithology_codes: Sequence[str] | None = None,
    ) -> Figure:
        hatch_context = {
            "hatch.color": HATCH_LINE_COLOR,
            "hatch.linewidth": 0.65,
        }
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

    def _render_figure(
        self,
        polygons: list[GeologicalPolygon],
        projected_df: pd.DataFrame,
        collar_depths: dict[str, float] | None,
        *,
        water_levels: Sequence[WaterLevel] | None = None,
        lithology_codes: Sequence[str] | None = None,
    ) -> Figure:
        fig_width = 13.5 if self.show_legend else 12.0
        fig, ax = plt.subplots(figsize=(fig_width, 6.8))
        fig.patch.set_facecolor("#F8FAFC")
        ax.set_facecolor("#FFFFFF")

        if lithology_codes is None:
            lithology_codes = collect_lithology_codes(projected_df, polygons)
        else:
            lithology_codes = list(lithology_codes)
        style_cache = {
            code: get_lithology_style(code, use_hatch=self.show_hatches) for code in lithology_codes
        }

        hole_summary = (
            projected_df.groupby("hole_id", as_index=False)
            .agg(
                x_profile=("x_profile", "first"),
                collar_elevation=("collar_elevation", "first"),
                bottom_elevation=("bottom_elevation", "min"),
                offset_distance=("offset_distance", "first"),
            )
            .sort_values("x_profile")
        )

        x_span = (
            float(hole_summary["x_profile"].max() - hole_summary["x_profile"].min())
            if len(hole_summary) >= 2
            else 10.0
        )
        stick_half_width = max(x_span * 0.015, 0.8)
        ve = self.vertical_exaggeration

        if self.interpretation_mode == "interpolated":
            z_min = float(hole_summary["bottom_elevation"].min()) * ve
            z_max = float(hole_summary["collar_elevation"].max()) * ve
            self._draw_uncertainty_zones(ax, hole_summary, z_min, z_max)

        if len(hole_summary) >= 2:
            surface_x = hole_summary["x_profile"].to_numpy(dtype=float)
            surface_y = hole_summary["collar_elevation"].to_numpy(dtype=float)
            sample_count = max(20, min(100, len(surface_x) * 20))
            x_dense = np.linspace(surface_x.min(), surface_x.max(), sample_count)
            y_dense = np.interp(x_dense, surface_x, surface_y) * ve
            ax.fill_between(
                x_dense,
                y_dense,
                y_dense.max() + 0.08 * (y_dense.max() - y_dense.min()),
                facecolor="#E8F5EE",
                edgecolor="none",
                alpha=0.55,
                zorder=1,
            )
            ax.plot(
                x_dense,
                y_dense,
                color=_SURFACE_COLOR,
                linewidth=2.5,
                solid_capstyle="round",
                zorder=6,
                label="Ground surface",
            )

        for geo_polygon in polygons:
            style = style_cache[geo_polygon.lithology_code]
            coords = list(geo_polygon.polygon.exterior.coords)
            linestyle = "--" if geo_polygon.is_pinch_out else "-"
            alpha = _PINCH_OUT_ALPHA if geo_polygon.is_pinch_out else 0.92
            patch = MplPolygon(
                self._transform_coords(coords, ve),
                closed=True,
                facecolor=style.color,
                edgecolor=style.edge_color,
                linewidth=0.75,
                linestyle=linestyle,
                hatch=style.hatch or None,
                alpha=alpha,
                zorder=2,
            )
            ax.add_patch(patch)

        if self.overlap_pairs:
            self._draw_overlap_markers(ax, ve)

        if water_levels:
            self._draw_water_table(ax, hole_summary, water_levels)

        collar_depths = collar_depths or {}
        labels = self._build_borehole_labels(hole_summary, collar_depths)
        labels = self._resolve_label_collisions(labels)

        if self.show_stick_logs:
            self._draw_stick_logs(ax, projected_df, style_cache, stick_half_width, ve)

        for row in hole_summary.itertuples(index=False):
            x = float(row.x_profile)
            top = float(row.collar_elevation)
            bottom = float(row.bottom_elevation)
            ax.plot(
                [x, x],
                [top * ve, bottom * ve],
                color=_STICK_COLOR,
                linewidth=4.0,
                solid_capstyle="butt",
                zorder=5,
            )
            ax.plot(
                [x],
                [top * ve],
                marker="v",
                markersize=7,
                color=_SURFACE_COLOR,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=7,
            )
            ax.annotate(
                f"{top:.1f} m RL",
                xy=(x, top * ve),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=7,
                color=_LABEL_COLOR,
                ha="left",
                va="bottom",
                zorder=8,
            )

        for label in labels:
            if label.draw_leader:
                ax.plot(
                    [label.x, label.x + label.dx],
                    self._transform_ys([label.y, label.y + label.dy]),
                    color=_STICK_COLOR,
                    linewidth=0.8,
                    linestyle="--",
                    zorder=7,
                )
            ax.annotate(
                label.text,
                xy=(label.x, self._transform_y(label.y)),
                xytext=(label.x + label.dx, self._transform_y(label.y + label.dy)),
                textcoords="data",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="600",
                color=_LABEL_COLOR,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#CBD5E1",
                    "alpha": 0.92,
                },
                zorder=8,
            )

        self._draw_scale_bar(ax)

        if self.show_legend and lithology_codes:
            legend_handles = [
                Patch(
                    facecolor=style_cache[code].color,
                    edgecolor=POLYGON_EDGE_COLOR,
                    hatch=style_cache[code].hatch or None,
                    linewidth=0.75,
                    label=code,
                )
                for code in lithology_codes
            ]
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

        ax.set_xlabel("Distance along transect (m)", fontsize=10, labelpad=8)
        ax.set_ylabel("Elevation (m)", fontsize=10, labelpad=8)
        ax.set_title(self.title, fontsize=13, fontweight="bold", pad=12, color="#1E293B")
        ax.set_aspect("auto")
        ax.grid(True, linestyle="--", alpha=0.35, color=_GRID_COLOR, zorder=0)
        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")

        if self.disclaimer:
            fig.text(
                0.5,
                0.01,
                self.disclaimer,
                ha="center",
                va="bottom",
                fontsize=8,
                color="#64748B",
                style="italic",
            )

        if self.overlap_pairs:
            overlap_note = (
                f"Polygon overlap markers ({len(self.overlap_pairs)}): "
                "review layer correlation between adjacent holes."
            )
            fig.text(
                0.5,
                0.045 if self.disclaimer else 0.01,
                overlap_note,
                ha="center",
                va="bottom",
                fontsize=7,
                color=_OVERLAP_MARKER_COLOR,
            )

        bottom_margin = 0.14 if self.disclaimer and self.overlap_pairs else 0.12
        if self.show_legend and lithology_codes:
            fig.subplots_adjust(right=0.78, bottom=bottom_margin)
        else:
            fig.tight_layout(rect=(0, bottom_margin - 0.02, 1, 1))
        return fig

    def _draw_overlap_markers(self, ax, ve: float) -> None:
        for overlap in self.overlap_pairs[:12]:
            ax.plot(
                overlap.centroid_x,
                overlap.centroid_y * ve,
                marker="x",
                markersize=8,
                color=_OVERLAP_MARKER_COLOR,
                markeredgewidth=1.5,
                linestyle="none",
                zorder=9,
            )

    def _draw_stick_logs(
        self,
        ax,
        projected_df: pd.DataFrame,
        style_cache: dict,
        stick_half_width: float,
        ve: float,
    ) -> None:
        x_values = projected_df["x_profile"].to_numpy(dtype=float)
        tops = projected_df["top_elevation"].to_numpy(dtype=float) * ve
        bottoms = projected_df["bottom_elevation"].to_numpy(dtype=float) * ve
        codes = projected_df["lithology_code"].astype(str).to_numpy()
        width = 2.0 * stick_half_width
        for index in range(len(x_values)):
            code = codes[index]
            style = style_cache.get(code)
            if style is None:
                style = get_lithology_style(code, use_hatch=self.show_hatches)
                style_cache[code] = style
            rect = Rectangle(
                (x_values[index] - stick_half_width, tops[index]),
                width,
                bottoms[index] - tops[index],
                facecolor=style.color,
                edgecolor=style.edge_color,
                linewidth=0.5,
                hatch=style.hatch or None,
                alpha=0.95,
                zorder=4,
            )
            ax.add_patch(rect)

    def _draw_uncertainty_zones(
        self,
        ax,
        hole_summary: pd.DataFrame,
        y_min: float,
        y_max: float,
    ) -> None:
        if len(hole_summary) < 2:
            return
        x_values = hole_summary["x_profile"].to_numpy(dtype=float)
        offset_values = hole_summary["offset_distance"].to_numpy(dtype=float)
        spacing = np.abs(np.diff(x_values))
        max_offset = np.maximum(offset_values[:-1], offset_values[1:])
        uncertain = (spacing > self.uncertainty_spacing_m) | (
            max_offset > self.uncertainty_offset_m
        )
        for index in np.flatnonzero(uncertain):
            x_left = float(x_values[index])
            x_right = float(x_values[index + 1])
            rect = Rectangle(
                (min(x_left, x_right), y_min),
                abs(x_right - x_left),
                y_max - y_min,
                facecolor=_UNCERTAINTY_COLOR,
                edgecolor="none",
                alpha=0.28,
                zorder=0,
            )
            ax.add_patch(rect)

    def _draw_water_table(
        self,
        ax,
        hole_summary: pd.DataFrame,
        water_levels: Sequence[WaterLevel],
    ) -> None:
        profile_lookup = {
            str(row.hole_id): (float(row.x_profile), float(row.collar_elevation))
            for row in hole_summary.itertuples(index=False)
        }
        points: list[tuple[float, float]] = []
        for level in water_levels:
            profile = profile_lookup.get(level.hole_id)
            if profile is None:
                continue
            x_profile, collar_rl = profile
            points.append((x_profile, collar_rl - level.depth))
        if not points:
            return
        if len(points) == 1:
            x_profile, water_rl = points[0]
            ax.plot(
                x_profile,
                self._transform_y(water_rl),
                marker="o",
                color=_WATER_COLOR,
                markersize=6,
                linestyle="None",
                zorder=6,
            )
            return
        points.sort(key=lambda item: item[0])
        xs = np.array([point[0] for point in points], dtype=float)
        ys = np.array([point[1] for point in points], dtype=float)
        x_dense = np.linspace(xs.min(), xs.max(), 100)
        y_dense = np.interp(x_dense, xs, ys)
        ax.plot(
            x_dense,
            self._transform_ys(y_dense),
            color=_WATER_COLOR,
            linewidth=2.0,
            linestyle="--",
            zorder=6,
            label="Water table",
        )

    def to_svg_bytes(self, fig: Figure) -> bytes:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
        buffer.seek(0)
        return buffer.getvalue()

    def _transform_y(self, value: float) -> float:
        return value * self.vertical_exaggeration

    def _transform_ys(self, values: list[float] | np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.vertical_exaggeration

    def _transform_coords(
        self,
        coords: list[tuple[float, float]],
        ve: float | None = None,
    ) -> np.ndarray:
        multiplier = self.vertical_exaggeration if ve is None else ve
        array = np.asarray(coords, dtype=float)
        array[:, 1] *= multiplier
        return array

    def _build_borehole_labels(
        self,
        hole_summary: pd.DataFrame,
        collar_depths: dict[str, float],
    ) -> list[_LabelSpec]:
        labels: list[_LabelSpec] = []
        for row in hole_summary.itertuples(index=False):
            depth = collar_depths.get(row.hole_id)
            depth_text = f"{depth:.1f} m TD" if depth is not None else ""
            text = f"{row.hole_id}\n{depth_text}".strip()
            labels.append(
                _LabelSpec(
                    x=float(row.x_profile),
                    y=float(row.collar_elevation),
                    text=text,
                )
            )
        return labels

    def _resolve_label_collisions(self, labels: list[_LabelSpec]) -> list[_LabelSpec]:
        if not labels:
            return labels

        x_span = max(label.x for label in labels) - min(label.x for label in labels)
        threshold = max(5.0, x_span * 0.08)
        resolved = [label for label in labels]

        for index in range(1, len(resolved)):
            current = resolved[index]
            previous = resolved[index - 1]
            if abs(current.x - previous.x) < threshold:
                offset = 4.0 * index
                current.dy = offset
                current.draw_leader = True
                previous.dy = max(previous.dy, offset / 2.0)
                previous.draw_leader = True

        return resolved

    def _draw_scale_bar(self, ax) -> None:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        bar_length = self.scale_bar_length_m
        if bar_length <= 0:
            return

        x_start = x_min + 0.04 * (x_max - x_min)
        y_pos = y_min + 0.06 * (y_max - y_min)
        ax.plot(
            [x_start, x_start + bar_length],
            [y_pos, y_pos],
            color=_STICK_COLOR,
            linewidth=4,
            solid_capstyle="butt",
            zorder=9,
        )
        ax.plot(
            [x_start, x_start],
            [y_pos - 0.01 * (y_max - y_min), y_pos + 0.01 * (y_max - y_min)],
            color=_STICK_COLOR,
            linewidth=1.5,
            zorder=9,
        )
        ax.plot(
            [x_start + bar_length, x_start + bar_length],
            [y_pos - 0.01 * (y_max - y_min), y_pos + 0.01 * (y_max - y_min)],
            color=_STICK_COLOR,
            linewidth=1.5,
            zorder=9,
        )
        ax.text(
            x_start + bar_length / 2.0,
            y_pos + 0.025 * (y_max - y_min),
            f"{bar_length:g} m",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="600",
            color=_LABEL_COLOR,
            zorder=9,
        )
