"""Chart layout drawing (mixin for CrossSectionRenderer)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from lithology_codes import collect_lithology_codes
from models import WaterLevel
from render_profiles import CHART_PROFILE
from render_theme import AXES_BG, FIGURE_BG, GRID_COLOR, LABEL_COLOR, STICK_COLOR, SURFACE_COLOR
from stratigraphy import GeologicalPolygon


@dataclass
class _LabelSpec:
    x: float
    y: float
    text: str
    dx: float = 0.0
    dy: float = 0.0
    draw_leader: bool = False


class ChartLayoutMixin:
    """Legacy debug chart layout."""

    def _render_chart_layout(
        self,
        polygons: list[GeologicalPolygon],
        projected_df: pd.DataFrame,
        collar_depths: dict[str, float] | None,
        *,
        water_levels: Sequence[WaterLevel] | None = None,
        lithology_codes: Sequence[str] | None = None,
    ) -> Figure:
        chart_profile = CHART_PROFILE
        original = self.profile
        self.profile = chart_profile
        try:
            fig_width = 13.5 if self.show_legend else 12.0
            fig, ax = plt.subplots(figsize=(fig_width, 6.8))
            fig.patch.set_facecolor(FIGURE_BG)
            ax.set_facecolor(AXES_BG)

            if lithology_codes is None:
                lithology_codes = collect_lithology_codes(projected_df, polygons)
            else:
                lithology_codes = list(lithology_codes)
            style_cache = self._style_cache_for(lithology_codes)
            ctx = self._hole_context(projected_df)
            hole_summary = ctx.summary
            collar_lookup = ctx.collar_lookup
            track_half = ctx.track_half
            ve = self.vertical_exaggeration

            if self.interpretation_mode in {"interpolated", "correlation_lines"}:
                z_min, z_max = self._uncertainty_y_bounds(hole_summary)
                self._draw_uncertainty_zones(ax, hole_summary, z_min, z_max, collar_lookup)

            if self.profile.show_ground_surface and len(hole_summary) >= 2:
                self._draw_sky_and_surface(ax, hole_summary, collar_lookup)

            self._draw_fence_polygons(ax, polygons, style_cache, ve, alpha=0.92)

            if self.profile.show_overlap_markers and self.overlap_pairs:
                self._draw_overlap_markers(ax, collar_lookup)
            if water_levels:
                self._draw_water_table(ax, hole_summary, water_levels, collar_lookup)
            self._draw_faults(ax, collar_lookup)
            self._draw_unconformities(ax, collar_lookup)
            self._draw_environmental_markers(ax, ctx.x_by_hole, collar_lookup)

            collar_depths = collar_depths or {}
            labels = self._resolve_label_collisions(
                self._build_borehole_labels(hole_summary, collar_depths)
            )
            if self.show_stick_logs:
                self._draw_track_lithology(ax, projected_df, style_cache, track_half, collar_lookup)

            if self.profile.show_centerline:
                for row in hole_summary.itertuples(index=False):
                    x = float(row.x_profile)
                    top = self._plot_y(float(row.collar_elevation), float(row.collar_elevation))
                    bottom = self._plot_y(float(row.bottom_elevation), float(row.collar_elevation))
                    ax.vlines(x, bottom, top, colors=STICK_COLOR, linewidth=4.0, zorder=5)
                    ax.plot(x, top, linestyle="none", marker="v", markersize=7, color=SURFACE_COLOR, zorder=7)
                    ax.annotate(
                        f"{float(row.collar_elevation):.1f} m RL",
                        xy=(x, top),
                        xytext=(6, 4),
                        textcoords="offset points",
                        fontsize=7,
                        color=LABEL_COLOR,
                        zorder=8,
                    )

            for label in labels:
                if label.draw_leader:
                    ax.plot(
                        [label.x, label.x + label.dx],
                        [
                            self._plot_y(label.y, label.y),
                            self._plot_y(label.y + label.dy, label.y),
                        ],
                        color=STICK_COLOR,
                        linewidth=0.8,
                        linestyle="--",
                        zorder=7,
                    )
                ax.annotate(
                    label.text,
                    xy=(label.x, self._plot_y(label.y, label.y)),
                    xytext=(label.x + label.dx, self._plot_y(label.y + label.dy, label.y)),
                    textcoords="data",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color=LABEL_COLOR,
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
                self._draw_legend(ax, style_cache, lithology_codes, polygons)

            ax.set_xlabel("Distance along transect (m)", fontsize=10, labelpad=8)
            ax.set_ylabel("Elevation (m)", fontsize=10, labelpad=8)
            ax.set_title(self.title, fontsize=13, fontweight="bold", pad=12, color=LABEL_COLOR)
            ax.set_aspect("auto")
            ax.grid(True, linestyle="--", alpha=0.35, color=GRID_COLOR, zorder=0)
            for spine in ax.spines.values():
                spine.set_color("#CBD5E1")

            self._draw_footers(fig)
            bottom_margin = 0.16
            if self.show_legend and lithology_codes:
                fig.subplots_adjust(right=0.78, bottom=bottom_margin)
            else:
                fig.tight_layout(rect=(0, bottom_margin - 0.02, 1, 1))
            return fig
        finally:
            self.profile = original

    def _build_borehole_labels(
        self,
        hole_summary: pd.DataFrame,
        collar_depths: dict[str, float],
    ) -> list[_LabelSpec]:
        labels: list[_LabelSpec] = []
        for row in hole_summary.itertuples(index=False):
            depth = collar_depths.get(row.hole_id)
            depth_text = f"{depth:.1f} m TD" if depth is not None else ""
            labels.append(
                _LabelSpec(
                    x=float(row.x_profile),
                    y=float(row.collar_elevation),
                    text=f"{row.hole_id}\n{depth_text}".strip(),
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
