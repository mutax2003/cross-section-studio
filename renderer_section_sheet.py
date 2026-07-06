"""Section-sheet layout drawing (mixin for CrossSectionRenderer)."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from lithology_codes import collect_lithology_codes
from models import WaterLevel
from render_theme import AXES_BG, FIGURE_BG, GRID_COLOR, LABEL_COLOR
from stratigraphy import GeologicalPolygon


class SectionSheetLayoutMixin:
    """Strater-like section sheet layout."""

    def _section_bottom_margin(self) -> float:
        margin = 0.17 if self.profile.title_block else 0.14
        if self.profile.show_column_headers:
            margin = max(margin, 0.26)
        return margin

    def _render_section_sheet(
        self,
        polygons: list[GeologicalPolygon],
        projected_df: pd.DataFrame,
        collar_depths: dict[str, float] | None,
        *,
        water_levels: Sequence[WaterLevel] | None = None,
        lithology_codes: Sequence[str] | None = None,
    ) -> Figure:
        fig_width = 14.0 if self.show_legend else 12.5
        fig, ax = plt.subplots(figsize=(fig_width, 7.2))
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

        self._draw_fence_polygons(ax, polygons, style_cache, ve, alpha=self.profile.fence_alpha)

        collar_depths = collar_depths or {}
        if self.profile.show_sky_fill or self.profile.show_ground_surface:
            self._draw_sky_and_surface(ax, hole_summary, collar_lookup)

        if self.show_stick_logs and self.profile.show_track_lithology:
            self._draw_track_lithology(ax, projected_df, style_cache, track_half, collar_lookup)
        if self.profile.show_track_border:
            self._draw_track_borders(ax, hole_summary, collar_depths, collar_lookup, track_half)
        if self.screen_intervals:
            self._draw_screen_intervals(
                ax,
                hole_summary,
                self.screen_intervals,
                collar_lookup,
                track_half,
            )
        if self.profile.show_contact_ticks:
            self._draw_contact_ticks(ax, projected_df, track_half, collar_lookup)
        self._draw_deviated_centerlines(ax, projected_df, collar_lookup)
        self._draw_raster_strips(ax, ctx.x_by_hole, collar_lookup, track_half)

        if self.profile.show_overlap_markers and self.overlap_pairs:
            self._draw_overlap_markers(ax, collar_lookup)
        if water_levels:
            self._draw_water_table(ax, hole_summary, water_levels, collar_lookup)
        self._draw_faults(ax, collar_lookup)
        self._draw_unconformities(ax, collar_lookup)
        self._draw_environmental_markers(ax, ctx.x_by_hole, collar_lookup)
        if self.profile.show_eol_bar:
            self._draw_eol_bars(ax, hole_summary, collar_depths, collar_lookup, track_half)

        self._draw_scale_bar(ax)
        if self.profile.show_ve_annotation:
            self._draw_ve_annotation(ax)

        if self.show_legend and lithology_codes:
            self._draw_legend(ax, style_cache, lithology_codes, polygons)

        y_label = (
            "Depth below collar (m)"
            if self.profile.y_axis_mode == "depth_below_collar"
            else "Elevation (m RL)"
        )
        ax.set_ylabel(y_label, fontsize=10, labelpad=8)
        ax.set_title(self.title, fontsize=14, fontweight="bold", pad=14, color=LABEL_COLOR)
        ax.set_aspect("auto")
        if self.profile.show_grid:
            ax.grid(True, linestyle="--", alpha=0.35, color=GRID_COLOR, zorder=0)
        else:
            ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color("#94A3B8")
            spine.set_linewidth(1.2)

        if self.profile.y_axis_mode == "depth_below_collar":
            ax.invert_yaxis()

        bottom_margin = self._section_bottom_margin()
        if self.show_legend and lithology_codes:
            fig.subplots_adjust(right=0.78, bottom=bottom_margin)
        else:
            fig.tight_layout(rect=(0, bottom_margin - 0.02, 1, 1))

        if self.profile.show_column_headers:
            self._draw_column_headers(ax, hole_summary, collar_depths, collar_lookup)
            fig.supxlabel(
                "Distance along transect (m)",
                fontsize=10,
                y=0.03,
                color=LABEL_COLOR,
            )
        else:
            ax.set_xlabel("Distance along transect (m)", fontsize=10, labelpad=8)

        self._draw_footers(fig)
        return fig
