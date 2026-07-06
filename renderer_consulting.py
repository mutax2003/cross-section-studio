"""Consulting-section layout drawing (mixin for CrossSectionRenderer)."""

from __future__ import annotations

import io
import logging
from typing import Sequence

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyArrow, Rectangle
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter, MultipleLocator

from constants import get_lithology_style
from lithology_codes import collect_lithology_codes
from models import ConsultingTitleBlock, VerticalGradient, WaterLevel
from stratigraphy import GeologicalPolygon
from render_theme import (
    CONSULTING_FIGURE_BG,
    CONSULTING_NM_COLOR,
    CONSULTING_SCALE_BAR_M,
    CONSULTING_SURFACE_COLOR,
    CONSULTING_WATER_COLOR,
    DEFAULT_CONSULTING_NOTES,
    LABEL_COLOR,
    REPORT_GRID_ALPHA,
    REPORT_GRID_COLOR,
    STICK_COLOR,
    TRACK_BORDER_COLOR,
    TRACK_FILL_COLOR,
    consulting_section_title,
    water_has_multiple_series,
)

logger = logging.getLogger(__name__)


class ConsultingLayoutMixin:
    """Consulting report-sheet layout methods. Expects CrossSectionRenderer attributes."""

    def _render_consulting_section(
        self,
        polygons: list[GeologicalPolygon],
        projected_df: pd.DataFrame,
        collar_depths: dict[str, float] | None,
        *,
        water_levels: Sequence[WaterLevel] | None = None,
        lithology_codes: Sequence[str] | None = None,
    ) -> Figure:
        title_block = self.consulting_title_block or ConsultingTitleBlock(section_label=self.title)
        if not title_block.notes:
            title_block = title_block.model_copy(update={"notes": DEFAULT_CONSULTING_NOTES})

        ctx = self._hole_context(projected_df)
        fig_width = max(11.0, min(24.0, 8.0 + ctx.x_span / 25.0))
        fig = plt.figure(figsize=(fig_width, 8.5))
        fig.patch.set_facecolor(CONSULTING_FIGURE_BG)
        grid = GridSpec(3, 1, figure=fig, height_ratios=[62, 8, 22], hspace=0.06)
        ax = fig.add_subplot(grid[0, 0])
        sub_gs = grid[1, 0].subgridspec(1, 3, width_ratios=[35, 30, 35], wspace=0.08)
        ax_scale = fig.add_subplot(sub_gs[0, 0])
        ax_center = fig.add_subplot(sub_gs[0, 1])
        ax_notes = fig.add_subplot(sub_gs[0, 2])
        ax_block = fig.add_subplot(grid[2, 0])
        ax.set_facecolor(CONSULTING_FIGURE_BG)

        with mpl.rc_context({"font.family": "sans-serif", "font.size": 9}):
            if lithology_codes is None:
                lithology_codes = collect_lithology_codes(projected_df, polygons)
            else:
                lithology_codes = list(lithology_codes)
            style_cache = self._style_cache_for(lithology_codes)

            hole_summary = ctx.summary
            collar_lookup = ctx.collar_lookup
            track_half = ctx.track_half
            ve = self.vertical_exaggeration
            collar_depths = collar_depths or {}
            water_levels_list = list(water_levels or [])
            profile_lookup = ctx.profile_lookup

            self._draw_fence_polygons(ax, polygons, style_cache, ve, alpha=self.profile.fence_alpha)
            self._draw_consulting_surface(ax, hole_summary, collar_lookup)
            self._draw_well_columns(ax, hole_summary, collar_depths, collar_lookup, track_half)
            self._draw_screen_intervals(
                ax,
                hole_summary,
                self.screen_intervals,
                collar_lookup,
                track_half,
                profile_lookup=profile_lookup,
            )
            multi_series = water_has_multiple_series(water_levels_list)
            self._draw_water_table(
                ax,
                hole_summary,
                water_levels_list,
                collar_lookup,
                label_elevations=True,
                label_dry_wells=not multi_series,
                water_color=CONSULTING_WATER_COLOR,
                profile_lookup=profile_lookup,
            )
            self._draw_vertical_gradients(
                ax,
                hole_summary,
                self.vertical_gradients,
                water_levels_list,
                collar_lookup,
                profile_lookup=profile_lookup,
            )
            self._draw_well_id_labels(ax, hole_summary)
            self._draw_transect_end_labels(ax, title_block)
            self._draw_faults(ax, collar_lookup)
            self._draw_unconformities(ax, collar_lookup)

            y_label = (
                "Depth below collar (m)"
                if self.profile.y_axis_mode == "depth_below_collar"
                else (title_block.y_axis_label or self.profile.y_axis_label or "ELEVATION (m)")
            )
            ax.set_xlabel("DISTANCE (m)", fontsize=10, labelpad=6, color=LABEL_COLOR)
            ax.set_ylabel(y_label, fontsize=10, labelpad=6, color=LABEL_COLOR)
            ax.set_aspect("auto")
            for spine in ax.spines.values():
                spine.set_color("#374151")
                spine.set_linewidth(1.0)

            if self.profile.y_axis_mode == "depth_below_collar":
                ax.invert_yaxis()

            ax_right: plt.Axes | None = None
            if self.profile.show_dual_y_axes:
                ax_right = ax.twinx()
                ax_right.set_ylim(ax.get_ylim())
                ax_right.set_ylabel(y_label, fontsize=10, labelpad=6, color=LABEL_COLOR)
                for spine in ax_right.spines.values():
                    spine.set_color("#374151")
                    spine.set_linewidth(1.0)

            if self.profile.show_report_grid:
                x_grid = 20.0 if ctx.x_span > 200.0 else float(getattr(self.profile, "x_major_grid_m", 10.0))
                self._apply_report_grid(ax, ax_right, consulting=True, x_major_step=x_grid)

            self._draw_subtitle_band(ax_scale, ax_center, ax_notes, title_block)
            self._draw_cad_title_block(ax_block, style_cache, lithology_codes, title_block)
        return fig

    def _apply_report_grid(self, ax, ax_right=None, *, consulting: bool = False, x_major_step: float | None = None) -> None:
        ve = self.vertical_exaggeration
        y_step = max(ve, 0.5)
        y_locator = MultipleLocator(y_step)
        ax.yaxis.set_major_locator(y_locator)
        if self.profile.y_axis_mode == "elevation_rl":
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos, v=ve: f"{value / v:.0f}"))
            if ax_right is not None:
                ax_right.yaxis.set_major_locator(y_locator)
                ax_right.yaxis.set_major_formatter(
                    FuncFormatter(lambda value, _pos, v=ve: f"{value / v:.0f}")
                )
        elif ax_right is not None:
            ax_right.yaxis.set_major_locator(y_locator)
        ax.xaxis.set_major_locator(MultipleLocator(x_major_step if x_major_step is not None else float(getattr(self.profile, "x_major_grid_m", 10.0))))
        ax.grid(True, which="major", color=REPORT_GRID_COLOR, alpha=REPORT_GRID_ALPHA, linewidth=0.6, zorder=0)
        if consulting:
            minor_y = max(ve * 0.5, 0.25)
            ax.yaxis.set_minor_locator(MultipleLocator(minor_y))
            ax.xaxis.set_minor_locator(MultipleLocator(5.0))
            ax.grid(True, which="minor", color=REPORT_GRID_COLOR, alpha=0.45, linewidth=0.35, zorder=0)
            ax.tick_params(axis="both", which="major", labelsize=8)
            if ax_right is not None:
                ax_right.tick_params(axis="y", which="major", labelsize=8)

    def _draw_well_columns(
        self,
        ax,
        hole_summary: pd.DataFrame,
        collar_depths: dict[str, float],
        collar_lookup: dict[str, float],
        track_half: float,
    ) -> None:
        geometry = self._well_rect_geometry(
            hole_summary, collar_depths, collar_lookup, track_half
        )
        if geometry is None:
            return
        self._add_rect_collection(
            ax,
            geometry,
            facecolors=TRACK_FILL_COLOR,
            edgecolors=TRACK_BORDER_COLOR,
            linewidths=0.8,
            zorder=8,
        )

    def _draw_consulting_surface(
        self,
        ax,
        hole_summary: pd.DataFrame,
        collar_lookup: dict[str, float],
    ) -> None:
        if len(hole_summary) < 2:
            return
        surface_x = hole_summary["x_profile"].to_numpy(dtype=float)
        if self.profile.y_axis_mode == "depth_below_collar":
            surface_y = np.zeros(len(hole_summary), dtype=float)
        else:
            collars = self._collar_values(
                hole_summary["hole_id"],
                hole_summary["collar_elevation"],
                collar_lookup,
            )
            surface_y = self._plot_y_values(collars, collars)
        sample_count = max(20, min(100, len(surface_x) * 20))
        x_dense = np.linspace(surface_x.min(), surface_x.max(), sample_count)
        y_dense = np.interp(x_dense, surface_x, surface_y)
        ax.plot(
            x_dense,
            y_dense,
            color=CONSULTING_SURFACE_COLOR,
            linewidth=1.0,
            solid_capstyle="round",
            zorder=6,
        )

    def _draw_vertical_gradients(
        self,
        ax,
        hole_summary: pd.DataFrame,
        vertical_gradients: Sequence[VerticalGradient],
        water_levels: Sequence[WaterLevel],
        collar_lookup: dict[str, float],
        *,
        profile_lookup: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if not vertical_gradients or hole_summary.empty:
            return
        water_depth_by_hole = {level.hole_id: level.depth for level in water_levels}
        if profile_lookup is None:
            profile_lookup = self._profile_lookup(hole_summary, collar_lookup)
        arrow_len = 0.12 * self.vertical_exaggeration
        for gradient in vertical_gradients:
            profile = profile_lookup.get(gradient.hole_id)
            if profile is None:
                continue
            x_profile, collar_rl = profile
            water_depth = water_depth_by_hole.get(gradient.hole_id, 1.0)
            anchor_rl = collar_rl - water_depth
            y = self._plot_y(anchor_rl, collar_rl)
            if gradient.direction == "up":
                arrow = FancyArrow(
                    float(x_profile),
                    float(y - arrow_len * 0.5),
                    0.0,
                    float(arrow_len),
                    width=0.35,
                    head_width=0.9,
                    head_length=0.25 * self.vertical_exaggeration,
                    length_includes_head=True,
                    facecolor=CONSULTING_WATER_COLOR,
                    edgecolor=CONSULTING_WATER_COLOR,
                    linewidth=0.0,
                    zorder=10,
                )
            else:
                arrow = FancyArrow(
                    float(x_profile),
                    float(y + arrow_len * 0.5),
                    0.0,
                    float(-arrow_len),
                    width=0.35,
                    head_width=0.9,
                    head_length=0.25 * self.vertical_exaggeration,
                    length_includes_head=True,
                    facecolor=CONSULTING_WATER_COLOR,
                    edgecolor=CONSULTING_WATER_COLOR,
                    linewidth=0.0,
                    zorder=10,
                )
            ax.add_patch(arrow)

    def _draw_thin_well_sticks(
        self,
        ax,
        hole_summary: pd.DataFrame,
        collar_depths: dict[str, float],
        collar_lookup: dict[str, float],
    ) -> None:
        extents = self._well_extents(hole_summary, collar_depths, collar_lookup)
        if extents is None:
            return
        x_values, top_y, bottom_y = extents
        segments = np.stack(
            [
                np.column_stack([x_values, top_y]),
                np.column_stack([x_values, bottom_y]),
            ],
            axis=1,
        )
        collection = LineCollection(segments, colors=STICK_COLOR, linewidths=1.2, zorder=7)
        ax.add_collection(collection)

    def _draw_well_id_labels(self, ax, hole_summary: pd.DataFrame) -> None:
        header_transform = ax.get_xaxis_transform()
        for row in hole_summary.itertuples(index=False):
            ax.text(
                float(row.x_profile),
                1.02,
                str(row.hole_id),
                transform=header_transform,
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color=LABEL_COLOR,
                clip_on=False,
                zorder=10,
            )

    def _transect_endpoint_lines(self, title_block: ConsultingTitleBlock) -> tuple[tuple[str, str], tuple[str, str]]:
        start_primary = title_block.transect_start_primary or title_block.transect_start_label
        start_secondary = title_block.transect_start_secondary
        end_primary = title_block.transect_end_primary or title_block.transect_end_label
        end_secondary = title_block.transect_end_secondary
        return (start_primary, start_secondary), (end_primary, end_secondary)

    def _draw_transect_end_labels(self, ax, title_block: ConsultingTitleBlock) -> None:
        (start_primary, start_secondary), (end_primary, end_secondary) = self._transect_endpoint_lines(
            title_block
        )
        if not start_primary and not start_secondary and not end_primary and not end_secondary:
            return
        header_transform = ax.get_xaxis_transform()
        if start_primary or start_secondary:
            start_lines = [line for line in (start_primary, start_secondary) if line]
            ax.text(
                0.0,
                1.06,
                "\n".join(start_lines),
                transform=header_transform,
                ha="left",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color=LABEL_COLOR,
                clip_on=False,
                zorder=10,
                linespacing=0.9,
            )
        if end_primary or end_secondary:
            end_lines = [line for line in (end_primary, end_secondary) if line]
            ax.text(
                1.0,
                1.06,
                "\n".join(end_lines),
                transform=header_transform,
                ha="right",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color=LABEL_COLOR,
                clip_on=False,
                zorder=10,
                linespacing=0.9,
            )

    def _draw_subtitle_band(
        self,
        ax_scale,
        ax_center,
        ax_notes,
        title_block: ConsultingTitleBlock,
    ) -> None:
        for panel in (ax_scale, ax_center, ax_notes):
            panel.set_axis_off()
            panel.set_xlim(0, 1)
            panel.set_ylim(0, 1)

        map_scale = title_block.map_scale or "1:1000"
        scale_bar_m = title_block.scale_bar_m or CONSULTING_SCALE_BAR_M
        bar_x = 0.02
        bar_y = 0.62
        bar_w = 0.88
        tick_step = 10.0
        tick_marks = tuple(np.arange(0.0, scale_bar_m + tick_step * 0.5, tick_step))
        for tick_m in tick_marks:
            tick_x = bar_x + (tick_m / scale_bar_m) * bar_w
            ax_scale.plot(
                [tick_x, tick_x],
                [bar_y - 0.06, bar_y + 0.06],
                color=STICK_COLOR,
                linewidth=1.0,
                transform=ax_scale.transAxes,
                clip_on=False,
            )
            ax_scale.text(
                tick_x,
                bar_y - 0.12,
                f"{int(tick_m)}",
                ha="center",
                va="top",
                fontsize=7,
                color=LABEL_COLOR,
                transform=ax_scale.transAxes,
            )
        ax_scale.plot(
            [bar_x, bar_x + bar_w],
            [bar_y, bar_y],
            color=STICK_COLOR,
            linewidth=2.5,
            solid_capstyle="butt",
            transform=ax_scale.transAxes,
            clip_on=False,
        )
        ax_scale.text(
            bar_x + bar_w / 2.0,
            bar_y - 0.22,
            "Metres",
            ha="center",
            va="top",
            fontsize=7,
            color=LABEL_COLOR,
            transform=ax_scale.transAxes,
        )
        ax_scale.text(
            bar_x + bar_w / 2.0,
            0.12,
            f"SCALE {map_scale}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            color=LABEL_COLOR,
            transform=ax_scale.transAxes,
        )

        section_title = consulting_section_title(title_block.section_label or self.title)
        ax_center.text(
            0.5,
            0.72,
            section_title,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=LABEL_COLOR,
            transform=ax_center.transAxes,
        )
        ax_center.plot(
            [0.12, 0.88],
            [0.64, 0.64],
            color=LABEL_COLOR,
            linewidth=0.8,
            transform=ax_center.transAxes,
            clip_on=False,
        )
        ve_text = f"{self.vertical_exaggeration:.0f}× VERTICAL EXAGGERATION"
        ax_center.text(
            0.5,
            0.28,
            ve_text,
            ha="center",
            va="center",
            fontsize=8,
            color=LABEL_COLOR,
            transform=ax_center.transAxes,
        )

        notes = title_block.notes or DEFAULT_CONSULTING_NOTES
        ax_notes.text(
            0.02,
            0.92,
            "NOTES:",
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=LABEL_COLOR,
            transform=ax_notes.transAxes,
        )
        note_y = 0.78
        for index, note in enumerate(notes[:4], start=1):
            ax_notes.text(
                0.02,
                note_y,
                f"{index}. {note}",
                ha="left",
                va="top",
                fontsize=7,
                color=LABEL_COLOR,
                transform=ax_notes.transAxes,
                wrap=True,
            )
            note_y -= 0.18

    def _draw_cad_title_block(
        self,
        ax,
        style_cache: dict,
        lithology_codes: list[str],
        title_block: ConsultingTitleBlock,
    ) -> None:
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        border = Rectangle(
            (0.01, 0.05),
            0.98,
            0.9,
            fill=False,
            edgecolor="#94A3B8",
            linewidth=1.0,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(border)
        for x_pos in (0.34, 0.66):
            ax.plot(
                [x_pos, x_pos],
                [0.05, 0.95],
                color="#94A3B8",
                linewidth=0.8,
                transform=ax.transAxes,
                clip_on=False,
            )
        ax.plot(
            [0.66, 0.66],
            [0.55, 0.95],
            color="#94A3B8",
            linewidth=0.8,
            transform=ax.transAxes,
            clip_on=False,
        )

        y = 0.82
        ax.text(0.04, y, "LEGEND", fontsize=9, fontweight="bold", color=LABEL_COLOR, transform=ax.transAxes)
        y -= 0.12
        swatch_w = 0.035
        for code in lithology_codes[:8]:
            style = self._resolve_style(code, style_cache)
            rect = Rectangle(
                (0.04, y - 0.03),
                swatch_w,
                0.05,
                facecolor=style.color,
                edgecolor=style.edge_color,
                linewidth=0.6,
                transform=ax.transAxes,
            )
            ax.add_patch(rect)
            ax.text(
                0.085,
                y,
                code.upper(),
                fontsize=8,
                va="center",
                color=LABEL_COLOR,
                transform=ax.transAxes,
            )
            y -= 0.11

        sym_y = y - 0.02
        screen_rect = Rectangle(
            (0.04, sym_y - 0.03),
            swatch_w,
            0.05,
            facecolor="none",
            edgecolor=TRACK_BORDER_COLOR,
            linewidth=0.6,
            hatch="///",
            transform=ax.transAxes,
        )
        ax.add_patch(screen_rect)
        ax.text(
            0.085,
            sym_y,
            title_block.screen_legend_label or "SCREENED INTERVAL",
            fontsize=7,
            va="center",
            color=LABEL_COLOR,
            transform=ax.transAxes,
        )
        sym_y -= 0.1
        if title_block.show_gradient_legend and self.vertical_gradients:
            arrow = FancyArrow(
                0.055,
                sym_y - 0.01,
                0.0,
                0.035,
                width=0.008,
                head_width=0.02,
                head_length=0.012,
                length_includes_head=True,
                transform=ax.transAxes,
                facecolor=CONSULTING_WATER_COLOR,
                edgecolor=CONSULTING_WATER_COLOR,
                clip_on=False,
            )
            ax.add_patch(arrow)
            ax.text(
                0.085,
                sym_y,
                "VERTICAL GRADIENT DIRECTION",
                fontsize=7,
                va="center",
                color=LABEL_COLOR,
                transform=ax.transAxes,
            )
            sym_y -= 0.1
        gw_legend = self.water_series_legend or [
            {
                "color": CONSULTING_WATER_COLOR,
                "marker": "v",
                "elevation_label": "GROUNDWATER ELEVATION masl",
                "level_label": "GROUNDWATER LEVEL (masl)",
            }
        ]
        for entry in gw_legend:
            color = entry.get("color", CONSULTING_WATER_COLOR)
            marker = entry.get("marker", "v")
            ax.plot(
                [0.04, 0.075],
                [sym_y, sym_y - 0.03],
                marker=marker,
                color=color,
                linewidth=0,
                markersize=6,
                transform=ax.transAxes,
                clip_on=False,
            )
            ax.text(
                0.085,
                sym_y - 0.015,
                entry.get("elevation_label", "GROUNDWATER ELEVATION masl"),
                fontsize=7,
                va="center",
                color=LABEL_COLOR,
                transform=ax.transAxes,
            )
            sym_y -= 0.07
            ax.plot(
                [0.04, 0.075],
                [sym_y, sym_y],
                color=color,
                linewidth=1.5,
                linestyle="--",
                transform=ax.transAxes,
                clip_on=False,
            )
            ax.text(
                0.085,
                sym_y,
                entry.get("level_label", "GROUNDWATER LEVEL (masl)"),
                fontsize=7,
                va="center",
                color=LABEL_COLOR,
                transform=ax.transAxes,
            )
            sym_y -= 0.1

        meta_rows: list[tuple[str, str]] = []
        if title_block.project_number:
            meta_rows.append(("PROJECT", title_block.project_number))
        if title_block.section_label:
            meta_rows.append(("TITLE", consulting_section_title(title_block.section_label)))
        if title_block.source:
            meta_rows.append(("SOURCE", title_block.source))
        if title_block.map_scale:
            meta_rows.append(("SCALE", title_block.map_scale))
        if title_block.date:
            meta_rows.append(("DATE", title_block.date))
        if title_block.drawn_by:
            meta_rows.append(("DRAWN BY", title_block.drawn_by))
        if title_block.revised:
            meta_rows.append(("REVISED", title_block.revised))
        if title_block.figure_number:
            meta_rows.append(("FIGURE NO.", title_block.figure_number))
        self._draw_title_block_metadata_table(ax, meta_rows)

        right_x = 0.68
        if title_block.prepared_for:
            ax.text(
                right_x,
                0.88,
                "PREPARED FOR",
                fontsize=8,
                fontweight="bold",
                color=LABEL_COLOR,
                transform=ax.transAxes,
            )
            ax.text(
                right_x,
                0.78,
                title_block.prepared_for,
                fontsize=8,
                color=LABEL_COLOR,
                transform=ax.transAxes,
            )
            self._draw_logo_image(ax, title_block.logo_prepared_for_bytes, (0.78, 0.62))
        if title_block.prepared_by:
            ax.text(
                right_x,
                0.48,
                "PREPARED BY",
                fontsize=8,
                fontweight="bold",
                color=LABEL_COLOR,
                transform=ax.transAxes,
            )
            ax.text(
                right_x,
                0.38,
                title_block.prepared_by,
                fontsize=8,
                color=LABEL_COLOR,
                transform=ax.transAxes,
            )
            self._draw_logo_image(ax, title_block.logo_prepared_by_bytes, (0.78, 0.22))

    def _draw_logo_image(self, ax, logo_bytes: bytes | None, position: tuple[float, float]) -> None:
        if not logo_bytes:
            return
        try:
            image = mpimg.imread(io.BytesIO(logo_bytes), format="png")
        except Exception:
            logger.warning("Could not decode consulting logo image")
            return
        imagebox = OffsetImage(image, zoom=0.18)
        ab = AnnotationBbox(
            imagebox,
            position,
            xycoords=ax.transAxes,
            frameon=False,
            box_alignment=(0.0, 0.5),
        )
        ax.add_artist(ab)

    def _draw_title_block_metadata_table(
        self,
        ax,
        rows: list[tuple[str, str]],
    ) -> None:
        if not rows:
            return
        table_left = 0.35
        table_bottom = 0.08
        table_width = 0.30
        row_height = 0.84 / max(len(rows), 1)
        for index, (label, value) in enumerate(rows):
            row_bottom = table_bottom + (len(rows) - index - 1) * row_height
            ax.plot(
                [table_left, table_left + table_width],
                [row_bottom, row_bottom],
                color="#94A3B8",
                linewidth=0.6,
                transform=ax.transAxes,
                clip_on=False,
            )
            ax.plot(
                [table_left + 0.11, table_left + 0.11],
                [row_bottom, row_bottom + row_height],
                color="#94A3B8",
                linewidth=0.6,
                transform=ax.transAxes,
                clip_on=False,
            )
            ax.text(
                table_left + 0.01,
                row_bottom + row_height * 0.55,
                label,
                fontsize=7,
                fontweight="bold",
                va="center",
                color=LABEL_COLOR,
                transform=ax.transAxes,
            )
            ax.text(
                table_left + 0.12,
                row_bottom + row_height * 0.55,
                value,
                fontsize=7,
                va="center",
                color=LABEL_COLOR,
                transform=ax.transAxes,
                wrap=True,
            )
        top_y = table_bottom + len(rows) * row_height
        ax.plot(
            [table_left, table_left + table_width],
            [top_y, top_y],
            color="#94A3B8",
            linewidth=0.6,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.plot(
            [table_left, table_left],
            [table_bottom, top_y],
            color="#94A3B8",
            linewidth=0.6,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.plot(
            [table_left + table_width, table_left + table_width],
            [table_bottom, top_y],
            color="#94A3B8",
            linewidth=0.6,
            transform=ax.transAxes,
            clip_on=False,
        )

