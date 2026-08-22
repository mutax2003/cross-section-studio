"""Consulting-section layout drawing (mixin for CrossSectionRenderer)."""

from __future__ import annotations

import io
import logging
import textwrap
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
from matplotlib.ticker import FuncFormatter, MultipleLocator

from constants import get_lithology_style
from lithology_codes import collect_lithology_codes
from models import ConsultingTitleBlock, VerticalGradient, WaterLevel
from stratigraphy import GeologicalPolygon
from render_theme import (
    CONSULTING_COLUMN_FILL,
    CONSULTING_FIGURE_BG,
    CONSULTING_NM_COLOR,
    CONSULTING_SCALE_BAR_M,
    CONSULTING_SURFACE_COLOR,
    CONSULTING_WATER_COLOR,
    DEFAULT_CONSULTING_NOTES,
    LABEL_COLOR,
    PARAMETER_READING_COLOR,
    OVERLAP_MARKER_COLOR,
    PINCH_OUT_ALPHA,
    REPORT_GRID_ALPHA,
    REPORT_GRID_COLOR,
    STICK_COLOR,
    TRACK_BORDER_COLOR,
    TRACK_FILL_COLOR,
    consulting_section_title,
    export_font_rc,
    primary_water_depth_by_hole,
    water_has_multiple_series,
)

logger = logging.getLogger(__name__)

# Minimum legend column width (axes fraction) before wrapping to two columns.
_LEGEND_MIN_COL_WIDTH = 0.14
_LEGEND_CHAR_WIDTH = 0.0065  # approx. axes fraction per character at 7.5 pt


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
        if (
            self.disclaimer
            and self.interpretation_mode in {"interpolated", "correlation_lines"}
            and self.disclaimer not in title_block.notes
        ):
            title_block = title_block.model_copy(
                update={"notes": (*title_block.notes, self.disclaimer)}
            )

        ctx = self._hole_context(projected_df)
        # Lock letter landscape (11×8.5 in) so PNG/PDF match client page extracts.
        fig = plt.figure(figsize=(11.0, 8.5))
        fig.patch.set_facecolor(CONSULTING_FIGURE_BG)
        fig.subplots_adjust(left=0.06, right=0.97, top=0.97, bottom=0.04)
        grid = GridSpec(3, 1, figure=fig, height_ratios=[58, 12, 22], hspace=0.12)
        ax = fig.add_subplot(grid[0, 0])
        sub_gs = grid[1, 0].subgridspec(1, 3, width_ratios=[32, 36, 32], wspace=0.14)
        ax_scale = fig.add_subplot(sub_gs[0, 0])
        ax_center = fig.add_subplot(sub_gs[0, 1])
        ax_notes = fig.add_subplot(sub_gs[0, 2])
        ax_block = fig.add_subplot(grid[2, 0])
        ax.set_facecolor(CONSULTING_FIGURE_BG)

        with mpl.rc_context(
            export_font_rc(
                self.profile.export_font_family,
                max(self.profile.export_font_size, 9.0),
            )
        ):
            if lithology_codes is None:
                lithology_codes = collect_lithology_codes(projected_df, polygons)
            elif not isinstance(lithology_codes, list):
                lithology_codes = list(lithology_codes)
            style_cache = self._style_cache_for(lithology_codes)

            hole_summary = ctx.summary
            collar_lookup = ctx.collar_lookup
            track_half = ctx.track_half
            ve = self.vertical_exaggeration
            collar_depths = collar_depths or {}
            water_levels_list = water_levels or ()
            profile_lookup = ctx.profile_lookup
            show_nm = self.profile.show_dry_well_nm

            if polygons:
                self._draw_fence_polygons(
                    ax, polygons, style_cache, ve, alpha=self.profile.fence_alpha, collar_lookup=collar_lookup
                )
            else:
                self._has_pinch_out = False
            self._draw_consulting_surface(ax, hole_summary, collar_lookup)
            self._draw_well_columns(ax, hole_summary, collar_depths, collar_lookup, track_half)
            if self.profile.show_track_lithology:
                self._draw_lithology_interval_rects(
                    ax,
                    projected_df,
                    style_cache,
                    track_half * 0.92,
                    collar_lookup,
                    zorder=9,
                    alpha=1.0,
                )
            if self.screen_intervals:
                self._draw_screen_intervals(
                    ax,
                    hole_summary,
                    self.screen_intervals,
                    collar_lookup,
                    track_half,
                    profile_lookup=profile_lookup,
                )
            if water_levels_list or show_nm:
                multi_series = water_has_multiple_series(water_levels_list)
                self._draw_water_table(
                    ax,
                    hole_summary,
                    water_levels_list,
                    collar_lookup,
                    label_elevations=self.profile.show_water_elevation_labels,
                    label_dry_wells=show_nm and not multi_series,
                    label_series_gaps=show_nm,
                    water_color=CONSULTING_WATER_COLOR,
                    profile_lookup=profile_lookup,
                )
            if self.vertical_gradients:
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
            if self.profile.show_overlap_markers and self.overlap_pairs:
                self._draw_overlap_markers(ax, collar_lookup, hole_summary=hole_summary)
            if self.faults:
                self._draw_faults(ax, collar_lookup, hole_summary=hole_summary)
            if self.unconformities:
                self._draw_unconformities(ax, collar_lookup, hole_summary=hole_summary)

            y_label = title_block.y_axis_label or self.profile.y_axis_label or (
                "DEPTH (mbgs)"
                if self.profile.y_axis_mode == "depth_below_collar"
                else "ELEVATION (m)"
            )
            ax.set_xlabel("DISTANCE (m)", fontsize=10, labelpad=2, color=LABEL_COLOR)
            ax.set_ylabel(y_label, fontsize=10, labelpad=6, color=LABEL_COLOR)
            ax.set_aspect("auto")
            for spine in ax.spines.values():
                spine.set_color("#374151")
                spine.set_linewidth(1.0)

            if self.profile.y_axis_mode == "depth_below_collar":
                ax.invert_yaxis()

            if (
                self.profile.consulting_axis_from_zero
                and self.profile.y_axis_mode != "depth_below_collar"
            ):
                self._apply_consulting_axis_limits(ax, hole_summary, track_half, water_levels_list)

            # Parameter labels after axis limits so collision spacing uses final ylim.
            self._draw_parameter_readings(
                ax,
                hole_summary,
                collar_lookup,
                profile_lookup=profile_lookup,
            )

            ax_right: plt.Axes | None = None
            if self.profile.show_dual_y_axes:
                ax_right = ax.twinx()
                ax_right.set_ylim(ax.get_ylim())
                ax_right.set_ylabel(y_label, fontsize=10, labelpad=6, color=LABEL_COLOR)
                for spine in ax_right.spines.values():
                    spine.set_color("#374151")
                    spine.set_linewidth(1.0)

            if self.profile.show_report_grid:
                x_grid = 20.0 if ctx.x_span > 200.0 else self.profile.x_major_grid_m
                self._apply_report_grid(ax, ax_right, consulting=True, x_major_step=x_grid)

            self._draw_subtitle_band(ax_scale, ax_center, ax_notes, title_block)
            self._draw_cad_title_block(ax_block, style_cache, lithology_codes, title_block)
            self._draw_consulting_footers(fig)
        return fig

    def _apply_consulting_axis_limits(
        self,
        ax,
        hole_summary: pd.DataFrame,
        track_half: float,
        water_levels: Sequence[WaterLevel],
    ) -> None:
        if hole_summary.empty:
            return
        ve = self.vertical_exaggeration
        x_max = float(hole_summary["x_profile"].max())
        x_pad = max(track_half, 5.0)
        ax.set_xlim(0.0, x_max + x_pad)
        y_min, y_max = self._uncertainty_y_bounds(hole_summary)
        y_pad = max(ve * 0.5, 1.0)
        collar_lookup = {
            str(row.hole_id): float(row.collar_elevation)
            for row in hole_summary.itertuples(index=False)
        }
        if water_levels and self.profile.show_water_elevation_labels:
            for level in water_levels:
                collar_rl = collar_lookup.get(level.hole_id)
                if collar_rl is None:
                    continue
                water_y = self._plot_y(collar_rl - level.depth, collar_rl)
                y_min = min(y_min, water_y)
                y_max = max(y_max, water_y)
        # Include active environmental sample depths so chloride markers are not clipped.
        active = {name.strip() for name in (self.environmental_parameters or ()) if name.strip()}
        if active and self.environmental_readings:
            for reading in self.environmental_readings:
                if reading.parameter not in active:
                    continue
                collar_rl = collar_lookup.get(reading.hole_id)
                if collar_rl is None:
                    continue
                for depth in (
                    reading.sample_depth,
                    reading.from_depth,
                    reading.to_depth,
                ):
                    if depth is None:
                        continue
                    sample_y = self._plot_y(collar_rl - float(depth), collar_rl)
                    y_min = min(y_min, sample_y)
                    y_max = max(y_max, sample_y)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    def _draw_consulting_footers(self, fig: Figure) -> None:
        footer_y = 0.01
        if self.overlap_pairs and self.profile.show_overlap_footer:
            fig.text(
                0.5,
                footer_y,
                (
                    f"Polygon overlap markers ({len(self.overlap_pairs)}): "
                    "review layer correlation between adjacent holes."
                ),
                ha="center",
                va="bottom",
                fontsize=7,
                color=OVERLAP_MARKER_COLOR,
            )
            footer_y += 0.02
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
        ax.xaxis.set_major_locator(MultipleLocator(x_major_step if x_major_step is not None else self.profile.x_major_grid_m))
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
            facecolors=CONSULTING_COLUMN_FILL,
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
        ax.plot(
            surface_x,
            surface_y,
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
        water_depth_by_hole = primary_water_depth_by_hole(water_levels)
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
        header_transform = ax.transAxes
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
        if self.profile.show_scale_bar:
            bar_x = 0.02
            bar_y = 0.55
            bar_w = 0.72
            tick_step = 10.0
            tick_marks = tuple(np.arange(0.0, scale_bar_m + tick_step * 0.5, tick_step))
            for tick_m in tick_marks:
                tick_x = bar_x + (tick_m / scale_bar_m) * bar_w
                ax_scale.plot(
                    [tick_x, tick_x],
                    [bar_y - 0.05, bar_y + 0.05],
                    color=STICK_COLOR,
                    linewidth=1.0,
                    transform=ax_scale.transAxes,
                    clip_on=False,
                )
                ax_scale.text(
                    tick_x,
                    bar_y - 0.10,
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
                bar_x + bar_w + 0.03,
                bar_y,
                "Metres",
                ha="left",
                va="center",
                fontsize=7,
                color=LABEL_COLOR,
                transform=ax_scale.transAxes,
            )
            ax_scale.text(
                bar_x + bar_w / 2.0,
                0.12,
                f"SCALE {map_scale}",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color=LABEL_COLOR,
                transform=ax_scale.transAxes,
            )

        section_title = consulting_section_title(title_block.section_label or self.title)
        ax_center.text(
            0.5,
            0.58,
            section_title,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=LABEL_COLOR,
            transform=ax_center.transAxes,
            wrap=True,
        )
        ax_center.plot(
            [0.10, 0.90],
            [0.38, 0.38],
            color=LABEL_COLOR,
            linewidth=0.8,
            transform=ax_center.transAxes,
            clip_on=False,
        )
        # Subtitle VE follows report chrome: on with the scale band, or when
        # show_ve_annotation is explicitly enabled. Both off = GIS paste mode.
        if self.profile.show_scale_bar or self.profile.show_ve_annotation:
            ve_text = (
                "NO VERTICAL EXAGGERATION"
                if abs(float(self.vertical_exaggeration) - 1.0) < 1e-9
                else f"{self.vertical_exaggeration:.0f}× VERTICAL EXAGGERATION"
            )
            ax_center.text(
                0.5,
                0.18,
                ve_text,
                ha="center",
                va="center",
                fontsize=8,
                color=LABEL_COLOR,
                transform=ax_center.transAxes,
            )

        notes = title_block.notes or DEFAULT_CONSULTING_NOTES
        ax_notes.text(
            0.04,
            0.86,
            "NOTES:",
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=LABEL_COLOR,
            transform=ax_notes.transAxes,
        )
        note_y = 0.68
        for index, note in enumerate(notes[:4], start=1):
            ax_notes.text(
                0.04,
                note_y,
                f"{index}. {note}",
                ha="left",
                va="top",
                fontsize=6.5,
                color=LABEL_COLOR,
                transform=ax_notes.transAxes,
                wrap=True,
            )
            note_y -= 0.22

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

        # Three independent panels so legend / title / prepared content stay boxed.
        legend_box = (0.01, 0.05, 0.32, 0.90)
        has_right = bool(title_block.prepared_for or title_block.prepared_by)
        if has_right:
            title_box = (0.34, 0.05, 0.31, 0.90)
            right_box = (0.66, 0.05, 0.33, 0.90)
            panels = (legend_box, title_box, right_box)
        else:
            # Give the title/meta table the remaining width when no prepared logos.
            title_box = (0.34, 0.05, 0.65, 0.90)
            right_box = None
            panels = (legend_box, title_box)
        for box in panels:
            ax.add_patch(
                Rectangle(
                    (box[0], box[1]),
                    box[2],
                    box[3],
                    fill=False,
                    edgecolor="#94A3B8",
                    linewidth=1.0,
                    transform=ax.transAxes,
                    clip_on=False,
                )
            )

        self._draw_legend_panel(
            ax,
            style_cache,
            lithology_codes,
            title_block,
            panel=legend_box,
        )

        meta_rows: list[tuple[str, str]] = []
        if title_block.project_number:
            meta_rows.append(("PROJECT", title_block.project_number))
        section_label = title_block.section_label or self.title
        if section_label:
            meta_rows.append(("TITLE", consulting_section_title(section_label)))
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
        self._draw_title_block_metadata_table(ax, meta_rows, panel=title_box)

        if right_box is None:
            return
        right_x = right_box[0] + 0.03
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

    @staticmethod
    def _legend_label_char_budget(
        col_width_axes: float,
        font_size: float,
        *,
        swatch_w: float,
    ) -> int:
        """Estimate how many characters fit in a legend column without bleeding sideways."""
        usable = max(col_width_axes - swatch_w - 0.018, 0.02)
        char_w = _LEGEND_CHAR_WIDTH * (font_size / 7.5)
        return max(8, int(usable / char_w))

    @staticmethod
    def _legend_panel_clip(ax, panel: tuple[float, float, float, float]) -> Rectangle:
        left, bottom, width, height = panel
        clip_rect = Rectangle(
            (left, bottom),
            width,
            height,
            transform=ax.transAxes,
            visible=False,
        )
        ax.add_patch(clip_rect)
        return clip_rect

    def _draw_legend_panel(
        self,
        ax,
        style_cache: dict,
        lithology_codes: list[str],
        title_block: ConsultingTitleBlock,
        *,
        panel: tuple[float, float, float, float],
    ) -> None:
        left, bottom, width, height = panel
        pad_x = 0.025
        pad_y = 0.04
        content_left = left + pad_x
        content_top = bottom + height - pad_y
        content_bottom = bottom + pad_y
        swatch_w = 0.03

        entries: list[tuple[str, str, dict[str, object]]] = []
        for code in lithology_codes[:12]:
            style = self._resolve_style(code, style_cache)
            entries.append(
                (
                    "swatch",
                    code.upper(),
                    {
                        "facecolor": style.color,
                        "edgecolor": style.edge_color,
                        "hatch": None,
                    },
                )
            )
        if self.screen_intervals:
            entries.append(
                (
                    "swatch",
                    title_block.screen_legend_label or "SCREENED INTERVAL",
                    {
                        "facecolor": TRACK_FILL_COLOR,
                        "edgecolor": TRACK_BORDER_COLOR,
                        "hatch": "///",
                    },
                )
            )
        if title_block.show_gradient_legend and self.vertical_gradients:
            entries.append(("gradient", "VERTICAL GRADIENT DIRECTION", {}))

        gw_legend = self.water_series_legend or []
        compact_gw = self.profile.compact_water_legend
        gw_linestyle = "-" if self.profile.water_line_solid else "--"
        if self.profile.show_water_legend and gw_legend:
            relative = self.profile.y_axis_mode == "depth_below_collar"
            default_elev = (
                "GROUNDWATER DEPTH (mbgs)" if relative else "GROUNDWATER ELEVATION masl"
            )
            default_level = (
                "GROUNDWATER LEVEL (mbgs)" if relative else "GROUNDWATER LEVEL (masl)"
            )
            for entry in gw_legend:
                if compact_gw:
                    legend_label = entry.get("level_label") or entry.get(
                        "elevation_label", default_level
                    )
                    entries.append(
                        (
                            "line_marker",
                            str(legend_label),
                            {
                                "color": entry.get("color", CONSULTING_WATER_COLOR),
                                "marker": entry.get("marker", "v"),
                                "linestyle": gw_linestyle,
                            },
                        )
                    )
                else:
                    entries.append(
                        (
                            "marker",
                            str(entry.get("elevation_label", default_elev)),
                            {
                                "color": entry.get("color", CONSULTING_WATER_COLOR),
                                "marker": entry.get("marker", "v"),
                            },
                        )
                    )
                    entries.append(
                        (
                            "line",
                            str(entry.get("level_label", default_level)),
                            {
                                "color": entry.get("color", CONSULTING_WATER_COLOR),
                                "linestyle": gw_linestyle,
                            },
                        )
                    )

        draw_param_markers = self.profile.parameter_draw_markers
        for entry in self.parameter_series_legend or []:
            kind = "line_marker" if draw_param_markers else "text"
            entries.append(
                (
                    kind,
                    str(entry.get("label", entry.get("parameter", "PARAMETER"))),
                    {
                        "color": entry.get("color", PARAMETER_READING_COLOR),
                        "marker": entry.get("marker", "D"),
                        "linestyle": "--",
                    },
                )
            )
        if getattr(self, "_has_pinch_out", False) and self.profile.show_pinch_out_legend:
            entries.append(
                (
                    "line",
                    "INFERRED PINCH-OUT",
                    {"color": LABEL_COLOR, "linestyle": "--"},
                )
            )

        # Header + entries must stay inside the legend box (never bleed into title block).
        ncol = max(1, self.profile.legend_ncol)
        usable_width = width - 2 * pad_x
        col_width_single = usable_width
        col_width_two = (usable_width - 0.02) / 2
        use_two_cols = (
            ncol >= 2
            and len(entries) > 6
            and col_width_two >= _LEGEND_MIN_COL_WIDTH
        )
        if use_two_cols:
            mid = (len(entries) + 1) // 2
            column_groups: list[list[tuple[str, str, dict[str, object]]]] = [
                entries[:mid],
                entries[mid:],
            ]
            col_gap = 0.02
            col_width = col_width_two
            column_layouts = [
                (content_left, content_left + swatch_w + 0.012),
                (
                    content_left + col_width + col_gap,
                    content_left + col_width + col_gap + swatch_w + 0.012,
                ),
            ]
        else:
            column_groups = [entries]
            col_width = col_width_single
            column_layouts = [(content_left, content_left + swatch_w + 0.012)]

        clip_rect = self._legend_panel_clip(ax, panel)

        n_rows = 1 + max(len(group) for group in column_groups)
        step = min(0.10, max(0.055, (content_top - content_bottom) / max(n_rows, 1)))
        font_size = 7.5 if step >= 0.08 else 6.5
        y_header = content_top
        header = ax.text(
            content_left,
            y_header,
            "LEGEND",
            fontsize=8.5,
            fontweight="bold",
            color=LABEL_COLOR,
            transform=ax.transAxes,
            va="top",
            clip_on=True,
        )
        header.set_clip_path(clip_rect)
        entry_top = y_header - step

        for group, (col_left, col_text_x) in zip(column_groups, column_layouts, strict=True):
            y = entry_top
            max_label_chars = self._legend_label_char_budget(
                col_width, font_size, swatch_w=swatch_w
            )
            for kind, label, style in group:
                if y < content_bottom + 0.02:
                    break
                display = label
                if len(label) > max_label_chars:
                    paren = label.rfind("(")
                    if paren > 0 and label.endswith(")") and len(label) - paren <= 14:
                        prefix_budget = max_label_chars - (len(label) - paren) - 1
                        if prefix_budget >= 8:
                            display = label[:prefix_budget].rstrip(" -:") + "…" + label[paren:]
                        else:
                            display = label[: max_label_chars - 1] + "…"
                    else:
                        display = label[: max_label_chars - 1] + "…"
                if kind == "swatch":
                    rect = Rectangle(
                        (col_left, y - 0.022),
                        swatch_w,
                        0.04,
                        facecolor=style["facecolor"],
                        edgecolor=style["edgecolor"],
                        linewidth=0.6,
                        hatch=style.get("hatch"),
                        transform=ax.transAxes,
                        clip_on=True,
                    )
                    ax.add_patch(rect)
                    rect.set_clip_path(clip_rect)
                elif kind == "gradient":
                    arrow = FancyArrow(
                        col_left + 0.012,
                        y - 0.01,
                        0.0,
                        0.03,
                        width=0.006,
                        head_width=0.016,
                        head_length=0.01,
                        length_includes_head=True,
                        transform=ax.transAxes,
                        facecolor=CONSULTING_WATER_COLOR,
                        edgecolor=CONSULTING_WATER_COLOR,
                        clip_on=True,
                    )
                    ax.add_patch(arrow)
                    arrow.set_clip_path(clip_rect)
                elif kind == "marker":
                    (marker_line,) = ax.plot(
                        [col_left, col_left + 0.03],
                        [y, y - 0.02],
                        marker=style.get("marker", "v"),
                        color=style.get("color", CONSULTING_WATER_COLOR),
                        linewidth=0,
                        markersize=5,
                        transform=ax.transAxes,
                        clip_on=True,
                    )
                    marker_line.set_clip_path(clip_rect)
                elif kind == "line":
                    (line,) = ax.plot(
                        [col_left, col_left + 0.03],
                        [y, y],
                        color=style.get("color", LABEL_COLOR),
                        linewidth=1.4,
                        linestyle=style.get("linestyle", "--"),
                        transform=ax.transAxes,
                        clip_on=True,
                    )
                    line.set_clip_path(clip_rect)
                elif kind == "text":
                    pass
                else:  # line_marker
                    (lm_line,) = ax.plot(
                        [col_left, col_left + 0.03],
                        [y, y],
                        marker=style.get("marker", "D"),
                        color=style.get("color", PARAMETER_READING_COLOR),
                        linewidth=1.4,
                        linestyle=style.get("linestyle", "--"),
                        markersize=5,
                        transform=ax.transAxes,
                        clip_on=True,
                    )
                    lm_line.set_clip_path(clip_rect)
                label_artist = ax.text(
                    col_left if kind == "text" else col_text_x,
                    y,
                    display,
                    fontsize=font_size,
                    va="center",
                    color=style.get("color", LABEL_COLOR) if kind == "text" else LABEL_COLOR,
                    transform=ax.transAxes,
                    clip_on=True,
                )
                label_artist.set_clip_path(clip_rect)
                y -= step

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
        *,
        panel: tuple[float, float, float, float] | None = None,
    ) -> None:
        if not rows:
            return
        if panel is None:
            table_left, table_bottom, table_width, table_height = 0.35, 0.08, 0.30, 0.84
        else:
            table_left, table_bottom, table_width, table_height = panel
            # Inset slightly so cell rules sit inside the panel border.
            inset = 0.01
            table_left += inset
            table_bottom += inset
            table_width -= 2 * inset
            table_height -= 2 * inset

        label_col_w = min(0.09, table_width * 0.28)
        value_col_w = table_width - label_col_w
        row_height = table_height / max(len(rows), 1)
        # Character budget from panel fraction; consulting sheets are typically wide.
        max_chars = max(28, int(value_col_w * 170))

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
                [table_left + label_col_w, table_left + label_col_w],
                [row_bottom, row_bottom + row_height],
                color="#94A3B8",
                linewidth=0.6,
                transform=ax.transAxes,
                clip_on=False,
            )
            ax.text(
                table_left + 0.01,
                row_bottom + row_height * 0.5,
                label,
                fontsize=7,
                fontweight="bold",
                va="center",
                color=LABEL_COLOR,
                transform=ax.transAxes,
                clip_on=True,
            )
            wrapped = textwrap.wrap(str(value), width=max_chars) or [""]
            max_lines = max(2, int(row_height / 0.09))
            if len(wrapped) > max_lines:
                wrapped = wrapped[:max_lines]
                if len(wrapped[-1]) > 3:
                    wrapped[-1] = wrapped[-1][: max(3, len(wrapped[-1]) - 1)] + "…"
            line_gap = min(0.10, (row_height * 0.7) / max(len(wrapped), 1))
            text_top = row_bottom + row_height * 0.5 + (len(wrapped) - 1) * line_gap * 0.5
            for line_index, line in enumerate(wrapped):
                ax.text(
                    table_left + label_col_w + 0.012,
                    text_top - line_index * line_gap,
                    line,
                    fontsize=6.5 if len(wrapped) > 1 else 7,
                    va="center",
                    ha="left",
                    color=LABEL_COLOR,
                    transform=ax.transAxes,
                    clip_on=True,
                )

        top_y = table_bottom + len(rows) * row_height
        for x_pos in (table_left, table_left + table_width):
            ax.plot(
                [x_pos, x_pos],
                [table_bottom, top_y],
                color="#94A3B8",
                linewidth=0.6,
                transform=ax.transAxes,
                clip_on=False,
            )
        ax.plot(
            [table_left, table_left + table_width],
            [top_y, top_y],
            color="#94A3B8",
            linewidth=0.6,
            transform=ax.transAxes,
            clip_on=False,
        )

