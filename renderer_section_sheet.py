"""Section-sheet layout drawing (mixin for CrossSectionRenderer)."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from lithology_codes import collect_lithology_codes
from models import WaterLevel
from render_theme import AXES_BG, FIGURE_BG, GRID_COLOR, LABEL_COLOR
from renderer_common import apply_true_value_y_axis, legend_source_keys, settle_figure_layout
from stratigraphy import GeologicalPolygon

# Clearance between the threshold key and the footer lines below it.
_KEY_FOOTER_PAD_PT = 3.0
# Gap between the hole-name row and the distance label hung under it.
_SUPX_GAP_PT = 6.0


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
        self._ve_main_ax = ax

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

        self._draw_fence_polygons(
            ax,
            polygons,
            style_cache,
            ve,
            alpha=self.profile.fence_alpha,
            collar_lookup=collar_lookup,
            hole_x_lookup=ctx.x_by_hole,
        )

        collar_depths = collar_depths or {}
        if self.profile.show_ground_surface:
            # The sky band was decoration (no legend meaning): ground line only.
            self._draw_sky_and_surface(ax, hole_summary, collar_lookup, draw_sky=False)

        interval_collars = self._collar_values(
            projected_df["hole_id"],
            projected_df["collar_elevation"],
            collar_lookup,
        )
        if self.show_stick_logs and self.profile.show_track_lithology:
            self._draw_track_lithology(
                ax,
                projected_df,
                style_cache,
                track_half,
                collar_lookup,
                collar_arr=interval_collars,
            )
            self._draw_unlogged_intervals(
                ax, projected_df, collar_depths, collar_lookup, track_half
            )
        if self.profile.show_track_border:
            self._draw_track_borders(ax, hole_summary, collar_depths, collar_lookup, track_half)
        profile_lookup = ctx.profile_lookup
        if self.screen_intervals:
            self._draw_screen_intervals(
                ax,
                hole_summary,
                self.screen_intervals,
                collar_lookup,
                track_half,
                profile_lookup=profile_lookup,
            )
        if self.profile.show_contact_ticks:
            self._draw_contact_ticks(
                ax,
                projected_df,
                track_half,
                collar_lookup,
                collar_arr=interval_collars,
            )
        self._draw_deviated_centerlines(ax, projected_df, collar_lookup)
        self._draw_raster_strips(ax, ctx.x_by_hole, collar_lookup, track_half)

        if self.profile.show_overlap_markers and self.overlap_pairs:
            self._draw_overlap_markers(ax, collar_lookup, hole_summary=hole_summary)
        if water_levels:
            self._draw_water_table(
                ax,
                hole_summary,
                water_levels,
                collar_lookup,
                label_elevations=self.profile.show_water_elevation_labels,
                label_dry_wells=self.profile.show_dry_well_nm,
                label_series_gaps=self.profile.show_dry_well_nm,
                profile_lookup=profile_lookup,
                track_half=track_half,
            )
            if self.profile.show_water_legend and self.water_series_legend:
                self._draw_compact_water_legend(ax)
            self._match_water_key_to_drawn(ax, hole_summary, water_levels)
        self._draw_faults(ax, collar_lookup, hole_summary=hole_summary)
        self._draw_unconformities(ax, collar_lookup, hole_summary=hole_summary)
        if not hole_summary.empty:
            # A long list of values on the last hole zig-zags right of it;
            # widen the frame so none are dropped at the right edge.
            x_max = float(hole_summary["x_profile"].max())
            span = x_max - float(hole_summary["x_profile"].min())
            label_room = self._last_hole_label_room(ax, hole_summary, span)
            if label_room:
                left, right = ax.get_xlim()
                ax.set_xlim(left, max(right, x_max + track_half + label_room))
        self._draw_parameter_readings(
            ax,
            hole_summary,
            collar_lookup,
            profile_lookup=profile_lookup,
            column_half_m=track_half,
        )
        if self.parameter_series_legend and self.profile.show_parameter_legend_text:
            self._draw_compact_parameter_legend(ax)
        if self.profile.show_eol_bar:
            self._draw_eol_bars(ax, hole_summary, collar_depths, collar_lookup, track_half)

        if self.profile.show_scale_bar:
            self._draw_scale_bar(ax)
        if self.profile.show_ve_annotation:
            self._draw_ve_annotation(ax)

        if self.show_legend and lithology_codes:
            self._draw_legend(
                ax, style_cache, lithology_codes, polygons, swatch_alpha=self.profile.fence_alpha
            )
            ground = self.profile.show_ground_surface and len(hole_summary) >= 2
            extras = self._legend_extra_keys(ax, ground_surface=ground)
            # The groundwater series key joins the lithology legend outside the
            # frame: in the plot (lower left) it sat over the geology.
            water_key = next((a for a in ax.artists if a.get_gid() == "water_series_key"), None)
            if water_key is not None:
                extras = legend_source_keys(water_key) + extras
                water_key.remove()
            self._rebuild_outside_legend(
                ax, extras, title="Legend" if water_key is not None else None
            )

        y_label = (
            "Depth below collar (m)"
            if self.profile.y_axis_mode == "depth_below_collar"
            else "Elevation (masl)"
        )
        ax.set_ylabel(y_label, fontsize=10, labelpad=8)
        # Data are plotted at value*VE; label the axis with the true values.
        apply_true_value_y_axis(ax, ve)
        # Depth mode draws the hole headers above the axes; lift the title clear.
        title_pad = 14 if self.profile.y_axis_mode == "elevation_rl" else 48
        ax.set_title(self.title, fontsize=14, fontweight="bold", pad=title_pad, color=LABEL_COLOR)
        # Exact VE: the plot box is locked to the chosen VE (centred in its
        # frame); auto: it fills the frame and the captions print the measured VE.
        self._apply_ve_aspect(ax)
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
            self.fit_right_margin_for_legend(fig)
        else:
            fig.subplots_adjust(bottom=bottom_margin)

        if self.profile.show_column_headers:
            self._draw_column_headers(ax, hole_summary, collar_depths, collar_lookup)
            fig.supxlabel(
                "Distance along transect (m)",
                fontsize=10,
                y=0.058,
                va="top",
                color=LABEL_COLOR,
            )
        else:
            ax.set_xlabel("Distance along transect (m)", fontsize=10, labelpad=8)

        if self.profile.show_column_headers and self.profile.y_axis_mode != "elevation_rl":
            # Depth mode: headers above the plot; keep the title over them
            # (measured at draw time, after any tier / vertical header pass).
            from renderer_chart import _ChartTitleLayout

            fig.set_layout_engine(
                _ChartTitleLayout(
                    ax,
                    self.title,
                    self._header_labels,
                    title_kwargs={"fontsize": 14, "fontweight": "bold", "color": LABEL_COLOR},
                )
            )
        self._draw_footers(fig)
        self._fit_section_sheet_bottom(fig)
        self._lift_threshold_key_over_footers(fig)
        return fig

    def _match_water_key_to_drawn(self, ax, hole_summary: pd.DataFrame, water_levels) -> None:
        """The in-plot series key shows a joining line only for series drawn with one."""
        key = next((a for a in ax.artists if a.get_gid() == "water_series_key"), None)
        if key is None:
            return
        on_section = set(hole_summary["hole_id"].astype(str))
        measured = self._water_series_measured(water_levels, on_section)
        sources = [handle for handle, _label in legend_source_keys(key)]
        for index, entry in enumerate(self.water_series_legend):
            if self._water_line_drawn(measured.get(entry["series_id"], 0)):
                continue
            # Drawn key artist now; the source proxy for a later rebuild.
            for handles in (key.legend_handles, sources):
                if index < len(handles) and handles[index] is not None:
                    handles[index].set_linestyle("none")

    def _fit_section_sheet_bottom(self, fig: Figure) -> None:
        """Hang the distance label just under the hole-name row and close the gap below.

        The label sat at a fixed page height, leaving ~1.5 in of white space
        under the hole names on tabloid. It now sits a few points under the
        lowest of the hole names and x tick labels, the threshold key follows
        it, and the bottom margin is set so that stack ends just above the
        footer lines (or the printable margin). Re-run after page resizes.
        """
        supx = getattr(fig, "_supxlabel", None)
        axes = getattr(self, "_ve_main_ax", None)
        if axes is None or axes.figure is not fig:
            return
        renderer = fig.canvas.get_renderer()
        dpi = fig.dpi
        height = fig.bbox.height
        margins = getattr(fig, "_css_page_margins_in", None)
        floor_px = (margins[3] if margins else 0.06) * dpi
        supx_id = id(supx)
        for text in fig.texts:
            if id(text) == supx_id or not text.get_visible() or not text.get_text().strip():
                continue
            if text.get_position()[1] < 0.2:
                floor_px = max(floor_px, text.get_window_extent(renderer).y1 + _KEY_FOOTER_PAD_PT * dpi / 72.0)
        for _attempt in range(3):
            settle_figure_layout(fig)
            frame = axes.get_window_extent(renderer)
            lowest = frame.y0
            tight = axes.xaxis.get_tightbbox(renderer)
            if tight is not None and tight.height > 0:
                lowest = min(lowest, tight.y0)
            for header in getattr(self, "_header_labels", None) or []:
                if header.figure is fig and header.get_visible() and header.get_text().strip():
                    box = header.get_window_extent(renderer)
                    if box.y1 <= frame.y0 + 1.0:
                        lowest = min(lowest, box.y0)
            if supx is not None and supx.get_text():
                supx.set_y((lowest - _SUPX_GAP_PT * dpi / 72.0) / height)
                # Centred on the drawn plot box (an exact-VE box is narrower
                # than the page; centred on the page the label ran past it).
                supx.set_x(0.5 * (frame.x0 + frame.x1) / fig.bbox.width)
            stack_bottom = lowest
            if supx is not None and supx.get_text():
                stack_bottom = min(stack_bottom, supx.get_window_extent(renderer).y0)
            key = getattr(self, "_chemistry_threshold_key", None)
            if key is not None and key.figure is fig:
                stack_bottom = min(stack_bottom, key.get_window_extent(renderer).y0)
            delta = (stack_bottom - floor_px) / height
            if abs(delta) * height < 1.0:
                break
            bottom = min(max(fig.subplotpars.bottom - delta, 0.04), fig.subplotpars.top - 0.2)
            fig.subplots_adjust(bottom=bottom)

    def _lift_threshold_key_over_footers(self, fig: Figure) -> None:
        """Raise the distance label (and the threshold key under it) clear of the footers.

        The threshold key hangs below the distance label; the disclaimer /
        metadata footer lines sit at the page foot. Measured at the render
        size: export pages are as tall or taller, so the clearance only grows.
        """
        key = getattr(self, "_chemistry_threshold_key", None)
        if key is None or key.figure is not fig:
            return
        supx = getattr(fig, "_supxlabel", None)
        footers = [
            text for text in fig.texts
            if text is not supx and text.get_visible() and text.get_text().strip()
        ]
        if not footers:
            return
        renderer = fig.canvas.get_renderer()
        top = max(text.get_window_extent(renderer).y1 for text in footers)
        need_px = top + _KEY_FOOTER_PAD_PT * fig.dpi / 72.0 - key.get_window_extent(renderer).y0
        if need_px <= 0.0:
            return
        need = need_px / fig.bbox.height
        if supx is not None and supx.get_text():
            supx.set_y(supx.get_position()[1] + need)
        else:
            fig.subplots_adjust(bottom=fig.subplotpars.bottom + need)
