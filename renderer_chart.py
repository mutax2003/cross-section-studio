"""Chart layout drawing (mixin for CrossSectionRenderer)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.layout_engine import LayoutEngine
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox, blended_transform_factory, offset_copy

from hydro_metrics import water_status
from lithology_codes import collect_lithology_codes
from models import WaterLevel
from render_profiles import CHART_PROFILE
from render_theme import AXES_BG, FIGURE_BG, GRID_COLOR, LABEL_COLOR, STICK_COLOR, SURFACE_COLOR
from renderer_common import apply_true_value_y_axis
from renderer_water import resolve_header_collisions
from stratigraphy import GeologicalPolygon


@dataclass
class _LabelSpec:
    x: float
    y: float
    text: str
    dx: float = 0.0
    dy: float = 0.0
    draw_leader: bool = False


# Reading / chemistry settings the caller's profile keeps on the chart layout.
# Everything else (cosmetics, axes, columns) stays pinned to CHART_PROFILE.
_PARAMETER_PROFILE_FIELDS: tuple[str, ...] = (
    "show_parameter_markers",
    "show_parameter_labels",
    "show_parameter_legend_text",
    "parameter_interpolate_segments",
    "parameter_interpolate_across_gaps",
    "parameter_draw_markers",
    "parameter_marker",
    "parameter_marker_size",
    "parameter_draw_leaders",
    "parameter_label_include_units",
    "chemistry_color_mode",
    "chemistry_threshold_green_max",
    "chemistry_threshold_yellow_max",
    "chemistry_label_style",
    # Elevation-mode radio in the sidebar (RL vs depth below collar).
    "y_axis_mode",
)


_TITLE_KWARGS: dict[str, object] = {
    "fontsize": 13,
    "fontweight": "bold",
    "color": LABEL_COLOR,
}
COLLAR_LEGEND_LABEL = "Ground surface / collar"
# Gap (points) between the title and the hole headers, and above the title.
_TITLE_HEADER_GAP_PT = 8.0
_TITLE_TOP_MARGIN_PT = 8.0
# Header boxes sit this far (points) above the plot frame; their rounded
# frame (pad 0.25 em at 8 pt) adds about this much around the text.
_HEADER_OFFSET_PT = 4.0
_HEADER_BOX_PAD_PT = 2.0


def _header_box(text, renderer):
    """Header extent including its framed box, not just the glyphs."""
    box = text.get_window_extent(renderer)
    patch = text.get_bbox_patch()
    if patch is not None:
        text.update_bbox_position_size(renderer)
        box = Bbox.union([box, patch.get_window_extent(renderer)])
    return box


class _ChartTitleLayout(LayoutEngine):
    """Draw-time hook: keep the chart title above the hole headers and on the page.

    Export resizes the page to a letter / tabloid preset and re-runs header
    collision resolution, which can raise a header a tier (or turn the headers
    vertical). A fixed title pad and top margin, set at the render size, then
    put the title off the top edge or under a raised header. Measuring at
    draw time (after any header pass) keeps both on every page.
    """

    _adjust_compatible = True
    _colorbar_gridspec = True

    def __init__(self, ax, title: str, headers: Sequence) -> None:
        super().__init__()
        self._ax = ax
        self._title = title
        self._headers = headers

    def set(self, **kwargs) -> None:  # pragma: no cover - no tunable params
        self._params.update(kwargs)

    def execute(self, fig) -> None:
        ax = self._ax
        if ax.figure is not fig or not self._title:
            return
        renderer = fig._get_renderer()
        px_per_pt = renderer.points_to_pixels(1.0)
        ax_top = ax.get_window_extent(renderer).y1
        headers = [
            text
            for text in self._headers
            if text.figure is fig and text.get_visible() and text.get_text().strip()
        ]
        # Reserve two header tiers (a neighbour may raise one; the header pass
        # steps by text height + 1 pt) and more when a header reaches higher.
        above_pt = 0.0
        if headers:
            tier_pt = max(t.get_window_extent(renderer).height for t in headers) / px_per_pt
            above_pt = _HEADER_OFFSET_PT + 2.0 * (tier_pt + 1.0) + _HEADER_BOX_PAD_PT
            reach_pt = (max(_header_box(t, renderer).y1 for t in headers) - ax_top) / px_per_pt
            above_pt = max(above_pt, reach_pt)
        pad_pt = max(6.0, above_pt + _TITLE_HEADER_GAP_PT)
        ax.set_title(self._title, pad=pad_pt, **_TITLE_KWARGS)
        title_h = ax.title.get_window_extent(renderer).height
        need_px = pad_pt * px_per_pt + title_h + _TITLE_TOP_MARGIN_PT * px_per_pt
        fig_h = fig.bbox.height
        if fig_h <= 0:
            return
        bottom = fig.subplotpars.bottom
        top = min(1.0 - need_px / fig_h, 0.98)
        top = max(top, bottom + 0.25)
        if abs(top - fig.subplotpars.top) > 1e-4:
            fig.subplots_adjust(top=top)


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
        original = self.profile
        # The stock chart profile has readings off; keep the caller's
        # parameter / chemistry settings so environmental readings draw.
        parameter_updates = {
            name: getattr(original, name)
            for name in _PARAMETER_PROFILE_FIELDS
            if getattr(original, name) != getattr(CHART_PROFILE, name)
        }
        chart_profile = (
            CHART_PROFILE.model_copy(update=parameter_updates) if parameter_updates else CHART_PROFILE
        )
        self.profile = chart_profile
        self._chart_collar_markers_drawn = False
        try:
            fig_width = 13.5 if self.show_legend else 12.0
            fig, ax = plt.subplots(figsize=(fig_width, 6.8))
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

            if self.profile.show_ground_surface and len(hole_summary) >= 2:
                self._draw_sky_and_surface(ax, hole_summary, collar_lookup)

            self._draw_fence_polygons(
                ax,
                polygons,
                style_cache,
                ve,
                alpha=0.92,
                collar_lookup=collar_lookup,
                hole_x_lookup=ctx.x_by_hole,
            )

            if self.profile.show_overlap_markers and self.overlap_pairs:
                self._draw_overlap_markers(ax, collar_lookup, hole_summary=hole_summary)
            profile_lookup = ctx.profile_lookup
            if water_levels:
                self._draw_water_table(
                    ax,
                    hole_summary,
                    water_levels,
                    collar_lookup,
                    profile_lookup=profile_lookup,
                )
            self._draw_faults(ax, collar_lookup, hole_summary=hole_summary)
            self._draw_unconformities(ax, collar_lookup, hole_summary=hole_summary)
            self._draw_parameter_readings(
                ax,
                hole_summary,
                collar_lookup,
                profile_lookup=profile_lookup,
                column_half_m=track_half,
            )
            if self.parameter_series_legend and self.profile.show_parameter_legend_text:
                self._draw_compact_parameter_legend(ax)

            collar_depths = collar_depths or {}
            labels = self._resolve_label_collisions(
                self._build_borehole_labels(hole_summary, collar_depths)
            )
            if self.show_stick_logs:
                self._draw_track_lithology(ax, projected_df, style_cache, track_half, collar_lookup)
                self._draw_unlogged_intervals(
                    ax, projected_df, collar_depths, collar_lookup, track_half
                )

            if self.profile.show_centerline:
                self._draw_chart_centerlines(ax, hole_summary, collar_depths, collar_lookup)

            # Hole headers sit just above the plot frame, over their columns
            # (as on the section sheet). Anchored at the collars inside the
            # plot, the 3-line boxes rose into the title and touched each other
            # on short sections; the shared header pass staggers them instead.
            header_transform = offset_copy(
                blended_transform_factory(ax.transData, ax.transAxes),
                fig=fig,
                y=4.0,
                units="points",
            )
            self._header_labels = []
            for label in labels:
                header = ax.text(
                    label.x,
                    1.0,
                    label.text,
                    transform=header_transform,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color=LABEL_COLOR,
                    linespacing=1.1,
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": "white",
                        "edgecolor": "#CBD5E1",
                        "alpha": 0.92,
                    },
                    zorder=8,
                    clip_on=False,
                )
                self._header_labels.append(header)

            self._draw_scale_bar(ax)
            # The chart has no title block: the VE note is its only VE caption.
            self._draw_ve_annotation(ax)
            if self.show_legend and lithology_codes:
                self._draw_legend(ax, style_cache, lithology_codes, polygons)
                self._extend_chart_legend(ax, hole_summary, water_levels, profile_lookup)

            ax.set_xlabel("Distance along transect (m)", fontsize=10, labelpad=8)
            depth_mode = self.profile.y_axis_mode == "depth_below_collar"
            ax.set_ylabel(
                "Depth below collar (m)" if depth_mode else "Elevation (m)",
                fontsize=10,
                labelpad=8,
            )
            apply_true_value_y_axis(ax, ve)
            if depth_mode:
                ax.invert_yaxis()
            # Room for up to two tiers of 3-line headers between plot and title;
            # _ChartTitleLayout re-measures at draw time (export page sizes).
            ax.set_title(self.title, pad=72, **_TITLE_KWARGS)
            self._apply_ve_aspect(ax)
            ax.grid(True, linestyle="--", alpha=0.35, color=GRID_COLOR, zorder=0)
            for spine in ax.spines.values():
                spine.set_color("#CBD5E1")

            self._draw_footers(fig)
            bottom_margin = 0.16
            if self.show_legend and lithology_codes:
                fig.subplots_adjust(right=0.78, bottom=bottom_margin)
                self.fit_right_margin_for_legend(fig)
            else:
                fig.tight_layout(rect=(0, bottom_margin - 0.02, 1, 1))
            fig.set_layout_engine(_ChartTitleLayout(ax, self.title, self._header_labels))
            resolve_header_collisions(fig, self._header_labels)
            return fig
        finally:
            self.profile = original

    def _draw_chart_centerlines(
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
        collection = LineCollection(segments, colors=STICK_COLOR, linewidths=4.0, zorder=5)
        ax.add_collection(collection)
        ax.scatter(x_values, top_y, marker="v", s=49, c=SURFACE_COLOR, zorder=7)
        self._chart_collar_markers_drawn = len(x_values) > 0

    def _chart_symbol_handles(
        self,
        hole_summary: pd.DataFrame,
        water_levels: Sequence[WaterLevel] | None,
        profile_lookup: dict[str, tuple[float, float]] | None,
    ) -> list[tuple[Line2D, str]]:
        """Legend proxies for the drawn groundwater series and collar markers."""
        handles: list[tuple[Line2D, str]] = []
        entries = list(getattr(self, "water_series_legend", None) or []) if water_levels else []
        if entries:
            on_section = set(profile_lookup or {}) or set(hole_summary["hole_id"].astype(str))
            measured: dict[str, int] = {}
            for level in water_levels or ():
                if str(level.hole_id) not in on_section:
                    continue
                if water_status(level) in {"dry", "nm"}:
                    continue
                series_id = level.series_id or "default"
                measured[series_id] = measured.get(series_id, 0) + 1
            interpolate = bool(
                self.interpolate_water_table or self.profile.interpolate_water_table_default
            )
            for entry in entries:
                count = measured.get(entry["series_id"], 0)
                if count == 0:
                    continue  # all dry / NM: no marker on the chart
                linestyle = entry.get("linestyle", "--") if interpolate and count >= 2 else "none"
                handles.append(
                    (
                        Line2D(
                            [],
                            [],
                            color=entry["color"],
                            marker=entry["marker"],
                            markersize=7,
                            linewidth=2.0,
                            linestyle=linestyle,
                        ),
                        entry["level_label"],
                    )
                )
        if getattr(self, "_chart_collar_markers_drawn", False):
            handles.append(
                (
                    Line2D([], [], color=SURFACE_COLOR, marker="v", markersize=7, linestyle="none"),
                    COLLAR_LEGEND_LABEL,
                )
            )
        return handles

    def _extend_chart_legend(
        self,
        ax,
        hole_summary: pd.DataFrame,
        water_levels: Sequence[WaterLevel] | None,
        profile_lookup: dict[str, tuple[float, float]] | None,
    ) -> None:
        """Append groundwater / collar symbols to the outside lithology legend."""
        extras = self._chart_symbol_handles(hole_summary, water_levels, profile_lookup)
        legend = ax.get_legend()
        if not extras or legend is None:
            return
        handles = list(legend.legend_handles)
        labels = [text.get_text() for text in legend.get_texts()]
        handles.extend(handle for handle, _label in extras)
        labels.extend(label for _handle, label in extras)
        gid = legend.get_gid()
        rebuilt = ax.legend(
            handles=handles,
            labels=labels,
            title="Legend",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            framealpha=0.95,
            edgecolor="#CBD5E1",
            fontsize=8,
            title_fontsize=9,
            ncol=legend._ncols,
        )
        if gid:
            rebuilt.set_gid(gid)

    def _build_borehole_labels(
        self,
        hole_summary: pd.DataFrame,
        collar_depths: dict[str, float],
    ) -> list[_LabelSpec]:
        labels: list[_LabelSpec] = []
        for row in hole_summary.itertuples(index=False):
            depth = collar_depths.get(row.hole_id)
            depth_text = f"{depth:.1f} m TD" if depth is not None else ""
            # The collar RL rides in the header box: as a separate label it
            # sat under the box and only "m RL" showed.
            rl_text = f"{float(row.collar_elevation):.1f} m RL"
            lines = [str(row.hole_id), depth_text, rl_text]
            labels.append(
                _LabelSpec(
                    x=float(row.x_profile),
                    y=float(row.collar_elevation),
                    text="\n".join(line for line in lines if line),
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
