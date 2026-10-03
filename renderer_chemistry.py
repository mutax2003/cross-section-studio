"""Environmental / chemistry parameter drawing mixin for CrossSectionRenderer."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from typing import TypedDict

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patheffects import withStroke
from matplotlib.transforms import offset_copy

from models import EnvironmentalReading
from render_theme import (
    CHEMISTRY_FIXED_COLORS,
    CHEMISTRY_LABEL_BLACK,
    LABEL_COLOR,
    chemistry_label_color,
    parameter_series_colors,
)
from renderer_water import _GW_MARKER_MAP


class ParameterLegendEntry(TypedDict):
    parameter: str
    color: str
    marker: str
    label: str


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
# "box" readability style: fully opaque so hatch lines never show through.
_PARAMETER_LABEL_BBOX_SOLID = {**_PARAMETER_LABEL_BBOX, "alpha": 1.0, "boxstyle": "square,pad=0.18"}


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
    invert_y: bool = False,
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
    # Work in a screen-oriented space: larger value = higher on screen. In
    # depth_below_collar mode (inverted axis) data-y grows downward, so flip.
    sign = -1.0 if (invert_y or float(y0) > float(y1)) else 1.0
    data_span = abs(float(y1) - float(y0)) or 1.0
    pos = ax.get_position()
    height_pts = float(ax.figure.get_figheight()) * float(pos.height) * 72.0
    pts_per_data = height_pts / data_span if height_pts > 0 else 1.0

    offsets: list[tuple[float, float, bool]] = [
        (_PARAMETER_LABEL_DX, _PARAMETER_LABEL_BASE_DY, False)
        for _ in marker_labels
    ]

    for indices in groups.values():
        indices_sorted = sorted(indices, key=lambda i: sign * marker_labels[i][1], reverse=True)
        last_text_y: float | None = None
        for label_index in indices_sorted:
            _x, y, _text = marker_labels[label_index]
            marker_y_pts = sign * float(y) * pts_per_data
            dy = _PARAMETER_LABEL_BASE_DY
            text_y = marker_y_pts + dy
            if last_text_y is not None and text_y > last_text_y - min_gap_pts:
                text_y = last_text_y - min_gap_pts
                dy = text_y - marker_y_pts
            # Keep labels inside axes (inverted Y still has y0/y1 span).
            bound_a = sign * float(y0) * pts_per_data
            bound_b = sign * float(y1) * pts_per_data
            y_lo_pts, y_hi_pts = min(bound_a, bound_b), max(bound_a, bound_b)
            text_y = min(max(text_y, y_lo_pts + min_gap_pts * 0.25), y_hi_pts - min_gap_pts * 0.25)
            dy = text_y - marker_y_pts
            draw_leader = abs(dy - _PARAMETER_LABEL_BASE_DY) > _PARAMETER_LABEL_LEADER_EPS_PTS
            offsets[label_index] = (_PARAMETER_LABEL_DX, dy, draw_leader)
            last_text_y = text_y
    return offsets


class RendererChemistryMixin:
    """Parameter markers, fence segments, and compact chemistry legend."""

    def _register_chemistry_label(self, annotation, color: str, *, allow_leader: bool) -> None:
        """Hand a value label to the shared collision pass (renderer_water).

        The base position is beside the reading (not the stacked offset), so
        a leader appears only when a label ends up away from its value.
        """
        annotation._water_base_xyann = (_PARAMETER_LABEL_DX, _PARAMETER_LABEL_BASE_DY)
        annotation._leader_allowed = allow_leader
        if not hasattr(self, "_water_labels"):
            self._water_labels = []
        self._water_labels.append(("chem", annotation, color))

    def _draw_parameter_readings(
        self,
        ax,
        hole_summary: pd.DataFrame,
        collar_lookup: dict[str, float],
        *,
        profile_lookup: dict[str, tuple[float, float]] | None = None,
        column_half_m: float = 0.0,
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
        draw_markers = self.profile.parameter_draw_markers
        marker = _GW_MARKER_MAP.get(self.profile.parameter_marker, self.profile.parameter_marker)
        transect_hole_ids = hole_summary["hole_id"].astype(str).tolist()
        transect_x = (
            dict(zip(transect_hole_ids, hole_summary["x_profile"].to_numpy(dtype=float)))
            if draw_markers
            else {}
        )

        by_parameter: dict[str, list[EnvironmentalReading]] = {}
        profile_holes = set(profile_lookup)
        for reading in self.environmental_readings:
            if reading.parameter not in active_parameters or reading.hole_id not in profile_holes:
                continue
            by_parameter.setdefault(reading.parameter, []).append(reading)

        consulting = self.profile.layout == "consulting_section"
        font_size = (
            _PARAMETER_LABEL_FONTSIZE_CONSULTING if consulting else _PARAMETER_LABEL_FONTSIZE
        )
        # Column spans the collision pass must keep value labels off.
        self._column_spans = [
            (ax, float(profile_lookup[h][0]) - column_half_m, float(profile_lookup[h][0]) + column_half_m)
            for h in transect_hole_ids
            if h in profile_lookup
        ]
        self.parameter_series_legend = []
        # Stable per parameter name (chloride keeps its colour on every
        # section), with collisions on one sheet resolved to distinct colours.
        series_colors = parameter_series_colors(by_parameter)
        for parameter, readings in sorted(by_parameter.items()):
            if draw_markers:
                color = series_colors[parameter]
            else:
                color = CHEMISTRY_LABEL_BLACK
            readings_by_hole: dict[str, list[EnvironmentalReading]] = {}
            for reading in readings:
                readings_by_hole.setdefault(reading.hole_id, []).append(reading)
            for hole_readings in readings_by_hole.values():
                hole_readings.sort(key=lambda item: item.sample_depth)

            marker_xs: list[float] = []
            marker_ys: list[float] = []
            marker_colors: list[str] = []
            marker_labels: list[tuple[float, float, str, str]] = []
            interval_sticks: list[np.ndarray] = []
            y_cache: dict[tuple[str, float], float] = {}
            for hole_id, hole_readings in readings_by_hole.items():
                depths: list[float] = [reading.sample_depth for reading in hole_readings]
                if draw_markers:
                    depths.extend(
                        depth
                        for reading in hole_readings
                        for depth in (reading.from_depth, reading.to_depth)
                        if depth is not None
                    )
                unique_depths = list(dict.fromkeys(depths))
                plotted = self._plot_depths_below_collar(
                    hole_id, unique_depths, profile_lookup
                )
                for depth, y in zip(unique_depths, plotted, strict=True):
                    y_cache[(hole_id, float(depth))] = float(y)

            for hole_id in transect_hole_ids:
                hole_readings = readings_by_hole.get(hole_id)
                if not hole_readings:
                    continue
                x_profile = float(profile_lookup[hole_id][0])
                for reading in hole_readings:
                    y = y_cache[(hole_id, float(reading.sample_depth))]
                    if draw_markers:
                        marker_xs.append(x_profile)
                        marker_ys.append(y)
                        # "dot" style carries the workbook colour on the marker
                        # itself instead of adding a second dot beside the label.
                        marker_colors.append(
                            (CHEMISTRY_FIXED_COLORS.get(reading.label_color) or color)
                            if str(self.profile.chemistry_label_style or "plain") == "dot"
                            else color
                        )
                        if (
                            reading.from_depth is not None
                            and reading.to_depth is not None
                            and abs(reading.to_depth - reading.from_depth) > 1e-9
                        ):
                            y_top = y_cache[(hole_id, float(reading.from_depth))]
                            y_bottom = y_cache[(hole_id, float(reading.to_depth))]
                            interval_sticks.append(
                                np.asarray(
                                    [[x_profile, y_top], [x_profile, y_bottom]],
                                    dtype=float,
                                )
                            )
                    if label_values:
                        # Prefer compact numeric text; unit belongs in the legend
                        # unless parameter_label_include_units is enabled.
                        if reading.value_label:
                            label_text = reading.value_label
                        elif self.profile.parameter_label_include_units:
                            label_text = reading.display_label
                        else:
                            label_text = f"{reading.value:g}"
                        # A colour picked in the workbook wins over the
                        # Configure threshold / black setting.
                        label_color = CHEMISTRY_FIXED_COLORS.get(
                            reading.label_color
                        ) or chemistry_label_color(
                            reading.value,
                            self.profile.chemistry_color_mode,
                            green_max=self.profile.chemistry_threshold_green_max,
                            yellow_max=self.profile.chemistry_threshold_yellow_max,
                        )
                        marker_labels.append((x_profile, y, label_text, label_color))

            if draw_markers:
                if not marker_xs:
                    continue
                if interval_sticks:
                    stick_collection = LineCollection(
                        interval_sticks,
                        colors=color,
                        linewidths=2.0,
                        linestyles="-",
                        zorder=7,
                        capstyle="round",
                    )
                    ax.add_collection(stick_collection)
                ax.scatter(
                    marker_xs,
                    marker_ys,
                    marker=marker,
                    c=marker_colors,
                    s=float(self.profile.parameter_marker_size),
                    zorder=8,
                )
            elif not marker_labels:
                units = sorted(
                    {
                        (reading.unit or "").strip()
                        for reading in readings
                        if (reading.unit or "").strip()
                    }
                )
                empty_label = (
                    f"{parameter.upper()} CONCENTRATION ({units[0]})"
                    if len(units) == 1
                    else f"{parameter.upper()} CONCENTRATION"
                )
                self.parameter_series_legend.append(
                    {
                        "parameter": parameter,
                        "color": color,
                        "marker": marker,
                        "label": empty_label,
                    }
                )
                continue
            label_offsets = _resolve_parameter_label_offsets(
                ax,
                [(x, y, text) for x, y, text, _ in marker_labels],
                invert_y=self.profile.y_axis_mode == "depth_below_collar",
            )
            label_style = str(self.profile.chemistry_label_style or "plain")
            label_base_kwargs: dict[str, object] = {
                "textcoords": "offset points",
                "fontsize": font_size,
                "zorder": 9,
                "clip_on": False,
                "ha": "left",
                "va": "center",
            }
            if draw_markers or label_style == "box":
                label_base_kwargs["bbox"] = (
                    _PARAMETER_LABEL_BBOX_SOLID if label_style == "box" else _PARAMETER_LABEL_BBOX
                )
            draw_leaders = self.profile.parameter_draw_leaders
            for (x_profile, y, label_text, label_color), (dx, dy, _draw_leader) in zip(
                marker_labels, label_offsets, strict=True
            ):
                anchor = (x_profile + column_half_m, y)
                if label_style == "dot" and not draw_markers:
                    dx += 5.0
                text_color = CHEMISTRY_LABEL_BLACK if label_style == "dot" else label_color
                annotation = ax.annotate(
                    label_text,
                    **label_base_kwargs,
                    # Anchor at the column's right edge: anchored at its centre,
                    # labels started inside wider (auto-fit) columns.
                    xy=anchor,
                    xytext=(dx, dy),
                    color=text_color,
                    # Leader exists from the start, hidden until the collision
                    # pass moves the label away from its reading.
                    arrowprops={
                        "arrowstyle": "-",
                        "color": label_color,
                        "lw": 0.55,
                        "linestyle": "--",
                        "shrinkA": 2,
                        "shrinkB": 1,
                        "alpha": 0.7,
                    },
                )
                annotation.arrow_patch.set_visible(False)
                if label_style == "stroke":
                    # The halo is a white twin drawn as outlines beneath the
                    # label; the label itself stays real text so PDF values
                    # remain searchable (a path effect on the label would not).
                    halo = ax.annotate(
                        label_text,
                        xy=anchor,
                        xytext=(dx, dy),
                        textcoords="offset points",
                        fontsize=font_size,
                        ha="left",
                        va="center",
                        color="white",
                        zorder=8.9,
                        clip_on=False,
                        path_effects=[withStroke(linewidth=2.6, foreground="white")],
                    )
                    annotation._halo = halo
                if label_style == "dot" and not draw_markers:
                    # Colour travels on a dot just left of the value; the dot is
                    # positioned in points off the anchor so it follows the
                    # label when the collision pass moves it.
                    (dot,) = ax.plot(
                        [anchor[0]],
                        [y],
                        marker="o",
                        markersize=3.6,
                        color=label_color,
                        linestyle="none",
                        zorder=9,
                        clip_on=False,
                        transform=offset_copy(ax.transData, fig=ax.figure, x=dx - 4.5, y=dy, units="points"),
                    )
                    annotation._chem_dot = dot
                self._register_chemistry_label(annotation, label_color, allow_leader=draw_leaders)

            if draw_markers and (
                use_segments or across_gaps
            ):
                self._draw_parameter_fence(
                    ax,
                    transect_hole_ids,
                    transect_x,
                    readings_by_hole,
                    y_cache,
                    profile_lookup,
                    color,
                    use_segments=use_segments,
                    across_gaps=across_gaps,
                )
            legend_label = parameter.upper()
            units = sorted(
                {
                    (reading.unit or "").strip()
                    for reading in readings
                    if (reading.unit or "").strip()
                }
            )
            if not self.profile.parameter_label_include_units and len(units) == 1:
                legend_label = f"{parameter.upper()} CONCENTRATION ({units[0]})"
            elif not draw_markers:
                legend_label = (
                    f"{parameter.upper()} CONCENTRATION ({units[0]})"
                    if len(units) == 1
                    else f"{parameter.upper()} CONCENTRATION"
                )
            self.parameter_series_legend.append(
                {
                    "parameter": parameter,
                    "color": color,
                    "marker": marker,
                    "label": legend_label,
                }
            )

    def _draw_parameter_fence(
        self,
        ax,
        transect_hole_ids: list[str],
        transect_x: dict[str, float],
        readings_by_hole: dict[str, list[EnvironmentalReading]],
        y_cache: dict[tuple[str, float], float],
        profile_lookup: dict[str, tuple[float, float]],
        color: str,
        *,
        use_segments: bool,
        across_gaps: bool,
    ) -> None:
        measured_holes = [hole_id for hole_id in transect_hole_ids if readings_by_hole.get(hole_id)]
        fence_segments: list[np.ndarray] = []
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
                    y0 = y_cache[(left_id, float(left_reading.sample_depth))]
                    y1 = y_cache[(right_id, float(right_reading.sample_depth))]
                    fence_segments.append(
                        np.asarray([[x0, y0], [x1, y1]], dtype=float)
                    )
            if fence_segments:
                collection = LineCollection(
                    fence_segments,
                    colors=color,
                    linewidths=1.5,
                    linestyles="--",
                    zorder=7,
                )
                ax.add_collection(collection)
            return
        if len(measured_holes) < 2:
            return
        if all(len(readings_by_hole[hole_id]) == 1 for hole_id in measured_holes):
            bands = [[(hole_id, readings_by_hole[hole_id][0]) for hole_id in measured_holes]]
        else:
            bands = _cluster_parameter_bands_by_depth(
                measured_holes, readings_by_hole, depth_tol=1.5
            )
        for band in bands:
            if len(band) < 2:
                continue
            band.sort(key=lambda item: float(profile_lookup[item[0]][0]))
            xs_arr = np.asarray(
                [float(profile_lookup[hole_id][0]) for hole_id, _reading in band],
                dtype=float,
            )
            ys_arr = np.asarray(
                [y_cache[(hole_id, float(reading.sample_depth))] for hole_id, reading in band],
                dtype=float,
            )
            if across_gaps and len(xs_arr) >= 2:
                x_dense = np.linspace(float(xs_arr.min()), float(xs_arr.max()), 100)
                y_dense = np.interp(x_dense, xs_arr, ys_arr)
                fence_segments.append(np.column_stack([x_dense, y_dense]))
            else:
                fence_segments.append(np.column_stack([xs_arr, ys_arr]))
        if fence_segments:
            collection = LineCollection(
                fence_segments,
                colors=color,
                linewidths=1.5,
                linestyles="--",
                zorder=7,
            )
            ax.add_collection(collection)

    def _draw_compact_parameter_legend(self, ax) -> None:
        if not self.profile.show_parameter_legend_text or not self.parameter_series_legend:
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
