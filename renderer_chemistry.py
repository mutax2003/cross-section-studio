"""Environmental / chemistry parameter drawing mixin for CrossSectionRenderer."""

from __future__ import annotations

from bisect import bisect_left
from typing import Sequence, TypedDict

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from models import EnvironmentalReading
from render_theme import (
    CHEMISTRY_LABEL_BLACK,
    LABEL_COLOR,
    PARAMETER_PALETTE,
    chemistry_label_color,
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


class RendererChemistryMixin:
    """Parameter markers, fence segments, and compact chemistry legend."""

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
        self.parameter_series_legend = []
        for index, (parameter, readings) in enumerate(sorted(by_parameter.items())):
            if draw_markers:
                color = PARAMETER_PALETTE[index % len(PARAMETER_PALETTE)]
            else:
                color = CHEMISTRY_LABEL_BLACK
            readings_by_hole: dict[str, list[EnvironmentalReading]] = {}
            for reading in readings:
                readings_by_hole.setdefault(reading.hole_id, []).append(reading)
            for hole_readings in readings_by_hole.values():
                hole_readings.sort(key=lambda item: item.sample_depth)

            marker_xs: list[float] = []
            marker_ys: list[float] = []
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
                        label_color = chemistry_label_color(
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
                    c=color,
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
                ax, [(x, y, text) for x, y, text, _ in marker_labels]
            )
            label_base_kwargs: dict[str, object] = {
                "textcoords": "offset points",
                "fontsize": font_size,
                "color": color,
                "zorder": 9,
                "clip_on": False,
                "ha": "left",
                "va": "center",
            }
            if draw_markers:
                label_base_kwargs["bbox"] = _PARAMETER_LABEL_BBOX
            draw_leaders = self.profile.parameter_draw_leaders
            for (x_profile, y, label_text, label_color), (dx, dy, draw_leader) in zip(
                marker_labels, label_offsets, strict=True
            ):
                annotate_kwargs = {
                    **label_base_kwargs,
                    "xy": (x_profile, y),
                    "xytext": (dx, dy),
                    "color": label_color,
                }
                if draw_leaders and draw_leader:
                    leader_props = {
                        "arrowstyle": "-",
                        "color": label_color,
                        "lw": 0.55,
                        "linestyle": "--",
                        "shrinkA": 2,
                        "shrinkB": 1,
                        "alpha": 0.7,
                    }
                    annotate_kwargs["arrowprops"] = leader_props
                ax.annotate(label_text, **annotate_kwargs)

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
