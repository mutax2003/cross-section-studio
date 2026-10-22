"""Environmental / chemistry parameter drawing mixin for CrossSectionRenderer."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from typing import NotRequired, TypedDict

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
from matplotlib.patheffects import withStroke
from matplotlib.transforms import Bbox, offset_copy

from chemistry_qc import harmonize_units, rdl_above_guideline_note, rdl_exceeds_guideline
from models import EnvironmentalReading
from render_profiles import resolved_chemistry_thresholds_for
from render_theme import (
    CHEMISTRY_FIXED_COLORS,
    CHEMISTRY_LABEL_BLACK,
    CHEMISTRY_LABEL_INCONCLUSIVE,
    LABEL_COLOR,
    chemistry_fixed_mode_color,
    chemistry_label_color,
    chemistry_threshold_bands,
    parameter_series_colors,
)
from renderer_common import LEGEND_LABEL_MAX_CHARS, THRESHOLD_KEY_GID, fit_label
from renderer_water import _GW_MARKER_MAP, _memoise_label, register_marker_set


class ParameterLegendEntry(TypedDict):
    parameter: str
    color: str
    marker: str
    label: str
    # "--" when a dashed fence joins the readings on the plot, "none" when
    # the markers stand alone; the legend glyph follows it.
    linestyle: NotRequired[str]
    # Single fixed colour mode other than black (P2 red): a value from the
    # figure in that colour, drawn beside the label as the client legend does.
    sample_text: NotRequired[str]
    sample_color: NotRequired[str]
    # Labels-only figures with several parameters: the glyph (matplotlib
    # marker) each value carries, drawn before the sample in the key.
    sample_marker: NotRequired[str]
    # Threshold colour mode: the band colours (green, orange, red) the legend
    # shows as a small three-dot sample instead of the series line/marker.
    threshold_colors: NotRequired[tuple[str, ...]]


_PARAMETER_LABEL_MIN_GAP_PTS = 26.0
# The "N values not printed" line under the threshold key.
_OMITTED_FOOTNOTE_GID = "chemistry-omitted-footnote"
# Clearance between the lowest x-axis text and the threshold key below it.
_THRESHOLD_KEY_GAP_PTS = 3.0
# Threshold mode: a fence segment joining readings in different bands.
_THRESHOLD_MIXED_SEGMENT_COLOR = "#6B7280"
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
# Marker shapes handed out to the second, third, ... plotted parameter (the
# first keeps the Configure marker): with two parameters at one hole the dots
# and the legend samples are the only thing telling "0.4" from "380".
_PARAMETER_MARKER_CYCLE = ("o", "s", "^", "D", "v", "P", "X", "*", "h")
# Threshold key: a parameter name longer than this takes its own row above its
# band text (inline, a 40-character name pushed the row past the frame).
_THRESHOLD_KEY_INLINE_NAME_CHARS = 12
# Glyph (points) beside each value on labels-only figures with several
# parameters: the parameter's marker shape, in the value's colour.
_PARAMETER_GLYPH_SIZE = 4.2
# Legend sample text with a glyph starts with two figure spaces (about 1.27 em)
# where the glyph is drawn, centred this many em in.
_GLYPH_SPACER = "\u2007\u2007"
_GLYPH_SPACER_CENTRE_EM = 0.55


def _legend_units(readings: Sequence[EnvironmentalReading], main_unit: str) -> list[str]:
    """Unit(s) a parameter's legend entry names: its main unit when it has one
    (readings in other units were converted to it, or print their own unit)."""
    if main_unit:
        return [main_unit]
    return sorted({(reading.unit or "").strip() for reading in readings if (reading.unit or "").strip()})


def parameter_series_markers(parameters: Sequence[str], base_marker: str) -> dict[str, str]:
    """Distinct matplotlib marker per parameter, in the given (Configure) order.

    The first parameter keeps ``base_marker`` (the Configure choice, circle by
    default) so a single-parameter figure is unchanged; each further parameter
    takes the next unused shape from ``_PARAMETER_MARKER_CYCLE``.
    """
    markers: dict[str, str] = {}
    used: set[str] = set()
    for index, name in enumerate(parameters):
        if name in markers:
            continue
        marker = base_marker
        if index:
            marker = next(
                (shape for shape in _PARAMETER_MARKER_CYCLE if shape not in used and shape != base_marker),
                base_marker,
            )
        markers[name] = marker
        used.add(marker)
    return markers


def _fit_legend_label(text: str) -> str:
    """Legend text wrapped / ellipsized like the lithology entries (a 40-character
    parameter name pushed the key off a letter page)."""
    return fit_label(text, max_chars=LEGEND_LABEL_MAX_CHARS)


def _shown_threshold_bands(green_max: float, yellow_max: float):
    """Threshold bands a value can fall in: no orange band when its limits
    coincide (green 50 / orange 50 reads "green <= 50 . red > 50", not
    "orange 50-50")."""
    bands = chemistry_threshold_bands(green_max, yellow_max)
    if float(yellow_max) <= float(green_max):
        bands = tuple(band for band in bands if band[0] != "orange")
    return bands


def _inconclusive_band(green_max: float):
    """Key band for non-detects whose detection limit is above the green limit."""
    return (("grey", CHEMISTRY_LABEL_INCONCLUSIVE, f"ND (RDL > {float(green_max):g})"),)


def _threshold_key_text(
    green_max: float,
    yellow_max: float,
    unit: str | None = None,
    *,
    inconclusive: bool = False,
) -> str:
    """Plain-text key, e.g. ``green <= 100 . orange 100-300 . red > 300 mg/L``
    (plus ``. grey ND (RDL > 100)`` when a non-detect's limit is above green)."""
    bands = _shown_threshold_bands(green_max, yellow_max)
    if inconclusive:
        bands = bands + _inconclusive_band(green_max)
    text = " \u00b7 ".join(f"{name} {span}" for name, _hex, span in bands)
    unit = (unit or "").strip()
    return f"{text} {unit}" if unit else text


def _threshold_key_lines(
    entries: Sequence[tuple[str, str, float, float]],
    inconclusive: set[str] | frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """One key line per plotted parameter: ``(parameter, unit, green, orange)``.

    A single parameter keeps the bare band text (``green <= 100 . ...``); with
    several parameters each line is prefixed by its parameter name, since
    every parameter carries its own limits in its own units.
    """
    if len(entries) == 1:
        parameter, unit, green_max, yellow_max = entries[0]
        return (
            _threshold_key_text(
                green_max, yellow_max, unit, inconclusive=parameter in inconclusive
            ),
        )
    return tuple(
        f"{parameter}: "
        + _threshold_key_text(green_max, yellow_max, unit, inconclusive=parameter in inconclusive)
        for parameter, unit, green_max, yellow_max in entries
    )


def _below_x_axis_anchor(ax):
    """Display bbox centred on the distance label, bottomed at the lowest x-axis text.

    Covers the tick labels, the axes x label and a figure-level ``supxlabel``
    (section sheets with column headers label the distance axis that way), so
    the key centres under whichever label is printed and never rides up into it.
    """

    def _bbox(renderer) -> Bbox:
        axes_box = ax.get_window_extent(renderer)
        bottom = axes_box.y0
        centre = 0.5 * (axes_box.x0 + axes_box.x1)
        tight = ax.xaxis.get_tightbbox(renderer)
        if tight is not None and tight.height > 0:
            bottom = min(bottom, tight.y0)
        supx = getattr(ax.figure, "_supxlabel", None)
        labels = [ax.xaxis.label, supx]
        for label in labels:
            if label is None or not label.get_visible() or not label.get_text():
                continue
            box = label.get_window_extent(renderer)
            if box.height > 0 and box.y0 <= bottom:
                bottom = box.y0
                centre = 0.5 * (box.x0 + box.x1)
        return Bbox.from_extents(centre - 1.0, bottom, centre + 1.0, axes_box.y1)

    return _bbox


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
        # Values (and their leaders) draw just above water labels, so a
        # leader is never hidden under a water label's white box.
        annotation.set_zorder(max(float(annotation.get_zorder()), 9.1))
        # Memoised text layout / leader path (same artist, same output).
        _memoise_label(annotation)
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
        # Values dropped by the label placement (filled by renderer_water).
        self.chemistry_label_notes = []
        # Threshold mode: non-detects whose detection limit is above the
        # parameter's green limit (inconclusive), one QA note each.
        self.chemistry_rdl_notes = []
        # Column spans the collision pass keeps labels off (chemistry values
        # and water numbers alike), set even when no chemistry is drawn.
        self._column_spans = []
        # (ax, x0, x1, y_lo, y_hi) of column pieces off the collar x.
        self._column_extra_spans = []
        if not hole_summary.empty:
            if profile_lookup is None:
                profile_lookup = self._profile_lookup(hole_summary, collar_lookup)
            self._column_spans = [
                (ax, float(profile_lookup[h][0]) - column_half_m, float(profile_lookup[h][0]) + column_half_m)
                for h in hole_summary["hole_id"].astype(str).tolist()
                if h in profile_lookup
            ]
            # Inclined (surveyed) holes step sideways down the hole: the
            # stepped pieces away from the collar x are obstacles too (a
            # value or gradient printed across one hides the column), over
            # their own height only.
            pieces_of = getattr(self, "_deviated_pieces", None)
            if pieces_of is not None and getattr(self, "_deviated_x", None):
                for hole_id in hole_summary["hole_id"].astype(str).tolist():
                    if hole_id not in profile_lookup:
                        continue
                    x_collar, collar_rl = profile_lookup[hole_id]
                    for y_lo, y_hi, piece_x in pieces_of(hole_id, float(collar_rl), -np.inf, np.inf) or ():
                        if abs(float(piece_x) - float(x_collar)) <= 1e-6:
                            continue
                        self._column_extra_spans.append(
                            (ax, float(piece_x) - column_half_m, float(piece_x) + column_half_m, y_lo, y_hi)
                        )
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
        base_marker = _GW_MARKER_MAP.get(self.profile.parameter_marker, self.profile.parameter_marker)
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
        # One unit per parameter on the figure: readings in another,
        # convertible unit are shown (and threshold-coloured) in the
        # parameter's main unit; unconvertible ones print with their own unit
        # in a neutral colour. Each case gets a QA note (chemistry_qc).
        harmonized = harmonize_units(
            [reading for readings in by_parameter.values() for reading in readings]
        )
        if harmonized.notes:
            converted = iter(harmonized.readings)
            by_parameter = {
                parameter: [next(converted) for _reading in readings]
                for parameter, readings in by_parameter.items()
            }
            # Read with the RDL notes into the section / PDF QA notes.
            self.chemistry_rdl_notes.extend(harmonized.notes)
        unconverted_ids = {id(reading) for reading in harmonized.unconverted}

        consulting = self.profile.layout == "consulting_section"
        font_size = (
            _PARAMETER_LABEL_FONTSIZE_CONSULTING if consulting else _PARAMETER_LABEL_FONTSIZE
        )
        self.parameter_series_legend = []
        # Per parameter: the units of readings whose label colour came from
        # the threshold bands (not a workbook colour) and the limits used;
        # drives the on-figure threshold key, one line per parameter.
        threshold_key_units: dict[str, set[str]] = {}
        threshold_key_limits: dict[str, tuple[float, float]] = {}
        # Parameters with an inconclusive (grey) non-detect on the figure: their
        # key line names the grey too.
        threshold_key_inconclusive: set[str] = set()
        threshold_mode = self.profile.chemistry_color_mode == "threshold"
        threshold_active = bool(label_values) and self.profile.chemistry_color_mode == "threshold"
        self.chemistry_threshold_key_text = None
        self.chemistry_threshold_key_lines = ()
        self._chemistry_threshold_key = None
        # Fixed label colour (P2 red or black) earns a sample value in the
        # consulting legend, as on the client P2 figures; threshold mode has
        # its own colour key under the distance label instead.
        fixed_color = chemistry_fixed_mode_color(self.profile.chemistry_color_mode)
        legend_sample_color = fixed_color if label_values and consulting and fixed_color else None
        # Stable per parameter name (chloride keeps its colour on every
        # section), with collisions on one sheet resolved to distinct colours.
        series_colors = parameter_series_colors(by_parameter)
        # Marker shape per parameter, in Configure order: the first keeps the
        # Configure marker, the rest take distinct shapes (see the cycle).
        configured_order = [
            name for name in dict.fromkeys(n.strip() for n in self.environmental_parameters) if name in by_parameter
        ]
        configured_order += sorted(name for name in by_parameter if name not in configured_order)
        series_markers = parameter_series_markers(configured_order, base_marker)
        several_parameters = len(by_parameter) >= 2
        for parameter, readings in sorted(by_parameter.items()):
            marker = series_markers[parameter]
            if draw_markers:
                color = series_colors[parameter]
            else:
                color = CHEMISTRY_LABEL_BLACK
            # Each parameter is banded by its own limits (in its own units);
            # a parameter without an entry uses the sheet-wide pair, and any
            # limit still missing the Configure defaults (the pipeline fills
            # those too), so labels, dots and the key always agree.
            green_max, yellow_max = resolved_chemistry_thresholds_for(
                parameter,
                self.profile.chemistry_thresholds,
                self.profile.chemistry_threshold_green_max,
                self.profile.chemistry_threshold_yellow_max,
            )
            readings_by_hole: dict[str, list[EnvironmentalReading]] = {}
            for reading in readings:
                readings_by_hole.setdefault(reading.hole_id, []).append(reading)
            for hole_readings in readings_by_hole.values():
                hole_readings.sort(key=lambda item: item.sample_depth)

            marker_xs: list[float] = []
            marker_ys: list[float] = []
            marker_colors: list[str] = []
            marker_labels: list[tuple[float, float, str, str]] = []
            label_holes: list[str] = []
            interval_sticks: list[np.ndarray] = []
            stick_colors: list[str] = []
            # Threshold mode: each reading's band colour, for its interval
            # stick and the fence segments that join it to its neighbours.
            reading_colors: dict[int, str] = {}
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
                    # A unit that cannot be converted to the parameter's main
                    # unit: never banded against limits in another unit.
                    foreign_unit = id(reading) in unconverted_ids
                    # A colour picked in the workbook wins over the Configure
                    # threshold / black setting.
                    label_color = CHEMISTRY_FIXED_COLORS.get(reading.label_color) or (
                        CHEMISTRY_LABEL_BLACK
                        if foreign_unit
                        else chemistry_label_color(
                            reading.value,
                            self.profile.chemistry_color_mode,
                            green_max=green_max,
                            yellow_max=yellow_max,
                            detected=reading.detected,
                        )
                    )
                    if threshold_mode and not foreign_unit and rdl_exceeds_guideline(reading, green_max):
                        self.chemistry_rdl_notes.append(
                            rdl_above_guideline_note(reading, green_max)
                        )
                    if threshold_active:
                        reading_colors[id(reading)] = label_color
                    if draw_markers:
                        marker_xs.append(x_profile)
                        marker_ys.append(y)
                        # Threshold mode colours each reading's dot like its
                        # value (green / orange / red, or the workbook colour),
                        # so a dot never contradicts its label. "dot" style
                        # carries the workbook colour on the marker itself
                        # instead of adding a second dot beside the label.
                        if threshold_active:
                            marker_colors.append(label_color)
                        elif str(self.profile.chemistry_label_style or "plain") == "dot":
                            marker_colors.append(
                                CHEMISTRY_FIXED_COLORS.get(reading.label_color) or color
                            )
                        else:
                            marker_colors.append(color)
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
                            # The stick behind a dot takes the dot's colour.
                            stick_colors.append(marker_colors[-1])
                    if label_values:
                        # Prefer compact numeric text; unit belongs in the legend
                        # unless parameter_label_include_units is enabled.
                        if reading.value_label:
                            label_text = reading.value_label
                        elif self.profile.parameter_label_include_units or foreign_unit:
                            # A value left in another unit always says which.
                            label_text = reading.display_label
                        else:
                            # "<5" for a non-detect: never a bare detection limit.
                            label_text = reading.value_text
                        marker_labels.append((x_profile, y, label_text, label_color))
                        label_holes.append(hole_id)
                        if (
                            threshold_active
                            and not foreign_unit
                            and reading.label_color not in CHEMISTRY_FIXED_COLORS
                        ):
                            threshold_key_units.setdefault(parameter, set()).add(
                                (reading.unit or "").strip()
                            )
                            threshold_key_limits[parameter] = (green_max, yellow_max)
                            if label_color == CHEMISTRY_LABEL_INCONCLUSIVE:
                                threshold_key_inconclusive.add(parameter)

            if draw_markers:
                if not marker_xs:
                    continue
                if interval_sticks:
                    stick_collection = LineCollection(
                        interval_sticks,
                        colors=stick_colors if threshold_active else color,
                        linewidths=2.0,
                        linestyles="-",
                        zorder=7,
                        capstyle="round",
                    )
                    ax.add_collection(stick_collection)
                dots = ax.scatter(
                    marker_xs,
                    marker_ys,
                    marker=marker,
                    c=marker_colors,
                    s=float(self.profile.parameter_marker_size),
                    # Over an LNAPL band (9.1) at the same depth, under the
                    # groundwater markers (9.3).
                    zorder=9.2,
                )
                # Same-depth readings and groundwater markers on a dot are
                # nudged apart by the label pass.
                register_marker_set(ax, dots, "chem")
            elif not marker_labels:
                units = _legend_units(readings, harmonized.dominant_unit_for(parameter))
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
                        "label": _fit_legend_label(empty_label),
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
            # Only "box" draws a background box. "strip" values sit on a
            # knocked-out strip (drawn by the collision pass once they are
            # placed), "stroke" carries a white halo only and "plain" is bare
            # text, so the four styles read differently.
            if label_style == "box":
                label_base_kwargs["bbox"] = _PARAMETER_LABEL_BBOX_SOLID
            draw_leaders = self.profile.parameter_draw_leaders
            # Labels only (no dots at the holes): each value carries a glyph
            # in the "dot" style, and whenever several parameters are on the
            # sheet, so a value reads as its parameter's (the legend keys the
            # same shape).
            glyph_labels = not draw_markers and (label_style == "dot" or several_parameters)
            for (x_profile, y, label_text, label_color), (dx, dy, _draw_leader), hole_id in zip(
                marker_labels, label_offsets, label_holes, strict=True
            ):
                anchor = (x_profile + column_half_m, y)
                if glyph_labels:
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
                # Named in the "values not printed" QA note if it is dropped.
                annotation._chem_hole_id = hole_id
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
                if glyph_labels:
                    # Colour (and the parameter's shape) travels on a glyph
                    # just left of the value; positioned in points off the
                    # anchor so it follows the label when the collision pass
                    # moves it.
                    (dot,) = ax.plot(
                        [anchor[0]],
                        [y],
                        marker=marker,
                        markersize=3.6 if label_style == "dot" else _PARAMETER_GLYPH_SIZE,
                        color=label_color,
                        linestyle="none",
                        zorder=9,
                        clip_on=False,
                        transform=offset_copy(ax.transData, fig=ax.figure, x=dx - 4.5, y=dy, units="points"),
                    )
                    annotation._chem_dot = dot
                self._register_chemistry_label(annotation, label_color, allow_leader=draw_leaders)

            fence_drawn = False
            if draw_markers and (
                use_segments or across_gaps
            ):
                fence_drawn = self._draw_parameter_fence(
                    ax,
                    transect_hole_ids,
                    transect_x,
                    readings_by_hole,
                    y_cache,
                    profile_lookup,
                    color,
                    use_segments=use_segments,
                    across_gaps=across_gaps,
                    reading_colors=reading_colors or None,
                )
            legend_label = parameter.upper()
            units = _legend_units(readings, harmonized.dominant_unit_for(parameter))
            if not self.profile.parameter_label_include_units and len(units) == 1:
                legend_label = f"{parameter.upper()} CONCENTRATION ({units[0]})"
            elif not draw_markers:
                legend_label = (
                    f"{parameter.upper()} CONCENTRATION ({units[0]})"
                    if len(units) == 1
                    else f"{parameter.upper()} CONCENTRATION"
                )
            legend_entry: ParameterLegendEntry = {
                "parameter": parameter,
                "color": color,
                "marker": marker,
                "label": _fit_legend_label(legend_label),
                # The key draws what the plot draws: a dashed fence line only
                # when one joins the readings, else the isolated markers.
                "linestyle": "--" if fence_drawn else "none",
            }
            if legend_sample_color is not None and not draw_markers and marker_labels:
                legend_entry["sample_text"] = next(
                    (text for _x, _y, text, c in marker_labels if c == legend_sample_color),
                    marker_labels[0][2],
                )
                legend_entry["sample_color"] = legend_sample_color
            if glyph_labels and marker_labels:
                # The key shows the glyph the values carry, in their colour.
                legend_entry["sample_marker"] = marker
                if legend_entry.get("sample_text"):
                    legend_entry["sample_text"] = _GLYPH_SPACER + legend_entry["sample_text"]
                if fixed_color and not threshold_active:
                    legend_entry["sample_color"] = legend_entry.get("sample_color") or fixed_color
            if threshold_active:
                legend_entry["threshold_colors"] = tuple(
                    hex_color
                    for _name, hex_color, _span in _shown_threshold_bands(green_max, yellow_max)
                )
            self.parameter_series_legend.append(legend_entry)
        if threshold_key_limits:
            entries = []
            for parameter in sorted(threshold_key_limits):
                named_units = sorted(unit for unit in threshold_key_units[parameter] if unit)
                unit = named_units[0] if len(named_units) == 1 else ""
                green_limit, yellow_limit = threshold_key_limits[parameter]
                entries.append((parameter, unit, green_limit, yellow_limit))
            self._draw_chemistry_threshold_key(
                ax, entries, font_size, inconclusive=threshold_key_inconclusive
            )

    def _draw_chemistry_threshold_key(
        self,
        ax,
        entries: Sequence[tuple[str, str, float, float]],
        font_size: float,
        *,
        inconclusive: set[str] | frozenset[str] = frozenset(),
    ) -> None:
        """State the threshold bands in words on the figure (threshold mode only).

        Colour alone must not carry the band (WCAG 1.4.1): each band is named
        and given its range, e.g. ``green ≤ 100 · orange 100–300 · red > 300 mg/L``,
        with the name drawn in the band colour as a secondary cue. One row per
        plotted parameter (``(parameter, unit, green_max, orange_max)``): each
        parameter has its own limits in its own units, so a shared row would
        mis-state one of them.
        """
        self.chemistry_threshold_key_lines = _threshold_key_lines(entries, inconclusive)
        self.chemistry_threshold_key_text = "\n".join(self.chemistry_threshold_key_lines)
        text_props = {"fontsize": font_size, "color": LABEL_COLOR}
        rows: list[HPacker] = []
        for index, (parameter, unit, green_max, yellow_max) in enumerate(entries):
            parts: list[TextArea] = []
            if index == 0:
                parts.append(TextArea("Label colour:", textprops=text_props))
            if len(entries) > 1:
                if len(parameter) > _THRESHOLD_KEY_INLINE_NAME_CHARS:
                    # A long name takes its own row (wrapped to at most two
                    # lines) so the band text under it never runs past the frame.
                    name = fit_label(parameter, max_chars=LEGEND_LABEL_MAX_CHARS)
                    parts.append(TextArea(f"{name}:", textprops=text_props))
                    rows.append(HPacker(children=parts, align="baseline", pad=0, sep=3))
                    parts = []
                else:
                    parts.append(TextArea(f"{parameter}:", textprops=text_props))
            bands = _shown_threshold_bands(green_max, yellow_max)
            if parameter in inconclusive:
                bands = bands + _inconclusive_band(green_max)
            for band_index, (name, hex_color, span) in enumerate(bands):
                parts.append(
                    TextArea(name, textprops={**text_props, "color": hex_color, "fontweight": "bold"})
                )
                suffix = span if band_index == len(bands) - 1 else f"{span} \u00b7"
                parts.append(TextArea(suffix, textprops=text_props))
            if unit:
                parts.append(TextArea(unit, textprops=text_props))
            rows.append(HPacker(children=parts, align="baseline", pad=0, sep=3))
        # Off the plot, in the white space under the x-axis label: anchored to
        # the axis' drawn extent at draw time, so it follows the axis through
        # export page resizes and margin refits without a re-layout pass.
        # A column: the key rows, plus (after the label pass) a footnote when
        # values had to be left off. Every layout that makes room for the
        # key measures the whole artist, so extra rows get room too.
        stack = VPacker(
            children=rows,
            align="center",
            pad=0,
            sep=2,
        )
        key = AnnotationBbox(
            stack,
            (0.5, 0.0),
            xycoords=_below_x_axis_anchor(ax),
            xybox=(0.0, -_THRESHOLD_KEY_GAP_PTS),
            boxcoords="offset points",
            box_alignment=(0.5, 1.0),
            frameon=False,
            pad=0.0,
            annotation_clip=False,
        )
        key.set_gid(THRESHOLD_KEY_GID)
        key.set_zorder(9.5)
        key._css_stack = stack
        key._css_rows = list(stack.get_children())
        ax.add_artist(key)
        self._chemistry_threshold_key = key

    def _update_chemistry_omitted_footnote(self, fig, labels) -> None:
        """On-figure note when the label pass had to leave values off.

        The QA note (app / PDF notes) names them; the figure itself says how
        many are missing, as a line under the threshold key (or, without a
        key, in the key's place under the distance label), so a reader of
        the printed sheet knows values exist.
        """
        key = getattr(self, "_chemistry_threshold_key", None)
        if key is not None and getattr(key, "_css_footnote_only", False):
            key.remove()
            self._chemistry_threshold_key = key = None
        if key is not None and key.figure is not fig:
            key = None
        dropped = [
            annotation
            for kind, annotation, _color in labels
            if kind == "chem" and getattr(annotation, "_water_dropped", False) and annotation.figure is fig
        ]
        stack = getattr(key, "_css_stack", None)
        if stack is not None:
            stack._children = list(key._css_rows)
            stack.stale = True
            key.stale = True
        if not dropped:
            return
        count = len(dropped)
        noun = "value" if count == 1 else "values"
        text = f"{count} {noun} not printed (no room at this scale) \u2014 see QA notes"
        font_size = (
            _PARAMETER_LABEL_FONTSIZE_CONSULTING
            if self.profile.layout == "consulting_section"
            else _PARAMETER_LABEL_FONTSIZE
        )
        note = TextArea(text, textprops={"fontsize": font_size, "color": LABEL_COLOR, "fontstyle": "italic"})
        note._text.set_gid(_OMITTED_FOOTNOTE_GID)
        if stack is not None:
            note.set_figure(fig)
            stack._children.append(note)
            return
        ax = dropped[0].axes
        # No threshold key: the note takes the key's row (same artist id, so
        # the scale bar row and page layouts make room for it).
        box = AnnotationBbox(
            note,
            (0.5, 0.0),
            xycoords=_below_x_axis_anchor(ax),
            xybox=(0.0, -_THRESHOLD_KEY_GAP_PTS),
            boxcoords="offset points",
            box_alignment=(0.5, 1.0),
            frameon=False,
            pad=0.0,
            annotation_clip=False,
        )
        box.set_gid(THRESHOLD_KEY_GID)
        box.set_zorder(9.5)
        box._css_footnote_only = True
        ax.add_artist(box)
        self._chemistry_threshold_key = box

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
        reading_colors: dict[int, str] | None = None,
    ) -> bool:
        """Dashed lines joining a parameter's readings between holes.

        With ``reading_colors`` (threshold mode) each hole-to-hole segment
        takes the band colour its two readings share, or a neutral grey when
        they fall in different bands: a red line under a green dot (or half
        red / half green) would contradict the values it joins. Returns True
        when at least one segment was drawn.
        """
        segment_colors: list[str] = []

        def segment_color(left_reading, right_reading) -> str:
            left = reading_colors.get(id(left_reading), color)
            right = reading_colors.get(id(right_reading), color)
            return left if left == right else _THRESHOLD_MIXED_SEGMENT_COLOR

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
                    if reading_colors:
                        segment_colors.append(segment_color(left_reading, right_reading))
            if fence_segments:
                collection = LineCollection(
                    fence_segments,
                    colors=segment_colors if reading_colors else color,
                    linewidths=1.5,
                    linestyles="--",
                    zorder=7,
                )
                ax.add_collection(collection)
            return bool(fence_segments)
        if len(measured_holes) < 2:
            return False
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
            if reading_colors:
                # One segment per hole pair, each in its readings' band colour
                # (the interpolation is linear, so the path is unchanged).
                for index in range(len(band) - 1):
                    fence_segments.append(
                        np.asarray(
                            [[xs_arr[index], ys_arr[index]], [xs_arr[index + 1], ys_arr[index + 1]]],
                            dtype=float,
                        )
                    )
                    segment_colors.append(segment_color(band[index][1], band[index + 1][1]))
            elif across_gaps and len(xs_arr) >= 2:
                x_dense = np.linspace(float(xs_arr.min()), float(xs_arr.max()), 100)
                y_dense = np.interp(x_dense, xs_arr, ys_arr)
                fence_segments.append(np.column_stack([x_dense, y_dense]))
            else:
                fence_segments.append(np.column_stack([xs_arr, ys_arr]))
        if fence_segments:
            collection = LineCollection(
                fence_segments,
                colors=segment_colors if reading_colors else color,
                linewidths=1.5,
                linestyles="--",
                zorder=7,
            )
            ax.add_collection(collection)
        return bool(fence_segments)

    @staticmethod
    def _draw_parameter_legend_sample(
        ax,
        x: float,
        y: float,
        entry: ParameterLegendEntry,
        *,
        font_size: float,
        transform=None,
        clip_path=None,
    ):
        """Draw an entry's sample value (e.g. red ``120``) at ``(x, y)``.

        Legend panels call this in the swatch column for parameter entries;
        returns the text artist, or ``None`` when the entry has no sample.
        """
        sample = entry.get("sample_text")
        if not sample:
            return None
        base_transform = transform if transform is not None else ax.transAxes
        color = entry.get("sample_color", CHEMISTRY_LABEL_BLACK)
        glyph_marker = entry.get("sample_marker")
        if glyph_marker:
            # The glyph the figure's values carry, in the room the sample
            # text keeps for it (it starts with _GLYPH_SPACER, so legend
            # layouts measuring the text make room for the glyph too).
            (glyph,) = ax.plot(
                [x],
                [y],
                marker=glyph_marker,
                markersize=_PARAMETER_GLYPH_SIZE,
                color=color,
                linestyle="none",
                transform=offset_copy(
                    base_transform, fig=ax.figure, x=_GLYPH_SPACER_CENTRE_EM * font_size, units="points"
                ),
                clip_on=True,
                gid="parameter-legend-glyph",
            )
            if clip_path is not None:
                glyph.set_clip_path(clip_path)
        artist = ax.text(
            x,
            y,
            sample,
            fontsize=font_size,
            va="center",
            ha="left",
            color=color,
            transform=base_transform,
            clip_on=True,
            gid="parameter-legend-sample",
        )
        if clip_path is not None:
            artist.set_clip_path(clip_path)
        return artist

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
