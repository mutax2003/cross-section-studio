"""Groundwater / water-table drawing mixin for CrossSectionRenderer."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TypedDict

import numpy as np
import pandas as pd
from matplotlib import _path as _mpl_path
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.markers import MarkerStyle
from matplotlib.patches import ArrowStyle, ConnectionStyle, FancyArrowPatch, Patch, Rectangle
from matplotlib.path import Path
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox, IdentityTransform, offset_copy

from hydro_metrics import (
    format_lnapl_thickness_label,
    has_lnapl,
    horizontal_gradients_along_profile,
    latest_water_by_hole,
    lnapl_correction_note,
    lnapl_interval_masl,
    series_legend_date,
    water_status,
)
from models import WaterLevel
from render_theme import (
    CONSULTING_NM_COLOR,
    CONSULTING_WATER_COLOR,
    LABEL_COLOR,
    LNAPL_EDGE_COLOR,
    LNAPL_FILL_COLOR,
    LNAPL_HATCH,
    LNAPL_LEGEND_LABEL,
    WATER_COLOR,
    consulting_gw_series_style,
)
from renderer_common import HatchBatchedPolyCollection, settle_figure_layout

logger = logging.getLogger(__name__)

_GW_MARKER_MAP = {
    "circle": "o",
    "triangle": "v",
    "diamond": "D",
    "plus": "P",
    "x": "x",
}


# Water numbers (RL / depth, DRY / NM notes) sit beside their own marker:
# left or right of the column, at the marker's height or stacked above /
# below it. Candidates are (side, extra dx, dy) in points from the column
# edge; a spot more than about one label height above/below the marker (or
# pushed sideways past half a label width) carries a leader back to it.
_WATER_COLUMN_GAP_PT = 2.0
_WATER_EXTRA_DX_PT = (0.0, 4.0, 10.0, 18.0, 28.0, 40.0, 54.0, 70.0)
# A water number keeps within this many label heights of its marker whenever
# any slot that close (either side, however far out sideways) is free: a
# short leader sideways reads better than a long one down the hole.
_WATER_NEAR_HEIGHTS = 2.0
# "Adjacent to its own well": at most this sideways step (points, one of
# _WATER_EXTRA_DX_PT) off the column edge and within this many label heights
# of the marker. Such spots (off every line) beat any further out, so a
# number never floats mid-span between two wells (GWM Fig 3, QA round 7).
_WATER_ADJACENT_EXTRA_PT = 10.0
_WATER_ADJACENT_HEIGHTS = _WATER_NEAR_HEIGHTS
_WATER_DY_STEP_PT = 2.0
_WATER_DY_MAX_PT = 96.0
# Preference costs (points of displacement): the left side is nearly as good
# as the right one; a leadered spot only wins when no adjacent spot is free.
_WATER_LEFT_COST_PT = 3.0
_WATER_LEADER_COST_PT = 30.0
_WATER_EXTRA_DX_COST = 0.7
# Two series' markers nudged apart on one hole: each number keeps to its own
# marker's side, even at the cost of a leader.
_WATER_WRONG_SIDE_COST_PT = 40.0
# Clearances (points): a label box keeps this far off groundwater / ground
# lines (half the 2 pt line plus a hair), markers and columns.
_WATER_LINE_CLEAR_PT = 1.6
_WATER_MARKER_CLEAR_PT = 1.0
_WATER_COLUMN_CLEAR_PT = 1.0
# Leader stops this far short of the marker (annotation shrinkB).
_WATER_LEADER_SHRINK_PT = 2.0
# Gradient (i=) labels: spots along their own segment (fraction of its
# length, midpoint first), offset perpendicular to it by the box half-extent
# plus a small extra; never on another line, never with a leader.
_GRADIENT_T = (0.5, 0.42, 0.58, 0.34, 0.66, 0.26, 0.74, 0.18, 0.82)
_GRADIENT_PERP_EXTRA_PT = (0.0, 2.0, 5.0, 9.0, 14.0, 20.0)
_GRADIENT_BELOW_COST = 1.5
# Water-number compaction ladder, one level for the whole figure (as the
# chemistry values shrink together): 0 full text; 1 unit suffix dropped
# ("737.00 m" -> "737.00"; the axis names the unit); 2 one decimal; 3 text
# at _WATER_COMPACT_FONT_SCALE (gradients shrink with the numbers). The
# level is raised only while a level number would otherwise be dropped or
# printed over a neighbouring column; a neighbouring column is the last
# resort before the drop (its own column is never covered).
_WATER_MAX_COMPACT_LEVEL = 3
_WATER_COMPACT_FONT_LEVEL = 3
_WATER_COMPACT_FONT_SCALE = 0.8
# Water label boxes: opaque white so lines and hatching never show through.
_WATER_LABEL_BBOX = {"boxstyle": "square,pad=0.12", "fc": "white", "ec": "none", "alpha": 1.0}
_WATER_LABEL_BBOX_PAD = 0.12
# Groundwater markers draw above the borehole columns (8) and the in-column
# lithology fills (9), with a thin white edge so they read on the grey
# column; water numbers (9.5) sit above them.
WATER_MARKER_ZORDER = 9.3
_WATER_LABEL_ZORDER = 9.5
_WATER_MARKER_EDGE_PT = 0.6
# Gap (points) kept between a groundwater marker and a chemistry dot, and
# between two chemistry dots, when one has to be nudged sideways.
_MARKER_SEPARATION_GAP_PT = 0.6
# Chemistry values sit only to the RIGHT of their own column: a value beside
# the wrong stick misattributes it. Candidates are (ddx, ddy) shifts in points
# from the label's base spot; the right side is searched nearest-first, with
# a fine vertical step so crowded readings stack down/up the column.
# Within _CHEM_NEAR_DY_PT / _CHEM_NEAR_DX_PT of the reading the nearest spot
# wins; beyond it
# sideways steps are cheap (0.25 per point vs 1 per point vertically): a
# hole's values must keep depth order, so crowded readings fan out into a
# second column beside their depth instead of cascading down the stick.
_CHEM_DX_STEPS = (0.0, 6.0, 12.0, 22.0, 34.0, 46.0, 58.0)
_CHEM_NEAR_DY_PT = 8.0
# Only small sideways nudges count as "near": a value pushed past a water
# label must not jump a long way right at its own depth (it read as the
# next hole's) while a short step down beside its column is free.
_CHEM_NEAR_DX_PT = 12.0
_CHEM_DY_STEP_PT = 3.0
_CHEM_DY_MAX_PT = 150.0
_CHEM_OFFSETS = sorted(
    (
        (ddx, sign * k * _CHEM_DY_STEP_PT)
        for ddx in _CHEM_DX_STEPS
        for k in range(int(_CHEM_DY_MAX_PT / _CHEM_DY_STEP_PT) + 1)
        for sign in ((1.0,) if k == 0 else (-1.0, 1.0))
    ),
    key=lambda item: (
        (0, abs(item[1]) + 1.5 * item[0])
        if abs(item[1]) <= _CHEM_NEAR_DY_PT and item[0] <= _CHEM_NEAR_DX_PT
        else (1, abs(item[1]) + 0.25 * item[0])
    ),
)
# Crowded-hole fallback (a hole whose values would otherwise be dropped, e.g.
# a long list on the last hole): sideways steps are nearly free so values
# zig-zag between two or more columns right of the stick (each only needs to
# sit below the previous value's centre), the vertical search reaches the
# whole frame, values sit one label pad apart (not two), and the text may
# shrink a little further. Level 1 keeps the normal sizes; level 2 starts one
# step smaller so every value fits.
_CHEM_COMPACT_DX_STEPS = (0.0, 6.0, 12.0, 22.0, 30.0, 38.0, 46.0, 58.0, 72.0, 88.0, 104.0, 120.0)
_CHEM_COMPACT_DY_MAX_PT = 720.0
_CHEM_COMPACT_DX_COST = 0.05
_CHEM_COMPACT_FONT_SCALES = {1: (1.0, 0.88, 0.76), 2: (0.88, 0.76, 0.68)}
_CHEM_MAX_COMPACT_LEVEL = 2


def _compact_chem_offsets() -> list[tuple[float, float]]:
    return sorted(
        (
            (ddx, sign * k * _CHEM_DY_STEP_PT)
            for ddx in _CHEM_COMPACT_DX_STEPS
            for k in range(int(_CHEM_COMPACT_DY_MAX_PT / _CHEM_DY_STEP_PT) + 1)
            for sign in ((1.0,) if k == 0 else (-1.0, 1.0))
        ),
        key=lambda item: (
            (0, abs(item[1]) + 1.5 * item[0])
            if abs(item[1]) <= _CHEM_NEAR_DY_PT and item[0] <= _CHEM_NEAR_DX_PT
            else (1, abs(item[1]) + _CHEM_COMPACT_DX_COST * item[0])
        ),
    )


_CHEM_COMPACT_OFFSETS = _compact_chem_offsets()
# A value whose reading shares its depth with the hole's previous value (the
# base spot collides with it) stacks directly below in depth order rather
# than sliding sideways along the other value's baseline.
_CHEM_STACK_OFFSETS = sorted(_CHEM_OFFSETS, key=lambda item: (abs(item[1]) + item[0], item[1] > 0))
_CHEM_STACK_OFFSETS_ARR = np.asarray(_CHEM_STACK_OFFSETS, dtype=float)
# Clearance (points) between a value and a groundwater line (half the 2 pt
# line width plus a hair).
_CHEM_WATER_LINE_CLEAR_PT = 1.5
_CHEM_OFFSETS_ARR = np.asarray(_CHEM_OFFSETS, dtype=float)
_CHEM_COMPACT_OFFSETS_ARR = np.asarray(_CHEM_COMPACT_OFFSETS, dtype=float)
# Clear gap (points) between a chemistry label (box/dot included) and the
# right edge of its own column.
_CHEM_COLUMN_GAP_PT = 2.0
# Re-fits of the x range so the last hole's values stay inside the frame.
_CHEM_FRAME_FIT_ATTEMPTS = 2
# "strip" style: background knock-out beside a labelled column. Above the
# lithology fills (2) and contact lines (3), below track fills, water lines /
# markers (5-8) and the values (9).
_CHEM_STRIP_ZORDER = 3.5
# Pad (points) past the value's right, top and bottom edges; the strip starts
# flush at the column's right edge (no pad on that side).
_CHEM_STRIP_PAD_PT = 2.5
# Clearance (points) between a value's top and the ground line above it: a
# surface reading's value sits just below the ground, never across it.
_CHEM_GROUND_CLEAR_PT = 2.0
# Last resorts before a value is dropped (the stick/marker stays): slightly
# smaller text. A value is never printed over a column or another label.
_CHEM_FONT_SCALES = (1.0, 0.88, 0.76)
_CHEM_MIN_FONT_SCALE = min(
    min(_CHEM_FONT_SCALES), *(min(scales) for scales in _CHEM_COMPACT_FONT_SCALES.values())
)
# A value moved more than one label height (and at least this many points)
# from its reading always gets a leader, even when leaders are otherwise off.
_CHEM_MIN_LEADER_SHIFT_PT = 6.0
# Placement order: chemistry values first. A value must sit beside its own
# column at its reading's depth (it has nowhere else to go), while a water
# number may sit either side of its groundwater marker; RL numbers before
# DRY / NM notes, gradients last.
_LABEL_PRIORITY = {"chem": 0, "rl": 1, "nm": 2, "gradient": 3}
# A value never drifts past this share of the gap to the next column (it
# would read as belonging to the next hole, or to none) unless the value
# itself needs more room than that.
_CHEM_MAX_GAP_DRIFT_SHARE = 0.5
# Re-runs that settle one font size per hole (no tiny stragglers).
_CHEM_FONT_SETTLE_PASSES = 3
# Most full placement passes one label resolution may run (compact
# escalation x font settling); bounds the time on pathological input.
_MAX_LABEL_PASSES = (_CHEM_MAX_COMPACT_LEVEL + 1) * (_CHEM_FONT_SETTLE_PASSES + 1)
# Clearance around each label box (points) so neighbours never touch.
_LABEL_PAD_PT = 2.0
# Sideways nudge (points) between coincident markers of different series on
# non-consulting sheets, so one series cannot hide another.
_SERIES_MARKER_STEP_PT = 7.5
# Line dash per series slot on non-consulting sheets (consulting keeps its
# solid client lines; marker shape separates series there).
_SERIES_DASHES = ("--", "-.", ":", (0, (6, 2, 1, 2)))
# Significant figures of an on-section i=Δh/Δx label (0.027, 0.033, 0.21).
_GRADIENT_SIG_FIGS = 2
# LNAPL product band: never thinner / narrower than this on the page (points)
# so a few-cm band at the marker is not hidden under the water marker
# (7 pt); grown about its centre only when the true size is smaller.
_LNAPL_MIN_HEIGHT_PT = 10.0
_LNAPL_MIN_WIDTH_PT = 26.0
# Outline width (points) of a groundwater marker drawn hollow over the LNAPL
# band, so the band reads through it.
_LNAPL_HOLLOW_EDGE_PT = 1.6
# Depth (mbgs) mode: water depths are joined (line, i=) only when the joined
# wells' collars agree within this many metres.
_DEPTH_MODE_COLLAR_TOL_M = 0.01
_DEPTH_MODE_WATER_NOTE = (
    "Depth mode (mbgs): groundwater depths are shown as markers only — the wells' collar "
    "elevations differ, so a line or gradient joining depths below ground would not "
    "represent the water table (use elevation mode for heads and gradients)"
)
# What the "i=" labels mean, shown only when one is drawn: a key row in the
# section-sheet series key (sample "i=" + the sentence), a numbered note in
# the consulting title block (a legend row there re-flowed the legend).
GRADIENT_LEGEND_LABEL = "i = horizontal hydraulic gradient Δh/Δx between adjacent wells (schematic)"
GRADIENT_KEY_SAMPLE = "i="
# Flowing-artesian head (above ground): dashed riser from the ground to the
# head plus an upward triangle on the ground line.
ARTESIAN_LEGEND_LABEL = "Flowing artesian head (above ground)"
_ARTESIAN_MARKER = "^"
_ARTESIAN_LINESTYLE = (0, (2.0, 1.5))
_ARTESIAN_TOL_M = 1e-6


def format_sig_figs(value: float, sig: int = _GRADIENT_SIG_FIGS) -> str:
    """``value`` to ``sig`` significant figures, never in exponent form."""
    value = float(value)
    if value == 0.0 or not np.isfinite(value):
        return "0" if value == 0.0 else str(value)
    rounded = round(value, sig - 1 - int(np.floor(np.log10(abs(value)))))
    # Rounding up may add a digit (0.0996 -> 0.10): keep sig figures.
    decimals = max(0, sig - 1 - int(np.floor(np.log10(abs(rounded)))))
    return f"{rounded:.{decimals}f}"


def format_gradient_text(gradient: float, series_prefix: str = "") -> str:
    """On-section ``i=0.027`` label (2 significant figures); several water
    series prefix the series name (``Fall 2025 i=0.033``)."""
    text = f"i={format_sig_figs(gradient)}"
    return f"{series_prefix} {text}" if series_prefix else text


def distinct_series_styles(
    proposed: Sequence[tuple[str, str, bool, bool]],
    candidates,
) -> list[tuple[str, str]]:
    """De-duplicate per-series ``(colour, marker)`` defaults.

    ``proposed`` rows are ``(colour, marker, colour_fixed, marker_fixed)`` in
    series order (``*_fixed``: set in the workbook — never changed). A series
    whose pair repeats an earlier one moves to the first candidate slot
    (``candidates(k) -> (colour, marker)``) with an unused pair, changing only
    the parts the workbook left blank. The first series never changes.
    """
    used: set[tuple[str, str]] = set()
    resolved: list[tuple[str, str]] = []
    for index, (color, marker, color_fixed, marker_fixed) in enumerate(proposed):
        pair = (str(color).lower(), str(marker))
        if pair in used and not (color_fixed and marker_fixed):
            for slot in range(index, index + 16):
                cand_color, cand_marker = candidates(slot)
                new_color = color if color_fixed else cand_color
                new_marker = marker if marker_fixed else cand_marker
                new_pair = (str(new_color).lower(), str(new_marker))
                if new_pair not in used:
                    color, marker, pair = new_color, new_marker, new_pair
                    break
        used.add(pair)
        resolved.append((color, marker))
    return resolved


class MinSizeBandCollection(HatchBatchedPolyCollection):
    """Rectangles in data units, grown about their centre at draw time to a
    minimum on-page size (points) so thin bands stay visible at any VE."""

    def __init__(self, boxes: np.ndarray, *, min_width_pt: float, min_height_pt: float, **kwargs) -> None:
        self._data_boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)  # x0, y0, x1, y1
        self._min_width_pt = float(min_width_pt)
        self._min_height_pt = float(min_height_pt)
        super().__init__(self._box_verts(self._data_boxes), **kwargs)

    @staticmethod
    def _box_verts(boxes: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = boxes.T
        return np.stack(
            [np.column_stack([x0, y0]), np.column_stack([x1, y0]), np.column_stack([x1, y1]), np.column_stack([x0, y1])],
            axis=1,
        )

    def draw(self, renderer) -> None:  # type: ignore[override]
        ax = self.axes
        if ax is not None and len(self._data_boxes):
            to_px = ax.transData
            lo = to_px.transform(self._data_boxes[:, :2])
            hi = to_px.transform(self._data_boxes[:, 2:])
            centre = 0.5 * (lo + hi)
            half = 0.5 * np.abs(hi - lo)
            min_half = 0.5 * renderer.points_to_pixels(np.asarray([self._min_width_pt, self._min_height_pt]))
            half = np.maximum(half, min_half)
            from_px = to_px.inverted()
            boxes = np.column_stack([from_px.transform(centre - half), from_px.transform(centre + half)])
            stale = self.stale
            self.set_verts(self._box_verts(boxes))
            self.stale = stale
        super().draw(renderer)


def _fmt_water_number(value: float, decimals: int = 2) -> str:
    """Water level / depth text: always 2 decimals (745.29, 745.30, 745.00).

    One fixed precision per figure: "744" beside "743.24" read as different
    survey accuracies. A crowded figure drops to one decimal for every
    number at once (``decimals=1``), never for a few of them.
    """
    decimals = max(0, int(decimals))
    text = f"{float(value):.{decimals}f}"
    zero = f"{0.0:.{decimals}f}"
    return zero if text == f"-{zero}" else text


def _water_compact_text(annotation, level: int) -> str:
    """Text of a water number at a compaction level (see the ladder above)."""
    value = getattr(annotation, "_water_value", None)
    if value is None:
        # A DRY / NM note (merged from both statuses after it was drawn):
        # its text never compacts, so it keeps what it says now.
        return annotation.get_text()
    suffix = str(getattr(annotation, "_water_suffix", "") or "")
    number = _fmt_water_number(value, 1 if level >= 2 else 2)
    text = number if level >= 1 or not suffix else f"{number} {suffix}"
    note = getattr(annotation, "_water_note", None)
    return f"{text}\n{note}" if note else text


def _apply_water_compaction(labels, level: int) -> bool:
    """Set every water number / gradient to a compaction level; True if any changed."""
    changed = False
    scale = _WATER_COMPACT_FONT_SCALE if level >= _WATER_COMPACT_FONT_LEVEL else 1.0
    for kind, annotation, _color in labels:
        if kind == "chem":
            continue
        base_size = getattr(annotation, "_water_base_fontsize", None)
        if base_size is None:
            base_size = float(annotation.get_fontsize())
            annotation._water_base_fontsize = base_size
        size = float(base_size) * scale
        if abs(float(annotation.get_fontsize()) - size) > 1e-9:
            annotation.set_fontsize(size)
            changed = True
        if kind in ("rl", "nm"):
            text = _water_compact_text(annotation, level)
            if text != annotation.get_text():
                annotation.set_text(text)
                changed = True
    return changed


class _ObstacleArray:
    """Axis-aligned boxes kept as an (N, 4) array for vectorised overlap sums."""

    def __init__(self, boxes) -> None:
        rows = [(b.x0, b.y0, b.x1, b.y1) for b in boxes]
        self._arr = np.array(rows, dtype=float).reshape(-1, 4)

    def add(self, box) -> None:
        self._arr = np.vstack([self._arr, [[box.x0, box.y0, box.x1, box.y1]]])

    def overlap(self, box) -> float:
        if self._arr.shape[0] == 0:
            return 0.0
        w = np.minimum(self._arr[:, 2], box.x1) - np.maximum(self._arr[:, 0], box.x0)
        h = np.minimum(self._arr[:, 3], box.y1) - np.maximum(self._arr[:, 1], box.y0)
        return float(np.sum(np.clip(w, 0.0, None) * np.clip(h, 0.0, None)))

    def raw(self, pad: float) -> np.ndarray:
        """Stored boxes shrunk by ``pad`` (the label extents without clearance)."""
        return self._arr + np.array([pad, pad, -pad, -pad])

    def overlaps_many(self, boxes: np.ndarray) -> np.ndarray:
        """Boolean per row of ``boxes`` (K, 4): does it overlap any stored box?"""
        if self._arr.shape[0] == 0 or boxes.shape[0] == 0:
            return np.zeros(boxes.shape[0], dtype=bool)
        hit = np.zeros(boxes.shape[0], dtype=bool)
        # Only obstacles reaching into the candidates' hull can overlap any
        # of them (the strict tests mirror ``w > 0`` / ``h > 0`` below), so
        # the far side of a long section is skipped without changing a hit.
        near = self._arr[
            (self._arr[:, 2] > boxes[:, 0].min())
            & (self._arr[:, 0] < boxes[:, 2].max())
            & (self._arr[:, 3] > boxes[:, 1].min())
            & (self._arr[:, 1] < boxes[:, 3].max())
        ]
        # Chunked so K candidates x N obstacles stays small in memory.
        for start in range(0, near.shape[0], 256):
            arr = near[start : start + 256]
            w = np.minimum(arr[None, :, 2], boxes[:, None, 2]) - np.maximum(arr[None, :, 0], boxes[:, None, 0])
            h = np.minimum(arr[None, :, 3], boxes[:, None, 3]) - np.maximum(arr[None, :, 1], boxes[:, None, 1])
            hit |= np.any((w > 0.0) & (h > 0.0), axis=1)
        return hit


def _register_water_line(ax, xs, ys) -> None:
    """Remember a drawn groundwater polyline (data coords) on its axes.

    Chemistry values treat these as obstacles: a water line drawn through a
    value strikes it out (lines sit above the label strips by design).
    """
    points = np.column_stack((np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)))
    if len(points) >= 2:
        ax.__dict__.setdefault("_css_water_lines", []).append(points)


def _water_line_segments(axes) -> np.ndarray:
    """(S, 4) display-space segments of every registered water polyline."""
    rows = []
    for ax in axes:
        for points in ax.__dict__.get("_css_water_lines", ()):
            display = ax.transData.transform(points)
            rows.append(np.column_stack((display[:-1], display[1:])))
    if not rows:
        return np.zeros((0, 4), dtype=float)
    return np.vstack(rows)


def _segment_box_hits(segs: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """(S, K) bool: does segment s (x0, y0, x1, y1) cross box k (x0, y0, x1, y1)?

    Liang-Barsky slab clipping, vectorised over every segment / box pair.
    """
    if segs.shape[0] == 0 or boxes.shape[0] == 0:
        return np.zeros((segs.shape[0], boxes.shape[0]), dtype=bool)
    t_lo = np.zeros((segs.shape[0], boxes.shape[0]))
    t_hi = np.ones_like(t_lo)
    miss = np.zeros_like(t_lo, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for start, end, lo_col, hi_col in ((0, 2, 0, 2), (1, 3, 1, 3)):
            p = segs[:, start][:, None]
            d = (segs[:, end] - segs[:, start])[:, None]
            lo = boxes[:, lo_col][None, :]
            hi = boxes[:, hi_col][None, :]
            flat = np.abs(d) < 1e-12
            miss |= flat & ((p < lo) | (p > hi))
            t1 = (lo - p) / np.where(flat, 1.0, d)
            t2 = (hi - p) / np.where(flat, 1.0, d)
            t_lo = np.where(flat, t_lo, np.maximum(t_lo, np.minimum(t1, t2)))
            t_hi = np.where(flat, t_hi, np.minimum(t_hi, np.maximum(t1, t2)))
    return ~miss & (t_lo <= t_hi)


# Slack (display px) for the hull pre-filters below: far beyond float noise in
# the exact tests, far below any distance that could change a hit.
_HULL_EPS = 1e-6


def _hull(arr: np.ndarray, margin=0.0) -> tuple[float, float, float, float]:
    """``(x0, y0, x1, y1)`` hull of (N, 4) boxes or segments, grown by ``margin``."""
    xs = arr[:, (0, 2)]
    ys = arr[:, (1, 3)]
    pad = margin + _HULL_EPS
    return (float(xs.min()) - pad, float(ys.min()) - pad, float(xs.max()) + pad, float(ys.max()) + pad)


def _rows_touching(arr: np.ndarray, hull) -> np.ndarray:
    """Rows of (N, 4) boxes / segments whose own bounding box meets ``hull``."""
    x0, y0, x1, y1 = hull
    lo_x = np.minimum(arr[:, 0], arr[:, 2])
    hi_x = np.maximum(arr[:, 0], arr[:, 2])
    lo_y = np.minimum(arr[:, 1], arr[:, 3])
    hi_y = np.maximum(arr[:, 1], arr[:, 3])
    return arr[(hi_x >= x0) & (lo_x <= x1) & (hi_y >= y0) & (lo_y <= y1)]


def _any_segment_hits(segs: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """``_segment_box_hits(segs, boxes).any(axis=0)``: (K,) per box.

    Segments clear of every box's hull cannot cross any of them, so only the
    nearby ones are tested (a long section's water / ground lines have
    hundreds of segments; the candidate spots of one label touch a few).
    """
    if segs.shape[0] == 0 or boxes.shape[0] == 0:
        return np.zeros(boxes.shape[0], dtype=bool)
    return _segment_box_hits(_rows_touching(segs, _hull(boxes)), boxes).any(axis=0)


def _segment_hits_any(segs: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """``_segment_box_hits(segs, boxes).any(axis=1)``: (S,) per segment."""
    if segs.shape[0] == 0 or boxes.shape[0] == 0:
        return np.zeros(segs.shape[0], dtype=bool)
    return _segment_box_hits(segs, _rows_touching(boxes, _hull(segs))).any(axis=1)


def _segments_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(K, L) bool: does segment a[k] properly cross segment b[l]?

    Touching (a shared end point, e.g. two leaders to one marker) is not a
    crossing; collinear overlaps are ignored.
    """
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=bool)
    p, r = a[:, None, :2], (a[:, 2:] - a[:, :2])[:, None, :]
    q, s = b[None, :, :2], (b[:, 2:] - b[:, :2])[None, :, :]
    denom = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]
    qp = q - p
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[..., 0] * s[..., 1] - qp[..., 1] * s[..., 0]) / denom
        u = (qp[..., 0] * r[..., 1] - qp[..., 1] * r[..., 0]) / denom
    eps = 1e-6
    return (np.abs(denom) > 1e-12) & (t > eps) & (t < 1 - eps) & (u > eps) & (u < 1 - eps)


def _ground_display_y(polylines, xs: np.ndarray) -> np.ndarray:
    """Display y of the ground surface above each display x (lowest if several)."""
    result = np.full(xs.shape, np.inf)
    for line in polylines:
        order = np.argsort(line[:, 0])
        result = np.minimum(result, np.interp(xs, line[order, 0], line[order, 1]))
    return result


def _register_ground_line(ax, xs, ys) -> None:
    """Remember the ground surface (data coords): water numbers keep off it."""
    points = np.column_stack((np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)))
    if len(points) >= 2:
        ax.__dict__.setdefault("_css_ground_lines", []).append(points)


def _ground_line_segments(axes) -> np.ndarray:
    rows = []
    for ax in axes:
        for points in ax.__dict__.get("_css_ground_lines", ()):
            display = ax.transData.transform(points)
            rows.append(np.column_stack((display[:-1], display[1:])))
    if not rows:
        return np.zeros((0, 4), dtype=float)
    return np.vstack(rows)


def register_marker_set(ax, collection, kind: str) -> None:
    """Remember a groundwater ("water") or chemistry ("chem") marker scatter.

    The label pass reads where they are drawn (water numbers keep off every
    marker) and nudges overlapping ones apart sideways; the original offsets
    are kept so each pass starts again from the drawn positions.
    """
    collection._css_base_offsets = np.asarray(collection.get_offsets(), dtype=float).copy()
    collection._css_marker_kind = kind
    ax.__dict__.setdefault("_css_marker_sets", []).append(collection)


def _marker_radius_px(collection, renderer) -> float:
    sizes = collection.get_sizes()
    area = float(sizes[0]) if len(sizes) else 36.0
    return 0.5 * float(np.sqrt(max(area, 0.0))) * renderer.points_to_pixels(1.0)


def _marker_points(collections, renderer) -> list[list]:
    """``[collection, index, display x, display y, radius px, dx]`` per point."""
    points = []
    for collection in collections:
        display = collection.get_offset_transform().transform(collection._css_base_offsets)
        radius = _marker_radius_px(collection, renderer)
        for index, (x, y) in enumerate(display):
            points.append([collection, index, float(x), float(y), radius, 0.0])
    return points


def _spread_clusters(points, gap: float, *, overlap_share: float, frame=None) -> None:
    """Spread markers of one hole that overlap left / right of the stick.

    Points closer than ``overlap_share`` of their summed radii (vertically,
    same x) form a cluster; its members sit side by side, centred on the
    stick, one marker width apart. A cluster at the frame edge spreads inwards
    only (a marker past the frame line is clipped).
    """
    by_hole: dict[float, list] = {}
    for point in points:
        by_hole.setdefault(round(point[2], 1), []).append(point)
    for hole_points in by_hole.values():
        hole_points.sort(key=lambda item: -item[3])
        cluster = [hole_points[0]]
        for point in hole_points[1:] + [None]:
            prev = cluster[-1]
            if point is not None and abs(prev[3] - point[3]) < overlap_share * (prev[4] + point[4]):
                cluster.append(point)
                continue
            if len(cluster) >= 2:
                step = 2.0 * max(item[4] for item in cluster) + gap
                # Top marker leftmost, side by side (two: half a step each way).
                for j, item in enumerate(cluster):
                    offset = (j - 0.5 * (len(cluster) - 1)) * step
                    item[5] += offset
                    item[2] += offset
                if frame is not None:
                    low = min(item[2] - item[4] - gap for item in cluster) - frame.x0
                    high = max(item[2] + item[4] + gap for item in cluster) - frame.x1
                    inward = -low if low < 0 else (-high if high > 0 else 0.0)
                    for item in cluster:
                        item[5] += inward
                        item[2] += inward
            if point is not None:
                cluster = [point]


def _separate_markers(axes, renderer) -> None:
    """Nudge chemistry dots and groundwater markers that overlap sideways.

    Two chemistry readings at (almost) the same depth of one hole spread a
    little left / right of the stick, as do two groundwater series with
    (almost) the same head; a groundwater marker landing on a chemistry dot
    steps to the left of it (values print right of the column), so no
    symbol hides another.
    """
    gap = renderer.points_to_pixels(_MARKER_SEPARATION_GAP_PT)
    for ax in axes:
        sets = ax.__dict__.get("_css_marker_sets", ())
        for collection in sets:
            collection.set_offsets(collection._css_base_offsets)
        chem = [c for c in sets if c._css_marker_kind == "chem" and len(c._css_base_offsets)]
        water = [c for c in sets if c._css_marker_kind == "water" and len(c._css_base_offsets)]
        dots = _marker_points(chem, renderer)
        heads = _marker_points(water, renderer)
        frame = ax.get_window_extent(renderer)
        _spread_clusters(dots, gap, overlap_share=1.0, frame=frame)
        # Heads of two series: only when one symbol mostly hides the other.
        _spread_clusters(heads, gap, overlap_share=0.6, frame=frame)
        if dots and heads:
            dot_arr = np.array([[d[2], d[3], d[4]] for d in dots], dtype=float)
            for head in heads:
                need = dot_arr[:, 2] + head[4] + gap
                hit = (np.abs(dot_arr[:, 1] - head[3]) < need) & (np.abs(dot_arr[:, 0] - head[2]) < need)
                if not hit.any():
                    continue
                target = float(np.min(dot_arr[hit, 0] - need[hit]))
                if target - head[4] < frame.x0:
                    # No room left of the dots inside the frame: step right.
                    target = float(np.max(dot_arr[hit, 0] + need[hit]))
                head[5] += target - head[2]
                head[2] = target
        for collection in chem + water:
            moved = [
                (point[1], point[5])
                for point in dots + heads
                if point[0] is collection and point[5] != 0.0
            ]
            if moved:
                _shift_offsets(collection, moved)


def _shift_offsets(collection, moved) -> None:
    """Shift some scatter points sideways by display pixels."""
    transform = collection.get_offset_transform()
    base = np.asarray(collection._css_base_offsets, dtype=float)
    display = transform.transform(base)
    current = np.asarray(collection.get_offsets(), dtype=float).copy()
    inverse = transform.inverted()
    for index, dx in moved:
        current[index] = inverse.transform([[display[index, 0] + dx, display[index, 1]]])[0]
    collection.set_offsets(current)


def _water_marker_points(axes, renderer) -> np.ndarray:
    """(M, 3) display x, y, radius of every drawn groundwater marker."""
    rows = []
    for ax in axes:
        for collection in ax.__dict__.get("_css_marker_sets", ()):
            if collection._css_marker_kind != "water" or not collection.get_visible():
                continue
            offsets = np.asarray(collection.get_offsets(), dtype=float)
            if not len(offsets):
                continue
            display = collection.get_offset_transform().transform(offsets)
            edge = renderer.points_to_pixels(_WATER_MARKER_EDGE_PT)
            radius = _marker_radius_px(collection, renderer) + edge
            rows.append(np.column_stack((display, np.full(len(display), radius))))
    if not rows:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(rows)


def _marker_boxes(axes, renderer, clear: float) -> list:
    """Display boxes (grown by ``clear``) of every drawn chemistry / groundwater marker."""
    boxes = []
    for ax in axes:
        for collection in ax.__dict__.get("_css_marker_sets", ()):
            if not collection.get_visible():
                continue
            offsets = np.asarray(collection.get_offsets(), dtype=float)
            if not len(offsets):
                continue
            display = collection.get_offset_transform().transform(offsets)
            radius = _marker_radius_px(collection, renderer) + clear
            if collection._css_marker_kind == "water":
                radius += renderer.points_to_pixels(_WATER_MARKER_EDGE_PT)
            boxes.extend(
                Bbox.from_extents(x - radius, y - radius, x + radius, y + radius)
                for x, y in display
                if np.isfinite(x) and np.isfinite(y)
            )
    return boxes


def _circles_hit_boxes(circles: np.ndarray, boxes: np.ndarray, clear: float) -> np.ndarray:
    """(K,) bool: does box k come within ``clear`` of any circle?"""
    if circles.shape[0] == 0 or boxes.shape[0] == 0:
        return np.zeros(boxes.shape[0], dtype=bool)
    # Circles farther than their reach from the boxes' hull hit none of them.
    reach = circles[:, 2] + clear + _HULL_EPS
    circles = circles[
        (circles[:, 0] + reach >= boxes[:, 0].min())
        & (circles[:, 0] - reach <= boxes[:, 2].max())
        & (circles[:, 1] + reach >= boxes[:, 1].min())
        & (circles[:, 1] - reach <= boxes[:, 3].max())
    ]
    if circles.shape[0] == 0:
        return np.zeros(boxes.shape[0], dtype=bool)
    cx, cy, r = circles[None, :, 0], circles[None, :, 1], circles[None, :, 2] + clear
    nx = np.clip(cx, boxes[:, None, 0], boxes[:, None, 2])
    ny = np.clip(cy, boxes[:, None, 1], boxes[:, None, 3])
    return np.any((nx - cx) ** 2 + (ny - cy) ** 2 < r**2, axis=1)


def _boxes_overlap_any(obstacles: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if obstacles.shape[0] == 0 or boxes.shape[0] == 0:
        return np.zeros(boxes.shape[0], dtype=bool)
    if obstacles.shape[0] > 8:
        obstacles = _rows_touching(obstacles, _hull(boxes))
        if obstacles.shape[0] == 0:
            return np.zeros(boxes.shape[0], dtype=bool)
    w = np.minimum(obstacles[None, :, 2], boxes[:, None, 2]) - np.maximum(obstacles[None, :, 0], boxes[:, None, 0])
    h = np.minimum(obstacles[None, :, 3], boxes[:, None, 3]) - np.maximum(obstacles[None, :, 1], boxes[:, None, 1])
    return np.any((w > 0.0) & (h > 0.0), axis=1)


def _set_label_visible(annotation, visible: bool) -> None:
    """Show/hide a label together with its halo twin and colour dot."""
    annotation.set_visible(visible)
    for companion in (getattr(annotation, "_halo", None), getattr(annotation, "_chem_dot", None)):
        if companion is not None:
            companion.set_visible(visible)


def _sync_label_companions(annotation, fig, dx: float, dy: float, ha: str) -> None:
    """Keep a label's halo twin and colour dot on the label after it moves."""
    halo = getattr(annotation, "_halo", None)
    if halo is not None:
        halo.xyann = (dx, dy)
        halo.set_horizontalalignment(ha)
    dot = getattr(annotation, "_chem_dot", None)
    if dot is not None:
        dot.set_transform(
            offset_copy(annotation.axes.transData, fig=fig, x=dx - 4.5, y=dy, units="points")
        )


# Display anchors of the labels during one placement pass (the plot box and
# limits are fixed while it runs): every rule asks for a label's anchor many
# times, and each transform through the axes' composite transform rebuilds
# its matrix. Keyed on the label and its anchor (water labels move theirs).
_ANCHOR_SCOPES: list[dict] = []


def _anchor_px(annotation) -> np.ndarray:
    """``annotation.axes.transData.transform([annotation.xy])[0]`` (memoised in a pass)."""
    if not _ANCHOR_SCOPES:
        return annotation.axes.transData.transform([annotation.xy])[0]
    xy = annotation.xy
    try:
        key = (id(annotation), id(annotation.axes), float(xy[0]), float(xy[1]))
    except (TypeError, ValueError):  # pragma: no cover - unit-bearing anchors
        return annotation.axes.transData.transform([xy])[0]
    cache = _ANCHOR_SCOPES[-1]
    anchor = cache.get(key)
    if anchor is None:
        anchor = cache[key] = annotation.axes.transData.transform([xy])[0]
        anchor.flags.writeable = False
    return anchor


def _chem_footprint(annotation, renderer):
    """Display box of a chemistry label: text plus its bbox patch and colour dot.

    Memoised per label on every input the measurement reads (text, font,
    alignment, offset, anchor in display space, dpi, visibility): placement
    re-runs (compact escalation, frame refits) re-measure the same spots many
    times, and each measurement lays the text out several times. A hit returns
    a copy of the exact box a fresh measurement would give.
    """
    anchor = _anchor_px(annotation)
    key = (
        annotation.get_text(),
        hash(annotation.get_fontproperties()),
        annotation.get_horizontalalignment(),
        annotation.get_verticalalignment(),
        annotation.get_rotation(),
        tuple(annotation.xyann),
        annotation.anncoords if isinstance(annotation.anncoords, str) else id(annotation.anncoords),
        float(anchor[0]),
        float(anchor[1]),
        float(annotation.figure.dpi),
        float(renderer.points_to_pixels(1.0)),
        annotation.get_visible(),
        # A strip value swapped to a box (too narrow for a strip) grows by it.
        annotation.get_bbox_patch() is not None,
    )
    cache = annotation.__dict__.setdefault("_chem_footprint_cache", {})
    cached = cache.get(key)
    if cached is not None:
        return cached.frozen()
    box = _measure_chem_footprint(annotation, renderer)
    if len(cache) > 512:
        cache.clear()
    cache[key] = box.frozen()
    return box


def _measure_chem_footprint(annotation, renderer):
    box = _text_box(annotation, renderer)
    patch = annotation.get_bbox_patch()
    if patch is not None:
        annotation.update_bbox_position_size(renderer)
        box = Bbox.union([box, patch.get_window_extent(renderer)])
    dot = getattr(annotation, "_chem_dot", None)
    if dot is not None:
        # The dot sits 4.5 pt left of the text anchor (see _sync_label_companions).
        anchor_x, anchor_y = _anchor_px(annotation)
        dx, dy = annotation.xyann
        cx = anchor_x + renderer.points_to_pixels(dx - 4.5)
        cy = anchor_y + renderer.points_to_pixels(dy)
        radius = renderer.points_to_pixels(dot.get_markersize() / 2.0 + 0.3)
        box = Bbox.union([box, Bbox([[cx - radius, cy - radius], [cx + radius, cy + radius]])])
    return box


def _own_column(annotation, column_boxes):
    """The column a chemistry label belongs to (anchored at its right edge)."""
    if not column_boxes:
        return None
    anchor_x = _anchor_px(annotation)[0]
    return min(column_boxes, key=lambda col: abs(col.x1 - anchor_x))


def _chem_right_limit(
    own,
    column_boxes,
    frame,
    *,
    consulting: bool,
    clearance: float,
    need: float,
    typical_gap: float = np.inf,
    memo=None,
) -> float:
    """Right-most display x a hole's value may reach.

    Normally the next column's left edge: a value past it reads as the
    neighbour's. The frame bounds the last hole on consulting sheets (their x
    range is fixed by the printed scale). Only a twin hole (its gap to the
    next column under half the section's ``typical_gap``, e.g. a re-drill a
    few decimetres away) whose value (``need`` pixels wide at the smallest
    font, plus ``clearance``) cannot fit before the twin may step past that
    one column into the gap the pair shares, with a leader. Never two
    columns, and never on an evenly spaced section.
    """
    if own is None:
        return np.inf
    # Inside the frame line, not on it (the spine would strike the text).
    edge = float(frame.x1) - 0.5 * clearance if consulting else np.inf
    right = memo.get(id(own)) if memo is not None else None
    if right is None:
        right = sorted(
            (
                col
                for col in column_boxes
                if col is not own and col.x1 > own.x1 and col.y1 > own.y0 and col.y0 < own.y1
            ),
            key=lambda col: col.x0,
        )[:2]
        if memo is not None:
            memo[id(own)] = right
    if not right:
        return edge
    first = right[0]
    own_gap = first.x0 - own.x1
    if own_gap >= need + clearance or own_gap >= 0.5 * typical_gap:
        # Near its own hole: never past the middle of the gap (on a long,
        # thin section a value slid sideways read as the next hole's), unless
        # the value alone (at full size) needs more than that.
        full_width = need / _CHEM_MIN_FONT_SCALE + 2.0 * clearance
        near = own.x1 + max(_CHEM_MAX_GAP_DRIFT_SHARE * own_gap, full_width)
        return float(min(first.x0, near))
    beyond = float(right[1].x0) if len(right) > 1 else edge
    if beyond - first.x1 >= need + clearance:
        return beyond
    return float(first.x0)


def _typical_column_gap(column_boxes) -> float:
    """Median clear gap between neighbouring columns (display px)."""
    ordered = sorted(column_boxes, key=lambda col: col.x0)
    gaps = [b.x0 - a.x1 for a, b in zip(ordered, ordered[1:], strict=False)]
    return float(np.median(gaps)) if gaps else np.inf


def _chem_min_width(annotation, renderer) -> float:
    """Display width of a value at the smallest font the placement may use."""
    if not hasattr(annotation, "_chem_base_fontsize"):
        annotation._chem_base_fontsize = float(annotation.get_fontsize())
    _set_chem_fontsize(annotation, annotation._chem_base_fontsize)
    annotation.set_horizontalalignment("left")
    annotation.xyann = annotation._water_base_xyann
    return _chem_footprint(annotation, renderer).width * _CHEM_MIN_FONT_SCALE


def _set_chem_fontsize(annotation, size: float) -> None:
    annotation.set_fontsize(size)
    halo = getattr(annotation, "_halo", None)
    if halo is not None:
        halo.set_fontsize(size)


_CHEM_UNIQUE_DY: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _chem_unique_dy(offsets: np.ndarray) -> np.ndarray:
    """Distinct dy (pt) of a candidate offset table (memoised per table)."""
    entry = _CHEM_UNIQUE_DY.get(id(offsets))
    if entry is None or entry[0] is not offsets:
        entry = _CHEM_UNIQUE_DY[id(offsets)] = (offsets, np.unique(offsets[:, 1]))
    return entry[1]


def _chem_any_height_fits(sy, box0, *, low, high, ceiling, top, ground_clear) -> bool:
    """Does any vertical shift ``sy`` (px) pass the placement's height rules?

    Mirrors the y terms of ``_place_chem_label``'s candidate filter with the
    same float operations, so a False here means every candidate fails.
    """
    y0 = box0.y0 + sy
    y1 = box0.y1 + sy
    ok = (y0 >= low) & (y1 <= high) & (0.5 * (y0 + y1) < ceiling)
    if top is not None:
        ok &= y1 + ground_clear <= top
    return bool(ok.any())


def _place_chem_label(
    annotation,
    renderer,
    fig,
    *,
    placed_arr,
    column_arr,
    own,
    pad,
    marker_arr=None,
    ceiling=np.inf,
    compact=0,
    line_segs=None,
    prev_box=None,
    right_limit=np.inf,
    font_cap=1.0,
):
    """Right-of-column placement with hard constraints.

    The label (box and dot included) must start right of its own column's
    right edge, overlap no borehole column and no placed label, and stay on
    the page and short of ``right_limit`` (the next column / consulting
    frame: a value never drifts across a neighbouring hole); leaving the axes
    sideways is otherwise only penalised. Its centre must also sit
    below ``ceiling`` (display y of the hole's previous, shallower label) so a
    hole's values keep their readings' depth order. ``compact`` (1 or 2)
    switches a crowded hole to the zig-zag / full-frame / smaller-text
    fallback. Returns the chosen (dx, dy), or None when nothing fits at any
    font size (the caller drops the label).

    The value also stays under the ground surface (the registered ground
    line, else the top of its own column): a surface reading's value sits
    just below the ground line rather than across it or the frame top.

    Soft constraints, kept whenever any free spot satisfies them: the value
    is not crossed by a groundwater line (``line_segs``, display segments),
    and its leader (when the shift earns one) does not run under another
    label. ``prev_box`` ((footprint, anchor y) of the hole's previous value)
    switches a value read at practically the same depth (anchors less than
    half a label height apart) to stacking directly below it. ``font_cap``
    (a font scale) starts the search at that size, so every value of a hole
    whose crowded value needed smaller text prints at one size.
    """
    base = annotation._water_base_xyann
    if not hasattr(annotation, "_chem_base_fontsize"):
        annotation._chem_base_fontsize = float(annotation.get_fontsize())
    frame = annotation.axes.get_window_extent(renderer)
    page = fig.bbox
    px_per_pt = renderer.points_to_pixels(1.0)
    min_x0 = own.x1 + _CHEM_COLUMN_GAP_PT * px_per_pt if own is not None else -np.inf
    annotation.set_horizontalalignment("left")
    if compact:
        scales = _CHEM_COMPACT_FONT_SCALES[min(compact, _CHEM_MAX_COMPACT_LEVEL)]
        offsets = _CHEM_COMPACT_OFFSETS_ARR
    else:
        scales = _CHEM_FONT_SCALES
        offsets = _CHEM_OFFSETS_ARR
    if font_cap < 1.0:
        scales = tuple(scale for scale in scales if scale <= font_cap + 1e-9) or (font_cap,)
    line_clear = _CHEM_WATER_LINE_CLEAR_PT * px_per_pt
    anchor_x, anchor_y = _chem_anchor(annotation)
    ground_lines = [
        annotation.axes.transData.transform(points)
        for points in annotation.axes.__dict__.get("_css_ground_lines", ())
    ]
    ground_clear = _CHEM_GROUND_CLEAR_PT * px_per_pt
    for scale in scales:
        _set_chem_fontsize(annotation, annotation._chem_base_fontsize * scale)
        annotation.xyann = base
        # With ha fixed the footprint only translates with the offset, so it
        # is measured once per font size and shifted per candidate (all
        # candidates at once).
        box0 = _chem_footprint(annotation, renderer)
        if (
            not compact
            and prev_box is not None
            and abs(prev_box[1] - anchor_y) < 0.5 * box0.height
            and box0.y0 < prev_box[0].y1 + pad
            and box0.y1 > prev_box[0].y0 - pad
        ):
            offsets = _CHEM_STACK_OFFSETS_ARR
        ddx_min = max(0.0, (min_x0 - box0.x0) / px_per_pt)
        if box0.x1 + ddx_min * px_per_pt > right_limit:
            # Narrow gap before the next column: start right at the column's
            # clearance instead of the usual spot a few points further out.
            ddx_min = (min_x0 - box0.x0) / px_per_pt
        # The vertical rules below only see each candidate's dy: test them
        # on the distinct dy values first (the same float ops per value), so
        # a value with no free height left (a crowded hole below its last
        # placed value) skips the full candidate grid.
        if not _chem_any_height_fits(
            _chem_unique_dy(offsets) * px_per_pt,
            box0,
            low=max(page.y0, frame.y0),
            high=min(page.y1, frame.y1),
            ceiling=ceiling,
            top=None if ground_lines or own is None else own.y1,
            ground_clear=ground_clear,
        ):
            continue
        ddx = offsets[:, 0] + ddx_min
        ddy = offsets[:, 1]
        sx = ddx * px_per_pt
        sy = ddy * px_per_pt
        boxes = np.column_stack((box0.x0 + sx, box0.y0 + sy, box0.x1 + sx, box0.y1 + sy))
        # On the page; above/below the plot it would land on headers, notes
        # or the title block (sideways past the frame is only penalised);
        # below the hole's previous value.
        ok = (
            (boxes[:, 0] >= page.x0)
            & (boxes[:, 2] <= min(page.x1, right_limit))
            & (boxes[:, 1] >= max(page.y0, frame.y0))
            & (boxes[:, 3] <= min(page.y1, frame.y1))
            & (0.5 * (boxes[:, 1] + boxes[:, 3]) < ceiling)
        )
        # Under the ground surface across the value's width (its own column
        # top when no ground line is registered): a surface reading's value
        # sits just below the ground line, never across it.
        if ground_lines:
            top_limit = np.minimum(
                _ground_display_y(ground_lines, boxes[:, 0]), _ground_display_y(ground_lines, boxes[:, 2])
            )
            ok &= boxes[:, 3] + ground_clear <= top_limit
        elif own is not None:
            ok &= boxes[:, 3] + ground_clear <= own.y1
        idx = np.flatnonzero(ok)
        if idx.size:
            idx = idx[~column_arr.overlaps_many(boxes[idx])]
        if idx.size and marker_arr is not None:
            # Never printed over a marker (another parameter's dot nudged
            # beside the stick, a groundwater symbol).
            idx = idx[~marker_arr.overlaps_many(boxes[idx])]
        if idx.size:
            # Placed boxes already carry the pad; a crowded hole's candidates
            # skip their own so values stack one pad apart instead of two.
            own_pad = 0.0 if compact else pad
            padded = boxes[idx] + np.array([-own_pad, -own_pad, own_pad, own_pad])
            idx = idx[~placed_arr.overlaps_many(padded)]
        if not idx.size:
            continue
        idx = _prefer_clear_spots(
            idx,
            boxes,
            ddx,
            ddy,
            line_segs=line_segs,
            line_clear=line_clear,
            # Clear of the padded boxes (a label's white box reaches past its
            # glyphs) when possible, else at least of the glyph extents.
            label_arrs=(placed_arr.raw(0.0), placed_arr.raw(pad)),
            anchor=(anchor_x, anchor_y),
            label_height_pt=box0.height / px_per_pt,
            px_per_pt=px_per_pt,
            # A crowded hole needs every spot near its readings; there a
            # leader may cross a neighbour rather than drop values.
            check_leaders=not compact,
        )
        cand = boxes[idx]
        inside_w = np.clip(np.minimum(cand[:, 2], frame.x1) - np.maximum(cand[:, 0], frame.x0), 0.0, None)
        inside_h = np.clip(np.minimum(cand[:, 3], frame.y1) - np.maximum(cand[:, 1], frame.y0), 0.0, None)
        outside = (cand[:, 2] - cand[:, 0]) * (cand[:, 3] - cand[:, 1]) - inside_w * inside_h
        best: tuple[float, int] | None = None
        for position, value in enumerate(outside.tolist()):
            # Tolerance: the overhang is the same for every dy at one dx, and
            # float noise must not pull a label far from its reading.
            if best is None or value < best[0] - 0.5:
                best = (value, position)
            if value <= 0.0:
                break
        assert best is not None
        chosen = int(idx[best[1]])
        return (base[0] + float(ddx[chosen]), base[1] + float(ddy[chosen]))
    _set_chem_fontsize(annotation, annotation._chem_base_fontsize)
    annotation.xyann = base
    return None


def _prefer_clear_spots(
    idx, boxes, ddx, ddy, *, line_segs, line_clear, label_arrs, anchor, label_height_pt, px_per_pt,
    check_leaders=True,
):
    """Narrow free candidates to those no water line crosses and whose leader
    clears other labels (each criterion only while some candidate meets it).

    A leader that would cross a groundwater line (the value parked on the
    far side of the line from its sample) is the worst case: such a spot is
    used only when every free spot has one. A spot near the sample whose box
    touches the line beats it (QA round 8: values leadered 6-12 m down,
    across the water line, while room beside the sample was free).
    """
    cand = boxes[idx]
    line_hit = np.zeros(idx.size, dtype=bool)
    if line_segs is not None and line_segs.shape[0]:
        grown = cand + np.array([-line_clear, -line_clear, line_clear, line_clear])
        line_hit = _any_segment_hits(line_segs, grown)
    leader_hits = [np.zeros(idx.size, dtype=bool) for _arr in label_arrs]
    shift = np.maximum(np.maximum(ddx[idx], 0.0), np.abs(ddy[idx]))
    leadered = shift > max(label_height_pt, _CHEM_MIN_LEADER_SHIFT_PT)
    crosses_line = np.zeros(idx.size, dtype=bool)
    segs = None
    if leadered.any():
        ax0, ay0 = anchor
        ends = np.column_stack((cand[:, 0], 0.5 * (cand[:, 1] + cand[:, 3])))
        vec = ends - np.array([ax0, ay0])
        length = np.hypot(vec[:, 0], vec[:, 1])
        # The drawn leader stops 2 pt short of its reading (shrinkB).
        trim = np.clip(2.0 * px_per_pt / np.maximum(length, 1e-9), 0.0, 1.0)[:, None]
        starts = np.array([ax0, ay0]) + vec * trim
        segs = np.column_stack((starts, ends))[leadered]
        if line_segs is not None and line_segs.shape[0]:
            near_lines = _rows_touching(line_segs, _hull(segs))
            if near_lines.shape[0]:
                crosses_line[np.flatnonzero(leadered)] = _segments_cross(segs, near_lines).any(axis=1)
    if check_leaders and segs is not None and label_arrs[0].shape[0]:
        # Labels outside the leaders' hull cannot be crossed by any of them.
        lo_x = np.minimum(segs[:, 0], segs[:, 2]).min()
        hi_x = np.maximum(segs[:, 0], segs[:, 2]).max()
        lo_y = np.minimum(segs[:, 1], segs[:, 3]).min()
        hi_y = np.maximum(segs[:, 1], segs[:, 3]).max()
        for leader_hit, label_arr in zip(leader_hits, label_arrs):
            near = label_arr[
                (label_arr[:, 2] >= lo_x)
                & (label_arr[:, 0] <= hi_x)
                & (label_arr[:, 3] >= lo_y)
                & (label_arr[:, 1] <= hi_y)
            ]
            leader_hit[np.flatnonzero(leadered)] = _segment_hits_any(segs, near)
    clear = ~crosses_line
    tiers = (
        [clear & ~line_hit & ~leader_hit for leader_hit in leader_hits]
        + [clear & ~line_hit, clear]
        + [~line_hit & ~leader_hit for leader_hit in leader_hits]
        + [~line_hit]
    )
    for keep in tiers:
        if keep.any():
            return idx[keep]
    return idx


_CHEM_DROP_NOTE_MAX_HOLES = 8


def _chem_drop_notes(labels) -> list[str]:
    """User-facing note listing chemistry values the placement had to drop.

    A value is only dropped when no spot beside its own column (right of it,
    short of the next column, inside the frame) fits at any allowed font
    size; its marker stays on the section.
    """
    by_hole: dict[str, list[str]] = {}
    for kind, annotation, _color in labels:
        if kind == "chem" and getattr(annotation, "_water_dropped", False):
            hole = str(getattr(annotation, "_chem_hole_id", "") or "?")
            by_hole.setdefault(hole, []).append(annotation.get_text())
    if not by_hole:
        return []
    count = sum(len(values) for values in by_hole.values())
    shown = list(by_hole.items())[:_CHEM_DROP_NOTE_MAX_HOLES]
    detail = "; ".join(f"{hole}: {', '.join(values)}" for hole, values in shown)
    if len(by_hole) > len(shown):
        detail += f"; +{len(by_hole) - len(shown)} more hole(s)"
    return [
        f"Chemistry labels: {count} value(s) not printed - no room beside their borehole "
        f"at this scale ({detail}). Markers are still drawn; a wider page or fewer "
        "boreholes per section shows them."
    ]


def _water_drop_notes(labels) -> list[str]:
    """Section notes for groundwater labels the placement left off.

    Gradient (i=) labels go first when a band is crowded; level numbers are
    only dropped when no spot beside their hole is free at all.
    """
    gradients = sum(
        1 for kind, annotation, _c in labels if kind == "gradient" and getattr(annotation, "_water_dropped", False)
    )
    levels = sum(
        1 for kind, annotation, _c in labels if kind in ("rl", "nm") and getattr(annotation, "_water_dropped", False)
    )
    notes = []
    if gradients:
        notes.append(f"{gradients} gradient label{'s' if gradients != 1 else ''} omitted — crowded.")
    if levels:
        notes.append(
            f"{levels} groundwater level label{'s' if levels != 1 else ''} omitted — crowded "
            "(markers are still drawn)."
        )
    return notes


def _strip_rect(own, boxes, pad: float) -> tuple[float, float, float, float]:
    """Knock-out from the column edge over a run of adjacent values."""
    return (
        own.x1,
        min(box.y0 for box in boxes) - pad,
        max(box.x1 for box in boxes) + pad,
        max(box.y1 for box in boxes) + pad,
    )


def _chem_mixed_size_holes(labels) -> dict[float, float]:
    """Hole key -> font scale cap when the printed values differ in size.

    One reduced size for the whole figure: when any value had to shrink,
    every hole is capped at that smallest scale (a few tiny values among
    normal ones read as a different kind of label).
    """
    holes: set[float] = set()
    scales: list[float] = []
    for kind, annotation, _color in labels:
        if kind != "chem" or not annotation.get_visible():
            continue
        base = getattr(annotation, "_chem_base_fontsize", None)
        if not base:
            continue
        scales.append(float(annotation.get_fontsize()) / float(base))
        holes.add(_chem_hole_key(annotation))
    if not scales or max(scales) - min(scales) <= 1e-6:
        return {}
    return dict.fromkeys(holes, min(scales))


def _chem_anchor(annotation) -> tuple[float, float]:
    x, y = _anchor_px(annotation)
    return float(x), float(y)


def _chem_hole_key(annotation) -> float:
    """Chemistry labels of one hole share the anchor x (its column's right edge)."""
    return round(_chem_anchor(annotation)[0], 1)


def _label_order_key(item) -> tuple:
    kind, annotation, _color = item
    priority = _LABEL_PRIORITY.get(kind, 9)
    if kind == "chem":
        return (priority, 0.0, 0.0)
    xy = getattr(annotation, "_water_base_xy", annotation.xy)
    x, y = annotation.axes.transData.transform([xy])[0]
    return (priority, round(float(x), 1), -float(y))


def _side_candidates(anchor, w: float, h: float, ppt: float, column_boxes, marker_extent=None):
    """Spots beside a marker's column: ``(centres (K, 2), costs (K,), leadered (K,))``.

    Right or left of the column, level with the marker or stacked above /
    below it in a tidy column; more than one label height off the marker
    (or pushed out past half a label width) the spot needs a leader, drawn
    from the box side facing the column. Also returns each spot's side
    (+1 right, -1 left).
    """
    ax_px, ay_px = float(anchor[0]), float(anchor[1])
    own = None
    for box in column_boxes:
        if box.x0 - 0.5 <= ax_px <= box.x1 + 0.5 and box.y0 - 0.5 <= ay_px <= box.y1 + 0.5:
            own = box
            break
    gap = _WATER_COLUMN_GAP_PT * ppt
    right_edge = own.x1 if own is not None else ax_px + 3.5 * ppt
    left_edge = own.x0 if own is not None else ax_px - 3.5 * ppt
    if marker_extent is not None:
        # Markers nudged past the column edge: numbers start beyond them, so
        # a stacked number's leader never runs back through its neighbour.
        left_edge = min(left_edge, marker_extent[0])
        right_edge = max(right_edge, marker_extent[1])
    right_edge += gap
    left_edge -= gap
    steps = int(_WATER_DY_MAX_PT / _WATER_DY_STEP_PT)
    dys = np.arange(-steps, steps + 1, dtype=float) * _WATER_DY_STEP_PT
    h_pt, w_pt = h / ppt, w / ppt
    # Plus the two spots that just clear the water line through the marker
    # (box edge one clearance off it): on a shallow section the grid steps
    # straddle it, landing either on the line or past the ground / frame.
    snug = 0.5 * h_pt + _WATER_LINE_CLEAR_PT + 0.25
    dys = np.sort(np.concatenate((dys, [-snug, snug])))
    centres, costs, leadered, sides, extras = [], [], [], [], []
    for side, edge, side_cost in ((1.0, right_edge, 0.0), (-1.0, left_edge, _WATER_LEFT_COST_PT)):
        for extra in _WATER_EXTRA_DX_PT:
            cx = edge + side * (extra * ppt + w / 2)
            lead = (np.abs(dys) > h_pt) | (extra > 0.5 * w_pt)
            centres.append(np.column_stack((np.full(dys.size, cx), ay_px + dys * ppt)))
            costs.append(np.abs(dys) + side_cost + _WATER_EXTRA_DX_COST * extra + _WATER_LEADER_COST_PT * lead)
            leadered.append(lead)
            sides.append(np.full(dys.size, side))
            extras.append(np.full(dys.size, extra))
    return (
        np.vstack(centres),
        np.concatenate(costs),
        np.concatenate(leadered),
        np.concatenate(sides),
        np.concatenate(extras),
    )


def _chem_depth_ordered(labels: list) -> list:
    """Reorder each hole's chemistry labels top-down (shallowest first on the page).

    The slots a hole's labels occupy in the placement order are kept, so the
    interleaving with other holes and label kinds is unchanged.
    """
    slots: dict[float, list[int]] = {}
    for index, (kind, annotation, _color) in enumerate(labels):
        if kind == "chem":
            slots.setdefault(_chem_hole_key(annotation), []).append(index)
    result = list(labels)
    for indices in slots.values():
        by_depth = sorted((labels[i] for i in indices), key=lambda item: -_chem_anchor(item[1])[1])
        for slot, item in zip(indices, by_depth):
            result[slot] = item
    return result



def _overlap_area(a, b) -> float:
    width = min(a.x1, b.x1) - max(a.x0, b.x0)
    height = min(a.y1, b.y1) - max(a.y0, b.y0)
    return width * height if width > 0 and height > 0 else 0.0


class _BoxStack:
    """Growable (N, 4) array of bbox extents for vectorised overlap sums.

    ``overlap_sum`` returns exactly ``sum(_overlap_area(box, b) for b in boxes)``:
    the per-pair areas use the same float ops, and only the non-zero terms are
    added in the original order, so layout decisions are bit-for-bit unchanged.
    """

    __slots__ = ("_data", "_size")

    def __init__(self, boxes=()) -> None:
        self._data = np.empty((max(16, len(boxes)), 4), dtype=float)
        self._size = 0
        for box in boxes:
            self.append(box)

    def append(self, box) -> None:
        if self._size == len(self._data):
            self._data = np.concatenate([self._data, np.empty_like(self._data)])
        self._data[self._size] = (box.x0, box.y0, box.x1, box.y1)
        self._size += 1

    def overlap_sum(self, box, start: int = 0) -> float:
        data = self._data[start : self._size]
        if not len(data):
            return 0
        x0, y0, x1, y1 = box.x0, box.y0, box.x1, box.y1
        width = np.minimum(x1, data[:, 2]) - np.maximum(x0, data[:, 0])
        height = np.minimum(y1, data[:, 3]) - np.maximum(y0, data[:, 1])
        hit = (width > 0) & (height > 0)
        if not hit.any():
            return 0
        return sum((width[hit] * height[hit]).tolist())


def _outside_area(box, frame) -> float:
    inside_w = max(0.0, min(box.x1, frame.x1) - max(box.x0, frame.x0))
    inside_h = max(0.0, min(box.y1, frame.y1) - max(box.y0, frame.y0))
    return box.width * box.height - inside_w * inside_h


def _text_box(annotation, renderer):
    # Text-only extent: Annotation.get_window_extent includes a visible leader,
    # which would make a moved label block the whole strip back to its point.
    # Only the text transform is refreshed (the first step of
    # Annotation.update_positions); positioning the leader as well would lay
    # the text out a second time, and draw() re-positions it anyway.
    annotation.set_transform(annotation._get_xy_transform(renderer, annotation.anncoords))
    return _text_window_extent(annotation, renderer)


# rcParams the text layout reads besides the text's own properties: generic
# family names resolve through them, and Agg metrics depend on the hinting.
_LAYOUT_RC_KEYS = (
    "font.family",
    "font.serif",
    "font.sans-serif",
    "font.monospace",
    "font.cursive",
    "font.fantasy",
    "mathtext.fontset",
    "text.hinting",
    "text.hinting_factor",
    "text.kerning_factor",
)
_LAYOUT_CACHE_MAX = 8192


def _text_layout_key(text, renderer, fig) -> tuple:
    """Every input ``Text._get_layout`` reads (the box it returns is relative
    to the text's position, so it does not depend on where the text sits)."""
    from matplotlib import rcParams

    props = text.get_fontproperties()
    return (
        text.get_text(),
        tuple(props.get_family()),
        props.get_slant(),
        props.get_variant(),
        props.get_weight(),
        props.get_stretch(),
        props.get_size(),
        props.get_file(),
        props.get_math_fontfamily(),
        text.get_usetex(),
        text.get_parse_math(),
        text._linespacing,
        text.get_rotation(),
        text.get_rotation_mode(),
        text.get_horizontalalignment(),
        text.get_verticalalignment(),
        float(fig.dpi),
        float(getattr(renderer, "dpi", 0.0) or 0.0),
        tuple(
            tuple(value) if isinstance(value, list) else value
            # dict lookup: the stored (validated) values, without the
            # deprecation / backend checks of ``rcParams[name]``.
            for value in map(rcParams._get, _LAYOUT_RC_KEYS)
        ),
    )


def _memo_text_layout(text, renderer, fig, compute=None):
    """``text._get_layout(renderer)`` memoised per renderer on every input it
    reads (None when the inputs cannot be keyed)."""
    try:
        key = _text_layout_key(text, renderer, fig)
        hash(key)
        cache = renderer.__dict__.setdefault("_css_text_layouts", {})
    except (TypeError, AttributeError):  # pragma: no cover - unusual fonts / renderers
        return None
    layout = cache.get(key)
    if layout is None:
        text._renderer = renderer
        layout = (compute or text._get_layout)(renderer)
        if len(cache) >= _LAYOUT_CACHE_MAX:
            cache.clear()
        cache[key] = layout
    return layout


class _MemoLayoutAnnotation(Annotation):
    """Value / water label whose text layout is memoised (see ``_text_window_extent``).

    Draws, tight-bbox passes and leader updates lay the same label text out
    again at every call; the layout depends only on the keyed inputs.
    """

    def _get_layout(self, renderer):
        fig = self.get_figure(root=True)
        if renderer is None or fig is None or self.get_wrap():
            return super()._get_layout(renderer)
        layout = _memo_text_layout(self, renderer, fig, compute=super()._get_layout)
        return super()._get_layout(renderer) if layout is None else layout


def _text_window_extent(text, renderer):
    """``Text.get_window_extent(text, renderer)``, bit for bit, with the text
    layout memoised per renderer.

    The layout (the costly part: font lookup, metrics, line boxes) depends
    only on the string, font, alignment, rotation and dpi; the extent is that
    box translated to the text's display position, exactly as matplotlib
    computes it. Label placement measures the same strings at the same sizes
    thousands of times while only their positions change.
    """
    fig = text.get_figure(root=True)
    if (
        renderer is None
        or fig is None
        or not text.get_visible()
        or text.get_text() == ""
        or text.get_wrap()
    ):
        return Text.get_window_extent(text, renderer)
    layout = _memo_text_layout(text, renderer, fig)
    if layout is None:
        return Text.get_window_extent(text, renderer)
    layout = layout[0]
    text._renderer = renderer
    x, y = text.get_unitless_position()
    x, y = text.get_transform().transform((x, y))
    return layout.translated(x, y)


# Hole-ID header candidates as (dx_pt, tier, ha): first shift alignment away
# from the neighbour, then stagger outward one or two text heights; negative
# tiers step inward for headers already at the page edge (consulting top row).
_HEADER_CANDIDATES = tuple(
    (dx, tier, "center" if dx == 0.0 else ("left" if dx > 0 else "right"))
    for tier in (0, 1, -1, 2)
    # Small nudges first; the wider steps let a corner header slide clear of a
    # transect end label ("A / NORTHWEST") that spans two lines.
    for dx in (0.0, 2.0, -2.0, 14.0, -14.0, 28.0, -28.0)
)


def resolve_header_collisions(fig, headers) -> None:
    """Keep hole-ID column headers from fusing when holes are close together.

    Headers sit outside the plot, so they are never leadered: each stays above
    (or below) its own column, only shifting alignment or stepping one tier out.
    """
    headers = [text for text in headers if text.get_visible() and text.get_text().strip()]
    if len(headers) < 2:
        return
    renderer = _figure_renderer(fig)
    pad = renderer.points_to_pixels(2.0)
    for text in headers:
        # Re-runs (export resizes the page) must start from the drawn position,
        # not stack another offset on the previous pass's choice.
        if not hasattr(text, "_header_base"):
            text._header_base = (
                text.get_transform(),
                text.get_horizontalalignment(),
                text.get_rotation(),
            )
        base_transform, base_ha, base_rotation = text._header_base
        text.set_transform(base_transform)
        text.set_horizontalalignment(base_ha)
        text.set_rotation(base_rotation)
    page = fig.bbox
    obstacles = _header_obstacles(
        fig,
        renderer,
        0.0,
        exclude={id(text) for text in headers},
        header_bands=_header_bands(fig, renderer, headers, pad),
    )
    _stagger_headers(fig, renderer, headers, obstacles, page, pad)
    staggered_cost = _header_layout_cost(renderer, headers, obstacles, page, pad)
    if staggered_cost == 0.0:
        return
    # Dense well fields: no horizontal stagger fits, so try vertical headers
    # (CAD convention) and keep whichever layout overlaps less.
    staggered = [(t.get_transform(), t.get_horizontalalignment()) for t in headers]
    for text in headers:
        text.set_transform(text._header_base[0])
        text.set_horizontalalignment("center")
        text.set_rotation(90)
    if _header_layout_cost(renderer, headers, obstacles, page, pad) >= staggered_cost:
        for text, (transform, ha) in zip(headers, staggered):
            text.set_rotation(text._header_base[2])
            text.set_transform(transform)
            text.set_horizontalalignment(ha)


def _header_layout_cost(renderer, headers, obstacles, page, pad) -> float:
    boxes = [text.get_window_extent(renderer).padded(pad) for text in headers]
    header_stack = _BoxStack(boxes)
    obstacle_stack = _BoxStack(obstacles)
    cost = 0.0
    for index, box in enumerate(boxes):
        cost += header_stack.overlap_sum(box, start=index + 1)
        cost += obstacle_stack.overlap_sum(box)
        cost += 4.0 * _outside_area(box, page)
    return cost


def _candidate_extent_fn(text, renderer):
    """``(transform, ha) -> Text.get_window_extent`` for one header's candidates.

    Only the transform and horizontal alignment differ between candidates, so
    the (alignment-dependent) text layout is computed once per alignment and
    translated exactly as ``Text.get_window_extent`` does, instead of
    re-measuring the string for all 28 candidate spots.
    """
    if text.get_wrap() or text.get_transform_rotates_text() or not text.get_text():
        def measure(transform, ha):
            text.set_transform(transform)
            text.set_horizontalalignment(ha)
            return text.get_window_extent(renderer)

        return measure
    position = text.get_unitless_position()
    layouts: dict[str, Bbox] = {}

    def extent(transform, ha):
        layout = layouts.get(ha)
        if layout is None:
            text.set_horizontalalignment(ha)
            layout = layouts[ha] = text._get_layout(renderer)[0]
        x, y = transform.transform(position)
        return layout.translated(x, y)

    return extent


def _stagger_headers(fig, renderer, headers, obstacles, page, pad) -> None:
    """Greedy pass: nudge alignment, then step headers outward by text-height tiers."""
    placed = _BoxStack(obstacles)
    for text in sorted(headers, key=lambda item: item.get_window_extent(renderer).x0):
        base_transform = text._header_base[0]
        height_pt = text.get_window_extent(renderer).height * 72.0 / fig.dpi + 1.0
        outward = 1.0 if text.get_verticalalignment() == "bottom" else -1.0
        best: tuple[float, float, int, str] | None = None
        candidate_box = _candidate_extent_fn(text, renderer)
        for index, (dx, tier, ha) in enumerate(_HEADER_CANDIDATES):
            transform = offset_copy(
                base_transform, fig=fig, x=dx, y=outward * tier * height_pt, units="points"
            )
            box = candidate_box(transform, ha).padded(pad)
            collision = placed.overlap_sum(box)
            outside = _outside_area(box, page)
            score = (collision + 4.0 * outside) * 1000.0 + index
            if best is None or score < best[0]:
                best = (score, dx, tier, ha)
            if collision == 0.0 and outside == 0.0:
                break
        assert best is not None
        _score, dx, tier, ha = best
        text.set_transform(
            offset_copy(base_transform, fig=fig, x=dx, y=outward * tier * height_pt, units="points")
        )
        text.set_horizontalalignment(ha)
        placed.append(text.get_window_extent(renderer).padded(pad))


def _drawn_tick_labels(axis, limits) -> list:
    """Major tick labels inside the view limits (matplotlib keeps but never
    draws labels for ticks past the axis ends)."""
    low, high = sorted(limits)
    span = (high - low) * 1e-9
    locs = axis.get_majorticklocs()
    labels = []
    for tick, loc in zip(axis.get_major_ticks(len(locs)), locs):
        if low - span <= loc <= high + span:
            labels.extend((tick.label1, tick.label2))
    return labels


def _header_bands(fig, renderer, headers, pad) -> list[tuple[float, float]]:
    """Display-y ranges every candidate header box (any tier / alignment, or the
    vertical alternative) stays within, padded like the boxes themselves."""
    bands = []
    for text in headers:
        box = text.get_window_extent(renderer)
        # Tiers step up to two (text height + 1 pt) either way; the vertical
        # alternative spans at most the text's width + height from its anchor.
        reach = 2.0 * (box.height + fig.dpi / 72.0) + box.width + box.height + pad + 1.0
        bands.append((box.y0 - reach, box.y1 + reach))
    return bands


def _straight_leader_bound(annotation, renderer):
    """Cheap superset of ``annotation.get_window_extent`` for a straight "-" leader.

    The leader path lies on the segment between its two end points (clipping
    and shrinking only trim it), so text box + segment box contains the exact
    extent without the costly patch-clipped path.  None when that cannot be
    guaranteed (other arrow / connection styles, transforms, clipped xy).
    """
    arrow = annotation.arrow_patch
    if arrow is None or not annotation._check_xy(renderer):
        return None
    style = arrow.get_arrowstyle()
    connection = arrow.get_connectionstyle()
    if not (
        type(style) is ArrowStyle.Curve
        and isinstance(connection, ConnectionStyle.Arc3)
        and connection.rad == 0.0
        and arrow.get_transform().is_affine
        and np.array_equal(arrow.get_transform().get_matrix(), np.eye(3))
        and arrow._posA_posB is not None
    ):
        return None
    annotation.update_positions(renderer)
    (xa, ya), (xb, yb) = arrow._posA_posB
    segment = Bbox([[min(xa, xb), min(ya, yb)], [max(xa, xb), max(ya, yb)]])
    return Bbox.union([_text_window_extent(annotation, renderer), segment])


def _annotation_extent_key(annotation, renderer) -> tuple:
    """Everything an annotation's drawn extent depends on (text layout, its
    anchor in display space, offset, alignment, font, leader, dpi)."""
    arrow = annotation.arrow_patch
    return (
        annotation.get_text(),
        tuple(np.asarray(annotation.xyann, dtype=float).ravel()),
        annotation.get_horizontalalignment(),
        annotation.get_verticalalignment(),
        annotation.get_rotation(),
        hash(annotation.get_fontproperties()),
        float(renderer.dpi),
        annotation.get_visible(),
        None if arrow is None else (arrow.get_visible(), getattr(annotation, "_arrow_relpos", None)),
        _bbox_patch_key(annotation.get_bbox_patch()),
    )


def _bbox_patch_key(patch):
    if patch is None:
        return None
    style = patch.get_boxstyle()
    return (type(style).__name__, getattr(style, "pad", None), patch.get_mutation_scale())


def _cached_annotation_extent(annotation, renderer, kind: str, measure):
    """Memoised ``measure(annotation, renderer)`` relative to the label's anchor.

    Header passes re-measure the same value labels (and their leader
    paths: a clipped-path extent per label) after every layout step, where
    only the plot moved. The extent of a label in offset points is a pure
    translation of its anchor, so it is measured once per text / offset /
    font / dpi and moved with the anchor.
    """
    try:
        key = (kind, _annotation_extent_key(annotation, renderer))
        anchor = np.asarray(annotation._get_position_xy(renderer), dtype=float)
    except Exception:  # pragma: no cover - unusual coordinate systems
        return measure(annotation, renderer)
    if not np.all(np.isfinite(anchor)):
        return measure(annotation, renderer)
    cache = annotation.__dict__.setdefault("_css_extent_cache", {})
    if key in cache:
        rel = cache[key]
        return None if rel is None else Bbox(rel + anchor)
    value = measure(annotation, renderer)
    if len(cache) > 8:
        cache.clear()
    cache[key] = None if value is None else np.asarray(value.get_points(), dtype=float) - anchor
    return value


def _header_obstacles(fig, renderer, pad, *, exclude: set[int], header_bands=None) -> list:
    """Tick labels, axis labels, titles and figure text a header must not cover.

    With ``header_bands``, annotations provably clear of every band (leadered
    chemistry / water labels deep in the plot) are left out: they cannot
    overlap any candidate header box, so every overlap sum is unchanged.
    """
    settle_figure_layout(fig)  # settle tick label positions for this page size
    artists: list = list(fig.texts)
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        artists.extend(_drawn_tick_labels(ax.xaxis, ax.get_xlim()))
        artists.extend(_drawn_tick_labels(ax.yaxis, ax.get_ylim()))
        # Other axes text (transect end labels, notes) is an obstacle too.
        artists.extend(ax.texts)
        artists.extend((ax.xaxis.label, ax.yaxis.label, ax.title, ax._left_title, ax._right_title))
    boxes = []
    for artist in artists:
        if id(artist) in exclude or not artist.get_visible() or not artist.get_text().strip():
            continue
        if isinstance(artist, Annotation):
            if header_bands is not None:
                bound = _cached_annotation_extent(artist, renderer, "leader", _straight_leader_bound)
                if bound is not None:
                    y0, y1 = bound.y0 - pad, bound.y1 + pad
                    if not any(y0 < top and y1 > bottom for bottom, top in header_bands):
                        continue
            extent = _cached_annotation_extent(
                artist, renderer, "extent", lambda a, r: a.get_window_extent(r)
            )
            boxes.append(extent.padded(pad))
            continue
        boxes.append(artist.get_window_extent(renderer).padded(pad))
    return boxes

def _clip_patch_key(patch):
    """Inputs of ``patch.contains`` for a leader's clip patch, or False when
    they cannot be captured (only the plain display-space rectangle
    ``Annotation.update_positions`` builds around unboxed text)."""
    if patch is None:
        return None
    if type(patch) is not Rectangle or not isinstance(patch.get_data_transform(), IdentityTransform):
        return False
    picker = patch.get_picker()
    if picker is not None and not isinstance(picker, (int, float, bool)):
        return False
    return (
        float(patch.get_x()),
        float(patch.get_y()),
        float(patch.get_width()),
        float(patch.get_height()),
        float(patch.get_angle()),
        patch.rotation_point if isinstance(patch.rotation_point, str) else tuple(patch.rotation_point),
        float(patch.get_linewidth()),
        tuple(float(v) for v in patch.get_edgecolor()),
        picker,
    )


def _leader_path_key(arrow):
    """Every input of ``FancyArrowPatch._get_path_in_displaycoord`` for an
    annotation leader, or None when it is not a plain display-space one."""
    positions = arrow._posA_posB
    if positions is None or not isinstance(arrow.get_transform(), IdentityTransform):
        return None
    if arrow.patchB is not None:
        return None
    patch_a = _clip_patch_key(arrow.patchA)
    if patch_a is False:
        return None
    try:
        (xa, ya), (xb, yb) = positions
        return (
            float(xa),
            float(ya),
            float(xb),
            float(yb),
            patch_a,
            float(arrow.shrinkA),
            float(arrow.shrinkB),
            float(arrow._dpi_cor),
            float(arrow.get_mutation_scale()),
            float(arrow.get_mutation_aspect()),
            float(arrow.get_linewidth()),
            id(arrow.get_arrowstyle()),
            id(arrow.get_connectionstyle()),
        )
    except (TypeError, ValueError):  # pragma: no cover - unit-bearing positions
        return None


def _prepared_contains(patch):
    """``patch.contains`` with its per-call set-up done once.

    ``Patch.contains`` rebuilds the patch path, transform and sub-paths on
    every call, and clipping a leader bisects it against its text box with
    dozens of calls while the box stays put. The prepared version runs the
    same ``point_in_path`` test with the same arguments, so every answer is
    the one ``Patch.contains`` gives.
    """
    contains = type(patch).contains
    if contains is not Patch.contains:  # pragma: no cover - custom hit tests
        return None
    path = patch.get_path()
    codes = path.codes
    if codes is not None:
        (idxs,) = np.where(codes == Path.MOVETO)
        idxs = idxs[1:]
        subpaths = list(map(Path, np.split(path.vertices, idxs), np.split(codes, idxs)))
    else:
        subpaths = [path]
    transform = patch.get_transform().frozen()
    if transform and not transform.is_affine:  # pragma: no cover - display boxes are affine
        subpaths = [transform.transform_path(sub) for sub in subpaths]
        transform = None
    radius = patch._process_radius(None)

    def prepared(mouseevent, radius_arg=None):
        if radius_arg is not None or patch._different_canvas(mouseevent):  # pragma: no cover
            return Patch.contains(patch, mouseevent, radius_arg)
        x, y = mouseevent.x, mouseevent.y
        return any(_mpl_path.point_in_path(x, y, radius, sub, transform) for sub in subpaths), {}

    return prepared


class _MemoLeaderPatch(FancyArrowPatch):
    """Label leader whose clipped display path is memoised on its inputs.

    Clipping the leader to the box around its text bisects the path against
    the box, which dominated the draws and extent queries of figures with
    hundreds of leadered values. Every draw / tight-bbox pass at an
    unchanged geometry and dpi recomputed the same path; a repeat now returns
    the path computed for exactly those inputs.
    """

    def _get_path_in_displaycoord(self):
        key = _leader_path_key(self)
        if key is not None:
            memo = self.__dict__.setdefault("_css_leader_paths", {})
            hit = memo.get(key)
            if hit is not None and hit[0] is self.get_arrowstyle() and hit[1] is self.get_connectionstyle():
                return hit[2]
        value = self._clipped_path()
        if key is None:
            return value
        if len(memo) >= 8:
            memo.clear()
        memo[key] = (self.get_arrowstyle(), self.get_connectionstyle(), value)
        return value

    def _clipped_path(self):
        """``FancyArrowPatch._get_path_in_displaycoord`` with the clip boxes'
        hit tests prepared once for the clip (removed again afterwards)."""
        prepared = []
        for patch in (self.patchA, self.patchB):
            if patch is not None and "contains" not in patch.__dict__:
                contains = _prepared_contains(patch)
                if contains is not None:
                    patch.contains = contains
                    prepared.append(patch)
        try:
            return super()._get_path_in_displaycoord()
        finally:
            for patch in prepared:
                patch.__dict__.pop("contains", None)


def _memoise_label(annotation) -> None:
    """Switch a placed label (and its halo twin) to the memoising subclasses.

    Same artists and state, only the layout / leader path computations are
    memoised, so everything drawn is unchanged.
    """
    for text in (annotation, getattr(annotation, "_halo", None)):
        if text is not None and type(text) is Annotation:
            text.__class__ = _MemoLayoutAnnotation
    arrow = getattr(annotation, "arrow_patch", None)
    if arrow is not None and type(arrow) is FancyArrowPatch:
        arrow.__class__ = _MemoLeaderPatch


def _figure_renderer(fig):
    try:
        return fig.canvas.get_renderer()
    except AttributeError:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        return FigureCanvasAgg(fig).get_renderer()


class WaterSeriesLegendEntry(TypedDict):
    series_id: str
    color: str
    marker: str
    linestyle: str | tuple
    level_label: str
    elevation_label: str


def _group_water_levels(
    water_levels: Sequence[WaterLevel],
    profile_lookup: dict[str, tuple[float, float]],
) -> dict[str, list[WaterLevel]]:
    groups: dict[str, list[WaterLevel]] = {}
    for level in water_levels:
        if level.hole_id not in profile_lookup:
            continue
        series_id = level.series_id or "default"
        groups.setdefault(series_id, []).append(level)
    return groups


def _connect_subgroups(
    levels: Sequence[WaterLevel],
) -> dict[str, list[WaterLevel]]:
    """Split a series into connect_group nests (blank = one shared polyline)."""
    subgroups: dict[str, list[WaterLevel]] = {}
    for level in levels:
        group_id = (level.connect_group or "").strip()
        subgroups.setdefault(group_id, []).append(level)
    return subgroups


# Plot-unit tolerance for "above the ground line" (heads equal to the collar
# RL are on the ground, not above it).
_GROUND_CLIP_TOL = 1e-6


def _ground_profile(
    profile_lookup: dict[str, tuple[float, float]], plot_y_values
) -> tuple[np.ndarray, np.ndarray]:
    """Ground polyline (plot coords) through every collar, x ascending.

    Coincident collars (nested wells) keep the highest RL, so a line is never
    clipped below a collar it passes.
    """
    by_x: dict[float, float] = {}
    for x, rl in profile_lookup.values():
        by_x[float(x)] = max(float(rl), by_x.get(float(x), float("-inf")))
    xs = np.asarray(sorted(by_x), dtype=float)
    rls = np.asarray([by_x[x] for x in xs], dtype=float)
    return xs, np.asarray(plot_y_values(rls, rls), dtype=float)


def clip_water_line_to_ground(
    anchor_x: np.ndarray,
    anchor_y: np.ndarray,
    ground_x: np.ndarray,
    ground_y: np.ndarray,
) -> tuple[np.ndarray | None, list[int], list[int]]:
    """Clip a straight-segment water line to the ground polyline.

    ``anchor_*`` are the heads joined by straight segments (x ascending).
    Between two heads the line is replaced by ``min(line, ground)`` — with the
    ground vertices and line/ground crossings inserted so the clip is exact —
    unless both heads are above ground (an artesian zone: left as drawn).
    When one head is above ground (a flowing artesian well) the line must
    still reach that measured head: the run above ground ending at that well
    stays straight, and only other runs between the heads (a hill or valley
    rim in between) are clipped.

    Returns ``(xy, clipped_spans, artesian_spans)``: ``xy`` is the clipped
    ``(N, 2)`` polyline or ``None`` when nothing was clipped (draw the
    original, unchanged); spans are indices ``i`` of anchor pairs ``(i, i+1)``
    where the straight line rose above ground and was clipped, or was left
    above ground (both heads, or the run next to an artesian head) — a span
    may be in both lists.
    """
    anchor_x = np.asarray(anchor_x, dtype=float)
    anchor_y = np.asarray(anchor_y, dtype=float)
    if anchor_x.size < 2 or ground_x.size < 2:
        return None, [], []
    ground_at_anchor = np.interp(anchor_x, ground_x, ground_y)
    above_anchor = anchor_y > ground_at_anchor + _GROUND_CLIP_TOL
    clipped: list[int] = []
    artesian: list[int] = []
    pieces: list[np.ndarray] = []
    for index in range(anchor_x.size - 1):
        x0, x1 = float(anchor_x[index]), float(anchor_x[index + 1])
        y0, y1 = float(anchor_y[index]), float(anchor_y[index + 1])
        straight = np.asarray([[x0, y0], [x1, y1]], dtype=float)
        if x1 - x0 <= 1e-12:
            pieces.append(straight)
            continue
        inner = ground_x[(ground_x > x0) & (ground_x < x1)]
        xs = np.concatenate(([x0], inner, [x1]))
        line = y0 + (y1 - y0) * (xs - x0) / (x1 - x0)
        diff = line - np.interp(xs, ground_x, ground_y)
        if not np.any(diff > _GROUND_CLIP_TOL):
            pieces.append(straight)
            continue
        if above_anchor[index] and above_anchor[index + 1]:
            artesian.append(index)
            pieces.append(straight)
            continue
        cross = np.flatnonzero(diff[:-1] * diff[1:] < 0.0)
        if cross.size:
            share = diff[cross] / (diff[cross] - diff[cross + 1])
            xs = np.sort(np.concatenate((xs, xs[cross] + (xs[cross + 1] - xs[cross]) * share)))
            line = y0 + (y1 - y0) * (xs - x0) / (x1 - x0)
        ground = np.interp(xs, ground_x, ground_y)
        ys = np.minimum(line, ground)
        # One head measured above ground (flowing artesian well): the line
        # must reach that head. The run above ground that ends at the
        # artesian well stays straight (it rises out of the ground towards
        # the head); any other run above ground is still clipped.
        keep = np.zeros(xs.size, dtype=bool)
        above = line - ground > _GROUND_CLIP_TOL
        if above_anchor[index + 1]:
            below = np.flatnonzero(~above)
            keep[(int(below[-1]) if below.size else 0):] = True
        elif above_anchor[index]:
            below = np.flatnonzero(~above)
            keep[: (int(below[0]) + 1 if below.size else xs.size)] = True
        if keep.any():
            ys = np.where(keep, line, ys)
            artesian.append(index)
        if np.any(above & ~keep):
            clipped.append(index)
        elif keep.any():
            # Only the artesian well's own run was above ground: the
            # straight line is drawn as is.
            pieces.append(straight)
            continue
        pieces.append(np.column_stack((xs, ys)))
    if not clipped:
        return None, clipped, artesian
    joined = [pieces[0]] + [piece[1:] if np.allclose(piece[0], prev[-1]) else piece
                            for prev, piece in zip(pieces, pieces[1:], strict=False)]
    return np.vstack(joined), clipped, artesian


class RendererWaterMixin:
    """Water-table markers, polylines, and compact GW legend."""

    def _note_water_line_clip(
        self,
        line_name: str,
        clipped_pairs: Sequence[tuple[str, str]],
        artesian_pairs: Sequence[tuple[str, str]],
        artesian_holes: set[str] | frozenset[str] = frozenset(),
    ) -> None:
        """Section QA notes for water-line spans clipped to / left above the ground.

        ``artesian_holes``: holes whose plotted head is above ground; a span
        with only one of them rises above the ground towards that head.
        """
        notes = getattr(self, "water_line_notes", None)
        if notes is None:
            notes = self.water_line_notes = []
        series = "" if line_name == "default" else f" ({line_name})"
        for left_id, right_id in clipped_pairs:
            text = (
                f"Groundwater line {left_id}–{right_id}{series}: clipped to the ground surface "
                "where the straight line between the heads rose above ground"
            )
            if text not in notes:
                notes.append(text)
        for left_id, right_id in artesian_pairs:
            ends = [hole for hole in (left_id, right_id) if hole in artesian_holes]
            if len(ends) == 1:
                text = (
                    f"Groundwater line {left_id}–{right_id}{series}: rises above the ground "
                    f"surface to the flowing artesian head measured at {ends[0]}"
                )
            else:
                text = (
                    f"Groundwater line {left_id}–{right_id}{series}: drawn above the ground "
                    "surface — both heads are above ground (flowing artesian zone)"
                )
            if text not in notes:
                notes.append(text)

    def _water_elevation_parts(self, level: WaterLevel, collar_rl: float) -> tuple[float, str]:
        """``(value, unit suffix)`` of a water number: RL (masl) in elevation
        mode, depth (mbgs) in relative mode; consulting sheets print no unit."""
        if self.profile.y_axis_mode == "depth_below_collar":
            # Vertical depth below ground: btoc readings lose the stick-up,
            # inclined holes report true vertical depth.
            return self._water_depth_below_ground(level, collar_rl), "mbgs"
        water_rl = self._water_head(level, collar_rl)
        if self.profile.layout == "consulting_section":
            return float(water_rl), ""
        return float(water_rl), "m"

    def _water_elevation_label(self, level: WaterLevel, collar_rl: float) -> str:
        """Annotate water as RL (masl) in elevation mode, or depth (mbgs) in relative mode."""
        value, suffix = self._water_elevation_parts(level, collar_rl)
        number = _fmt_water_number(value)
        return f"{number} {suffix}" if suffix else number

    def _water_legend_captions(
        self,
        series_id: str,
        label: str,
        default_label: str,
    ) -> tuple[str, str]:
        relative = self.profile.y_axis_mode == "depth_below_collar"
        datum = "mbgs" if relative else "masl"
        if not self._legend_upper_case():
            # Section sheet / chart legends: sentence case beside the
            # title-case lithology rows (the series name as given).
            if series_id == "default" and not label:
                if relative:
                    return "Groundwater level (mbgs)", "Groundwater depth (mbgs)"
                return "Groundwater level (masl)", "Groundwater elevation (masl)"
            display_label = label or default_label or series_id
            if relative:
                return (
                    f"Groundwater level ({display_label})",
                    f"Groundwater depth {datum} ({display_label})",
                )
            return (
                f"Groundwater level ({display_label})",
                f"Groundwater elevation masl ({display_label})",
            )
        if series_id == "default" and not label:
            if relative:
                return "GROUNDWATER LEVEL (mbgs)", "GROUNDWATER DEPTH (mbgs)"
            return "GROUNDWATER LEVEL (masl)", "GROUNDWATER ELEVATION (masl)"
        display_label = (label or default_label or series_id).upper()
        if relative:
            return (
                f"GROUNDWATER LEVEL ({display_label})",
                f"GROUNDWATER DEPTH {datum} ({display_label})",
            )
        return (
            f"GROUNDWATER LEVEL ({display_label})",
            f"GROUNDWATER ELEVATION masl ({display_label})",
        )

    def _legend_upper_case(self) -> bool:
        """Consulting title-block legends are upper case; the section sheet
        and chart legends follow their title-case lithology rows."""
        return str(getattr(self.profile, "layout", "")) == "consulting_section"

    def _water_status_note(self, ax, hole_id: str, xy: tuple[float, float], text: str) -> None:
        """One NM / DRY note per hole: a second status merges into it ("DRY / NM")
        instead of a second label crowding the first and the hole's values."""
        notes = self.__dict__.setdefault("_water_status_notes", {})
        key = (id(ax), str(hole_id))
        existing = notes.get(key)
        if existing is not None:
            parts = existing.get_text().split(" / ")
            if text not in parts:
                parts.append(text)
                order = {"DRY": 0, "NM": 1}
                parts.sort(key=lambda part: order.get(part, 2))
                existing.set_text(" / ".join(parts))
                halo = getattr(existing, "_halo", None)
                if halo is not None:
                    halo.set_text(existing.get_text())
            return
        notes[key] = self._water_annotate(
            ax,
            text,
            xy,
            kind="nm",
            color=CONSULTING_NM_COLOR,
            fontsize=8,
            xytext=(4, 0),
        )

    def _water_annotate(
        self,
        ax,
        text: str,
        xy: tuple[float, float],
        *,
        kind: str,
        color: str,
        fontsize: float,
        xytext: tuple[float, float],
        ha: str = "left",
    ):
        """Draw a water number and register it for collision avoidance."""
        annotation = ax.annotate(
            text,
            xy=xy,
            xytext=xytext,
            textcoords="offset points",
            fontsize=fontsize,
            color=color,
            ha=ha,
            va="center",
            zorder=_WATER_LABEL_ZORDER,
            bbox=dict(_WATER_LABEL_BBOX),
            # Leader exists from the start but stays hidden unless the label
            # is moved away from its point by the collision pass.
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "lw": 0.5,
                "shrinkA": 0,
                "shrinkB": _WATER_LEADER_SHRINK_PT,
            },
        )
        annotation.arrow_patch.set_visible(False)
        annotation._water_base_text = text
        annotation._water_base_fontsize = float(fontsize)
        _memoise_label(annotation)
        if not hasattr(self, "_water_labels"):
            self._water_labels = []
        self._water_labels.append((kind, annotation, color))
        return annotation

    def _resolve_water_label_collisions(self, fig) -> None:
        """Move water numbers apart once the figure layout is final.

        Greedy placement: each label takes the nearest candidate offset that
        stays inside its axes and clears every label placed so far plus other
        axes text (chemistry values, annotations). Labels moved away from their
        point get a thin leader line, as on the client CAD figures.
        """
        labels = getattr(self, "_water_labels", None) or []
        if not labels:
            return
        renderer = _figure_renderer(fig)
        for _kind, annotation, _color in labels:
            _memoise_label(annotation)
        # Nothing the placement reads changed since the last run ended (an
        # export of the figure as rendered): the passes would repeat it
        # exactly, so its result stands.
        inputs = self._label_pass_inputs(fig, labels, renderer)
        if inputs is not None and fig.__dict__.get("_css_label_pass_inputs") == inputs:
            return
        fig.__dict__.pop("_css_label_pass_inputs", None)
        for attempt in range(_CHEM_FRAME_FIT_ATTEMPTS + 1):
            if attempt:
                inputs = self._label_pass_inputs(fig, labels, renderer)
            level_before = int(getattr(self, "_water_compact_level", 0))
            self._run_label_passes(fig, labels, renderer)
            # Non-consulting sheets: a value that still crosses the right
            # frame line (the last hole's label room was estimated at the
            # render size; export re-frames the page) widens the x range so
            # it fits, then labels are placed again. Consulting sheets print
            # a scale ratio from the x range, so their range never moves.
            if attempt == _CHEM_FRAME_FIT_ATTEMPTS or not self._widen_frame_for_chem_labels(
                fig, labels, renderer
            ):
                break
            headers = [t for t in getattr(self, "_header_labels", None) or [] if t.figure is fig]
            resolve_header_collisions(fig, headers)
        self._draw_chem_label_strips(fig, labels, renderer)
        footnote = getattr(self, "_update_chemistry_omitted_footnote", None)
        if footnote is not None:
            footnote(fig, labels)
        notes = _chem_drop_notes(labels) + _water_drop_notes(labels)
        if notes != getattr(self, "chemistry_label_notes", None):
            for note in notes:
                logger.warning(note)
        self.chemistry_label_notes = notes
        if (
            inputs is not None
            # A run that raised the water compaction starts the next one at
            # the raised level, which is not the run that produced this.
            and int(getattr(self, "_water_compact_level", 0)) == level_before
            and self._label_pass_inputs(fig, labels, renderer) == inputs
        ):
            fig._css_label_pass_inputs = inputs

    def _label_pass_inputs(self, fig, labels, renderer):
        """Fingerprint of everything a placement run reads, or None.

        Only taken while the figure's layout is settled (the run's own settle
        would not draw), so the obstacle extents measured here are the ones
        the run measures: other axes text, drawn tick labels, the plot boxes
        and limits, columns, labels and the water compaction level.
        """
        from renderer_common import _figure_draw_count, _layout_fingerprint, _same_settled_layout

        draws = _figure_draw_count(fig)
        layout = _layout_fingerprint(fig)
        previous = fig.__dict__.get("_css_settled_layout")
        if (
            draws is None
            or layout is None
            or previous is None
            or previous[0] != draws
            or previous[1] is None
            or not _same_settled_layout(previous[1], layout)
        ):
            return None
        label_ids = {id(annotation) for _kind, annotation, _color in labels}
        label_ids |= {id(annotation._halo) for _k, annotation, _c in labels if hasattr(annotation, "_halo")}
        axes = list(dict.fromkeys(annotation.axes for _kind, annotation, _color in labels))
        texts = []
        for ax in axes:
            for text in ax.texts:
                if id(text) not in label_ids and text.get_visible() and text.get_text().strip():
                    texts.append((id(text), text.get_text(), tuple(text.get_window_extent(renderer).extents)))
        ticks = []
        for ax in fig.axes:
            if not ax.get_visible():
                continue
            for tick in _drawn_tick_labels(ax.xaxis, ax.get_xlim()) + _drawn_tick_labels(ax.yaxis, ax.get_ylim()):
                if tick.get_visible() and tick.get_text().strip():
                    ticks.append((id(tick), tick.get_text(), tuple(tick.get_window_extent(renderer).extents)))
        label_rows = tuple(
            (
                kind,
                id(annotation),
                getattr(annotation, "_water_base_text", annotation.get_text()),
                tuple(float(v) for v in getattr(annotation, "_water_base_xy", annotation.xy)),
                tuple(getattr(annotation, "_water_base_xyann", ())),
                bool(annotation.get_visible() or getattr(annotation, "_water_dropped", False)),
            )
            for kind, annotation, _color in labels
        )
        registered = tuple(
            (
                id(ax),
                tuple(id(points) for points in ax.__dict__.get("_css_water_lines", ())),
                tuple(id(points) for points in ax.__dict__.get("_css_ground_lines", ())),
                tuple(id(collection) for collection in ax.__dict__.get("_css_marker_sets", ())),
            )
            for ax in axes
        )
        spans = tuple(
            (id(ax), float(x0), float(x1)) for ax, x0, x1 in (getattr(self, "_column_spans", None) or [])
        ) + tuple(
            (id(ax), float(x0), float(x1), float(y0), float(y1))
            for ax, x0, x1, y0, y1 in (getattr(self, "_column_extra_spans", None) or [])
        )
        return (
            layout[0],
            id(renderer),
            float(renderer.points_to_pixels(1.0)),
            tuple(texts),
            tuple(ticks),
            label_rows,
            registered,
            spans,
            int(getattr(self, "_water_compact_level", 0)),
            str(getattr(self.profile, "layout", "")),
        )

    def _run_label_passes(self, fig, labels, renderer) -> None:
        # A hole whose values would be dropped is re-placed in compact mode
        # (zig-zag columns, full-frame stack, then smaller text); the whole
        # pass restarts so its earlier values make room for the later ones.
        compact: dict[float, int] = {}
        # Values that came out at mixed sizes (a crowded value shrank) are
        # placed again with every value at the smallest size used: one
        # consistent size per figure reads better than a few tiny values
        # among normal ones.
        font_caps: dict[float, float] = {}
        # Hard cap on full placement passes: escalation raises every hole
        # with drops at once, so a well-formed figure needs at most
        # _CHEM_MAX_COMPACT_LEVEL + 1 passes per font settle.
        budget = _MAX_LABEL_PASSES + _WATER_MAX_COMPACT_LEVEL
        # Water numbers: one compaction level for the whole figure, raised
        # (unit dropped, one decimal, smaller text) while a level number
        # would otherwise be dropped or printed over a neighbouring column.
        water_level = int(getattr(self, "_water_compact_level", 0))
        _apply_water_compaction(labels, water_level)
        for _settle in range(_CHEM_FONT_SETTLE_PASSES + 1):
            while True:
                dropped_holes = self._label_collision_pass(
                    fig, labels, renderer, compact, font_caps=font_caps
                )
                self._label_pass_count = getattr(self, "_label_pass_count", 0) + 1
                budget -= 1
                escalate = {
                    hole
                    for hole in dropped_holes
                    if np.isfinite(hole) and compact.get(hole, 0) < _CHEM_MAX_COMPACT_LEVEL
                }
                raise_water = False
                if getattr(self, "_water_labels_short", False):
                    while water_level < _WATER_MAX_COMPACT_LEVEL and not raise_water:
                        water_level += 1
                        # A level that changes nothing (consulting numbers
                        # carry no unit to drop) costs no pass.
                        raise_water = _apply_water_compaction(labels, water_level)
                    self._water_compact_level = water_level
                if (not escalate and not raise_water) or budget <= 0:
                    break
                for hole in escalate:
                    compact[hole] = compact.get(hole, 0) + 1
            mixed = _chem_mixed_size_holes(labels)
            mixed = {
                hole: scale
                for hole, scale in mixed.items()
                if scale < font_caps.get(hole, 1.0) - 1e-9
            }
            if not mixed or _settle == _CHEM_FONT_SETTLE_PASSES or budget <= 0:
                break
            font_caps.update(mixed)

    def _widen_frame_for_chem_labels(self, fig, labels, renderer) -> bool:
        """Extend the right x limit so no visible chemistry value crosses the frame.

        Returns True when a limit moved (the caller re-places the labels).
        """
        if str(getattr(self.profile, "layout", "")) == "consulting_section":
            return False
        gap = renderer.points_to_pixels(_CHEM_COLUMN_GAP_PT)
        needed: dict[int, tuple[object, float]] = {}
        for kind, annotation, _color in labels:
            if kind != "chem" or not annotation.get_visible():
                continue
            ax = annotation.axes
            frame = ax.get_window_extent(renderer)
            box = _chem_footprint(annotation, renderer)
            if box.x1 <= frame.x1 - gap:
                continue
            left, right = ax.get_xlim()
            if right <= left or frame.width <= 0:
                continue
            anchor_px = _chem_anchor(annotation)[0]
            reach = box.x1 - anchor_px + gap  # label extent right of its anchor
            usable = frame.width - reach
            anchor_x = float(annotation.xy[0])
            if usable <= 0.1 * frame.width or anchor_x <= left:
                continue
            target = left + (anchor_x - left) * frame.width / usable
            if target > right and target > needed.get(id(ax), (ax, right))[1]:
                needed[id(ax)] = (ax, target)
        for ax, target in needed.values():
            left, _right = ax.get_xlim()
            ax.set_xlim(left, target)
        return bool(needed)

    def _chem_strip_layout(self, fig, labels, renderer, column_boxes=None):
        """Where the "strip" knock-outs go: ``(ax, x0, y0, x1, y1)`` display rectangles.

        One knock-out per placed value, flush with its column's right edge
        (no pad on that side, square corners) and running to the value's
        right edge plus the pad; values of one hole less than about one line
        apart share a single strip. A strip never reaches the next column,
        the frame or above the ground line, and a widely spaced hole's values
        never merge into one tall slab over the geology between them.
        """
        if str(getattr(self.profile, "chemistry_label_style", "") or "") != "strip":
            return []
        if column_boxes is None:
            column_boxes = self._column_obstacle_boxes(fig)
        if not column_boxes:
            return []
        pad = renderer.points_to_pixels(_CHEM_STRIP_PAD_PT)
        holes: dict[tuple[int, float], list] = {}
        for kind, annotation, _color in labels:
            if kind == "chem" and annotation.get_visible():
                key = (id(annotation.axes), _chem_hole_key(annotation))
                holes.setdefault(key, []).append(annotation)
        result: list[tuple[object, float, float, float, float]] = []
        for hole_labels in holes.values():
            ax = hole_labels[0].axes
            own = _own_column(hole_labels[0], column_boxes)
            if own is None:
                continue
            boxes = [_chem_footprint(annotation, renderer) for annotation in hole_labels]
            frame = ax.get_window_extent(renderer)
            # Never over the next column to the right, nor past the frame.
            next_x0 = [col.x0 for col in column_boxes if col is not own and col.x0 > own.x1]
            right_limit = min([frame.x1] + next_x0)
            # One knock-out per value (from the column edge), merged only with
            # a neighbour less than about one line away: a single strip over
            # every value of the hole blanked tall slabs of geology between
            # widely spaced readings.
            rects = []
            group: list = []
            for box in sorted(boxes, key=lambda item: -item.y1):
                if group and min(item.y0 for item in group) - box.y1 > box.height:
                    rects.append(_strip_rect(own, group, pad))
                    group = []
                group.append(box)
            if group:
                rects.append(_strip_rect(own, group, pad))
            for x0, y0, x1, y1 in rects:
                x0 = max(x0, own.x1)
                x1 = min(x1, right_limit)
                y0 = max(y0, frame.y0)
                # Never above the ground line: the column box starts at the collar.
                y1 = min(y1, frame.y1, own.y1)
                if x1 > x0 and y1 > y0:
                    result.append((ax, x0, y0, x1, y1))
        return result

    def _draw_chem_label_strips(self, fig, labels, renderer) -> None:
        """Knock a background strip out of the fills beside each labelled column.

        "strip" style: values read on a clean strip starting flush at their
        column's right edge and running to the value's right edge (plus the
        pad), with the hatching resuming beyond it. Unlike the "box" style
        (a padded white box around the text, detached from the column by the
        normal gap) a strip has no gap and no pad on the column side. The
        strip sits above lithology fills and contacts but below columns,
        water lines and markers, and is clipped so it never reaches another
        column or leaves the frame.
        """
        # Strips belong to their figure: a second render() on this renderer
        # must not strip the knock-outs off an earlier, still-open figure.
        for strip in getattr(fig, "_chem_strips", ()):
            strip.remove()
        fig._chem_strips = []
        self._chem_strips = fig._chem_strips
        for ax, x0, y0, x1, y1 in self._chem_strip_layout(fig, labels, renderer):
            (dx0, dy0), (dx1, dy1) = ax.transData.inverted().transform([[x0, y0], [x1, y1]])
            strip = Rectangle(
                (min(dx0, dx1), min(dy0, dy1)),
                abs(dx1 - dx0),
                abs(dy1 - dy0),
                facecolor=ax.get_facecolor(),
                edgecolor="none",
                linewidth=0.0,
                zorder=_CHEM_STRIP_ZORDER,
            )
            strip.set_gid("chemistry-label-strip")
            ax.add_patch(strip)
            self._chem_strips.append(strip)

    def _label_collision_pass(
        self,
        fig,
        labels,
        renderer,
        compact: dict[float, int],
        *,
        font_caps: dict[float, float] | None = None,
    ) -> set[float]:
        """One greedy placement pass; returns hole keys with dropped chemistry values."""
        dropped_holes: set[float] = set()
        font_caps = font_caps or {}
        pad = renderer.points_to_pixels(_LABEL_PAD_PT)
        # Set when a level number / note is dropped or lands on a
        # neighbouring column: the figure's water numbers then compact.
        self._water_labels_short = False
        # A re-run (page resize) starts from scratch: restore labels a previous
        # pass dropped so they get another chance at the new size.
        for _kind, annotation, _color in labels:
            if getattr(annotation, "_water_dropped", False):
                annotation._water_dropped = False
                _set_label_visible(annotation, True)
        water_artists = {id(annotation) for _kind, annotation, _color in labels}
        water_artists |= {
            id(annotation._halo) for _k, annotation, _c in labels if hasattr(annotation, "_halo")
        }
        placed: list = []
        for ax in {annotation.axes for _kind, annotation, _color in labels}:
            for text in ax.texts:
                if id(text) not in water_artists and text.get_visible() and text.get_text().strip():
                    placed.append(text.get_window_extent(renderer).padded(pad))
        # Tick labels too — the twin RL axis on consulting sheets sits exactly
        # where a last-hole value label wants to go.
        settle_figure_layout(fig)
        # Anchors are fixed from here on (the layout is settled): memoise them.
        _ANCHOR_SCOPES.append({})
        try:
            for ax in fig.axes:
                if not ax.get_visible():
                    continue
                for tick in _drawn_tick_labels(ax.xaxis, ax.get_xlim()) + _drawn_tick_labels(
                    ax.yaxis, ax.get_ylim()
                ):
                    if tick.get_visible() and tick.get_text().strip():
                        placed.append(tick.get_window_extent(renderer).padded(pad))
            _separate_markers({annotation.axes for _k, annotation, _c in labels}, renderer)
            # Borehole columns are obstacles for value labels: a label over a
            # column hides the stick/markers, and over a neighbour's column it
            # misattributes the value.
            column_boxes = self._column_obstacle_boxes(fig)
            # Inclined holes' stepped pieces: hard obstacles like the columns,
            # but not "a hole's column" for the side / next-hole rules.
            extra_boxes = self._column_extra_obstacle_boxes(fig)
            # Every drawn marker (chemistry dots nudged off the stick,
            # groundwater markers): a value printed on one hides it.
            marker_arr = _ObstacleArray(
                _marker_boxes(
                    {annotation.axes for _k, annotation, _c in labels},
                    renderer,
                    renderer.points_to_pixels(_WATER_MARKER_CLEAR_PT),
                )
            )
            # Water numbers hole by hole (left to right), each hole's top label
            # first, so crowded holes stack their labels in level order.
            ordered = _chem_depth_ordered(sorted(labels, key=_label_order_key))
            # Display y of each hole's lowest placed chemistry label: the next
            # (deeper) value of that hole must land below it.
            chem_ceiling: dict[float, float] = {}
            chem_prev_box: dict[float, tuple[Bbox, float]] = {}
            line_segs = _water_line_segments({annotation.axes for _k, annotation, _c in labels})
            consulting = str(getattr(self.profile, "layout", "")) == "consulting_section"
            px_per_pt = renderer.points_to_pixels(1.0)
            right_columns: dict[int, list] = {}  # own column -> next two columns right
            typical_gap = _typical_column_gap(column_boxes)
            # Obstacles as an (N, 4) array: with hundreds of labels x dozens of
            # candidates the per-box Python loop dominated render time.
            placed_arr = _ObstacleArray(placed)
            column_arr = _ObstacleArray(column_boxes + extra_boxes)
            # Strip knock-outs of the placed values (chemistry goes first), built
            # once the first water label comes up: water numbers and DRY / NM
            # notes keep off them whenever they can.
            strip_arr: _ObstacleArray | None = None
            scene: dict | None = None
            for kind, annotation, _color in ordered:
                frame = annotation.axes.get_window_extent(renderer)
                if kind != "chem" and strip_arr is None:
                    strip_arr = _ObstacleArray(
                        Bbox.from_extents(x0, y0, x1, y1)
                        for _ax, x0, y0, x1, y1 in self._chem_strip_layout(fig, labels, renderer, column_boxes)
                    )
                if not hasattr(annotation, "_water_base_xyann"):
                    annotation._water_base_xyann = tuple(annotation.xyann)
                base = annotation._water_base_xyann
                if not np.all(np.isfinite(_anchor_px(annotation))):
                    # Degenerate axes (every hole at one x, a zero span): the
                    # label has no place on the page. Drop it once; it never
                    # escalates a re-run (a NaN hole key matched no earlier
                    # pass and re-ran the placement without end).
                    annotation._water_dropped = True
                    annotation.arrow_patch.set_visible(False)
                    _set_label_visible(annotation, False)
                    continue
                if kind == "chem":
                    hole_key = _chem_hole_key(annotation)
                    own = _own_column(annotation, column_boxes)
                    clearance = _CHEM_COLUMN_GAP_PT * px_per_pt
                    need = _chem_min_width(annotation, renderer)
                    right_limit = _chem_right_limit(
                        own,
                        column_boxes,
                        frame,
                        consulting=consulting,
                        clearance=clearance,
                        need=need,
                        typical_gap=typical_gap,
                        memo=right_columns,
                    )
                    # Wider than the gap beside its column even at the smallest
                    # font: no compact level can place it, so the hole is not
                    # re-run for it (its other values keep their normal spots).
                    too_wide = own is not None and right_limit - (own.x1 + clearance) < need
                    spot = None
                    if not too_wide:
                        spot = _place_chem_label(
                            annotation,
                            renderer,
                            fig,
                            placed_arr=placed_arr,
                            column_arr=column_arr,
                            marker_arr=marker_arr,
                            own=own,
                            right_limit=right_limit,
                            pad=pad,
                            ceiling=chem_ceiling.get(hole_key, np.inf),
                            compact=compact.get(hole_key, 0),
                            line_segs=line_segs,
                            prev_box=chem_prev_box.get(hole_key),
                            font_cap=font_caps.get(hole_key, 1.0),
                        )
                    if spot is None:
                        if too_wide:
                            _set_chem_fontsize(annotation, annotation._chem_base_fontsize)
                            annotation.xyann = base
                        else:
                            dropped_holes.add(hole_key)
                        # Nowhere right of its column is free: keep the stick/marker,
                        # never print the value over a column or another label.
                        annotation._water_dropped = True
                        annotation.arrow_patch.set_visible(False)
                        _set_label_visible(annotation, False)
                        continue
                    dx, dy = spot
                    annotation.xyann = (dx, dy)
                    _sync_label_companions(annotation, fig, dx, dy, "left")
                    chem_box = _chem_footprint(annotation, renderer)
                    chem_ceiling[hole_key] = 0.5 * (chem_box.y0 + chem_box.y1)
                    chem_prev_box[hole_key] = (chem_box, _chem_anchor(annotation)[1])
                    final_box = chem_box.padded(pad)
                    placed.append(final_box)
                    placed_arr.add(final_box)
                    # A value moved more than about one label height from its
                    # reading (either direction) always gets a thin leader back to
                    # it, even with leaders off, so it is never read against the
                    # wrong hole or depth.
                    # A value pulled in towards its column (narrow gap) is closer
                    # to its reading, not away from it: only rightward moves count.
                    shift = max(abs(dy - base[1]), max(0.0, dx - base[0]))
                    label_height_pt = chem_box.height / renderer.points_to_pixels(1.0)
                    leader = shift > max(label_height_pt, _CHEM_MIN_LEADER_SHIFT_PT)
                    annotation.arrow_patch.set_visible(leader)
                    continue
                if scene is None:
                    scene = self._water_label_scene(
                        fig, labels, renderer, column_boxes, strip_arr, extra_boxes=extra_boxes
                    )
                if not self._place_water_label(kind, annotation, renderer, scene, placed_arr, pad):
                    # No free spot: keep the marker, never print overlapping text.
                    annotation._water_dropped = True
                    annotation.arrow_patch.set_visible(False)
                    _set_label_visible(annotation, False)
                    if kind in ("rl", "nm"):
                        self._water_labels_short = True
                elif kind in ("rl", "nm") and getattr(annotation, "_water_on_neighbour", False):
                    self._water_labels_short = True
            return dropped_holes
        finally:
            _ANCHOR_SCOPES.pop()

    def _water_label_scene(self, fig, labels, renderer, column_boxes, strip_arr, *, extra_boxes=()) -> dict:
        """Obstacles the water numbers keep off, read once the values are placed."""
        axes = {annotation.axes for _k, annotation, _c in labels}
        leaders = []
        for kind, annotation, _color in labels:
            if kind != "chem" or not annotation.get_visible() or not annotation.arrow_patch.get_visible():
                continue
            box = _chem_footprint(annotation, renderer)
            x, y = _chem_anchor(annotation)
            leaders.append((0.5 * (box.x0 + box.x1), 0.5 * (box.y0 + box.y1), x, y))
        water_segs = _water_line_segments(axes)
        ground_segs = _ground_line_segments(axes)
        return {
            "lines": np.vstack((water_segs, ground_segs)),
            "ground_lines": ground_segs,
            "ground": [
                ax.transData.transform(points)
                for ax in axes
                for points in ax.__dict__.get("_css_ground_lines", ())
            ],
            "markers": _water_marker_points(axes, renderer),
            "columns": np.array([(b.x0, b.y0, b.x1, b.y1) for b in column_boxes], dtype=float).reshape(-1, 4),
            # Inclined holes' stepped column pieces: never covered by any
            # water number, note or gradient (hard, like the own column).
            "column_pieces": np.array([(b.x0, b.y0, b.x1, b.y1) for b in extra_boxes], dtype=float).reshape(-1, 4),
            "column_boxes": column_boxes,
            "strips": strip_arr._arr if strip_arr is not None else np.zeros((0, 4)),
            "leaders": np.array(leaders, dtype=float).reshape(-1, 4),
        }

    def _place_water_label(self, kind, annotation, renderer, scene, placed_arr, pad) -> bool:
        """Put one water number / note / gradient on its best free spot.

        Hard rules (a spot breaking one is never used): inside the frame, off
        every placed label and leader, off its own borehole column and the
        groundwater markers, and a leader (if any) crossing no other label or
        leader. Neighbouring columns are soft: every spot clear of them is
        tried first (the figure compacts its numbers before one is used).
        RL numbers and NM notes prefer spots off every water / ground
        line and the chemistry strips, but take a line over being dropped;
        gradients take a line only as the last resort. A spot within ``_WATER_NEAR_HEIGHTS``
        label heights of the marker (beside it, however far out sideways)
        beats any further down or up the hole, even one on the water line
        itself (never the ground line): a number beside its marker on the
        line reads better than one at the end of a long leader across the
        geology. A spot wholly above the ground surface is the last choice
        before dropping the label. Returns False when nothing fits.
        """
        ppt = renderer.points_to_pixels(1.0)
        ax = annotation.axes
        frame = ax.get_window_extent(renderer)
        if not hasattr(annotation, "_water_base_xy"):
            annotation._water_base_xy = tuple(annotation.xy)
        annotation.xy = annotation._water_base_xy
        anchor = _anchor_px(annotation)
        nudge = 0.0
        marker = getattr(annotation, "_water_marker", None)
        marker_px = None
        if marker is not None:
            collection, index = marker
            transform = collection.get_offset_transform()
            now = transform.transform([collection.get_offsets()[index]])[0]
            marker_px = (
                float(now[0]),
                _marker_radius_px(collection, renderer) + renderer.points_to_pixels(_WATER_MARKER_EDGE_PT),
            )
            nudge = float(now[0] - transform.transform([collection._css_base_offsets[index]])[0][0])
            if abs(nudge) > 0.5 and transform is ax.transData:
                # Leader to the drawn (nudged) marker, not the column centre.
                annotation.xy = tuple(float(v) for v in collection.get_offsets()[index])
        lead_to = _anchor_px(annotation)
        annotation.set_horizontalalignment("center")
        annotation.set_verticalalignment("center")
        annotation.xyann = (0.0, 0.0)
        raw = _text_box(annotation, renderer)
        grow = _WATER_LABEL_BBOX_PAD * float(annotation.get_fontsize()) * ppt
        w, h = raw.width + 2 * grow, raw.height + 2 * grow
        if kind == "gradient":
            centres, costs, leadered, feet = self._gradient_candidates(annotation, anchor, w, h, ppt)
            sides = np.zeros(centres.shape[0])
            adjacent = None
        else:
            # Every marker drawn on this hole (nudged ones may stick out of
            # the column): numbers start beyond them, in one tidy column.
            markers = scene["markers"]
            reach = 4.0 * (marker_px[1] if marker_px is not None else 5.0 * ppt)
            near = markers[np.abs(markers[:, 0] - anchor[0]) <= reach] if markers.shape[0] else markers
            if near.shape[0]:
                extent = (float(np.min(near[:, 0] - near[:, 2])), float(np.max(near[:, 0] + near[:, 2])))
            else:
                extent = None
            centres, costs, leadered, sides, extras = _side_candidates(
                anchor, w, h, ppt, scene["column_boxes"], marker_extent=extent
            )
            # Right beside its own column (a small sideways step at most) and
            # within a few label heights of the marker: a number pushed out
            # into the middle of the span reads as nobody's (or the next
            # well's), so these spots come first.
            adjacent = (extras <= _WATER_ADJACENT_EXTRA_PT) & (
                np.abs(centres[:, 1] - float(anchor[1])) <= _WATER_ADJACENT_HEIGHTS * h
            )
            if abs(nudge) > 0.5:
                # Number on the side its marker was nudged to.
                # The other side is left to the number nudged that way.
                costs = costs + np.where(sides * nudge < 0, _WATER_WRONG_SIDE_COST_PT, 0.0)
                costs = costs - np.where(sides < 0, _WATER_LEFT_COST_PT, 0.0) * (nudge < 0)
        if centres.shape[0] == 0:
            return False
        boxes = np.column_stack(
            (centres[:, 0] - w / 2, centres[:, 1] - h / 2, centres[:, 0] + w / 2, centres[:, 1] + h / 2)
        )
        ok = (
            (boxes[:, 0] >= frame.x0)
            & (boxes[:, 2] <= frame.x1)
            & (boxes[:, 1] >= frame.y0)
            & (boxes[:, 3] <= frame.y1)
        )
        if kind == "gradient" and scene["ground"]:
            # Never above the ground surface (an i= in the sky reads as
            # nobody's): the box top stays under the ground line.
            top = boxes[:, 3] + _WATER_LINE_CLEAR_PT * ppt
            for x_col in (boxes[:, 0], centres[:, 0], boxes[:, 2]):
                ok &= top <= _ground_display_y(scene["ground"], x_col)
        if kind == "gradient" and scene["lines"].shape[0]:
            # Nothing between the label and its own segment: across another
            # series' line it reads as that line's gradient.
            ok &= ~_segments_cross(np.column_stack((feet, centres)), scene["lines"]).any(axis=1)
        idx = np.flatnonzero(ok)
        # Own column (the one the marker / note sits in): never covered.
        # Every other column is a soft obstacle, the last resort before
        # the label is dropped (the figure compacts its numbers first).
        columns = scene["columns"]
        own_col = np.zeros(columns.shape[0], dtype=bool)
        if kind != "gradient" and columns.shape[0]:
            own_col = (
                (columns[:, 0] - 0.5 <= anchor[0])
                & (anchor[0] <= columns[:, 2] + 0.5)
                & (columns[:, 1] - 0.5 <= anchor[1])
                & (anchor[1] <= columns[:, 3] + 0.5)
            )
        col_clear = _WATER_COLUMN_CLEAR_PT * ppt
        other_col_hit = np.zeros(boxes.shape[0], dtype=bool)
        if idx.size:
            sub = boxes[idx]
            bad = placed_arr.overlaps_many(sub)
            cleared = sub + np.array([-col_clear, 0.0, col_clear, 0.0])
            bad |= _boxes_overlap_any(columns[own_col], cleared)
            if scene["column_pieces"].shape[0]:
                bad |= _boxes_overlap_any(scene["column_pieces"], cleared)
            bad |= _circles_hit_boxes(scene["markers"], sub, _WATER_MARKER_CLEAR_PT * ppt)
            if scene["leaders"].shape[0]:
                bad |= _any_segment_hits(scene["leaders"], sub + np.array([-pad, -pad, pad, pad]))
            other_col_hit[idx] = _boxes_overlap_any(columns[~own_col], cleared)
            idx = idx[~bad]
        leader_segs = None
        if idx.size and leadered[idx].any():
            # Leader: from the middle of the box side facing the column to
            # the marker, stopping short of it.
            starts = centres - np.column_stack((sides * w / 2, np.zeros(sides.size)))
            vec = lead_to[None, :] - starts
            length = np.maximum(np.hypot(vec[:, 0], vec[:, 1]), 1e-9)
            end = lead_to[None, :] - vec * (_WATER_LEADER_SHRINK_PT * ppt / length)[:, None]
            leader_segs = np.column_stack((starts, end))
            lead_idx = idx[leadered[idx]]
            segs = leader_segs[lead_idx]
            bad = np.zeros(lead_idx.size, dtype=bool)
            placed_raw = placed_arr.raw(pad)
            if placed_raw.shape[0]:
                bad |= _segment_hits_any(segs, placed_raw)
            if scene["leaders"].shape[0]:
                bad |= _segments_cross(segs, scene["leaders"]).any(axis=1)
            idx = np.setdiff1d(idx, lead_idx[bad], assume_unique=True)
        if not idx.size:
            return False
        sub = boxes[idx]
        line_clear = _WATER_LINE_CLEAR_PT * ppt
        on_line = np.zeros(idx.size, dtype=bool)
        if scene["lines"].shape[0]:
            grown = sub + np.array([-line_clear, -line_clear, line_clear, line_clear])
            on_line = _any_segment_hits(scene["lines"], grown)
        on_strip = np.zeros(idx.size, dtype=bool)
        if scene["strips"].shape[0]:
            on_strip = _boxes_overlap_any(scene["strips"], sub)
        near = np.abs(centres[idx, 1] - float(anchor[1])) <= _WATER_NEAR_HEIGHTS * h
        on_other = other_col_hit[idx]
        if kind == "gradient":
            # Off every line first; a spot touching a line (the bend at a
            # hole, not its own segment) beats dropping the gradient.
            tiers = [~on_line & ~on_strip, ~on_line, ~on_strip, np.ones(idx.size, dtype=bool)]
        else:
            on_ground = np.zeros(idx.size, dtype=bool)
            if scene["ground_lines"].shape[0]:
                on_ground = _any_segment_hits(scene["ground_lines"], grown)
            # Wholly above the ground surface ("in the sky"): a number there
            # reads as nobody's level, so every other spot comes first.
            sky = np.zeros(idx.size, dtype=bool)
            if scene["ground"]:
                sky = np.ones(idx.size, dtype=bool)
                for x_col in (sub[:, 0], 0.5 * (sub[:, 0] + sub[:, 2]), sub[:, 2]):
                    sky &= sub[:, 1] > _ground_display_y(scene["ground"], x_col)
            tiers = [
                near & ~on_line & ~on_strip & ~sky,
                near & ~on_line & ~sky,
                near & ~on_ground & ~on_strip & ~sky,
                near & ~on_ground & ~sky,
                ~on_line & ~on_strip & ~sky,
                ~on_line & ~sky,
                ~on_line,
                np.ones(idx.size, dtype=bool),
            ]
            if adjacent is not None:
                # Beside its own column first (off every line), then the
                # usual ladder: never mid-span when the column side is free.
                adj = adjacent[idx]
                tiers = [adj & ~on_line & ~on_strip & ~sky, adj & ~on_line & ~sky] + tiers
        # Every tier clear of the neighbouring columns first, then the same
        # ladder over them (never the own column, filtered above).
        tiers = [keep & ~on_other for keep in tiers] + tiers
        choice = None
        for keep in tiers:
            if keep.any():
                pick = idx[keep]
                choice = int(pick[np.argmin(costs[pick])])
                break
        if choice is None:
            return False
        annotation._water_on_neighbour = bool(other_col_hit[choice])
        cx, cy = centres[choice]
        drawn = _anchor_px(annotation)
        dx, dy = (cx - drawn[0]) / ppt, (cy - drawn[1]) / ppt
        annotation.xyann = (dx, dy)
        # The drawn leader leaves the box side facing the column.
        annotation._arrow_relpos = (0.5 - 0.5 * float(sides[choice]), 0.5)
        _sync_label_companions(annotation, annotation.figure, dx, dy, "center")
        final = Bbox.from_extents(*boxes[choice]).padded(pad)
        placed_arr.add(final)
        leader = bool(leadered[choice])
        annotation.arrow_patch.set_visible(leader)
        if leader and leader_segs is not None:
            scene["leaders"] = np.vstack((scene["leaders"], leader_segs[choice][None, :]))
        return True

    def _gradient_candidates(self, annotation, anchor, w, h, ppt):
        """Spots along the label's own water segment, offset perpendicular to it:
        ``(centres, costs, leadered, feet)`` where ``feet`` are the matching
        points on the segment."""
        seg = getattr(annotation, "_gradient_segment", None)
        ax = annotation.axes
        if seg is None:
            p0 = p1 = anchor
        else:
            p0, p1 = ax.transData.transform([seg[0], seg[1]])
        vec = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
        length = float(np.hypot(*vec))
        if length < 1e-9:
            u = np.array([1.0, 0.0])
        else:
            u = vec / length
        normal = np.array([-u[1], u[0]])
        if normal[1] < 0:
            normal = -normal
        half = abs(normal[0]) * w / 2 + abs(normal[1]) * h / 2 + _WATER_LINE_CLEAR_PT * ppt
        centres, costs, feet = [], [], []
        for t in _GRADIENT_T:
            point = np.asarray(p0, dtype=float) + t * vec
            for side, side_cost in ((1.0, 0.0), (-1.0, _GRADIENT_BELOW_COST)):
                for extra in _GRADIENT_PERP_EXTRA_PT:
                    centres.append(point + side * normal * (half + extra * ppt))
                    costs.append(abs(t - 0.5) * 40.0 + side_cost + extra)
                    # Just off the segment, on the label's side.
                    feet.append(point + side * normal * 0.5)
        centres_arr = np.asarray(centres, dtype=float)
        return (
            centres_arr,
            np.asarray(costs, dtype=float),
            np.zeros(len(centres), dtype=bool),
            np.asarray(feet, dtype=float),
        )

    def _column_extra_obstacle_boxes(self, fig) -> list:
        """Display boxes of inclined holes' column pieces away from the collar x
        (over their own height; the top / bottom piece runs to the frame),
        plus the LNAPL bands (grown to their drawn minimum size): a value or
        water number printed over a band hid it (QA round 8)."""
        boxes = []
        for ax, x0, x1, y_lo, y_hi in getattr(self, "_column_extra_spans", None) or []:
            lim0, lim1 = ax.get_ylim()
            lo_lim, hi_lim = min(lim0, lim1), max(lim0, lim1)
            lo, hi = max(float(y_lo), lo_lim), min(float(y_hi), hi_lim)
            if hi <= lo:
                continue
            (ax0, ay0), (ax1, ay1) = ax.transData.transform([[x0, lo], [x1, hi]])
            boxes.append(Bbox.from_extents(min(ax0, ax1), min(ay0, ay1), max(ax0, ax1), max(ay0, ay1)))
        boxes += self._lnapl_band_boxes(fig)
        return boxes

    @staticmethod
    def _lnapl_band_boxes(fig) -> list:
        """Display boxes of the drawn LNAPL bands (``MinSizeBandCollection``)."""
        renderer = fig.canvas.get_renderer()
        min_half = 0.5 * renderer.points_to_pixels(
            np.asarray([_LNAPL_MIN_WIDTH_PT, _LNAPL_MIN_HEIGHT_PT], dtype=float)
        )
        boxes = []
        for ax in fig.axes:
            for band in ax.collections:
                data_boxes = getattr(band, "_data_boxes", None)
                if band.get_gid() != "lnapl_band" or data_boxes is None or not band.get_visible():
                    continue
                lo = ax.transData.transform(data_boxes[:, :2])
                hi = ax.transData.transform(data_boxes[:, 2:])
                centre = 0.5 * (lo + hi)
                half = np.maximum(0.5 * np.abs(hi - lo), min_half)
                for (cx, cy), (hx, hy) in zip(centre, half, strict=True):
                    boxes.append(Bbox.from_extents(cx - hx, cy - hy, cx + hx, cy + hy))
        return boxes

    def _column_obstacle_boxes(self, fig) -> list:
        """Display-space boxes of every borehole column on the section."""
        spans = getattr(self, "_column_spans", None) or []
        boxes = []
        for ax, x0, x1 in spans:
            y0, y1 = ax.get_ylim()
            lo, hi = min(y0, y1), max(y0, y1)
            (ax0, ay0), (ax1, ay1) = ax.transData.transform([[x0, lo], [x1, hi]])
            # Normalised: an inverted (depth) y axis maps lo above hi, and an
            # un-normalised Bbox has y0 > y1 — every overlap test and the
            # strip's top clamp would silently fail.
            boxes.append(Bbox.from_extents(min(ax0, ax1), min(ay0, ay1), max(ax0, ax1), max(ay0, ay1)))
        return boxes

    def _draw_water_table(
        self,
        ax,
        hole_summary: pd.DataFrame,
        water_levels: Sequence[WaterLevel],
        collar_lookup: dict[str, float],
        *,
        label_elevations: bool = False,
        label_dry_wells: bool = False,
        label_series_gaps: bool = False,
        water_color: str | None = None,
        profile_lookup: dict[str, tuple[float, float]] | None = None,
        track_half: float | None = None,
    ) -> None:
        # Section notes on the drawn water lines (ground clipping / artesian)
        # and the LNAPL head correction.
        self.water_line_notes = []
        # Whether an LNAPL product band was drawn (legend entry only then).
        self.lnapl_drawn = False
        self.artesian_drawn = False
        self.artesian_color = None
        # Whether any "i=" gradient label was drawn (legend / notes key then).
        self.gradient_labels_drawn = False
        if hole_summary.empty:
            return
        self._water_status_notes = {}
        if not water_levels:
            # No water data at all: 'NM' would just label every hole (documented:
            # NM only when dry-well labeling is on AND water data exist).
            self.water_series_legend = []
            return
        if profile_lookup is None:
            profile_lookup = self._profile_lookup(hole_summary, collar_lookup)
        if len(profile_lookup) >= 2:
            ground = sorted((float(x), float(rl)) for x, rl in profile_lookup.values())
            ground_rl = np.asarray([rl for _x, rl in ground], dtype=float)
            _register_ground_line(
                ax, [x for x, _rl in ground], self._plot_y_values(ground_rl, ground_rl)
            )
        # Straight lines between heads may rise above a topographic low between
        # wells (creek valley): they are clipped to the ground in elevation
        # mode (depth mode has no common ground line to clip against).
        clip_ground = (
            _ground_profile(profile_lookup, self._plot_y_values)
            if len(profile_lookup) >= 2 and self.profile.y_axis_mode != "depth_below_collar"
            else None
        )
        series_groups = _group_water_levels(water_levels, profile_lookup)
        holes_with_any_water = {
            level.hole_id for levels in series_groups.values() for level in levels
        }
        fully_dry_nm_drawn: set[str] = set()
        if label_dry_wells:
            dry_lookup = {
                hole_id: profile
                for hole_id, profile in profile_lookup.items()
                if hole_id not in holes_with_any_water
            }
            if dry_lookup:
                dry_x = np.fromiter((p[0] for p in dry_lookup.values()), dtype=float, count=len(dry_lookup))
                dry_collars = np.fromiter((p[1] for p in dry_lookup.values()), dtype=float, count=len(dry_lookup))
                dry_elev = dry_collars - 1.0
                if getattr(self, "along_hole_tvd", None):
                    # Note anchors 1 m down the hole (true vertical on surveyed holes).
                    dry_elev = np.asarray(
                        [
                            self._hole_elevation_at_depth(str(hole_id), 1.0, float(collar))
                            for hole_id, collar in zip(dry_lookup.keys(), dry_collars, strict=True)
                        ],
                        dtype=float,
                    )
                dry_y = self._plot_y_values(dry_elev, dry_collars)
                for hole_id, x_profile, y in zip(dry_lookup.keys(), dry_x, dry_y, strict=True):
                    fully_dry_nm_drawn.add(str(hole_id))
                    self._water_status_note(ax, str(hole_id), (float(x_profile), float(y)), "NM")
        if not series_groups:
            return
        self.water_series_legend = []
        interpolate = self.interpolate_water_table or self.profile.interpolate_water_table_default
        use_segments = self.profile.water_interpolate_segments
        across_gaps = self.profile.water_interpolate_across_gaps
        default_water_color = (
            CONSULTING_WATER_COLOR
            if self.profile.layout == "consulting_section"
            else WATER_COLOR
        )
        profile_marker = _GW_MARKER_MAP.get(self.profile.water_symbol, self.profile.water_symbol)
        transect_hole_ids = hole_summary.sort_values("x_profile")["hole_id"].astype(str).tolist()
        transect_x = {
            str(row.hole_id): float(row.x_profile)
            for row in hole_summary.itertuples(index=False)
        }
        consulting = self.profile.layout == "consulting_section"
        depth_mode = self.profile.y_axis_mode == "depth_below_collar"
        # Series drawn with a joining line: the legend keys a line only for
        # those (markers only for a single reading, a series broken by every
        # dry / NM gap, or depth mode across different collar elevations).
        joined_series: set[str] = set()
        n_series = len(series_groups)
        multi_series = n_series >= 2
        # (x, collar RL, reading) of measured readings with free product.
        lnapl_levels: list[tuple[float, float, WaterLevel]] = []
        # Holes with an LNAPL band (any series): every head drawn there is
        # hollow, so neither series' marker hides the band.
        product_holes = {
            str(level.hole_id)
            for levels in series_groups.values()
            for level in levels
            if has_lnapl(level) and water_status(level) == "measured"
        }
        # One "i=" per segment: series with equal heads produce identical text.
        gradient_drawn: set[tuple[str, str, str]] = set()
        # Flowing-artesian heads drawn (legend entry only then).
        self.artesian_drawn = False
        self.artesian_color = None
        # connect_group(s) each hole has readings in (any series): a hole read
        # only in another nest (a deep well between shallow wells) is not a
        # gap in this nest's line.
        groups_by_hole: dict[str, set[str]] = {}
        for series_levels in series_groups.values():
            for level in series_levels:
                groups_by_hole.setdefault(str(level.hole_id), set()).add(
                    (level.connect_group or "").strip()
                )
        series_items = sorted(series_groups.items())
        series_styles: list[tuple[str, str, str]] = []
        for series_index, (series_id, levels) in enumerate(series_items):
            first = levels[0]
            default_color, default_marker, default_label = consulting_gw_series_style(
                series_id,
                first.series_label,
                series_index=series_index,
            )
            color = first.color or water_color or default_color or default_water_color
            if consulting or multi_series:
                # Several series: per-series shapes so they read apart.
                raw_marker = first.marker or default_marker or profile_marker
            else:
                raw_marker = first.marker or profile_marker or default_marker
            lowered = str(raw_marker).lower()
            marker = _GW_MARKER_MAP.get(lowered, raw_marker)
            if marker not in MarkerStyle.markers:
                marker = lowered if lowered in MarkerStyle.markers else "v"
            series_styles.append((color, marker, default_label))
        if multi_series:
            # Two series must never share colour + marker (an unknown id can
            # land on a known id's slot): later ones move to a free slot.
            resolved = distinct_series_styles(
                [
                    (color, marker, bool(levels[0].color or water_color), bool(levels[0].marker))
                    for (color, marker, _label), (_sid, levels) in zip(series_styles, series_items, strict=True)
                ],
                lambda slot: consulting_gw_series_style("", series_index=slot)[:2],
            )
            series_styles = [
                (color, marker, label)
                for (color, marker), (_c, _m, label) in zip(resolved, series_styles, strict=True)
            ]
        for series_index, (series_id, levels) in enumerate(series_items):
            first = levels[0]
            plotted_levels: list[WaterLevel] = []
            color, marker, default_label = series_styles[series_index]
            label = first.series_label or default_label or series_id
            # Several series: each i= names its series (colour alone is not
            # enough), in the legend's case (series names are upper case there).
            gradient_prefix = label.upper() if multi_series else ""
            series_linestyle = "-" if self.profile.water_line_solid else "--"
            if multi_series and not consulting:
                series_linestyle = _SERIES_DASHES[series_index % len(_SERIES_DASHES)]
            level_by_hole = {level.hole_id: level for level in levels}
            if label_series_gaps:
                # Fully dry holes: one NM only (skip if label_dry_wells already drew them,
                # or draw once across series when dry-well labeling is off).
                for hole_id in transect_hole_ids:
                    if hole_id in level_by_hole:
                        continue
                    if hole_id in holes_with_any_water:
                        # Read in another series: no row here is not "not
                        # measured" (an nm row says that) - nothing drawn.
                        continue
                    if label_dry_wells or hole_id in fully_dry_nm_drawn:
                        continue
                    fully_dry_nm_drawn.add(hole_id)
                    profile = profile_lookup.get(hole_id)
                    if profile is None:
                        continue
                    x_profile, collar_rl = profile
                    y = self._plot_y(self._hole_elevation_at_depth(hole_id, 1.0, collar_rl), collar_rl)
                    self._water_status_note(ax, hole_id, (float(x_profile), float(y)), "NM")
            # Draw each connect_group nest separately so shallow/deep do not join.
            for group_id, group_levels in _connect_subgroups(levels).items():
                # One reading per hole: latest date wins (undated: last row).
                level_by_id = latest_water_by_hole(group_levels)
                plotted_levels.extend(level_by_id.values())
                # Holes of this nest's line, in section order: a hole read
                # only in another connect_group is skipped (no gap), while a
                # hole with a dry / NM row here, or no reading at all, still
                # breaks a segments-mode line.
                group_hole_ids = [
                    hole_id
                    for hole_id in transect_hole_ids
                    if hole_id in level_by_id
                    or not groups_by_hole.get(hole_id)
                    or group_id in groups_by_hole[hole_id]
                ]
                xs: list[float] = []
                water_rls: list[float] = []
                collars: list[float] = []
                measured_levels: list[WaterLevel] = []
                for hole_id in transect_hole_ids:
                    level = level_by_id.get(hole_id)
                    if level is None:
                        continue
                    profile = profile_lookup.get(hole_id)
                    if profile is None:
                        continue
                    x_profile, collar_rl = profile
                    status = water_status(level)
                    if status in {"dry", "nm"}:
                        y_nm = self._plot_y(
                            self._hole_elevation_at_depth(hole_id, 1.0, collar_rl), collar_rl
                        )
                        self._water_status_note(
                            ax, hole_id, (float(x_profile), float(y_nm)), "NM" if status == "nm" else "DRY"
                        )
                        continue
                    xs.append(x_profile)
                    water_rls.append(self._water_head(level, collar_rl))
                    collars.append(collar_rl)
                    measured_levels.append(level)
                if not xs:
                    continue
                xs_arr = np.asarray(xs, dtype=float)
                water_arr = np.asarray(water_rls, dtype=float)
                collar_arr = np.asarray(collars, dtype=float)
                ys = self._plot_y_values(water_arr, collar_arr)
                if getattr(self, "_deviated_x", None):
                    # Surveyed holes: the marker sits on the column step at its head.
                    xs_arr = np.asarray(
                        [
                            self._deviated_x_at(str(level.hole_id), float(collar), float(y), float(x))
                            for level, collar, y, x in zip(measured_levels, collar_arr, ys, xs_arr, strict=True)
                        ],
                        dtype=float,
                    )
                marker_x = {str(level.hole_id): float(x) for level, x in zip(measured_levels, xs_arr, strict=True)}
                y_by_level = {
                    str(level.hole_id): (float(x), float(y))
                    for level, x, y in zip(measured_levels, xs_arr, ys, strict=True)
                }
                marker_transform = ax.transData
                if multi_series and not consulting:
                    # Coincident heads of different series: fan the markers
                    # out sideways so one series cannot hide another.
                    shift = (series_index - 0.5 * (n_series - 1)) * _SERIES_MARKER_STEP_PT
                    marker_transform = offset_copy(ax.transData, fig=ax.figure, x=shift, units="points")
                # A head on an LNAPL band is drawn hollow (series-colour
                # outline, no fill) so the product band under it stays visible.
                on_product = [str(level.hole_id) in product_holes for level in measured_levels]
                face = to_rgba(color)
                markers = ax.scatter(
                    xs_arr,
                    ys,
                    marker=marker,
                    facecolors=[(0.0, 0.0, 0.0, 0.0) if lnapl else face for lnapl in on_product],
                    s=49,
                    zorder=WATER_MARKER_ZORDER,
                    transform=marker_transform,
                    edgecolors=[face if lnapl else "white" for lnapl in on_product],
                    linewidths=[
                        _LNAPL_HOLLOW_EDGE_PT if lnapl else _WATER_MARKER_EDGE_PT for lnapl in on_product
                    ],
                    clip_on=True,
                )
                # Never past the frame (a level at the collar of an edge hole
                # poked its triangle out over the axis line).
                markers.set_clip_path(ax.patch)
                register_marker_set(ax, markers, "water")
                artesian = water_arr > collar_arr + _ARTESIAN_TOL_M
                artesian_holes = {
                    str(level.hole_id) for level, above in zip(measured_levels, artesian, strict=True) if above
                }
                if np.any(artesian):
                    self._draw_artesian_risers(
                        ax,
                        xs_arr[artesian],
                        self._plot_y_values(collar_arr[artesian], collar_arr[artesian]),
                        ys[artesian],
                        color,
                        marker_transform,
                    )
                lnapl_levels.extend(
                    (float(x), float(collar_rl), level)
                    for x, collar_rl, level in zip(xs_arr, collars, measured_levels, strict=True)
                    if has_lnapl(level)
                )
                if label_elevations:
                    for marker_index, (x_profile, _water_rl, y, level, collar_rl) in enumerate(
                        zip(xs_arr, water_arr, ys, measured_levels, collars, strict=True)
                    ):
                        rl_label = self._water_annotate(
                            ax,
                            self._water_elevation_label(level, collar_rl),
                            (float(x_profile), float(y)),
                            kind="rl",
                            color=color,
                            fontsize=7,
                            xytext=(4, -8),
                        )
                        # The number's parts, for the figure-wide compaction
                        # ladder (unit dropped, one decimal, smaller text).
                        rl_label._water_value, rl_label._water_suffix = self._water_elevation_parts(
                            level, collar_rl
                        )
                        if has_lnapl(level):
                            # Corrected head + the measured product thickness.
                            rl_label._water_note = format_lnapl_thickness_label(level)
                            rl_label.set_text(_water_compact_text(rl_label, 0))
                            rl_label._water_base_text = rl_label.get_text()
                        # Its own marker: a marker nudged off a coincident
                        # one pulls the number to its side.
                        rl_label._water_marker = (markers, marker_index)
                # Depth mode: depths below ground of wells with different
                # collar elevations do not describe one water table; joining
                # them (or an i= between them) would be meaningless.
                depth_join_ok = not depth_mode or float(np.ptp(collar_arr)) <= _DEPTH_MODE_COLLAR_TOL_M
                if len(xs_arr) >= 2 and interpolate and not depth_join_ok:
                    if _DEPTH_MODE_WATER_NOTE not in self.water_line_notes:
                        self.water_line_notes.append(_DEPTH_MODE_WATER_NOTE)
                if len(xs_arr) >= 2 and interpolate and depth_join_ok:
                    gw_linestyle = series_linestyle
                    line_name = series_id if not group_id else f"{series_id} [{group_id}]"
                    clipped_xy = None
                    if clip_ground is not None and not (use_segments and not across_gaps):
                        clipped_xy, clipped_spans, artesian_spans = clip_water_line_to_ground(
                            xs_arr, ys, *clip_ground
                        )
                        hole_pairs = [
                            (str(left.hole_id), str(right.hole_id))
                            for left, right in zip(measured_levels, measured_levels[1:], strict=False)
                        ]
                        self._note_water_line_clip(
                            line_name,
                            [hole_pairs[i] for i in clipped_spans],
                            [hole_pairs[i] for i in artesian_spans],
                            artesian_holes,
                        )
                    if across_gaps:
                        x_dense = np.linspace(float(xs_arr.min()), float(xs_arr.max()), 100)
                        y_dense = np.interp(x_dense, xs_arr, ys)
                        if clipped_xy is not None:
                            x_dense, y_dense = clipped_xy[:, 0], clipped_xy[:, 1]
                        ax.plot(
                            x_dense,
                            y_dense,
                            color=color,
                            linewidth=2.0,
                            linestyle=gw_linestyle,
                            zorder=6,
                        )
                        _register_water_line(ax, x_dense, y_dense)
                        joined_series.add(series_id)
                    elif use_segments:
                        y_by_hole = {
                            level.hole_id: float(y)
                            for level, y in zip(measured_levels, ys, strict=True)
                        }
                        segments: list[np.ndarray] = []
                        for left_id, right_id in zip(
                            group_hole_ids, group_hole_ids[1:], strict=False
                        ):
                            if left_id not in y_by_hole or right_id not in y_by_hole:
                                continue
                            segment = np.asarray(
                                [
                                    [marker_x.get(left_id, transect_x[left_id]), y_by_hole[left_id]],
                                    [marker_x.get(right_id, transect_x[right_id]), y_by_hole[right_id]],
                                ],
                                dtype=float,
                            )
                            if clip_ground is not None and segment[1, 0] > segment[0, 0]:
                                seg_xy, seg_clipped, seg_artesian = clip_water_line_to_ground(
                                    segment[:, 0], segment[:, 1], *clip_ground
                                )
                                pair = (str(left_id), str(right_id))
                                self._note_water_line_clip(
                                    line_name,
                                    [pair] if seg_clipped else [],
                                    [pair] if seg_artesian else [],
                                    artesian_holes,
                                )
                                if seg_xy is not None:
                                    segment = seg_xy
                            segments.append(segment)
                        if segments:
                            collection = LineCollection(
                                segments,
                                colors=color,
                                linewidths=2.0,
                                linestyles=gw_linestyle,
                                zorder=6,
                            )
                            ax.add_collection(collection)
                            joined_series.add(series_id)
                            if self._cad_svg_layers_enabled():
                                self._set_cad_gid(collection, "water")
                            for segment in segments:
                                _register_water_line(ax, segment[:, 0], segment[:, 1])
                    else:
                        line_x, line_y = xs_arr, ys
                        if clipped_xy is not None:
                            line_x, line_y = clipped_xy[:, 0], clipped_xy[:, 1]
                        plotted = ax.plot(
                            line_x,
                            line_y,
                            color=color,
                            linewidth=2.0,
                            linestyle=gw_linestyle,
                            zorder=6,
                        )
                        if self._cad_svg_layers_enabled() and plotted:
                            self._set_cad_gid(plotted[0], "water")
                        _register_water_line(ax, line_x, line_y)
                        joined_series.add(series_id)
                    # Schematic horizontal i = Δh/Δx between adjacent measured heads.
                    gradient_collars = {
                        hid: float(profile_lookup[hid][1])
                        for hid in transect_hole_ids
                        if hid in profile_lookup
                    }
                    gradient_segments = horizontal_gradients_along_profile(
                        measured_levels,
                        hole_order=transect_hole_ids,
                        x_by_hole=transect_x,
                        collar_rl_by_hole=gradient_collars,
                        series_id=series_id,
                        connect_group=group_id,
                        stick_up_by_hole=getattr(self, "stick_up_by_hole", None),
                        md_to_elevation_by_hole=self._md_to_elevation_by_hole(gradient_collars),
                    )
                    adjacent_pairs = set(
                        zip(group_hole_ids, group_hole_ids[1:], strict=False)
                    )
                    for segment in gradient_segments:
                        if (
                            use_segments
                            and not across_gaps
                            and (segment.left_hole_id, segment.right_hole_id)
                            not in adjacent_pairs
                        ):
                            # Segments mode draws no water line across a dry/NM
                            # gap — do not float an i= label over the open gap.
                            continue
                        gradient_key = (
                            str(segment.left_hole_id),
                            str(segment.right_hole_id),
                            format_gradient_text(segment.gradient),
                        )
                        gradient_text = format_gradient_text(segment.gradient, gradient_prefix)
                        if gradient_key in gradient_drawn:
                            continue
                        gradient_drawn.add(gradient_key)
                        mid_collar = 0.5 * (
                            float(profile_lookup[segment.left_hole_id][1])
                            + float(profile_lookup[segment.right_hole_id][1])
                        )
                        mid_y = self._plot_y(segment.mid_head_masl, mid_collar)
                        self.gradient_labels_drawn = True
                        gradient_label = self._water_annotate(
                            ax,
                            gradient_text,
                            (segment.mid_x, float(mid_y)),
                            kind="gradient",
                            color=color,
                            fontsize=6,
                            xytext=(0, 6),
                            ha="center",
                        )
                        left_end = y_by_level.get(str(segment.left_hole_id))
                        right_end = y_by_level.get(str(segment.right_hole_id))
                        if left_end is not None and right_end is not None:
                            # Its own segment: the label sits along it.
                            gradient_label._gradient_segment = (left_end, right_end)
            # An unlabelled default series has no name worth printing: the
            # caption falls back to its datum ("GROUNDWATER LEVEL (masl)").
            caption_label = (
                "" if series_id == "default" and not first.series_label and not default_label
                else label
            )
            if not first.series_label:
                # Unlabelled series from one monitoring event: name the date
                # ("GROUNDWATER LEVEL (2026-04-02)"). Undated: unchanged.
                # Dated from the readings plotted (latest per hole and nest):
                # a superseded duplicate does not widen the range.
                legend_date = series_legend_date(plotted_levels or levels)
                if legend_date:
                    base = "" if series_id == "default" else (default_label or series_id)
                    caption_label = f"{base}, {legend_date}" if base else legend_date
            level_label_text, elevation_label_text = self._water_legend_captions(
                series_id, caption_label, default_label
            )
            self.water_series_legend.append(
                {
                    "series_id": series_id,
                    "color": color,
                    "marker": marker,
                    "linestyle": series_linestyle if series_id in joined_series else "none",
                    "level_label": level_label_text,
                    "elevation_label": elevation_label_text,
                }
            )
        if lnapl_levels:
            self._draw_lnapl_bands(ax, lnapl_levels, track_half)
        key_min = int(getattr(self.profile, "water_series_key_min_series", 0) or 0)
        # A connect_group nest series ("June 2025 deep") keys its marker even
        # when it is the only series left (its parent "June 2025" was
        # deselected): the plain-text fallback shows no marker, and only the
        # marker tells which nest the heads are. A plain single series keeps
        # the text line.
        nested = any(
            (level.connect_group or "").strip() for levels in series_groups.values() for level in levels
        )
        if not consulting and key_min and (len(self.water_series_legend) >= key_min or nested):
            self._draw_water_series_key(ax)

    def _draw_artesian_risers(self, ax, xs, ground_ys, head_ys, color: str, transform) -> None:
        """Flowing artesian heads: a dashed riser from the ground line up to
        the head (the marker floats above ground otherwise) and an upward
        triangle on the ground line; keyed only when drawn."""
        risers = [
            np.asarray([[float(x), float(g)], [float(x), float(h)]], dtype=float)
            for x, g, h in zip(xs, ground_ys, head_ys, strict=True)
        ]
        collection = LineCollection(
            risers,
            colors=color,
            linewidths=1.0,
            linestyles=[_ARTESIAN_LINESTYLE],
            zorder=WATER_MARKER_ZORDER - 0.05,
            transform=transform,
        )
        ax.add_collection(collection)
        flags = ax.scatter(
            xs,
            ground_ys,
            marker=_ARTESIAN_MARKER,
            c=color,
            s=22,
            zorder=WATER_MARKER_ZORDER - 0.02,
            transform=transform,
            edgecolors="white",
            linewidths=_WATER_MARKER_EDGE_PT,
        )
        flags.set_gid("artesian_head")
        collection.set_gid("artesian_riser")
        self.artesian_drawn = True
        if self.artesian_color is None:
            self.artesian_color = color

    def artesian_legend_proxy(self):
        """Legend key for flowing artesian heads (``None`` when none drawn)."""
        if not getattr(self, "artesian_drawn", False):
            return None
        from matplotlib.lines import Line2D

        return Line2D(
            [],
            [],
            color=self.artesian_color or CONSULTING_WATER_COLOR,
            marker=_ARTESIAN_MARKER,
            markersize=6,
            linewidth=1.0,
            linestyle=_ARTESIAN_LINESTYLE,
            label=ARTESIAN_LEGEND_LABEL,
        )

    def _draw_lnapl_bands(
        self,
        ax,
        lnapl_levels: Sequence[tuple[float, float, WaterLevel]],
        track_half: float | None,
    ) -> None:
        """Hatched product band in the well column (product top → oil/water
        interface) and the QA note naming the corrected wells.

        The band shows the apparent (measured) thickness at the measured
        interface; the water marker sits at the corrected head inside it.
        """
        if track_half is None:
            xs = np.sort(np.asarray([item[0] for item in lnapl_levels], dtype=float))
            track_half = self._track_half_width(xs, x_sorted=True)
        half = float(track_half) * 0.92
        stick_ups = getattr(self, "stick_up_by_hole", None) or {}
        drawn: list[WaterLevel] = []
        seen: set[tuple[str, float, float]] = set()
        x_left: list[float] = []
        y_low: list[float] = []
        heights: list[float] = []
        for x_profile, collar_rl, level in lnapl_levels:
            interval = lnapl_interval_masl(
                level,
                collar_rl,
                stick_up_m=stick_ups.get(str(level.hole_id)),
                md_to_elevation=self._md_to_elevation(level.hole_id, collar_rl),
            )
            if interval is None:
                continue
            drawn.append(level)
            key = (str(level.hole_id), round(interval[0], 6), round(interval[1], 6))
            if key in seen:
                continue  # same band from another series / nest
            seen.add(key)
            top_y, bottom_y = self._plot_y_values(
                np.asarray(interval, dtype=float), np.full(2, float(collar_rl))
            )
            height = abs(float(top_y) - float(bottom_y))
            if height <= 1e-9:
                continue
            if getattr(self, "_deviated_x", None):
                # Surveyed holes: the band sits on the trace at its depth.
                x_profile = self._deviated_x_at(
                    str(level.hole_id), float(collar_rl), 0.5 * (float(top_y) + float(bottom_y)), float(x_profile)
                )
            x_left.append(float(x_profile) - half)
            y_low.append(min(float(top_y), float(bottom_y)))
            heights.append(height)
        if x_left:
            x0 = np.asarray(x_left, dtype=float)
            y0 = np.asarray(y_low, dtype=float)
            # A band of a few cm is a hairline at report scale and the water
            # marker (at the corrected head, inside it) covered it: grown at
            # draw time to a minimum page size, wider than the marker.
            band = MinSizeBandCollection(
                np.column_stack([x0, y0, x0 + 2.0 * half, y0 + np.asarray(heights, dtype=float)]),
                min_width_pt=_LNAPL_MIN_WIDTH_PT,
                min_height_pt=_LNAPL_MIN_HEIGHT_PT,
                facecolors=LNAPL_FILL_COLOR,
                edgecolors=LNAPL_EDGE_COLOR,
                linewidths=0.6,
                hatch=LNAPL_HATCH,
            )
            # Over the column fills and screen hatch (9), under the water
            # markers (9.3) and labels (9.5).
            band.set_zorder(9.1)
            band.set_gid("lnapl_band")
            ax.add_collection(band)
            self.lnapl_drawn = True
        note = lnapl_correction_note(drawn)
        if note and note not in self.water_line_notes:
            self.water_line_notes.append(note)

    def lnapl_legend_proxy(self):
        """Legend swatch for the LNAPL band (``None`` when no band was drawn)."""
        if not getattr(self, "lnapl_drawn", False):
            return None
        from matplotlib.patches import Patch

        return Patch(
            facecolor=LNAPL_FILL_COLOR,
            edgecolor=LNAPL_EDGE_COLOR,
            hatch=LNAPL_HATCH,
            linewidth=0.6,
            label=LNAPL_LEGEND_LABEL,
        )

    def _draw_water_series_key(self, ax) -> None:
        """Marker + line + label per GW series, as a small framed key."""
        from matplotlib.legend import Legend
        from matplotlib.lines import Line2D

        from renderer_common import LEGEND_KEY_HANDLER_MAP, SampleTextKey, remember_legend_keys

        handles = [
            Line2D(
                [],
                [],
                color=entry["color"],
                marker=entry["marker"],
                markersize=6,
                linewidth=1.6,
                linestyle=entry.get("linestyle", "--"),
            )
            for entry in self.water_series_legend
        ]
        labels = [entry["level_label"] for entry in self.water_series_legend]
        lnapl_proxy = self.lnapl_legend_proxy()
        if lnapl_proxy is not None:
            handles.append(lnapl_proxy)
            labels.append(LNAPL_LEGEND_LABEL)
        artesian_proxy = self.artesian_legend_proxy()
        if artesian_proxy is not None:
            handles.append(artesian_proxy)
            labels.append(ARTESIAN_LEGEND_LABEL)
        if getattr(self, "gradient_labels_drawn", False):
            handles.append(SampleTextKey(GRADIENT_KEY_SAMPLE, self.water_series_legend[0]["color"]))
            gradient_label = GRADIENT_LEGEND_LABEL.split(" = ", 1)[1]
            labels.append(gradient_label.upper() if self._legend_upper_case() else gradient_label)
        key = Legend(
            ax,
            handles,
            labels,
            handler_map=LEGEND_KEY_HANDLER_MAP,
            loc="lower left",
            fontsize=7,
            frameon=True,
            framealpha=0.9,
            handlelength=3.0,
            borderaxespad=0.6,
        )
        remember_legend_keys(key, handles, labels)
        key.set_zorder(20)
        key.set_gid("water_series_key")
        ax.add_artist(key)
        self._water_series_key_ax = ax

    def _draw_compact_water_legend(self, ax) -> None:
        if not self.water_series_legend:
            return
        if getattr(self, "_water_series_key_ax", None) is ax:
            return  # the series key already names every series
        lines = []
        for entry in self.water_series_legend:
            lines.append(entry["level_label"])
        upper = self._legend_upper_case()
        if getattr(self, "lnapl_drawn", False):
            lines.append(LNAPL_LEGEND_LABEL.upper() if upper else LNAPL_LEGEND_LABEL)
        if getattr(self, "artesian_drawn", False):
            lines.append(ARTESIAN_LEGEND_LABEL.upper() if upper else ARTESIAN_LEGEND_LABEL)
        if getattr(self, "gradient_labels_drawn", False):
            lines.append(GRADIENT_LEGEND_LABEL)
        ax.text(
            0.01,
            0.01,
            " | ".join(lines),
            transform=ax.transAxes,
            fontsize=7,
            color=LABEL_COLOR,
            va="bottom",
            ha="left",
            zorder=20,
        )
