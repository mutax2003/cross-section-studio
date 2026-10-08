"""Groundwater / water-table drawing mixin for CrossSectionRenderer."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TypedDict

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.markers import MarkerStyle
from matplotlib.patches import ArrowStyle, ConnectionStyle, Rectangle
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox, offset_copy

from hydro_metrics import (
    format_gradient_label,
    horizontal_gradients_along_profile,
    water_head_masl,
    water_status,
)
from models import WaterLevel
from render_theme import (
    CONSULTING_NM_COLOR,
    CONSULTING_WATER_COLOR,
    LABEL_COLOR,
    WATER_COLOR,
    consulting_gw_series_style,
)
from renderer_common import settle_figure_layout

logger = logging.getLogger(__name__)

_GW_MARKER_MAP = {
    "circle": "o",
    "triangle": "v",
    "diamond": "D",
    "plus": "P",
    "x": "x",
}


# Candidate label offsets (points) as (dx, dy, ha), sorted nearest-first from
# the default spot so a label takes the closest free position in any direction.
def _nearest_first(
    base: tuple[float, float],
    offsets: Sequence[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    return sorted(offsets, key=lambda item: ((item[0] - base[0]) ** 2 + (item[1] - base[1]) ** 2))


_DY_GRID = (0.0, -8.0, 8.0, -16.0, 16.0, -26.0, 26.0, -38.0, 38.0, -52.0, 52.0)
_SIDE_OFFSETS = [
    (dx, dy, "left" if dx > 0 else "right")
    for dx in (5.0, -5.0, 22.0, -22.0, 40.0, -40.0, 60.0, -60.0)
    for dy in _DY_GRID
]
_LABEL_CANDIDATES: dict[str, list[tuple[float, float, str]]] = {
    "rl": _nearest_first((5.0, -8.0), _SIDE_OFFSETS),
    "nm": _nearest_first((5.0, 0.0), _SIDE_OFFSETS),
    "gradient": _nearest_first(
        (0.0, 6.0),
        [(dx, dy, "center") for dx in (0.0, 20.0, -20.0, 40.0, -40.0, 60.0, -60.0) for dy in _DY_GRID],
    ),
}
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
_CHEM_STRIP_PAD_PT = 2.5
# Values starting within this many points of a hole's left-most value form
# its main column (one strip); the rest get their own knock-outs.
_CHEM_STRIP_COLUMN_TOL_PT = 8.0
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
# Water labels avoid the chemistry strips (knock-outs beside labelled
# columns) whenever a clear spot exists; overlapping one costs like a label
# collision but never drops the water number.
_STRIP_OBSTACLE_WEIGHT = 1.0
# "strip" style: a strip wider than this share of the gap to the next column
# hides the geology between the holes; that hole's values get solid boxes
# ("box" style) instead.
_CHEM_STRIP_MAX_GAP_SHARE = 0.45
# A value never drifts past this share of the gap to the next column (it
# would read as belonging to the next hole, or to none) unless the value
# itself needs more room than that.
_CHEM_MAX_GAP_DRIFT_SHARE = 0.5
# Re-runs that settle one font size per hole (no tiny stragglers).
_CHEM_FONT_SETTLE_PASSES = 3
# Water numbers that still collide after every candidate are dropped (marker
# stays) — overlapping digits misreport a level. Chemistry values use their
# own hard-constraint pass (_place_chem_label) and are dropped as a last resort.
_DROP_ON_COLLISION = frozenset({"rl", "nm", "gradient"})
# Clearance around each label box (points) so neighbours never touch.
_LABEL_PAD_PT = 2.0
# Sideways nudge (points) between coincident markers of different series on
# non-consulting sheets, so one series cannot hide another.
_SERIES_MARKER_STEP_PT = 7.5
# Line dash per series slot on non-consulting sheets (consulting keeps its
# solid client lines; marker shape separates series there).
_SERIES_DASHES = ("--", "-.", ":", (0, (6, 2, 1, 2)))


def _fmt_water_number(value: float) -> str:
    """Water level / depth text: at most 2 decimals, trailing zeros stripped
    (745.29, 745.3, 745), like the chemistry values."""
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


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


def _chem_footprint(annotation, renderer):
    """Display box of a chemistry label: text plus its bbox patch and colour dot.

    Memoised per label on every input the measurement reads (text, font,
    alignment, offset, anchor in display space, dpi, visibility): placement
    re-runs (compact escalation, frame refits) re-measure the same spots many
    times, and each measurement lays the text out several times. A hit returns
    a copy of the exact box a fresh measurement would give.
    """
    anchor = annotation.axes.transData.transform([annotation.xy])[0]
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
        anchor_x, anchor_y = annotation.axes.transData.transform([annotation.xy])[0]
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
    anchor_x = annotation.axes.transData.transform([annotation.xy])[0][0]
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


# Solid box a "strip" value takes when its hole is too close to the next one
# for a strip (matches the "box" label style).
_CHEM_FALLBACK_BBOX = {
    "boxstyle": "square,pad=0.18",
    "facecolor": "white",
    "edgecolor": "none",
    "alpha": 1.0,
}


def _set_chem_box_fallback(annotation, on: bool) -> None:
    """Give a strip-style value a solid box (or take it away again)."""
    if bool(getattr(annotation, "_chem_box_fallback", False)) == on:
        return
    annotation._chem_box_fallback = on
    annotation.set_bbox(dict(_CHEM_FALLBACK_BBOX) if on else None)


def _set_chem_fontsize(annotation, size: float) -> None:
    annotation.set_fontsize(size)
    halo = getattr(annotation, "_halo", None)
    if halo is not None:
        halo.set_fontsize(size)


def _place_chem_label(
    annotation,
    renderer,
    fig,
    *,
    placed_arr,
    column_arr,
    own,
    pad,
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
        idx = np.flatnonzero(ok)
        if idx.size:
            idx = idx[~column_arr.overlaps_many(boxes[idx])]
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
    clears other labels (each criterion only while some candidate meets it)."""
    cand = boxes[idx]
    line_hit = np.zeros(idx.size, dtype=bool)
    if line_segs is not None and line_segs.shape[0]:
        grown = cand + np.array([-line_clear, -line_clear, line_clear, line_clear])
        line_hit = _segment_box_hits(line_segs, grown).any(axis=0)
    leader_hits = [np.zeros(idx.size, dtype=bool) for _arr in label_arrs]
    shift = np.maximum(np.maximum(ddx[idx], 0.0), np.abs(ddy[idx]))
    leadered = shift > max(label_height_pt, _CHEM_MIN_LEADER_SHIFT_PT)
    if check_leaders and leadered.any() and label_arrs[0].shape[0]:
        ax0, ay0 = anchor
        ends = np.column_stack((cand[:, 0], 0.5 * (cand[:, 1] + cand[:, 3])))
        vec = ends - np.array([ax0, ay0])
        length = np.hypot(vec[:, 0], vec[:, 1])
        # The drawn leader stops 2 pt short of its reading (shrinkB).
        trim = np.clip(2.0 * px_per_pt / np.maximum(length, 1e-9), 0.0, 1.0)[:, None]
        starts = np.array([ax0, ay0]) + vec * trim
        segs = np.column_stack((starts, ends))[leadered]
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
            leader_hit[np.flatnonzero(leadered)] = _segment_box_hits(segs, near).any(axis=1)
    tiers = [~line_hit & ~leader_hit for leader_hit in leader_hits] + [~line_hit]
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
    x, y = annotation.axes.transData.transform([annotation.xy])[0]
    return float(x), float(y)


def _chem_hole_key(annotation) -> float:
    """Chemistry labels of one hole share the anchor x (its column's right edge)."""
    return round(_chem_anchor(annotation)[0], 1)


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


_LEADER_THRESHOLD_PT = 14.0


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
    return Text.get_window_extent(annotation, renderer)


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
    return Bbox.union([Text.get_window_extent(annotation, renderer), segment])


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
        if header_bands is not None and isinstance(artist, Annotation):
            bound = _straight_leader_bound(artist, renderer)
            if bound is not None:
                y0, y1 = bound.y0 - pad, bound.y1 + pad
                if not any(y0 < top and y1 > bottom for bottom, top in header_bands):
                    continue
        boxes.append(artist.get_window_extent(renderer).padded(pad))
    return boxes

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


class RendererWaterMixin:
    """Water-table markers, polylines, and compact GW legend."""

    def _water_elevation_label(self, level: WaterLevel, collar_rl: float) -> str:
        """Annotate water as RL (masl) in elevation mode, or depth (mbgs) in relative mode."""
        if self.profile.y_axis_mode == "depth_below_collar":
            return f"{_fmt_water_number(level.depth)} mbgs"
        water_rl = (
            float(level.elevation_masl)
            if level.elevation_masl is not None
            else collar_rl - level.depth
        )
        if self.profile.layout == "consulting_section":
            return _fmt_water_number(water_rl)
        return f"{_fmt_water_number(water_rl)} m"

    def _water_legend_captions(
        self,
        series_id: str,
        label: str,
        default_label: str,
    ) -> tuple[str, str]:
        relative = self.profile.y_axis_mode == "depth_below_collar"
        datum = "mbgs" if relative else "masl"
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
            zorder=9,
            bbox={"boxstyle": "square,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.85},
            # Leader exists from the start but stays hidden unless the label
            # is moved away from its point by the collision pass.
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.5, "shrinkA": 0, "shrinkB": 2},
        )
        annotation.arrow_patch.set_visible(False)
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
        for attempt in range(_CHEM_FRAME_FIT_ATTEMPTS + 1):
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
        notes = _chem_drop_notes(labels)
        if notes != getattr(self, "chemistry_label_notes", None):
            for note in notes:
                logger.warning(note)
        self.chemistry_label_notes = notes

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
        for _settle in range(_CHEM_FONT_SETTLE_PASSES + 1):
            while True:
                dropped_holes = self._label_collision_pass(
                    fig, labels, renderer, compact, font_caps=font_caps
                )
                escalate = {
                    hole for hole in dropped_holes if compact.get(hole, 0) < _CHEM_MAX_COMPACT_LEVEL
                }
                if not escalate:
                    break
                for hole in escalate:
                    compact[hole] = compact.get(hole, 0) + 1
            mixed = _chem_mixed_size_holes(labels)
            mixed = {
                hole: scale
                for hole, scale in mixed.items()
                if scale < font_caps.get(hole, 1.0) - 1e-9
            }
            if not mixed or _settle == _CHEM_FONT_SETTLE_PASSES:
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
        """Where the "strip" knock-outs go: ``(rects, boxed)``.

        ``rects`` are ``(ax, x0, y0, x1, y1)`` display rectangles; ``boxed``
        lists the values of holes too close to their right-hand neighbour for
        a strip (wider than ``_CHEM_STRIP_MAX_GAP_SHARE`` of the gap): those
        take solid boxes instead, so the geology between the holes shows.
        """
        if str(getattr(self.profile, "chemistry_label_style", "") or "") != "strip":
            return [], []
        if column_boxes is None:
            column_boxes = self._column_obstacle_boxes(fig)
        if not column_boxes:
            return [], []
        pad = renderer.points_to_pixels(_CHEM_STRIP_PAD_PT)
        tol = renderer.points_to_pixels(_CHEM_STRIP_COLUMN_TOL_PT)
        holes: dict[tuple[int, float], list] = {}
        for kind, annotation, _color in labels:
            if kind == "chem" and annotation.get_visible():
                key = (id(annotation.axes), _chem_hole_key(annotation))
                holes.setdefault(key, []).append(annotation)
        result: list[tuple[object, float, float, float, float]] = []
        boxed: list = []
        for hole_labels in holes.values():
            ax = hole_labels[0].axes
            own = _own_column(hole_labels[0], column_boxes)
            if own is None:
                continue
            if any(getattr(annotation, "_chem_box_fallback", False) for annotation in hole_labels):
                boxed.extend(hole_labels)
                continue
            boxes = [_chem_footprint(annotation, renderer) for annotation in hole_labels]
            frame = ax.get_window_extent(renderer)
            # Never over the next column to the right, nor past the frame.
            next_x0 = [col.x0 for col in column_boxes if col is not own and col.x0 > own.x1]
            right_limit = min([frame.x1] + next_x0)
            # The strip spans the hole's main value column; a value nudged
            # further right (around a water label, or a crowded hole's
            # zig-zag column) gets its own knock-out instead of widening the
            # whole strip.
            first_x0 = min(box.x0 for box in boxes)
            main = [box for box in boxes if box.x0 <= first_x0 + tol]
            main_x1 = max(box.x1 for box in main) + pad
            if next_x0 and main_x1 - own.x1 > _CHEM_STRIP_MAX_GAP_SHARE * (min(next_x0) - own.x1):
                boxed.extend(hole_labels)
                continue
            rects = [
                (
                    own.x1,
                    min(box.y0 for box in main) - pad,
                    max(box.x1 for box in main) + pad,
                    max(box.y1 for box in main) + pad,
                )
            ]
            rects += [
                (box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad)
                for box in boxes
                if box.x0 > first_x0 + tol
            ]
            for x0, y0, x1, y1 in rects:
                x0 = max(x0, own.x1)
                x1 = min(x1, right_limit)
                y0 = max(y0, frame.y0)
                # Never above the ground line: the column box starts at the collar.
                y1 = min(y1, frame.y1, own.y1)
                if x1 > x0 and y1 > y0:
                    result.append((ax, x0, y0, x1, y1))
        return result, boxed

    def _pick_chem_box_fallback(self, labels, renderer, column_boxes) -> None:
        """Before placement: box the values of holes too close to the next one for a strip.

        A hole's strip is at least its widest value (full size) plus the
        column clearance and strip pad wide; when that exceeds
        ``_CHEM_STRIP_MAX_GAP_SHARE`` of the gap to the next column the strip
        would hide the geology between the holes, so its values take solid
        boxes, measured with the box from the start.
        """
        if str(getattr(self.profile, "chemistry_label_style", "") or "") != "strip":
            return
        if not column_boxes:
            return
        extra = renderer.points_to_pixels(_CHEM_COLUMN_GAP_PT + _CHEM_STRIP_PAD_PT)
        holes: dict[tuple[int, float], list] = {}
        for kind, annotation, _color in labels:
            if kind == "chem":
                holes.setdefault((id(annotation.axes), _chem_hole_key(annotation)), []).append(annotation)
        for hole_labels in holes.values():
            own = _own_column(hole_labels[0], column_boxes)
            if own is None:
                continue
            next_x0 = [col.x0 for col in column_boxes if col is not own and col.x0 > own.x1]
            if not next_x0:
                continue
            gap = min(next_x0) - own.x1
            widest = max(
                _chem_min_width(annotation, renderer) / _CHEM_MIN_FONT_SCALE for annotation in hole_labels
            )
            if widest + extra > _CHEM_STRIP_MAX_GAP_SHARE * gap:
                for annotation in hole_labels:
                    _set_chem_box_fallback(annotation, True)

    def _draw_chem_label_strips(self, fig, labels, renderer) -> None:
        """Knock a background strip out of the fills beside each labelled column.

        "strip" style: values read on a clean strip immediately right of their
        column (widest placed value + pad, spanning the hole's placed values)
        with the hatching resuming beyond it. The strip sits above lithology
        fills and contacts but below columns, water lines and markers, and is
        clipped so it never reaches another column or leaves the frame. A hole
        too close to the next one for a strip gets boxed values instead.
        """
        # Strips belong to their figure: a second render() on this renderer
        # must not strip the knock-outs off an earlier, still-open figure.
        for strip in getattr(fig, "_chem_strips", ()):
            strip.remove()
        fig._chem_strips = []
        self._chem_strips = fig._chem_strips
        rects, boxed = self._chem_strip_layout(fig, labels, renderer)
        for annotation in boxed:
            _set_chem_box_fallback(annotation, True)
        for ax, x0, y0, x1, y1 in rects:
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
        # A re-run (page resize) starts from scratch: restore labels a previous
        # pass dropped so they get another chance at the new size, and take
        # off the boxes a too-narrow strip swapped in (re-decided afterwards).
        for _kind, annotation, _color in labels:
            if getattr(annotation, "_water_dropped", False):
                annotation._water_dropped = False
                _set_label_visible(annotation, True)
            _set_chem_box_fallback(annotation, False)
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
        for ax in fig.axes:
            if not ax.get_visible():
                continue
            for tick in _drawn_tick_labels(ax.xaxis, ax.get_xlim()) + _drawn_tick_labels(
                ax.yaxis, ax.get_ylim()
            ):
                if tick.get_visible() and tick.get_text().strip():
                    placed.append(tick.get_window_extent(renderer).padded(pad))
        # Borehole columns are obstacles for value labels: a label over a
        # column hides the stick/markers, and over a neighbour's column it
        # misattributes the value.
        column_boxes = self._column_obstacle_boxes(fig)
        self._pick_chem_box_fallback(labels, renderer, column_boxes)
        ordered = _chem_depth_ordered(sorted(labels, key=lambda item: _LABEL_PRIORITY.get(item[0], 9)))
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
        column_arr = _ObstacleArray(column_boxes)
        # Strip knock-outs of the placed values (chemistry goes first), built
        # once the first water label comes up: water numbers and DRY / NM
        # notes keep off them whenever they can.
        strip_arr: _ObstacleArray | None = None
        for kind, annotation, _color in ordered:
            frame = annotation.axes.get_window_extent(renderer)
            if kind != "chem" and strip_arr is None:
                rects, _boxed = self._chem_strip_layout(fig, labels, renderer, column_boxes)
                strip_arr = _ObstacleArray(
                    Bbox.from_extents(x0, y0, x1, y1) for _ax, x0, y0, x1, y1 in rects
                )
            if not hasattr(annotation, "_water_base_xyann"):
                annotation._water_base_xyann = tuple(annotation.xyann)
            base = annotation._water_base_xyann
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
            best: tuple[float, tuple[float, float, str]] | None = None
            for index, (dx, dy, ha) in enumerate(_LABEL_CANDIDATES[kind]):
                annotation.xyann = (dx, dy)
                annotation.set_horizontalalignment(ha)
                box = _text_box(annotation, renderer).padded(pad)
                collision = placed_arr.overlap(box)
                if kind == "nm":
                    # A DRY / NM note anchors on the column centre: keep it
                    # off the column so it reads beside its hole.
                    collision += column_arr.overlap(box)
                if strip_arr is not None:
                    collision += _STRIP_OBSTACLE_WEIGHT * strip_arr.overlap(box)
                # Off the axes is bad; off the page is worse (it is cut off).
                outside = _outside_area(box, frame) + 4.0 * _outside_area(box, fig.bbox)
                score = (collision + 4.0 * outside) * 1000.0 + index
                if best is None or score < best[0]:
                    best = (score, (dx, dy, ha))
                if collision == 0.0 and outside == 0.0:
                    break
            assert best is not None
            dx, dy, ha = best[1]
            annotation.xyann = (dx, dy)
            annotation.set_horizontalalignment(ha)
            _sync_label_companions(annotation, fig, dx, dy, ha)
            final_box = _text_box(annotation, renderer).padded(pad)
            if kind in _DROP_ON_COLLISION and placed_arr.overlap(final_box) > 0.0:
                # No free spot: keep the marker, never print overlapping text.
                annotation._water_dropped = True
                annotation.arrow_patch.set_visible(False)
                _set_label_visible(annotation, False)
                continue
            placed.append(final_box)
            placed_arr.add(final_box)
            moved = abs(dy - base[1]) > _LEADER_THRESHOLD_PT or abs(dx - base[0]) > _LEADER_THRESHOLD_PT
            annotation.arrow_patch.set_visible(moved and getattr(annotation, "_leader_allowed", True))
        return dropped_holes

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
    ) -> None:
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
                dry_y = self._plot_y_values(dry_collars - 1.0, dry_collars)
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
        n_series = len(series_groups)
        multi_series = n_series >= 2
        # One "i=" per segment: series with equal heads produce identical text.
        gradient_drawn: set[tuple[str, str, str]] = set()
        for series_index, (series_id, levels) in enumerate(sorted(series_groups.items())):
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
            label = first.series_label or default_label or series_id
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
                    if hole_id not in holes_with_any_water:
                        if label_dry_wells or hole_id in fully_dry_nm_drawn:
                            continue
                        fully_dry_nm_drawn.add(hole_id)
                    profile = profile_lookup.get(hole_id)
                    if profile is None:
                        continue
                    x_profile, collar_rl = profile
                    y = self._plot_y(collar_rl - 1.0, collar_rl)
                    self._water_status_note(ax, hole_id, (float(x_profile), float(y)), "NM")
            # Draw each connect_group nest separately so shallow/deep do not join.
            for group_id, group_levels in _connect_subgroups(levels).items():
                level_by_id = {item.hole_id: item for item in group_levels}
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
                        y_nm = self._plot_y(collar_rl - 1.0, collar_rl)
                        self._water_status_note(
                            ax, hole_id, (float(x_profile), float(y_nm)), "NM" if status == "nm" else "DRY"
                        )
                        continue
                    xs.append(x_profile)
                    water_rls.append(water_head_masl(level, collar_rl))
                    collars.append(collar_rl)
                    measured_levels.append(level)
                if not xs:
                    continue
                xs_arr = np.asarray(xs, dtype=float)
                water_arr = np.asarray(water_rls, dtype=float)
                collar_arr = np.asarray(collars, dtype=float)
                ys = self._plot_y_values(water_arr, collar_arr)
                marker_transform = ax.transData
                if multi_series and not consulting:
                    # Coincident heads of different series: fan the markers
                    # out sideways so one series cannot hide another.
                    shift = (series_index - 0.5 * (n_series - 1)) * _SERIES_MARKER_STEP_PT
                    marker_transform = offset_copy(ax.transData, fig=ax.figure, x=shift, units="points")
                ax.scatter(
                    xs_arr, ys, marker=marker, c=color, s=49, zorder=7, transform=marker_transform
                )
                if label_elevations:
                    for x_profile, water_rl, y, level, collar_rl in zip(
                        xs_arr, water_arr, ys, measured_levels, collars, strict=True
                    ):
                        self._water_annotate(
                            ax,
                            self._water_elevation_label(level, collar_rl),
                            (float(x_profile), float(y)),
                            kind="rl",
                            color=color,
                            fontsize=7,
                            xytext=(4, -8),
                        )
                if len(xs_arr) >= 2 and interpolate:
                    gw_linestyle = series_linestyle
                    if across_gaps:
                        x_dense = np.linspace(float(xs_arr.min()), float(xs_arr.max()), 100)
                        y_dense = np.interp(x_dense, xs_arr, ys)
                        ax.plot(
                            x_dense,
                            y_dense,
                            color=color,
                            linewidth=2.0,
                            linestyle=gw_linestyle,
                            zorder=6,
                        )
                        _register_water_line(ax, x_dense, y_dense)
                    elif use_segments:
                        y_by_hole = {
                            level.hole_id: float(y)
                            for level, y in zip(measured_levels, ys, strict=True)
                        }
                        segments: list[np.ndarray] = []
                        for left_id, right_id in zip(
                            transect_hole_ids, transect_hole_ids[1:], strict=False
                        ):
                            if left_id not in y_by_hole or right_id not in y_by_hole:
                                continue
                            segments.append(
                                np.asarray(
                                    [
                                        [transect_x[left_id], y_by_hole[left_id]],
                                        [transect_x[right_id], y_by_hole[right_id]],
                                    ],
                                    dtype=float,
                                )
                            )
                        if segments:
                            collection = LineCollection(
                                segments,
                                colors=color,
                                linewidths=2.0,
                                linestyles=gw_linestyle,
                                zorder=6,
                            )
                            ax.add_collection(collection)
                            if self._cad_svg_layers_enabled():
                                self._set_cad_gid(collection, "water")
                            for segment in segments:
                                _register_water_line(ax, segment[:, 0], segment[:, 1])
                    else:
                        plotted = ax.plot(
                            xs_arr,
                            ys,
                            color=color,
                            linewidth=2.0,
                            linestyle=gw_linestyle,
                            zorder=6,
                        )
                        if self._cad_svg_layers_enabled() and plotted:
                            self._set_cad_gid(plotted[0], "water")
                        _register_water_line(ax, xs_arr, ys)
                    # Schematic horizontal i = Δh/Δx between adjacent measured heads.
                    gradient_segments = horizontal_gradients_along_profile(
                        measured_levels,
                        hole_order=transect_hole_ids,
                        x_by_hole=transect_x,
                        collar_rl_by_hole={
                            hid: float(profile_lookup[hid][1])
                            for hid in transect_hole_ids
                            if hid in profile_lookup
                        },
                        series_id=series_id,
                        connect_group=group_id,
                    )
                    adjacent_pairs = set(
                        zip(transect_hole_ids, transect_hole_ids[1:], strict=False)
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
                        gradient_text = format_gradient_label(segment)
                        gradient_key = (
                            str(segment.left_hole_id),
                            str(segment.right_hole_id),
                            gradient_text,
                        )
                        if gradient_key in gradient_drawn:
                            continue
                        gradient_drawn.add(gradient_key)
                        mid_collar = 0.5 * (
                            float(profile_lookup[segment.left_hole_id][1])
                            + float(profile_lookup[segment.right_hole_id][1])
                        )
                        mid_y = self._plot_y(segment.mid_head_masl, mid_collar)
                        self._water_annotate(
                            ax,
                            gradient_text,
                            (segment.mid_x, float(mid_y)),
                            kind="gradient",
                            color=color,
                            fontsize=6,
                            xytext=(0, 6),
                            ha="center",
                        )
            level_label_text, elevation_label_text = self._water_legend_captions(
                series_id, label, default_label
            )
            self.water_series_legend.append(
                {
                    "series_id": series_id,
                    "color": color,
                    "marker": marker,
                    "linestyle": series_linestyle,
                    "level_label": level_label_text,
                    "elevation_label": elevation_label_text,
                }
            )
        key_min = int(getattr(self.profile, "water_series_key_min_series", 0) or 0)
        if not consulting and key_min and len(self.water_series_legend) >= key_min:
            self._draw_water_series_key(ax)

    def _draw_water_series_key(self, ax) -> None:
        """Marker + line + label per GW series, as a small framed key."""
        from matplotlib.legend import Legend
        from matplotlib.lines import Line2D

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
        key = Legend(
            ax,
            handles,
            labels,
            loc="lower left",
            fontsize=7,
            frameon=True,
            framealpha=0.9,
            handlelength=3.0,
            borderaxespad=0.6,
        )
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
