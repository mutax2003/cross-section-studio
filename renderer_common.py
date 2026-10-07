"""Shared geometry and style helpers for CrossSectionRenderer."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.collections import PolyCollection
from matplotlib.path import Path
from matplotlib.text import Annotation
from matplotlib.ticker import FuncFormatter, Locator, MaxNLocator
from shapely.geometry import Polygon as ShapelyPolygon

import hatch_patterns
from constants import get_lithology_style
from models import ScreenInterval
from render_theme import SCREEN_INTERVAL_HATCH, TRACK_BORDER_COLOR, TRACK_FILL_COLOR

# Draw '.', '/' and '+' as the client legend template's marks (fine stipple,
# short silty dashes, isolated plus marks) in every figure and export.
hatch_patterns.install()


class _ExaggeratedAxisLocator(Locator):
    """Place ticks on round TRUE values of a vertically exaggerated axis."""

    def __init__(self, ve: float) -> None:
        self._ve = ve
        self._nice = MaxNLocator(nbins="auto", steps=[1, 2, 2.5, 5, 10])

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        self._nice.set_axis(self.axis)
        return np.asarray(self._nice.tick_values(vmin / self._ve, vmax / self._ve)) * self._ve


def apply_true_value_y_axis(ax, ve: float) -> None:
    """Label a y axis plotted at value*VE with the true (unexaggerated) values."""
    if not ve or ve == 1.0:
        return
    ax.yaxis.set_major_locator(_ExaggeratedAxisLocator(ve))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos, v=ve: f"{value / v:.6g}"))


# Column intervals drawn shorter than this (in inches on the output page) get a
# densified hatch so a sparse single-character pattern ('.', '/', 'O', '+')
# still leaves visible marks. 0.1 in is ~28-30 px at 300 dpi.  For a base
# hatch that is already N characters dense the threshold scales to 0.1 / N in,
# because its rows are N times closer together (see ``hatch_density``).
THIN_UNIT_MIN_HEIGHT_IN = 0.1
# Matplotlib lays a single-character hatch out on a 1 in unit cell at
# rcParams-independent density 6, i.e. rows of marks every 1/6 in; every
# repeat of the character adds another 6 rows per inch.
BASE_HATCH_ROW_SPACING_IN = 1.0 / 6.0
# Total pattern density (base characters x repeat factor) stays within
# [2, 4]: densify a single-character hatch at least 2x below the threshold and
# never let any resolved hatch exceed 4 characters, so the pattern still reads
# as the legend's stipple / diagonal rather than solid texture.  A base hatch
# that is already 4+ characters ('xxx' is 3, so 2x would be 6) is never
# densified.
THIN_UNIT_MIN_DENSIFY = 2
THIN_UNIT_MAX_DENSIFY = 4


def hatch_density(hatch: str | None) -> int:
    """Density of a matplotlib hatch relative to a single-character pattern.

    ``'.'`` → 1, ``'..'`` → 2, ``'xxx'`` → 3.  Mixed patterns such as ``'/.'``
    count every character (2) because the extra mark type adds texture even
    though each individual mark type is still at single density.  Empty / None
    (plain unit) is reported as 1 so callers can divide by it.
    """
    return max(1, len(hatch or ""))


def densify_hatch(hatch: str | None, factor: int = THIN_UNIT_MIN_DENSIFY) -> str:
    """Repeat a sparse hatch so matplotlib draws it ``factor`` times denser.

    Matplotlib hatch density scales with the number of pattern characters, so
    ``'.'`` → ``'..'``, ``'/'`` → ``'//'``, ``'O'`` → ``'OO'``, ``'+'`` → ``'++'``.
    An empty hatch (plain unit) stays empty.
    """
    if not hatch:
        return ""
    return hatch * max(1, int(factor))


def thin_unit_densify_factor(
    height_in: float,
    *,
    min_height_in: float = THIN_UNIT_MIN_HEIGHT_IN,
    base_hatch: str | None = None,
) -> int:
    """Hatch repeat factor for an interval drawn ``height_in`` inches tall.

    ``base_hatch`` is the legend pattern that will be repeated; its own density
    (``hatch_density``) scales both the thin threshold and the row spacing, so
    ``'..'`` is treated as thin below ``min_height_in / 2`` and ``'xxx'`` is
    never densified (3 chars x 2 would exceed ``THIN_UNIT_MAX_DENSIFY``).

    Returns 1 (base hatch) at or above the effective threshold.  Below it, the
    smallest repeat whose effective row spacing fits inside the interval (so
    at least one row of marks lands in the band whatever the pattern phase),
    clamped so the resolved pattern density ``hatch_density(base) * factor``
    lies in ``[THIN_UNIT_MIN_DENSIFY, THIN_UNIT_MAX_DENSIFY]``.  A
    single-character base therefore gets a factor in ``[2, 4]``.
    """
    density = hatch_density(base_hatch)
    max_factor = THIN_UNIT_MAX_DENSIFY // density
    if max_factor < 2:
        # Base is already at least half the cap: a single repeat would exceed it.
        return 1
    # Template marks ('.', '/', '+') sit closer than matplotlib's stock 1/6 in
    # rows; scale the thin threshold and spacing by the pattern's own pitch.
    pitch_scale = hatch_patterns.STOCK_ROWS_PER_INCH / hatch_patterns.rows_per_inch(base_hatch)
    threshold_in = min_height_in * pitch_scale / density
    if not np.isfinite(height_in) or height_in >= threshold_in:
        return 1
    min_factor = max(1, -(-THIN_UNIT_MIN_DENSIFY // density))  # ceil division
    if height_in <= 0.0:
        return max_factor
    row_spacing_in = BASE_HATCH_ROW_SPACING_IN * pitch_scale / density
    needed = int(np.ceil(row_spacing_in / height_in))
    return int(min(max_factor, max(min_factor, needed)))



def legend_swatch_hatch(hatch: str | None, height_in: float) -> str | None:
    """Hatch for a legend swatch ``height_in`` inches tall.

    A swatch shorter than one hatch row can miss every mark (Sand then reads
    as plain in the legend). Densify just enough that a row always lands
    inside, with the same caps as thin beds; taller swatches keep the base.
    """
    if not hatch:
        return hatch
    if "O" in hatch:
        # Gravel stones are drawn at one size everywhere (section, legend,
        # legend sheet) so the legend shows exactly what the figure shows.
        return hatch
    factor = thin_unit_densify_factor(
        height_in, min_height_in=BASE_HATCH_ROW_SPACING_IN, base_hatch=hatch
    )
    if "+" in hatch:
        # Separate "+" marks packed 3-4x tighter read as a grid, unlike the
        # section; double density at most keeps them separate marks.
        factor = min(factor, 2)
    return densify_hatch(hatch, factor) if factor > 1 else hatch


_AGG_SNAP_MAX_VERTICES = 1024  # Agg PathSnapper: SNAP_AUTO never snaps longer paths


def _agg_auto_snaps(display_verts: np.ndarray, codes: np.ndarray | None) -> bool:
    """Replicate Agg's SNAP_AUTO ``should_snap`` test for one path (display coords).

    Only LINETO segments are checked; every vertex (MOVETO / CLOSEPOLY too)
    becomes the previous point, exactly as the C++ iterator walks the path.
    """
    if len(display_verts) > _AGG_SNAP_MAX_VERTICES or len(display_verts) < 1:
        return False
    if codes is not None and np.any((codes == Path.CURVE3) | (codes == Path.CURVE4)):
        return False
    delta = np.abs(np.diff(display_verts, axis=0))
    if codes is None:
        line_to = np.ones(len(delta), dtype=bool)
    else:
        line_to = codes[1:] == Path.LINETO
    bad = line_to & (delta[:, 0] >= 1e-4) & (delta[:, 1] >= 1e-4)
    return not bool(np.any(bad))


def _overlaps_by_area(verts_a, codes_a, verts_b, codes_b, *, min_area_px: float = 0.25) -> bool:
    """True unless two single-ring paths provably meet only along edges / points."""
    for codes in (codes_a, codes_b):
        if codes is not None and np.count_nonzero(codes == Path.MOVETO) > 1:
            return True
    try:
        ring_a = ShapelyPolygon(verts_a)
        ring_b = ShapelyPolygon(verts_b)
        if not (ring_a.is_valid and ring_b.is_valid):
            return True
        return bool(ring_a.intersection(ring_b).area > min_area_px)
    except Exception:  # degenerate ring: keep the original order
        return True


class HatchBatchedPolyCollection(PolyCollection):
    """PolyCollection whose hatch is rasterised once per batch of disjoint paths.

    Agg re-renders the full one-inch hatch tile (300 x 300 px at export dpi)
    for every path in a hatched collection, which made dense gravel / sand
    stipple dominate PNG export time.  On the Agg renderer this draw merges
    paths whose (padded) display boxes do not touch into one compound path for
    the face + hatch pass, then strokes the same paths in a second pass.

    Pixels are unchanged: members of a batch are separated by more than the
    stroke width plus antialiasing, overlapping paths keep their original
    relative draw order (a later path always lands in a later batch), the
    hatch pattern is anchored to the canvas origin either way, and each pass
    keeps the original line width, face presence (so no path clipping or
    simplification kicks in) and Agg pixel-snapping decision.  Vector
    backends (SVG / PDF) and any collection outside the simple single-style
    case fall back to the stock draw.

    ``merge_shared_edges=True`` (fence polygons) also batches neighbours that
    only share an edge (zero-area intersection), e.g. one unit's panels either
    side of a borehole.  Their common edge is stroked identically by both, so
    the only change is antialiasing on that seam (at most 1/255 per channel in the
    60-hole parity check; up to 5/255 under the semi-opaque borehole columns
    in the 12-hole sweep); paths that overlap by area stay ordered.
    """

    def __init__(self, *args, merge_shared_edges: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.merge_shared_edges = bool(merge_shared_edges)

    def draw(self, renderer) -> None:  # type: ignore[override]
        batches = self._hatch_batches(renderer)
        if batches is None:
            super().draw(renderer)
            return
        paths = self._paths
        facecolors = self._facecolors
        edgecolors = self._edgecolors
        hatchcolors = self._hatchcolors
        hatch = self._hatch
        resolved_edge = np.array(self.get_edgecolor(), dtype=float)
        resolved_hatch_color = np.array(self.get_hatchcolor(), dtype=float)
        clear_edge = resolved_edge.copy()
        clear_edge[:, 3] = 0.0
        clear_face = np.zeros((1, 4), dtype=float)
        try:
            for fill_groups, members in batches:
                # Face + hatch: one compound path per snap-consistent group;
                # a fully transparent stroke keeps the original line width so
                # Agg snaps the fill exactly as it did per path.
                self._paths = [Path.make_compound_path(*group) for group in fill_groups]
                self._facecolors = facecolors
                self._edgecolors = clear_edge
                self._hatchcolors = resolved_hatch_color
                self._hatch = hatch
                PolyCollection.draw(self, renderer)
                # Strokes: transparent face (not 'none') so the stroke path
                # is neither clipped nor simplified, as in the stock draw.
                self._paths = members
                self._facecolors = clear_face
                self._edgecolors = resolved_edge
                self._hatch = None
                PolyCollection.draw(self, renderer)
        finally:
            self._paths = paths
            self._facecolors = facecolors
            self._edgecolors = edgecolors
            self._hatchcolors = hatchcolors
            self._hatch = hatch

    def _hatch_batches(self, renderer):
        """``[(fill_groups, member_paths), ...]`` in draw order, or None for the stock draw."""
        if not isinstance(renderer, RendererAgg) or not self._hatch or not self.get_visible():
            return None
        if hasattr(vars(renderer).get("draw_path_collection"), "__wrapped__"):
            # RendererBase._draw_disabled (draw_without_rendering) swaps in
            # wrapped no-ops: nothing is rasterised, so there is nothing to batch.
            return None
        paths = self.get_paths()
        if len(paths) < 2:
            return None
        if (
            self.get_path_effects()
            or self.get_sketch_params() is not None
            or self._gapcolor is not None
            or self.get_array() is not None
            or self.get_agg_filter() is not None
            or len(self.get_transforms())
            or len(self._facecolors) != 1
            or len(self.get_edgecolor()) != 1
            or len(self.get_hatchcolor()) != 1
            or len(self._linewidths) != 1
            or len(self._linestyles) != 1
            or len(self._antialiaseds) != 1
            or len(self._urls) != 1
        ):
            return None
        offsets = self.get_offsets()
        if len(offsets) != 1 or np.any(np.asarray(offsets) != 0):
            return None
        transform = self.get_transform()
        if not transform.is_affine:
            return None
        affine = transform.get_affine()
        display = []
        for path in paths:
            verts = np.asarray(path.vertices, dtype=float)
            if verts.ndim != 2 or len(verts) == 0 or not np.all(np.isfinite(verts)):
                return None
            display.append(affine.transform(verts))
        snap = self.get_snap()
        if snap is None:
            snaps = [_agg_auto_snaps(d, p.codes) for d, p in zip(display, paths, strict=True)]
        else:
            snaps = [bool(snap)] * len(paths)
        # Pad: half the stroke, plus antialiasing / snapping slack on each side.
        pad = renderer.points_to_pixels(float(self._linewidths[0])) / 2.0 + 2.0
        boxes = np.array(
            [(d[:, 0].min(), d[:, 1].min(), d[:, 0].max(), d[:, 1].max()) for d in display]
        )
        boxes[:, :2] -= pad
        boxes[:, 2:] += pad
        # Level of path j = 1 + max level of any earlier path it touches, so
        # touching paths keep their original order and each level is disjoint.
        levels = np.zeros(len(paths), dtype=int)
        for j in range(1, len(paths)):
            earlier = boxes[:j]
            touch = (
                (earlier[:, 0] <= boxes[j, 2])
                & (earlier[:, 2] >= boxes[j, 0])
                & (earlier[:, 1] <= boxes[j, 3])
                & (earlier[:, 3] >= boxes[j, 1])
            )
            if touch.any() and self.merge_shared_edges:
                touch[touch] = [
                    _overlaps_by_area(display[i], paths[i].codes, display[j], paths[j].codes)
                    for i in np.flatnonzero(touch)
                ]
            if touch.any():
                levels[j] = int(levels[:j][touch].max()) + 1
        if int(levels.max()) + 1 >= len(paths):
            return None  # nothing to merge
        batches = []
        for level in range(int(levels.max()) + 1):
            members = [paths[i] for i in np.flatnonzero(levels == level)]
            groups: list[list] = []
            snapped_group: list = []
            snapped_vertices = 0
            unsnapped_group: list = []
            for i in np.flatnonzero(levels == level):
                path = paths[i]
                if snaps[i] and snap is None:
                    # Keep each merged snapped path under Agg's vertex limit.
                    if (
                        snapped_vertices + len(path.vertices) > _AGG_SNAP_MAX_VERTICES
                        and snapped_group
                    ):
                        groups.append(snapped_group)
                        snapped_group, snapped_vertices = [], 0
                    snapped_group.append(path)
                    snapped_vertices += len(path.vertices)
                else:
                    unsnapped_group.append(path)
            groups.extend(group for group in (snapped_group, unsnapped_group) if group)
            batches.append((groups, members))
        return batches


# SVG / PDF backends name each collection group after ``type(obj).__name__``
# ("PolyCollection_7"); keep that name so vector exports stay byte-identical.
# Pickle and repr use ``__qualname__``, which still names this class.
HatchBatchedPolyCollection.__name__ = "PolyCollection"


class ThinUnitHatchCollection(HatchBatchedPolyCollection):
    """Lithology rectangles whose hatch is densified for thin intervals at draw time.

    The split between "thin" and "thick" intervals needs the final axes
    transform (view limits, ``subplots_adjust`` and the output dpi are only
    known when the figure is drawn), so the decision is taken inside
    :meth:`draw`.  Each densify factor in use is drawn as its own batch with
    the matching hatch; ``resolved_hatches`` records the per-rectangle outcome
    of the last draw for inspection and tests.
    """

    def __init__(
        self,
        verts,
        *,
        base_hatch: str,
        min_height_in: float = THIN_UNIT_MIN_HEIGHT_IN,
        **kwargs,
    ) -> None:
        super().__init__(verts, hatch=base_hatch or None, **kwargs)
        self.base_hatch = base_hatch or ""
        self.min_height_in = float(min_height_in)
        self.resolved_hatches: list[str] = [self.base_hatch] * len(self.get_paths())

    def _interval_heights_in(self) -> np.ndarray | None:
        """Drawn height of each rectangle in inches, or None when unknowable."""
        figure = self.get_figure()
        paths = self.get_paths()
        if figure is None or not paths:
            return None
        dpi = float(getattr(figure, "dpi", 0.0) or 0.0)
        if dpi <= 0.0:
            return None
        verts = np.asarray([path.vertices for path in paths], dtype=float)
        if verts.ndim != 3 or verts.shape[1] < 4:
            return None
        # Rect verts are (x_left, y0), (x_right, y0), (x_right, y_top), (x_left, y_top).
        bottom = verts[:, 0, :]
        top = verts[:, 3, :]
        transform = self.get_transform()
        bottom_px = transform.transform(bottom)
        top_px = transform.transform(top)
        return np.abs(top_px[:, 1] - bottom_px[:, 1]) / dpi

    def densify_factors(self) -> np.ndarray:
        """Per-rectangle hatch repeat factor for the current figure transform."""
        heights = self._interval_heights_in()
        if heights is None:
            return np.ones(len(self.get_paths()), dtype=int)
        return np.asarray(
            [
                thin_unit_densify_factor(
                    float(h), min_height_in=self.min_height_in, base_hatch=self.base_hatch
                )
                for h in heights
            ],
            dtype=int,
        )

    def thin_mask(self) -> np.ndarray:
        """Boolean mask of rectangles currently shorter than ``min_height_in``."""
        return self.densify_factors() > 1

    def draw(self, renderer) -> None:  # type: ignore[override]
        if not self.base_hatch:
            super().draw(renderer)
            return
        factors = self.densify_factors()
        self.resolved_hatches = [densify_hatch(self.base_hatch, int(f)) for f in factors]
        distinct = sorted({int(f) for f in factors})
        if distinct == [1]:
            self._set_hatch_quietly(self.base_hatch)
            super().draw(renderer)
            return
        all_paths = self._paths
        try:
            for factor in distinct:
                subset = [
                    path for path, f in zip(all_paths, factors, strict=True) if int(f) == factor
                ]
                self._paths = subset
                self._set_hatch_quietly(densify_hatch(self.base_hatch, factor))
                super().draw(renderer)
        finally:
            # Restore the base hatch without ``set_hatch``: that marks the
            # collection (and hence the figure) stale after every draw, which
            # would force a redundant redraw on the next savefig / canvas.draw.
            self._paths = all_paths
            self._set_hatch_quietly(self.base_hatch)

    def _set_hatch_quietly(self, hatch: str) -> None:
        """Swap the draw-time hatch without flagging the artist stale."""
        if self.get_hatch() != hatch:
            self._hatch = hatch


# --- Layout settling for the label-collision passes ------------------------
#
# The header / tick / water-label passes each call ``draw_without_rendering``
# so tick labels, axis labels and titles sit where a draw puts them.  Most of
# those draws repeat an identical layout: only label texts moved in between,
# and text extents are computed lazily anyway.  ``settle_figure_layout`` skips
# the draw when no draw of any kind has happened since its previous settle of
# the same figure (tracked with a ``draw_event`` counter, so a savefig at
# another dpi always forces a fresh settle) and nothing that positions ticks,
# axis labels or titles has changed (``_layout_fingerprint``).


def _figure_draw_count(fig) -> int | None:
    """Draws of ``fig`` seen by our ``draw_event`` hook, or None if just (re)installed."""
    registry = fig.canvas.callbacks
    holder = fig.__dict__.get("_css_draw_count")
    cid = fig.__dict__.get("_css_draw_cid")
    if holder is not None and cid is not None:
        ref = registry.callbacks.get("draw_event", {}).get(cid)
        hook = ref() if ref is not None else None
        if getattr(hook, "_css_holder", None) is holder:
            return holder[0]
    # First use, or the hook was lost (unpickled / copied figure): no history.
    holder = [0]

    def count_draw(_event, holder=holder) -> None:
        holder[0] += 1

    count_draw._css_holder = holder
    fig._css_draw_count = holder
    fig._css_draw_cid = registry.connect("draw_event", count_draw)
    return None


def _axis_layout_key(axis) -> tuple:
    formatter = axis.get_major_formatter()
    suppressed = getattr(formatter, "suppressed", None)
    return (
        id(formatter),
        id(axis.get_major_locator()),
        id(axis.get_minor_formatter()),
        id(axis.get_minor_locator()),
        tuple(sorted(suppressed)) if suppressed is not None else None,
        axis.get_visible(),
        axis.get_scale(),
        axis.get_ticks_position(),
        axis.get_label_position(),
        axis.label.get_text(),
        axis.label.get_visible(),
        axis.labelpad,
        repr(sorted(axis._major_tick_kw.items())),
        repr(sorted(axis._minor_tick_kw.items())),
    )


def _layout_fingerprint(fig) -> tuple | None:
    """Everything a no-render draw commits for tick, axis-label and title layout."""
    if fig.get_layout_engine() is not None:
        return None  # layout engines reposition axes from every artist's extent
    params = fig.subplotpars
    parts: list = [
        tuple(fig.get_size_inches()),
        float(fig.dpi),
        (params.left, params.right, params.bottom, params.top, params.wspace, params.hspace),
    ]
    for ax in fig.axes:
        parts.append(
            (
                id(ax),
                ax.get_visible(),
                ax.get_position().bounds,
                tuple(ax.get_xlim()),
                tuple(ax.get_ylim()),
                tuple(title.get_text() for title in (ax.title, ax._left_title, ax._right_title)),
                tuple(
                    (name, repr(spine._position), spine.get_visible())
                    for name, spine in ax.spines.items()
                ),
                _axis_layout_key(ax.xaxis),
                _axis_layout_key(ax.yaxis),
            )
        )
    # Axes that draw their axis artists, kept apart: turning an axes' axis
    # *off* (title-block panels) changes nothing a no-render draw commits, as
    # its ticks are then simply not updated; turning one on needs a draw.
    axis_on = frozenset(id(ax) for ax in fig.axes if ax.axison)
    return tuple(parts), axis_on


def _same_settled_layout(previous: tuple, current: tuple) -> bool:
    previous_parts, previous_on = previous
    current_parts, current_on = current
    return previous_parts == current_parts and current_on <= previous_on


def _annotation_arrows(fig) -> list:
    arrows = []
    for container in (fig, *fig.axes):
        for text in container.texts:
            if isinstance(text, Annotation) and text.arrow_patch is not None:
                arrows.append(text.arrow_patch)
    return arrows


def _settle_arrow_draw(arrow):
    def draw(renderer) -> None:
        # The only state FancyArrowPatch.draw keeps is the dpi correction its
        # get_window_extent reads; the clipped leader path itself is only
        # needed when pixels are produced.
        if arrow.get_visible():
            arrow._dpi_cor = renderer.points_to_pixels(1.0)

    return draw


def _draw_without_rendering_fast(fig) -> None:
    """``fig.draw_without_rendering()`` minus annotation leader path geometry."""
    arrows = _annotation_arrows(fig)
    for arrow in arrows:
        arrow.draw = _settle_arrow_draw(arrow)
    try:
        fig.draw_without_rendering()
    finally:
        for arrow in arrows:
            arrow.__dict__.pop("draw", None)


def settle_figure_layout(fig) -> None:
    """``fig.draw_without_rendering()`` for the label passes, skipped when the
    figure has not been drawn since its last settle and its layout is unchanged."""
    draws = _figure_draw_count(fig)
    key = _layout_fingerprint(fig)
    previous = fig.__dict__.get("_css_settled_layout")
    if (
        draws is not None
        and key is not None
        and previous is not None
        and previous[0] == draws
        and previous[1] is not None
        and _same_settled_layout(previous[1], key)
    ):
        return
    _draw_without_rendering_fast(fig)
    fig._css_settled_layout = (_figure_draw_count(fig), _layout_fingerprint(fig))


class RendererGeometryMixin:
    """Well extents, profile lookup, and lithology style resolution."""

    def _collar_values(self, hole_ids: pd.Series, fallback: pd.Series, collar_lookup: dict[str, float]) -> np.ndarray:
        mapped = hole_ids.astype(str).map(collar_lookup)
        return mapped.fillna(fallback).to_numpy(dtype=float)

    def _profile_lookup(
        self,
        hole_summary: pd.DataFrame,
        collar_lookup: dict[str, float] | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Map hole_id → (x_profile, collar_rl)."""
        if hole_summary.empty:
            return {}
        hole_ids = hole_summary["hole_id"].astype(str).to_numpy()
        x_values = hole_summary["x_profile"].to_numpy(dtype=float)
        if collar_lookup:
            collars = self._collar_values(
                hole_summary["hole_id"],
                hole_summary["collar_elevation"],
                collar_lookup,
            )
        else:
            collars = hole_summary["collar_elevation"].to_numpy(dtype=float)
        return {
            str(hole_id): (float(x_profile), float(collar_rl))
            for hole_id, x_profile, collar_rl in zip(hole_ids, x_values, collars, strict=True)
        }

    def _well_extents(
        self,
        hole_summary: pd.DataFrame,
        collar_depths: dict[str, float],
        collar_lookup: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return (x_profile, top_y, bottom_y) for well sticks/columns, or None if empty.

        When ``stick_up_by_hole`` is set on the renderer, column tops extend above
        the collar by that stick-up height (Wave B MW construction).
        """
        if hole_summary.empty:
            return None
        hole_ids = hole_summary["hole_id"].astype(str).to_numpy()
        x_values = hole_summary["x_profile"].to_numpy(dtype=float)
        collars = self._collar_values(
            hole_summary["hole_id"],
            hole_summary["collar_elevation"],
            collar_lookup,
        )
        row_bottoms = hole_summary["bottom_elevation"].to_numpy(dtype=float)
        td_values = np.fromiter(
            (collar_depths.get(str(hole_id), np.nan) for hole_id in hole_ids),
            dtype=float,
            count=len(hole_ids),
        )
        bottom_elev = np.where(np.isfinite(td_values), collars - td_values, row_bottoms)
        stick_ups = getattr(self, "stick_up_by_hole", None) or {}
        top_elev = np.asarray(
            [
                float(collars[i]) + float(stick_ups.get(str(hole_ids[i]), 0.0) or 0.0)
                for i in range(len(hole_ids))
            ],
            dtype=float,
        )
        top_y = self._plot_y_values(top_elev, collars)
        bottom_y = self._plot_y_values(bottom_elev, collars)
        return x_values, top_y, bottom_y

    def _well_rect_geometry(
        self,
        hole_summary: pd.DataFrame,
        collar_depths: dict[str, float],
        collar_lookup: dict[str, float],
        track_half: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Return (x_left, y0, width, height) for filled/bordered well rectangles."""
        extents = self._well_extents(hole_summary, collar_depths, collar_lookup)
        if extents is None:
            return None
        x_values, top_y, bottom_y = extents
        heights = np.abs(top_y - bottom_y)
        mask = heights > 1e-9
        if not np.any(mask):
            return None
        x_values = x_values[mask]
        y0 = np.minimum(top_y[mask], bottom_y[mask])
        heights = heights[mask]
        return x_values - track_half, y0, np.full(len(x_values), track_half * 2.0), heights

    @staticmethod
    def _rect_verts(
        x_left: np.ndarray,
        y0: np.ndarray,
        widths: np.ndarray,
        heights: np.ndarray,
    ) -> np.ndarray:
        x_right = x_left + widths
        y_top = y0 + heights
        return np.stack(
            [
                np.column_stack([x_left, y0]),
                np.column_stack([x_right, y0]),
                np.column_stack([x_right, y_top]),
                np.column_stack([x_left, y_top]),
            ],
            axis=1,
        )

    def _add_rect_collection(
        self,
        ax,
        geometry: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        *,
        facecolors,
        edgecolors,
        linewidths: float,
        zorder: int,
        hatch: str | None = None,
        alpha: float | None = None,
    ) -> None:
        collection = (HatchBatchedPolyCollection if hatch else PolyCollection)(
            self._rect_verts(*geometry),
            facecolors=facecolors,
            edgecolors=edgecolors,
            linewidths=linewidths,
            hatch=hatch,
            alpha=alpha,
        )
        collection.set_zorder(zorder)
        ax.add_collection(collection)

    def _draw_lithology_interval_rects(
        self,
        ax,
        projected_df: pd.DataFrame,
        style_cache: dict,
        track_half_width: float,
        collar_lookup: dict[str, float],
        *,
        zorder: int = 5,
        alpha: float = 0.96,
        collar_arr: np.ndarray | None = None,
    ) -> None:
        """Vectorized track lithology rectangles grouped by resolved style."""
        if projected_df.empty:
            return
        collars = (
            collar_arr
            if collar_arr is not None
            else self._collar_values(
                projected_df["hole_id"],
                projected_df["collar_elevation"],
                collar_lookup,
            )
        )
        x_values = projected_df["x_profile"].to_numpy(dtype=float)
        tops = self._plot_y_values(
            projected_df["top_elevation"].to_numpy(dtype=float),
            collars,
        )
        bottoms = self._plot_y_values(
            projected_df["bottom_elevation"].to_numpy(dtype=float),
            collars,
        )
        y0 = np.minimum(tops, bottoms)
        heights = np.abs(tops - bottoms)
        mask = heights > 1e-9
        if not np.any(mask):
            return
        lithology_codes = projected_df["lithology_code"].astype(str).to_numpy()[mask]
        x_masked = x_values[mask]
        y0_masked = y0[mask]
        heights_masked = heights[mask]
        width = 2.0 * track_half_width
        order = np.argsort(lithology_codes, kind="stable")
        sorted_codes = lithology_codes[order]
        split_at = np.concatenate(
            ([0], np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1, [len(sorted_codes)])
        )
        thin_min_in = float(getattr(self, "thin_unit_min_height_in", THIN_UNIT_MIN_HEIGHT_IN))
        for start, end in zip(split_at[:-1], split_at[1:], strict=True):
            code = str(sorted_codes[start])
            style = self._resolve_style(code, style_cache)
            slice_idx = order[start:end]
            count = end - start
            geometry = (
                x_masked[slice_idx] - track_half_width,
                y0_masked[slice_idx],
                np.full(count, width),
                heights_masked[slice_idx],
            )
            hatch = style.hatch if self.show_hatches else ""
            if not hatch:
                self._add_rect_collection(
                    ax,
                    geometry,
                    facecolors=style.color,
                    edgecolors=style.edge_color,
                    linewidths=0.6,
                    zorder=zorder,
                    hatch=None,
                    alpha=alpha,
                )
                continue
            # One collection per code; thin intervals pick up the densified
            # hatch when the figure is drawn (see ThinUnitHatchCollection).
            collection = ThinUnitHatchCollection(
                self._rect_verts(*geometry),
                base_hatch=hatch,
                min_height_in=thin_min_in,
                facecolors=style.color,
                edgecolors=style.edge_color,
                linewidths=0.6,
                alpha=alpha,
            )
            collection.set_zorder(zorder)
            ax.add_collection(collection)

    def _style_cache_for(self, lithology_codes: Sequence[str]) -> dict:
        consulting_palette = bool(getattr(self.profile, "use_consulting_palette", False))
        use_hatch = self.show_hatches
        return {
            code: get_lithology_style(
                code,
                use_hatch=use_hatch,
                consulting_palette=consulting_palette,
            )
            for code in lithology_codes
        }

    def _resolve_style(self, code: str, style_cache: dict):
        style = style_cache.get(code)
        if style is None:
            consulting = getattr(self.profile, "use_consulting_palette", False)
            style = get_lithology_style(
                code,
                use_hatch=self.show_hatches,
                consulting_palette=consulting,
            )
            style_cache[code] = style
        return style

    def _draw_screen_intervals(
        self,
        ax,
        hole_summary: pd.DataFrame,
        screen_intervals: Sequence[ScreenInterval],
        collar_lookup: dict[str, float],
        track_half: float,
        *,
        profile_lookup: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if not screen_intervals or hole_summary.empty:
            return
        if profile_lookup is None:
            profile_lookup = self._profile_lookup(hole_summary, collar_lookup)
        half = track_half * 0.92
        width = track_half * 1.84
        x_profiles: list[float] = []
        collar_rls: list[float] = []
        from_depths: list[float] = []
        to_depths: list[float] = []
        for interval in screen_intervals:
            profile = profile_lookup.get(interval.hole_id)
            if profile is None:
                continue
            x_profile, collar_rl = profile
            x_profiles.append(x_profile)
            collar_rls.append(collar_rl)
            from_depths.append(interval.from_depth)
            to_depths.append(interval.to_depth)
        if not x_profiles:
            return
        x_arr = np.asarray(x_profiles, dtype=float)
        collar_arr = np.asarray(collar_rls, dtype=float)
        from_arr = np.asarray(from_depths, dtype=float)
        to_arr = np.asarray(to_depths, dtype=float)
        top_ys = self._plot_y_values(collar_arr - from_arr, collar_arr)
        bottom_ys = self._plot_y_values(collar_arr - to_arr, collar_arr)
        heights = np.abs(top_ys - bottom_ys)
        mask = heights > 1e-9
        if not np.any(mask):
            return
        x_arr = x_arr[mask] - half
        y_arr = np.minimum(top_ys[mask], bottom_ys[mask])
        h_arr = heights[mask]
        self._add_rect_collection(
            ax,
            (x_arr, y_arr, np.full(int(mask.sum()), width), h_arr),
            facecolors=TRACK_FILL_COLOR,
            edgecolors=TRACK_BORDER_COLOR,
            linewidths=0.6,
            zorder=9,
            hatch=SCREEN_INTERVAL_HATCH,
        )

