"""Shared geometry and style helpers for CrossSectionRenderer."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.collections import PolyCollection
from matplotlib.patches import PathPatch, Polygon, Rectangle
from matplotlib.path import Path
from matplotlib.text import Annotation, Text
from matplotlib.ticker import FuncFormatter, Locator, MaxNLocator
from matplotlib.transforms import Affine2D
from shapely.geometry import Polygon as ShapelyPolygon

import hatch_patterns
from constants import HATCH_LINE_COLOR, POLYGON_EDGE_COLOR, get_lithology_style
from hydro_metrics import water_depth_below_ground_m, water_head_masl
from models import ScreenInterval, WaterLevel
from render_theme import (
    CONSULTING_COLUMN_FILL,
    PINCH_OUT_ALPHA,
    SCREEN_INTERVAL_HATCH,
    TRACK_BORDER_COLOR,
    TRACK_FILL_COLOR,
)

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


# --- Vertical exaggeration (VE) -------------------------------------------
#
# Data are stored at ``value * storage_ve`` on the y axis. In *exact* mode the
# axes box is locked to the chosen VE (aspect 1 in stored units, adjustable
# "box", centred in its frame region); in *auto* mode the axes fill their frame
# and the caption prints the VE the figure really ends up with, measured from
# the final axes box at draw time (so export page re-framing is honoured).

# A measured auto VE within this of 1 prints as "no vertical exaggeration".
VE_UNITY_TOLERANCE = 0.05


def measured_vertical_exaggeration(ax, storage_ve: float = 1.0) -> float | None:
    """True VE drawn by ``ax``: (display px per vertical metre) / (px per horizontal metre).

    ``storage_ve`` is the factor the y data were multiplied by before plotting.
    Uses the axes' active (aspect-applied) box, so it is exact at draw time.
    Measured in display pixels: during a ``bbox_inches="tight"`` save
    matplotlib temporarily changes the figure size while the axes fractions
    stay relative to the original page, so fractions x page inches were wrong.
    """
    figure = ax.figure
    if figure is None:
        return None
    if ax.get_axes_locator() is None:
        ax.apply_aspect()
    box = ax.get_window_extent()
    box_w = float(box.width)
    box_h = float(box.height)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = abs(float(x1) - float(x0))
    dy = abs(float(y1) - float(y0))
    storage = float(storage_ve) if storage_ve and storage_ve > 0 else 1.0
    if box_w <= 0 or box_h <= 0 or dx <= 0 or dy <= 0:
        return None
    if not all(np.isfinite(v) for v in (dx, dy, box_w, box_h)):
        return None
    vertical = box_h / (dy / storage)
    horizontal = box_w / dx
    return vertical / horizontal


def round_vertical_exaggeration(ve: float) -> float:
    """Sensible rounding for a measured VE: 1 decimal under 10x, else whole."""
    ve = float(ve)
    if ve < 10.0:
        return round(ve, 1)
    return float(round(ve))


def is_unit_ve(ve: float, *, auto: bool) -> bool:
    """True when the drawing has no vertical exaggeration (auto: within 5 %)."""
    if auto:
        return abs(float(ve) - 1.0) <= VE_UNITY_TOLERANCE
    return abs(float(ve) - 1.0) < 1e-9


def format_ve_value(ve: float, *, auto: bool) -> str:
    """``"5×"`` for a chosen VE, ``"≈2.2×"`` for a measured (auto) one."""
    if auto:
        return f"≈{round_vertical_exaggeration(ve):g}×"
    return f"{float(ve):g}×"


def format_ve_caption(ve: float | None, *, auto: bool, style: str = "consulting") -> str:
    """VE wording for the figure.

    ``style``: ``"consulting"`` (subtitle band, upper case), ``"short"``
    (in-plot "V.E." note) or ``"footer"`` (metadata footer item).
    """
    if ve is None or not np.isfinite(ve) or ve <= 0:
        return {
            "consulting": "VERTICAL EXAGGERATION: AUTO",
            "short": "V.E. auto",
            "footer": "VE: auto",
        }.get(style, "VE: auto")
    unit = is_unit_ve(ve, auto=auto)
    if style == "consulting":
        if unit:
            return "NO VERTICAL EXAGGERATION"
        if auto:
            return f"VERTICAL EXAGGERATION {format_ve_value(ve, auto=True)}"
        return f"{format_ve_value(ve, auto=False)} VERTICAL EXAGGERATION"
    if style == "short":
        if unit:
            return "No V.E. (1×)"
        return f"V.E. {format_ve_value(ve, auto=auto)}"
    if unit:
        return "VE: none (1×)"
    return f"VE: {format_ve_value(ve, auto=auto)}"


class VECaptionText(Text):
    """Text whose VE wording is recomputed from the final axes box at draw time.

    ``template`` holds ``{ve}`` where the caption goes (e.g. a footer line).
    In exact mode the chosen VE is printed; in auto mode the VE measured from
    ``main_ax`` once layout (and any export page resize) is final.
    """

    def __init__(
        self,
        x: float,
        y: float,
        *,
        main_ax,
        storage_ve: float,
        auto: bool,
        chosen_ve: float | None,
        style: str = "consulting",
        template: str = "{ve}",
        fit_axes=None,
        **kwargs,
    ) -> None:
        self._ve_main_ax = main_ax
        self._ve_storage = float(storage_ve)
        self._ve_auto = bool(auto)
        self._ve_chosen = chosen_ve
        self._ve_style = style
        self._ve_template = template
        # Shrink (never grow) to stay inside this panel's width.
        self._ve_fit_axes = fit_axes
        super().__init__(x, y, "", **kwargs)
        self._ve_base_size = float(self.get_fontsize())
        self.refresh_ve_text()

    def _fit_to_panel(self, renderer=None) -> None:
        panel = getattr(self, "_ve_fit_axes", None)
        if panel is None:
            return
        self.set_fontsize(self._ve_base_size)
        limit = 0.98 * float(panel.bbox.width)
        width = float(Text.get_window_extent(self, renderer).width)
        if limit > 0.0 and width > limit:
            self.set_fontsize(max(5.0, self._ve_base_size * limit / width))

    def current_ve(self) -> float | None:
        if not self._ve_auto:
            return self._ve_chosen
        return measured_vertical_exaggeration(self._ve_main_ax, self._ve_storage)

    def refresh_ve_text(self) -> None:
        caption = format_ve_caption(self.current_ve(), auto=self._ve_auto, style=self._ve_style)
        text = self._ve_template.replace("{ve}", caption)
        if text != self.get_text():
            self.set_text(text)

    def get_window_extent(self, renderer=None, dpi=None):
        if self._ve_auto:
            self.refresh_ve_text()
        self._fit_to_panel(renderer)
        return super().get_window_extent(renderer=renderer, dpi=dpi)

    def draw(self, renderer) -> None:
        if self._ve_auto:
            self.refresh_ve_text()
        self._fit_to_panel(renderer)
        super().draw(renderer)


class VECaptionAnnotation(Annotation, VECaptionText):
    """:class:`VECaptionText` placed like an annotation (e.g. under the axes)."""

    def __init__(
        self,
        xy,
        *,
        main_ax,
        storage_ve: float,
        auto: bool,
        chosen_ve: float | None,
        style: str = "consulting",
        template: str = "{ve}",
        **kwargs,
    ) -> None:
        self._ve_main_ax = main_ax
        self._ve_storage = float(storage_ve)
        self._ve_auto = bool(auto)
        self._ve_chosen = chosen_ve
        self._ve_style = style
        self._ve_template = template
        Annotation.__init__(self, "", xy, **kwargs)
        self.refresh_ve_text()

    def get_window_extent(self, renderer=None):
        if self._ve_auto:
            self.refresh_ve_text()
        return Annotation.get_window_extent(self, renderer)

    def draw(self, renderer) -> None:
        if self._ve_auto:
            self.refresh_ve_text()
        Annotation.draw(self, renderer)


# gid of the chemistry threshold key hung under the distance label.
THRESHOLD_KEY_GID = "chemistry-threshold-key"


def below_axes_anchor(ax):
    """Callable anchor: the axes' x range, bottomed under every x-axis text.

    The bottom is the lowest of the axes frame, its tick labels, the x label,
    a figure ``supxlabel`` and the chemistry threshold key, so an annotation
    placed at ``(x, 0)`` of it sits in the clear row below all of them and
    follows the axes through page resizes (evaluated at draw time).
    """
    from matplotlib.transforms import Bbox

    def _bbox(renderer):
        frame = ax.get_window_extent(renderer)
        bottom = frame.y0
        tight = ax.xaxis.get_tightbbox(renderer)
        if tight is not None and tight.height > 0:
            bottom = min(bottom, tight.y0)
        extras = [ax.xaxis.label, getattr(ax.figure, "_supxlabel", None)]
        extras += [a for a in ax.artists if a.get_gid() == THRESHOLD_KEY_GID]
        for artist in extras:
            if artist is None or not artist.get_visible():
                continue
            if isinstance(artist, Text) and not artist.get_text():
                continue
            box = artist.get_window_extent(renderer)
            if box.height > 0:
                bottom = min(bottom, box.y0)
        return Bbox.from_extents(frame.x0, bottom, frame.x1, frame.y1)

    return _bbox


def add_ve_caption(container, x: float, y: float, **kwargs) -> VECaptionText:
    """Add a :class:`VECaptionText` to an Axes or Figure (like ``.text``)."""
    from matplotlib.axes import Axes

    if isinstance(container, Axes):
        kwargs.setdefault("transform", container.transData)
        text = VECaptionText(x, y, **kwargs)
        text.set_clip_path(container.patch)
        container._add_text(text)
    else:
        kwargs.setdefault("transform", container.transSubfigure)
        text = VECaptionText(x, y, **kwargs)
        container.texts.append(text)
        text._remove_method = container.texts.remove
        text.set_figure(container)
        container.stale = True
    return text


def apply_ve_aspect(ax, *, exact: bool, twins: Sequence = (), anchor: str = "C") -> None:
    """Lock ``ax`` to the chosen VE (data stored at x VE) or let it fill its frame.

    Exact mode: aspect 1 in stored units with adjustable "box" — the plot
    shrinks inside its frame region, anchored at ``anchor``. Twinned axes (the
    consulting right-hand RL axis) are released from matplotlib's twin group
    (which forbids adjustable "box") and follow the main axes box instead.
    Auto mode keeps ``aspect="auto"`` (the axes fill their frame) unless that
    would compress the vertical (VE < 1): the box is then narrowed (centred)
    to exactly 1x — see :class:`AutoVEAspect`.
    """
    if not exact:
        ax.set_aspect("auto")
        release_twins(ax, twins)
        ax.apply_aspect = AutoVEAspect(ax)
        return
    if isinstance(ax.__dict__.get("apply_aspect"), AutoVEAspect):
        del ax.apply_aspect
    release_twins(ax, twins)
    ax.set_adjustable("box")
    ax.set_aspect(1.0, adjustable="box", anchor=anchor)


def release_twins(ax, twins: Sequence = ()) -> None:
    """Detach ``twins`` from ``ax``'s twin group; they follow ``ax``'s drawn box.

    Matplotlib's twin group copies one shared position to every member, which
    breaks any layout that sizes the main axes box itself (exact VE's
    adjustable "box", the consulting exact map scale). Each released twin is
    located on ``ax``'s axes box instead, so it tracks whatever box ``ax`` gets.
    """
    from matplotlib.axes._base import _TransformedBoundsLocator

    for twin in twins:
        if twin is None:
            continue
        group = ax._twinned_axes
        if group.joined(ax, twin):
            group.remove(twin)
            if len(group.get_siblings(ax)) <= 1:
                group.remove(ax)
        twin.set_adjustable("box")
        twin.set_aspect("auto")
        twin.set_axes_locator(_TransformedBoundsLocator([0, 0, 1, 1], ax.transAxes))


class AutoVEAspect:
    """``Axes.apply_aspect`` for auto VE: fill the frame, but never below 1x.

    Fit-to-page on a short, wide page (a short section on tabloid) gave a
    measured VE under 1 — the vertical *compressed*. A geological section is
    never drawn compressed, so the box is narrowed (centred in its frame) to
    exactly 1x instead, and the measured caption then reads "no vertical
    exaggeration". Data are stored at true scale in auto mode (factor 1).
    A position handed in by an axes locator (tight-bbox saves freeze the
    drawn box) is used as is.
    """

    def __init__(self, ax) -> None:
        self.ax = ax

    def __call__(self, position=None) -> None:
        from matplotlib.axes import Axes
        from matplotlib.transforms import Bbox

        ax = self.ax
        if position is not None:
            Axes.apply_aspect(ax, position)
            return
        frame = ax.get_position(original=True)
        Axes.apply_aspect(ax, frame)
        figure = ax.figure
        if figure is None:
            return
        width_in, height_in = (float(v) for v in figure.get_size_inches())
        box_w = frame.width * width_in
        box_h = frame.height * height_in
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        dx = abs(float(x1) - float(x0))
        dy = abs(float(y1) - float(y0))
        if min(box_w, box_h, dx, dy) <= 0.0 or not np.isfinite(dx + dy):
            return
        ve = (box_h / dy) / (box_w / dx)
        if ve >= 1.0 - 1e-9:
            return
        width = frame.width * ve
        ax._set_position(
            Bbox.from_bounds(frame.x0 + 0.5 * (frame.width - width), frame.y0, width, frame.height),
            which="active",
        )


# --- Page margins ------------------------------------------------------------

# Least printable margin (inches) every exported page keeps clear on all four
# sides; the export framing margins raise it per side.
MIN_PRINT_MARGIN_IN = 0.25


def page_margins_in(framing) -> tuple[float, float, float, float]:
    """``(left, right, top, bottom)`` printable margins for an export page."""
    if framing is None:
        return (MIN_PRINT_MARGIN_IN,) * 4
    return tuple(
        max(MIN_PRINT_MARGIN_IN, float(getattr(framing, f"margin_{side}_in", 0.0) or 0.0))
        for side in ("left", "right", "top", "bottom")
    )


def figure_content_bbox_in(fig, renderer=None):
    """Union of every drawn artist's extent, in inches (like ``bbox_inches="tight"``)."""
    if renderer is None:
        renderer = fig.canvas.get_renderer()
    return fig.get_tightbbox(renderer)


# --- X ticks that fit their box ------------------------------------------------


class FittedStepLocator(Locator):
    """Major x ticks at ``base`` m, or a coarser 1/2/5 step when labels would touch.

    A fixed 10 m step printed "0 102030" on a narrow exact-VE / map-scale box.
    The step is chosen at draw time from the drawn axes width and the measured
    width of the widest label, so every page size and box lock is covered.
    ``minor=True`` gives the matching unlabelled step (5 m up to a 20 m major).
    """

    def __init__(self, base: float | None = None, *, minor: bool = False, pad_pt: float = 6.0) -> None:
        self._base = float(base) if base and base > 0 else None
        self._minor = minor
        self._pad_pt = float(pad_pt)
        self._major_ref: FittedStepLocator | None = None

    @staticmethod
    def _steps_from(start: float):
        exponent = 10.0 ** np.floor(np.log10(max(start, 1e-9)))
        while True:
            for mantissa in (1.0, 2.0, 2.5, 5.0):
                step = mantissa * exponent
                if step >= start * (1.0 - 1e-9):
                    yield step
            exponent *= 10.0

    def _label_font(self):
        ticks = getattr(self.axis, "majorTicks", None) or []
        if ticks:
            return ticks[0].label1.get_fontproperties()
        from matplotlib.font_manager import FontProperties

        return FontProperties()

    def major_step(self, vmin: float, vmax: float) -> float:
        axes = self.axis.axes
        figure = axes.figure
        span = abs(float(vmax) - float(vmin))
        width_px = float(axes.bbox.width)
        if span <= 0.0 or width_px <= 0.0 or figure is None:
            return self._base or 1.0
        try:
            renderer = figure.canvas.get_renderer()
        except Exception:  # pragma: no cover - canvas without a renderer
            return self._base or 1.0
        prop = self._label_font()
        formatter = self.axis.get_major_formatter()
        low, high = sorted((float(vmin), float(vmax)))
        dpi = float(getattr(renderer, "dpi", figure.dpi))
        # Every draw, tick query and minor locator asks again for the same
        # box: answer from the last result while nothing it reads changed.
        # A stateful formatter (ScalarFormatter offset / magnitude) shows in
        # the labels it gives the range ends.
        probes = tuple(
            str(formatter(value)) if callable(formatter) else f"{value:g}"
            for value in (low, 0.5 * (low + high), high)
        )
        key = (low, high, width_px, float(figure.dpi), dpi, hash(prop), id(formatter), probes)
        cached = self.__dict__.get("_step_cache")
        if cached is not None and cached[0] == key:
            return cached[1]
        widths = self.__dict__.setdefault("_width_cache", {})
        width_key = (hash(prop), dpi)
        if widths.get("_key") != width_key:
            widths.clear()
            widths["_key"] = width_key

        def text_width(text: str) -> float:
            width = widths.get(text)
            if width is None:
                width = widths[text] = renderer.get_text_width_height_descent(text, prop, ismath=False)[0]
            return width

        px_per_unit = width_px / span
        pad_px = self._pad_pt * figure.dpi / 72.0
        start = self._base or _round_step_up(span / 12.0)
        step = start
        for count, step in enumerate(self._steps_from(start)):
            if step >= span or count > 30:
                break
            # A step narrower than the pad fails whatever its labels measure.
            if step * px_per_unit < pad_px:
                continue
            values = np.arange(np.ceil(low / step - 1e-9) * step, high + step * 1e-6, step)
            widest = 0.0
            for value in values[:200]:
                text = formatter(value) if callable(formatter) else f"{value:g}"
                if text:
                    widest = max(widest, text_width(str(text)))
            if step * px_per_unit >= widest + pad_px:
                break
        self._step_cache = (key, float(step))
        return float(step)

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        low, high = sorted((float(vmin), float(vmax)))
        major = self._major_ref or self
        step = major.major_step(low, high)
        if self._minor:
            if major._base is None:
                step = step / 2.0
            else:
                step = 5.0 if step <= 20.0 + 1e-9 else step / 5.0
        first = np.ceil(low / step - 1e-9) * step
        values = np.arange(first, high + step * 1e-6, step)
        return self.raise_if_exceeds(np.round(values, 9) + 0.0)


def fitted_x_locators(axis, base: float | None) -> tuple[FittedStepLocator, FittedStepLocator]:
    """Install a fitted major / minor x locator pair on ``axis``; returns them."""
    major = FittedStepLocator(base)
    minor = FittedStepLocator(base, minor=True)
    minor._major_ref = major
    major.set_axis(axis)
    minor.set_axis(axis)
    axis.set_major_locator(major)
    axis.set_minor_locator(minor)
    return major, minor


def _round_step_up(value: float) -> float:
    exponent = 10.0 ** np.floor(np.log10(max(value, 1e-9)))
    for mantissa in (1.0, 2.0, 2.5, 5.0, 10.0):
        if mantissa * exponent >= value * (1.0 - 1e-9):
            return mantissa * exponent
    return 10.0 * exponent  # pragma: no cover


# --- Long labels ------------------------------------------------------------------

# A hole ID / legend label longer than this wraps; past ``max_lines`` lines it
# is cut with an ellipsis (a 120-character ID ran past both page edges and
# shrank the section to a strip). 20 keeps paired IDs such as
# "2017-BH11 / BH24-11" on one line.
HOLE_LABEL_MAX_CHARS = 20
LEGEND_LABEL_MAX_CHARS = 32


# Label pieces: runs of whitespace, or text up to and including a '-', '_'
# or '/' (a line may break after one), or text with none of those.
_LABEL_PIECE_RE = re.compile(r"\s+|[^\s\-_/]*[-_/]|[^\s\-_/]+")


def _wrap_label(text: str, width: int) -> list[str]:
    """Lines of at most ``width`` characters: broken at spaces or after '-',
    '_' and '/' ("MW-2017-01-DEEP-" / "NESTED", never "...-NEST" / "ED");
    only a piece longer than the width breaks mid-token."""
    width = max(1, int(width))
    lines: list[str] = []
    current = ""
    for piece in _LABEL_PIECE_RE.findall(text):
        if piece.isspace():
            if current:
                current += " "
            continue
        if current.strip() and len(current) + len(piece) > width:
            lines.append(current.rstrip())
            current = ""
        while len(piece) > width:
            lines.append(piece[:width])
            piece = piece[width:]
        current += piece
    if current.strip():
        lines.append(current.rstrip())
    return lines


def fit_label(label: str, *, max_chars: int = HOLE_LABEL_MAX_CHARS, max_lines: int = 2) -> str:
    """``label`` wrapped at ``max_chars`` (at spaces or after '-' '_' '/',
    then hard breaks), at most ``max_lines`` lines, the last ellipsized when
    text was dropped."""
    text = str(label)
    if len(text) <= max_chars or "\n" in text:
        return text
    lines = _wrap_label(text, max_chars)
    if len(lines) <= max_lines:
        return "\n".join(lines)
    kept = lines[:max_lines]
    kept[-1] = kept[-1][: max_chars - 1].rstrip() + "\u2026"
    return "\n".join(kept)


# --- Hole-ID header stacking ------------------------------------------------------
#
# The first header pass (``renderer_water.resolve_header_collisions``) nudges
# each header sideways or one tier out. On a short section at exact VE (holes a
# few millimetres apart on the page) that is not enough and long IDs overprint.
# ``stack_hole_headers`` is the fallback: tiers outward with a leader tick down
# to the hole, wrapping "A / B" IDs, smaller text, and finally vertical text.

_HEADER_MAX_TIERS = 3
_HEADER_SHRINK = (1.0, 0.85, 0.75)
_HEADER_MIN_PT = 6.0
_HEADER_PAD_PT = 1.5
# Clear space kept between side-by-side horizontal headers: at ~0.06 in
# "BH23-04 2017-BH11 / BH24-11" read as one string.
HEADER_MIN_GAP_IN = 0.15
# Vertical (rotated) headers in a fan: a narrower gap reads apart.
_HEADER_FAN_GAP_PT = 4.0


def _header_extent(text, renderer):
    """Header text extent including its framed box.

    During ``stack_hole_headers`` (no draw happens inside it) extents are
    cached by the text's layout signature: the fallbacks try the same
    labels / sizes / offsets many times over.
    """
    figure = text.figure
    cache = figure.__dict__.get("_css_extent_cache") if figure is not None else None
    if cache is None:
        return _measure_header_extent(text, renderer)
    key = (text_layout_signature(text), id(renderer))
    box = cache.get(key)
    if box is None:
        box = cache[key] = _measure_header_extent(text, renderer)
    return box


def _measure_header_extent(text, renderer):
    box = text.get_window_extent(renderer)
    patch = text.get_bbox_patch()
    if patch is not None:
        text.update_bbox_position_size(renderer)
        from matplotlib.transforms import Bbox

        box = Bbox.union([box, patch.get_window_extent(renderer)])
    return box


def _boxes_overlap(a, b, pad: float = 0.0) -> bool:
    return a.x0 < b.x1 + pad and b.x0 < a.x1 + pad and a.y0 < b.y1 + pad and b.y0 < a.y1 + pad


def _is_horizontal(text) -> bool:
    return abs(float(text.get_rotation())) % 180.0 < 1.0


def _too_close(a, b, gap_x: float, pad: float) -> bool:
    """Boxes overlap within ``pad``, or sit side by side less than ``gap_x`` apart."""
    return a.x0 < b.x1 + gap_x and b.x0 < a.x1 + gap_x and a.y0 < b.y1 + pad and b.y0 < a.y1 + pad


def header_overlaps(
    fig, headers, renderer=None, *, pad_pt: float = 0.5, min_gap_in: float = 0.0
) -> list[tuple[int, int]]:
    """Index pairs of visible headers whose (framed) boxes touch.

    ``min_gap_in``: two horizontal headers on one row also count when the
    clear space between them is less than this (they read as one string).
    """
    if renderer is None:
        renderer = fig.canvas.get_renderer()
    pad = pad_pt * fig.dpi / 72.0
    gap = max(pad, min_gap_in * fig.dpi)
    shown = [t for t in headers if t.get_visible() and t.get_text().strip()]
    boxes = [_header_extent(t, renderer) for t in shown]
    flat = [_is_horizontal(t) for t in shown]
    return [
        (i, j)
        for i in range(len(boxes))
        for j in range(i + 1, len(boxes))
        if _too_close(boxes[i], boxes[j], gap if flat[i] and flat[j] else pad, pad)
    ]


def _header_leader_transform(ax, outward: float):
    """x: data; y: points from the axes' top (or bottom) edge."""
    from matplotlib.transforms import Affine2D, ScaledTranslation, blended_transform_factory

    edge = (0.0, 1.0) if outward > 0 else (0.0, 0.0)
    y_trans = Affine2D().scale(1.0 / 72.0) + ax.figure.dpi_scale_trans + ScaledTranslation(*edge, ax.transAxes)
    return blended_transform_factory(ax.transData, y_trans)


def _remove_header_leaders(fig) -> None:
    for line in fig.__dict__.pop("_css_header_leaders", []) or []:
        try:
            line.remove()
        except (ValueError, NotImplementedError):  # pragma: no cover - already gone
            pass


def _outward(text) -> float:
    """+1 for a header row above the plot (va bottom), -1 below it."""
    return 1.0 if text.get_verticalalignment() == "bottom" else -1.0


def _off_page(
    box,
    page,
    outward: float,
    *,
    horizontal: bool = False,
    flexible_top: bool = False,
    top_limit: float | None = None,
) -> bool:
    """Header box past the page edge it grows toward.

    Rows below the plot (section sheet) are exempt vertically: the sheet's
    bottom fit makes room under whatever the header pass draws. So are rows
    above the plot when a title layout engine lowers the plot to fit them
    (``flexible_top``: section sheet depth mode, chart) — up to
    ``top_limit`` (display y), the most the engine can lower the plot by
    while keeping the title on the page (``_header_top_limit``).
    """
    if horizontal and (box.x0 < page.x0 or box.x1 > page.x1):
        return True
    if outward > 0:
        if flexible_top:
            return top_limit is not None and box.y1 > top_limit
        return box.y1 > page.y1
    return False


def _header_top_limit(fig, headers) -> float | None:
    """Display y a header row above the plot may reach under a title layout
    engine (``renderer_chart._ChartTitleLayout.header_room_px``); None when
    the engine sets no bound."""
    room = getattr(fig.get_layout_engine(), "header_room_px", None)
    if room is None:
        return None
    room_px = room(fig)
    frame = next((t.axes.bbox for t in headers if t.axes is not None), None)
    if room_px is None or frame is None:
        return None
    return float(frame.y1 + room_px)


def _title_fits_headers(fig) -> bool:
    """True when the figure's layout engine places the title over the header row."""
    engine = fig.get_layout_engine()
    return engine is not None and hasattr(engine, "_headers") and hasattr(engine, "_title")


def _header_reads_as_own(text, renderer, page) -> bool:
    """On the page and still over its own hole (the nudge pass may slide a
    header 28 pt aside, onto a neighbour's column, with no leader)."""
    box = _header_extent(text, renderer)
    if box.x0 < page.x0 or box.x1 > page.x1:
        return False
    base = getattr(text, "_header_base", None)
    transform = base[0] if base is not None else text.get_transform()
    position = text.get_unitless_position()
    anchor_x, base_y = (float(v) for v in transform.transform(position))
    # Stepped a tier out with no leader: which column it names is a guess.
    if abs(float(text.get_transform().transform(position)[1]) - base_y) > 0.5:
        return False
    return box.x0 - 1.0 <= anchor_x <= box.x1 + 1.0


def restore_stacked_headers(headers) -> None:
    """Undo ``stack_hole_headers`` text / size changes (before a fresh header pass)."""
    for text in headers:
        if not hasattr(text, "_header_stack_orig"):
            text._header_stack_orig = (text.get_text(), float(text.get_fontsize()))
        if getattr(text, "_header_stacked", False):
            original_text, original_size = text._header_stack_orig
            text.set_text(original_text)
            text.set_fontsize(original_size)
            text.set_rotation(getattr(text, "_header_base", (None, None, 0.0))[2])
            text._header_stacked = False


def stack_hole_headers(fig, headers, *, max_tiers: int = _HEADER_MAX_TIERS, robust: bool = False) -> None:
    """Make overlapping hole headers readable (see ``_stack_hole_headers``)."""
    try:
        fig._css_extent_cache = {}
        _stack_hole_headers(fig, headers, max_tiers=max_tiers, robust=robust)
    finally:
        fig.__dict__.pop("_css_extent_cache", None)


def _stack_hole_headers(fig, headers, *, max_tiers: int = _HEADER_MAX_TIERS, robust: bool = False) -> None:
    """Make overlapping hole headers readable: tiers + leaders, wrap, shrink, rotate.

    Runs after the first (nudging) header pass; a no-op when no two headers
    touch and none leaves the page. Re-runnable (export page resizes): every
    run starts from the drawn text, size and position.
    """
    from matplotlib.lines import Line2D
    from matplotlib.transforms import offset_copy

    headers = [t for t in headers if t.figure is fig and t.get_visible() and t.get_text().strip()]
    _remove_header_leaders(fig)
    restore_stacked_headers(headers)
    if len(headers) < 2:
        return
    renderer = fig.canvas.get_renderer()
    settle_figure_layout(fig)
    page = fig.bbox
    if not header_overlaps(fig, headers, renderer, min_gap_in=HEADER_MIN_GAP_IN) and all(
        _header_reads_as_own(t, renderer, page) for t in headers
    ):
        return

    px_per_pt = fig.dpi / 72.0
    pad = _HEADER_PAD_PT * px_per_pt
    flexible_top = _title_fits_headers(fig)
    ordered = sorted(headers, key=lambda t: t.get_unitless_position()[0])
    # How high a row above the plot may reach under the title engine.
    top_limit = _header_top_limit(fig, ordered) if flexible_top else None
    # The nudge pass's layout, kept when nothing here does better.
    first_pass = [
        (t.get_transform(), t.get_horizontalalignment(), t.get_rotation(), t.get_text(), t.get_fontsize())
        for t in ordered
    ]
    first_cost = len(header_overlaps(fig, ordered, renderer, min_gap_in=HEADER_MIN_GAP_IN)) + sum(
        1
        for t in ordered
        if _off_page(_header_extent(t, renderer), page, _outward(t), horizontal=True)
    )
    base_states = []
    for text in ordered:
        base = getattr(text, "_header_base", None) or (
            text.get_transform(),
            text.get_horizontalalignment(),
            text.get_rotation(),
        )
        text._header_base = base
        base_states.append(base)
    obstacles = _header_stack_obstacles(fig, renderer, headers, skip_title=flexible_top)
    # RL / elevation axis labels beside the frame: a fan reaching past the
    # frame edge printed over the rotated left axis label.
    side_boxes = _axis_label_boxes(fig, renderer)
    obstacles += side_boxes

    def reset(text, base, *, label: str, size: float, rotation: float) -> None:
        text.set_text(label)
        text.set_fontsize(size)
        text.set_transform(base[0])
        text.set_horizontalalignment("center")
        text.set_rotation(rotation)
        text.set_rotation_mode("default")

    def anchor_px(text) -> float:
        return float(text.get_transform().transform(text.get_unitless_position())[0])

    gap_x = max(pad, HEADER_MIN_GAP_IN * fig.dpi)

    def try_tiers(labels, size) -> tuple[float, list[int]] | None:
        placed: list = list(obstacles)
        placed_headers: list = []
        leaders: list[tuple[float, float, float]] = []
        tiers: list[int] = []
        for text, base, label in zip(ordered, base_states, labels, strict=True):
            reset(text, base, label=label, size=size, rotation=0.0)
            box0 = _header_extent(text, renderer)
            outward = 1.0 if text.get_verticalalignment() == "bottom" else -1.0
            step = box0.height + 1.0 * px_per_pt
            x = anchor_px(text)
            # Leader span: from the frame (above the plot) or the home row
            # level under the tick labels (below it) out to the header.
            edge = text.axes.bbox.y1 if outward > 0 and text.axes is not None else (
                box0.y0 if outward > 0 else box0.y1
            )
            chosen = None
            for tier in range(max_tiers):
                box = box0.translated(0.0, outward * tier * step)
                if box.x0 < page.x0 or box.x1 > page.x1:
                    shift = (page.x0 - box.x0 + pad) if box.x0 < page.x0 else (page.x1 - box.x1 - pad)
                    box = box.translated(shift, 0.0)
                if _off_page(box, page, outward, flexible_top=flexible_top, top_limit=top_limit):
                    break
                if any(_boxes_overlap(box, other, pad) for other in placed):
                    continue
                if any(_too_close(box, other, gap_x, pad) for other in placed_headers):
                    continue
                if tier:
                    near = box.y0 if outward > 0 else box.y1
                    lo, hi = sorted((edge, near))
                    if any(o.x0 - pad < x < o.x1 + pad and o.y0 < hi and o.y1 > lo for o in placed):
                        continue
                if any(box.x0 - pad < lx < box.x1 + pad and box.y0 < ly1 and box.y1 > ly0 for lx, ly0, ly1 in leaders):
                    continue
                chosen = (tier, box)
                break
            if chosen is None:
                return None
            tier, box = chosen
            placed.append(box)
            placed_headers.append(box)
            if tier:
                near = box.y0 if outward > 0 else box.y1
                leaders.append((x, *sorted((edge, near))))
            tiers.append(tier)
        return 0.0, tiers

    originals = [t._header_stack_orig for t in ordered]
    base_size = max(size for _label, size in originals)
    wrapped = [label.replace(" / ", " /\n") for label, _size in originals]
    plain = [label for label, _size in originals]
    attempts = [(plain, 1.0)]
    if wrapped != plain:
        attempts.append((wrapped, 1.0))
    attempts += [(wrapped, f) for f in _HEADER_SHRINK[1:]]
    # ``robust``: the plot box kept moving between passes (the header height
    # feeds the title layout of an aspect-locked box); tiers and in-place
    # vertical text depend on the exact hole spacing, so go to the fan,
    # whose spacing comes from the header widths alone.
    for labels, factor in [] if robust else attempts:
        size = max(_HEADER_MIN_PT, base_size * factor)
        result = try_tiers(labels, size)
        if result is None:
            continue
        _apply_tiers(fig, ordered, base_states, labels, size, result[1], renderer, reset, Line2D, offset_copy)
        return
    # Vertical text (CAD convention) when every header clears its neighbours.
    rotated_cost = None
    for factor in _HEADER_SHRINK:
        size = max(_HEADER_MIN_PT, base_size * factor)
        for text, base, (label, _s) in zip(ordered, base_states, originals, strict=True):
            reset(text, base, label=label, size=size, rotation=90.0)
        boxes = [_header_extent(t, renderer) for t in ordered]
        cost = sum(
            1
            for i in range(len(boxes))
            for j in range(i + 1, len(boxes))
            # A clear gap, not just no overlap: a slightly narrower box
            # (the title layout reacting to the header height) must not
            # make neighbours touch.
            if _boxes_overlap(boxes[i], boxes[j], 2.0 * pad)
        ) + sum(
            1
            for t, box in zip(ordered, boxes, strict=True)
            if _off_page(box, page, _outward(t), flexible_top=flexible_top, top_limit=top_limit)
        )
        if rotated_cost is None or cost < rotated_cost[0]:
            rotated_cost = (cost, size)
        if cost == 0 and not robust:
            for text in ordered:
                text._header_stacked = True
            return
    # Holes closer than a text height: one row of headers fanned out above
    # the frame in hole order, each joined to its hole by a leader.
    margins = getattr(fig, "_css_page_margins_in", None) or (0.05,) * 4
    label_sets = [(plain, attempts)]
    # Multi-line headers (chart: ID / TD / RL): the hole ID alone, when the
    # full headers cannot be spread over the page width. The ID as drawn
    # (``_header_hole_id``, wrapped by ``fit_label``), not the first wrapped
    # line of it, which cut a long ID short.
    first_lines = [
        str(getattr(text, "_header_hole_id", "") or label.split("\n")[0])
        for text, label in zip(ordered, plain, strict=True)
    ]
    if first_lines != plain:
        first_wrapped = [label.replace(" / ", " /\n") for label in first_lines]
        first_attempts = [(first_lines, 1.0)]
        if first_wrapped != first_lines:
            first_attempts.append((first_wrapped, 1.0))
        first_attempts += [(first_wrapped, f) for f in _HEADER_SHRINK[1:]]
        label_sets.append((first_lines, first_attempts))
    tops = [flexible_top]
    if not flexible_top and getattr(fig, "_css_headers_lift_top", False):
        # A fixed page whose margin fit lowers the plot top for content past
        # the top margin: a fan taller than the room above the frame gets it.
        tops.append(True)
    for top_flexible in tops:
        for vertical_labels, horizontal_attempts in label_sets:
            for labels, factor in horizontal_attempts:
                size = max(_HEADER_MIN_PT, base_size * factor)
                if _apply_fan(
                    fig, ordered, base_states, labels, size, renderer, reset, obstacles, margins,
                    flexible_top=top_flexible, side_boxes=side_boxes, top_limit=top_limit,
                ):
                    if top_flexible and not flexible_top:
                        # Sticky: the page fit keeps the lowered top from now on.
                        fig._css_header_fan_lifted = True
                    return
            # Vertical headers fanned out (a text height wide each).
            for factor in _HEADER_SHRINK:
                size = max(_HEADER_MIN_PT, base_size * factor)
                if _apply_fan(
                    fig, ordered, base_states, vertical_labels, size, renderer, reset, obstacles, margins,
                    flexible_top=top_flexible, rotation=90.0, side_boxes=side_boxes, top_limit=top_limit,
                ):
                    if top_flexible and not flexible_top:
                        # Sticky: the page fit keeps the lowered top from now on.
                        fig._css_header_fan_lifted = True
                    return
    _cost, size = rotated_cost
    if first_cost <= _cost:
        for text, (transform, ha, rotation, label, fontsize) in zip(ordered, first_pass, strict=True):
            text.set_transform(transform)
            text.set_horizontalalignment(ha)
            text.set_rotation(rotation)
            text.set_text(label)
            text.set_fontsize(fontsize)
        return
    for text, base, (label, _s) in zip(ordered, base_states, originals, strict=True):
        reset(text, base, label=label, size=size, rotation=90.0)
        text._header_stacked = True


def _apply_fan(
    fig, ordered, base_states, labels, size, renderer, reset, obstacles, margins,
    *, flexible_top: bool = False, rotation: float = 0.0, side_boxes=(), top_limit: float | None = None,
) -> bool:
    """One row of headers spread in hole order with straight leaders; False if it can't fit.

    Each leader runs from the header row's home level (the frame edge above
    the plot; under the x tick labels below it) to its header, so it never
    crosses the tick labels.
    """
    from matplotlib.transforms import offset_copy

    px_per_pt = fig.dpi / 72.0
    gap = _HEADER_FAN_GAP_PT * px_per_pt if rotation else max(_HEADER_FAN_GAP_PT * px_per_pt, HEADER_MIN_GAP_IN * fig.dpi)
    page = fig.bbox
    lo = page.x0 + margins[0] * fig.dpi
    hi = page.x1 - margins[1] * fig.dpi
    boxes0, anchors = [], []
    for text, base, label in zip(ordered, base_states, labels, strict=True):
        reset(text, base, label=label, size=size, rotation=rotation)
        boxes0.append(_header_extent(text, renderer))
        anchors.append(float(text.get_transform().transform(text.get_unitless_position())[0]))
    outward0 = _outward(ordered[0])
    frame = ordered[0].axes.bbox if ordered[0].axes is not None else None
    # The fan row starts at least a header height + 8 pt out from its home row.
    reach = max(box.height for box in boxes0) + 8.0 * px_per_pt
    for box in side_boxes:
        # Axis labels reaching into the fan row's band bound the fan.
        if frame is not None and not (
            box.y1 > frame.y1 + reach if outward0 > 0 else box.y0 < frame.y0 - reach
        ):
            continue
        centre = 0.5 * (box.x0 + box.x1)
        if centre <= min(anchors):
            lo = max(lo, box.x1 + gap)
        elif centre >= max(anchors):
            hi = min(hi, box.x0 - gap)
    widths = [box.width + gap for box in boxes0]
    if sum(widths) > hi - lo:
        return False
    centres = list(anchors)
    for index in range(1, len(centres)):
        centres[index] = max(centres[index], centres[index - 1] + 0.5 * (widths[index - 1] + widths[index]))
    left_edge = centres[0] - 0.5 * widths[0]
    right_edge = centres[-1] + 0.5 * widths[-1]
    shift = 0.5 * (min(anchors) + max(anchors)) - 0.5 * (left_edge + right_edge)
    shift = max(shift, lo - (left_edge))
    shift = min(shift, hi - right_edge)
    centres = [c + shift for c in centres]
    outward = 1.0 if ordered[0].get_verticalalignment() == "bottom" else -1.0
    # Steeper leaders for a wider fan (flat ones read as underlines).
    spread_pt = max(abs(c - a) for c, a in zip(centres, anchors, strict=True)) / px_per_pt
    lift_pt = max(box.height for box in boxes0) / px_per_pt + min(max(8.0, 0.12 * spread_pt), 30.0)
    placed = []
    for box, centre, anchor in zip(boxes0, centres, anchors, strict=True):
        moved = box.translated(centre - anchor, outward * lift_pt * px_per_pt)
        if _off_page(moved, page, outward, flexible_top=flexible_top, top_limit=top_limit):
            return False
        if any(_boxes_overlap(moved, other, 0.0) for other in obstacles):
            return False
        placed.append(moved)
    leaders = []
    for text, base, box0, centre, anchor, moved in zip(
        ordered, base_states, boxes0, centres, anchors, placed, strict=True
    ):
        dx_pt = (centre - anchor) / px_per_pt
        text.set_transform(offset_copy(base[0], fig=fig, x=dx_pt, y=outward * lift_pt, units="points"))
        text._header_stacked = True
        ax = text.axes
        if ax is None:
            continue
        edge_px = ax.bbox.y1 if outward > 0 else ax.bbox.y0
        near_px = moved.y0 if outward > 0 else moved.y1
        home_px = box0.y0 if outward > 0 else box0.y1
        # Above the plot nothing sits between the frame and the header row;
        # below it the x tick labels do, so start under them.
        start_pt = 0.0 if outward > 0 else (home_px - edge_px) / px_per_pt
        leader = ax.annotate(
            "",
            xy=(float(text.get_unitless_position()[0]), start_pt),
            xycoords=_header_leader_transform(ax, outward),
            xytext=(dx_pt, (near_px - edge_px) / px_per_pt - start_pt - outward * 0.5),
            textcoords="offset points",
            arrowprops={
                "arrowstyle": "-",
                "color": text.get_color(),
                "lw": 0.5,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            },
            annotation_clip=False,
            zorder=text.get_zorder(),
        )
        leader.set_gid("hole-header-leader")
        leaders.append(leader)
    fig._css_header_leaders = leaders
    return True


def _apply_tiers(fig, ordered, base_states, labels, size, tiers, renderer, reset, line_cls, offset_copy) -> None:
    px_per_pt = fig.dpi / 72.0
    leaders = []
    page = fig.bbox
    for text, base, label, tier in zip(ordered, base_states, labels, tiers, strict=True):
        reset(text, base, label=label, size=size, rotation=0.0)
        text._header_stacked = True
        box0 = _header_extent(text, renderer)
        outward = 1.0 if text.get_verticalalignment() == "bottom" else -1.0
        step_pt = (box0.height + px_per_pt) / px_per_pt
        dx_pt = 0.0
        box = box0.translated(0.0, outward * tier * step_pt * px_per_pt)
        if box.x0 < page.x0:
            dx_pt = (page.x0 - box.x0) / px_per_pt + _HEADER_PAD_PT
        elif box.x1 > page.x1:
            dx_pt = (page.x1 - box.x1) / px_per_pt - _HEADER_PAD_PT
        if tier or dx_pt:
            text.set_transform(
                offset_copy(base[0], fig=fig, x=dx_pt, y=outward * tier * step_pt, units="points")
            )
        if not tier:
            continue
        ax = text.axes
        if ax is None:
            continue
        edge_px = ax.bbox.y1 if outward > 0 else ax.bbox.y0
        home_px = box0.y0 if outward > 0 else box0.y1
        end_px = home_px + outward * tier * step_pt * px_per_pt
        # Above the plot the leader starts at the frame (nothing between);
        # below it, under the x tick labels at the header row's home level.
        start_px = edge_px if outward > 0 else home_px
        start_pt = (start_px - edge_px) / px_per_pt
        end_pt = (end_px - edge_px) / px_per_pt
        x_data = float(text.get_unitless_position()[0])
        # A thin tick from the header row's home level (clear of the tick
        # labels) over the hole out to the stepped header.
        leader = line_cls(
            [x_data, x_data],
            [start_pt, end_pt - outward * 0.5],
            transform=_header_leader_transform(ax, outward),
            color=text.get_color(),
            linewidth=0.5,
            clip_on=False,
            zorder=text.get_zorder(),
        )
        leader.set_gid("hole-header-leader")
        ax.add_artist(leader)
        leaders.append(leader)
    fig._css_header_leaders = leaders


def _axis_label_boxes(fig, renderer) -> list:
    """Extents of the visible y-axis labels (RL / elevation) of every axes."""
    boxes = []
    for ax in fig.axes:
        if not ax.get_visible() or not ax.axison:
            continue
        label = ax.yaxis.label
        if label.get_visible() and label.get_text().strip():
            box = label.get_window_extent(renderer)
            if box.width > 0 and box.height > 0:
                boxes.append(box)
    return boxes


def _header_stack_obstacles(fig, renderer, headers, *, skip_title: bool = False) -> list:
    """Non-header axes text outside the plot frame (transect end labels, titles).

    ``skip_title``: the layout engine lifts the title over whatever the
    header rows reach, so it is not an obstacle.
    """
    skip = {id(t) for t in headers}
    boxes = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        frame = ax.bbox
        for artist in [*ax.texts, *([] if skip_title else [ax.title])]:
            if id(artist) in skip or not artist.get_visible() or not artist.get_text().strip():
                continue
            if isinstance(artist, Annotation):
                continue
            box = artist.get_window_extent(renderer)
            if box.y0 >= frame.y1 - 1.0 or box.y1 <= frame.y0 + 1.0:
                boxes.append(box)
    return boxes


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

# Depth ranges of a borehole with no lithology row (log gaps, no recovery, or
# an unlogged top / bottom of hole) show the plain grey column fill. They get
# their own legend key so the grey does not read as an unlisted lithology.
UNLOGGED_FILL_COLOR = CONSULTING_COLUMN_FILL
# "No Recovery" is a lithology code of its own in template 261002 (white);
# a depth with no row at all is simply not logged.
UNLOGGED_LEGEND_LABEL = "Not logged"
# Gaps thinner than this (metres) are rounding noise between logged rows.
UNLOGGED_MIN_THICKNESS_M = 0.01


def unlogged_intervals(
    projected_df: pd.DataFrame,
    collar_depths: dict[str, float] | None = None,
    collar_lookup: dict[str, float] | None = None,
    along_hole_tvd: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Return the unlogged depth ranges of each hole as elevation intervals.

    ``along_hole_tvd`` (surveyed holes) converts the total depth to true
    vertical depth; other holes use collar − total depth.

    Covers gaps between logged rows, a logged top below the collar, and a
    total depth (``collar_depths``) below the deepest logged row. Columns:
    ``hole_id``, ``x_profile``, ``collar_elevation``, ``top_elevation``,
    ``bottom_elevation`` (top above bottom).
    """
    columns = ["hole_id", "x_profile", "collar_elevation", "top_elevation", "bottom_elevation"]
    if projected_df is None or projected_df.empty:
        return pd.DataFrame(columns=columns)
    collar_depths = collar_depths or {}
    collar_lookup = collar_lookup or {}
    tol = UNLOGGED_MIN_THICKNESS_M
    rows: list[tuple[str, float, float, float, float]] = []
    for hole_id, hole_df in projected_df.groupby("hole_id", sort=False):
        hole = str(hole_id)
        x_profile = float(hole_df["x_profile"].iloc[0])
        collar = float(collar_lookup.get(hole, hole_df["collar_elevation"].iloc[0]))
        tops = np.maximum(
            hole_df["top_elevation"].to_numpy(dtype=float),
            hole_df["bottom_elevation"].to_numpy(dtype=float),
        )
        bottoms = np.minimum(
            hole_df["top_elevation"].to_numpy(dtype=float),
            hole_df["bottom_elevation"].to_numpy(dtype=float),
        )
        finite = np.isfinite(tops) & np.isfinite(bottoms)
        if not np.any(finite) or not np.isfinite(collar):
            continue
        order = np.argsort(-tops[finite], kind="stable")
        tops = tops[finite][order]
        bottoms = bottoms[finite][order]
        # Walk down the hole; ``reached`` is the deepest elevation logged so far.
        reached = collar
        for top, bottom in zip(tops, bottoms, strict=True):
            if reached - top > tol:
                rows.append((hole, x_profile, collar, reached, float(top)))
            reached = min(reached, float(bottom))
        td = collar_depths.get(hole)
        if td is not None and np.isfinite(td):
            td_elevation = float(
                hole_elevations_at_depths(along_hole_tvd, hole, [float(td)], collar)[0]
            )
            if reached - td_elevation > tol:
                rows.append((hole, x_profile, collar, reached, td_elevation))
    return pd.DataFrame(rows, columns=columns)


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



# Hatch marks in legend keys are drawn by ``swatch_hatch_artists`` at the
# section's marks per inch. The line width and colour match the section's
# hatch rc (``CrossSectionRenderer.render``), which a later re-fit outside
# that rc context would not see, so they are fixed here.
LEGEND_HATCH_LINEWIDTH_PT = 0.65
# Smallest swatch height (inches): one whole template mark (a 4.35 pt "+",
# a 5 pt-high silty dash) with its stroke and a clear gap to the border, and
# two staggered rows of the sand stipple (rows 1/14 in apart), so the key
# reads as the same texture as the section.
MIN_LEGEND_SWATCH_HEIGHT_IN = 8.0 / 72.0
PINCH_OUT_KEY_FACE = "#D1D5DB"


def swatch_hatch_artists(
    hatch: str | None,
    width_in: float,
    height_in: float,
    transform,
) -> list[PathPatch]:
    """Hatch marks for a legend swatch, identical to the section's hatch.

    ``transform`` maps swatch-local inches (origin at the swatch's lower-left
    corner) to display. Isolated marks are whole and centred; line hatches
    are clipped to the swatch. Matplotlib's own hatch would show a random,
    page-anchored slice of the 1 in pattern cell instead.
    """
    marks, lines = hatch_patterns.swatch_marks_path(hatch, width_in, height_in)
    linewidth = hatch_patterns.hatch_linewidth_for(hatch) or LEGEND_HATCH_LINEWIDTH_PT
    artists: list[PathPatch] = []
    for path, clip in ((marks, False), (lines, True)):
        if path is None:
            continue
        patch = PathPatch(
            path,
            facecolor=HATCH_LINE_COLOR,
            edgecolor=HATCH_LINE_COLOR,
            linewidth=linewidth,
            transform=transform,
        )
        patch.set_gid("legend-swatch-hatch")
        if clip:
            # A path clip (a Rectangle would become a clip *box*, which
            # ``Axes.add_patch`` then replaces with the axes box).
            box = Path([(0.0, 0.0), (width_in, 0.0), (width_in, height_in), (0.0, height_in), (0.0, 0.0)], closed=True)
            patch.set_clip_path(box, transform)
        artists.append(patch)
    return artists


def pinch_out_key_artist(width_in: float, height_in: float, transform, *, linewidth: float = 0.75) -> Polygon:
    """An inferred pinch-out key: a tapering wedge drawn like the section's.

    Same alpha and dashed edge as the faded wedges in the section (a grey box
    did not read as the wedge it keys).
    """
    wedge = Polygon(
        [(0.0, 0.0), (0.0, height_in), (width_in, height_in * 0.5)],
        closed=True,
        facecolor=PINCH_OUT_KEY_FACE,
        edgecolor=POLYGON_EDGE_COLOR,
        alpha=PINCH_OUT_ALPHA,
        linestyle="--",
        linewidth=linewidth,
        transform=transform,
    )
    wedge.set_gid("pinch-out-legend-key")
    return wedge


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
        # A separate stroke pass does not reproduce the stock hatched draw's
        # edge pixels (edges came out up to ~100/255 darker), so collections
        # with a visible outline keep the stock draw.
        edges = np.asarray(self.get_edgecolor(), dtype=float).reshape(-1, 4)
        widths = np.asarray(self.get_linewidth(), dtype=float)
        if edges.size and np.any(edges[:, 3] > 0) and np.any(widths > 0):
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


def _transform_signature(transform) -> tuple:
    """Structure and offsets of a text transform (``offset_copy`` chains are new
    objects each header pass; the axes / figure transforms they wrap are shared)."""
    from matplotlib.transforms import BlendedGenericTransform, ScaledTranslation

    if isinstance(transform, ScaledTranslation):
        return ("st", tuple(float(v) for v in transform._t), id(transform._scale_trans))
    a = getattr(transform, "_a", None)
    b = getattr(transform, "_b", None)
    if a is not None and b is not None:
        return ("c", _transform_signature(a), _transform_signature(b))
    if isinstance(transform, BlendedGenericTransform):
        return ("b", _transform_signature(transform._x), _transform_signature(transform._y))
    return ("id", id(transform))


def text_layout_signature(text) -> tuple:
    """What a text's drawn extent depends on, besides the axes it hangs from."""
    patch = text.get_bbox_patch()
    return (
        id(text),
        text.get_visible(),
        text.get_text(),
        float(text.get_fontsize()),
        float(text.get_rotation()),
        text.get_rotation_mode(),
        text.get_horizontalalignment(),
        text.get_verticalalignment(),
        tuple(float(v) for v in text.get_unitless_position()),
        _transform_signature(text.get_transform()),
        None if patch is None else id(patch),
    )


def _engine_fingerprint(engine, fig) -> tuple | None:
    """Inputs of a header / title layout engine (``renderer_chart._ChartTitleLayout``).

    Its ``execute`` reads only the title and the header extents relative to
    the plot top, so a figure whose headers and title are unchanged since its
    last settle lays out the same again. Other engines: unknown (None).
    """
    headers = getattr(engine, "_headers", None)
    if headers is None or not hasattr(engine, "_title"):
        return None
    return (
        id(engine),
        engine._title,
        repr(sorted(getattr(engine, "_title_kwargs", {}).items())),
        tuple(getattr(fig, "_css_page_margins_in", None) or ()),
        tuple(text_layout_signature(t) for t in headers if t.figure is fig),
    )


def _layout_fingerprint(fig) -> tuple | None:
    """Everything a no-render draw commits for tick, axis-label and title layout."""
    engine = fig.get_layout_engine()
    engine_key = None
    if engine is not None:
        # Layout engines reposition axes from artists' extents; only a known
        # header / title engine's inputs can be fingerprinted.
        engine_key = _engine_fingerprint(engine, fig)
        if engine_key is None:
            return None
    params = fig.subplotpars
    parts: list = [
        engine_key,
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


def draw_without_rendering_fast(fig) -> None:
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
    draw_without_rendering_fast(fig)
    fig._css_settled_layout = (_figure_draw_count(fig), _layout_fingerprint(fig))


# --- Chemistry / status legend keys for matplotlib legends ---------------------
#
# The consulting title-block legend draws its own keys; section sheets and the
# chart use a matplotlib ``Legend``. These proxies give them the same keys: the
# three G / O / R dots in threshold mode, the drawn marker in black / red mode,
# or a sample value in its colour when values print without markers.


class ThresholdDotsKey:
    """Legend proxy: one dot per threshold band colour."""

    def __init__(self, colors: Sequence[str], marker: str = "o") -> None:
        self.colors = tuple(colors)
        self.marker = marker


class SampleTextKey:
    """Legend proxy: a short sample text (``"120"``, ``"NM"``) in its colour."""

    def __init__(self, text: str, color: str, *, fontweight: str = "normal") -> None:
        self.text = str(text)
        self.color = color
        self.fontweight = fontweight


class _ThresholdDotsHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        from matplotlib.lines import Line2D

        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        count = max(len(orig_handle.colors), 1)
        artists = []
        for index, color in enumerate(orig_handle.colors):
            dot = Line2D(
                [x0 + width * (index + 0.5) / count],
                [y0 + height / 2.0],
                marker=orig_handle.marker,
                # Three dots share a 2 em key: 4 pt dots stay apart (5 pt
                # dots touched and read as one blob).
                markersize=min(5.0, 0.75 * width / count),
                markerfacecolor=color,
                markeredgecolor=color,
                linestyle="none",
                transform=handlebox.get_transform(),
            )
            dot.set_gid("parameter-legend-threshold-dots")
            handlebox.add_artist(dot)
            artists.append(dot)
        return artists[0] if artists else None


class _SampleTextHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        sample = Text(
            handlebox.xdescent + handlebox.width / 2.0,
            handlebox.ydescent + handlebox.height / 2.0,
            orig_handle.text,
            ha="center",
            va="center",
            color=orig_handle.color,
            fontsize=fontsize * 0.95,
            fontweight=orig_handle.fontweight,
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(sample)
        return sample


class PinchOutKey:
    """Legend proxy: the inferred pinch-out wedge."""


class _PinchOutHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        to_display = Affine2D().scale(72.0).translate(handlebox.xdescent, handlebox.ydescent)
        wedge = pinch_out_key_artist(
            handlebox.width / 72.0,
            handlebox.height / 72.0,
            to_display + handlebox.get_transform(),
        )
        handlebox.add_artist(wedge)
        return wedge


class _LithologySwatchHandler:
    """A lithology ``Patch`` key with its hatch drawn as whole, centred marks."""

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        rect = Rectangle(
            (x0, y0),
            width,
            height,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
            linewidth=orig_handle.get_linewidth(),
            linestyle=orig_handle.get_linestyle(),
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(rect)
        to_display = Affine2D().scale(72.0).translate(x0, y0) + handlebox.get_transform()
        for artist in swatch_hatch_artists(orig_handle.get_hatch(), width / 72.0, height / 72.0, to_display):
            handlebox.add_artist(artist)
        return rect


def _legend_handler_map() -> dict:
    from matplotlib.patches import Patch

    return {
        ThresholdDotsKey: _ThresholdDotsHandler(),
        SampleTextKey: _SampleTextHandler(),
        PinchOutKey: _PinchOutHandler(),
        Patch: _LithologySwatchHandler(),
    }


LEGEND_KEY_HANDLER_MAP = _legend_handler_map()
# Key height (em) for matplotlib legends holding lithology swatches: tall
# enough for one whole hatch mark at 8 pt (the default 0.7 em is not).
LEGEND_SWATCH_HANDLEHEIGHT = 1.0


_LEGEND_SOURCE_ATTR = "_css_source_keys"


def remember_legend_keys(legend, handles: Sequence[object], labels: Sequence[str]) -> None:
    """Keep the proxy ``(handle, label)`` pairs a legend was built from.

    ``Legend.legend_handles`` are the artists the handlers *drew*, not the
    proxies: a threshold-dots key comes back as one green dot and a
    ``SampleTextKey`` as a bare ``Text`` (which a new legend rejects). A
    rebuild reads these pairs instead.
    """
    setattr(legend, _LEGEND_SOURCE_ATTR, list(zip(handles, labels, strict=False)))


def legend_source_keys(legend) -> list[tuple[object, str]]:
    """``(handle, label)`` pairs to rebuild ``legend`` from (see ``remember_legend_keys``)."""
    stored = getattr(legend, _LEGEND_SOURCE_ATTR, None)
    if stored is not None:
        return list(stored)
    return [
        (handle, text.get_text())
        for handle, text in zip(legend.legend_handles, legend.get_texts(), strict=False)
        if handle is not None
    ]


def parameter_legend_keys(entries, *, draw_markers: bool, upper: bool = False) -> list[tuple[object, str]]:
    """``(handle, label)`` per chemistry legend entry (``parameter_series_legend``)."""
    from matplotlib.lines import Line2D

    keys: list[tuple[object, str]] = []
    for entry in entries or ():
        label = str(entry.get("label") or entry.get("parameter") or "Parameter")
        label = label.upper() if upper else label
        # Each parameter's own marker shape (set per entry by the chemistry
        # pass) keys its samples, with or without markers on the section.
        marker = entry.get("marker") or "o"
        if entry.get("threshold_colors"):
            keys.append((ThresholdDotsKey(entry["threshold_colors"], marker=marker), label))
        elif entry.get("sample_text"):
            color = entry.get("sample_color") or entry.get("color") or "#111827"
            keys.append((SampleTextKey(entry["sample_text"], color), label))
        else:
            color = entry.get("sample_color") or entry.get("color") or "#111827"
            handle = Line2D([], [], color=color, marker=marker, markersize=6, linestyle="none")
            keys.append((handle, label))
    return keys


def water_status_texts(renderer_obj, fig) -> list[str]:
    """Distinct NM / DRY water-status notes drawn on ``fig`` (e.g. ``["NM", "DRY"]``)."""
    notes = getattr(renderer_obj, "__dict__", {}).get("_water_status_notes") or {}
    found: set[str] = set()
    for annotation in notes.values():
        if annotation is None or annotation.figure is not fig or not annotation.get_visible():
            continue
        for part in annotation.get_text().split(" / "):
            if part.strip():
                found.add(part.strip().upper())
    order = {"NM": 0, "DRY": 1}
    return sorted(found, key=lambda part: order.get(part, 2))


def water_status_legend_label(parts: Sequence[str], *, upper: bool = False) -> str:
    """``"NM / DRY — not measured / dry"`` for the status notes actually drawn."""
    words = {"NM": "not measured", "DRY": "dry"}
    label = " / ".join(parts) + " — " + " / ".join(words.get(p, p.lower()) for p in parts)
    return label.upper() if upper else label


def _along_hole_tvd_table(along_hole_tvd: Mapping[str, object] | None, hole_id: str):
    if not along_hole_tvd:
        return None
    return along_hole_tvd.get(str(hole_id))


def hole_elevations_at_depths(
    along_hole_tvd: Mapping[str, object] | None,
    hole_id: str,
    depths: Sequence[float] | np.ndarray,
    collar_rl: float | np.ndarray,
) -> np.ndarray:
    """Elevations (masl) of measured depths down ``hole_id``.

    Surveyed holes use their MD→TVD table (``projection.AlongHoleTVD``);
    vertical holes keep the exact legacy ``collar − depth``.
    """
    depth_arr = np.asarray(depths, dtype=float)
    table = _along_hole_tvd_table(along_hole_tvd, hole_id)
    if table is None:
        return collar_rl - depth_arr
    return collar_rl - table.tvd_many(depth_arr)


class RendererGeometryMixin:
    """Well extents, profile lookup, and lithology style resolution."""

    def _hole_elevations_at_depths(
        self,
        hole_id: str,
        depths: Sequence[float] | np.ndarray,
        collar_rl: float | np.ndarray,
    ) -> np.ndarray:
        return hole_elevations_at_depths(
            getattr(self, "along_hole_tvd", None), hole_id, depths, collar_rl
        )

    def _hole_elevation_at_depth(self, hole_id: str, depth: float, collar_rl: float) -> float:
        table = _along_hole_tvd_table(getattr(self, "along_hole_tvd", None), hole_id)
        if table is None:
            return collar_rl - depth
        return float(collar_rl) - table.tvd_at(float(depth))

    def _md_to_elevation(self, hole_id: str, collar_rl: float) -> Callable[[float], float] | None:
        """``md → elevation`` for a surveyed hole, ``None`` for a vertical one."""
        table = _along_hole_tvd_table(getattr(self, "along_hole_tvd", None), hole_id)
        if table is None:
            return None
        collar = float(collar_rl)
        return lambda md: collar - table.tvd_at(float(md))

    def _md_to_elevation_by_hole(
        self, collar_rl_by_hole: Mapping[str, float]
    ) -> dict[str, Callable[[float], float]]:
        return {
            hole_id: fn
            for hole_id, collar_rl in collar_rl_by_hole.items()
            if (fn := self._md_to_elevation(hole_id, collar_rl)) is not None
        }

    def _water_head(self, level: WaterLevel, collar_rl: float) -> float:
        """Head (masl) via ``hydro_metrics.water_head_masl`` with stick-up and survey."""
        stick_ups = getattr(self, "stick_up_by_hole", None) or {}
        return water_head_masl(
            level,
            collar_rl,
            stick_up_m=stick_ups.get(str(level.hole_id)),
            md_to_elevation=self._md_to_elevation(level.hole_id, collar_rl),
        )

    def _water_depth_below_ground(self, level: WaterLevel, collar_rl: float) -> float:
        """Vertical depth of the head below ground (the "mbgs" label value)."""
        stick_ups = getattr(self, "stick_up_by_hole", None) or {}
        return water_depth_below_ground_m(
            level,
            collar_rl,
            stick_up_m=stick_ups.get(str(level.hole_id)),
            md_to_elevation=self._md_to_elevation(level.hole_id, collar_rl),
        )

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
        td_elev = collars - td_values
        tvd_tables = getattr(self, "along_hole_tvd", None) or {}
        for index, hole_id in enumerate(hole_ids):
            # Surveyed holes end at the true vertical depth of TD.
            if str(hole_id) in tvd_tables and np.isfinite(td_values[index]):
                td_elev[index] = self._hole_elevation_at_depth(
                    str(hole_id), float(td_values[index]), float(collars[index])
                )
        bottom_elev = np.where(np.isfinite(td_values), td_elev, row_bottoms)
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

    def _draw_unlogged_intervals(
        self,
        ax,
        projected_df: pd.DataFrame,
        collar_depths: dict[str, float] | None,
        collar_lookup: dict[str, float],
        track_half_width: float,
        *,
        zorder: int = 5,
        draw: bool = True,
    ) -> bool:
        """Fill unlogged column depths grey; record whether any exist for the legend.

        With ``draw=False`` only the legend flag is set (consulting columns are
        already grey underneath the logged intervals).
        """
        gaps = unlogged_intervals(
            projected_df,
            collar_depths,
            collar_lookup,
            getattr(self, "along_hole_tvd", None),
        )
        found = not gaps.empty
        self._has_unlogged_intervals = bool(getattr(self, "_has_unlogged_intervals", False) or found)
        if not found or not draw:
            return found
        collars = gaps["collar_elevation"].to_numpy(dtype=float)
        tops = self._plot_y_values(gaps["top_elevation"].to_numpy(dtype=float), collars)
        bottoms = self._plot_y_values(gaps["bottom_elevation"].to_numpy(dtype=float), collars)
        count = len(gaps)
        self._add_rect_collection(
            ax,
            (
                gaps["x_profile"].to_numpy(dtype=float) - track_half_width,
                np.minimum(tops, bottoms),
                np.full(count, 2.0 * track_half_width),
                np.abs(tops - bottoms),
            ),
            facecolors=UNLOGGED_FILL_COLOR,
            edgecolors=TRACK_BORDER_COLOR,
            linewidths=0.6,
            zorder=zorder,
        )
        return found

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
        top_elevs: list[float] = []
        bottom_elevs: list[float] = []
        for interval in screen_intervals:
            profile = profile_lookup.get(interval.hole_id)
            if profile is None:
                continue
            x_profile, collar_rl = profile
            x_profiles.append(x_profile)
            collar_rls.append(collar_rl)
            # Screen depths are along the hole: surveyed holes use TVD.
            top_elevs.append(
                self._hole_elevation_at_depth(interval.hole_id, interval.from_depth, collar_rl)
            )
            bottom_elevs.append(
                self._hole_elevation_at_depth(interval.hole_id, interval.to_depth, collar_rl)
            )
        if not x_profiles:
            return
        x_arr = np.asarray(x_profiles, dtype=float)
        collar_arr = np.asarray(collar_rls, dtype=float)
        top_ys = self._plot_y_values(np.asarray(top_elevs, dtype=float), collar_arr)
        bottom_ys = self._plot_y_values(np.asarray(bottom_elevs, dtype=float), collar_arr)
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

