"""Hatch marks drawn like the client CAD legend (Cross_Section_Litho_Legend_261002).

Matplotlib's built-in hatches do not match the template:

* ``.`` draws large dots on a 1/6 in grid; the template stipple is fine
  (~0.8 pt dots, ~5 pt apart, staggered).
* ``/`` draws continuous diagonals; the template's silty pattern is short
  45° dashes (~7 pt long, ~9 pt apart, staggered).
* ``+`` draws a full grid; the template's clay-loam pattern is isolated small
  ``+`` marks (4.35 pt strokes, ~12 pt apart, staggered).

``install()`` swaps the matplotlib pattern classes for those three characters
(every other character keeps its stock pattern; ``x`` keeps continuous
diagonals). The hatch strings themselves are unchanged, so they stay valid
matplotlib hatches, repeat-to-densify still works, and SVG/PDF/PNG all draw
the same path. Matplotlib exposes no public custom-hatch API yet, so this
patches ``matplotlib.hatch._hatch_types``; ``tests/test_hatch_patterns.py``
guards it.
"""

from __future__ import annotations

import functools

import matplotlib.hatch as mhatch
import numpy as np
from matplotlib.path import Path

# Marks per inch along each axis for ONE pattern character at matplotlib's
# default hatch density (6). Matplotlib's hatch unit cell is one inch.
TEMPLATE_ROWS_PER_INCH: dict[str, int] = {".": 14, "/": 8, "+": 6, "O": 8}
STOCK_ROWS_PER_INCH = 6

# Mark sizes, in inches (template measurements at 1:1).
_DOT_RADIUS_IN = 0.42 / 72.0
_DASH_HALF_SPAN_IN = 2.52 / 72.0  # each axis: 5.04 pt box -> ~7.1 pt dash
_PLUS_ARM_IN = 4.35 / 2 / 72.0  # template "+" strokes are 4.35 pt long

_DEFAULT_DENSITY = 6

# Gravel outlines are drawn thinner than other hatch marks: at the smaller
# legend-size stones the shared 0.65 pt hatch line filled them in (the
# template uses 0.42 pt lines on 17 pt stones).
GRAVEL_HATCH_LINEWIDTH_PT = 0.3


def hatch_linewidth_for(hatch: str | None) -> float | None:
    """Hatch line width override for *hatch* (``None`` = use the rc default)."""
    if hatch and "O" in hatch:
        return GRAVEL_HATCH_LINEWIDTH_PT
    return None

# Gravel cobbles: one repeat tile of the template's stone outlines, traced
# from Cross_Section_Litho_Legend_261002 (17.1 pt square tile). Drawn at 8
# tiles per inch (9 pt) in sections and legend swatches alike, so the figure
# shows the stones the legend shows; whole tiles fit the 1 in cell. Each
# entry is a straight segment (x0, y0, x1, y1) in tile units, y up; outlines
# that cross the tile edge continue in the neighbouring tile.
COBBLE_TILE_SEGMENTS: tuple[tuple[float, float, float, float], ...] = (
    (0.042, 0.157, 0.212, 0.106),
    (0.053, 0.438, 0.212, 0.487),
    (0.074, 0.746, 0.204, 0.657),
    (0.133, 0.987, -0.067, 1.017),
    (0.163, 0.517, -0.047, 0.408),
    (0.163, 0.768, 0.212, 0.857),
    (0.204, 0.657, 0.163, 0.517),
    (0.212, 0.106, 0.382, 0.157),
    (0.212, 0.487, 0.374, 0.427),
    (0.212, 0.717, 0.293, 0.527),
    (0.212, 0.857, 0.212, 0.917),
    (0.212, 0.917, 0.133, 0.987),
    (0.223, 0.927, 0.212, 0.717),
    (0.293, 0.527, 0.504, 0.517),
    (0.374, 0.427, 0.433, 0.297),
    (0.382, 0.157, 0.433, 0.297),
    (0.453, 0.138, 0.223, -0.073),
    (0.504, 0.297, 0.533, 0.376),
    (0.504, 0.517, 0.723, 0.687),
    (0.612, 0.176, 0.504, 0.297),
    (0.633, 0.546, 0.814, 0.746),
    (0.693, 0.117, 0.453, 0.138),
    (0.704, 0.427, 0.633, 0.546),
    (0.723, 0.397, 0.533, 0.376),
    (0.723, 0.687, 0.804, 0.938),
    (0.793, 0.106, 0.833, 0.006),
    (0.804, 0.938, 0.693, 1.117),
    (0.814, 0.746, 1.074, 0.746),
    (0.833, 0.006, 0.982, 0.097),
    (0.833, 0.917, 0.914, 0.797),
    (0.844, 0.197, 0.612, 0.176),
    (0.863, 0.138, 0.793, 0.106),
    (0.914, 0.797, 1.163, 0.768),
    (0.923, 0.346, 0.723, 0.397),
    (0.933, 0.017, 0.833, -0.083),
    (0.933, 0.297, 0.844, 0.197),
    (0.933, 0.297, 0.923, 0.346),
    (0.953, 0.408, 0.704, 0.427),
    (0.982, 0.097, 0.863, 0.138),
    (0.982, 0.297, 1.044, 0.157),
    (0.993, 0.297, 1.053, 0.438),
)



def rows_per_inch(hatch: str | None) -> int:
    """Mark rows per inch for one character of *hatch*'s dominant mark."""
    for char in hatch or "":
        if char in TEMPLATE_ROWS_PER_INCH:
            return TEMPLATE_ROWS_PER_INCH[char]
    return STOCK_ROWS_PER_INCH


def _rows(hatch: str, char: str, density: int) -> int:
    count = hatch.count(char)
    if not count:
        return 0
    return max(1, round(count * density * TEMPLATE_ROWS_PER_INCH[char] / _DEFAULT_DENSITY))


class _TemplateShapes(mhatch.Shapes):
    """Staggered grid of one mark shape (matplotlib's Shapes layout)."""

    char = ""
    half_size_in = 0.0

    def __init__(self, hatch, density):
        self.num_rows = _rows(hatch, self.char, density)
        if self.num_rows:
            # Shapes scales vertices by (1 / num_rows) * size.
            self.size = self.half_size_in * self.num_rows
        super().__init__(hatch, density)


class TemplateDots(_TemplateShapes):
    """Fine stipple for sandy units (``.``)."""

    char = "."
    half_size_in = _DOT_RADIUS_IN
    filled = True

    def __init__(self, hatch, density):
        circle = Path.unit_circle()
        self.shape_vertices = circle.vertices
        self.shape_codes = circle.codes
        super().__init__(hatch, density)


class TemplateDashes(_TemplateShapes):
    """Short 45° dashes for silty units (``/``)."""

    char = "/"
    half_size_in = _DASH_HALF_SPAN_IN
    filled = True  # open polyline: stroke it once, no reversed copy

    def __init__(self, hatch, density):
        self.shape_vertices = np.array([[-1.0, -1.0], [1.0, 1.0]])
        self.shape_codes = np.array([Path.MOVETO, Path.LINETO], dtype=Path.code_type)
        super().__init__(hatch, density)


class TemplatePlus(_TemplateShapes):
    """Isolated small ``+`` marks for clay-loam mixes (``+``)."""

    char = "+"
    half_size_in = _PLUS_ARM_IN
    filled = True  # two open strokes

    def __init__(self, hatch, density):
        self.shape_vertices = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
        self.shape_codes = np.array(
            [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO], dtype=Path.code_type
        )
        super().__init__(hatch, density)



class TemplateCobbles(mhatch.HatchPatternBase):
    """Irregular stone outlines for gravel (``O``), as on the template."""

    def __init__(self, hatch, density):
        count = hatch.count("O")
        self.num_tiles = (
            max(1, round(count * density * TEMPLATE_ROWS_PER_INCH["O"] / _DEFAULT_DENSITY))
            if count
            else 0
        )
        # One extra tile on each side so outlines crossing the cell edge are
        # complete once the backend clips the hatch cell.
        span = self.num_tiles + 2
        self.num_vertices = len(COBBLE_TILE_SEGMENTS) * 2 * span * span if count else 0

    def set_vertices_and_codes(self, vertices, codes):
        size = 1.0 / self.num_tiles
        segs = np.asarray(COBBLE_TILE_SEGMENTS, dtype=float).reshape(-1, 2, 2)
        parts = []
        for ix in range(-1, self.num_tiles + 1):
            for iy in range(-1, self.num_tiles + 1):
                parts.append((segs + (ix, iy)) * size)
        vertices[:] = np.concatenate(parts).reshape(-1, 2)
        codes[0::2] = Path.MOVETO
        codes[1::2] = Path.LINETO

class _HorizontalOnly(mhatch.HorizontalHatch):
    """Stock horizontal lines for ``-`` only (``+`` is now a mark)."""

    def __init__(self, hatch, density):
        super().__init__(hatch.replace("+", ""), density)


class _VerticalOnly(mhatch.VerticalHatch):
    """Stock vertical lines for ``|`` only (``+`` is now a mark)."""

    def __init__(self, hatch, density):
        super().__init__(hatch.replace("+", ""), density)


class _NorthEastLinesOnly(mhatch.NorthEastHatch):
    """Stock continuous diagonals for ``x``/``X`` only (``/`` is now dashes)."""

    def __init__(self, hatch, density):
        super().__init__(hatch.replace("/", ""), density)


_REPLACEMENTS = {
    mhatch.HorizontalHatch: (_HorizontalOnly,),
    mhatch.VerticalHatch: (_VerticalOnly,),
    mhatch.NorthEastHatch: (_NorthEastLinesOnly, TemplateDashes),
    mhatch.SmallFilledCircles: (TemplateDots,),
    mhatch.LargeCircles: (TemplateCobbles,),
}
_TEMPLATE_TYPES = (
    _HorizontalOnly,
    _VerticalOnly,
    _NorthEastLinesOnly,
    TemplateDashes,
    TemplateDots,
    TemplatePlus,
    TemplateCobbles,
)


def is_installed() -> bool:
    return all(cls in mhatch._hatch_types for cls in _TEMPLATE_TYPES)


def _with_gravel_linewidth(set_hatch):
    """Wrap an artist's ``set_hatch`` to apply :func:`hatch_linewidth_for`."""
    if getattr(set_hatch, "_template_gravel_linewidth", False):
        return set_hatch

    def wrapped(self, hatch):
        set_hatch(self, hatch)
        linewidth = hatch_linewidth_for(hatch)
        if linewidth is not None:
            self._hatch_linewidth = linewidth
        elif getattr(self, "_hatch_linewidth", None) == GRAVEL_HATCH_LINEWIDTH_PT:
            # Hatch changed away from gravel: back to the normal hatch line.
            import matplotlib

            self._hatch_linewidth = matplotlib.rcParams["hatch.linewidth"]

    wrapped._template_gravel_linewidth = True
    wrapped.__doc__ = set_hatch.__doc__
    return wrapped


def install() -> None:
    """Use the template marks for ``.``, ``/``, ``+`` and ``O`` (idempotent).

    Also gives gravel (``O``) hatches their thinner outline on every patch and
    collection, so section fills, legend swatches and exports match.
    """
    from matplotlib.collections import Collection
    from matplotlib.patches import Patch

    Patch.set_hatch = _with_gravel_linewidth(Patch.set_hatch)
    Collection.set_hatch = _with_gravel_linewidth(Collection.set_hatch)
    if is_installed():
        return
    types: list[type] = []
    for cls in mhatch._hatch_types:
        types.extend(_REPLACEMENTS.get(cls, (cls,)))
    types.append(TemplatePlus)
    mhatch._hatch_types[:] = types


# --- Legend swatch marks ------------------------------------------------------
#
# A matplotlib hatch is anchored to the page, so a small legend swatch shows
# whatever part of the 1 in cell happens to fall inside it: marks cut by the
# border, a "+" reduced to a stub, or none at all on a short swatch. Legend
# keys therefore draw the same hatch path explicitly: the identical marks at
# the identical marks-per-inch (``get_path(hatch, 6)`` on a 1 in cell, exactly
# what the backends tile in the section), with isolated marks kept only when
# whole and the pattern centred in the swatch. Continuous line hatches
# (``-``, ``|``, ``x``, gravel outlines) are clipped to the swatch instead.

_ISOLATED_MARK_MAX_IN = 0.25  # a subpath group wider than this is a line hatch
_MARK_PHASE_STEP_IN = 0.5 / 72.0


def _subpaths(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    verts, codes = path.vertices, path.codes
    if codes is None:
        return [(verts, np.full(len(verts), Path.LINETO, dtype=Path.code_type))]
    starts = list(np.flatnonzero(codes == Path.MOVETO)) + [len(codes)]
    return [(verts[a:b], codes[a:b]) for a, b in zip(starts[:-1], starts[1:], strict=False) if b > a]


def _mark_groups(path: Path) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    """Split a hatch cell path into isolated marks (grouped by centre) and line pieces."""
    marks: dict[tuple[float, float], list[tuple[np.ndarray, np.ndarray]]] = {}
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for verts, codes in _subpaths(path):
        drawn = verts[codes != Path.CLOSEPOLY] if len(verts) > 1 else verts
        lo, hi = drawn.min(axis=0), drawn.max(axis=0)
        if float(np.max(hi - lo)) > _ISOLATED_MARK_MAX_IN:
            lines.append((verts, codes))
            continue
        centre = tuple(np.round((lo + hi) / 2.0, 5))
        marks.setdefault(centre, []).append((verts, codes))
    groups = []
    for parts in marks.values():
        verts = np.concatenate([v for v, _c in parts])
        codes = np.concatenate([c for _v, c in parts])
        groups.append((verts, codes))
    return groups, lines


@functools.lru_cache(maxsize=256)
def swatch_marks_path(
    hatch: str | None,
    width_in: float,
    height_in: float,
    *,
    inset_in: float = 0.9 / 72.0,
    density: int = _DEFAULT_DENSITY,
) -> tuple[Path | None, Path | None]:
    """Hatch marks for a ``width_in`` x ``height_in`` legend swatch, in inches.

    Returns ``(marks, lines)``, both in swatch coordinates (origin at the
    swatch's lower-left corner, inches) or ``None``: ``marks`` holds only whole
    isolated marks (dots, dashes, ``+``) at the section's marks per inch,
    phased for the most whole marks and centred; ``lines`` holds continuous
    hatch pieces to clip to the swatch. Both draw like a hatch: filled and
    stroked in the hatch colour.
    """
    import matplotlib.hatch as mh

    if not hatch or width_in <= 0 or height_in <= 0:
        return None, None
    cell = mh.get_path(hatch, density)
    if cell is None or len(cell.vertices) == 0:
        return None, None
    groups, lines = _mark_groups(cell)
    if "O" in hatch:
        # Gravel outlines are short segments of stones that cross the cell
        # edge: clip them like the section does, never drop pieces.
        lines = lines + groups
        groups = []
    nx = int(np.ceil(width_in)) + 2
    ny = int(np.ceil(height_in)) + 2

    def tiled(parts):
        """Line pieces, tiled over the swatch, that reach inside it."""
        if parts and all(len(v) == 2 for v, _c in parts):
            # Straight segments (gravel outlines, stock lines): vectorised.
            segs = np.stack([v for v, _c in parts])  # (n, 2, 2)
            shifts = np.array([(ix, iy) for ix in range(-1, nx) for iy in range(-1, ny)], dtype=float)
            moved = (segs[None, :, :, :] + shifts[:, None, None, :]).reshape(-1, 2, 2)
            lo, hi = moved.min(axis=1), moved.max(axis=1)
            keep = (hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] <= width_in) & (lo[:, 1] <= height_in)
            moved = np.unique(np.round(moved[keep], 6), axis=0)
            codes = np.array([Path.MOVETO, Path.LINETO], dtype=Path.code_type)
            return [(seg, codes) for seg in moved]
        out = []
        seen = set()
        for ix in range(-1, nx):
            for iy in range(-1, ny):
                for verts, codes in parts:
                    moved = verts + (ix, iy)
                    lo, hi = moved.min(axis=0), moved.max(axis=0)
                    if hi[0] < 0 or hi[1] < 0 or lo[0] > width_in or lo[1] > height_in:
                        continue
                    key = tuple(np.round(moved, 5).ravel())
                    if key in seen:  # padded cells repeat their neighbours' pieces
                        continue
                    seen.add(key)
                    out.append((moved, codes))
        return out

    marks_path = None
    if groups:
        centres = []
        halves = []
        for verts, codes in groups:
            drawn = verts[codes != Path.CLOSEPOLY]
            lo, hi = drawn.min(axis=0), drawn.max(axis=0)
            centres.append((lo + hi) / 2.0)
            halves.append((hi - lo) / 2.0)
        centres = np.asarray(centres)
        halves = np.asarray(halves)
        offsets = np.array([(ix, iy) for ix in range(-1, nx) for iy in range(-1, ny)], dtype=float)
        all_c = (centres[None, :, :] + offsets[:, None, :]).reshape(-1, 2)
        all_h = np.broadcast_to(halves[None, :, :], (len(offsets), len(halves), 2)).reshape(-1, 2)
        all_i = np.arange(len(all_c))
        pitch = 1.0 / max(rows_per_inch(hatch), 1)
        steps = np.arange(0.0, 2.0 * pitch, _MARK_PHASE_STEP_IN)
        # Only marks that some phase shift can bring into the swatch.
        reach = 2.0 * pitch
        near = np.all(
            (all_c >= (-reach, -reach)) & (all_c <= (width_in + reach, height_in + reach)), axis=1
        )
        all_c, all_h, all_i = all_c[near], all_h[near], all_i[near]
        lo_lim = np.array([inset_in, inset_in])
        hi_lim = np.array([width_in - inset_in, height_in - inset_in])
        best = None
        for dx in steps:
            for dy in steps:
                c = all_c + (dx, dy)
                inside = np.all((c - all_h >= lo_lim) & (c + all_h <= hi_lim), axis=1)
                count = int(inside.sum())
                if not count:
                    continue
                lo = (c[inside] - all_h[inside]).min(axis=0)
                hi = (c[inside] + all_h[inside]).max(axis=0)
                off_centre = float(np.abs((lo + hi) / 2.0 - (width_in / 2.0, height_in / 2.0)).sum())
                key = (count, -off_centre)
                if best is None or key > best[0]:
                    best = (key, dx, dy, inside, lo, hi)
        if best is not None:
            _key, dx, dy, inside, lo, hi = best
            shift = np.array([width_in / 2.0, height_in / 2.0]) - (lo + hi) / 2.0
            verts_list, codes_list = [], []
            flat = [(verts, codes) for _o in offsets for verts, codes in groups]
            for index in all_i[inside]:
                verts, codes = flat[index]
                ox, oy = offsets[index // len(groups)]
                verts_list.append(verts + (ox + dx + shift[0], oy + dy + shift[1]))
                codes_list.append(codes)
            marks_path = Path(np.concatenate(verts_list), np.concatenate(codes_list))
    lines_path = None
    pieces = tiled(lines) if lines else []
    if pieces:
        lines_path = Path(
            np.concatenate([v for v, _c in pieces]), np.concatenate([c for _v, c in pieces])
        )
    return marks_path, lines_path


def isolated_mark_centres(path: Path | None) -> np.ndarray:
    """Centres (inches) of the isolated marks in a ``swatch_marks_path`` result."""
    if path is None:
        return np.empty((0, 2))
    groups, _lines = _mark_groups(path)
    centres = []
    for verts, codes in groups:
        drawn = verts[codes != Path.CLOSEPOLY]
        centres.append((drawn.min(axis=0) + drawn.max(axis=0)) / 2.0)
    return np.asarray(centres).reshape(-1, 2)
