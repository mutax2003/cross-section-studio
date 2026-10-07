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

import matplotlib.hatch as mhatch
import numpy as np
from matplotlib.path import Path

# Marks per inch along each axis for ONE pattern character at matplotlib's
# default hatch density (6). Matplotlib's hatch unit cell is one inch.
TEMPLATE_ROWS_PER_INCH: dict[str, int] = {".": 14, "/": 8, "+": 6, "O": 4}
STOCK_ROWS_PER_INCH = 6

# Mark sizes, in inches (template measurements at 1:1).
_DOT_RADIUS_IN = 0.42 / 72.0
_DASH_HALF_SPAN_IN = 2.52 / 72.0  # each axis: 5.04 pt box -> ~7.1 pt dash
_PLUS_ARM_IN = 4.35 / 2 / 72.0  # template "+" strokes are 4.35 pt long

_DEFAULT_DENSITY = 6

# Gravel cobbles: one repeat tile of the template's stone outlines, traced
# from Cross_Section_Litho_Legend_261002 (17.1 pt square tile; drawn here at
# 4 tiles per inch = 18 pt so the 1 in hatch cell holds whole tiles). Each
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


def install() -> None:
    """Use the template marks for ``.``, ``/``, ``+`` and ``O`` (idempotent)."""
    if is_installed():
        return
    types: list[type] = []
    for cls in mhatch._hatch_types:
        types.extend(_REPLACEMENTS.get(cls, (cls,)))
    types.append(TemplatePlus)
    mhatch._hatch_types[:] = types
