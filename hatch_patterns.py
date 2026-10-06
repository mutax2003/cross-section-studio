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
TEMPLATE_ROWS_PER_INCH: dict[str, int] = {".": 14, "/": 8, "+": 6}
STOCK_ROWS_PER_INCH = 6

# Mark sizes, in inches (template measurements at 1:1).
_DOT_RADIUS_IN = 0.42 / 72.0
_DASH_HALF_SPAN_IN = 2.52 / 72.0  # each axis: 5.04 pt box -> ~7.1 pt dash
_PLUS_ARM_IN = 4.35 / 2 / 72.0  # template "+" strokes are 4.35 pt long

_DEFAULT_DENSITY = 6


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
}
_TEMPLATE_TYPES = (_HorizontalOnly, _VerticalOnly, _NorthEastLinesOnly, TemplateDashes, TemplateDots, TemplatePlus)


def is_installed() -> bool:
    return all(cls in mhatch._hatch_types for cls in _TEMPLATE_TYPES)


def install() -> None:
    """Use the template marks for ``.``, ``/`` and ``+`` (idempotent)."""
    if is_installed():
        return
    types: list[type] = []
    for cls in mhatch._hatch_types:
        types.extend(_REPLACEMENTS.get(cls, (cls,)))
    types.append(TemplatePlus)
    mhatch._hatch_types[:] = types
