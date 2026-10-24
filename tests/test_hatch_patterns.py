"""Hatch marks match the client CAD legend 261023 (DOTS / DASH / CROSS / GRAVEL)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.hatch as mhatch  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.path import Path  # noqa: E402

import hatch_patterns  # noqa: E402
import renderer_common  # noqa: E402,F401  (installs the template marks)


def _marks(hatch: str) -> list[np.ndarray]:
    """Split a hatch path into its separate marks (one MOVETO each)."""
    path = mhatch.get_path(hatch, 6)
    marks, current = [], []
    for vertex, code in zip(path.vertices, path.codes, strict=True):
        if code == Path.MOVETO and current:
            marks.append(np.array(current))
            current = []
        current.append(vertex)
    if current:
        marks.append(np.array(current))
    return marks


def test_template_marks_are_installed_once() -> None:
    assert hatch_patterns.is_installed()
    before = list(mhatch._hatch_types)
    hatch_patterns.install()
    assert list(mhatch._hatch_types) == before


def _centres(hatch: str) -> np.ndarray:
    """Mark centres in one 1 in cell, edge duplicates dropped."""
    groups, _lines = hatch_patterns._mark_groups(mhatch.get_path(hatch, 6))
    centres = np.array([(v.min(axis=0) + v.max(axis=0)) / 2.0 for v, _c in groups])
    return centres[np.all((centres >= 0.0) & (centres < 1.0 - 1e-9), axis=1)]


def _nearest_pt(centres: np.ndarray) -> float:
    d = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)) * 72)


def test_sandy_dots_are_a_fine_stipple() -> None:
    """CAD DOTS: 0.84 pt dots on a staggered 4.69 pt lattice (matplotlib's
    stock '.' drew ~2.4 pt dots 12 pt apart)."""
    path = mhatch.get_path(".", 6)
    extent = path.vertices.max(axis=0) - path.vertices.min(axis=0)
    assert extent.max() == pytest.approx(1.0, abs=0.05)  # one-inch cell
    circles = path.vertices.reshape(-1, len(Path.unit_circle().vertices), 2)
    diameters_pt = (circles.max(axis=1) - circles.min(axis=1)).max(axis=1) * 72
    # Filled and stroked with the 0.65 pt hatch line: printed diameter 0.84 pt.
    assert np.allclose(diameters_pt + 0.65, hatch_patterns.DOT_PRINTED_DIAMETER_PT, atol=0.05)
    centres = _centres(".")
    assert len(centres) == 16 * 16  # 16 staggered rows per inch
    assert _nearest_pt(centres) == pytest.approx(4.5, abs=0.05)  # CAD: 4.69 pt


def test_silty_pattern_is_short_dashes_on_a_square_lattice() -> None:
    """CAD DASH at 45°: 6.25 pt dashes, rows and columns aligned ~8.85 pt apart."""
    marks = _marks("/")
    lengths_pt = [np.hypot(*(m[-1] - m[0])) * 72 for m in marks]
    assert np.allclose(lengths_pt, 6.25, atol=0.05)
    assert all(m[-1][0] > m[0][0] and m[-1][1] > m[0][1] for m in marks)  # 45 degrees
    centres = _centres("/")
    assert len(centres) == 8 * 8
    assert len(np.unique(np.round(centres[:, 0], 6))) == 8  # columns line up (not staggered)
    assert _nearest_pt(centres) == pytest.approx(9.0, abs=0.05)


def test_loam_pattern_is_isolated_plus_marks_on_a_checkerboard() -> None:
    """CAD CROSS: 4.38 pt "+" marks, each column offset half a step (8.76 pt grid)."""
    marks = _marks("+")
    lengths_pt = {round(float(np.hypot(*(m[-1] - m[0])) * 72), 2) for m in marks}
    assert lengths_pt == {4.38}  # two 4.38 pt strokes per mark, no grid lines
    assert len(marks) % 2 == 0
    centres = _centres("+")
    assert len(centres) == 8 * 8 // 2
    assert _nearest_pt(centres) == pytest.approx(9.0 * np.sqrt(2.0), abs=0.05)
    # Checkerboard: neighbouring columns are offset by one grid step.
    col0 = centres[np.isclose(centres[:, 0], 0.0)][:, 1]
    col1 = centres[np.isclose(centres[:, 0], 1 / 8)][:, 1]
    assert np.isclose(min(col1) - min(col0), 1 / 8)


def test_template_marks_tile_seamlessly() -> None:
    """Repeating a character keeps whole-inch tiling (even row counts)."""
    for hatch in (".", "..", "+", "++", "+++"):
        centres = _centres(hatch)
        assert len(centres) > 0
        for column in np.unique(np.round(centres[:, 0], 6)):
            ys = np.sort(centres[np.isclose(centres[:, 0], column)][:, 1])
            steps = np.diff(np.concatenate([ys, ys[:1] + 1.0]))
            assert np.allclose(steps, steps[0]), (hatch, column)


def test_other_patterns_keep_stock_matplotlib_shapes() -> None:
    """'x' keeps continuous diagonals and '-'/'|' keep lines; only '.', '/'
    and '+' changed."""
    long_strokes = [m for m in _marks("x") if np.hypot(*(m[-1] - m[0])) > 0.5]
    assert long_strokes
    assert len(mhatch.get_path("-", 6).vertices) > 0
    assert len(mhatch.get_path("O", 6).vertices) > 0


def test_gravel_draws_template_cobble_outlines_not_circles() -> None:
    """CAD GRAVEL draws irregular stone outlines (15 pt tile on the 261023
    plot); matplotlib's "O" drew even circles."""
    from hatch_patterns import COBBLE_TILE_SEGMENTS

    path = mhatch.get_path("O", 6)
    codes = path.codes
    assert set(codes) <= {Path.MOVETO, Path.LINETO}  # straight outline segments, no curves
    # 8 tiles per inch (the legend size) plus one tile of overlap on each side.
    assert len(path.vertices) == len(COBBLE_TILE_SEGMENTS) * 2 * 10 * 10
    assert len(COBBLE_TILE_SEGMENTS) == 41


def test_gravel_outlines_are_thin_on_every_hatched_artist() -> None:
    """Small legend-size stones filled in at the shared 0.65 pt hatch line."""
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Rectangle

    from hatch_patterns import GRAVEL_HATCH_LINEWIDTH_PT

    patch = Rectangle((0, 0), 1, 1, hatch="O")
    collection = PolyCollection([[(0, 0), (1, 0), (1, 1)]], hatch="OO")
    assert patch.get_hatch_linewidth() == GRAVEL_HATCH_LINEWIDTH_PT
    assert collection.get_hatch_linewidth() == GRAVEL_HATCH_LINEWIDTH_PT
    assert Rectangle((0, 0), 1, 1, hatch="/").get_hatch_linewidth() != GRAVEL_HATCH_LINEWIDTH_PT


def test_hatch_change_away_from_gravel_restores_the_normal_line() -> None:
    import matplotlib
    from matplotlib.patches import Rectangle

    patch = Rectangle((0, 0), 1, 1, hatch="O")
    patch.set_hatch("/")
    assert patch.get_hatch_linewidth() == matplotlib.rcParams["hatch.linewidth"]
