"""Hatch marks match the client legend template (Cross_Section_Litho_Legend_261002)."""

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


def test_sandy_dots_are_a_fine_stipple() -> None:
    """Template: ~0.84 pt dots about 5 pt apart (matplotlib's stock '.' drew
    ~2.4 pt dots 12 pt apart)."""
    path = mhatch.get_path(".", 6)
    extent = path.vertices.max(axis=0) - path.vertices.min(axis=0)
    assert extent.max() == pytest.approx(1.0, abs=0.05)  # one-inch cell
    circles = path.vertices.reshape(-1, len(Path.unit_circle().vertices), 2)
    diameters_pt = (circles.max(axis=1) - circles.min(axis=1)).max(axis=1) * 72
    assert np.allclose(diameters_pt, 0.84, atol=0.05)
    # ~14 rows per inch, staggered: roughly 196 dots per square inch.
    assert 150 <= len(circles) <= 260


def test_silty_pattern_is_short_dashes_not_lines() -> None:
    marks = _marks("/")
    lengths_pt = [np.hypot(*(m[-1] - m[0])) * 72 for m in marks]
    assert np.allclose(lengths_pt, 7.13, atol=0.1)  # 5.04 pt box diagonal
    assert all(m[-1][0] > m[0][0] and m[-1][1] > m[0][1] for m in marks)  # 45 degrees


def test_clay_loam_pattern_is_isolated_plus_marks() -> None:
    marks = _marks("+")
    lengths_pt = {round(float(np.hypot(*(m[-1] - m[0])) * 72), 1) for m in marks}
    assert lengths_pt <= {4.3, 4.4}  # template: two 4.35 pt strokes per mark, no grid lines
    assert len(marks) % 2 == 0


def test_other_patterns_keep_stock_matplotlib_shapes() -> None:
    """'x' keeps continuous diagonals and '-'/'|' keep lines; only '.', '/'
    and '+' changed."""
    long_strokes = [m for m in _marks("x") if np.hypot(*(m[-1] - m[0])) > 0.5]
    assert long_strokes
    assert len(mhatch.get_path("-", 6).vertices) > 0
    assert len(mhatch.get_path("O", 6).vertices) > 0


def test_gravel_draws_template_cobble_outlines_not_circles() -> None:
    """Template 261002 draws gravel as irregular stone outlines on a ~17 pt
    tile; matplotlib's "O" drew even circles."""
    from hatch_patterns import COBBLE_TILE_SEGMENTS

    path = mhatch.get_path("O", 6)
    codes = path.codes
    assert set(codes) <= {Path.MOVETO, Path.LINETO}  # straight outline segments, no curves
    # 4 tiles per inch plus one tile of overlap on each side.
    assert len(path.vertices) == len(COBBLE_TILE_SEGMENTS) * 2 * 6 * 6
    assert len(COBBLE_TILE_SEGMENTS) == 41
