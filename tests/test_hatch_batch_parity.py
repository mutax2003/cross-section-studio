"""HatchBatchedPolyCollection must draw exactly like a stock PolyCollection."""

from __future__ import annotations

import io
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from PIL import Image  # noqa: E402

import renderer_common as RC  # noqa: E402

_SHAPES = [
    [(10, 10), (30, 10), (30, 30), (10, 30)],
    [(50, 50), (70, 52), (60, 70)],
    [(30, 10), (50, 10), (50, 30), (30, 30)],  # shares an edge with the first
]


def _png(cls, *, linewidth: float, edgecolor) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.add_collection(
        cls(_SHAPES, facecolors=[(0.8, 0.6, 0.2, 1)], edgecolors=edgecolor, linewidths=linewidth, hatch="..")
    )
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100)
    plt.close(fig)
    return np.asarray(Image.open(buffer)).astype(int)


@pytest.mark.parametrize(("linewidth", "edgecolor"), [(0.4, "k"), (0.0, "k"), (0.4, "none")])
def test_batched_hatch_draw_matches_stock_pixels(linewidth, edgecolor) -> None:
    batched = _png(RC.HatchBatchedPolyCollection, linewidth=linewidth, edgecolor=edgecolor)
    stock = _png(PolyCollection, linewidth=linewidth, edgecolor=edgecolor)
    assert np.array_equal(batched, stock)


@pytest.mark.parametrize("first", ["renderer_water", "renderer_chemistry", "renderer_common"])
def test_renderer_modules_import_in_any_order(first: str) -> None:
    # A renamed collection subclass once broke matplotlib's docstring
    # registry ("KeyError: 'PolyCollection:kwdoc'") when imported before pyplot.
    code = f"import {first}; import renderer; import matplotlib.pyplot"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr[-2000:]
