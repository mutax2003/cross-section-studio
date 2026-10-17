"""Full lithology legend sheet: every code, readable, nothing printed over a code."""

from __future__ import annotations

import itertools

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from constants import USGS_LITHOLOGY_COLORS  # noqa: E402
from lithology_legend_sheet import (  # noqa: E402
    build_legend_sheet_figure,
    build_legend_sheet_png,
    pattern_name,
)


def test_sheet_lists_every_palette_code_once() -> None:
    fig = build_legend_sheet_figure()
    try:
        texts = [t.get_text() for t in fig.axes[0].texts]
        for code in USGS_LITHOLOGY_COLORS:
            assert texts.count(code.upper()) == 1, code
    finally:
        plt.close(fig)


def test_no_text_prints_over_another() -> None:
    """The palette image used to print descriptions over the codes."""
    fig = build_legend_sheet_figure()
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [t.get_window_extent(renderer) for t in fig.axes[0].texts if t.get_text()]
        overlaps = [(a, b) for a, b in itertools.combinations(boxes, 2) if a.overlaps(b)]
        assert not overlaps
        assert all(fig.bbox.contains(b.x0, b.y0) and fig.bbox.contains(b.x1, b.y1) for b in boxes)
    finally:
        plt.close(fig)


def test_pattern_names_are_plain_words() -> None:
    assert pattern_name("") == "plain"
    assert pattern_name(".") == "dots"
    assert pattern_name("+") == "plus marks"
    assert pattern_name("/.") == "45° dashes + dots"


def test_png_is_valid_and_can_be_limited_to_codes() -> None:
    png = build_legend_sheet_png(["Sand", "Clay Loam"])
    assert png.startswith(b"\x89PNG")
