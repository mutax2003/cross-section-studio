"""Lithology legend sheet: every code, readable, nothing printed over a code.

Default layout is the client legend 261023 look (tall swatches, name above);
the row layout (swatch, name, colour, pattern) stays available.
"""

from __future__ import annotations

import itertools

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from constants import LITHOLOGY_LEGEND_ORDER, USGS_LITHOLOGY_COLORS  # noqa: E402
from lithology_legend_sheet import (  # noqa: E402
    build_legend_sheet_figure,
    build_legend_sheet_png,
    pattern_name,
    tall_label,
)


def _texts(fig) -> list[str]:
    return [t.get_text().replace("\n", " ") for t in fig.axes[0].texts]


def test_default_tall_sheet_lists_the_client_legend_in_order() -> None:
    fig = build_legend_sheet_figure()
    try:
        texts = _texts(fig)
        for code in LITHOLOGY_LEGEND_ORDER:
            assert texts.count(code.upper()) == 1, code
        labels = [t for t in fig.axes[0].texts if (t.get_gid() or "").startswith("legend-label:")]
        assert [t.get_gid().split(":", 1)[1] for t in labels] == list(LITHOLOGY_LEGEND_ORDER)
        swatches = fig.axes[0].patches
        assert len(swatches) == len(LITHOLOGY_LEGEND_ORDER)
        assert all(p.get_height() > 5 * p.get_width() for p in swatches)  # tall, like 261023
    finally:
        plt.close(fig)


def test_row_sheet_lists_every_palette_code_once() -> None:
    fig = build_legend_sheet_figure(layout="rows")
    try:
        texts = _texts(fig)
        for code in USGS_LITHOLOGY_COLORS:
            assert texts.count(code.upper()) == 1, code
    finally:
        plt.close(fig)


def test_long_names_wrap_on_words() -> None:
    assert tall_label("Sandy Clay Loam") == "SANDY CLAY\nLOAM"
    assert tall_label("Clay") == "CLAY"
    with pytest.raises(ValueError, match="layout"):
        build_legend_sheet_figure(layout="bogus")


@pytest.mark.parametrize("layout", ["tall", "rows"])
def test_no_text_prints_over_another(layout: str) -> None:
    """The palette image used to print descriptions over the codes."""
    fig = build_legend_sheet_figure(layout=layout)
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
    assert build_legend_sheet_png(["Sand"], layout="rows").startswith(b"\x89PNG")
