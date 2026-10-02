"""Consulting-sheet typography and title-block fixes.

Covers three defects on the consulting report layout:

1. ``export_font_size`` must scale the explicit consulting point sizes
   (headers, legend, axis labels, ticks) proportionally above the 9 pt base,
   and sizes at or below the base must render identically.
2. A long section label must stay inside its TITLE cell in the title block
   (no text crossing a row rule or the panel edge).
3. The app default label "Borehole Cross-Section" must not print as
   "CROSS SECTION BOREHOLE CROSS-SECTION".
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest
from matplotlib.patches import Rectangle

from models import Collar, ConsultingTitleBlock, Lithology
from render_profiles import CONSULTING_SECTION_PROFILE
from renderer import CrossSectionRenderer
from tests.conftest import run_pipeline

LONG_LABEL = "A-A' ALONG THE NORTHERN PROPERTY BOUNDARY WITH GROUNDWATER LEVELS 2025"
assert len(LONG_LABEL) == 70


@pytest.fixture(scope="module")
def two_hole_section():
    collars = [
        Collar(hole_id="MW-01", easting=0.0, northing=0.0, elevation=665.0, total_depth=20.0),
        Collar(hole_id="MW-02", easting=50.0, northing=0.0, elevation=664.0, total_depth=18.0),
    ]
    lithologies = [
        Lithology(hole_id="MW-01", from_depth=0.0, to_depth=8.0, lithology_code="Sand"),
        Lithology(hole_id="MW-01", from_depth=8.0, to_depth=20.0, lithology_code="Clay"),
        Lithology(hole_id="MW-02", from_depth=0.0, to_depth=10.0, lithology_code="Sand"),
        Lithology(hole_id="MW-02", from_depth=10.0, to_depth=18.0, lithology_code="Clay"),
    ]
    projected, polygons, _ = run_pipeline(collars, lithologies, [(0.0, 0.0), (50.0, 0.0)])
    return projected, polygons, {"MW-01": 20.0, "MW-02": 18.0}


def _render(two_hole_section, title_block: ConsultingTitleBlock, *, export_font_size: float | None = None):
    projected, polygons, collar_depths = two_hole_section
    profile = CONSULTING_SECTION_PROFILE
    if export_font_size is not None:
        profile = profile.model_copy(update={"export_font_size": export_font_size})
    renderer = CrossSectionRenderer(
        show_legend=False,
        render_profile=profile,
        consulting_title_block=title_block,
    )
    figure = renderer.render(polygons, projected, collar_depths=collar_depths)
    return renderer, figure


def _title_block_axes(figure):
    """The CAD title-block axes: the one holding the TITLE row label."""
    return next(ax for ax in figure.axes if any(t.get_text() == "TITLE" for t in ax.texts))


def _font_sizes(figure) -> dict[str, float]:
    ax = figure.axes[0]
    block = _title_block_axes(figure)
    return {
        "hole_header": next(t.get_fontsize() for t in ax.texts if t.get_text() == "MW-01"),
        "legend_header": next(t.get_fontsize() for t in block.texts if t.get_text() == "LEGEND"),
        "x_label": ax.xaxis.label.get_fontsize(),
        "tick": ax.xaxis.get_ticklabels()[0].get_fontsize(),
        "title_row": next(t.get_fontsize() for t in block.texts if t.get_text() == "TITLE"),
    }


# --- (1) export_font_size scales the sheet ---------------------------------


def test_export_font_size_scales_consulting_text_proportionally(two_hole_section) -> None:
    block = ConsultingTitleBlock(section_label="A-A'", map_scale="1:1000")
    _, fig_9 = _render(two_hole_section, block, export_font_size=9.0)
    _, fig_13 = _render(two_hole_section, block, export_font_size=13.0)
    sizes_9 = _font_sizes(fig_9)
    sizes_13 = _font_sizes(fig_13)
    expected_scale = 13.0 / 9.0
    for key, base in sizes_9.items():
        assert sizes_13[key] == pytest.approx(base * expected_scale), key
    # Design sizes at the base: header 8, legend 8.5, axis label 10, ticks 8.
    assert sizes_9["hole_header"] == pytest.approx(8.0)
    assert sizes_9["legend_header"] == pytest.approx(8.5)
    assert sizes_9["x_label"] == pytest.approx(10.0)
    assert sizes_9["tick"] == pytest.approx(8.0)


def test_export_font_size_below_base_renders_identically(two_hole_section) -> None:
    block = ConsultingTitleBlock(section_label="A-A'", map_scale="1:1000")
    renderer_8, fig_8 = _render(two_hole_section, block, export_font_size=8.0)
    renderer_9, fig_9 = _render(two_hole_section, block, export_font_size=9.0)
    assert _font_sizes(fig_8) == _font_sizes(fig_9)
    assert renderer_8.to_png_bytes(fig_8) == renderer_9.to_png_bytes(fig_9)


# --- (2) long TITLE stays inside its cell ------------------------------------


def _title_cell_check(figure, label: str) -> None:
    figure.draw_without_rendering()
    block = _title_block_axes(figure)
    # The title panel is the boxed Rectangle starting at x=0.34 (axes fraction).
    panel = next(
        p
        for p in block.patches
        if isinstance(p, Rectangle) and p.get_visible() and abs(p.get_x() - 0.34) < 1e-6
    )
    panel_bbox = panel.get_window_extent()
    # Horizontal cell rules drawn by the metadata table (constant-y, two points).
    rule_ys = sorted(
        {
            float(block.transAxes.transform((0.0, line.get_ydata()[0]))[1])
            for line in block.lines
            if len(line.get_ydata()) == 2 and line.get_ydata()[0] == line.get_ydata()[1]
        }
    )
    words = [w for w in label.split() if len(w) > 3]
    title_texts = [t for t in block.texts if any(w in t.get_text() for w in words)]
    assert title_texts, "title text not found in title block"
    for text in title_texts:
        extent = text.get_window_extent()
        assert extent.x0 >= panel_bbox.x0 - 0.5
        assert extent.x1 <= panel_bbox.x1 + 0.5, (text.get_text(), extent.x1, panel_bbox.x1)
        # No row rule may pass through the text box.
        crossing = [y for y in rule_ys if extent.y0 < y < extent.y1]
        assert not crossing, (text.get_text(), extent.extents, crossing)
    # All fragments share a size no smaller than the hard minimum.
    assert min(t.get_fontsize() for t in title_texts) >= 5.0


def test_long_section_label_stays_inside_title_cell(two_hole_section) -> None:
    block = ConsultingTitleBlock(
        section_label=LONG_LABEL,
        map_scale="1:1000",
        project_number="100/09-29 & 100/16-29-057-19 W4M",
        source="ECOVENTURE 2025",
        date="05/11/26",
        drawn_by="SL/JG",
        revised="EC 04/12/22-xsec",
        figure_number="3",
        prepared_for="C-GROUP ENERGY INC.",
        prepared_by="ECOVENTURE",
    )
    _, figure = _render(two_hole_section, block)
    _title_cell_check(figure, LONG_LABEL)


def test_long_section_label_fits_wide_title_cell_on_one_line(two_hole_section) -> None:
    # Without prepared-for/by the title panel takes the remaining width and
    # the 70-character label shrinks instead of wrapping.
    block = ConsultingTitleBlock(section_label=LONG_LABEL, map_scale="1:1000", figure_number="3")
    _, figure = _render(two_hole_section, block)
    _title_cell_check(figure, LONG_LABEL)
    block_ax = _title_block_axes(figure)
    fragments = [t for t in block_ax.texts if "PROPERTY BOUNDARY" in t.get_text()]
    assert len(fragments) == 1
    assert fragments[0].get_fontsize() < 7.0


def test_short_section_label_keeps_base_title_size(two_hole_section) -> None:
    block = ConsultingTitleBlock(section_label="A-A'", map_scale="1:1000")
    _, figure = _render(two_hole_section, block)
    block_ax = _title_block_axes(figure)
    title = next(t for t in block_ax.texts if t.get_text() == "CROSS SECTION A-A'")
    assert title.get_fontsize() == pytest.approx(7.0)


# --- (3) default label is not doubled ----------------------------------------


def test_default_label_does_not_double_cross_section(two_hole_section) -> None:
    renderer, figure = _render(
        two_hole_section, ConsultingTitleBlock(section_label="Borehole Cross-Section")
    )
    svg_text = renderer.to_svg_bytes(figure).decode("utf-8", errors="ignore")
    assert "CROSS SECTION BOREHOLE" not in svg_text.upper()
    assert "Borehole Cross-Section" in svg_text
    block_ax = _title_block_axes(figure)
    assert any(t.get_text() == "Borehole Cross-Section" for t in block_ax.texts)


def test_plain_label_still_gets_cross_section_prefix(two_hole_section) -> None:
    _, figure = _render(two_hole_section, ConsultingTitleBlock(section_label="B-B'"))
    block_ax = _title_block_axes(figure)
    assert any(t.get_text() == "CROSS SECTION B-B'" for t in block_ax.texts)
