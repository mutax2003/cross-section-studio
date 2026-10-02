"""Consulting-sheet typography and title-block fixes.

Covers three defects on the consulting report layout:

1. ``export_font_size`` must scale the explicit consulting point sizes
   (headers, legend, axis labels, ticks) proportionally above the 9 pt base,
   and sizes at or below the base must render identically.
2. A long section label must stay inside its TITLE cell in the title block
   (no text crossing a row rule or the panel edge).
3. The app default label "Borehole Cross-Section" must not print as
   "CROSS SECTION BOREHOLE CROSS-SECTION".
4. At a large ``export_font_size`` the sheet geometry (legend pitch, notes
   wrap, title-block columns, header reserve) follows the font scale, so text
   does not overprint or leave the page.
5. Fitted TITLE text is re-fitted when export resizes the page
   (letter portrait), and the subtitle band title is fitted to its panel.
"""

from __future__ import annotations

import itertools

import matplotlib

matplotlib.use("Agg")

import pytest
from matplotlib.patches import Rectangle

from export_framing import ExportFramingConfig
from models import Collar, ConsultingTitleBlock, EnvironmentalReading, Lithology, WaterLevel
from render_profiles import CONSULTING_SECTION_PROFILE
from renderer import CrossSectionRenderer
from renderer_consulting import ConsultingLayoutMixin
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


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Intersection of Main St", "CROSS SECTION Intersection of Main St"),
        ("Section line 2", "Section line 2"),
        ("SECTION B-B'", "SECTION B-B'"),
        ("Cross-Section", "Cross-Section"),
        ("Borehole Cross-Section", "Borehole Cross-Section"),
        ("A-A' (section)", "A-A' (section)"),
        ("Bisection A", "CROSS SECTION Bisection A"),
        ("Sections 1-3", "CROSS SECTION Sections 1-3"),
        ("B-B'", "CROSS SECTION B-B'"),
        ("", "CROSS SECTION"),
    ],
)
def test_display_title_section_word_boundary(label: str, expected: str) -> None:
    assert ConsultingLayoutMixin._consulting_display_title(label) == expected


# --- (4) geometry follows export_font_size -----------------------------------


@pytest.fixture(scope="module")
def dense_sheet():
    """12 holes, 31 lithology codes, water and chloride: the crowded case."""
    codes = [f"Code{i:02d} Silty Sandy Gravel" for i in range(31)]
    ids = [f"MW-2025-{i:02d}" for i in range(12)]
    collars = [
        Collar(hole_id=h, easting=i * 25.0, northing=0.0, elevation=100.0 + i * 0.3, total_depth=20.0)
        for i, h in enumerate(ids)
    ]
    lith = []
    for i, h in enumerate(ids):
        for k in range(5):
            lith.append(
                Lithology(
                    hole_id=h,
                    from_depth=4.0 * k,
                    to_depth=4.0 * (k + 1),
                    lithology_code=codes[(i * 5 + k) % 31],
                )
            )
    water = [WaterLevel(hole_id=h, depth=3.0 + (i % 4)) for i, h in enumerate(ids)]
    readings = [
        EnvironmentalReading(
            hole_id=h, parameter="Chloride", value=float(100 + k * 37), depth=1.0 + k * 4.0, unit="mg/L"
        )
        for h in ids
        for k in range(4)
    ]
    projected, polygons, _ = run_pipeline(collars, lith, [(0.0, 0.0), (275.0, 0.0)])
    block = ConsultingTitleBlock(
        section_label="A-A'",
        figure_number="3",
        project_number="P-1",
        source="Field logs",
        date="2026-10",
        notes=("Note one about the sheet " * 3, "Note two"),
        drawn_by="AL",
        prepared_for="Client Co",
        prepared_by="Consult Inc",
        transect_start_label="A",
        transect_end_label="A'",
    )
    return codes, ids, water, readings, projected, polygons, block


def _render_dense(dense_sheet, export_font_size: float):
    codes, ids, water, readings, projected, polygons, block = dense_sheet
    profile = CONSULTING_SECTION_PROFILE.model_copy(
        update={
            "export_font_size": export_font_size,
            "show_parameter_markers": True,
            "show_parameter_labels": True,
            "show_water_elevation_labels": True,
        }
    )
    renderer = CrossSectionRenderer(
        show_legend=True,
        render_profile=profile,
        consulting_title_block=block,
        environmental_readings=readings,
        environmental_parameters=("Chloride",),
    )
    return renderer.render(
        polygons, projected, collar_depths={h: 20.0 for h in ids}, water_levels=water, lithology_codes=codes
    )


def _text_layout_faults(figure) -> tuple[list[str], int]:
    figure.draw_without_rendering()
    renderer = figure.canvas.get_renderer()
    texts = [t for ax in figure.axes for t in ax.texts if t.get_visible() and t.get_text().strip()]
    texts += [t for t in figure.texts if t.get_text().strip()]
    boxes = [(t, t.get_window_extent(renderer)) for t in texts]
    page = figure.bbox
    off_page = [
        t.get_text()
        for t, b in boxes
        if b.x0 < -1 or b.y0 < -1 or b.x1 > page.width + 1 or b.y1 > page.height + 1
    ]
    overlaps = sum(
        1
        for (_t1, b1), (_t2, b2) in itertools.combinations(boxes, 2)
        if b1.width > 0 and b2.width > 0 and b1.overlaps(b2)
    )
    return off_page, overlaps


def test_large_export_font_keeps_text_on_page_and_apart(dense_sheet) -> None:
    off_8, overlaps_8 = _text_layout_faults(_render_dense(dense_sheet, 8.0))
    assert off_8 == []
    for size in (11.0, 14.0):
        off, overlaps = _text_layout_faults(_render_dense(dense_sheet, size))
        assert off == [], (size, off)
        # Before the geometry followed the scale, 14 pt gave 40 overlaps
        # against a 16 baseline (water/chemistry label stacks).
        assert overlaps <= overlaps_8 + 3, (size, overlaps, overlaps_8)


# --- (5) re-fit on export resize; band title fitted -------------------------


@pytest.mark.parametrize("page_preset", ["letter_portrait", "letter_landscape"])
def test_title_cell_refits_after_export_page_resize(two_hole_section, page_preset: str) -> None:
    projected, polygons, collar_depths = two_hole_section
    block = ConsultingTitleBlock(
        section_label=LONG_LABEL,
        map_scale="1:1000",
        figure_number="3",
        prepared_for="C-GROUP ENERGY INC.",
        prepared_by="ECOVENTURE",
    )
    renderer = CrossSectionRenderer(
        show_legend=False,
        render_profile=CONSULTING_SECTION_PROFILE,
        consulting_title_block=block,
        export_framing=ExportFramingConfig(page_preset=page_preset),
    )
    figure = renderer.render(polygons, projected, collar_depths=collar_depths)
    renderer.to_png_bytes(figure, dpi=72)
    expected = (8.5, 11.0) if page_preset == "letter_portrait" else (11.0, 8.5)
    assert tuple(figure.get_size_inches()) == pytest.approx(expected)
    _title_cell_check(figure, LONG_LABEL)


def test_long_label_band_title_fits_centre_panel(two_hole_section) -> None:
    block = ConsultingTitleBlock(section_label=LONG_LABEL, map_scale="1:1000")
    _, figure = _render(two_hole_section, block)
    figure.draw_without_rendering()
    renderer = figure.canvas.get_renderer()
    centre = next(
        ax for ax in figure.axes if any("VERTICAL EXAGGERATION" in t.get_text() for t in ax.texts)
    )
    band_title = [t for t in centre.texts if "PROPERTY" in t.get_text() or "GROUNDWATER" in t.get_text()]
    assert band_title
    neighbours = [
        t
        for ax in figure.axes
        for t in ax.texts
        if t.get_text() == "Metres" or t.get_text().startswith(("NOTES", "1. "))
    ]
    assert len(neighbours) >= 3
    panel = centre.get_window_extent(renderer)
    gutter = 0.13 * panel.width
    for text in band_title:
        box = text.get_window_extent(renderer)
        assert box.x0 >= panel.x0 - gutter and box.x1 <= panel.x1 + gutter, text.get_text()
        for other in neighbours:
            assert not box.overlaps(other.get_window_extent(renderer)), (text.get_text(), other.get_text())


# --- (6) long notes / prepared-for / portrait at every export size -----------

E2E_NOTES = (
    "Stratigraphy interpolated between boreholes; actual conditions may vary between locations.",
    "Groundwater levels measured June 2025 and may fluctuate seasonally.",
    "Chloride results in mg/L; see Table 2 for laboratory analytical data.",
    "Interpreted fence diagram — contacts are linear between adjacent boreholes. Groundwater "
    "markers/lines are schematic linear connectors between measured levels — not a "
    "potentiometric surface.",
)


@pytest.fixture(scope="module")
def ten_hole_sheet():
    collars, lith, water, readings = [], [], [], []
    sequence = [
        (0.0, 0.5, "Topsoil"),
        (0.5, 3.0, "Sandy Clay"),
        (3.0, 7.0, "Sand"),
        (7.0, 9.0, "Clay"),
        (9.0, 12.0, "Bedrock"),
    ]
    for i in range(10):
        hole = f"BH-{i + 1:02d}"
        collars.append(
            Collar(hole_id=hole, easting=i * 25.0, northing=0.0, elevation=100.0 - i * 0.3, total_depth=12.0)
        )
        lith += [
            Lithology(hole_id=hole, from_depth=a, to_depth=b, lithology_code=code) for a, b, code in sequence
        ]
        water.append(WaterLevel(hole_id=hole, depth=2.5 + 0.1 * i))
        readings.append(
            EnvironmentalReading(hole_id=hole, parameter="Chloride", value=5.0 + i * 3, depth=4.0, unit="mg/L")
        )
    projected, polygons, _ = run_pipeline(collars, lith, [(0.0, 0.0), (225.0, 0.0)])
    block = ConsultingTitleBlock(
        section_label="A-A'",
        transect_start_label="A",
        transect_end_label="A'",
        figure_number="3",
        project_number="PRJ-2026-0142",
        date="2026-10-02",
        drawn_by="AL",
        prepared_for="Northern Prairie Midstream Holdings Ltd.",
        prepared_by="Ecoventure Environmental Consulting Inc.",
        notes=E2E_NOTES,
    )
    return projected, polygons, water, readings, block


def _drawn_texts(figure):
    """Texts that are actually drawn (no tick labels on axis-off panels)."""
    texts = []
    for ax in figure.axes:
        texts += [t for t in ax.texts if t.get_visible() and t.get_text().strip()]
        if not ax.axison:
            continue
        for axis in (ax.xaxis, ax.yaxis):
            if axis.label.get_visible() and axis.label.get_text().strip():
                texts.append(axis.label)
            for tick in axis._update_ticks():
                texts += [lab for lab in (tick.label1, tick.label2) if lab.get_visible() and lab.get_text().strip()]
    return texts


@pytest.mark.parametrize("page_preset", ["letter_landscape", "letter_portrait"])
@pytest.mark.parametrize("export_font_size", [8.0, 11.0, 14.0])
def test_long_notes_and_prepared_values_stay_on_page_and_apart(
    ten_hole_sheet, page_preset: str, export_font_size: float
) -> None:
    projected, polygons, water, readings, block = ten_hole_sheet
    profile = CONSULTING_SECTION_PROFILE.model_copy(update={"export_font_size": export_font_size})
    renderer = CrossSectionRenderer(
        show_legend=True,
        render_profile=profile,
        consulting_title_block=block,
        environmental_readings=readings,
        environmental_parameters=("Chloride",),
        export_framing=ExportFramingConfig(page_preset=page_preset),
    )
    figure = renderer.render(
        polygons, projected, collar_depths={c: 12.0 for c in projected["hole_id"].unique()}, water_levels=water
    )
    renderer.to_png_bytes(figure, dpi=72)
    figure.draw_without_rendering()
    page = figure.bbox
    main_axes = {id(figure.axes[0])} | {
        id(ax) for ax in figure.axes if ax.bbox.bounds == figure.axes[0].bbox.bounds
    }
    boxes = [(t, t.get_window_extent()) for t in _drawn_texts(figure)]
    off_page = [
        t.get_text()[:40]
        for t, b in boxes
        if b.x0 < -1 or b.y0 < -1 or b.x1 > page.width + 1 or b.y1 > page.height + 1
    ]
    assert off_page == [], off_page
    # Notes, title-block and prepared-for/by text (everything below the plot).
    sheet = [(t, b) for t, b in boxes if t.axes is not None and id(t.axes) not in main_axes and b.width > 0]
    assert any(t.get_text().startswith("1. ") for t, _b in sheet)
    clashes = [
        (t1.get_text()[:30], t2.get_text()[:30])
        for (t1, b1), (t2, b2) in itertools.combinations(sheet, 2)
        if min(b1.x1, b2.x1) - max(b1.x0, b2.x0) > 1.0 and min(b1.y1, b2.y1) - max(b1.y0, b2.y0) > 1.0
    ]
    assert clashes == [], clashes
