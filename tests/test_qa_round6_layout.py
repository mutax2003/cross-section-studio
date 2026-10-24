"""QA round 6 — layout regressions in the renderer family.

1. Water-level numbers on ordinary fence densities (12 holes @ 12 m): a
   neighbouring column is a soft tier and the figure compacts its numbers
   (unit, decimals, size) before any is dropped; gradients take a line as
   the last resort.
2. ``_TrueMetreLocator`` caps its tick counts on an extreme elevation span.
3. Chart title and headers with 2-line hole IDs stay on fixed pages.
4. ``_legend_panel_layout`` on an empty entry list.
5. ``fit_label`` breaks at '-', '_', '/' or spaces before a hard break.
6. Threshold-mode legend samples use each parameter's own marker.
7. A legend taller than a short exact-VE plot box stays on the page.
8. The "values not printed" footnote stacks clear of the VE footer line.
"""

from __future__ import annotations

import math
import time
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.text import Text

from export_framing import ExportFramingConfig
from models import (
    Collar,
    ConsultingTitleBlock,
    EnvironmentalReading,
    Lithology,
    ScreenInterval,
    WaterLevel,
)
from pipeline import OverlayExclusions, compute_section_geometry, render_cross_section_from_geometry
from renderer_common import THRESHOLD_KEY_GID, ThresholdDotsKey, fit_label, legend_source_keys
from renderer_water import _figure_renderer, _text_box

CODES = ["CLAY", "SILT", "SAND", "GRAVEL", "TILL", "SHALE", "SANDSTONE", "PEAT", "FILL", "BEDROCK"]


def _site(n: int, spacing: float, *, chem: bool = True, water: bool = True, id_fmt: str = "BH{i:02d}",
          elevation=None, depth: float = 30.0):
    collars, liths, waters, chems, screens = [], [], [], [], []
    for i in range(n):
        hid = id_fmt.format(i=i + 1)
        elev = 740.0 + 2.0 * math.sin(i) if elevation is None else float(elevation(i))
        td = depth + (i % 3) * 2.0
        collars.append(Collar(hole_id=hid, easting=i * spacing, northing=0.0, elevation=elev, total_depth=td))
        top, k = 0.0, 0
        while top < td - 1e-9:
            bot = min(td, top + 2.0 + ((i + k) % 4))
            liths.append(Lithology(hole_id=hid, from_depth=top, to_depth=bot, lithology_code=CODES[(i + k) % len(CODES)]))
            top, k = bot, k + 1
        if water:
            depth_w = 3.0 + 0.5 * (i % 4)
            waters.append(WaterLevel(hole_id=hid, depth=depth_w, elevation_masl=elev - depth_w))
        if chem:
            for d in (2.0, 2.3, 5.0, 9.0, 14.0):
                if d < td:
                    chems.append(EnvironmentalReading(
                        hole_id=hid, parameter="Chloride", value=50.0 + 97.0 * ((i * 7 + int(d * 3)) % 9), depth=d, unit="mg/L"
                    ))
        screens.append(ScreenInterval(hole_id=hid, from_depth=min(td - 3, 8.0), to_depth=min(td - 0.5, 12.0)))
    return collars, liths, waters, chems, screens


def _render(site, *, layout: str, page: str = "letter_landscape", ve=None, title: str | None = None, **extra):
    collars, liths, waters, chems, screens = site
    points = [(c.easting, c.northing) for c in collars]
    geometry = compute_section_geometry(collars, liths, points)
    kwargs = dict(
        vertical_exaggeration=ve,
        water_levels=waters or None,
        environmental_readings=chems,
        environmental_parameters=tuple(sorted({r.parameter for r in chems})) if chems else None,
        show_parameter_labels=bool(chems),
        chemistry_color_mode="threshold" if chems else None,
        chemistry_threshold_green_max=120.0,
        chemistry_threshold_yellow_max=400.0,
        chemistry_label_style="strip",
        render_layout=layout,
        consulting_title_block=(
            ConsultingTitleBlock(section_label="A-A'", prepared_for="Client", prepared_by="Firm")
            if layout == "consulting_section" else None
        ),
        screen_intervals=screens,
        export_framing=ExportFramingConfig(page_preset=page, export_dpi=150),
        export_formats=frozenset({"png"}),
        close_figure=False,
    )
    if title is not None:
        kwargs["title"] = title
    kwargs.update(extra)
    result = render_cross_section_from_geometry(geometry, points, **kwargs)
    bundle = result.retained_export_bundle
    bundle["figure"].draw_without_rendering()
    return bundle["renderer"], bundle["figure"], result


def _visible(renderer, kind: str) -> list:
    return [a for k, a, _c in renderer._water_labels if k == kind and a.get_visible()]


# --- 1. water numbers on an ordinary fence density ------------------------------


def test_dense_section_sheet_prints_its_water_numbers() -> None:
    renderer, figure, result = _render(
        _site(12, 12.0), layout="section_sheet",
        show_water_elevation_labels=True, interpolate_water_table=True,
    )
    try:
        shown = _visible(renderer, "rl")
        assert len(shown) >= 10, [w for w in result.overlap_warnings if "omitted" in w]
        # One compaction level for the whole figure: every number prints at
        # the same precision and size.
        assert len({a.get_text().count(".") and len(a.get_text().split(".")[1]) for a in shown}) == 1
        assert len({round(float(a.get_fontsize()), 3) for a in shown}) == 1
        # Never on its own column's interior.
        mpl_renderer = _figure_renderer(figure)
        columns = renderer._column_obstacle_boxes(figure)
        for annotation in shown:
            box = _text_box(annotation, mpl_renderer)
            x, y = annotation.axes.transData.transform([annotation._water_base_xy])[0]
            own = [c for c in columns if c.x0 - 0.5 <= x <= c.x1 + 0.5 and c.y0 - 0.5 <= y <= c.y1 + 0.5]
            assert own, annotation.get_text()
            assert not box.padded(-1.0).overlaps(own[0]), annotation.get_text()
    finally:
        plt.close(figure)


def test_dense_consulting_sheet_prints_its_gradients() -> None:
    renderer, figure, result = _render(
        _site(14, 14.0), layout="consulting_section",
        show_water_elevation_labels=True, interpolate_water_table=True,
    )
    try:
        assert len(_visible(renderer, "gradient")) >= 10, [w for w in result.overlap_warnings if "omitted" in w]
        assert len(_visible(renderer, "rl")) >= 12
    finally:
        plt.close(figure)


# --- 2. tick cap -------------------------------------------------------------------


def test_true_metre_locator_caps_ticks_on_an_extreme_span() -> None:
    site = _site(4, 60.0, chem=False, water=False, elevation=lambda i: -100_000.0 if i % 2 else 100_000.0, depth=400.0)
    started = time.perf_counter()
    renderer, figure, _result = _render(site, layout="consulting_section", ve=25.0)
    elapsed = time.perf_counter() - started
    try:
        assert elapsed < 10.0, elapsed
        counted = 0
        for ax in figure.axes:
            lo, hi = sorted(ax.get_ylim())
            if hi - lo < 1e5:
                continue
            counted += 1
            assert len(ax.yaxis.get_majorticklocs()) <= 500
            assert len(ax.yaxis.get_minorticklocs()) <= 500
        assert counted >= 1
    finally:
        plt.close(figure)


# --- 3. chart title on fixed pages with 2-line IDs ----------------------------------


@pytest.mark.parametrize("page", ["title_block", "letter_landscape"])
def test_chart_title_and_headers_stay_inside_the_page(page: str) -> None:
    site = _site(24, 25.0, chem=False, water=False, id_fmt="MW-2017-{i:02d}-DEEP-NESTED-WELL-NORTH-BANK")
    renderer, figure, _result = _render(site, layout="chart", page=page, title="Borehole Cross-Section")
    try:
        mpl_renderer = figure.canvas.get_renderer()
        height_in = float(figure.get_size_inches()[1])
        ax = renderer._ve_main_ax
        title = ax.title.get_window_extent(mpl_renderer).transformed(figure.dpi_scale_trans.inverted())
        assert title.y1 <= height_in - 0.25 + 1e-3, (title.y0, title.y1, height_in)
        assert title.y0 >= 0.25
        headers = [t for t in renderer._header_labels if t.figure is figure and t.get_visible()]
        assert headers
        for text in headers:
            box = text.get_window_extent(mpl_renderer).transformed(figure.dpi_scale_trans.inverted())
            assert box.y1 <= height_in - 0.25 + 1e-3, (text.get_text(), box.y1)
        # The plot keeps a readable share of the page under them.
        assert ax.get_position().height >= 0.2
    finally:
        plt.close(figure)


# --- 4. empty legend panel --------------------------------------------------------


def test_legend_panel_layout_handles_no_entries() -> None:
    from render_profiles import CONSULTING_SECTION_PROFILE
    from renderer import CrossSectionRenderer

    renderer = CrossSectionRenderer(render_profile=CONSULTING_SECTION_PROFILE)
    figure = plt.figure(figsize=(4.0, 3.0), dpi=100)
    try:
        layout = renderer._legend_panel_layout(figure, [], 0, panel_w_in=2.0, panel_h_in=1.5)
        assert layout.rows == []
        assert layout.hidden_units == 0
        assert layout.font_pt > 0 and layout.ncols == 1
    finally:
        plt.close(figure)


# --- 5. fit_label token breaks -----------------------------------------------------


def test_fit_label_breaks_after_separators_before_a_hard_break() -> None:
    assert fit_label("MW-2017-01-DEEP-NESTED") == "MW-2017-01-DEEP-\nNESTED"
    assert fit_label("WELL_SITE_A_SHALLOW_NORTH") == "WELL_SITE_A_SHALLOW_\nNORTH"
    assert fit_label("2017-BH11/BH24-11/DEEP-NEST") == "2017-BH11/BH24-11/\nDEEP-NEST"
    assert fit_label("MW-01 deep nest north bank") == "MW-01 deep nest\nnorth bank"
    # A token longer than the width still breaks hard; three lines ellipsize.
    lines = fit_label("X" * 120).split("\n")
    assert len(lines) == 2 and lines[0] == "X" * 20 and lines[1].endswith("…")
    # Paired IDs within the limit stay whole.
    assert fit_label("2017-BH11 / BH24-11") == "2017-BH11 / BH24-11"


# --- 6. per-parameter legend markers --------------------------------------------


def _two_parameter_site():
    collars, liths, waters, _chems, screens = _site(4, 25.0, chem=False)
    chems = []
    for i, collar in enumerate(collars):
        for parameter, base in (("Chloride", 60.0), ("Benzene", 500.0)):
            chems.append(EnvironmentalReading(
                hole_id=collar.hole_id, parameter=parameter, value=base + 90.0 * i, depth=4.0 + 3.0 * (parameter == "Benzene"), unit="mg/L",
            ))
    return collars, liths, waters, chems, screens


def test_threshold_legend_samples_use_each_parameters_marker() -> None:
    renderer, figure, _result = _render(_two_parameter_site(), layout="section_sheet")
    try:
        entries = renderer.parameter_series_legend
        assert len(entries) == 2 and len({e.get("marker") for e in entries}) == 2
        legend = renderer._ve_main_ax.get_legend()
        keys = [handle for handle, _label in legend_source_keys(legend) if isinstance(handle, ThresholdDotsKey)]
        assert len(keys) == 2
        assert len({key.marker for key in keys}) == 2
    finally:
        plt.close(figure)

    renderer, figure, _result = _render(_two_parameter_site(), layout="consulting_section")
    try:
        dots = [a for ax in figure.axes for a in ax.collections if a.get_gid() == "parameter-legend-threshold-dots"]
        assert len(dots) == 2
        shapes = {tuple(np.round(c.get_paths()[0].vertices, 4).ravel()) for c in dots}
        assert len(shapes) == 2
    finally:
        plt.close(figure)


# --- 7 / 8. tall legend and footnote on a portrait section sheet -------------------


def _portrait_sheet(*, ve, span: float, depth: float, parameter: str):
    collars = [Collar(hole_id=f"BH-{i}", easting=i * span / 2, northing=0.0, elevation=10.0, total_depth=depth) for i in range(3)]
    lith = [
        Lithology(hole_id=c.hole_id, from_depth=a, to_depth=b, lithology_code=code)
        for c in collars
        for a, b, code in ((0.0, depth * 0.4, "Sand"), (depth * 0.4, depth, "Clay"))
    ]
    readings = [
        # Readings past the hole bottom (as the fuzz case had them): the
        # label pass cannot print them, so the footnote is drawn.
        EnvironmentalReading(hole_id="BH-0", parameter=parameter, value=1000.0, from_depth=59.0, to_depth=63.0, label_color="green"),
        EnvironmentalReading(hole_id="BH-1", parameter="PHC F2", value=608.6, depth=43.0, unit="µg/kg"),
    ]
    points = [(0.0, 0.0), (span, 0.0)]
    geometry = compute_section_geometry(collars, lith, points)
    # The pipeline now leaves readings this far below a 20 m hole off the
    # figure (find_implausible_overlays); keep them drawn here so the test
    # still exercises the footnote layout it was written for.
    with mock.patch("pipeline.find_implausible_overlays", return_value=OverlayExclusions()):
        result = _render_portrait(geometry, points, ve, readings, parameter)
    bundle = result.retained_export_bundle
    bundle["figure"].draw_without_rendering()
    return bundle["renderer"], bundle["figure"], result


def _render_portrait(geometry, points, ve, readings, parameter: str):
    return render_cross_section_from_geometry(
        geometry,
        points,
        render_layout="section_sheet",
        export_formats=frozenset({"png"}),
        vertical_exaggeration=ve,
        export_framing=ExportFramingConfig(
            page_preset="letter_portrait", export_dpi=100,
            margin_top_in=0.5, margin_bottom_in=0.5, margin_left_in=0.5, margin_right_in=0.5,
        ),
        title="Section",
        environmental_readings=readings,
        environmental_parameters=[parameter, "PHC F2"],
        chemistry_color_mode="red",
        chemistry_label_style="stroke",
        parameter_draw_markers=False,
        close_figure=False,
    )


def test_tall_legend_beside_a_short_exact_ve_plot_stays_on_the_page() -> None:
    renderer, figure, _result = _portrait_sheet(ve=1.0, span=7500.0, depth=2000.0, parameter="ü" * 50)
    try:
        mpl_renderer = figure.canvas.get_renderer()
        legend = renderer._ve_main_ax.get_legend()
        assert legend is not None
        box = legend.get_window_extent(mpl_renderer).transformed(figure.dpi_scale_trans.inverted())
        width_in, height_in = (float(v) for v in figure.get_size_inches())
        assert box.y0 >= 0.25 - 1e-3 and box.y1 <= height_in - 0.25 + 1e-3, (box.y0, box.y1)
        assert box.x0 >= 0.25 - 1e-3 and box.x1 <= width_in - 0.25 + 1e-3, (box.x0, box.x1)
    finally:
        plt.close(figure)


def test_omitted_values_footnote_stacks_clear_of_the_footer() -> None:
    renderer, figure, _result = _portrait_sheet(ve=None, span=235.0, depth=20.0, parameter="ü" * 50)
    try:
        mpl_renderer = figure.canvas.get_renderer()
        key = renderer._chemistry_threshold_key
        assert key is not None and key.figure is figure and key.get_gid() == THRESHOLD_KEY_GID
        texts = [t.get_text() for t in key.findobj(Text)]
        assert any("not printed" in text for text in texts), texts
        note = key.get_window_extent(mpl_renderer)
        footers = [
            t for t in figure.texts
            if t is not getattr(figure, "_supxlabel", None) and t.get_visible() and t.get_text().strip()
            and t.get_position()[1] < 0.2
        ]
        assert footers
        for text in footers:
            box = text.get_window_extent(mpl_renderer)
            assert not (box.x0 < note.x1 and note.x0 < box.x1 and box.y0 < note.y1 and note.y0 < box.y1), text.get_text()
        assert note.y0 >= 0.25 * figure.dpi - 1.0
    finally:
        plt.close(figure)
