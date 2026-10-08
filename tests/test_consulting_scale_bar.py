"""Consulting-sheet scale bar is drawn to the section's true horizontal scale.

The bar used to sit at a fixed fraction of its panel whatever the section
length, with 10 m ticks and a nominal "SCALE 1:1000"; on a 480 m section a
"30 m" bar measured ~100 m. These tests measure the drawn bar through the main
axes' data->display transform after the export page geometry is applied.
"""

from __future__ import annotations

import logging
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from export_framing import ExportFramingConfig
from models import Collar, ConsultingTitleBlock, Lithology
from render_profiles import CONSULTING_SECTION_PROFILE
from renderer import CrossSectionRenderer
from renderer_consulting import (
    nearest_standard_scale,
    nice_scale_bar_length,
    parse_map_scale,
    scale_bar_tick_step,
    scale_ratio_text,
)
from tests.conftest import run_pipeline

_SECTIONS = {"p2_short": (3, 6.0), "gwm_160m": (5, 40.0), "synth_480m": (17, 30.0)}


def _section(n_holes: int, spacing_m: float):
    collars, lithologies = [], []
    for i in range(n_holes):
        hole = f"BH{i:02d}"
        depth = 12.0 + (i % 3)
        collars.append(
            Collar(hole_id=hole, easting=i * spacing_m, northing=0.0, elevation=630.0 + 0.3 * i, total_depth=depth)
        )
        lithologies += [
            Lithology(hole_id=hole, from_depth=0.0, to_depth=depth / 2, lithology_code="Sand"),
            Lithology(hole_id=hole, from_depth=depth / 2, to_depth=depth, lithology_code="Clay"),
        ]
    end = (n_holes - 1) * spacing_m
    projected, polygons, _ = run_pipeline(collars, lithologies, [(0.0, 0.0), (end, 0.0)])
    return projected, polygons, {c.hole_id: c.total_depth for c in collars}


@pytest.fixture(scope="module", params=sorted(_SECTIONS))
def section(request):
    return request.param, _section(*_SECTIONS[request.param])


def _render(section_data, page_preset: str, title_block: ConsultingTitleBlock | None = None):
    projected, polygons, depths = section_data
    renderer = CrossSectionRenderer(
        show_legend=False,
        render_profile=CONSULTING_SECTION_PROFILE,
        consulting_title_block=title_block or ConsultingTitleBlock(section_label="SCALE TEST"),
        export_framing=ExportFramingConfig(page_preset=page_preset),
    )
    figure = renderer.render(polygons, projected, collar_depths=depths)
    renderer.to_png_bytes(figure, dpi=72)  # applies the export page + re-fit
    return figure


def _scale_panel(figure):
    return next(ax for ax in figure.axes if any(t.get_text() == "Metres" for t in ax.texts))


def _measure(figure):
    main = figure._css_main_axes[0]
    panel = _scale_panel(figure)
    bar = max(
        (line for line in panel.lines if line.get_linewidth() >= 2.0),
        key=lambda line: abs(line.get_xdata()[1] - line.get_xdata()[0]),
    )
    x0, x1 = bar.get_xdata()
    px = abs(panel.transAxes.transform((x1, 0))[0] - panel.transAxes.transform((x0, 0))[0])
    lo, hi = main.get_xlim()
    metres_per_px = abs(hi - lo) / main.bbox.width
    labels = [t for t in panel.texts if re.fullmatch(r"\d+(\.\d+)?", t.get_text())]
    scale_text = next(t.get_text() for t in panel.texts if "SCALE" in t.get_text())
    return px * metres_per_px, labels, scale_text, metres_per_px, panel, px


@pytest.mark.parametrize("page_preset", ["letter_landscape", "letter_portrait"])
def test_scale_bar_length_matches_label(section, page_preset: str) -> None:
    name, data = section
    figure = _render(data, page_preset)
    try:
        drawn_m, labels, scale_text, metres_per_px, panel, bar_px = _measure(figure)
        labelled = max(float(t.get_text()) for t in labels)
        assert drawn_m == pytest.approx(labelled, rel=0.02), name
        assert {"0"} <= {t.get_text() for t in labels}
        assert bar_px <= 0.81 * panel.bbox.width

        ratio = metres_per_px * figure.dpi * 39.37
        denominator = float(scale_text.rsplit(":", 1)[1])
        if scale_text.startswith("SCALE 1:"):
            assert denominator == pytest.approx(ratio, rel=0.03)
        else:
            assert scale_text.startswith("APPROX. SCALE 1:")
            assert denominator == pytest.approx(ratio, rel=0.05)

        renderer = figure.canvas.get_renderer()
        boxes = sorted((t.get_window_extent(renderer) for t in labels), key=lambda b: b.x0)
        for left, right in zip(boxes, boxes[1:]):
            assert left.x1 <= right.x0, name
        units = next(t for t in panel.texts if t.get_text() == "Metres").get_window_extent(renderer)
        for box in boxes:
            assert not box.overlaps(units)
    finally:
        plt.close(figure)


def test_portrait_export_recomputes_printed_scale() -> None:
    data = _section(*_SECTIONS["synth_480m"])
    landscape = _render(data, "letter_landscape")
    portrait = _render(data, "letter_portrait")
    try:
        assert _measure(landscape)[2] != _measure(portrait)[2]
    finally:
        plt.close(landscape)
        plt.close(portrait)


def test_map_scale_that_does_not_fit_prints_as_shown_and_notes_it(caplog) -> None:
    """A 480 m section can't be drawn at 1:1000 on letter (needs ~19 in)."""
    data = _section(*_SECTIONS["synth_480m"])
    block = ConsultingTitleBlock(section_label="SCALE TEST", map_scale="1:1000")
    with caplog.at_level(logging.WARNING, logger="renderer_consulting"):
        figure = _render(data, "letter_landscape", block)
    try:
        notes = [r.getMessage() for r in caplog.records if "Map scale 1:1000" in r.getMessage()]
        assert notes and "doesn't fit a letter landscape page" in notes[-1]
        assert "(AS SHOWN)" in notes[-1]
        texts = [t.get_text() for ax in figure.axes for t in ax.texts]
        assert "AS SHOWN" in texts and "1:1000" not in texts
        scale_text = _measure(figure)[2]
        assert scale_text.startswith("APPROX. SCALE 1:") and "1:1000" not in scale_text
    finally:
        plt.close(figure)


def _png_metres_per_px(figure, png: bytes, dpi: int) -> float:
    """Horizontal metres per pixel measured in the PNG: main frame spines vs x limits."""
    import io

    import numpy as np
    from PIL import Image

    image = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)
    main = figure._css_main_axes[0]
    pos = main.get_position()  # only to pick the rows the frame spans
    height = image.shape[0]
    rows = slice(int((1.0 - pos.y1) * height) + 3, int((1.0 - pos.y0) * height) - 3)
    spine = np.array([0x37, 0x41, 0x51])  # consulting frame colour
    mask = (np.abs(image[rows] - spine).sum(axis=2) <= 30).mean(axis=0) > 0.9
    columns = np.flatnonzero(mask)
    assert columns.size >= 2, "frame spines not found in the PNG"
    left = columns[columns < columns.min() + 6].mean()
    right = columns[columns > columns.max() - 6].mean()
    lo, hi = main.get_xlim()
    return abs(hi - lo) / (right - left)


@pytest.mark.parametrize("page_preset", ["letter_landscape", "letter_portrait", "tabloid_landscape"])
@pytest.mark.parametrize("ve", [None, 5.0])
def test_fitting_map_scale_is_drawn_exactly(page_preset: str, ve: float | None) -> None:
    """160 m at 1:1000 (~6.6 in) fits every page: the PNG measures 1:1000 (±2 %)."""
    projected, polygons, depths = _section(*_SECTIONS["gwm_160m"])
    renderer = CrossSectionRenderer(
        show_legend=False,
        vertical_exaggeration=ve,
        render_profile=CONSULTING_SECTION_PROFILE,
        consulting_title_block=ConsultingTitleBlock(section_label="SCALE TEST", map_scale="1:1 000"),
        export_framing=ExportFramingConfig(page_preset=page_preset),
    )
    figure = renderer.render(polygons, projected, collar_depths=depths)
    try:
        dpi = 100
        png = renderer.to_png_bytes(figure, dpi=dpi)
        ratio = _png_metres_per_px(figure, png, dpi) * dpi * 39.37
        assert ratio == pytest.approx(1000.0, rel=0.02)
        texts = [t.get_text() for ax in figure.axes for t in ax.texts]
        assert "1:1 000" in texts and "AS SHOWN" not in texts
        assert _measure(figure)[2] == "SCALE 1:1000"
        assert renderer.map_scale_notes(figure) == ()
        # The box sits inside its frame region, centred horizontally.
        main = figure._css_main_axes[0]
        frame, box = main.get_position(original=True), main.get_position()
        assert box.x0 >= frame.x0 - 1e-9 and box.x1 <= frame.x1 + 1e-9
        assert box.x0 - frame.x0 == pytest.approx(frame.x1 - box.x1, abs=1e-6)
        twin = figure._css_main_axes[1]
        if twin is not None:
            figure.canvas.draw()
            assert twin.get_position().bounds == pytest.approx(main.get_position().bounds)
    finally:
        plt.close(figure)


def test_map_scale_leaving_a_tiny_plot_falls_back() -> None:
    """160 m at 1:5000 would be a ~1.3 in strip (< 30 % of the frame)."""
    projected, polygons, depths = _section(*_SECTIONS["gwm_160m"])
    renderer = CrossSectionRenderer(
        show_legend=False,
        render_profile=CONSULTING_SECTION_PROFILE,
        consulting_title_block=ConsultingTitleBlock(section_label="SCALE TEST", map_scale="1:5000"),
        export_framing=ExportFramingConfig(page_preset="letter_landscape"),
    )
    figure = renderer.render(polygons, projected, collar_depths=depths)
    try:
        renderer.to_png_bytes(figure, dpi=72)
        (note,) = renderer.map_scale_notes(figure)
        assert note.startswith("Map scale 1:5000 would leave the section under 30%")
        texts = [t.get_text() for ax in figure.axes for t in ax.texts]
        assert "AS SHOWN" in texts
        main = figure._css_main_axes[0]
        assert main.get_position().width == pytest.approx(main.get_position(original=True).width)
    finally:
        plt.close(figure)


def test_map_scale_box_geometry() -> None:
    from renderer_consulting import format_scale_ratio, map_scale_box, page_description

    frame = (0.1, 0.5, 0.8, 0.4)  # 8.8 x 3.4 in on an 11 x 8.5 page
    box, status, printed = map_scale_box(frame, (11.0, 8.5), 100.0, 30.0, 1000.0, exact_ve=False)
    assert status == "exact" and printed == 1000.0
    assert box[2] * 11.0 == pytest.approx(100.0 * 39.37 / 1000.0)
    assert box[3] == pytest.approx(0.4)  # auto VE fills the frame height
    assert box[0] - 0.1 == pytest.approx(0.9 - (box[0] + box[2]))
    # Exact VE: the height follows the width (aspect 1 in stored units).
    box, status, _ = map_scale_box(frame, (11.0, 8.5), 100.0, 30.0, 1000.0, exact_ve=True)
    assert status == "exact"
    assert box[3] * 8.5 == pytest.approx(box[2] * 11.0 * 0.3)
    # Too tall at that scale with exact VE: fits the frame at the VE instead.
    box, status, printed = map_scale_box(frame, (11.0, 8.5), 100.0, 120.0, 1000.0, exact_ve=True)
    assert status == "no_fit" and box[3] == pytest.approx(0.4)
    assert printed > 1000.0
    _box, status, printed = map_scale_box(frame, (11.0, 8.5), 400.0, 30.0, 1000.0, exact_ve=False)
    assert status == "no_fit" and printed == pytest.approx(400.0 * 39.37 / 8.8)
    assert map_scale_box(frame, (11.0, 8.5), 20.0, 30.0, 1000.0, exact_ve=False)[1] == "too_small"
    assert format_scale_ratio(1000.0) == "1:1000"
    assert format_scale_ratio(2.5) == "1:2.5"
    assert page_description((8.5, 11.0)) == "letter portrait"
    assert page_description((8.0, 6.0)) == "8 x 6 in"


def test_scale_helpers() -> None:
    assert nice_scale_bar_length(87.0) == 50.0
    assert nice_scale_bar_length(9.9) == 5.0
    assert nice_scale_bar_length(2.1) == 2.0
    assert scale_bar_tick_step(100.0) == 20.0
    assert scale_bar_tick_step(50.0) == 10.0
    assert scale_bar_tick_step(20.0) == 5.0
    assert scale_bar_tick_step(30.0) == 10.0
    assert nearest_standard_scale(1934.0) == 2000
    assert nearest_standard_scale(1240.0) == 1250
    # A fitted drawing is never claimed to be at a round scale.
    assert scale_ratio_text(1010.0) == "APPROX. SCALE 1:1010"
    assert scale_ratio_text(1278.0) == "APPROX. SCALE 1:1280"
    assert scale_ratio_text(1934.0) == "APPROX. SCALE 1:1930"
    assert scale_ratio_text(671.0) == "APPROX. SCALE 1:671"
    assert scale_ratio_text(1010.0, snap=True) == "SCALE 1:1000"
    assert parse_map_scale("1:1 500") == 1500.0
    assert parse_map_scale("1:1,000") == 1000.0
    assert parse_map_scale("NTS") is None


def test_scale_text_keeps_a_decimal_on_very_short_sections() -> None:
    from renderer_consulting import scale_ratio_text

    assert scale_ratio_text(3.3) == "APPROX. SCALE 1:3.3"
    assert scale_ratio_text(671) == "APPROX. SCALE 1:671"


@pytest.mark.parametrize("page_preset", ["letter_landscape", "letter_portrait", "tabloid_landscape"])
def test_fit_to_page_without_map_scale_always_prints_approx(section, page_preset: str) -> None:
    """No map scale: the fitted ratio (e.g. 1:1278) printed as "SCALE 1:1250"
    without "APPROX." although the bar drew 1:1278."""
    name, data = section
    figure = _render(data, page_preset)
    try:
        _drawn, _labels, scale_text, metres_per_px, _panel, _px = _measure(figure)
        ratio = metres_per_px * figure.dpi * 39.37
        assert scale_text.startswith("APPROX. SCALE 1:"), (name, scale_text)
        assert float(scale_text.rsplit(":", 1)[1]) == pytest.approx(ratio, rel=0.006)
    finally:
        plt.close(figure)


def test_title_block_scale_defers_to_the_bar_when_not_set() -> None:
    """The model default "1:1000" printed in the SCALE row contradicted the
    true-scale bar ("APPROX. SCALE 1:170")."""
    import matplotlib.pyplot as plt

    import renderer as renderer_mod
    from models import Collar, ConsultingTitleBlock, Lithology
    from pipeline import build_cross_section

    collars = [
        Collar(hole_id=h, easting=20.0 * i, northing=0.0, elevation=100.0, total_depth=10.0)
        for i, h in enumerate(("BH-1", "BH-2", "BH-3"))
    ]
    liths = [Lithology(hole_id=c.hole_id, from_depth=0, to_depth=10, lithology_code="Clay") for c in collars]
    texts = {}
    original = renderer_mod.CrossSectionRenderer.render

    def spy(self, *args, **kwargs):
        fig = original(self, *args, **kwargs)
        texts["t"] = [t.get_text() for t in fig.findobj(lambda o: hasattr(o, "get_text"))]
        texts.setdefault("figs", []).append(fig)
        return fig

    renderer_mod.CrossSectionRenderer.render = spy
    try:
        for block, expected in (
            (ConsultingTitleBlock(section_label="A-A'"), "AS SHOWN"),
            (ConsultingTitleBlock(section_label="A-A'", map_scale="1:500"), "1:500"),
        ):
            build_cross_section(
                collars, liths, [(0.0, 0.0), (40.0, 0.0)],
                render_layout="consulting_section", consulting_title_block=block,
            )
            assert expected in texts["t"]
    finally:
        renderer_mod.CrossSectionRenderer.render = original
        for fig in texts.get("figs", []):
            plt.close(fig)
