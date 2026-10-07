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


def test_mismatched_map_scale_is_kept_in_title_block_and_logged(caplog) -> None:
    data = _section(*_SECTIONS["synth_480m"])
    block = ConsultingTitleBlock(section_label="SCALE TEST", map_scale="1:1000")
    with caplog.at_level(logging.WARNING, logger="renderer_consulting"):
        figure = _render(data, "letter_landscape", block)
    try:
        assert any("differs from the printed section scale" in r.getMessage() for r in caplog.records)
        texts = [t.get_text() for ax in figure.axes for t in ax.texts]
        assert "1:1000" in texts  # user's value stays in the title block
        assert "SCALE 1:1000" not in _measure(figure)[2]
    finally:
        plt.close(figure)


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
    assert scale_ratio_text(1010.0) == "SCALE 1:1000"
    assert scale_ratio_text(1934.0) == "APPROX. SCALE 1:1900"
    assert scale_ratio_text(671.0) == "APPROX. SCALE 1:670"
    assert parse_map_scale("1:1 500") == 1500.0
    assert parse_map_scale("1:1,000") == 1000.0
    assert parse_map_scale("NTS") is None


def test_scale_text_keeps_a_decimal_on_very_short_sections() -> None:
    from renderer_consulting import scale_ratio_text

    assert scale_ratio_text(3.3) == "APPROX. SCALE 1:3.3"
    assert scale_ratio_text(671) == "APPROX. SCALE 1:670"


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
