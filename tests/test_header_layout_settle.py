"""Header / tick settling: bounded work, readable headers, map-scale fallback.

Dense sections (30 holes, many chemistry labels) re-ran the header passes
and no-render draws dozens of times per export, the fitted x locator
re-measured hundreds of tick labels per ladder step on long sections, and
dense header rows could still overprint with no note. These tests count work
(text measurements, draws) with generous bounds rather than timing it.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure

from export_framing import ExportFramingConfig
from models import Collar, ConsultingTitleBlock, EnvironmentalReading, Lithology, WaterLevel
from pipeline import compute_section_geometry, render_cross_section_from_geometry
from renderer_common import HEADER_MIN_GAP_IN, fitted_x_locators, header_overlaps

_CODES = ("Fill", "Clay", "Sand", "Clay Loam", "Silty Clay")
_VALUES = (8.0, 1110.0, 1850.0, 494.0, 30.9, 760.0)


def _section(n: int, spacing: float, ids=None, xs=None):
    collars, lith, readings, water = [], [], [], []
    for i in range(n):
        hole = ids[i] if ids else (f"BH24-{i:02d} / HA25-{i:02d}" if i % 2 == 0 else f"BH23-{i:02d}")
        td = 12.0 + (i % 3)
        x = xs[i] if xs else i * spacing
        collars.append(Collar(hole_id=hole, easting=x, northing=0.0, elevation=635.0 - (i % 4), total_depth=td))
        top = 0.0
        for j, code in enumerate(_CODES):
            bottom = min(td, top + (0.5 if j == 0 else td / 4.0))
            if bottom > top:
                lith.append(Lithology(hole_id=hole, from_depth=top, to_depth=bottom, lithology_code=code))
            top = bottom
        for k in range(4):
            readings.append(
                EnvironmentalReading(
                    hole_id=hole, parameter="Chloride", value=_VALUES[(i + k) % 6],
                    depth=0.8 + k * (td - 1.0) / 4.0, unit="mg/kg",
                )
            )
        water.append(WaterLevel(hole_id=hole, depth=2.0 + (i % 3) * 0.3))
    end = xs[-1] if xs else max(spacing * (n - 1), 1.0)
    return collars, lith, readings, water, [(0.0, 0.0), (end, 0.0)]


def _render(data, layout, page, ve, *, title_block=None):
    collars, lith, readings, water, points = data
    geometry = compute_section_geometry(collars, lith, points)
    result = render_cross_section_from_geometry(
        geometry,
        points,
        vertical_exaggeration=ve,
        render_layout=layout,
        environmental_readings=readings,
        environmental_parameters=("Chloride",),
        show_parameter_labels=True,
        water_levels=water,
        chemistry_color_mode="threshold",
        chemistry_threshold_green_max=250.0,
        chemistry_threshold_yellow_max=1000.0,
        export_framing=ExportFramingConfig(page_preset=page, export_dpi=72) if page else None,
        export_formats=frozenset({"png"}),
        close_figure=False,
        consulting_title_block=title_block,
    )
    bundle = result.retained_export_bundle
    return result, bundle["figure"], bundle["renderer"]


def _headers(renderer, figure):
    return [
        t for t in getattr(renderer, "_header_labels", [])
        if t.figure is figure and t.get_visible() and t.get_text().strip()
    ]


@pytest.fixture
def count_text_measurements(monkeypatch):
    calls = {"n": 0}
    original = RendererAgg.get_text_width_height_descent

    def counted(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RendererAgg, "get_text_width_height_descent", counted)
    return calls


def test_fitted_x_locator_measures_each_label_once_and_caches_the_step(count_text_measurements) -> None:
    # 100 km on one axes: the 10 m base step ladders up through ~20 steps,
    # each of which measured up to 200 labels (712k measurements per export).
    fig = plt.figure(figsize=(11.0, 8.5), dpi=150)
    try:
        ax = fig.add_subplot(111)
        ax.set_xlim(0.0, 100_000.0)
        major, _minor = fitted_x_locators(ax.xaxis, 10.0)
        fig.canvas.draw()
        step = major.major_step(*ax.get_xlim())
        assert step >= 10_000.0
        first = count_text_measurements["n"]
        assert first < 1500
        for _ in range(5):
            ax.xaxis.get_majorticklocs()
            ax.xaxis.get_minorticklocs()
        assert count_text_measurements["n"] == first  # cached for the same box
    finally:
        plt.close(fig)


def test_long_section_export_bounded_text_measurements(count_text_measurements) -> None:
    _result, figure, _renderer = _render(_section(6, 20_000.0), "consulting_section", "letter_landscape", 20.0)
    plt.close(figure)
    assert count_text_measurements["n"] < 60_000


def test_dense_chart_export_bounded_no_render_draws(monkeypatch) -> None:
    draws = {"n": 0}
    original = Figure.draw_without_rendering

    def counted(self, *args, **kwargs):
        draws["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "draw_without_rendering", counted)
    _result, figure, _renderer = _render(_section(30, 10.0), "chart", "letter_landscape", 5.0)
    plt.close(figure)
    # 46 before the settle passes were deduplicated.
    assert draws["n"] <= 24


@pytest.mark.parametrize(
    ("n", "spacing", "layout", "page", "ve"),
    [
        (30, 5.0, "consulting_section", "letter_landscape", 20.0),
        (30, 5.0, "chart", "letter_landscape", None),
        (15, 4.0, "consulting_section", "letter_portrait", 20.0),
        (15, 4.0, "consulting_section", "tabloid_landscape", 5.0),
    ],
)
def test_dense_headers_never_overprint_without_a_note(n, spacing, layout, page, ve) -> None:
    result, figure, renderer = _render(_section(n, spacing), layout, page, ve)
    try:
        figure.canvas.draw()
        overlaps = header_overlaps(figure, _headers(renderer, figure), figure.canvas.get_renderer(), pad_pt=0.0)
        noted = any(w.startswith("Hole labels overlap at this scale") for w in result.overlap_warnings)
        assert not overlaps or noted
        if overlaps:
            assert any(w.startswith("Hole labels overlap") for w in result.qa_notes)
        # Every header stays on the page.
        r = figure.canvas.get_renderer()
        page_box = figure.bbox
        for text in _headers(renderer, figure):
            box = text.get_window_extent(r)
            assert box.y1 <= page_box.y1 + 0.5 and box.x0 >= -0.5 and box.x1 <= page_box.x1 + 0.5, text.get_text()
    finally:
        plt.close(figure)


_CLIENT_IDS = [
    "BH24-12 / HA25-02", "BH23-04", "2017-BH11 / BH24-11", "BH23-03", "2017-BH12", "BH24-08", "BH23-01",
]
_CLIENT_XS = [0.0, 5.0, 10.0, 15.0, 20.0, 26.0, 32.0]


@pytest.mark.parametrize("ve", [5.0, 20.0])
def test_side_by_side_headers_keep_a_clear_gap(ve) -> None:
    _result, figure, renderer = _render(
        _section(7, 5.0, ids=_CLIENT_IDS, xs=_CLIENT_XS), "consulting_section", "letter_landscape", ve
    )
    try:
        figure.canvas.draw()
        r = figure.canvas.get_renderer()
        headers = [t for t in _headers(renderer, figure) if abs(t.get_rotation()) % 180 < 1.0]
        boxes = sorted((t.get_window_extent(r) for t in headers), key=lambda b: b.x0)
        gap_px = HEADER_MIN_GAP_IN * figure.dpi - 1.0
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                same_row = a.y0 < b.y1 and b.y0 < a.y1
                if same_row and b.x0 >= a.x0:
                    assert b.x0 - a.x1 >= gap_px or b.x0 >= a.x1 + gap_px, (a, b)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("scale", ["1:5000", "1:100000"])
def test_map_scale_too_coarse_keeps_the_fitted_x_range(scale) -> None:
    data = _section(7, 5.0)
    _r0, plain_fig, _ = _render(data, "consulting_section", "letter_landscape", 5.0)
    plain_right = plain_fig._css_main_axes[0].get_xlim()[1]
    plt.close(plain_fig)
    block = ConsultingTitleBlock(map_scale=scale, section_label="A - A'")
    result, figure, renderer = _render(data, "consulting_section", "letter_landscape", 5.0, title_block=block)
    try:
        assert any("AS SHOWN" in note for note in result.qa_notes)
        ax = figure._css_main_axes[0]
        # Was 0-108 m at 1:5000 and 0-1583 m at 1:100000 (section ~2% of the frame).
        assert ax.get_xlim()[1] == pytest.approx(plain_right, rel=0.02)
        # The fanned / stacked headers keep off the rotated RL axis labels.
        figure.canvas.draw()
        r = figure.canvas.get_renderer()
        labels = [a.yaxis.label.get_window_extent(r) for a in figure.axes if a.axison and a.yaxis.label.get_text()]
        for text in _headers(renderer, figure):
            box = text.get_window_extent(r)
            assert not any(box.overlaps(label) for label in labels), text.get_text()
    finally:
        plt.close(figure)


def test_exact_map_scale_still_adds_last_hole_label_room() -> None:
    data = _section(7, 5.0)
    block = ConsultingTitleBlock(map_scale="1:200", section_label="A - A'")
    result, figure, _renderer = _render(data, "consulting_section", "letter_landscape", None, title_block=block)
    try:
        assert not any("Map scale" in note for note in result.qa_notes)
        ax = figure._css_main_axes[0]
        assert ax.get_xlim()[1] > 30.0 + 2.0
    finally:
        plt.close(figure)


# --- Long hole IDs / lithology codes and the raster size cap -------------------


def test_fit_label_wraps_then_ellipsizes() -> None:
    from renderer_common import fit_label

    assert fit_label("2017-BH11 / BH24-11") == "2017-BH11 / BH24-11"
    assert fit_label("MW-01 deep nest north bank") == "MW-01 deep nest\nnorth bank"
    lines = fit_label("X" * 120).split("\n")
    assert len(lines) == 2 and lines[-1].endswith("…")
    assert all(len(line) <= 20 for line in lines)


def _long_id_section(id_len: int, code_len: int):
    ids = ["BH-1", "X" * id_len, "BH-3"]
    collars = [
        Collar(hole_id=h, easting=i * 50.0, northing=0.0, elevation=100.0, total_depth=20.0)
        for i, h in enumerate(ids)
    ]
    lith = [Lithology(hole_id=h, from_depth=0.0, to_depth=10.0, lithology_code="Clay") for h in ids]
    lith += [Lithology(hole_id=h, from_depth=10.0, to_depth=20.0, lithology_code="L" * code_len) for h in ids]
    return collars, lith


@pytest.mark.parametrize("layout", ["section_sheet", "consulting_section", "chart"])
def test_very_long_hole_id_and_code_stay_on_the_page(layout) -> None:
    collars, lith = _long_id_section(120, 120)
    points = [(0.0, 0.0), (100.0, 0.0)]
    result = render_cross_section_from_geometry(
        compute_section_geometry(collars, lith, points),
        points,
        render_layout=layout,
        export_framing=ExportFramingConfig(page_preset="letter_portrait", export_dpi=72),
        export_formats=frozenset({"png"}),
        close_figure=False,
    )
    figure = result.retained_export_bundle["figure"]
    try:
        figure.canvas.draw()
        r = figure.canvas.get_renderer()
        content = figure.get_tightbbox(r)
        width, height = figure.get_size_inches()
        assert content.x0 >= 0.2 and content.y0 >= 0.2
        assert content.x1 <= width - 0.2 and content.y1 <= height - 0.2
        ax = getattr(figure, "_css_ve_axes", None) or figure.axes[0]
        # The section keeps a usable width (was a ~1.7 in strip).
        assert ax.get_window_extent(r).width / figure.dpi > 3.0
    finally:
        plt.close(figure)


def test_png_resolution_capped_with_a_note(monkeypatch) -> None:
    import io

    from PIL import Image

    from renderer import CrossSectionRenderer

    monkeypatch.setattr(CrossSectionRenderer, "MAX_PNG_PIXELS", 200_000)
    result, figure, _renderer = _render(_section(3, 10.0), "section_sheet", "letter_landscape", None)
    plt.close(figure)
    w, h = Image.open(io.BytesIO(result.png_bytes)).size
    assert w * h <= 200_000
    assert any(note.startswith("PNG resolution lowered") for note in result.qa_notes)
