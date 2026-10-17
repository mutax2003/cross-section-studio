"""Exported page layout at exact VE, map scale and auto VE.

Short sections (holes a few metres apart) printed at a fixed VE on letter /
tabloid pages: hole headers overprinted, x tick labels ran together,
chemistry values sat on the neighbouring column, content reached into the
printable margin, and legends missed keys for symbols the figure drew. These
tests check the exported figure (after the export page prep), not the
render-size one.
"""

from __future__ import annotations

import io
import re

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.lines import Line2D

from export_framing import ExportFramingConfig
from models import Collar, EnvironmentalReading, Lithology, WaterLevel
from pipeline import compute_section_geometry, render_cross_section_from_geometry
from render_theme import CHEMISTRY_LABEL_GREEN, CHEMISTRY_LABEL_ORANGE, CHEMISTRY_LABEL_RED
from renderer_common import (
    MIN_PRINT_MARGIN_IN,
    ThresholdDotsKey,
    header_overlaps,
    measured_vertical_exaggeration,
)

# Client-like short transect: long, paired IDs a few metres apart.
_HOLES = [
    ("BH24-12 / HA25-02", 0.0, 12.0),
    ("BH23-04", 5.0, 15.3),
    ("2017-BH11 / BH24-11", 10.0, 14.5),
    ("BH23-03", 15.0, 12.4),
    ("2017-BH12", 20.0, 9.0),
    ("BH24-08", 26.0, 8.0),
    ("BH23-01", 32.0, 6.0),
]
_POINTS = [(0.0, 0.0), (32.0, 0.0)]


def _short_section():
    collars = [
        Collar(hole_id=h, easting=x, northing=0.0, elevation=635.0, total_depth=td) for h, x, td in _HOLES
    ]
    lith = []
    for h, _x, td in _HOLES:
        top = 0.0
        for j, code in enumerate(("Fill", "Clay", "Sand", "Clay Loam", "Silty Clay")):
            bottom = min(td, top + (0.5 if j == 0 else td / 4.0))
            if bottom > top:
                lith.append(Lithology(hole_id=h, from_depth=top, to_depth=bottom, lithology_code=code))
            top = bottom
    readings = []
    values = (8.0, 1110.0, 1850.0, 494.0, 30.9, 760.0)
    for i, (h, _x, td) in enumerate(_HOLES):
        for k in range(4):
            readings.append(
                EnvironmentalReading(
                    hole_id=h,
                    parameter="Chloride",
                    value=values[(i + k) % len(values)],
                    depth=0.8 + k * (td - 1.0) / 4.0,
                    unit="mg/kg",
                )
            )
    return collars, lith, readings


_GEOMETRY_CACHE: dict = {}


def _geometry():
    if "short" not in _GEOMETRY_CACHE:
        collars, lith, readings = _short_section()
        _GEOMETRY_CACHE["short"] = (compute_section_geometry(collars, lith, _POINTS), readings)
    return _GEOMETRY_CACHE["short"]


def _export(layout, page, ve, **kwargs):
    geometry, readings = _geometry()
    framing = ExportFramingConfig(page_preset=page, export_dpi=72) if page else None
    kwargs.setdefault("environmental_readings", readings)
    kwargs.setdefault("environmental_parameters", ("Chloride",))
    kwargs.setdefault("show_parameter_labels", True)
    result = render_cross_section_from_geometry(
        geometry,
        _POINTS,
        vertical_exaggeration=ve,
        render_layout=layout,
        chemistry_color_mode=kwargs.pop("chemistry_color_mode", "threshold"),
        chemistry_threshold_green_max=kwargs.pop("green", 250.0),
        chemistry_threshold_yellow_max=kwargs.pop("orange", 1000.0),
        export_framing=framing,
        export_formats=frozenset({"png"}),
        close_figure=False,
        **kwargs,
    )
    bundle = result.retained_export_bundle
    figure = bundle["figure"]
    figure.canvas.draw()
    return result, figure, bundle["renderer"]


def _close(figure):
    plt.close(figure)


def _headers(renderer, figure):
    return [
        t
        for t in getattr(renderer, "_header_labels", [])
        if t.figure is figure and t.get_visible() and t.get_text().strip()
    ]


def _main_ax(figure):
    return getattr(figure, "_css_ve_axes", None) or figure.axes[0]


_LAYOUT_PAGES = [
    (layout, page, ve)
    for layout in ("consulting_section", "section_sheet", "chart")
    for page, ve in (("letter_landscape", 5.0), ("letter_landscape", 20.0), ("letter_portrait", 20.0))
]


_DEPTH_MODE_PAGES = [
    ("section_sheet", page, ve, "relative")
    for page, ve in (("tabloid_landscape", 20.0), ("letter_portrait", 20.0), (None, 20.0), ("letter_landscape", 5.0))
]


@pytest.mark.parametrize(
    ("layout", "page", "ve", "elevation_mode"),
    [(*row, "absolute") for row in _LAYOUT_PAGES] + _DEPTH_MODE_PAGES,
)
def test_exact_ve_short_section_headers_and_ticks_never_overprint(layout, page, ve, elevation_mode) -> None:
    # Depth mode puts the headers above the plot, under a title whose layout
    # reacts to their height (and so re-sizes the aspect-locked box).
    _result, figure, renderer = _export(layout, page, ve, elevation_mode=elevation_mode)
    try:
        r = figure.canvas.get_renderer()
        headers = _headers(renderer, figure)
        assert len(headers) == len(_HOLES)
        assert header_overlaps(figure, headers, r, pad_pt=0.0) == []
        # Every header reads as its own hole: over its column, or joined by a leader.
        leaders = figure.__dict__.get("_css_header_leaders") or []
        if not leaders:
            for text in headers:
                box = text.get_window_extent(r)
                base = getattr(text, "_header_base", None)
                transform = base[0] if base else text.get_transform()
                anchor = float(transform.transform(text.get_unitless_position())[0])
                assert box.x0 - 1.0 <= anchor <= box.x1 + 1.0, text.get_text()
        ax = _main_ax(figure)
        low, high = sorted(ax.get_xlim())
        labels = [
            lab
            for loc, lab in zip(ax.xaxis.get_majorticklocs(), ax.xaxis.get_ticklabels(), strict=False)
            if low - 1e-9 <= loc <= high + 1e-9 and lab.get_visible() and lab.get_text()
        ]
        assert len(labels) >= 2
        boxes = [lab.get_window_extent(r) for lab in labels]
        for left, right in zip(boxes, boxes[1:], strict=False):
            assert left.x1 + 1.0 <= right.x0, "x tick labels touch"
    finally:
        _close(figure)


@pytest.mark.parametrize("ve", [5.0, 20.0])
@pytest.mark.parametrize("elevation_mode", ["absolute", "relative"])
def test_section_sheet_fixed_ve_chemistry_values_stay_beside_their_own_column(ve, elevation_mode) -> None:
    result, figure, renderer = _export(
        "section_sheet", "letter_landscape", ve, elevation_mode=elevation_mode, chemistry_color_mode="red"
    )
    try:
        r = figure.canvas.get_renderer()
        ax = _main_ax(figure)
        hole_px = {
            h: float(ax.transData.transform((x, 0.0))[0]) for h, x, _td in _HOLES
        }
        labels = [a for kind, a, _c in renderer._water_labels if kind == "chem"]
        shown = [a for a in labels if a.get_visible()]
        dropped = [a for a in labels if getattr(a, "_water_dropped", False)]
        assert shown
        for annotation in shown:
            box = annotation.get_window_extent(r)
            own = str(annotation._chem_hole_id)
            for hole, x_px in hole_px.items():
                if hole != own:
                    assert not (box.x0 + 1.0 < x_px < box.x1 - 1.0), (annotation.get_text(), own, hole)
            frame = ax.get_window_extent(r)
            assert box.x1 <= frame.x1 + 1.0
        # Any value that had to go is named in the build notes.
        if dropped:
            assert any(w.startswith("Chemistry labels:") for w in result.overlap_warnings)
    finally:
        _close(figure)


@pytest.mark.parametrize("layout", ["consulting_section", "section_sheet", "chart"])
@pytest.mark.parametrize("page", ["letter_portrait", "letter_landscape", "tabloid_landscape"])
@pytest.mark.parametrize("ve", [None, 5.0])
def test_exported_page_keeps_quarter_inch_margins(layout, page, ve) -> None:
    _result, figure, _renderer = _export(layout, page, ve)
    try:
        r = figure.canvas.get_renderer()
        content = figure.get_tightbbox(r)
        width, height = figure.get_size_inches()
        margins = (content.x0, width - content.x1, height - content.y1, content.y0)
        assert min(margins) >= MIN_PRINT_MARGIN_IN - 0.01, margins
    finally:
        _close(figure)


def _legend_keys(figure):
    legend = _main_ax(figure).get_legend()
    assert legend is not None
    return {text.get_text(): handle for text, handle in zip(legend.get_texts(), legend.legend_handles, strict=False)}


def _chem_handle(figure):
    keys = _legend_keys(figure)
    label = next(k for k in keys if "CHLORIDE" in k.upper())
    return label, keys[label]


@pytest.mark.parametrize("layout", ["section_sheet", "chart"])
def test_section_sheet_and_chart_legends_key_chemistry(layout) -> None:
    _r, figure, renderer = _export(layout, None, None)
    try:
        legend = _main_ax(figure).get_legend()
        texts = [t.get_text() for t in legend.get_texts()]
        assert any("CHLORIDE" in t.upper() for t in texts)
        source = renderer.parameter_series_legend[0]
        assert source["threshold_colors"] == (CHEMISTRY_LABEL_GREEN, CHEMISTRY_LABEL_ORANGE, CHEMISTRY_LABEL_RED)
        dots = [
            child
            for child in legend.findobj(Line2D)
            if child.get_gid() == "parameter-legend-threshold-dots"
        ]
        assert len(dots) == 3
        if layout == "section_sheet":
            assert "Ground surface" in texts
    finally:
        _close(figure)
    # Empty orange band (orange limit == green limit): only green and red dots.
    _r, figure, renderer = _export("section_sheet", None, None, green=250.0, orange=250.0)
    try:
        entry = renderer.parameter_series_legend[0]
        assert entry["threshold_colors"] == (CHEMISTRY_LABEL_GREEN, CHEMISTRY_LABEL_RED)
        legend = _main_ax(figure).get_legend()
        dots = [c for c in legend.findobj(Line2D) if c.get_gid() == "parameter-legend-threshold-dots"]
        assert len(dots) == 2
    finally:
        _close(figure)


@pytest.mark.parametrize("mode", ["black", "red"])
def test_single_colour_chemistry_legend_key_is_the_drawn_marker(mode) -> None:
    from matplotlib.collections import PathCollection
    from matplotlib.colors import to_hex

    _r, figure, _renderer = _export("section_sheet", None, None, chemistry_color_mode=mode)
    try:
        ax = _main_ax(figure)
        legend = ax.get_legend()
        texts = [t.get_text() for t in legend.get_texts()]
        index = next(i for i, t in enumerate(texts) if "CHLORIDE" in t.upper())
        handle = legend.legend_handles[index]
        assert isinstance(handle, Line2D)
        drawn = {
            to_hex(c)
            for coll in ax.collections
            if isinstance(coll, PathCollection) and coll.get_zorder() == 8
            for c in coll.get_facecolors()
        }
        assert drawn == {to_hex(handle.get_markerfacecolor())}
        assert handle.get_marker() not in (None, "", "None")
    finally:
        _close(figure)


def test_legend_threshold_dots_key_draws_one_dot_per_band() -> None:
    from renderer_common import LEGEND_KEY_HANDLER_MAP

    figure, ax = plt.subplots()
    try:
        legend = ax.legend(
            handles=[ThresholdDotsKey(("#00aa00", "#cc0000"))],
            labels=["CHLORIDE"],
            handler_map=LEGEND_KEY_HANDLER_MAP,
        )
        figure.canvas.draw()
        dots = [
            child
            for child in legend.findobj(Line2D)
            if child.get_gid() == "parameter-legend-threshold-dots"
        ]
        assert len(dots) == 2
    finally:
        plt.close(figure)


def test_nm_dry_notes_get_a_legend_entry_only_when_drawn() -> None:
    hole_ids = [h for h, _x, _td in _HOLES]
    measured = WaterLevel(hole_id=hole_ids[0], depth=3.0, series_id="s1")
    dry = WaterLevel(hole_id=hole_ids[1], depth=0.0, series_id="s1", status="dry")
    for layout in ("section_sheet", "chart"):
        _r, figure, _renderer = _export(layout, None, None, water_levels=[measured, dry])
        try:
            texts = [t.get_text() for t in _main_ax(figure).get_legend().get_texts()]
            assert any("dry" in t.lower() and "—" in t for t in texts), texts
        finally:
            _close(figure)
    _r, figure, _renderer = _export("consulting_section", "letter_landscape", None, water_levels=[measured, dry])
    try:
        texts = [t.get_text() for ax in figure.axes for t in ax.texts]
        assert any("DRY" in t and "—" in t for t in texts), texts
    finally:
        _close(figure)
    _r, figure, _renderer = _export("section_sheet", None, None, water_levels=[measured])
    try:
        texts = [t.get_text() for t in _main_ax(figure).get_legend().get_texts()]
        assert not any("—" in t and ("dry" in t.lower() or "not measured" in t.lower()) for t in texts)
    finally:
        _close(figure)


def test_section_sheet_water_key_shows_a_line_only_for_joined_series() -> None:
    hole_ids = [h for h, _x, _td in _HOLES]
    levels = [WaterLevel(hole_id=hole_ids[0], depth=3.0, series_id="s1")]
    _r, figure, renderer = _export("section_sheet", None, None, water_levels=levels)
    try:
        ax = _main_ax(figure)
        keys = [a for a in ax.artists if a.get_gid() == "water_series_key"]
        legend = ax.get_legend()
        handles = [h for k in keys for h in k.legend_handles]
        handles += [
            h
            for t, h in zip(legend.get_texts(), legend.legend_handles, strict=False)
            if "GROUNDWATER" in t.get_text().upper() or "WATER" in t.get_text().upper()
        ]
        for handle in handles:
            if isinstance(handle, Line2D):
                assert handle.get_linestyle() in ("None", "none", ""), "one reading: no joining line"
    finally:
        _close(figure)


def test_consulting_pinch_out_entry_only_when_wedges_drawn() -> None:
    collars, lith, _readings = _short_section()
    # A unit in one hole only: drawn as a dashed wedge when pinch-outs are on.
    lith = [
        item.model_copy(update={"lithology_code": "Gravel"})
        if item.hole_id == "BH23-03" and item.lithology_code == "Sand"
        else item
        for item in lith
    ]
    seen = {}
    for pinch in (True, False):
        geometry = compute_section_geometry(collars, lith, _POINTS, allow_pinch_outs=pinch)
        result = render_cross_section_from_geometry(
            geometry, _POINTS, render_layout="consulting_section",
            export_framing=ExportFramingConfig(page_preset="letter_landscape", export_dpi=72),
            export_formats=frozenset({"png"}), close_figure=False,
        )
        figure = result.retained_export_bundle["figure"]
        renderer = result.retained_export_bundle["renderer"]
        try:
            texts = [t.get_text() for ax in figure.axes for t in ax.texts]
            seen[pinch] = ("INFERRED PINCH-OUT" in texts, bool(renderer._has_pinch_out))
        finally:
            plt.close(figure)
    assert seen[True] == (True, True)
    for shown, drawn in seen.values():
        assert shown == drawn


def _long_thin_section():
    collars = [
        Collar(hole_id=f"BH-{i + 1:02d}", easting=60.0 * i, northing=0.0, elevation=748.0 - 0.6 * i, total_depth=14.0)
        for i in range(6)
    ]
    lith = [
        Lithology(hole_id=c.hole_id, from_depth=d, to_depth=d + 3.5, lithology_code=code)
        for c in collars
        for d, code in zip((0.0, 3.5, 7.0, 10.5), ("Clay", "Sand", "Gravel", "Clay"), strict=True)
    ]
    levels = [WaterLevel(hole_id=c.hole_id, depth=3.0, series_id="s1") for c in collars]
    return collars, lith, levels


@pytest.mark.parametrize("page", [None, "letter_landscape"])
def test_chart_ve1_long_section_puts_ve_note_and_scale_bar_under_the_plot(page) -> None:
    collars, lith, levels = _long_thin_section()
    points = [(0.0, 0.0), (300.0, 0.0)]
    geometry = compute_section_geometry(collars, lith, points)
    framing = ExportFramingConfig(page_preset=page, export_dpi=72) if page else None
    result = render_cross_section_from_geometry(
        geometry, points, vertical_exaggeration=1.0, render_layout="chart", water_levels=levels,
        export_framing=framing, export_formats=frozenset({"png"}), close_figure=False,
    )
    figure = result.retained_export_bundle["figure"]
    try:
        figure.canvas.draw()
        r = figure.canvas.get_renderer()
        ax = _main_ax(figure)
        frame = ax.get_window_extent(r)
        note = next(t for t in ax.texts if t.get_gid() == "ve-note")
        bar = next(a for a in ax.artists if a.get_gid() == "scale-bar")
        assert "No V.E." in note.get_text()
        for artist in (note, bar):
            assert artist.get_window_extent(r).y1 <= frame.y0 + 0.5
        # Clear of the distance label too.
        label = ax.xaxis.label.get_window_extent(r)
        for artist in (note, bar):
            box = artist.get_window_extent(r)
            assert box.y1 <= label.y0 + 0.5 or box.x1 <= label.x0 or box.x0 >= label.x1
    finally:
        plt.close(figure)


@pytest.mark.parametrize("page", [None, "letter_landscape", "tabloid_landscape"])
def test_section_sheet_distance_label_hangs_just_under_the_hole_names(page) -> None:
    _r, figure, renderer = _export("section_sheet", page, None)
    try:
        r = figure.canvas.get_renderer()
        supx = figure._supxlabel
        headers = _headers(renderer, figure)
        lowest = min(t.get_window_extent(r).y0 for t in headers)
        gap_pt = (lowest - supx.get_window_extent(r).y1) / figure.dpi * 72.0
        assert 0.0 <= gap_pt <= 14.0, gap_pt
    finally:
        _close(figure)


def _deep_short_section():
    collars = [
        Collar(hole_id=h, easting=x, northing=0.0, elevation=100.0, total_depth=60.0)
        for h, x in (("D1", 0.0), ("D2", 10.0), ("D3", 20.0))
    ]
    lith = [
        Lithology(hole_id=c.hole_id, from_depth=d, to_depth=d + 20.0, lithology_code=code)
        for c in collars
        for d, code in zip((0.0, 20.0, 40.0), ("Clay", "Sand", "Clay"), strict=True)
    ]
    return collars, lith


@pytest.mark.parametrize("layout", ["section_sheet", "chart", "consulting_section"])
@pytest.mark.parametrize("page", [None, "letter_landscape", "tabloid_landscape"])
def test_auto_ve_never_compresses_the_vertical(layout, page) -> None:
    collars, lith = _deep_short_section()
    points = [(0.0, 0.0), (20.0, 0.0)]
    geometry = compute_section_geometry(collars, lith, points)
    framing = ExportFramingConfig(page_preset=page, export_dpi=72) if page else None
    result = render_cross_section_from_geometry(
        geometry, points, vertical_exaggeration=None, render_layout=layout,
        export_framing=framing, export_formats=frozenset({"svg"}), close_figure=False,
    )
    figure = result.retained_export_bundle["figure"]
    try:
        figure.canvas.draw()
        ax = _main_ax(figure)
        ve = measured_vertical_exaggeration(ax, getattr(figure, "_css_ve_storage", 1.0))
        assert ve is not None and ve >= 1.0 - 1e-3
        svg = result.svg_bytes.decode("utf-8")
        assert not re.search(r"≈0\.\d+×", svg), "a compressed VE was printed"
    finally:
        plt.close(figure)


def test_auto_ve_held_at_one_reads_no_exaggeration_and_centres_the_box() -> None:
    from renderer_common import AutoVEAspect

    figure, ax = plt.subplots(figsize=(11.0, 8.5))
    try:
        ax.set_xlim(0.0, 20.0)
        ax.set_ylim(0.0, 60.0)
        ax.apply_aspect = AutoVEAspect(ax)
        figure.canvas.draw()
        assert measured_vertical_exaggeration(ax) == pytest.approx(1.0, rel=1e-3)
        frame = ax.get_position(original=True)
        box = ax.get_position(original=False)
        assert box.width < frame.width
        assert box.x0 + 0.5 * box.width == pytest.approx(frame.x0 + 0.5 * frame.width, abs=1e-6)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("layout", ["section_sheet", "chart", "consulting_section"])
def test_auto_ve_caption_matches_drawn_box_on_tight_crop_exports(layout) -> None:
    """Measured in display pixels: a bbox_inches='tight' save changes the figure size."""
    collars = [Collar(hole_id=f"B{i}", easting=40.0 * i, northing=0, elevation=100, total_depth=20) for i in range(4)]
    lith = [Lithology(hole_id=f"B{i}", from_depth=0, to_depth=20, lithology_code="Clay") for i in range(4)]
    points = [(0.0, 0.0), (120.0, 0.0)]
    geometry = compute_section_geometry(collars, lith, points)
    result = render_cross_section_from_geometry(
        geometry, points, render_layout=layout, export_formats=frozenset({"svg", "pdf"}), close_figure=False,
    )
    figure = result.retained_export_bundle["figure"]
    try:
        drawn = {}
        ax = _main_ax(figure)

        original_draw = type(figure).draw

        def spy(self, renderer):
            original_draw(self, renderer)
            box = ax.get_window_extent(renderer)
            (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
            storage = getattr(self, "_css_ve_storage", 1.0) or 1.0
            drawn["ve"] = (box.height / (abs(y1 - y0) / storage)) / (box.width / abs(x1 - x0))

        figure.draw = spy.__get__(figure)
        figure.savefig(io.BytesIO(), format="svg", bbox_inches="tight")
        svg = result.svg_bytes.decode("utf-8")
        printed = [float(v) for v in re.findall(r"≈([\d.]+)×", svg)]
        assert printed
        assert printed[0] == pytest.approx(drawn["ve"], abs=0.051 * max(1.0, drawn["ve"]))
        assert measured_vertical_exaggeration(ax, getattr(figure, "_css_ve_storage", 1.0)) == pytest.approx(
            drawn["ve"], rel=1e-3
        )
    finally:
        plt.close(figure)


@pytest.mark.parametrize("ve", [None, 2.0, 5.0, 10.0])
def test_consulting_exact_ve_keeps_last_hole_chemistry_values(ve) -> None:
    from tests.test_chem_threshold_dots_and_last_hole import _readings, _section

    collars, lith = _section()
    points = [(0.0, 0.0), (100.0, 0.0)]
    geometry = compute_section_geometry(collars, lith, points)
    result = render_cross_section_from_geometry(
        geometry, points, vertical_exaggeration=ve, render_layout="consulting_section",
        environmental_readings=_readings(5, 25.0), environmental_parameters=["Chloride"],
        show_parameter_labels=True, export_formats=frozenset({"png"}),
        export_framing=ExportFramingConfig(page_preset="tabloid_landscape", export_dpi=72),
        close_figure=False,
    )
    figure = result.retained_export_bundle["figure"]
    try:
        assert not [w for w in result.overlap_warnings if w.startswith("Chemistry")]
    finally:
        plt.close(figure)


def test_exact_ve_x_ticks_follow_the_drawn_box_width() -> None:
    _r, figure, _renderer = _export("consulting_section", "letter_portrait", 20.0)
    try:
        ax = _main_ax(figure)
        ticks = [t for t in ax.xaxis.get_majorticklocs() if min(ax.get_xlim()) <= t <= max(ax.get_xlim())]
        assert len(ticks) >= 2
        step = float(np.diff(ticks)[0])
        assert step > 10.0, "the 10 m client step cannot fit a 20x box on letter portrait"
    finally:
        _close(figure)
