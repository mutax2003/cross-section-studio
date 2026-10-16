"""Chemistry legend entries, threshold key and marker shapes on a fixed page.

A 40-character parameter name went into the legend unwrapped (the lithology
entries wrap at ``LEGEND_LABEL_MAX_CHARS``) and pushed the key 1.4 in past the
right edge of a letter page; the same name inline in the threshold key ran the
row past the frame. With two parameters plotted, every dot and every legend
sample was the same circle, so "0.4" and "380" at one hole could not be told
apart. These tests check the exported figure after the page prep.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
from matplotlib.lines import Line2D

from export_framing import ExportFramingConfig
from models import Collar, EnvironmentalReading, Lithology
from pipeline import compute_section_geometry, render_cross_section_from_geometry
from renderer_chemistry import parameter_series_markers
from renderer_common import LEGEND_LABEL_MAX_CHARS, THRESHOLD_KEY_GID

LONG_NAME = "ü" * 40
_POINTS = [(0.0, 0.0), (235.0, 0.0)]


def _section(parameters):
    collars = [
        Collar(hole_id=f"BH-{i}", easting=i * 117.5, northing=0.0, elevation=10.0, total_depth=20.0)
        for i in range(3)
    ]
    lith = [
        Lithology(hole_id=c.hole_id, from_depth=a, to_depth=b, lithology_code=code)
        for c in collars
        for a, b, code in ((0.0, 8.0, "Sand"), (8.0, 20.0, "Clay"))
    ]
    readings = [
        EnvironmentalReading(
            hole_id=c.hole_id, parameter=p, value=v * (i + 1), depth=3.0 + 4.0 * k, unit="mg/kg"
        )
        for i, c in enumerate(collars)
        for k, (p, v) in enumerate(zip(parameters, (0.4, 380.0, 12.0), strict=False))
    ]
    return collars, lith, readings


def _export(layout, parameters, **kwargs):
    collars, lith, readings = _section(parameters)
    geometry = compute_section_geometry(collars, lith, _POINTS)
    kwargs.setdefault("chemistry_color_mode", "threshold")
    result = render_cross_section_from_geometry(
        geometry,
        _POINTS,
        render_layout=layout,
        environmental_readings=readings,
        environmental_parameters=tuple(parameters),
        show_parameter_labels=True,
        chemistry_threshold_green_max=1.0,
        chemistry_threshold_yellow_max=100.0,
        export_framing=ExportFramingConfig(page_preset="letter_portrait", export_dpi=72, margin_right_in=0.5),
        export_formats=frozenset({"png"}),
        close_figure=False,
        **kwargs,
    )
    bundle = result.retained_export_bundle
    figure = bundle["figure"]
    figure.canvas.draw()
    return figure, bundle["renderer"]


def _main_ax(figure):
    return getattr(figure, "_css_ve_axes", None) or figure.axes[0]


def _legend_keys(figure):
    legend = _main_ax(figure).get_legend()
    assert legend is not None
    return {
        text.get_text(): handle
        for text, handle in zip(legend.get_texts(), legend.legend_handles, strict=False)
    }


def _inches(artist, figure):
    renderer = figure.canvas.get_renderer()
    return artist.get_window_extent(renderer).transformed(figure.dpi_scale_trans.inverted())


def _threshold_key(figure):
    ax = _main_ax(figure)
    return next(a for a in ax.artists if a.get_gid() == THRESHOLD_KEY_GID)


@pytest.mark.parametrize("layout", ["section_sheet", "chart"])
def test_long_parameter_name_keeps_the_legend_on_the_page(layout) -> None:
    figure, renderer = _export(layout, [LONG_NAME, "PHC F2"])
    try:
        width, height = (float(v) for v in figure.get_size_inches())
        box = _inches(_main_ax(figure).get_legend(), figure)
        assert box.x0 >= 0.0 and box.y0 >= 0.0
        assert box.x1 <= width - 0.1 and box.y1 <= height
        label = next(k for k in _legend_keys(figure) if "Ü" in k)
        lines = label.split("\n")
        assert len(lines) == 2
        assert all(len(line) <= LEGEND_LABEL_MAX_CHARS for line in lines)
        entry = next(e for e in renderer.parameter_series_legend if e["parameter"] == LONG_NAME)
        assert "\n" in entry["label"]
    finally:
        plt.close(figure)


def test_threshold_key_puts_a_long_parameter_name_on_its_own_row() -> None:
    long_figure, long_renderer = _export("section_sheet", ["Benzene", LONG_NAME])
    short_figure, _short_renderer = _export("section_sheet", ["Benzene", "Chloride"])
    try:
        long_key = _threshold_key(long_figure)
        short_key = _threshold_key(short_figure)
        long_box = _inches(long_key, long_figure)
        short_box = _inches(short_key, short_figure)
        width, _height = (float(v) for v in long_figure.get_size_inches())
        # Inside the page, and no wider than the key with short names (the
        # band row sets the width; the long name wraps on a row of its own).
        assert long_box.x0 >= 0.0 and long_box.y0 >= 0.0 and long_box.x1 <= width
        assert long_box.width <= short_box.width * 1.02
        assert len(long_key._css_rows) == 3 and len(short_key._css_rows) == 2
        # The text form of the key keeps the full name (not page-limited).
        assert long_renderer.chemistry_threshold_key_lines[1].startswith(LONG_NAME)
    finally:
        plt.close(long_figure)
        plt.close(short_figure)


def test_two_parameters_get_distinct_marker_shapes_on_the_figure_and_in_the_legend() -> None:
    figure, renderer = _export("section_sheet", ["Benzene", "Chloride"], chemistry_color_mode="red")
    try:
        markers = {entry["parameter"]: entry["marker"] for entry in renderer.parameter_series_legend}
        assert markers["Benzene"] == "o"
        assert markers["Chloride"] != "o"
        ax = _main_ax(figure)
        chem = [c for c in ax.__dict__.get("_css_marker_sets", []) if c._css_marker_kind == "chem"]
        assert len(chem) == 2
        paths = {tuple(map(tuple, c.get_paths()[0].vertices.round(6))) for c in chem}
        assert len(paths) == 2, "both parameters drew the same marker path"
        handles = {
            label: handle
            for label, handle in _legend_keys(figure).items()
            if "BENZENE" in label.upper() or "CHLORIDE" in label.upper()
        }
        assert len(handles) == 2 and all(isinstance(h, Line2D) for h in handles.values())
        assert {h.get_marker() for h in handles.values()} == {"o", markers["Chloride"]}
    finally:
        plt.close(figure)


def test_single_parameter_keeps_the_configure_marker() -> None:
    figure, renderer = _export("section_sheet", ["Chloride"], chemistry_color_mode="red")
    try:
        assert [entry["marker"] for entry in renderer.parameter_series_legend] == ["o"]
    finally:
        plt.close(figure)


def test_parameter_series_markers_follow_configure_order() -> None:
    assert parameter_series_markers(["Benzene", "Chloride", "pH"], "o") == {
        "Benzene": "o",
        "Chloride": "s",
        "pH": "^",
    }
    # A non-default Configure marker is kept for the first parameter and skipped after.
    assert parameter_series_markers(["A", "B"], "s") == {"A": "s", "B": "o"}
    assert parameter_series_markers(["A"], "D") == {"A": "D"}
    assert parameter_series_markers([], "o") == {}
