"""Chart layout on export pages: title stays on the sheet, legend names every symbol."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from export_framing import ExportFramingConfig
from models import Collar, Lithology, WaterLevel
from pipeline import compute_section_geometry, render_cross_section_from_geometry
from renderer_chart import COLLAR_LEGEND_LABEL

# Close holes with long IDs: the header pass has to raise one a tier.
_HOLES = (
    ("BH24-12 / HA25-02", 0.0),
    ("BH23-04", 5.0),
    ("2017-BH11 / BH24-11", 10.0),
    ("BH23-03", 15.0),
    ("2017-BH12", 20.0),
    ("BH24-08", 26.0),
)
_TRANSECT = [(0.0, 0.0), (26.0, 0.0)]


def _collars() -> list[Collar]:
    return [
        Collar(hole_id=h, easting=x, northing=0.0, elevation=635.0, total_depth=12.0)
        for h, x in _HOLES
    ]


def _lithologies() -> list[Lithology]:
    rows = []
    for h, _x in _HOLES:
        rows.append(Lithology(hole_id=h, from_depth=0.0, to_depth=4.0, lithology_code="Clay"))
        rows.append(Lithology(hole_id=h, from_depth=4.0, to_depth=12.0, lithology_code="Sand"))
    return rows


def _water_levels() -> list[WaterLevel]:
    levels = []
    for series_id, label, depth in (("s1", "May 14, 2026", 2.0), ("s2", "Sept 22, 2026", 3.0)):
        for h, _x in _HOLES:
            levels.append(
                WaterLevel(hole_id=h, depth=depth, series_id=series_id, series_label=label)
            )
    # A series with no measured head draws no marker, so it has no legend entry.
    levels.append(WaterLevel(hole_id=_HOLES[0][0], depth=0.0, series_id="s3", status="dry"))
    return levels


def _render(preset: str, **kwargs):
    geometry = compute_section_geometry(
        _collars(), _lithologies(), _TRANSECT, fail_on_overlaps=False
    )
    result = render_cross_section_from_geometry(
        geometry,
        _TRANSECT,
        title="Chart Layout Page Regression Section Title",
        render_layout="chart",
        export_formats=frozenset({"png"}),
        export_framing=ExportFramingConfig(page_preset=preset, export_dpi=72),
        close_figure=False,
        **kwargs,
    )
    bundle = result.retained_export_bundle
    assert bundle is not None
    return result, bundle["figure"], bundle["renderer"]


@pytest.mark.parametrize("preset", ["letter_landscape", "letter_portrait", "tabloid_landscape"])
def test_chart_title_and_headers_fit_the_export_page(preset: str) -> None:
    result, figure, renderer = _render(preset)
    try:
        assert result.png_bytes
        canvas_renderer = figure.canvas.get_renderer()
        page = figure.bbox
        title = figure.axes[0].title.get_window_extent(canvas_renderer)
        assert title.y1 <= page.y1 + 0.5, f"title clipped off the top on {preset}"
        assert title.x0 >= page.x0 - 0.5 and title.x1 <= page.x1 + 0.5
        headers = [h for h in renderer._header_labels if h.get_text().strip()]
        assert headers
        for header in headers:
            header.update_bbox_position_size(canvas_renderer)
            box = header.get_bbox_patch().get_window_extent(canvas_renderer)
            assert box.y1 <= page.y1 + 0.5, f"{header.get_text()!r} off the page"
            assert box.y1 <= title.y0, f"{header.get_text()!r} header runs into the title"
    finally:
        plt.close(figure)


def test_chart_legend_names_water_series_and_collar_marker() -> None:
    _result, figure, _renderer = _render(
        "letter_landscape", water_levels=_water_levels(), interpolate_water_table=True
    )
    try:
        legend = figure.axes[0].get_legend()
        assert legend is not None
        labels = [text.get_text() for text in legend.get_texts()]
        water = [label for label in labels if label.startswith("GROUNDWATER LEVEL")]
        assert water == [
            "GROUNDWATER LEVEL (MAY 14, 2026)",
            "GROUNDWATER LEVEL (SEPT 22, 2026)",
        ]
        assert labels[-1] == COLLAR_LEGEND_LABEL
        assert {"Clay", "Sand"} <= set(labels)
        handles = dict(zip(labels, legend.legend_handles, strict=True))
        series_handle = handles["GROUNDWATER LEVEL (MAY 14, 2026)"]
        assert series_handle.get_marker() not in (None, "", "None")
        assert series_handle.get_linestyle() not in ("None", "none", "")
        assert handles[COLLAR_LEGEND_LABEL].get_marker() == "v"
    finally:
        plt.close(figure)


def test_chart_legend_without_water_has_no_water_entries() -> None:
    _result, figure, _renderer = _render("letter_landscape")
    try:
        labels = [t.get_text() for t in figure.axes[0].get_legend().get_texts()]
        assert not any(label.startswith("GROUNDWATER") for label in labels)
        assert COLLAR_LEGEND_LABEL in labels
    finally:
        plt.close(figure)
