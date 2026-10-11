"""Legend keys: complete, rebuilt from their proxies, and drawn like the figure.

* The outside section-sheet / chart legend is rebuilt from the proxies it was
  made from: a second rebuild used to turn the three-dot threshold key into
  one green dot and drop the NM sample text.
* The consulting title-block legend lists every unit and every symbol it
  has room for (two or three columns, a smaller size) instead of rolling
  units up while half the box is empty; symbol keys are never dropped and
  labels stay inside the box.
* Swatch hatch marks are the section's marks at the section's marks per
  inch, whole and centred, on every page.
"""

from __future__ import annotations

import warnings
from io import BytesIO

import matplotlib
import matplotlib.hatch as mhatch
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.text import Text

import hatch_patterns
from export_framing import ExportFramingConfig
from ingestion import ingest_workbook
from models import ConsultingTitleBlock, subset_parse_result
from pipeline import compute_section_geometry, render_cross_section_from_geometry
from render_profiles import SECTION_SHEET_PROFILE
from render_theme import PINCH_OUT_ALPHA
from renderer import CrossSectionRenderer
from renderer_common import (
    LEGEND_KEY_HANDLER_MAP,
    MIN_LEGEND_SWATCH_HEIGHT_IN,
    SampleTextKey,
    ThresholdDotsKey,
    legend_source_keys,
)
from tests.test_multi_section_workbook import build_site_workbook_bytes

matplotlib.use("Agg")

_SYMBOL_LABELS = (
    "SCREENED INTERVAL",
    "GROUNDWATER LEVEL",
    "CHLORIDE CONCENTRATION",
    "NOT MEASURED",
    "INFERRED PINCH-OUT",
)


# --- outside legend rebuild ----------------------------------------------------


def test_outside_legend_rebuild_keeps_threshold_and_sample_text_keys() -> None:
    renderer = CrossSectionRenderer(show_legend=True, render_profile=SECTION_SHEET_PROFILE)
    renderer._has_pinch_out = True
    fig = Figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    renderer._draw_legend(ax, {}, ["Clay", "Sand"], [])
    colors = ("#047857", "#B45309", "#B91C1C")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # "Legend does not support handles for Text instances"
        renderer._rebuild_outside_legend(
            ax,
            [(ThresholdDotsKey(colors), "CHLORIDE"), (SampleTextKey("NM", "#64748B"), "NM — not measured")],
        )
        # The water key joining later rebuilds the legend a second time.
        renderer._rebuild_outside_legend(ax, [(Line2D([], [], marker="v"), "GROUNDWATER LEVEL (MAY)")])
    legend = ax.get_legend()
    labels = [t.get_text() for t in legend.get_texts()]
    assert labels == [
        "Clay",
        "Sand",
        "Inferred pinch-out",
        "CHLORIDE",
        "NM — not measured",
        "GROUNDWATER LEVEL (MAY)",
    ]
    fig.draw_without_rendering()
    dots = [c for c in legend.findobj(Line2D) if c.get_gid() == "parameter-legend-threshold-dots"]
    assert len(dots) == 3
    assert {d.get_markerfacecolor() for d in dots} == set(colors)
    assert any(isinstance(a, Text) and a.get_text() == "NM" for a in legend.findobj(Text))
    # Source proxies, not drawn artists, are kept for the next rebuild.
    assert isinstance(legend_source_keys(legend)[3][0], ThresholdDotsKey)


def test_pinch_out_key_is_a_faded_dashed_wedge() -> None:
    renderer = CrossSectionRenderer(show_legend=True, render_profile=SECTION_SHEET_PROFILE)
    renderer._has_pinch_out = True
    fig = Figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    renderer._draw_legend(ax, {}, ["Clay"], [])
    fig.draw_without_rendering()
    wedges = [p for p in ax.get_legend().findobj(Polygon) if p.get_gid() == "pinch-out-legend-key"]
    assert len(wedges) == 1
    assert wedges[0].get_alpha() == pytest.approx(PINCH_OUT_ALPHA)
    assert wedges[0].get_linestyle() == "--"
    assert len(wedges[0].get_xy()) >= 3  # a wedge, not the old box


def test_handler_map_draws_lithology_swatch_marks() -> None:
    from matplotlib.patches import Patch

    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    legend = ax.legend(
        handles=[Patch(facecolor="#FDE68A", edgecolor="k", hatch=".", label="Sand")],
        handler_map=LEGEND_KEY_HANDLER_MAP,
        handleheight=1.0,
        fontsize=8,
    )
    fig.draw_without_rendering()
    marks = [p for p in legend.findobj() if getattr(p, "get_gid", lambda: None)() == "legend-swatch-hatch"]
    assert marks, "sand swatch has no stipple"


# --- swatch marks match the figure --------------------------------------------


def _cell_marks_per_square_inch(hatch: str) -> int:
    cell = mhatch.get_path(hatch, 6)
    groups, _lines = hatch_patterns._mark_groups(cell)
    centres = np.array([(v.min(axis=0) + v.max(axis=0)) / 2.0 for v, _c in groups])
    inside = np.all((centres >= 0.0) & (centres < 1.0 - 1e-9), axis=1)
    return int(inside.sum())


@pytest.mark.parametrize("hatch", [".", "+", "/", ".."])
def test_swatch_marks_per_inch_equal_the_figure(hatch: str) -> None:
    side = 2.0
    marks, _lines = hatch_patterns.swatch_marks_path(hatch, side, side, inset_in=0.0)
    centres = hatch_patterns.isolated_mark_centres(marks)
    per_sq_in = len(centres) / side**2
    expected = _cell_marks_per_square_inch(hatch)
    # Whole marks only: the border loses at most a partial row on each side.
    assert expected * 0.8 <= per_sq_in <= expected * 1.01
    # Nearest-neighbour spacing is the figure's pitch.
    cell = mhatch.get_path(hatch, 6)
    cell_groups, _ = hatch_patterns._mark_groups(cell)
    cell_centres = np.array([(v.min(axis=0) + v.max(axis=0)) / 2.0 for v, _c in cell_groups])

    def nn(points):
        d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        return float(np.median(d.min(axis=1)))

    assert nn(centres) == pytest.approx(nn(cell_centres), rel=0.02)


@pytest.mark.parametrize("hatch", [".", "+", "/", "./"])
def test_small_swatch_marks_are_whole_and_centred(hatch: str) -> None:
    width, height = 0.24, MIN_LEGEND_SWATCH_HEIGHT_IN
    marks, _lines = hatch_patterns.swatch_marks_path(hatch, width, height)
    assert marks is not None
    verts = marks.vertices
    assert verts[:, 0].min() >= 0.0 and verts[:, 0].max() <= width
    assert verts[:, 1].min() >= 0.0 and verts[:, 1].max() <= height
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    assert (lo + hi) / 2.0 == pytest.approx((width / 2.0, height / 2.0), abs=1e-6)
    if hatch == ".":
        # Two staggered stipple rows, like the section, not one sparse row.
        rows = np.unique(np.round(hatch_patterns.isolated_mark_centres(marks)[:, 1], 4))
        assert len(rows) >= 2


# --- consulting title-block legend --------------------------------------------


def _consulting_section(page: str, section: str = "A-A'", **kwargs):
    pr, _rep = ingest_workbook(BytesIO(build_site_workbook_bytes(site_unit_order=True)))
    hole_ids = next(s.hole_ids for s in pr.section_specs if s.label == section)
    from batch_export import transect_points_from_collars

    sub = subset_parse_result(pr, hole_ids)
    params = sorted({e.parameter for e in sub.environmental_readings})
    points = transect_points_from_collars(pr.collars, hole_ids)
    geometry = compute_section_geometry(sub.collars, sub.lithologies, points, allow_pinch_outs=True)
    result = render_cross_section_from_geometry(
        geometry,
        points,
        vertical_exaggeration=None,
        show_hatches=True,
        title="QA",
        water_levels=sub.water_levels or None,
        environmental_readings=sub.environmental_readings,
        environmental_parameters=tuple(params[:1]),
        show_parameter_labels=True,
        chemistry_color_mode=kwargs.get("chem", "threshold"),
        chemistry_threshold_green_max=120.0,
        chemistry_threshold_yellow_max=400.0,
        render_layout="consulting_section",
        consulting_title_block=ConsultingTitleBlock(
            section_label=section, project_number="QA-1", prepared_for="CLIENT", prepared_by="QA",
            figure_number="1", date="2026-10-08",
        ),
        screen_intervals=sub.screen_intervals,
        export_framing=ExportFramingConfig(page_preset=page, export_dpi=72),
        export_formats=frozenset({"png"}),
        close_figure=False,
    )
    codes = sorted({lith.lithology_code for lith in sub.lithologies})
    return result, codes


def _legend_axes(figure):
    return next(ax for ax in figure.axes if any(t.get_text() == "LEGEND" for t in ax.texts))


@pytest.mark.parametrize("page", ["letter_portrait", "letter_landscape", "tabloid_landscape"])
def test_consulting_legend_lists_every_unit_and_symbol_inside_its_box(page: str) -> None:
    result, codes = _consulting_section(page)
    figure = result.retained_export_bundle["figure"]
    figure.draw_without_rendering()
    ax = _legend_axes(figure)
    renderer = figure._get_renderer()
    texts = [t for t in ax.texts if t.get_text().strip() and t.get_visible()]
    labels = " ".join(t.get_text().replace("\n", " ") for t in texts)
    assert "MORE UNITS" not in labels
    for code in codes:
        assert code.upper() in labels, (page, code)
    for symbol in _SYMBOL_LABELS:
        assert symbol in labels, (page, symbol)
    boxes = ax.__dict__["_css_title_block_boxes"]
    legend_box = list(boxes.values())[-1][0]
    left, bottom = ax.transAxes.transform((legend_box[0], legend_box[1]))
    right, top = ax.transAxes.transform((legend_box[0] + legend_box[2], legend_box[1] + legend_box[3]))
    legend_texts = []
    for text in texts:
        extent = Text.get_window_extent(text, renderer)
        if not (left <= (extent.x0 + extent.x1) / 2.0 <= right):
            continue  # title / prepared panels
        legend_texts.append(text)
        assert extent.x0 >= left and extent.x1 <= right, (page, text.get_text())
        assert extent.y0 >= bottom + 1 and extent.y1 <= top - 1, (page, text.get_text())
    sizes = {
        round(t.get_fontsize(), 2)
        for t in legend_texts
        if t.get_text() != "LEGEND" and t.get_gid() != "water-status-legend-sample"
    }
    assert min(sizes) >= 6.0 - 1e-6


def test_consulting_single_colour_chemistry_key_is_one_marker() -> None:
    result, _codes = _consulting_section("letter_landscape", chem="red")
    figure = result.retained_export_bundle["figure"]
    ax = _legend_axes(figure)
    keys = [
        line
        for line in ax.lines
        if line.get_marker() not in (None, "", "None", "none") and line.get_linestyle() in ("None", "none", "")
    ]
    chemistry = [line for line in keys if line.get_marker() != "v"]
    assert chemistry and all(len(line.get_xdata()) == 1 for line in chemistry)
    assert not any(isinstance(c, PathCollection) and c.get_gid() == "parameter-legend-threshold-dots" for c in ax.collections)
