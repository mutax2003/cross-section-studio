"""Visual-QA round 2 regressions for chemistry value labels.

1. Depth mode (inverted y axis): strip knock-outs were never drawn, and the
   column obstacle boxes were un-normalised (y0 > y1), so overlap tests failed.
2. Groundwater lines struck through values; leaders ran under water labels.
3. Section sheet on letter portrait: the last hole's values crossed the frame.
4. Consulting legend: the chloride key was ellipsised on letter portrait.
5. Two values at one depth: the second slid sideways along the first.
6. Threshold mode legend key: a three-dot band sample, in every mode.
A. Hidden parameters widened the last hole's label room.
B. A second render() removed the first figure's strips.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from matplotlib.text import Text

from models import EnvironmentalReading
from render_profiles import CONSULTING_SECTION_PROFILE, SECTION_SHEET_PROFILE
from renderer import CrossSectionRenderer
from tests.conftest import run_pipeline
from tests.test_chem_threshold_dots_and_last_hole import _HOLES, _readings, _section, _strips


def _capture(monkeypatch) -> list:
    captured: list = []
    original = CrossSectionRenderer.render

    def capture(self, *args, **kwargs):
        figure = original(self, *args, **kwargs)
        captured.append((self, figure))
        return figure

    monkeypatch.setattr(CrossSectionRenderer, "render", capture)
    return captured


def _render(profile, readings, *, parameters=("Chloride",), **update):
    collars, lith = _section()
    projected, polygons, _ = run_pipeline(collars, lith, [(0.0, 0.0), (100.0, 0.0)])
    renderer = CrossSectionRenderer(
        show_legend=False,
        render_profile=profile.model_copy(
            update={"show_parameter_markers": True, "show_parameter_labels": True, **update}
        ),
        environmental_readings=readings,
        environmental_parameters=parameters,
    )
    figure = renderer.render(polygons, projected, collar_depths={h: 30.0 for h in _HOLES})
    return renderer, figure


# --- 1. depth mode ---------------------------------------------------------


@pytest.mark.parametrize("profile", [CONSULTING_SECTION_PROFILE, SECTION_SHEET_PROFILE])
def test_depth_mode_draws_strips_and_normalised_column_boxes(profile) -> None:
    renderer, figure = _render(profile, _readings(6, 20.0), y_axis_mode="depth_below_collar")
    columns = renderer._column_obstacle_boxes(figure)
    assert columns and all(box.y0 < box.y1 and box.x0 < box.x1 for box in columns)
    assert len(_strips(figure)) >= len(_HOLES)


# --- B. strips belong to their figure --------------------------------------


def test_second_render_keeps_first_figures_strips() -> None:
    import matplotlib.pyplot as plt

    collars, lith = _section()
    projected, polygons, _ = run_pipeline(collars, lith, [(0.0, 0.0), (100.0, 0.0)])
    renderer = CrossSectionRenderer(
        show_legend=False,
        render_profile=CONSULTING_SECTION_PROFILE.model_copy(
            update={"show_parameter_markers": True, "show_parameter_labels": True}
        ),
        environmental_readings=_readings(8, 10.0),
        environmental_parameters=("Chloride",),
    )
    first = renderer.render(polygons, projected, collar_depths={h: 30.0 for h in _HOLES})
    count = len(_strips(first))
    assert count
    second = renderer.render(polygons, projected, collar_depths={h: 30.0 for h in _HOLES})
    try:
        assert len(_strips(first)) == count
        assert len(_strips(second)) == count
    finally:
        plt.close(first)
        plt.close(second)


# --- A. hidden parameters do not widen the frame ---------------------------


def test_hidden_parameters_do_not_widen_last_hole_label_room() -> None:
    last = _HOLES[-1]
    base = [
        EnvironmentalReading(
            hole_id=last, parameter="Chloride", value=12.0, depth=2.0, unit="mg/kg"
        )
    ]
    hidden = [
        EnvironmentalReading(
            hole_id=last, parameter=f"Metal{k}", value=1.0, depth=0.2 * k, unit="mg/kg"
        )
        for k in range(1, 41)
    ]
    limits = []
    for readings in (base, base + hidden):
        _renderer, figure = _render(CONSULTING_SECTION_PROFILE, readings)
        limits.append(tuple(figure.axes[0].get_xlim()))
    assert limits[0] == limits[1]
    # Labels off: no room reserved at all.
    renderer, figure = _render(CONSULTING_SECTION_PROFILE, base + hidden)
    renderer.profile = renderer.profile.model_copy(update={"show_parameter_labels": False})
    import pandas as pd

    summary = pd.DataFrame({"hole_id": [last], "x_profile": [100.0]})
    assert renderer._last_hole_label_room(figure.axes[0], summary, 100.0) == 0.0


# --- 2. water lines and leaders --------------------------------------------


def test_segment_box_hits() -> None:
    from renderer_water import _segment_box_hits

    segs = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 20.0, 10.0, 20.0], [5.0, -5.0, 5.0, 30.0]])
    boxes = np.array([[4.0, 4.0, 6.0, 6.0], [20.0, 0.0, 30.0, 10.0]])
    hits = _segment_box_hits(segs, boxes)
    assert hits.tolist() == [[True, False], [False, False], [True, False]]


def test_water_lines_never_strike_through_values(monkeypatch) -> None:
    from batch_export import transect_points_from_collars
    from ingestion import ingest_workbook
    from models import subset_parse_result
    from pipeline import build_cross_section
    from renderer_water import (
        _chem_footprint,
        _figure_renderer,
        _segment_box_hits,
        _water_line_segments,
    )
    from tests.test_multi_section_workbook import build_site_workbook_bytes

    captured = _capture(monkeypatch)
    parse_result, _report = ingest_workbook(BytesIO(build_site_workbook_bytes()))
    for spec in parse_result.section_specs[:2]:
        sub = subset_parse_result(parse_result, spec.hole_ids)
        build_cross_section(
            sub.collars,
            sub.lithologies,
            transect_points_from_collars(parse_result.collars, spec.hole_ids),
            vertical_exaggeration=5.0,
            water_levels=sub.water_levels or None,
            environmental_readings=sub.environmental_readings,
            environmental_parameters=("Chloride",),
            show_parameter_labels=True,
            chemistry_color_mode="threshold",
            chemistry_threshold_green_max=120.0,
            chemistry_threshold_yellow_max=400.0,
            render_layout="consulting_section",
            screen_intervals=sub.screen_intervals,
            export_formats=frozenset({"png"}),
        )
    assert len(captured) == 2
    for renderer, figure in captured:
        figure.draw_without_rendering()
        mpl_renderer = _figure_renderer(figure)
        segs = _water_line_segments(figure.axes)
        assert segs.shape[0]
        chem = [
            ann for kind, ann, _c in renderer._water_labels if kind == "chem" and ann.get_visible()
        ]
        assert chem
        boxes = np.array([tuple(_chem_footprint(ann, mpl_renderer).extents) for ann in chem])
        struck = _segment_box_hits(segs, boxes).any(axis=0)
        assert not struck.any(), [
            ann.get_text() for ann, hit in zip(chem, struck, strict=True) if hit
        ]
        # A shown leader never runs under another label's text.
        others = [
            Text.get_window_extent(ann, mpl_renderer)
            for _kind, ann, _c in renderer._water_labels
            if ann.get_visible()
        ]
        for ann in chem:
            if not ann.arrow_patch.get_visible():
                continue
            box = _chem_footprint(ann, mpl_renderer)
            ax_, ay_ = ann.axes.transData.transform([ann.xy])[0]
            end = (box.x0, 0.5 * (box.y0 + box.y1))
            seg = np.array([[ax_ + 0.15 * (end[0] - ax_), ay_ + 0.15 * (end[1] - ay_), *end]])
            own = Text.get_window_extent(ann, mpl_renderer)
            rects = np.array(
                [
                    o.padded(-1.0).extents
                    for o in others
                    if o.extents.tolist() != own.extents.tolist()
                ]
            )
            assert not _segment_box_hits(seg, rects).any(), ann.get_text()


# --- 3. last hole on letter portrait section sheet --------------------------


def test_section_sheet_letter_portrait_keeps_last_hole_values_inside_frame(monkeypatch) -> None:
    from export_framing import ExportFramingConfig
    from pipeline import build_cross_section
    from renderer_water import _chem_footprint, _figure_renderer

    captured = _capture(monkeypatch)
    collars, lith = _section()
    readings = _readings(3, 6.0)
    build_cross_section(
        collars,
        lith,
        [(0.0, 0.0), (100.0, 0.0)],
        environmental_readings=readings,
        environmental_parameters=["Chloride"],
        show_parameter_labels=True,
        chemistry_color_mode="threshold",
        chemistry_threshold_green_max=100.0,
        chemistry_threshold_yellow_max=300.0,
        render_layout="section_sheet",
        export_framing=ExportFramingConfig(page_preset="letter_portrait"),
        export_formats=frozenset({"png"}),
    )
    renderer, figure = captured[-1]
    figure.draw_without_rendering()
    mpl_renderer = _figure_renderer(figure)
    shown = [
        ann for kind, ann, _c in renderer._water_labels if kind == "chem" and ann.get_visible()
    ]
    assert len(shown) == len(readings)
    for ann in shown:
        frame = ann.axes.get_window_extent(mpl_renderer)
        box = _chem_footprint(ann, mpl_renderer)
        assert box.x1 <= frame.x1 + 0.5, ann.get_text()


# --- 4 + 6. consulting legend key -------------------------------------------


def _legend_texts(figure) -> list[str]:
    return [t.get_text() for ax in figure.axes for t in ax.texts]


@pytest.mark.parametrize("draw_markers", [True, False])
def test_threshold_legend_key_is_three_band_dots(monkeypatch, draw_markers) -> None:
    from matplotlib.colors import to_hex

    from export_framing import ExportFramingConfig
    from pipeline import build_cross_section
    from render_theme import CHEMISTRY_LABEL_GREEN, CHEMISTRY_LABEL_ORANGE, CHEMISTRY_LABEL_RED

    captured = _capture(monkeypatch)
    collars, lith = _section()
    build_cross_section(
        collars,
        lith,
        [(0.0, 0.0), (100.0, 0.0)],
        environmental_readings=_readings(3, 6.0),
        environmental_parameters=["Chloride"],
        show_parameter_labels=True,
        chemistry_color_mode="threshold",
        chemistry_threshold_green_max=100.0,
        chemistry_threshold_yellow_max=300.0,
        render_layout="consulting_section",
        interpretation_mode="interpolated" if draw_markers else "borehole_only",
        export_framing=ExportFramingConfig(page_preset="letter_portrait"),
        export_formats=frozenset({"png"}),
    )
    _renderer, figure = captured[-1]
    dots = [
        c
        for ax in figure.axes
        for c in ax.collections
        if c.get_gid() == "parameter-legend-threshold-dots"
    ]
    assert len(dots) == 1
    colours = [to_hex(c) for c in dots[0].get_facecolors()]
    assert colours == [
        to_hex(CHEMISTRY_LABEL_GREEN),
        to_hex(CHEMISTRY_LABEL_ORANGE),
        to_hex(CHEMISTRY_LABEL_RED),
    ]
    # Letter portrait: the key wraps instead of losing words.
    texts = _legend_texts(figure)
    key = [t for t in texts if t.startswith("CHLORIDE")]
    assert key and "…" not in key[0]
    assert "CONCENTRATION" in key[0] and "(mg/kg)" in key[0]


def test_threshold_legend_dots_without_markers() -> None:
    renderer, figure = _render(
        CONSULTING_SECTION_PROFILE,
        _readings(3, 6.0),
        parameter_draw_markers=False,
        chemistry_color_mode="threshold",
        chemistry_threshold_green_max=100.0,
        chemistry_threshold_yellow_max=300.0,
    )
    assert renderer.parameter_series_legend[0].get("threshold_colors")


def test_fixed_colour_legend_has_no_threshold_dots() -> None:
    renderer, _figure = _render(
        CONSULTING_SECTION_PROFILE, _readings(3, 6.0), chemistry_color_mode="red"
    )
    assert "threshold_colors" not in renderer.parameter_series_legend[0]


# --- 5. same-depth values stack below ---------------------------------------


def test_same_depth_values_stack_directly_below() -> None:
    from renderer_water import _chem_footprint, _figure_renderer

    hole = _HOLES[2]
    readings = [
        EnvironmentalReading(
            hole_id=hole,
            parameter="Chloride",
            value=367.0,
            from_depth=9.75,
            to_depth=10.5,
            unit="mg/kg",
        ),
        EnvironmentalReading(
            hole_id=hole,
            parameter="Chloride",
            value=224.0,
            from_depth=10.0,
            to_depth=10.5,
            unit="mg/kg",
        ),
    ]
    renderer, figure = _render(CONSULTING_SECTION_PROFILE, readings)
    figure.draw_without_rendering()
    mpl_renderer = _figure_renderer(figure)
    labels = {ann.get_text(): ann for kind, ann, _c in renderer._water_labels if kind == "chem"}
    upper = _chem_footprint(labels["367"], mpl_renderer)
    lower = _chem_footprint(labels["224"], mpl_renderer)
    assert labels["224"].get_visible() and labels["367"].get_visible()
    assert lower.y1 <= upper.y0 + 0.5  # below, in depth order
    # Same column: not pushed sideways along the first value's baseline.
    assert abs(lower.x0 - upper.x0) < 0.5 * upper.width
