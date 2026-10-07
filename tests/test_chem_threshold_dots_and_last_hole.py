"""Threshold-coloured reading dots and a crowded last hole that keeps every value.

User reports: in threshold mode the dots on the stick stayed the series red
while only the value text took green / orange / red; and a long list of
readings on the right-most hole lost values (dropped as a last resort).
"""

from __future__ import annotations

import pytest
from matplotlib.collections import PathCollection
from matplotlib.colors import to_hex

from models import Collar, EnvironmentalReading, Lithology
from render_profiles import CONSULTING_SECTION_PROFILE, SECTION_SHEET_PROFILE
from render_theme import (
    CHEMISTRY_FIXED_COLORS,
    CHEMISTRY_LABEL_GREEN,
    CHEMISTRY_LABEL_ORANGE,
    CHEMISTRY_LABEL_RED,
    parameter_series_color,
)
from renderer import CrossSectionRenderer
from tests.conftest import run_pipeline
from tests.test_chem_labels_right_of_column import _audit, _order_inversions

_HOLES = [f"B{i}" for i in range(6)]


def _readings(last_n: int, last_span_m: float, *, explicit_black_at: int | None = None):
    readings = []
    for i, hole in enumerate(_HOLES):
        n = last_n if i == len(_HOLES) - 1 else 5
        for k in range(n):
            readings.append(
                EnvironmentalReading(
                    hole_id=hole,
                    parameter="Chloride",
                    value=5.0 + 97.0 * k,
                    depth=0.5 + last_span_m * k / n,
                    unit="mg/kg",
                    label_color="black" if (i == len(_HOLES) - 1 and k == explicit_black_at) else None,
                )
            )
    return readings


def _section():
    collars = [
        Collar(hole_id=h, easting=i * 20.0, northing=0.0, elevation=100.0, total_depth=30.0)
        for i, h in enumerate(_HOLES)
    ]
    lith = [
        Lithology(hole_id=h, from_depth=3.0 * j, to_depth=3.0 * j + 3.0, lithology_code=("Clay", "Sand")[j % 2])
        for h in _HOLES
        for j in range(10)
    ]
    return collars, lith


def _marker_colours(figure) -> list[str]:
    colours: list[str] = []
    for ax in figure.axes:
        for coll in ax.collections:
            if isinstance(coll, PathCollection) and coll.get_zorder() == 8:
                colours += [to_hex(c) for c in coll.get_facecolors()]
    return colours


def _render_threshold(profile, readings, **update):
    collars, lith = _section()
    projected, polygons, _ = run_pipeline(collars, lith, [(0.0, 0.0), (100.0, 0.0)])
    renderer = CrossSectionRenderer(
        show_legend=False,
        render_profile=profile.model_copy(
            update={
                "show_parameter_markers": True,
                "show_parameter_labels": True,
                "parameter_draw_markers": True,
                "chemistry_color_mode": "threshold",
                "chemistry_threshold_green_max": 500.0,
                "chemistry_threshold_yellow_max": 1200.0,
                **update,
            }
        ),
        environmental_readings=readings,
        environmental_parameters=("Chloride",),
    )
    figure = renderer.render(polygons, projected, collar_depths={h: 30.0 for h in _HOLES})
    return renderer, figure


@pytest.mark.parametrize("profile", [CONSULTING_SECTION_PROFILE, SECTION_SHEET_PROFILE])
def test_threshold_mode_colours_each_dot_like_its_label(profile) -> None:
    readings = _readings(15, 28.0, explicit_black_at=3)
    renderer, figure = _render_threshold(profile, readings)
    dots = _marker_colours(figure)
    labels = [to_hex(ann.get_color()) for kind, ann, _c in renderer._water_labels if kind == "chem"]
    assert len(dots) == len(readings) == len(labels)
    # Dots and labels carry the same multiset of colours: G/O/R bands plus
    # the workbook's black, never the series colour.
    assert sorted(dots) == sorted(labels)
    assert to_hex(parameter_series_color("Chloride")) not in dots
    assert {to_hex(CHEMISTRY_LABEL_GREEN), to_hex(CHEMISTRY_LABEL_ORANGE), to_hex(CHEMISTRY_LABEL_RED)} <= set(dots)
    assert dots.count(to_hex(CHEMISTRY_FIXED_COLORS["black"])) == 1


@pytest.mark.parametrize("mode", ["black", "red"])
def test_fixed_modes_keep_series_coloured_dots(mode) -> None:
    renderer, figure = _render_threshold(SECTION_SHEET_PROFILE, _readings(5, 20.0), chemistry_color_mode=mode)
    dots = set(_marker_colours(figure))
    assert dots == {to_hex(parameter_series_color("Chloride"))}


@pytest.mark.parametrize("layout", ["consulting_section", "section_sheet"])
@pytest.mark.parametrize(("last_n", "span_m"), [(25, 10.0), (35, 20.0)])
def test_crowded_last_hole_keeps_every_value(monkeypatch, layout, last_n, span_m) -> None:
    # 6 holes on letter portrait; the right-most hole has a long, dense list
    # (readings 0.4-0.6 m apart). Every value prints, right of its column,
    # off every column and label, in depth order.
    from export_framing import ExportFramingConfig
    from pipeline import build_cross_section

    captured = {}
    original = CrossSectionRenderer.render

    def capture(self, *args, **kwargs):
        captured["renderer"] = self
        captured["figure"] = original(self, *args, **kwargs)
        return captured["figure"]

    monkeypatch.setattr(CrossSectionRenderer, "render", capture)
    collars, lith = _section()
    readings = _readings(last_n, span_m, explicit_black_at=3)
    build_cross_section(
        collars,
        lith,
        [(0.0, 0.0), (100.0, 0.0)],
        environmental_readings=readings,
        environmental_parameters=["Chloride"],
        show_parameter_labels=True,
        chemistry_color_mode="threshold",
        chemistry_threshold_green_max=500.0,
        chemistry_threshold_yellow_max=1200.0,
        render_layout=layout,
        export_framing=ExportFramingConfig(page_preset="letter_portrait"),
        export_formats=frozenset({"png"}),
    )
    renderer, figure = captured["renderer"], captured["figure"]
    total, dropped, left_of_own, over_column, overlaps = _audit(renderer, figure)
    assert total == len(readings)
    assert (dropped, left_of_own, over_column, overlaps) == (0, 0, 0, 0)
    assert _order_inversions(renderer, figure) == []


# --- "strip" readability style (default): clean background beside the column ---


def _strips(figure):
    return [p for ax in figure.axes for p in ax.patches if p.get_gid() == "chemistry-label-strip"]


def test_strip_is_the_default_label_style_and_others_remain() -> None:
    from typing import get_args

    from render_profiles import ChemistryLabelStyle, CrossSectionRenderProfile

    assert CrossSectionRenderProfile().chemistry_label_style == "strip"
    assert CONSULTING_SECTION_PROFILE.chemistry_label_style == "strip"
    assert set(get_args(ChemistryLabelStyle)) == {"strip", "plain", "box", "dot", "stroke"}


@pytest.mark.parametrize("profile", [CONSULTING_SECTION_PROFILE, SECTION_SHEET_PROFILE])
def test_strip_knocks_out_fills_beside_each_labelled_column(profile) -> None:
    from matplotlib.transforms import Bbox

    from models import WaterLevel
    from renderer_water import _chem_footprint, _figure_renderer, _overlap_area

    collars, lith = _section()
    projected, polygons, _ = run_pipeline(collars, lith, [(0.0, 0.0), (100.0, 0.0)])
    renderer = CrossSectionRenderer(
        show_legend=False,
        render_profile=profile.model_copy(update={"show_parameter_markers": True, "show_parameter_labels": True}),
        environmental_readings=_readings(12, 20.0),
        environmental_parameters=("Chloride",),
    )
    water = [WaterLevel(hole_id=h, depth=3.0 + 0.3 * i) for i, h in enumerate(_HOLES)]
    figure = renderer.render(polygons, projected, collar_depths={h: 30.0 for h in _HOLES}, water_levels=water)
    figure.draw_without_rendering()
    mpl_renderer = _figure_renderer(figure)
    strips = _strips(figure)
    assert len(strips) >= len(_HOLES)
    columns = renderer._column_obstacle_boxes(figure)
    strip_boxes = [s.get_window_extent(mpl_renderer) for s in strips]
    for box in strip_boxes:
        # Beside a column, never over any column (1 px float tolerance).
        assert not any(_overlap_area(box.padded(-1.0), col) > 0 for col in columns)
    ax = strips[0].axes
    frame = ax.get_window_extent(mpl_renderer)
    # Water-level lines stay drawn above the strip.
    water_lines = [line for line in ax.lines if line.get_zorder() >= 6]
    assert water_lines
    for strip in strips:
        # Above lithology fills and contacts, below track fills, water, markers, values.
        assert 3.0 < strip.get_zorder() < 5.0
    for kind, ann, _c in renderer._water_labels:
        if kind != "chem" or not ann.get_visible():
            continue
        assert ann.get_bbox_patch() is None  # plain coloured text, no box
        box = _chem_footprint(ann, mpl_renderer)
        inside = Bbox.intersection(box, frame)
        if inside is None or inside.width < 1 or inside.height < 1:
            continue
        covered = sum(_overlap_area(inside, s) for s in strip_boxes)
        assert covered >= 0.95 * inside.width * inside.height, ann.get_text()


def test_water_numbers_use_two_decimals_with_trailing_zeros_stripped() -> None:
    from renderer_water import _fmt_water_number

    assert [_fmt_water_number(v) for v in (745.29, 745.2904, 745.3, 745.0, 2.5, -0.001)] == [
        "745.29",
        "745.29",
        "745.3",
        "745",
        "2.5",
        "0",
    ]


def test_dry_and_nm_on_one_hole_merge_into_one_note() -> None:
    import matplotlib.pyplot as plt

    renderer = CrossSectionRenderer(show_legend=False)
    fig, ax = plt.subplots()
    try:
        renderer._water_status_notes = {}
        renderer._water_status_note(ax, "BH-12", (0.0, 0.0), "NM")
        renderer._water_status_note(ax, "BH-12", (0.0, 0.0), "DRY")
        renderer._water_status_note(ax, "BH-12", (0.0, 0.0), "NM")
        renderer._water_status_note(ax, "BH-13", (5.0, 0.0), "NM")
        notes = [ann.get_text() for kind, ann, _c in renderer._water_labels if kind == "nm"]
        assert notes == ["DRY / NM", "NM"]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("layout", ["consulting_section", "section_sheet"])
def test_multi_section_site_labels_stay_attributable(monkeypatch, layout) -> None:
    """QA on the 14-hole multi-section workbook: displaced chloride values need
    a leader, RLs print <= 2 decimals, DRY + NM merge into one note."""
    import re
    from io import BytesIO

    from batch_export import build_multi_transect_exports, clear_batch_geometry_memo
    from ingestion import ingest_workbook
    from renderer_water import _chem_footprint, _figure_renderer
    from tests.test_multi_section_workbook import (
        _base_request,
        _batch_specs,
        build_site_workbook_bytes,
    )

    captured: list = []
    original = CrossSectionRenderer.render

    def capture(self, *args, **kwargs):
        figure = original(self, *args, **kwargs)
        captured.append((self, figure))
        return figure

    monkeypatch.setattr(CrossSectionRenderer, "render", capture)
    parse_result, _report = ingest_workbook(BytesIO(build_site_workbook_bytes()))
    clear_batch_geometry_memo()
    build_multi_transect_exports(
        parse_result, _base_request(layout), _batch_specs(parse_result), export_formats=frozenset({"png"})
    )
    assert len(captured) == 3
    for renderer, figure in captured:
        figure.draw_without_rendering()
        mpl_renderer = _figure_renderer(figure)
        px_per_pt = mpl_renderer.points_to_pixels(1.0)
        for kind, ann, _c in renderer._water_labels:
            if not ann.get_visible():
                continue
            text = ann.get_text()
            if kind == "rl":
                assert not re.search(r"\d\.\d{3}", text), text
            if kind == "nm":
                assert text in {"NM", "DRY", "DRY / NM"}, text
            if kind != "chem":
                continue
            base = ann._water_base_xyann
            shift = max(abs(ann.xyann[0] - base[0]), abs(ann.xyann[1] - base[1]))
            height_pt = _chem_footprint(ann, mpl_renderer).height / px_per_pt
            if shift > height_pt:
                assert ann.arrow_patch.get_visible(), text
        nm_by_anchor: dict[tuple[float, float], int] = {}
        for kind, ann, _c in renderer._water_labels:
            if kind == "nm":
                key = (round(ann.xy[0], 3), round(ann.xy[1], 3))
                nm_by_anchor[key] = nm_by_anchor.get(key, 0) + 1
        assert all(count == 1 for count in nm_by_anchor.values())
    # C-C carries the dry well BH-12: one merged note there.
    texts = [ann.get_text() for kind, ann, _c in captured[2][0]._water_labels if kind == "nm"]
    assert "DRY" in " ".join(texts)


@pytest.mark.parametrize("style", ["plain", "box", "dot", "stroke"])
def test_other_styles_draw_no_strip(style) -> None:
    _renderer, figure = _render_threshold(SECTION_SHEET_PROFILE, _readings(5, 20.0), chemistry_label_style=style)
    assert _strips(figure) == []
