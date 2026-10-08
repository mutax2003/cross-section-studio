"""Visual-QA round 3 regressions for chemistry value labels.

1. A value read just below a groundwater level was pushed out of its strip
   (the water labels took the slot right of the column first); DRY / NM notes
   sat on the strip.
2. Holes close together on the page: opaque strips hid the geology between
   the columns; values shrank one by one to mixed sizes; the last hole's
   values sat on the right frame line.
3. Long, thin sections: values drifted far sideways from their hole.
4. "plain" / "stroke" drew the same white box as "box"; the legend glyph
   showed a dashed line the plot never draws.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.lines import Line2D

import renderer_water
from models import Collar, EnvironmentalReading, Lithology, WaterLevel
from render_profiles import CHART_PROFILE, CONSULTING_SECTION_PROFILE, SECTION_SHEET_PROFILE
from renderer import CrossSectionRenderer
from tests.conftest import run_pipeline


def _positions(holes, spacing_m) -> list[float]:
    if isinstance(spacing_m, (list, tuple)):
        return [float(x) for x in spacing_m]
    return [i * spacing_m for i in range(len(holes))]


def _section(holes, spacing_m, depth: float = 20.0):
    collars = [
        Collar(hole_id=h, easting=x, northing=0.0, elevation=100.0, total_depth=depth)
        for h, x in zip(holes, _positions(holes, spacing_m), strict=True)
    ]
    lith = [
        Lithology(
            hole_id=h,
            from_depth=2.0 * j,
            to_depth=2.0 * j + 2.0,
            lithology_code=("Clay", "Sand")[j % 2],
        )
        for h in holes
        for j in range(int(depth // 2))
    ]
    return collars, lith


def _render(profile, holes, spacing_m, readings, *, water=None, ve=None, **update):
    collars, lith = _section(holes, spacing_m)
    end = _positions(holes, spacing_m)[-1]
    projected, polygons, _ = run_pipeline(collars, lith, [(0.0, 0.0), (end, 0.0)])
    renderer = CrossSectionRenderer(
        show_legend=False,
        render_profile=profile.model_copy(
            update={"show_parameter_markers": True, "show_parameter_labels": True, **update}
        ),
        environmental_readings=readings,
        environmental_parameters=("Chloride",),
        vertical_exaggeration=ve,
    )
    figure = renderer.render(
        polygons, projected, collar_depths={h: 20.0 for h in holes}, water_levels=water
    )
    return renderer, figure


def _chem(renderer):
    return [a for kind, a, _c in renderer._water_labels if kind == "chem" and a.get_visible()]


def _labels(renderer, kind):
    return [a for k, a, _c in renderer._water_labels if k == kind and a.get_visible()]


def _px(figure):
    renderer = renderer_water._figure_renderer(figure)
    return renderer, renderer.points_to_pixels(1.0)


def _reading(hole, value, depth):
    return EnvironmentalReading(
        hole_id=hole, parameter="Chloride", value=value, depth=depth, unit="mg/kg"
    )


# --- 1. values keep the slot beside their column ---------------------------

_WIDE = [f"W{i}" for i in range(4)]


def _water_case(**update):
    readings = [_reading(h, 870.0 + k, 6.2) for k, h in enumerate(_WIDE)]
    readings += [_reading(h, 50.0, 1.0) for h in _WIDE]
    water = [WaterLevel(hole_id=h, depth=5.6, series_id="MAY") for h in _WIDE] + [
        WaterLevel(hole_id=h, depth=6.0, series_id="SEPT") for h in _WIDE
    ]
    return _render(CONSULTING_SECTION_PROFILE, _WIDE, 60.0, readings, water=water, **update)


@pytest.mark.parametrize("style", ["strip", "plain"])
def test_value_below_water_level_stays_beside_its_reading(style) -> None:
    renderer, figure = _water_case(chemistry_label_style=style)
    try:
        r, px = _px(figure)
        for annotation in _chem(renderer):
            if not annotation.get_text().startswith("87"):
                continue
            box = renderer_water._chem_footprint(annotation, r)
            anchor_x, anchor_y = renderer_water._chem_anchor(annotation)
            # Directly right of its column at (about) its reading's depth:
            # not shoved sideways past the water labels or metres down.
            assert box.x0 - anchor_x < 16.0 * px
            assert abs(0.5 * (box.y0 + box.y1) - anchor_y) < 1.5 * box.height
            assert not annotation.arrow_patch.get_visible()
        # No water number overlaps a value.
        chem_boxes = [renderer_water._chem_footprint(a, r) for a in _chem(renderer)]
        for annotation in _labels(renderer, "rl"):
            box = renderer_water._text_box(annotation, r)
            assert all(renderer_water._overlap_area(box, other) == 0.0 for other in chem_boxes)
    finally:
        plt.close(figure)


def test_value_box_is_not_crossed_by_a_water_line() -> None:
    renderer, figure = _water_case()
    try:
        r, _px_per_pt = _px(figure)
        segs = renderer_water._water_line_segments({figure.axes[0]})
        assert segs.shape[0]
        boxes = np.asarray(
            [tuple(renderer_water._chem_footprint(a, r).extents) for a in _chem(renderer)]
        )
        assert not renderer_water._segment_box_hits(segs, boxes).any()
    finally:
        plt.close(figure)


def test_water_numbers_and_dry_notes_keep_off_the_strips() -> None:
    readings = [
        _reading(h, 900.0 + k, d) for h in _WIDE for k, d in enumerate((0.8, 2.0, 3.2, 6.2))
    ]
    water = [WaterLevel(hole_id=h, depth=5.6, series_id="MAY") for h in _WIDE[::2]]
    water += [WaterLevel(hole_id=h, depth=0.0, series_id="MAY", status="dry") for h in _WIDE[1::2]]
    renderer, figure = _render(CONSULTING_SECTION_PROFILE, _WIDE, 60.0, readings, water=water)
    try:
        r, _px_per_pt = _px(figure)
        strips = [s.get_window_extent(r) for s in figure._chem_strips]
        assert strips
        for kind in ("rl", "nm"):
            for annotation in _labels(renderer, kind):
                box = renderer_water._text_box(annotation, r)
                assert all(renderer_water._overlap_area(box, s) == 0.0 for s in strips), kind
    finally:
        plt.close(figure)


# --- 2. holes close together -------------------------------------------------

_TIGHT = [f"T{i}" for i in range(7)]


def _tight_case(spacing=(0.0, 3.5, 7.0, 10.5, 14.0, 17.5, 60.0)):
    readings = [
        _reading(h, value, depth)
        for h in _TIGHT
        for value, depth in ((1110.0, 1.0), (1320.0, 2.0), (12.5, 6.0), (1850.0, 9.0))
    ]
    return _render(CONSULTING_SECTION_PROFILE, _TIGHT, spacing, readings)


def test_close_holes_get_boxed_values_not_strips_over_the_geology() -> None:
    renderer, figure = _tight_case()
    try:
        r, _px_per_pt = _px(figure)
        columns = sorted(renderer._column_obstacle_boxes(figure), key=lambda b: b.x0)
        gaps = {round(a.x1, 1): b.x0 - a.x1 for a, b in zip(columns, columns[1:], strict=False)}
        for strip in figure._chem_strips:
            box = strip.get_window_extent(r)
            own = max((c for c in columns if c.x1 <= box.x0 + 1.0), key=lambda c: c.x1)
            gap = gaps.get(round(own.x1, 1))
            if gap is not None:
                assert box.width <= renderer_water._CHEM_STRIP_MAX_GAP_SHARE * gap + 1.0
        boxed = [a for a in _chem(renderer) if getattr(a, "_chem_box_fallback", False)]
        assert boxed and all(a.get_bbox_patch() is not None for a in boxed)
    finally:
        plt.close(figure)


def test_values_share_one_font_size_per_figure() -> None:
    renderer, figure = _tight_case()
    try:
        sizes = {round(a.get_fontsize(), 3) for a in _chem(renderer)}
        assert len(sizes) == 1, sizes
    finally:
        plt.close(figure)


def test_mixed_value_sizes_cap_every_hole_at_the_smallest() -> None:
    fig, ax = plt.subplots()
    try:
        labels = []
        for x, size in ((1.0, 7.25), (1.0, 5.51), (2.0, 7.25)):
            annotation = ax.annotate("1", xy=(x, 0.5), fontsize=size)
            annotation._chem_base_fontsize = 7.25
            labels.append(("chem", annotation, "k"))
        caps = renderer_water._chem_mixed_size_holes(labels)
        assert len(caps) == 2
        assert all(cap == pytest.approx(5.51 / 7.25) for cap in caps.values())
        # One size throughout: nothing to settle.
        labels[1][1].set_fontsize(7.25)
        assert renderer_water._chem_mixed_size_holes(labels) == {}
    finally:
        plt.close(fig)


def test_last_hole_values_stay_inside_the_consulting_frame() -> None:
    renderer, figure = _tight_case()
    try:
        r, _px_per_pt = _px(figure)
        frame = figure.axes[0].get_window_extent(r)
        for annotation in _chem(renderer):
            assert renderer_water._chem_footprint(annotation, r).x1 < frame.x1
    finally:
        plt.close(figure)


def test_right_limit_keeps_inside_frame_and_half_the_gap() -> None:
    from matplotlib.transforms import Bbox

    own = Bbox.from_extents(100, 0, 110, 500)
    nxt = Bbox.from_extents(410, 0, 420, 500)
    frame = Bbox.from_extents(0, 0, 600, 500)
    limit = renderer_water._chem_right_limit(
        own, [own, nxt], frame, consulting=True, clearance=2.0, need=20.0
    )
    assert limit == pytest.approx(110 + 0.5 * 300)
    # A value wider than half the gap may still use what it needs.
    limit = renderer_water._chem_right_limit(
        own, [own, nxt], frame, consulting=True, clearance=2.0, need=150.0
    )
    assert 110 + 150 < limit <= 410
    # Last hole on a consulting sheet: inside the frame line, not on it.
    limit = renderer_water._chem_right_limit(
        own, [own], frame, consulting=True, clearance=2.0, need=20.0
    )
    assert limit < frame.x1


# --- 3. long, thin sections -------------------------------------------------


def test_values_on_a_long_thin_chart_stay_near_their_hole() -> None:
    holes = [f"L{i}" for i in range(6)]
    readings = [
        _reading(h, value, depth)
        for h in holes[2:5]
        for value, depth in ((45.0, 1.0), (380.0, 3.5), (1250.0, 6.0))
    ]
    renderer, figure = _render(
        CHART_PROFILE, holes, 62.0, readings, ve=1.0, chemistry_label_style="strip"
    )
    try:
        r, _px_per_pt = _px(figure)
        columns = sorted(renderer._column_obstacle_boxes(figure), key=lambda b: b.x0)
        for annotation in _chem(renderer):
            own = renderer_water._own_column(annotation, columns)
            nxt = min((c for c in columns if c.x0 > own.x1), key=lambda c: c.x0)
            box = renderer_water._chem_footprint(annotation, r)
            gap = nxt.x0 - own.x1
            assert box.x1 <= own.x1 + max(0.5 * gap, box.width / 0.68 + 8.0) + 1.0
    finally:
        plt.close(figure)


# --- 4. label styles and legend glyph ----------------------------------------


@pytest.mark.parametrize(
    ("style", "boxed", "halo"),
    [
        ("plain", False, False),
        ("stroke", False, True),
        ("box", True, False),
        ("strip", False, False),
    ],
)
def test_label_styles_draw_distinct_backgrounds(style, boxed, halo) -> None:
    readings = [_reading(h, 120.0, 3.0) for h in _WIDE]
    renderer, figure = _render(
        CONSULTING_SECTION_PROFILE, _WIDE, 60.0, readings, chemistry_label_style=style
    )
    try:
        for annotation in _chem(renderer):
            assert (annotation.get_bbox_patch() is not None) is boxed
            assert (getattr(annotation, "_halo", None) is not None) is halo
    finally:
        plt.close(figure)


@pytest.mark.parametrize("mode", ["black", "red", "threshold"])
@pytest.mark.parametrize("segments", [False, True])
def test_legend_glyph_matches_the_drawn_markers(mode, segments) -> None:
    readings = [_reading(h, 120.0 * (k + 1), 3.0) for k, h in enumerate(_WIDE)]
    renderer, figure = _render(
        CONSULTING_SECTION_PROFILE,
        _WIDE,
        60.0,
        readings,
        chemistry_color_mode=mode,
        parameter_interpolate_segments=segments,
    )
    try:
        (entry,) = renderer.parameter_series_legend
        assert entry["linestyle"] == ("--" if segments else "none")
        drawn = {
            tuple(np.round(c, 3))
            for coll in figure.axes[0].collections
            if coll.get_zorder() == 8
            for c in coll.get_facecolors()
        }
        if mode != "threshold":
            from matplotlib.colors import to_rgba

            assert tuple(np.round(to_rgba(entry["color"]), 3)) in drawn
    finally:
        plt.close(figure)


def test_consulting_legend_draws_markers_only_without_a_fence() -> None:
    readings = [_reading(h, 120.0, 3.0) for h in _WIDE]
    collars, lith = _section(_WIDE, 60.0)
    projected, polygons, _ = run_pipeline(collars, lith, [(0.0, 0.0), (180.0, 0.0)])
    renderer = CrossSectionRenderer(
        show_legend=True,
        render_profile=CONSULTING_SECTION_PROFILE.model_copy(
            update={
                "show_parameter_markers": True,
                "show_parameter_labels": True,
                "chemistry_color_mode": "black",
            }
        ),
        environmental_readings=readings,
        environmental_parameters=("Chloride",),
    )
    figure = renderer.render(polygons, projected, collar_depths={h: 20.0 for h in _WIDE})
    try:
        glyphs = [
            line
            for ax in figure.axes[1:]
            for line in ax.get_lines()
            if isinstance(line, Line2D) and line.get_marker() not in (None, "None", "", " ")
        ]
        assert glyphs
        assert all(line.get_linestyle() in ("None", "none", "") for line in glyphs)
    finally:
        plt.close(figure)


def test_section_sheet_strip_defaults_unchanged_for_wide_holes() -> None:
    readings = [_reading(h, 120.0, d) for h in _WIDE for d in (2.0, 5.0)]
    renderer, figure = _render(SECTION_SHEET_PROFILE, _WIDE, 60.0, readings)
    try:
        assert figure._chem_strips
        assert not any(getattr(a, "_chem_box_fallback", False) for a in _chem(renderer))
    finally:
        plt.close(figure)
