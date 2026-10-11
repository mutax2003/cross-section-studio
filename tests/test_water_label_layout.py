"""Groundwater markers / labels and chemistry strips (visual QA round 4).

* Markers draw above the borehole columns (they were hidden behind them).
* RL / NM / i= labels keep off every water and ground line, the columns,
  the markers and each other; RL numbers sit beside their own marker (or
  carry a leader), gradients sit on their own segment; leaders cross no
  label and no other leader; label boxes are opaque.
* Crowded bands drop i= labels before RL numbers, with a section note.
* One chemistry label style per figure; strips only as tall as their values.
* Overlapping markers (same-depth chemistry dots, a GW marker on a dot,
  two series at one head) are nudged apart.
* Values left off at a small scale are counted on the figure itself.
"""

from __future__ import annotations

import itertools
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.text import Text

from batch_export import transect_points_from_collars
from export_framing import ExportFramingConfig
from ingestion import ingest_workbook
from models import ConsultingTitleBlock, subset_parse_result
from pipeline import compute_section_geometry, render_cross_section_from_geometry
from renderer_common import THRESHOLD_KEY_GID
from renderer_water import (
    WATER_MARKER_ZORDER,
    _figure_renderer,
    _fmt_water_number,
    _ground_line_segments,
    _segment_box_hits,
    _segments_cross,
    _separate_markers,
    _water_drop_notes,
    _water_line_segments,
    register_marker_set,
)
from tests.test_multi_section_workbook import build_site_workbook_bytes


@pytest.fixture(scope="module")
def site():
    parse_result, _report = ingest_workbook(BytesIO(build_site_workbook_bytes()))
    return parse_result


def _render(site, *, layout="consulting_section", page="letter_landscape", ve=None, chem="threshold", style="strip"):
    holes = site.section_specs[0].hole_ids
    sub = subset_parse_result(site, holes)
    points = transect_points_from_collars(site.collars, holes)
    geometry = compute_section_geometry(sub.collars, sub.lithologies, points)
    params = sorted({reading.parameter for reading in sub.environmental_readings})
    result = render_cross_section_from_geometry(
        geometry,
        points,
        vertical_exaggeration=ve,
        water_levels=sub.water_levels,
        environmental_readings=sub.environmental_readings,
        environmental_parameters=tuple(params[:1]),
        show_parameter_labels=True,
        chemistry_color_mode=chem,
        chemistry_threshold_green_max=120.0,
        chemistry_threshold_yellow_max=400.0,
        chemistry_label_style=style,
        render_layout=layout,
        consulting_title_block=(
            ConsultingTitleBlock(section_label="A-A'") if layout == "consulting_section" else None
        ),
        screen_intervals=sub.screen_intervals,
        export_framing=ExportFramingConfig(page_preset=page),
        export_formats=frozenset({"png"}),
        close_figure=False,
    )
    bundle = result.retained_export_bundle
    # Refresh the label box patches at screen dpi (the export drew at 300).
    bundle["figure"].draw_without_rendering()
    return bundle["renderer"], bundle["figure"], result


@pytest.fixture(scope="module")
def consulting(site):
    renderer, figure, result = _render(site)
    yield renderer, figure, result
    plt.close(figure)


def _label_box(annotation, mpl_renderer):
    annotation.update_positions(mpl_renderer)
    box = Text.get_window_extent(annotation, mpl_renderer)
    patch = annotation.get_bbox_patch()
    if patch is not None:
        box = patch.get_window_extent(mpl_renderer)
    return box


def _water_labels(renderer, kinds=("rl", "nm", "gradient")):
    return [
        (kind, annotation)
        for kind, annotation, _c in renderer._water_labels
        if kind in kinds and annotation.get_visible()
    ]


def _arr(box):
    return np.array([[box.x0, box.y0, box.x1, box.y1]], dtype=float)


def test_groundwater_markers_draw_above_the_columns(consulting) -> None:
    renderer, figure, _result = consulting
    ax = renderer._water_labels[0][1].axes
    markers = [c for c in ax.__dict__["_css_marker_sets"] if c._css_marker_kind == "water"]
    assert markers
    # Columns draw at 8, the in-column lithology fills at 9.
    column_z = max(
        float(c.get_zorder())
        for c in ax.collections
        if not hasattr(c, "_css_marker_kind") and c.get_zorder() >= 8.0
    )
    mpl_renderer = _figure_renderer(figure)
    frame = ax.get_window_extent(mpl_renderer)
    for collection in markers:
        # Above the columns (8) and the in-column lithology fills (9).
        assert collection.get_zorder() == WATER_MARKER_ZORDER > column_z
        # Thin white edge so the symbol reads on the grey column.
        assert np.allclose(collection.get_edgecolor()[0][:3], (1.0, 1.0, 1.0))
        # Clipped to the frame, and nudged markers stay inside it.
        assert collection.get_clip_on()
        assert collection.get_clip_box() is not None or collection.get_clip_path() is not None
        xs = collection.get_offset_transform().transform(collection.get_offsets())[:, 0]
        assert np.all(xs >= frame.x0 - 0.5) and np.all(xs <= frame.x1 + 0.5)


def test_water_labels_keep_off_lines_columns_markers_and_each_other(consulting) -> None:
    renderer, figure, _result = consulting
    mpl_renderer = _figure_renderer(figure)
    labels = _water_labels(renderer)
    assert sum(kind == "rl" for kind, _a in labels) >= 10
    ax = labels[0][1].axes
    lines = np.vstack((_water_line_segments({ax}), _ground_line_segments({ax})))
    columns = renderer._column_obstacle_boxes(figure)
    frame = ax.get_window_extent(mpl_renderer)
    boxes = []
    for kind, annotation in labels:
        box = _label_box(annotation, mpl_renderer)
        boxes.append((annotation.get_text(), box))
        assert frame.x0 - 1 <= box.x0 and box.x1 <= frame.x1 + 1, annotation.get_text()
        assert not _segment_box_hits(lines, _arr(box.padded(-1.0))).any(), (kind, annotation.get_text())
        for column in columns:
            assert not box.padded(-1.0).overlaps(column), (kind, annotation.get_text())
        # Opaque white box: no line or hatch shows through the digits.
        assert annotation.get_bbox_patch().get_alpha() in (None, 1.0)
    overlaps = [(a[0], b[0]) for a, b in itertools.combinations(boxes, 2) if a[1].overlaps(b[1])]
    assert overlaps == []
    # Off every groundwater marker.
    for collection in ax.__dict__["_css_marker_sets"]:
        if collection._css_marker_kind != "water":
            continue
        for x, y in collection.get_offset_transform().transform(collection.get_offsets()):
            for text, box in boxes:
                assert not box.padded(-1.0).contains(x, y), text


def test_rl_numbers_sit_by_their_marker_or_carry_a_leader(consulting) -> None:
    renderer, figure, _result = consulting
    mpl_renderer = _figure_renderer(figure)
    for kind, annotation in _water_labels(renderer, ("rl",)):
        box = _label_box(annotation, mpl_renderer)
        x, y = annotation.axes.transData.transform([annotation.xy])[0]
        dx = max(box.x0 - x, x - box.x1, 0.0)
        dy = max(box.y0 - y, y - box.y1, 0.0)
        far = dx > box.width or dy > box.height
        assert not far or annotation.arrow_patch.get_visible(), annotation.get_text()


def test_gradient_labels_sit_beside_their_own_segment(consulting) -> None:
    renderer, figure, _result = consulting
    mpl_renderer = _figure_renderer(figure)
    gradients = _water_labels(renderer, ("gradient",))
    assert gradients
    for _kind, annotation in gradients:
        assert not annotation.arrow_patch.get_visible()
        box = _label_box(annotation, mpl_renderer)
        p0, p1 = annotation.axes.transData.transform(annotation._gradient_segment)
        centre = np.array([0.5 * (box.x0 + box.x1), 0.5 * (box.y0 + box.y1)])
        seg = p1 - p0
        t = float(np.dot(centre - p0, seg) / np.dot(seg, seg))
        assert 0.0 < t < 1.0, annotation.get_text()
        foot = p0 + t * seg
        # Just off the segment: half the box plus a few points.
        reach = 0.5 * np.hypot(box.width, box.height) + mpl_renderer.points_to_pixels(12.0)
        assert np.hypot(*(centre - foot)) <= reach, annotation.get_text()


def test_leaders_cross_no_label_and_no_other_leader(consulting) -> None:
    renderer, figure, _result = consulting
    mpl_renderer = _figure_renderer(figure)
    labels = [(k, a) for k, a, _c in renderer._water_labels if a.get_visible()]
    boxes = {id(a): _label_box(a, mpl_renderer) for _k, a in labels}
    leaders = []
    for kind, annotation in labels:
        if kind == "chem" or not annotation.arrow_patch.get_visible():
            continue
        box = boxes[id(annotation)]
        relx, rely = annotation._arrow_relpos
        start = (box.x0 + relx * box.width, box.y0 + rely * box.height)
        end = annotation.axes.transData.transform([annotation.xy])[0]
        leaders.append((id(annotation), np.array([[*start, *end]], dtype=float)))
    assert leaders, "the crowded synthetic section needs some leaders"
    for owner, seg in leaders:
        others = np.vstack([_arr(box.padded(-1.0)) for key, box in boxes.items() if key != owner])
        assert not _segment_box_hits(seg, others).any()
    for (_a, seg_a), (_b, seg_b) in itertools.combinations(leaders, 2):
        assert not _segments_cross(seg_a, seg_b).any()


def test_water_numbers_print_two_fixed_decimals() -> None:
    assert [_fmt_water_number(v) for v in (744.0, 743.2, 743.24, 630.4, 3.1, -0.001)] == [
        "744.00",
        "743.20",
        "743.24",
        "630.40",
        "3.10",
        "0.00",
    ]


def test_crowded_band_notes_omitted_gradients_before_levels() -> None:
    class _Label:
        def __init__(self, dropped):
            self._water_dropped = dropped

    labels = [("gradient", _Label(True), "c"), ("gradient", _Label(True), "c"), ("rl", _Label(False), "c")]
    assert _water_drop_notes(labels) == ["2 gradient labels omitted — crowded."]
    assert _water_drop_notes([("rl", _Label(False), "c")]) == []


def test_crowded_gwm_style_band_drops_gradients_first(site) -> None:
    """Three series 0.05 m apart at every hole: no level number is dropped
    while a gradient label is still printed, and drops reach the notes."""
    from render_profiles import CONSULTING_SECTION_PROFILE
    from tests.test_water_series_collisions import _stress_render

    renderer, figure = _stress_render(CONSULTING_SECTION_PROFILE)
    try:
        kinds = [(k, getattr(a, "_water_dropped", False)) for k, a, _c in renderer._water_labels]
        rl_dropped = any(dropped for kind, dropped in kinds if kind == "rl")
        gradients_shown = any(not dropped for kind, dropped in kinds if kind == "gradient")
        assert not (rl_dropped and gradients_shown)
        n_gradients = sum(dropped for kind, dropped in kinds if kind == "gradient")
        notes = " ".join(renderer.chemistry_label_notes)
        if n_gradients:
            assert f"{n_gradients} gradient label" in notes
    finally:
        plt.close(figure)


def test_one_chemistry_label_style_per_figure(site) -> None:
    """strip style: every hole gets a strip, or (one hole too tight) every
    value gets a box — never a mix (synthetic chart, letter portrait, VE 20)."""
    renderer, figure, _result = _render(site, layout="chart", page="letter_portrait", ve=20.0)
    try:
        values = [a for k, a, _c in renderer._water_labels if k == "chem" and a.get_visible()]
        boxed = [getattr(a, "_chem_box_fallback", False) for a in values]
        strips = list(getattr(figure, "_chem_strips", ()))
        assert values
        assert all(boxed) or not any(boxed)
        if all(boxed):
            assert strips == []
    finally:
        plt.close(figure)


def test_strips_cover_only_their_own_values(consulting) -> None:
    """Each knock-out spans adjacent values only (gap < one line), so the
    geology between widely spaced readings stays visible."""
    renderer, figure, _result = consulting
    mpl_renderer = _figure_renderer(figure)
    strips = list(getattr(figure, "_chem_strips", ()))
    assert strips
    values = [
        _label_box(a, mpl_renderer)
        for k, a, _c in renderer._water_labels
        if k == "chem" and a.get_visible()
    ]
    line = max(box.height for box in values)
    pad = mpl_renderer.points_to_pixels(3.0)
    for strip in strips:
        rect = strip.get_window_extent(mpl_renderer)
        inside = sorted(
            (box for box in values if rect.x0 - 1 <= box.x0 and rect.y0 - 1 <= box.y0 and box.y1 <= rect.y1 + 1),
            key=lambda box: -box.y1,
        )
        assert inside, "a strip with no value on it"
        assert rect.y1 - inside[0].y1 <= pad and inside[-1].y0 - rect.y0 <= pad
        for upper, lower in zip(inside, inside[1:]):
            assert upper.y0 - lower.y1 <= line + 1.0


def _scatter_set(ax, xy, kind, size=36.0):
    collection = ax.scatter([xy[0]], [xy[1]], s=size)
    register_marker_set(ax, collection, kind)
    return collection


def test_same_depth_dots_and_gw_marker_on_a_dot_are_nudged_apart() -> None:
    figure, ax = plt.subplots(figsize=(4, 4), dpi=100)
    try:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        dot_a = _scatter_set(ax, (5.0, 5.0), "chem")
        dot_b = _scatter_set(ax, (5.0, 5.01), "chem")
        head = _scatter_set(ax, (5.0, 5.0), "water", size=49.0)
        mpl_renderer = _figure_renderer(figure)
        _separate_markers({ax}, mpl_renderer)
        pos = {
            name: c.get_offset_transform().transform(c.get_offsets())[0]
            for name, c in (("a", dot_a), ("b", dot_b), ("head", head))
        }
        ppt = mpl_renderer.points_to_pixels(1.0)
        r_dot, r_head = 3.0 * ppt, 3.5 * ppt
        assert np.hypot(*(pos["a"] - pos["b"])) >= 2 * r_dot
        # The GW marker steps left of both dots (values print on the right).
        assert pos["head"][0] < min(pos["a"][0], pos["b"][0])
        for name in ("a", "b"):
            assert np.hypot(*(pos["head"] - pos[name])) >= r_dot + r_head
        # A second pass starts from the drawn positions again (idempotent).
        _separate_markers({ax}, mpl_renderer)
        again = head.get_offset_transform().transform(head.get_offsets())[0]
        assert np.allclose(again, pos["head"])
    finally:
        plt.close(figure)


def test_two_series_at_one_head_spread_and_label_their_side() -> None:
    from render_profiles import CONSULTING_SECTION_PROFILE
    from tests.test_water_series_collisions import _stress_render

    renderer, figure = _stress_render(CONSULTING_SECTION_PROFILE)
    try:
        figure.draw_without_rendering()
        mpl_renderer = _figure_renderer(figure)
        own_side = wrong_side = draws = 0
        for kind, annotation, _c in renderer._water_labels:
            marker = getattr(annotation, "_water_marker", None)
            if kind != "rl" or marker is None or not annotation.get_visible():
                continue
            collection, index = marker
            transform = collection.get_offset_transform()
            now = transform.transform([collection.get_offsets()[index]])[0][0]
            base = transform.transform([collection._css_base_offsets[index]])[0][0]
            if abs(now - base) < 1.0:
                continue
            box = _label_box(annotation, mpl_renderer)
            centre = 0.5 * (box.x0 + box.x1)
            frame = annotation.axes.get_window_extent(mpl_renderer)
            if min(base - frame.x0, frame.x1 - base) < 1.5 * box.width:
                continue  # end hole: one side is past the frame
            draws += 1
            # A marker nudged left is labelled on the left (and vice versa)
            # unless that side had no room at all.
            if (now - base) * (centre - base) < 0:
                wrong_side += 1
            else:
                own_side += 1
        assert draws and own_side > 0
        assert wrong_side <= 1
    finally:
        plt.close(figure)


def test_omitted_values_are_counted_on_the_figure(site) -> None:
    for chem in ("threshold", "black"):
        renderer, figure, _result = _render(site, chem=chem)
        try:
            labels = renderer._water_labels
            values = [a for k, a, _c in labels if k == "chem"]
            values[0]._water_dropped = True
            values[1]._water_dropped = True
            renderer._update_chemistry_omitted_footnote(figure, labels)
            key = renderer._chemistry_threshold_key
            assert key is not None and key.get_gid() == THRESHOLD_KEY_GID
            texts = [t.get_text() for t in key.findobj(Text)]
            assert "2 values not printed (no room at this scale) — see QA notes" in texts
            figure.draw_without_rendering()
            # Cleared again once every value prints.
            values[0]._water_dropped = values[1]._water_dropped = False
            renderer._update_chemistry_omitted_footnote(figure, labels)
            key = renderer._chemistry_threshold_key
            texts = [] if key is None else [t.get_text() for t in key.findobj(Text)]
            assert not any("not printed" in text for text in texts)
            if chem == "black":
                assert key is None
        finally:
            plt.close(figure)
