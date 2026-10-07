"""Chemistry value labels sit right of their own borehole column and never
over any column or another label (user report: chloride values printed on
columns / left of the hole)."""

from __future__ import annotations

import itertools

import pytest
from matplotlib.text import Text

from models import Collar, EnvironmentalReading, Lithology
from render_profiles import CONSULTING_SECTION_PROFILE, SECTION_SHEET_PROFILE
from renderer import CrossSectionRenderer
from renderer_water import _chem_footprint, _figure_renderer, _overlap_area
from tests.conftest import run_pipeline

# Client B-B shape: 7 holes over 32 m, uneven spacing, a few readings each.
_BB_EASTINGS = (0.0, 5.0, 10.0, 15.0, 20.0, 26.0, 32.0)
_BB_DEPTHS = (12.0, 15.3, 14.5, 12.4, 9.0, 8.0, 6.0)


def _section(eastings, depths, readings_per_hole, *, step=None):
    ids = [f"BH-{i:02d}" for i in range(len(eastings))]
    collars = [
        Collar(hole_id=h, easting=e, northing=0.0, elevation=100.0, total_depth=d)
        for h, e, d in zip(ids, eastings, depths, strict=True)
    ]
    lith = [
        Lithology(hole_id=h, from_depth=0.0, to_depth=d, lithology_code="Sandy Clay")
        for h, d in zip(ids, depths, strict=True)
    ]
    readings = []
    for h, d in zip(ids, depths, strict=True):
        gap = step if step is not None else (d - 1.0) / readings_per_hole
        readings += [
            EnvironmentalReading(
                hole_id=h, parameter="Chloride", value=float(10 + 37 * k), depth=0.1 + gap * k, unit="mg/kg"
            )
            for k in range(readings_per_hole)
            if 0.1 + gap * k < d
        ]
    projected, polygons, _ = run_pipeline(collars, lith, [(eastings[0], 0.0), (eastings[-1], 0.0)])
    return ids, dict(zip(ids, depths, strict=True)), projected, polygons, readings


def _render(section, profile, **update):
    ids, depths, projected, polygons, readings = section
    renderer = CrossSectionRenderer(
        show_legend=False,
        render_profile=profile.model_copy(
            update={"show_parameter_markers": True, "show_parameter_labels": True, **update}
        ),
        environmental_readings=readings,
        environmental_parameters=("Chloride",),
    )
    figure = renderer.render(polygons, projected, collar_depths=depths)
    return renderer, figure


def _audit(renderer, figure):
    mpl_renderer = _figure_renderer(figure)
    figure.draw_without_rendering()
    columns = renderer._column_obstacle_boxes(figure)
    chem = [ann for kind, ann, _c in renderer._water_labels if kind == "chem"]
    shown = [ann for ann in chem if ann.get_visible()]
    gap_px = mpl_renderer.points_to_pixels(1.5)
    left_of_own = over_column = 0
    for ann in shown:
        box = _chem_footprint(ann, mpl_renderer)
        anchor_x = ann.axes.transData.transform([ann.xy])[0][0]
        own = min(columns, key=lambda col: abs(col.x1 - anchor_x))
        left_of_own += box.x0 < own.x1 + gap_px
        over_column += any(_overlap_area(box, col) > 0 for col in columns)
        assert ann.get_horizontalalignment() == "left"
    boxes = [
        _chem_footprint(ann, mpl_renderer) if kind == "chem" else Text.get_window_extent(ann, mpl_renderer)
        for kind, ann, _c in renderer._water_labels
        if ann.get_visible()
    ]
    overlaps = sum(1 for a, b in itertools.combinations(boxes, 2) if _overlap_area(a, b) > 0)
    return len(chem), len(chem) - len(shown), left_of_own, over_column, overlaps


P2_STICKS = {"parameter_draw_markers": False, "chemistry_color_mode": "red"}


@pytest.mark.parametrize(
    ("profile", "update"),
    [
        (CONSULTING_SECTION_PROFILE, P2_STICKS),
        (CONSULTING_SECTION_PROFILE, {**P2_STICKS, "chemistry_label_style": "box"}),
        (CONSULTING_SECTION_PROFILE, {**P2_STICKS, "chemistry_label_style": "dot"}),
        (CONSULTING_SECTION_PROFILE, {**P2_STICKS, "chemistry_label_style": "stroke"}),
        (SECTION_SHEET_PROFILE, {}),
    ],
)
def test_bb_shaped_section_keeps_every_value_right_of_its_column(profile, update) -> None:
    section = _section(_BB_EASTINGS, _BB_DEPTHS, 6)
    renderer, figure = _render(section, profile, **update)
    total, dropped, left_of_own, over_column, overlaps = _audit(renderer, figure)
    assert total >= 37
    assert (dropped, left_of_own, over_column, overlaps) == (0, 0, 0, 0)


@pytest.mark.parametrize("style", ["plain", "dot", "stroke"])
def test_dense_readings_stack_right_never_over_columns(style) -> None:
    # Readings every 0.2 m, holes 4 m apart: right side is crowded.
    section = _section((0.0, 4.0, 8.0, 12.0), (10.0,) * 4, 49, step=0.2)
    renderer, figure = _render(section, CONSULTING_SECTION_PROFILE, **P2_STICKS, chemistry_label_style=style)
    total, dropped, left_of_own, over_column, overlaps = _audit(renderer, figure)
    assert total == 196
    assert (left_of_own, over_column, overlaps) == (0, 0, 0)
    assert dropped < total // 4  # most values still print; the rest keep their stick
    for kind, ann, _c in renderer._water_labels:
        if kind != "chem" or not ann.get_visible():
            continue
        halo = getattr(ann, "_halo", None)
        if halo is not None:  # outline twin follows moves and font shrinks
            assert tuple(halo.xyann) == tuple(ann.xyann)
            assert halo.get_fontsize() == ann.get_fontsize()
        if abs(ann.xyann[1]) > 20:  # stacked far from its reading: leader shown
            assert ann.arrow_patch.get_visible()
    assert _order_inversions(renderer, figure) == []


def _order_inversions(renderer, figure) -> list[tuple[str, str]]:
    """Pairs (shallower, deeper) of one hole whose labels print in reverse order."""
    mpl_renderer = _figure_renderer(figure)
    holes: dict[float, list[tuple[float, float, str]]] = {}
    for kind, ann, _c in renderer._water_labels:
        if kind != "chem" or not ann.get_visible():
            continue
        anchor_x, anchor_y = ann.axes.transData.transform([ann.xy])[0]
        label_y = anchor_y + mpl_renderer.points_to_pixels(ann.xyann[1])
        holes.setdefault(round(anchor_x, 1), []).append((anchor_y, label_y, ann.get_text()))
    return [
        (a[2], b[2])
        for labels in holes.values()
        for a, b in itertools.permutations(labels, 2)
        if a[0] > b[0] + 0.5 and a[1] <= b[1]
    ]


def test_bb_shaped_section_keeps_depth_order() -> None:
    renderer, figure = _render(_section(_BB_EASTINGS, _BB_DEPTHS, 6), CONSULTING_SECTION_PROFILE, **P2_STICKS)
    assert _order_inversions(renderer, figure) == []


@pytest.mark.parametrize("layout", ["consulting_section", "section_sheet"])
@pytest.mark.parametrize("font_size", [None, 14.0])
def test_crowded_last_hole_keeps_depth_order(monkeypatch, layout, font_size) -> None:
    # QA repro: 6 holes x 8 readings on letter portrait leave the last hole
    # little room before the frame; values printed out of depth order and
    # one drifted ~7 m below its reading.
    from export_framing import ExportFramingConfig
    from pipeline import build_cross_section

    captured = {}
    original = CrossSectionRenderer.render

    def capture(self, *args, **kwargs):
        captured["renderer"] = self
        captured["figure"] = original(self, *args, **kwargs)
        return captured["figure"]

    monkeypatch.setattr(CrossSectionRenderer, "render", capture)
    holes = [f"B{i}" for i in range(6)]
    collars = [
        Collar(hole_id=h, easting=i * 20.0, northing=0.0, elevation=100.0, total_depth=30.0)
        for i, h in enumerate(holes)
    ]
    lith = [
        Lithology(hole_id=h, from_depth=float(j), to_depth=float(j + 1), lithology_code=f"Unit {j}")
        for h in holes
        for j in range(30)
    ]
    readings = [
        EnvironmentalReading(hole_id=h, parameter="Chloride", value=5.0 + 300 * k, depth=1.0 + 3 * k, unit="mg/kg")
        for h in holes
        for k in range(8)
    ]
    build_cross_section(
        collars,
        lith,
        [(0.0, 0.0), (100.0, 0.0)],
        environmental_readings=readings,
        environmental_parameters=["Chloride"],
        show_parameter_labels=True,
        chemistry_color_mode="threshold",
        render_layout=layout,
        export_font_size=font_size,
        export_framing=ExportFramingConfig(page_preset="letter_portrait"),
        export_formats=frozenset({"png"}),
    )
    renderer, figure = captured["renderer"], captured["figure"]
    total, dropped, left_of_own, over_column, overlaps = _audit(renderer, figure)
    assert total == 48
    assert (dropped, left_of_own, over_column, overlaps) == (0, 0, 0, 0)
    assert _order_inversions(renderer, figure) == []
    for kind, ann, _c in renderer._water_labels:
        if kind == "chem":  # no float-noise detour far from the reading
            assert abs(ann.xyann[1] - ann._water_base_xyann[1]) <= 20
