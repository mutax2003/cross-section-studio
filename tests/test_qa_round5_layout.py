"""Visual QA round 5: legend columns, label clearance at the frame top,
water numbers beside their marker, and one-hole lenses inside the fence.

1. The consulting legend splits its rows into balanced columns (row counts
   within one of each other, the symbol keys kept together) at every panel
   width -- never "CLAY" alone beside eight rows.
2. On a wide shallow section drawn at VE 1 a surface reading's value sits
   just below the ground line, inside the frame; water numbers take a slot
   beside their marker (on the water line if need be) over a long leader
   down the hole.
3. With pinch-outs off, a one-hole unit's inferred lenses extend only toward
   existing neighbours and stay inside the fence envelope.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import renderer_consulting
from export_framing import ExportFramingConfig
from models import Collar, ConsultingTitleBlock, EnvironmentalReading, Lithology, WaterLevel
from pipeline import compute_section_geometry, render_cross_section_from_geometry
from render_profiles import CONSULTING_SECTION_PROFILE
from renderer import CrossSectionRenderer
from renderer_consulting import _balanced_splits
from renderer_water import (
    _chem_footprint,
    _figure_renderer,
    _ground_display_y,
    _text_box,
)
from tests.conftest import run_pipeline

# --- 1. balanced legend columns -----------------------------------------------


@pytest.mark.parametrize(("n", "ncols"), [(9, 2), (10, 2), (10, 3), (11, 3), (4, 3), (1, 2)])
def test_balanced_splits_differ_by_at_most_one_row(n: int, ncols: int) -> None:
    splits = list(_balanced_splits(n, ncols))
    assert splits
    for columns in splits:
        assert [i for col in columns for i in col] == list(range(n))
        sizes = [len(col) for col in columns]
        assert max(sizes) - min(sizes) <= 1


def test_balanced_splits_keep_the_symbol_keys_together() -> None:
    # Ten rows, the last two symbol keys: 5 | 5 keeps them in one column.
    for columns in _balanced_splits(10, 2, keep_tail=2):
        assert len(columns[-1]) >= 2
    # Nine rows, five symbol keys: only 4 | 5 keeps them together.
    assert [[len(c) for c in s] for s in _balanced_splits(9, 2, keep_tail=5)] == [[4, 5]]
    # Too many to keep together: every balanced split is still offered.
    assert list(_balanced_splits(4, 2, keep_tail=3))
    assert list(_balanced_splits(10, 2, max_rows=4)) == []


def _swatch(label: str):
    return ("swatch", label, {"facecolor": "#aaaaaa", "edgecolor": "#222222", "hatch": None})


_UNITS = ["CLAY", "CLAY LOAM", "FILL", "LOAMY SAND", "SAND", "SANDY CLAY", "SANDY CLAY LOAM", "SILTY CLAY"]
_SYMBOLS = [
    ("threshold_dots", "CHLORIDE CONCENTRATION (mg/kg)", {"color": "k", "marker": "o", "parameter_entry": {}}),
    ("pinch_out", "INFERRED PINCH-OUT", {"color": "k", "linestyle": "--"}),
]


def _column_counts(layout) -> list[int]:
    xs = sorted({round(row.x, 3) for row in layout.rows})
    return [sum(1 for row in layout.rows if round(row.x, 3) == x) for x in xs]


@pytest.mark.parametrize("panel_w_in", [1.6, 2.0, 2.2, 2.4, 2.9, 3.4, 4.2])
def test_legend_columns_balance_at_every_panel_width(panel_w_in: float) -> None:
    entries = [_swatch(u) for u in _UNITS] + _SYMBOLS
    renderer = CrossSectionRenderer(render_profile=CONSULTING_SECTION_PROFILE, show_legend=False)
    figure = plt.figure(figsize=(11.0, 8.5), dpi=100)
    try:
        layout = renderer._legend_panel_layout(
            figure, entries, len(_UNITS), panel_w_in=panel_w_in, panel_h_in=1.8, prefer_cols=2
        )
        counts = _column_counts(layout)
        assert len(layout.rows) == len(entries)
        assert layout.hidden_units == 0
        assert max(counts) - min(counts) <= 1, counts
        if layout.ncols > 1:
            # The two symbol keys share the last column.
            last_x = max(row.x for row in layout.rows)
            assert all(row.x == last_x for row in layout.rows if row.entry[0] != "swatch")
    finally:
        plt.close(figure)


@pytest.mark.parametrize("page", ["letter_portrait", "letter_landscape", "tabloid_landscape"])
def test_rendered_consulting_legend_columns_balance(monkeypatch, page: str) -> None:
    captured: list = []
    original = renderer_consulting.ConsultingLayoutMixin._legend_panel_layout

    def spy(self, *args, **kwargs):
        layout = original(self, *args, **kwargs)
        captured.append(layout)
        return layout

    monkeypatch.setattr(renderer_consulting.ConsultingLayoutMixin, "_legend_panel_layout", spy)
    codes = ["Clay", "Clay Loam", "Fill", "Loamy Sand", "Sand", "Sandy Clay", "Sandy Clay Loam", "Silty Clay"]
    collars = [
        Collar(hole_id=f"BH-{i}", easting=5.0 * i, northing=0.0, elevation=635.0, total_depth=16.0)
        for i in range(7)
    ]
    lith = [
        Lithology(hole_id=c.hole_id, from_depth=2.0 * j, to_depth=2.0 * j + 2.0, lithology_code=codes[(i + j) % 8])
        for i, c in enumerate(collars)
        for j in range(8)
    ]
    readings = [EnvironmentalReading(hole_id=c.hole_id, parameter="Chloride", value=120.0, depth=3.0) for c in collars]
    geometry = compute_section_geometry(collars, lith, [(0.0, 0.0), (30.0, 0.0)], allow_pinch_outs=False)
    result = render_cross_section_from_geometry(
        geometry,
        [(0.0, 0.0), (30.0, 0.0)],
        vertical_exaggeration=20.0,
        environmental_readings=readings,
        environmental_parameters=("Chloride",),
        show_parameter_labels=True,
        chemistry_color_mode="threshold",
        chemistry_threshold_green_max=250.0,
        chemistry_threshold_yellow_max=1000.0,
        render_layout="consulting_section",
        consulting_title_block=ConsultingTitleBlock(section_label="A-A'"),
        export_framing=ExportFramingConfig(page_preset=page),
        export_formats=frozenset({"png"}),
        close_figure=False,
    )
    bundle = result.retained_export_bundle
    try:
        assert captured
        layout = captured[-1]
        counts = _column_counts(layout)
        assert layout.hidden_units == 0
        assert max(counts) - min(counts) <= 1, counts
    finally:
        plt.close(bundle["figure"])


# --- 2. wide shallow section at VE 1 ---------------------------------------------

_SHALLOW = [f"BH-{i + 1:02d}" for i in range(6)]


def _shallow_render(*, layout: str, elevation_mode: str = "absolute", many: bool = False):
    collars = [
        Collar(hole_id=h, easting=40.0 * i, northing=0.0, elevation=round(100.0 - 0.15 * i, 2), total_depth=18.0)
        for i, h in enumerate(_SHALLOW)
    ]
    units = [(0.0, 0.3, "Topsoil"), (0.3, 0.5, "Sand"), (0.5, 5.0, "Clay"), (5.0, 7.0, "Silt"), (7.0, 11.0, "Silty Clay")]
    tail = (
        [(11.0, 13.0, "Gravel"), (13.0, 15.0, "Loamy Sand"), (15.0, 16.0, "Fill"), (16.0, 18.0, "Clay")]
        if many
        else [(11.0, 18.0, "Clay")]
    )
    lith = [
        Lithology(hole_id=c.hole_id, from_depth=a, to_depth=b, lithology_code=code)
        for c in collars
        for a, b, code in units + tail
    ]
    water = [WaterLevel(hole_id=c.hole_id, depth=3.0 + 0.3 * i, series_id="R1") for i, c in enumerate(collars)]
    readings = [
        EnvironmentalReading(hole_id=c.hole_id, parameter="Chloride", value=v, depth=d, unit="mg/kg")
        for i, c in enumerate(collars)
        for d, v in ((1.0, 80.0 + 300.0 * i), (4.0, 1500.0 if i % 2 else 200.0), (9.0, 400.0))
    ]
    points = [(0.0, 0.0), (200.0, 0.0)]
    geometry = compute_section_geometry(collars, lith, points)
    result = render_cross_section_from_geometry(
        geometry,
        points,
        vertical_exaggeration=1.0,
        water_levels=water,
        environmental_readings=readings,
        environmental_parameters=("Chloride",),
        show_parameter_labels=True,
        chemistry_color_mode="threshold",
        chemistry_threshold_green_max=250.0,
        chemistry_threshold_yellow_max=1000.0,
        chemistry_label_style="strip",
        render_layout=layout,
        elevation_mode=elevation_mode,
        consulting_title_block=(
            ConsultingTitleBlock(section_label="A-A'") if layout == "consulting_section" else None
        ),
        export_framing=ExportFramingConfig(page_preset="letter_landscape"),
        export_formats=frozenset({"png"}),
        close_figure=False,
    )
    bundle = result.retained_export_bundle
    bundle["figure"].draw_without_rendering()
    return bundle["renderer"], bundle["figure"]


@pytest.mark.parametrize("layout", ["section_sheet", "consulting_section"])
def test_surface_values_sit_under_the_ground_line_inside_the_frame(layout: str) -> None:
    renderer, figure = _shallow_render(layout=layout)
    try:
        mpl_renderer = _figure_renderer(figure)
        values = [a for k, a, _c in renderer._water_labels if k == "chem" and a.get_visible()]
        assert len(values) >= 12
        for annotation in values:
            ax = annotation.axes
            frame = ax.get_window_extent(mpl_renderer)
            box = _chem_footprint(annotation, mpl_renderer)
            assert box.y1 <= frame.y1 + 0.5, annotation.get_text()
            assert box.y0 >= frame.y0 - 0.5, annotation.get_text()
            ground = [ax.transData.transform(p) for p in ax.__dict__.get("_css_ground_lines", ())]
            assert ground
            top_limit = min(float(_ground_display_y(ground, np.array([x]))[0]) for x in (box.x0, box.x1))
            assert box.y1 <= top_limit + 0.5, annotation.get_text()
    finally:
        plt.close(figure)


def test_water_number_stays_beside_its_marker_over_a_long_leader() -> None:
    renderer, figure = _shallow_render(layout="consulting_section", elevation_mode="relative", many=True)
    try:
        mpl_renderer = _figure_renderer(figure)
        labels = [a for k, a, _c in renderer._water_labels if k == "rl" and a.get_visible()]
        assert labels
        first = min(labels, key=lambda a: a.axes.transData.transform([a.xy])[0][0])
        box = _text_box(first, mpl_renderer)
        marker_y = first.axes.transData.transform([first.xy])[0][1]
        assert abs(0.5 * (box.y0 + box.y1) - marker_y) <= 2.0 * box.height, first.get_text()
    finally:
        plt.close(figure)


# --- 3. one-hole lenses inside the fence ---------------------------------------


def _orphan_case(*, unit_hole: str, from_depth: float, to_depth: float, depths: dict[str, float]):
    collars = [
        Collar(hole_id=h, easting=x, northing=0.0, elevation=100.0, total_depth=depths[h])
        for h, x in (("BH-01", 0.0), ("BH-02", 40.0), ("BH-03", 80.0))
    ]
    lith = []
    for c in collars:
        if c.hole_id == unit_hole:
            lith += [
                Lithology(hole_id=c.hole_id, from_depth=0.0, to_depth=from_depth, lithology_code="Clay"),
                Lithology(hole_id=c.hole_id, from_depth=from_depth, to_depth=to_depth, lithology_code="Gravel"),
            ]
            if to_depth < c.total_depth:
                lith.append(
                    Lithology(hole_id=c.hole_id, from_depth=to_depth, to_depth=c.total_depth, lithology_code="Clay")
                )
        else:
            lith.append(Lithology(hole_id=c.hole_id, from_depth=0.0, to_depth=c.total_depth, lithology_code="Clay"))
    projected, polygons, _ = run_pipeline(
        collars, lith, [(0.0, 0.0), (80.0, 0.0)], allow_pinch_outs=False, render_layout="consulting_section"
    )
    renderer = CrossSectionRenderer(render_profile=CONSULTING_SECTION_PROFILE, show_legend=False)
    figure = renderer.render(
        polygons, projected, collar_depths={c.hole_id: c.total_depth for c in collars}
    )
    hole_summary = renderer._hole_context(projected).summary
    _rows, lenses = renderer._orphan_unit_geometry(projected, polygons, hole_summary, 1.0)
    envelope = renderer._fence_envelope(hole_summary)
    plt.close(figure)
    return lenses, envelope


def test_end_hole_lens_extends_inward_only() -> None:
    lenses, envelope = _orphan_case(
        unit_hole="BH-01", from_depth=4.0, to_depth=8.0, depths={"BH-01": 20.0, "BH-02": 20.0, "BH-03": 20.0}
    )
    assert len(lenses) == 1
    x_min, _, x_max, _ = lenses[0].polygon.bounds
    assert x_min >= -1e-9 and x_max <= 20.0 + 1e-9
    assert lenses[0].is_pinch_out
    assert lenses[0].polygon.within(envelope.buffer(1e-6))


def test_lenses_are_clipped_to_the_fence_envelope() -> None:
    # The unit sits at the bottom of a deep hole beside a shallow neighbour:
    # the lens toward it is cut off at the fence base, never drawn below it.
    lenses, envelope = _orphan_case(
        unit_hole="BH-02", from_depth=16.0, to_depth=20.0, depths={"BH-01": 20.0, "BH-02": 20.0, "BH-03": 10.0}
    )
    assert len(lenses) == 2
    for lens in lenses:
        assert lens.polygon.within(envelope.buffer(1e-6)), lens.polygon.bounds
        x_min, _, x_max, _ = lens.polygon.bounds
        assert 20.0 - 1e-9 <= x_min and x_max <= 60.0 + 1e-9
    right = max(lenses, key=lambda lens: lens.polygon.bounds[2])
    # Clipped: the full triangle (reach 10 m, 4 m thick) would reach below
    # the sloping fence base, so its apex is cut short.
    assert right.polygon.bounds[2] < 50.0 - 1e-6
    assert right.polygon.area < 20.0 - 1e-6
