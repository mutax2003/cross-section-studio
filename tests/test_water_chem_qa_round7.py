"""Regression tests for figure-QA round 7 (water + chemistry rendering, qa_hydro workbook).

1. Mixed units: readings in a convertible unit are shown and threshold-coloured
   in the parameter's main unit (MW-04 Chloride 45000 ug/L -> 45 mg/L, green),
   with a QA note; unconvertible units print their own unit in a neutral colour.
2. Labels-only figures with several parameters carry each parameter's glyph,
   matching the legend.
3. Depth (mbgs) mode: water markers only — no joining lines or i= across wells
   with different collars (unless all collars are equal), plus a QA note.
4. Deselecting a parent series keeps the nest series' DRY note and its legend
   marker.
5. Heads on an LNAPL band are drawn hollow so the band stays visible.
6. Inclined-hole column pieces and markers are label obstacles.
7. A water line reaches a measured flowing-artesian head.
8. i= labels: legend key, series names in the legend's case, numbers kept
   beside their own well.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.text import Text

from chemistry_qc import convert_unit, dominant_units, harmonize_units, mixed_unit_warnings
from models import Collar, DeviationReading, EnvironmentalReading, Lithology, WaterLevel
from pipeline import compute_section_geometry, render_cross_section_from_geometry
from render_theme import (
    CHEMISTRY_LABEL_BLACK,
    CHEMISTRY_LABEL_GREEN,
    filter_water_levels_for_plot,
)
from renderer_chemistry import parameter_series_markers
from renderer_water import GRADIENT_KEY_SAMPLE, GRADIENT_LEGEND_LABEL

POINTS = [(-10.0, 0.0), (210.0, 0.0)]


def _collars(rls, *, spacing=50.0, td=20.0):
    return [
        Collar(hole_id=hole_id, easting=spacing * index, northing=0.0, elevation=rl, total_depth=td)
        for index, (hole_id, rl) in enumerate(rls)
    ]


def _render(collars, water=None, *, deviations=(), points=POINTS, **kwargs):
    lithologies = [
        Lithology(hole_id=c.hole_id, from_depth=0.0, to_depth=c.total_depth / 2, lithology_code="SAND")
        for c in collars
    ] + [
        Lithology(hole_id=c.hole_id, from_depth=c.total_depth / 2, to_depth=c.total_depth, lithology_code="CLAY")
        for c in collars
    ]
    geometry = compute_section_geometry(collars, lithologies, points, deviation_readings=deviations)
    kwargs.setdefault("interpolate_water_table", True)
    result = render_cross_section_from_geometry(
        geometry,
        points,
        water_levels=water,
        export_formats=frozenset({"svg"}),
        close_figure=False,
        **kwargs,
    )
    bundle = result.retained_export_bundle
    fig = bundle["figure"]
    fig.canvas.draw()
    return result, bundle["renderer"], fig


def _chem_labels(renderer):
    return [a for kind, a, _c in getattr(renderer, "_water_labels", []) if kind == "chem" and a.get_visible()]


def _reading(hole, parameter, value, unit="mg/L", depth=3.0, **extra):
    return EnvironmentalReading(hole_id=hole, parameter=parameter, value=value, depth=depth, unit=unit, **extra)


# --- 1. Mixed units ------------------------------------------------------------------


def test_convert_unit_known_families_only() -> None:
    assert convert_unit(45000, "ug/L", "mg/L") == 45.0
    assert convert_unit(0.003, "mg/L", "µg/L") == 3.0
    assert convert_unit(2, "µg/g", "mg/kg") == 2.0
    assert convert_unit(500, "ng/L", "ug/L") == 0.5
    assert convert_unit(1500, "ppb", "ppm") == 1.5
    # ppm joins a matrix only when the parameter's other units name one.
    assert convert_unit(1.0, "ppm", "mg/L") is None
    assert convert_unit(1.0, "ppm", "mg/L", ratio_family="aqueous") == 1.0
    assert convert_unit(1.0, "ppm", "mg/L", ratio_family="solid") is None
    assert convert_unit(5, "mg/kg", "mg/L") is None
    assert convert_unit(5, "pCi/L", "mg/L") is None


def test_dominant_unit_is_the_most_used() -> None:
    readings = [
        _reading("MW-01", "Chloride", 10),
        _reading("MW-02", "Chloride", 20, unit="mg/l"),
        _reading("MW-03", "Chloride", 45000, unit="ug/L"),
        _reading("MW-03", "Benzene", 1, unit="µg/L"),
    ]
    assert dominant_units(readings) == {"chloride": "mg/L", "benzene": "µg/L"}


def test_harmonize_converts_with_note_and_flags_unconvertible() -> None:
    readings = [
        _reading("MW-01", "Chloride", 180),
        _reading("MW-02", "Chloride", 320),
        _reading("MW-04", "Chloride", 45000, unit="ug/L"),
        _reading("MW-05", "Chloride", 12, unit="mg/kg"),
        _reading("MW-06", "Chloride", 5000, unit="ug/L", detected=False),
    ]
    result = harmonize_units(readings)
    values = [(r.value, r.unit) for r in result.readings]
    assert values[2] == (45.0, "mg/L") and values[4] == (5.0, "mg/L")
    # Unconvertible: untouched and flagged.
    assert values[3] == (12.0, "mg/kg") and result.is_unconverted(result.readings[3])
    assert "MW-04 Chloride 45000 ug/L shown as 45 mg/L (converted to the parameter's main unit)" in result.notes
    assert any(note.startswith("MW-06 Chloride <5000 ug/L shown as <5 mg/L") for note in result.notes)
    assert any("MW-05 Chloride 12 mg/kg printed with its own unit" in note for note in result.notes)
    # Inputs untouched.
    assert readings[2].value == 45000
    (warning,) = mixed_unit_warnings(readings)
    assert "mg/kg cannot be converted to mg/L" in warning


def _chloride_section(extra_reading, **kwargs):
    collars = _collars([("MW-01", 100.0), ("MW-02", 100.0), ("MW-03", 100.0), ("MW-04", 100.0), ("MW-05", 100.0)])
    readings = [
        _reading("MW-01", "Chloride", 50),
        _reading("MW-02", "Chloride", 180),
        _reading("MW-03", "Chloride", 320),
        _reading("MW-05", "Chloride", 90),
        extra_reading,
    ]
    return _render(
        collars,
        None,
        environmental_readings=readings,
        environmental_parameters=("Chloride",),
        chemistry_color_mode="threshold",
        chemistry_thresholds={"Chloride": (100.0, 250.0)},
        render_layout="consulting_section",
        **kwargs,
    )


def test_converted_reading_is_shown_and_coloured_in_main_unit() -> None:
    result, renderer, fig = _chloride_section(_reading("MW-04", "Chloride", 45000, unit="ug/L"))
    try:
        labels = {a.get_text(): a for a in _chem_labels(renderer)}
        assert "45000" not in labels
        assert "45" in labels
        assert labels["45"].get_color() == CHEMISTRY_LABEL_GREEN
        note = "MW-04 Chloride 45000 ug/L shown as 45 mg/L (converted to the parameter's main unit)"
        assert note in result.qa_notes
        # Legend and threshold key keep the main unit.
        (entry,) = renderer.parameter_series_legend
        assert "(mg/L)" in entry["label"]
        assert renderer.chemistry_threshold_key_text.endswith("mg/L")
    finally:
        plt.close(fig)


def test_unconvertible_reading_prints_its_unit_in_neutral_colour() -> None:
    result, renderer, fig = _chloride_section(_reading("MW-04", "Chloride", 400, unit="mg/kg"))
    try:
        labels = {a.get_text(): a for a in _chem_labels(renderer)}
        assert "400 mg/kg" in labels
        assert labels["400 mg/kg"].get_color() == CHEMISTRY_LABEL_BLACK
        assert any("MW-04 Chloride 400 mg/kg printed with its own unit" in n for n in result.qa_notes)
        (entry,) = renderer.parameter_series_legend
        assert "(mg/L)" in entry["label"]
    finally:
        plt.close(fig)


# --- 2. Parameter glyphs ---------------------------------------------------------------


@pytest.mark.parametrize("mode", ["threshold", "black", "red"])
def test_labels_only_figures_carry_each_parameters_glyph(mode: str) -> None:
    collars = _collars([("MW-01", 100.0), ("MW-02", 100.0), ("MW-03", 100.0)])
    readings = [
        _reading(h, p, v, depth=d)
        for h in ("MW-01", "MW-02", "MW-03")
        for p, v, d in (("Benzene", 0.02, 3.0), ("Chloride", 120.0, 6.0))
    ]
    _result, renderer, fig = _render(
        collars,
        None,
        environmental_readings=readings,
        environmental_parameters=("Benzene", "Chloride"),
        chemistry_color_mode=mode,
        render_layout="consulting_section",
        parameter_draw_markers=False,
        show_parameter_labels=True,
    )
    try:
        markers = parameter_series_markers(["Benzene", "Chloride"], "o")
        assert markers["Benzene"] != markers["Chloride"]
        by_text = {"0.02": markers["Benzene"], "120": markers["Chloride"]}
        labels = _chem_labels(renderer)
        assert labels
        for label in labels:
            glyph = getattr(label, "_chem_dot", None)
            assert glyph is not None, label.get_text()
            assert glyph.get_marker() == by_text[label.get_text()]
            assert glyph.get_visible()
        for entry in renderer.parameter_series_legend:
            assert entry["marker"] == markers[entry["parameter"]]
            assert entry.get("sample_marker") == markers[entry["parameter"]]
        if mode != "threshold":
            glyphs = [
                artist
                for ax in fig.axes
                for artist in ax.lines
                if artist.get_gid() == "parameter-legend-glyph"
            ]
            assert {g.get_marker() for g in glyphs} == set(markers.values())
    finally:
        plt.close(fig)


def test_single_parameter_labels_have_no_glyph() -> None:
    collars = _collars([("MW-01", 100.0), ("MW-02", 100.0)])
    readings = [_reading("MW-01", "Chloride", 120.0), _reading("MW-02", "Chloride", 80.0)]
    _result, renderer, fig = _render(
        collars,
        None,
        environmental_readings=readings,
        environmental_parameters=("Chloride",),
        render_layout="consulting_section",
        parameter_draw_markers=False,
        show_parameter_labels=True,
    )
    try:
        labels = _chem_labels(renderer)
        assert labels and all(getattr(a, "_chem_dot", None) is None for a in labels)
    finally:
        plt.close(fig)


# --- 3. Depth mode -----------------------------------------------------------------------


def _water_lines(fig):
    return [points for ax in fig.axes for points in ax.__dict__.get("_css_water_lines", ())]


def _gradient_texts(renderer):
    return [a.get_text() for kind, a, _c in getattr(renderer, "_water_labels", []) if kind == "gradient"]


@pytest.mark.parametrize("layout", ["consulting_section", "section_sheet"])
def test_depth_mode_draws_markers_only_across_different_collars(layout: str) -> None:
    collars = _collars([("MW-1", 100.0), ("MW-2", 98.0), ("MW-3", 101.0)])
    water = [WaterLevel(hole_id=h, depth=d) for h, d in (("MW-1", 3.0), ("MW-2", 2.0), ("MW-3", 4.0))]
    result, renderer, fig = _render(collars, water, render_layout=layout, elevation_mode="relative")
    try:
        assert _water_lines(fig) == []
        assert _gradient_texts(renderer) == []
        assert any(note.startswith("Depth mode (mbgs): groundwater depths are shown as markers only") for note in result.qa_notes)
        assert all(entry["linestyle"] == "none" for entry in renderer.water_series_legend)
    finally:
        plt.close(fig)
    # Elevation mode: unchanged (line + i=).
    result, renderer, fig = _render(collars, water, render_layout=layout)
    try:
        assert _water_lines(fig)
        assert _gradient_texts(renderer)
    finally:
        plt.close(fig)


def test_depth_mode_joins_wells_with_equal_collars() -> None:
    collars = _collars([("MW-1", 100.0), ("MW-2", 100.005), ("MW-3", 100.0)])
    water = [WaterLevel(hole_id=h, depth=d) for h, d in (("MW-1", 3.0), ("MW-2", 2.0), ("MW-3", 4.0))]
    result, renderer, fig = _render(collars, water, render_layout="consulting_section", elevation_mode="relative")
    try:
        assert _water_lines(fig)
        assert not any(note.startswith("Depth mode (mbgs)") for note in result.qa_notes)
    finally:
        plt.close(fig)


# --- 4. Deselected parent series ------------------------------------------------------------


def _template_water():
    return [
        WaterLevel(hole_id="MW-01", depth=3.2, series_id="2025-06", series_label="June 2025", connect_group="shallow", date="2025-06-12"),
        WaterLevel(hole_id="MW-02", depth=3.5, series_id="2025-06", series_label="June 2025", connect_group="shallow", date="2025-06-12"),
        WaterLevel(hole_id="MW-03", depth=3.8, series_id="2025-06", series_label="June 2025", connect_group="shallow", date="2025-06-12"),
        WaterLevel(hole_id="MW-01", depth=8.0, series_id="2025-06-deep", series_label="June 2025 deep", connect_group="deep", date="2025-06-12"),
        WaterLevel(hole_id="MW-03", depth=7.5, series_id="2025-06-deep", series_label="June 2025 deep", connect_group="deep", date="2025-06-12"),
        WaterLevel(hole_id="MW-02", depth=0.0, status="dry", series_id="2025-06", series_label="June 2025", connect_group="deep", date="2025-06-12"),
    ]


def test_dry_row_of_a_nest_follows_the_plotted_nest_series() -> None:
    kept = filter_water_levels_for_plot(_template_water(), ("2025-06-deep",))
    dry = [level for level in kept if level.status == "dry"]
    assert len(dry) == 1 and dry[0].hole_id == "MW-02"
    assert dry[0].series_id == "2025-06-deep" and dry[0].series_label == "June 2025 deep"
    # Both selected: unchanged (the row stays in its own series).
    both = filter_water_levels_for_plot(_template_water(), ("2025-06", "2025-06-deep"))
    assert [level.series_id for level in both if level.status == "dry"] == ["2025-06"]
    # A different date is a different round: not adopted.
    other = [*_template_water()[:5], _template_water()[5].model_copy(update={"date": "2025-09-01"})]
    assert not [level for level in filter_water_levels_for_plot(other, ("2025-06-deep",)) if level.status == "dry"]


def test_nest_series_alone_keeps_dry_note_and_legend_marker() -> None:
    collars = _collars([("MW-01", 635.0), ("MW-02", 634.5), ("MW-03", 634.0)], spacing=45.0, td=12.0)
    water = filter_water_levels_for_plot(_template_water(), ("2025-06-deep",))
    _result, renderer, fig = _render(
        collars,
        water,
        render_layout="section_sheet",
        show_dry_well_nm=True,
        points=[(0.0, 0.0), (90.0, 0.0)],
    )
    try:
        notes = [a.get_text() for kind, a, _c in renderer._water_labels if kind == "nm" and a.get_visible()]
        assert "DRY" in notes
        ax = fig.axes[0]
        # The series key (in the plot, or merged into the outside legend).
        keys = [a for a in ax.artists if a.get_gid() == "water_series_key"]
        if ax.get_legend() is not None:
            keys.append(ax.get_legend())
        pairs = [
            (text.get_text(), handle)
            for key in keys
            for text, handle in zip(key.get_texts(), key.legend_handles, strict=False)
        ]
        (handle,) = [handle for text, handle in pairs if "JUNE 2025 DEEP" in text.upper()]
        assert handle.get_marker() not in (None, "", "None", "none")
        # MW-02 is dry: the deep line is broken, so the key shows no line.
        assert handle.get_linestyle() in ("None", "none", "")
    finally:
        plt.close(fig)


# --- 5. LNAPL band under hollow markers --------------------------------------------------------


def test_heads_on_an_lnapl_band_are_hollow() -> None:
    collars = _collars([("MW-1", 100.0), ("MW-2", 100.0), ("MW-3", 100.0)])
    water = [
        WaterLevel(hole_id="MW-1", depth=3.0, series_id="a", series_label="June"),
        WaterLevel(hole_id="MW-2", depth=3.5, series_id="a", series_label="June", product_thickness_m=0.35, product_density=0.8),
        WaterLevel(hole_id="MW-3", depth=3.2, series_id="a", series_label="June"),
        WaterLevel(hole_id="MW-2", depth=3.3, series_id="b", series_label="Fall"),
        WaterLevel(hole_id="MW-3", depth=3.4, series_id="b", series_label="Fall"),
    ]
    _result, renderer, fig = _render(collars, water, render_layout="consulting_section")
    try:
        ax = fig.axes[0]
        sets = [c for c in ax.__dict__.get("_css_marker_sets", ()) if c._css_marker_kind == "water"]
        x_mw2 = 50.0 + 10.0
        hollow, filled = 0, 0
        for collection in sets:
            faces = collection.get_facecolors()
            for (x, _y), face in zip(collection._css_base_offsets, faces, strict=False):
                if abs(x - x_mw2) < 1.0:
                    assert face[3] == 0.0, "head on the LNAPL band must be hollow"
                    hollow += 1
                else:
                    assert face[3] > 0.0
                    filled += 1
        assert hollow == 2 and filled == 3  # both series at MW-2
        assert renderer.lnapl_drawn
    finally:
        plt.close(fig)


# --- 6. Obstacles: inclined-hole pieces and markers ----------------------------------------------


def _boxes_overlap(a, b) -> bool:
    return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1


def test_labels_keep_off_inclined_column_pieces_and_markers() -> None:
    collars = _collars([("A", 100.0), ("B", 100.0), ("C", 100.0)])
    deviations = [
        DeviationReading(hole_id="B", depth=0.0, inclination_deg=0.0, azimuth_deg=90.0),
        DeviationReading(hole_id="B", depth=3.0, inclination_deg=30.0, azimuth_deg=90.0),
        DeviationReading(hole_id="B", depth=20.0, inclination_deg=30.0, azimuth_deg=90.0),
    ]
    readings = [
        _reading(h, p, v, depth=d)
        for h in "ABC"
        for p, v, d in (("Benzene", 0.001, 12.0), ("Chloride", 180.0, 12.0))
    ]
    water = [WaterLevel(hole_id=h, depth=11.5) for h in "ABC"]
    _result, renderer, fig = _render(
        collars,
        water,
        deviations=deviations,
        environmental_readings=readings,
        environmental_parameters=("Benzene", "Chloride"),
        chemistry_color_mode="threshold",
        render_layout="consulting_section",
    )
    try:
        canvas_renderer = fig.canvas.get_renderer()
        pieces = renderer._column_extra_obstacle_boxes(fig)
        assert pieces, "inclined hole B has column pieces off its collar x"
        from renderer_water import _marker_boxes

        markers = _marker_boxes({fig.axes[0]}, canvas_renderer, 0.0)
        for kind, label, _c in renderer._water_labels:
            if not label.get_visible() or kind == "nm":
                continue
            # The text (and its box), not the leader running back to the column.
            box = Text.get_window_extent(label, canvas_renderer)
            for piece in pieces:
                assert not _boxes_overlap(box, piece), (kind, label.get_text())
            if kind == "chem":
                for marker in markers:
                    assert not _boxes_overlap(box, marker), label.get_text()
    finally:
        plt.close(fig)


# --- 7. Flowing artesian head -------------------------------------------------------------------


def test_water_line_reaches_a_measured_artesian_head() -> None:
    # MW-2's head (101.0) is 1 m above its collar (100.0); its neighbours
    # are below ground: the line must still reach 101.0 at MW-2.
    collars = _collars([("MW-1", 104.0), ("MW-2", 100.0), ("MW-3", 103.0)])
    water = [
        WaterLevel(hole_id="MW-1", depth=2.0),
        WaterLevel(hole_id="MW-2", depth=0.0, elevation_masl=101.0),
        WaterLevel(hole_id="MW-3", depth=3.0),
    ]
    result, _renderer, fig = _render(collars, water, render_layout="consulting_section", water_interpolate_across_gaps=True)
    try:
        (line,) = _water_lines(fig)
        x_mw2 = 60.0
        assert float(np.interp(x_mw2, line[:, 0], line[:, 1])) == pytest.approx(101.0, abs=1e-6)
        assert any("rises above the ground surface to the flowing artesian head measured at MW-2" in n for n in result.qa_notes)
    finally:
        plt.close(fig)


# --- 8. Gradient labels ---------------------------------------------------------------------------


def test_gradient_key_and_series_case() -> None:
    collars = _collars([("A", 100.0), ("B", 100.0), ("C", 100.0)])
    water = [
        WaterLevel(hole_id="A", depth=3.0, series_id="a", series_label="June 2025"),
        WaterLevel(hole_id="B", depth=4.0, series_id="a", series_label="June 2025"),
        WaterLevel(hole_id="C", depth=4.5, series_id="a", series_label="June 2025"),
        WaterLevel(hole_id="A", depth=3.5, series_id="b", series_label="Fall 2025"),
        WaterLevel(hole_id="B", depth=4.4, series_id="b", series_label="Fall 2025"),
    ]
    _result, renderer, fig = _render(collars, water, render_layout="consulting_section")
    try:
        gradients = _gradient_texts(renderer)
        assert gradients and all(t.startswith(("JUNE 2025 i=", "FALL 2025 i=")) for t in gradients)
        assert renderer.gradient_labels_drawn
        # Consulting sheet: a numbered note in the title block.
        texts = [" ".join(t.get_text().split()) for ax in fig.axes for t in ax.texts]
        assert any(GRADIENT_LEGEND_LABEL in t for t in texts)
    finally:
        plt.close(fig)
    # No i= drawn (one reading per series): no note.
    _result, renderer, fig = _render(collars, water[:1], render_layout="consulting_section")
    try:
        texts = [" ".join(t.get_text().split()) for ax in fig.axes for t in ax.texts]
        assert not any(GRADIENT_LEGEND_LABEL in t for t in texts)
    finally:
        plt.close(fig)
    # Section sheet: a row in the series key (sample "i=" + the sentence).
    _result, renderer, fig = _render(collars, water, render_layout="section_sheet")
    try:
        ax = fig.axes[0]
        keys = [a for a in ax.artists if a.get_gid() == "water_series_key"]
        if ax.get_legend() is not None:
            keys.append(ax.get_legend())
        texts = [" ".join(t.get_text().split()).upper() for key in keys for t in key.get_texts()]
        assert any(GRADIENT_LEGEND_LABEL.split(" = ", 1)[1].upper() in t for t in texts)
        assert GRADIENT_KEY_SAMPLE in [t.get_text() for t in ax.texts] or keys
    finally:
        plt.close(fig)


def test_water_numbers_stay_beside_their_own_well() -> None:
    # Two series with heads close together at every well (GWM Fig 3): each
    # number sits right beside its own column, never out in the span.
    collars = _collars([(h, 632.0) for h in ("W1", "W2", "W3", "W4", "W5")], spacing=40.0)
    heads = {"W1": (631.6, 631.2), "W2": (630.5, 630.2), "W3": (629.9, 629.9), "W4": (630.5, 631.0), "W5": (629.9, 630.5)}
    water = [
        WaterLevel(hole_id=h, depth=0.0, elevation_masl=pair[k], series_id=sid, series_label=label)
        for h, pair in heads.items()
        for k, (sid, label) in enumerate((("may", "May 2024"), ("june", "June 2025")))
    ]
    _result, renderer, fig = _render(
        collars, water, render_layout="consulting_section", points=[(-10.0, 0.0), (170.0, 0.0)]
    )
    try:
        canvas_renderer = fig.canvas.get_renderer()
        columns = sorted(renderer._column_obstacle_boxes(fig), key=lambda b: b.x0)
        for kind, label, _c in renderer._water_labels:
            if kind != "rl" or not label.get_visible():
                continue
            box = label.get_window_extent(canvas_renderer)
            anchor_x = label.axes.transData.transform([label.xy])[0][0]
            own = min(columns, key=lambda col: abs(0.5 * (col.x0 + col.x1) - anchor_x))
            others = [col for col in columns if col is not own]
            centre = 0.5 * (box.x0 + box.x1)
            gap_own = max(box.x0 - own.x1, own.x0 - box.x1, 0.0)
            gap_other = min(max(box.x0 - col.x1, col.x0 - box.x1, 0.0) for col in others)
            assert gap_own < gap_other, (label.get_text(), centre)
    finally:
        plt.close(fig)
