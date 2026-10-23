"""Regression tests for the groundwater-figure QA round (qa_hydro workbook).

1. Segments mode joins a nest's wells across a hole read only in another
   connect_group (nested deep well), but not across a dry / NM row.
2. Several water series never share colour + marker.
3. The legend date range uses the readings plotted (superseded duplicates out).
4. Legends: chemistry-columns preset keys its water; threshold key lists every
   plotted parameter; flowing artesian heads get a riser and a key.
5. LNAPL band has a minimum page size; NM only for nm rows (not for holes
   without a row in one series); i= labels at 2 significant figures, prefixed
   by the series when several are drawn.
6. Inclined holes: column, screens and water markers follow the projected
   interval x positions instead of the collar x.
"""

from __future__ import annotations

import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.text import Text

from models import (
    Collar,
    DeviationReading,
    EnvironmentalReading,
    Lithology,
    ScreenInterval,
    WaterLevel,
)
from pipeline import compute_section_geometry, render_cross_section_from_geometry
from render_theme import LNAPL_HATCH, consulting_gw_series_style
from renderer_water import (
    ARTESIAN_LEGEND_LABEL,
    distinct_series_styles,
    format_gradient_text,
    format_sig_figs,
)
from ui_output_presets import resolve_output_preset

POINTS = [(-10.0, 0.0), (210.0, 0.0)]


def _collars(rls, *, spacing=50.0, td=20.0):
    return [
        Collar(hole_id=hole_id, easting=spacing * index, northing=0.0, elevation=rl, total_depth=td)
        for index, (hole_id, rl) in enumerate(rls)
    ]


def _render(collars, water, *, deviations=(), points=POINTS, **kwargs):
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


def _water_lines(fig):
    return [points for ax in fig.axes for points in ax.__dict__.get("_css_water_lines", ())]


def _texts(fig) -> list[str]:
    return [t.get_text() for t in fig.findobj(Text) if t.get_visible() and t.get_text()]


# --- 1. Segments adjacency within the nest -----------------------------------------


NEST = [("MW-5", 100.0), ("MW-6S", 100.0), ("MW-6D", 100.0), ("MW-7", 100.0)]


def _nest_collars():
    collars = _collars(NEST)
    # MW-6D nested 1.5 m from MW-6S.
    collars[2] = collars[2].model_copy(update={"easting": collars[1].easting + 1.5})
    collars[3] = collars[3].model_copy(update={"easting": 150.0})
    return collars


def test_nested_deep_well_does_not_break_shallow_segments() -> None:
    water = [
        WaterLevel(hole_id="MW-5", depth=3.0, connect_group="shallow"),
        WaterLevel(hole_id="MW-6S", depth=3.2, connect_group="shallow"),
        WaterLevel(hole_id="MW-6D", depth=4.0, connect_group="deep"),
        WaterLevel(hole_id="MW-7", depth=3.5, connect_group="shallow"),
    ]
    _result, _renderer, fig = _render(_nest_collars(), water, water_interpolate_across_gaps=False)
    lines = _water_lines(fig)
    plt.close(fig)
    xs = sorted({round(float(x), 3) for line in lines for x in line[:, 0]})
    spans = {(round(float(line[0, 0]), 3), round(float(line[-1, 0]), 3)) for line in lines}
    # MW-5 -> MW-6S and MW-6S -> MW-7 are both drawn (MW-6D skipped).
    assert len(spans) == 2
    assert xs[0] == pytest.approx(10.0) and xs[-1] == pytest.approx(160.0)


def test_nm_row_in_the_same_nest_still_breaks_the_line() -> None:
    water = [
        WaterLevel(hole_id="MW-5", depth=3.0, connect_group="shallow"),
        WaterLevel(hole_id="MW-6S", depth=0.0, status="nm", connect_group="shallow"),
        WaterLevel(hole_id="MW-6D", depth=4.0, connect_group="deep"),
        WaterLevel(hole_id="MW-7", depth=3.5, connect_group="shallow"),
    ]
    _result, _renderer, fig = _render(_nest_collars(), water, water_interpolate_across_gaps=False)
    lines = _water_lines(fig)
    texts = _texts(fig)
    plt.close(fig)
    assert lines == []
    assert "NM" in texts


# --- 2/3. Series styles and legend dates ---------------------------------------


def test_series_with_colliding_defaults_get_distinct_styles() -> None:
    # "2025-06" is a known id (navy circle); an unknown second id lands on the
    # same slot without de-duplication.
    collars = _collars([("A", 100.0), ("B", 100.0), ("C", 100.0)])
    water = [
        WaterLevel(hole_id=h, depth=3.0 + i * 0.2, series_id=sid, series_label=label)
        for sid, label in (("2025-06", ""), ("2025-09", "Fall 2025"))
        for i, h in enumerate("ABC")
    ]
    _result, renderer, fig = _render(collars, water, render_layout="consulting_section")
    plt.close(fig)
    styles = [(e["color"].lower(), e["marker"]) for e in renderer.water_series_legend]
    assert len(styles) == 2 and styles[0] != styles[1]
    # The first series keeps its stable style (figure parity).
    color, marker, _ = consulting_gw_series_style("2025-06", "", series_index=0)
    assert styles[0] == (color.lower(), marker)


def test_distinct_series_styles_keeps_workbook_choices() -> None:
    def slots(k):
        return consulting_gw_series_style("", series_index=k)[:2]

    out = distinct_series_styles(
        [("#000000", "o", False, False), ("#000000", "o", True, False), ("#000000", "o", True, True)], slots
    )
    assert out[0] == ("#000000", "o")
    assert out[1][0] == "#000000" and out[1][1] != "o"  # colour fixed, marker moved
    assert out[2] == ("#000000", "o")  # both fixed: never changed


def test_legend_date_ignores_superseded_reading() -> None:
    collars = _collars([("A", 100.0), ("B", 100.0)])
    water = [
        WaterLevel(hole_id="A", depth=4.0, date=datetime.date(2025, 6, 10)),
        WaterLevel(hole_id="A", depth=4.2, date=datetime.date(2025, 6, 12)),
        WaterLevel(hole_id="B", depth=4.5, date=datetime.date(2025, 6, 11)),
    ]
    _result, renderer, fig = _render(collars, water, render_layout="consulting_section")
    plt.close(fig)
    (entry,) = renderer.water_series_legend
    assert "2025-06-11 TO 06-12" in entry["level_label"].upper()
    assert "06-10" not in entry["level_label"]


# --- 4. Legends --------------------------------------------------------------------------


def test_chemistry_columns_preset_keys_drawn_water() -> None:
    preset = resolve_output_preset("p2_chemistry_sticks")
    assert preset.show_water_legend is True
    collars = _collars([("A", 100.0), ("B", 100.0)])
    water = [WaterLevel(hole_id="A", depth=3.0), WaterLevel(hole_id="B", depth=3.5)]
    _result, renderer, fig = _render(
        collars,
        water,
        render_layout="consulting_section",
        interpretation_mode="borehole_only",
        elevation_mode="relative",
        show_water_legend=preset.show_water_legend,
        show_water_elevation_labels=False,
        show_dry_well_nm=False,
    )
    texts = " ".join(_texts(fig))
    plt.close(fig)
    assert renderer.water_series_legend
    assert "GROUNDWATER LEVEL" in texts


def test_threshold_key_lists_every_plotted_parameter() -> None:
    collars = _collars([("A", 100.0), ("B", 100.0)])
    readings = [
        EnvironmentalReading(hole_id="A", parameter="Chloride", value=50.0, depth=2.0, unit="mg/L"),
        EnvironmentalReading(hole_id="B", parameter="Chloride", value=300.0, depth=2.0, unit="mg/L"),
        EnvironmentalReading(hole_id="A", parameter="Benzene", value=0.01, depth=2.0, unit="mg/L"),
    ]
    _result, renderer, fig = _render(
        collars,
        [WaterLevel(hole_id="A", depth=3.0)],
        render_layout="consulting_section",
        interpretation_mode="borehole_only",
        environmental_readings=readings,
        environmental_parameters=["Chloride", "Benzene"],
        chemistry_color_mode="threshold",
        chemistry_thresholds={"Chloride": (100.0, 250.0), "Benzene": (0.005, 0.05)},
        show_parameter_labels=True,
        parameter_draw_markers=False,
    )
    plt.close(fig)
    lines = renderer.chemistry_threshold_key_lines
    assert any(line.startswith("Benzene") for line in lines)
    assert any(line.startswith("Chloride") for line in lines)


@pytest.mark.parametrize("layout", ["consulting_section", "section_sheet"])
def test_artesian_head_gets_riser_and_key(layout: str) -> None:
    collars = _collars([("A", 100.0), ("B", 100.0), ("C", 100.0)])
    series = [("s1", ""), ("s2", "Fall")]
    water = [
        WaterLevel(hole_id="A", depth=3.0, series_id=sid, series_label=lab) for sid, lab in series
    ] + [
        WaterLevel(hole_id="B", depth=0.0, elevation_masl=101.0, series_id=sid, series_label=lab)
        for sid, lab in series
    ]
    _result, renderer, fig = _render(collars, water, render_layout=layout, show_water_legend=True)
    risers = [c for ax in fig.axes for c in ax.collections if c.get_gid() == "artesian_riser"]
    texts = " ".join(_texts(fig)).upper()
    plt.close(fig)
    assert renderer.artesian_drawn
    assert risers
    seg = risers[0].get_segments()[0]
    assert seg[0, 0] == pytest.approx(seg[1, 0])  # vertical riser at the hole
    assert ARTESIAN_LEGEND_LABEL.upper() in texts


def test_no_artesian_no_riser_or_key() -> None:
    collars = _collars([("A", 100.0), ("B", 100.0)])
    water = [WaterLevel(hole_id="A", depth=3.0), WaterLevel(hole_id="B", depth=3.5)]
    _result, renderer, fig = _render(collars, water, render_layout="consulting_section", show_water_legend=True)
    texts = " ".join(_texts(fig)).upper()
    plt.close(fig)
    assert not renderer.artesian_drawn
    assert ARTESIAN_LEGEND_LABEL.upper() not in texts


# --- 5. Band size, NM, gradient labels ------------------------------------------


def test_lnapl_band_has_minimum_page_size() -> None:
    collars = _collars([("A", 100.0), ("B", 100.0)])
    water = [
        WaterLevel(hole_id="A", depth=3.5, product_thickness_m=0.05, product_density=0.8),
        WaterLevel(hole_id="B", depth=3.0),
    ]
    _result, _renderer, fig = _render(collars, water, render_layout="consulting_section", vertical_exaggeration=5.0)
    (band,) = [c for ax in fig.axes for c in ax.collections if c.get_hatch() == LNAPL_HATCH]
    ax = band.axes
    verts = ax.transData.transform(band.get_paths()[0].vertices[:4])
    plt.close(fig)
    px_per_pt = fig.dpi / 72.0
    assert np.ptp(verts[:, 1]) >= 7.9 * px_per_pt
    assert np.ptp(verts[:, 0]) >= 19.9 * px_per_pt
    assert band.get_zorder() < 9.3  # under the water markers / labels


def test_nm_only_for_nm_rows_not_for_holes_missing_from_a_series() -> None:
    collars = _collars([("A", 100.0), ("B", 100.0), ("C", 100.0)])
    water = [
        WaterLevel(hole_id=h, depth=3.0, series_id="june") for h in "ABC"
    ] + [
        WaterLevel(hole_id="A", depth=3.1, series_id="fall"),
        WaterLevel(hole_id="B", depth=0.0, status="nm", series_id="fall"),
        # C: no fall row at all -> nothing printed.
    ]
    _result, renderer, fig = _render(
        collars, water, render_layout="consulting_section", show_dry_well_nm=True
    )
    notes = {hole: text.get_text() for (_ax, hole), text in renderer._water_status_notes.items()}
    plt.close(fig)
    assert notes == {"B": "NM"}


def test_gradient_label_precision_and_series_prefix() -> None:
    assert format_sig_figs(0.0332) == "0.033"
    assert format_sig_figs(-0.0270) == "-0.027"
    assert format_sig_figs(0.207) == "0.21"
    assert format_sig_figs(0.00333) == "0.0033"
    assert format_sig_figs(0.0996) == "0.10"
    assert format_sig_figs(123.0) == "120"
    assert format_gradient_text(0.0332) == "i=0.033"
    assert format_gradient_text(0.0332, "Fall 2025") == "Fall 2025 i=0.033"

    collars = _collars([("A", 100.0), ("B", 100.0)])
    water = [
        WaterLevel(hole_id="A", depth=3.0, series_id="a", series_label="June"),
        WaterLevel(hole_id="B", depth=4.0, series_id="a", series_label="June"),
        WaterLevel(hole_id="A", depth=3.5, series_id="b", series_label="Fall"),
        WaterLevel(hole_id="B", depth=4.4, series_id="b", series_label="Fall"),
    ]
    _result, _renderer, fig = _render(collars, water, render_layout="consulting_section")
    texts = _texts(fig)
    plt.close(fig)
    # Series names follow the legend's case (upper case there).
    assert "JUNE i=-0.020" in texts
    assert "FALL i=-0.018" in texts


# --- 6. Inclined holes follow the projected intervals ------------------------------


def test_inclined_hole_column_and_markers_follow_interval_x() -> None:
    collars = _collars([("A", 100.0), ("B", 100.0), ("C", 100.0)])
    deviations = [
        DeviationReading(hole_id="B", depth=0.0, inclination_deg=0.0, azimuth_deg=90.0),
        DeviationReading(hole_id="B", depth=3.0, inclination_deg=30.0, azimuth_deg=90.0),
        DeviationReading(hole_id="B", depth=20.0, inclination_deg=30.0, azimuth_deg=90.0),
    ]
    water = [WaterLevel(hole_id=h, depth=12.0) for h in "ABC"]
    screens = [ScreenInterval(hole_id="B", from_depth=8.0, to_depth=14.0)]
    _result, renderer, fig = _render(
        collars,
        water,
        deviations=deviations,
        render_layout="consulting_section",
        screen_intervals=screens,
        interpretation_mode="borehole_only",
    )
    steps = renderer._deviated_x["B"]
    assert "A" not in renderer._deviated_x and "C" not in renderer._deviated_x
    assert float(np.ptp(steps[2])) > 1.0
    collar_rl = 100.0
    trace_y, trace_x = renderer._deviated_trace_plot("B", collar_rl)
    collar_x = float(trace_x[0])
    head = renderer._hole_elevation_at_depth("B", 12.0, collar_rl)  # TVD of the 12 m reading
    water_y = float(renderer._plot_y_values(np.asarray([head]), np.asarray([collar_rl]))[0])
    # The 12 m water level is a measured depth; the survey bends the hole,
    # so the marker sits on the trace at the head's plot height.
    marker_xs = [
        float(x)
        for ax in fig.axes
        for coll in ax.__dict__.get("_css_marker_sets", [])
        if coll._css_marker_kind == "water"
        for x in coll._css_base_offsets[:, 0]
    ]
    column_polys = [
        path.vertices
        for ax in fig.axes
        for coll in ax.collections
        if coll.get_zorder() == 8 and hasattr(coll, "get_paths")
        for path in coll.get_paths()
        if len(path.vertices) > 5
    ]
    plt.close(fig)
    # B's water marker lies on the slanted trace, well off the collar x.
    on_trace = [x for x in marker_xs if x > collar_x + 1.0]
    assert on_trace, marker_xs
    assert abs(on_trace[0] - float(np.interp(water_y, trace_y[::-1], trace_x[::-1]))) < 0.5
    # B's column is ONE continuous slanted polygon (not stacked rectangles):
    # its outline spans from the collar x to the deep trace x.
    assert len(column_polys) == 1
    xs = column_polys[0][:, 0]
    assert float(xs.max() - xs.min()) > float(np.ptp(steps[2]))
