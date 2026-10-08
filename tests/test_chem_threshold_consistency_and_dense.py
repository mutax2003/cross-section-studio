"""Threshold-mode consistency, engine threshold defaults, dense-section label
placement and a placement work-count regression guard.

QA audit (Oct 2026):
1. threshold mode left the interval sticks / fence segments series red under
   green dots;
2. ``chemistry_color_mode="threshold"`` without limits printed black values
   beside red dots;
3. a 60-hole chloride export took ~10 s, dominated by label placement;
4. on a dense 60-hole section values drifted across neighbouring columns
   with long leaders, and dropped values went unreported.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pytest
from matplotlib.collections import LineCollection
from matplotlib.colors import to_hex

import renderer_water
from models import Collar, EnvironmentalReading, Lithology
from render_profiles import (
    CONSULTING_SECTION_PROFILE,
    DEFAULT_CHEMISTRY_THRESHOLD_GREEN_MAX,
    DEFAULT_CHEMISTRY_THRESHOLD_YELLOW_MAX,
    SECTION_SHEET_PROFILE,
    resolved_chemistry_thresholds,
)
from render_theme import (
    CHEMISTRY_LABEL_GREEN,
    CHEMISTRY_LABEL_ORANGE,
    CHEMISTRY_LABEL_RED,
    parameter_series_color,
)
from renderer import CrossSectionRenderer
from renderer_chemistry import _THRESHOLD_MIXED_SEGMENT_COLOR
from renderer_water import _chem_footprint, _figure_renderer, _overlap_area
from tests.conftest import run_pipeline

GREEN, ORANGE, RED = (to_hex(c) for c in (CHEMISTRY_LABEL_GREEN, CHEMISTRY_LABEL_ORANGE, CHEMISTRY_LABEL_RED))


def _two_hole_section():
    ids = ["A", "B"]
    collars = [
        Collar(hole_id=h, easting=i * 20.0, northing=0.0, elevation=100.0, total_depth=12.0)
        for i, h in enumerate(ids)
    ]
    lith = [Lithology(hole_id=h, from_depth=0.0, to_depth=12.0, lithology_code="Clay") for h in ids]
    projected, polygons, _ = run_pipeline(collars, lith, [(0.0, 0.0), (20.0, 0.0)])
    return ids, projected, polygons


def _interval(hole, value, top):
    return EnvironmentalReading(
        hole_id=hole, parameter="Chloride", value=value, from_depth=top, to_depth=top + 1.0, unit="mg/kg"
    )


def _render(profile, readings, **update):
    ids, projected, polygons = _two_hole_section()
    renderer = CrossSectionRenderer(
        show_legend=False,
        render_profile=profile.model_copy(
            update={
                "show_parameter_markers": True,
                "show_parameter_labels": True,
                "parameter_draw_markers": True,
                "chemistry_color_mode": "threshold",
                "chemistry_threshold_green_max": 100.0,
                "chemistry_threshold_yellow_max": 250.0,
                **update,
            }
        ),
        environmental_readings=readings,
        environmental_parameters=("Chloride",),
    )
    figure = renderer.render(polygons, projected, collar_depths={h: 12.0 for h in ids})
    return renderer, figure


def _line_collections(figure, *, linewidth):
    return [
        coll
        for ax in figure.axes
        for coll in ax.collections
        if isinstance(coll, LineCollection) and coll.get_zorder() == 7 and coll.get_linewidth()[0] == linewidth
    ]


# Hole A: green (top) then red; hole B: green then green.
_READINGS = [_interval("A", 20.0, 1.0), _interval("A", 900.0, 5.0), _interval("B", 30.0, 1.0), _interval("B", 60.0, 5.0)]


@pytest.mark.parametrize("profile", [SECTION_SHEET_PROFILE, CONSULTING_SECTION_PROFILE])
def test_threshold_interval_sticks_take_their_readings_colour(profile) -> None:
    import matplotlib.pyplot as plt

    _renderer, figure = _render(profile, _READINGS)
    (sticks,) = _line_collections(figure, linewidth=2.0)
    colours = sorted(to_hex(c) for c in sticks.get_colors())
    assert colours == sorted([GREEN, RED, GREEN, GREEN])
    assert to_hex(parameter_series_color("Chloride")) not in colours
    plt.close(figure)


def test_fixed_mode_sticks_keep_the_series_colour() -> None:
    import matplotlib.pyplot as plt

    _renderer, figure = _render(SECTION_SHEET_PROFILE, _READINGS, chemistry_color_mode="black")
    (sticks,) = _line_collections(figure, linewidth=2.0)
    assert {to_hex(c) for c in sticks.get_colors()} == {to_hex(parameter_series_color("Chloride"))}
    plt.close(figure)


@pytest.mark.parametrize(
    "update",
    [
        {"parameter_interpolate_segments": True},
        {"parameter_interpolate_across_gaps": True},
    ],
)
def test_threshold_fence_segments_share_a_band_colour_or_turn_grey(update) -> None:
    import matplotlib.pyplot as plt

    _renderer, figure = _render(SECTION_SHEET_PROFILE, _READINGS, **update)
    (fence,) = _line_collections(figure, linewidth=1.5)
    colours = sorted(to_hex(c) for c in fence.get_colors())
    # Shallow pair green-green keeps green; deep pair red-green is neutral.
    assert colours == sorted([GREEN, to_hex(_THRESHOLD_MIXED_SEGMENT_COLOR)])
    plt.close(figure)


def test_consulting_threshold_legend_is_three_dots_in_fence_mode() -> None:
    import matplotlib.pyplot as plt

    renderer, figure = _render(
        CONSULTING_SECTION_PROFILE, _READINGS, parameter_interpolate_segments=True
    )
    (entry,) = renderer.parameter_series_legend
    assert [to_hex(c) for c in entry["threshold_colors"]] == [GREEN, ORANGE, RED]
    plt.close(figure)


def test_resolved_thresholds_fill_gaps_with_configure_defaults() -> None:
    assert resolved_chemistry_thresholds(None, None) == (
        DEFAULT_CHEMISTRY_THRESHOLD_GREEN_MAX,
        DEFAULT_CHEMISTRY_THRESHOLD_YELLOW_MAX,
    ) == (100.0, 250.0)
    assert resolved_chemistry_thresholds(50.0, None) == (50.0, 250.0)
    assert resolved_chemistry_thresholds(300.0, None) == (300.0, 300.0)
    assert resolved_chemistry_thresholds(None, 400.0) == (100.0, 400.0)


def test_threshold_mode_without_limits_uses_defaults_for_labels_and_dots(caplog) -> None:
    import matplotlib.pyplot as plt

    import pipeline

    captured = {}
    original = CrossSectionRenderer.render

    def capture(self, *args, **kwargs):
        captured["renderer"] = self
        captured["figure"] = original(self, *args, **kwargs)
        return captured["figure"]

    collars = [
        Collar(hole_id=h, easting=i * 20.0, northing=0.0, elevation=100.0, total_depth=12.0)
        for i, h in enumerate(("A", "B"))
    ]
    lith = [Lithology(hole_id=c.hole_id, from_depth=0.0, to_depth=12.0, lithology_code="Clay") for c in collars]
    readings = [
        EnvironmentalReading(hole_id="A", parameter="Chloride", value=v, depth=d, unit="mg/kg")
        for v, d in ((20.0, 1.0), (180.0, 5.0), (900.0, 9.0))
    ]
    with pytest.MonkeyPatch.context() as mp, caplog.at_level(logging.WARNING, logger="pipeline"):
        mp.setattr(CrossSectionRenderer, "render", capture)
        pipeline.build_cross_section(
            collars,
            lith,
            [(0.0, 0.0), (20.0, 0.0)],
            render_layout="section_sheet",
            environmental_readings=readings,
            environmental_parameters=["Chloride"],
            chemistry_color_mode="threshold",
            export_formats=frozenset(),
        )
    assert any("without both limits" in record.getMessage() for record in caplog.records)
    renderer = captured["renderer"]
    labels = {ann.get_text(): to_hex(ann.get_color()) for kind, ann, _c in renderer._water_labels if kind == "chem"}
    assert labels == {"20": GREEN, "180": ORANGE, "900": RED}
    dots = sorted(
        to_hex(c)
        for ax in captured["figure"].axes
        for coll in ax.collections
        if coll.get_zorder() == 8
        for c in coll.get_facecolors()
    )
    assert dots == sorted(labels.values())
    assert renderer.chemistry_threshold_key_text is not None
    plt.close("all")


# ---------------------------------------------------------------------------
# Dense sections: values stay beside their own hole or are dropped with a note
# ---------------------------------------------------------------------------


def _dense_case(n_holes: int, *, spacing_m: float = 15.0, last_n: int = 14, wide_values: bool = True):
    rng = np.random.default_rng(1)
    collars, lith, readings = [], [], []
    for i in range(n_holes):
        hole = f"BH{i:02d}"
        collars.append(
            Collar(hole_id=hole, easting=i * spacing_m, northing=0.0, elevation=100.0 + rng.uniform(-1, 1), total_depth=12.0)
        )
        lith.append(Lithology(hole_id=hole, from_depth=0.0, to_depth=6.0, lithology_code="Clay", unit_order=1))
        lith.append(Lithology(hole_id=hole, from_depth=6.0, to_depth=12.0, lithology_code="Sand", unit_order=2))
        values = (5.0, 50.0, 150.0, 900.0) if wide_values else (5.0, 7.0)
        for k in range(last_n if i == n_holes - 1 else 4):
            readings.append(
                EnvironmentalReading(
                    hole_id=hole, parameter="Chloride", value=float(rng.choice(values)), depth=0.5 + 0.8 * k, unit="mg/kg"
                )
            )
    return collars, lith, readings, [(0.0, 0.0), ((n_holes - 1) * spacing_m, 0.0)]


def _build_kept(layout, case):
    import pipeline

    captured = {}
    original = CrossSectionRenderer.render

    def capture(self, *args, **kwargs):
        captured["renderer"] = self
        captured["figure"] = original(self, *args, **kwargs)
        return captured["figure"]

    collars, lith, readings, points = case
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(CrossSectionRenderer, "render", capture)
        mp.setattr(
            pipeline,
            "render_cross_section_from_geometry",
            lambda *a, _orig=pipeline.render_cross_section_from_geometry, **k: _orig(*a, **{**k, "close_figure": False}),
        )
        result = pipeline.build_cross_section(
            collars,
            lith,
            points,
            render_layout=layout,
            environmental_readings=readings,
            environmental_parameters=["Chloride"],
            export_formats=frozenset({"png"}),
        )
    return result, captured["renderer"], captured["figure"]


@pytest.mark.parametrize("layout", ["consulting_section", "section_sheet"])
def test_dense_section_values_never_cross_a_neighbour_and_drops_are_reported(layout) -> None:
    import matplotlib.pyplot as plt

    result, renderer, figure = _build_kept(layout, _dense_case(60))
    mpl_renderer = _figure_renderer(figure)
    columns = sorted(renderer._column_obstacle_boxes(figure), key=lambda col: col.x0)
    chem = [ann for kind, ann, _c in renderer._water_labels if kind == "chem"]
    shown = [ann for ann in chem if ann.get_visible()]
    dropped = len(chem) - len(shown)
    assert shown, "some values must still print"
    gap = mpl_renderer.points_to_pixels(1.5)
    for ann in shown:
        box = _chem_footprint(ann, mpl_renderer)
        frame = ann.axes.get_window_extent(mpl_renderer)
        anchor_x = ann.axes.transData.transform([ann.xy])[0][0]
        index = min(range(len(columns)), key=lambda i: abs(columns[i].x1 - anchor_x))
        own = columns[index]
        assert box.x0 >= own.x1 + gap  # right of its own column
        if index + 1 < len(columns):
            assert box.x1 <= columns[index + 1].x0  # never past the next column
        elif layout == "consulting_section":
            assert box.x1 <= frame.x1 + 0.5  # last hole: inside the fixed frame
        assert frame.y0 - 0.5 <= box.y0 and box.y1 <= frame.y1 + 0.5  # inside vertically
        assert not any(_overlap_area(box, col) > 0 for col in columns)
        # No long drifts: a value stays within a few label heights of its reading.
        anchor_y = ann.axes.transData.transform([ann.xy])[0][1]
        assert abs(0.5 * (box.y0 + box.y1) - anchor_y) < 12 * box.height
    assert dropped > 0  # 3-digit values cannot fit the ~9 pt gaps at this scale
    notes = [w for w in result.overlap_warnings if w.startswith("Chemistry labels:")]
    assert len(notes) == 1 and f"{dropped} value(s) not printed" in notes[0]
    assert renderer.chemistry_label_notes == notes
    # The crowded last hole keeps all 14 values (it has the frame to itself).
    last = [ann for ann in chem if getattr(ann, "_chem_hole_id", "") == "BH59"]
    assert len(last) == 14 and all(ann.get_visible() for ann in last)
    plt.close(figure)


def test_section_without_drops_reports_nothing() -> None:
    import matplotlib.pyplot as plt

    result, renderer, figure = _build_kept("section_sheet", _dense_case(6, spacing_m=40.0))
    assert renderer.chemistry_label_notes == []
    assert not any(w.startswith("Chemistry labels:") for w in result.overlap_warnings)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Placement work guard (call counts, not wall time, so it cannot flake)
# ---------------------------------------------------------------------------


def test_label_placement_measures_each_value_a_bounded_number_of_times(monkeypatch) -> None:
    """25-hole chemistry export with crowded values: footprints are memoised
    across placement passes (compact re-runs, the export re-placement), so a
    value's text is laid out a handful of times, not once per candidate font
    size per pass as before (~17 footprints per value on the 60-hole case)."""
    import matplotlib.pyplot as plt

    counts = {"requested": 0, "measured": 0}
    footprint = renderer_water._chem_footprint
    measure = renderer_water._measure_chem_footprint

    def requested(annotation, renderer):
        counts["requested"] += 1
        return footprint(annotation, renderer)

    def measured(annotation, renderer):
        counts["measured"] += 1
        return measure(annotation, renderer)

    monkeypatch.setattr(renderer_water, "_chem_footprint", requested)
    monkeypatch.setattr(renderer_water, "_measure_chem_footprint", measured)
    started = time.perf_counter()
    _result, renderer, figure = _build_kept("section_sheet", _dense_case(25, spacing_m=12.0))
    elapsed = time.perf_counter() - started
    labels = sum(1 for kind, _ann, _c in renderer._water_labels if kind == "chem")
    assert labels == 24 * 4 + 14
    assert counts["measured"] <= 6 * labels, counts
    assert counts["requested"] >= 2 * counts["measured"], counts  # the memo is doing its job
    # Generous wall-clock backstop only (about 2 s here; never a tight bound).
    assert elapsed < 60.0
    plt.close(figure)
