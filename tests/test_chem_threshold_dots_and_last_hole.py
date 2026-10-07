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
