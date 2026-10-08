"""Unlogged-interval legend key and one-hole units on consulting sheets.

1. Depths with no lithology row (log gaps, no recovery, unlogged base of hole)
   show the grey column fill; every layout with a lithology legend lists that
   grey as "Not logged" -- but only when such a gap is drawn.
2. A unit logged in a single hole gets no fence polygon when pinch-outs are
   off (the generic consulting default). Consulting columns are plain grey, so
   the unit was listed in the legend but drawn nowhere; it must now show in the
   column and as a short inferred lens.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

from constants import get_lithology_style
from models import Collar, Lithology
from render_profiles import CHART_PROFILE, CONSULTING_SECTION_PROFILE, SECTION_SHEET_PROFILE
from renderer import CrossSectionRenderer
from renderer_common import UNLOGGED_FILL_COLOR, UNLOGGED_LEGEND_LABEL, unlogged_intervals
from tests.conftest import run_pipeline

COLLARS = [
    Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0),
    Collar(hole_id="BH-02", easting=40.0, northing=0.0, elevation=100.0, total_depth=20.0),
    Collar(hole_id="BH-03", easting=80.0, northing=0.0, elevation=100.0, total_depth=20.0),
]
COLLAR_DEPTHS = {collar.hole_id: collar.total_depth for collar in COLLARS}
TRANSECT = [(0.0, 0.0), (80.0, 0.0)]


def _liths(*, gap: bool, one_hole_unit: bool) -> list[Lithology]:
    rows = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Sand"),
        Lithology(
            hole_id="BH-01", from_depth=7.0 if gap else 5.0, to_depth=20.0, lithology_code="Clay"
        ),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=4.0, lithology_code="Sand"),
        Lithology(hole_id="BH-03", from_depth=0.0, to_depth=6.0, lithology_code="Sand"),
        Lithology(hole_id="BH-03", from_depth=6.0, to_depth=20.0, lithology_code="Clay"),
    ]
    if one_hole_unit:
        rows += [
            Lithology(hole_id="BH-02", from_depth=4.0, to_depth=8.0, lithology_code="Gravel"),
            Lithology(hole_id="BH-02", from_depth=8.0, to_depth=20.0, lithology_code="Clay"),
        ]
    else:
        rows.append(
            Lithology(hole_id="BH-02", from_depth=4.0, to_depth=20.0, lithology_code="Clay")
        )
    return rows


def _render(profile, *, gap: bool, one_hole_unit: bool = False, mode: str = "interpolated", pinch: bool = True):
    projected, polygons, _ = run_pipeline(
        COLLARS,
        _liths(gap=gap, one_hole_unit=one_hole_unit),
        TRANSECT,
        interpretation_mode=mode,
        allow_pinch_outs=pinch,
        render_layout=profile.layout,
    )
    renderer = CrossSectionRenderer(
        render_profile=profile,
        interpretation_mode=mode,
        show_legend=profile.layout != "consulting_section",
    )
    figure = renderer.render(polygons, projected, collar_depths=COLLAR_DEPTHS)
    return renderer, figure, projected, polygons


def _figure_texts(figure) -> list[str]:
    texts = [text.get_text() for ax in figure.axes for text in ax.texts]
    for ax in figure.axes:
        legend = ax.get_legend()
        if legend is not None:
            texts += [text.get_text() for text in legend.get_texts()]
    return texts


def _has_face(ax, colour: str) -> bool:
    target = np.asarray(to_rgba(colour))[:3]
    for collection in ax.collections:
        faces = np.asarray(collection.get_facecolors())
        if faces.size and np.any(np.all(np.isclose(faces[:, :3], target, atol=1e-3), axis=1)):
            return True
    return False


def test_unlogged_intervals_finds_gaps_top_and_base() -> None:
    projected, _, _ = run_pipeline(
        [Collar(hole_id="H", easting=0.0, northing=0.0, elevation=50.0, total_depth=12.0)],
        [
            Lithology(hole_id="H", from_depth=0.5, to_depth=3.0, lithology_code="Sand"),
            Lithology(hole_id="H", from_depth=4.0, to_depth=9.0, lithology_code="Clay"),
            Lithology(hole_id="H", from_depth=9.0, to_depth=10.0, lithology_code="Silt"),
        ],
        [(-10.0, 0.0), (10.0, 0.0)],
        interpretation_mode="borehole_only",
    )
    gaps = unlogged_intervals(projected, {"H": 12.0})
    spans = sorted(
        (round(50.0 - top, 3), round(50.0 - bottom, 3))
        for top, bottom in zip(gaps["top_elevation"], gaps["bottom_elevation"], strict=True)
    )
    assert spans == [(0.0, 0.5), (3.0, 4.0), (10.0, 12.0)]
    # Without a total depth the base below the last row is not counted.
    assert len(unlogged_intervals(projected)) == 2
    assert unlogged_intervals(projected.iloc[0:0]).empty


@pytest.mark.parametrize("profile", [SECTION_SHEET_PROFILE, CHART_PROFILE], ids=["section_sheet", "chart"])
def test_outside_legend_lists_unlogged_only_when_a_gap_is_drawn(profile) -> None:
    _, figure, _, _ = _render(profile, gap=True)
    try:
        assert UNLOGGED_LEGEND_LABEL in _figure_texts(figure)
        assert _has_face(figure.axes[0], UNLOGGED_FILL_COLOR)
    finally:
        plt.close(figure)
    _, figure, _, _ = _render(profile, gap=False)
    try:
        assert UNLOGGED_LEGEND_LABEL not in _figure_texts(figure)
        assert not _has_face(figure.axes[0], UNLOGGED_FILL_COLOR)
    finally:
        plt.close(figure)


def test_consulting_legend_panel_lists_unlogged_when_columns_show_the_log() -> None:
    profile = CONSULTING_SECTION_PROFILE.model_copy(update={"show_track_lithology": True})
    label = UNLOGGED_LEGEND_LABEL.upper()
    _, figure, _, _ = _render(profile, gap=True, mode="borehole_only")
    try:
        assert label in _figure_texts(figure)
    finally:
        plt.close(figure)
    _, figure, _, _ = _render(profile, gap=False, mode="borehole_only")
    try:
        assert label not in _figure_texts(figure)
    finally:
        plt.close(figure)


def test_consulting_fence_columns_do_not_claim_unlogged_key() -> None:
    # Fence consulting columns are uniformly grey: a log gap is not visible.
    _, figure, _, _ = _render(CONSULTING_SECTION_PROFILE, gap=True)
    try:
        assert UNLOGGED_LEGEND_LABEL.upper() not in _figure_texts(figure)
    finally:
        plt.close(figure)


def test_consulting_draws_one_hole_unit_without_pinch_outs() -> None:
    renderer, figure, projected, polygons = _render(
        CONSULTING_SECTION_PROFILE, gap=False, one_hole_unit=True, pinch=False
    )
    try:
        # Stratigraphy builds no Gravel polygon with pinch-outs off ...
        assert "Gravel" not in {polygon.lithology_code for polygon in polygons}
        # ... yet the legend lists it, so the sheet must draw it.
        assert "GRAVEL" in _figure_texts(figure)
        gravel = get_lithology_style("Gravel", consulting_palette=True).color
        assert _has_face(figure.axes[0], gravel)
        hole_summary = renderer._hole_context(projected).summary
        rows, lenses = renderer._orphan_unit_geometry(projected, polygons, hole_summary, 1.0)
        assert list(rows["lithology_code"]) == ["Gravel"]
        # One short lens toward each neighbour, not reaching past the midpoint.
        assert len(lenses) == 2
        for lens in lenses:
            assert lens.is_pinch_out
            x_min, _, x_max, _ = lens.polygon.bounds
            assert 0.0 < x_max - x_min <= 20.0
            assert x_min >= 20.0 - 1e-9 and x_max <= 60.0 + 1e-9
    finally:
        plt.close(figure)


def test_consulting_matched_units_add_no_orphan_geometry() -> None:
    renderer, figure, projected, polygons = _render(
        CONSULTING_SECTION_PROFILE, gap=False, one_hole_unit=True, pinch=True
    )
    try:
        hole_summary = renderer._hole_context(projected).summary
        rows, lenses = renderer._orphan_unit_geometry(projected, polygons, hole_summary, 1.0)
        assert rows.empty and lenses == []
    finally:
        plt.close(figure)
