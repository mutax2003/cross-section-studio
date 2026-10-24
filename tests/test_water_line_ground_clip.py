"""Interpolated groundwater lines are clipped to the ground surface.

A straight line between two heads can rise above a topographic low between
the wells (creek valley); in elevation mode the drawn line follows
``min(line, ground)`` there. Where both heads are above ground (flowing
artesian zone) it is left as drawn and a QA note says so; where one head is
above ground the line still rises to it (only interior runs are clipped).
Sections where the line stays below ground draw exactly as before.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from models import Collar, Lithology, WaterLevel
from pipeline import compute_section_geometry, render_cross_section_from_geometry
from renderer_water import clip_water_line_to_ground

GROUND_X = np.asarray([0.0, 50.0, 100.0])
GROUND_Y = np.asarray([110.0, 96.0, 110.0])  # valley at x = 50
POINTS = [(-10.0, 0.0), (160.0, 0.0)]
X0 = 10.0  # profile x of the first collar (transect starts 10 m before it)


def test_clip_follows_ground_in_valley() -> None:
    xy, clipped, artesian = clip_water_line_to_ground(
        np.asarray([0.0, 100.0]), np.asarray([102.0, 102.0]), GROUND_X, GROUND_Y
    )
    assert clipped == [0] and artesian == []
    assert xy is not None
    ground = np.interp(xy[:, 0], GROUND_X, GROUND_Y)
    assert np.all(xy[:, 1] <= ground + 1e-9)
    # The valley floor and the two line/ground crossings are vertices.
    assert [50.0, 96.0] in xy.tolist()
    crossings = xy[np.isclose(xy[:, 1], 102.0) & (xy[:, 0] > 0.0) & (xy[:, 0] < 100.0)]
    assert crossings[:, 0].tolist() == pytest.approx([50.0 * 8.0 / 14.0, 100.0 - 50.0 * 8.0 / 14.0])
    assert xy[0].tolist() == [0.0, 102.0] and xy[-1].tolist() == [100.0, 102.0]


def test_line_below_ground_is_untouched() -> None:
    xy, clipped, artesian = clip_water_line_to_ground(
        np.asarray([0.0, 100.0]), np.asarray([94.0, 95.0]), GROUND_X, GROUND_Y
    )
    assert xy is None and clipped == [] and artesian == []


def test_both_heads_above_ground_is_left_as_artesian() -> None:
    flat_x = np.asarray([0.0, 100.0])
    flat_y = np.asarray([100.0, 100.0])
    xy, clipped, artesian = clip_water_line_to_ground(
        np.asarray([0.0, 100.0]), np.asarray([100.5, 100.3]), flat_x, flat_y
    )
    assert xy is None and clipped == [] and artesian == [0]


def test_one_artesian_head_line_reaches_the_measured_head() -> None:
    """Round 7: a measured above-ground (flowing artesian) head is real data —
    clipping the line to the ground made it miss the head it joins (MW-07,
    636.00 on a 635.00 collar). The run above ground next to the artesian well
    stays straight, so the line rises out of the ground to the head."""
    flat_x = np.asarray([0.0, 100.0])
    flat_y = np.asarray([100.0, 100.0])
    xy, clipped, artesian = clip_water_line_to_ground(
        np.asarray([0.0, 100.0]), np.asarray([101.0, 99.0]), flat_x, flat_y
    )
    assert xy is None and clipped == [] and artesian == [0]
    # Mirror: artesian head on the right.
    xy, clipped, artesian = clip_water_line_to_ground(
        np.asarray([0.0, 100.0]), np.asarray([99.0, 101.0]), flat_x, flat_y
    )
    assert xy is None and clipped == [] and artesian == [0]


def test_one_artesian_head_still_clips_an_interior_hill() -> None:
    """A valley between the wells (line above ground there) is still clipped;
    only the run ending at the artesian head is left above ground."""
    hill_x = np.asarray([0.0, 20.0, 50.0, 100.0])
    hill_y = np.asarray([100.0, 99.0, 101.0, 100.0])
    xy, clipped, artesian = clip_water_line_to_ground(
        np.asarray([0.0, 100.0]), np.asarray([99.5, 101.0]), hill_x, hill_y
    )
    assert clipped == [0] and artesian == [0]
    assert xy is not None
    assert xy[-1].tolist() == [100.0, 101.0]
    ground = np.interp(xy[:, 0], hill_x, hill_y)
    interior = xy[:, 0] <= 40.0
    assert np.all(xy[interior, 1] <= ground[interior] + 1e-9)
    assert np.any(xy[:, 1] > ground + 1e-6)


# --- Rendered sections -------------------------------------------------------------


def _section(collar_rls, water, **kwargs):
    collars = [
        Collar(hole_id=hole_id, easting=50.0 * index, northing=0.0, elevation=rl, total_depth=30.0)
        for index, (hole_id, rl) in enumerate(collar_rls)
    ]
    lithologies = [
        Lithology(hole_id=c.hole_id, from_depth=0.0, to_depth=30.0, lithology_code="SAND")
        for c in collars
    ]
    geometry = compute_section_geometry(collars, lithologies, POINTS)
    result = render_cross_section_from_geometry(
        geometry,
        POINTS,
        water_levels=water,
        interpolate_water_table=True,
        export_formats=frozenset({"svg"}),
        close_figure=False,
        **kwargs,
    )
    bundle = result.retained_export_bundle
    lines = [
        points
        for ax in bundle["figure"].axes
        for points in ax.__dict__.get("_css_water_lines", ())
    ]
    plt.close(bundle["figure"])
    return result, bundle["renderer"], lines


VALLEY = [("MW-1", 110.0), ("BH-2", 96.0), ("MW-3", 110.0)]
VALLEY_WATER = [WaterLevel(hole_id="MW-1", depth=8.0), WaterLevel(hole_id="MW-3", depth=8.0)]


def test_valley_water_line_is_clipped_to_ground() -> None:
    result, renderer, lines = _section(VALLEY, VALLEY_WATER, water_interpolate_across_gaps=True)
    (line,) = lines
    ground = np.interp(line[:, 0], [X0, X0 + 50.0, X0 + 100.0], [110.0, 96.0, 110.0])
    assert np.all(line[:, 1] <= ground + 1e-6)
    assert line[:, 1].min() == pytest.approx(96.0)
    assert any("MW-1–MW-3" in note and "clipped to the ground surface" in note for note in result.qa_notes)
    assert renderer.water_line_notes


def test_segments_mode_draws_no_line_over_unmeasured_valley() -> None:
    result, renderer, lines = _section(VALLEY, VALLEY_WATER, water_interpolate_across_gaps=False)
    assert lines == []
    assert renderer.water_line_notes == []


def test_simple_section_line_is_unchanged() -> None:
    water = [WaterLevel(hole_id="MW-1", depth=8.0), WaterLevel(hole_id="MW-3", depth=6.0)]
    result, renderer, lines = _section(
        [("MW-1", 110.0), ("BH-2", 109.0), ("MW-3", 108.0)],
        water,
        water_interpolate_across_gaps=True,
    )
    (line,) = lines
    expected_x = np.linspace(X0, X0 + 100.0, 100)
    assert np.array_equal(line[:, 0], expected_x)
    assert np.array_equal(line[:, 1], np.interp(expected_x, [X0, X0 + 100.0], [102.0, 102.0]))
    assert renderer.water_line_notes == []
    assert not any("Groundwater line" in note for note in result.qa_notes)


def test_depth_mode_is_not_clipped() -> None:
    _result, renderer, _lines = _section(
        VALLEY, VALLEY_WATER, water_interpolate_across_gaps=True, elevation_mode="relative"
    )
    assert renderer.water_line_notes == []


def test_artesian_zone_line_left_above_ground_with_note() -> None:
    water = [
        WaterLevel(hole_id="MW-1", depth=0.0, elevation_masl=100.6),
        WaterLevel(hole_id="MW-2", depth=0.0, elevation_masl=100.4),
    ]
    result, _renderer, lines = _section([("MW-1", 100.0), ("MW-2", 100.0)], water)
    assert lines and max(float(line[:, 1].max()) for line in lines) == pytest.approx(100.6)
    assert any("flowing artesian zone" in note for note in result.qa_notes)


@pytest.mark.parametrize("layout", ["section_sheet", "chart", "consulting_section"])
def test_artesian_marker_stays_inside_the_frame(layout: str) -> None:
    water = [WaterLevel(hole_id="MW-1", depth=0.0, elevation_masl=120.0)]
    collars = [("MW-1", 110.0), ("MW-2", 108.0)]
    hole_collars = [
        Collar(hole_id=hole_id, easting=50.0 * index, northing=0.0, elevation=rl, total_depth=30.0)
        for index, (hole_id, rl) in enumerate(collars)
    ]
    lithologies = [
        Lithology(hole_id=c.hole_id, from_depth=0.0, to_depth=30.0, lithology_code="SAND")
        for c in hole_collars
    ]
    geometry = compute_section_geometry(hole_collars, lithologies, POINTS)
    result = render_cross_section_from_geometry(
        geometry,
        POINTS,
        water_levels=water,
        render_layout=layout,
        show_water_elevation_labels=False,
        export_formats=frozenset({"svg"}),
        close_figure=False,
    )
    figure = result.retained_export_bundle["figure"]
    try:
        marker_axes = [
            (ax, collection)
            for ax in figure.axes
            for collection in ax.collections
            if getattr(collection, "_css_marker_kind", None) == "water"
        ]
        assert marker_axes, "water marker not drawn"
        for ax, collection in marker_axes:
            heads = [float(y) for _x, y in collection.get_offsets()]
            assert max(heads) == pytest.approx(120.0)
            assert max(ax.get_ylim()) >= 120.0
    finally:
        plt.close(figure)
