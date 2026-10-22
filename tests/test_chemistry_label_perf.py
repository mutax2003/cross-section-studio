"""Chemistry label placement performance guards.

Perf audit (Oct 2026): placement was superlinear in the number of values on
a hole (2000 readings on one hole took 45-71 s, 500 took ~10 s) and a
60-hole all-overlay export spent most of its time in repeated label passes.
The fixes memoise text layouts and leader paths, pre-filter collision tests
to nearby obstacles and skip a placement run whose inputs are unchanged; all
of them leave the drawn output bit-identical.

Work counts are asserted (they cannot flake); wall-clock bounds are generous
backstops for CI machines only.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Text

import renderer_water
from models import Collar, EnvironmentalReading, Lithology
from render_profiles import SECTION_SHEET_PROFILE
from renderer import CrossSectionRenderer

_N_READINGS = 300
_PARAMETERS = tuple(f"P{i:02d}" for i in range(20))


def _one_hole_case(n: int):
    rng = np.random.default_rng(3)
    collars = [
        Collar(hole_id="BH0", easting=0.0, northing=0.0, elevation=100.0, total_depth=30.0),
        Collar(hole_id="BHX", easting=40.0, northing=0.0, elevation=100.0, total_depth=30.0),
    ]
    lith = [
        Lithology(hole_id="BH0", from_depth=0.0, to_depth=15.0, lithology_code="Clay"),
        Lithology(hole_id="BH0", from_depth=15.0, to_depth=30.0, lithology_code="Sand"),
        Lithology(hole_id="BHX", from_depth=0.0, to_depth=30.0, lithology_code="Clay"),
    ]
    readings = [
        EnvironmentalReading(
            hole_id="BH0",
            parameter=_PARAMETERS[k % len(_PARAMETERS)],
            value=float(np.round(rng.uniform(0.1, 999.0), 2)),
            depth=0.2 + 29.5 * k / n,
            unit="mg/L",
        )
        for k in range(n)
    ]
    return collars, lith, readings


def _render(n: int):
    from tests.conftest import run_pipeline

    collars, lith, readings = _one_hole_case(n)
    projected, polygons, _ = run_pipeline(collars, lith, [(0.0, 0.0), (40.0, 0.0)])
    renderer = CrossSectionRenderer(
        show_legend=False,
        environmental_readings=readings,
        environmental_parameters=_PARAMETERS,
        render_profile=SECTION_SHEET_PROFILE.model_copy(update={"show_parameter_labels": True}),
    )
    figure = renderer.render(polygons, projected, collar_depths={"BH0": 30.0, "BHX": 30.0})
    return renderer, figure


@pytest.fixture
def counters(monkeypatch):
    counts = {"passes": 0, "pass_seconds": 0.0, "layouts": 0, "leader_paths": 0}
    run_passes = CrossSectionRenderer._run_label_passes

    def timed_passes(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return run_passes(self, *args, **kwargs)
        finally:
            counts["passes"] += 1
            counts["pass_seconds"] += time.perf_counter() - started

    layout = Text._get_layout

    def counted_layout(self, renderer):
        if isinstance(self, renderer_water._MemoLayoutAnnotation):
            counts["layouts"] += 1
        return layout(self, renderer)

    path = FancyArrowPatch._get_path_in_displaycoord

    def counted_path(self):
        counts["leader_paths"] += 1
        return path(self)

    monkeypatch.setattr(CrossSectionRenderer, "_run_label_passes", timed_passes)
    monkeypatch.setattr(Text, "_get_layout", counted_layout)
    monkeypatch.setattr(FancyArrowPatch, "_get_path_in_displaycoord", counted_path)
    return counts


def test_crowded_hole_places_within_budget(counters) -> None:
    """300 values on one hole: the label passes stay well under a second
    each on a dev machine (~2-3 s for every pass of the render); 20 s is a CI
    backstop that the pre-fix code (~6-7 s, superlinear) would approach at
    ~600 readings."""
    import matplotlib.pyplot as plt

    started = time.perf_counter()
    renderer, figure = _render(_N_READINGS)
    elapsed = time.perf_counter() - started
    chem = [ann for kind, ann, _c in renderer._water_labels if kind == "chem"]
    assert len(chem) == _N_READINGS
    assert any(ann.get_visible() for ann in chem)
    assert counters["pass_seconds"] < 20.0, counters
    assert elapsed < 60.0
    # Text layouts are memoised per string / size / renderer: a value is laid
    # out a handful of times (one per font size tried), not once per
    # measurement (~15-20 per value before).
    assert counters["layouts"] <= 6 * _N_READINGS, counters
    plt.close(figure)


def test_export_of_an_unchanged_figure_reuses_the_placement(counters) -> None:
    """A PNG export at the rendered page size changes nothing the placement
    reads: the label run is not repeated, and the result is the same."""
    import matplotlib.pyplot as plt

    renderer, figure = _render(120)
    before = [(tuple(ann.xyann), ann.get_visible()) for _k, ann, _c in renderer._water_labels]
    passes = counters["passes"]
    png = renderer.to_png_bytes(figure, dpi=72)
    assert png.startswith(b"\x89PNG")
    assert counters["passes"] == passes
    assert [(tuple(ann.xyann), ann.get_visible()) for _k, ann, _c in renderer._water_labels] == before
    plt.close(figure)


def test_leader_paths_are_computed_once_per_geometry(counters) -> None:
    """Draws and tight-bbox passes at one dpi reuse a leader's clipped path."""
    import matplotlib.pyplot as plt

    renderer, figure = _render(120)
    leaders = [
        ann
        for _k, ann, _c in renderer._water_labels
        if ann.get_visible() and ann.arrow_patch is not None and ann.arrow_patch.get_visible()
    ]
    assert leaders, "the crowded hole should leader some values"
    assert all(isinstance(ann.arrow_patch, renderer_water._MemoLeaderPatch) for ann in leaders)
    renderer_fig = figure.canvas.get_renderer()
    counters["leader_paths"] = 0
    for _ in range(3):
        figure.get_tightbbox(renderer_fig)
    # One computation per leader at this geometry (hidden leaders of visible
    # values count too: matplotlib includes them in the label extent).
    visible = sum(1 for _k, ann, _c in renderer._water_labels if ann.get_visible())
    assert counters["leader_paths"] <= visible, counters
    plt.close(figure)


def test_text_extent_matches_matplotlib_bit_for_bit() -> None:
    """The memoised extent is the exact ``Text.get_window_extent`` box."""
    import matplotlib.pyplot as plt

    renderer, figure = _render(60)
    mpl_renderer = renderer_water._figure_renderer(figure)
    for _kind, ann, _c in renderer._water_labels:
        if not ann.get_visible():
            continue
        for size in (6.5, 5.72, 4.42):
            ann.set_fontsize(size)
            expected = Text.get_window_extent(ann, mpl_renderer).get_points()
            assert np.array_equal(renderer_water._text_window_extent(ann, mpl_renderer).get_points(), expected)
            # The memoised layout also feeds matplotlib's own extent.
            assert np.array_equal(Text.get_window_extent(ann, mpl_renderer).get_points(), expected)
    plt.close(figure)


def test_prefiltered_hit_tests_match_the_full_tests() -> None:
    rng = np.random.default_rng(11)
    for _ in range(200):
        segs = rng.uniform(0, 400, size=(rng.integers(0, 40), 4))
        corners = rng.uniform(0, 400, size=(rng.integers(0, 30), 2))
        sizes = rng.uniform(1, 40, size=(corners.shape[0], 2))
        boxes = np.column_stack((corners, corners + sizes))
        full = renderer_water._segment_box_hits(segs, boxes)
        assert np.array_equal(renderer_water._any_segment_hits(segs, boxes), full.any(axis=0))
        assert np.array_equal(renderer_water._segment_hits_any(segs, boxes), full.any(axis=1))
        circles = np.column_stack((rng.uniform(0, 400, size=(segs.shape[0], 2)), rng.uniform(1, 10, segs.shape[0])))
        if boxes.shape[0]:
            cx, cy, r = circles[None, :, 0], circles[None, :, 1], circles[None, :, 2] + 2.0
            nx = np.clip(cx, boxes[:, None, 0], boxes[:, None, 2])
            ny = np.clip(cy, boxes[:, None, 1], boxes[:, None, 3])
            expected = np.any((nx - cx) ** 2 + (ny - cy) ** 2 < r**2, axis=1)
            assert np.array_equal(renderer_water._circles_hit_boxes(circles, boxes, 2.0), expected)

