"""Consulting RL / depth axis: round labelled steps, minor ticks, no crowding.

User report: "the graphs is displaying too many numbers" -- the consulting
sheet labelled every metre on both sides (70+ labels on a 35 m section).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from models import Collar, DataParser, Lithology  # noqa: E402
from pipeline import build_cross_section  # noqa: E402
from renderer import CrossSectionRenderer  # noqa: E402
from renderer_consulting import (  # noqa: E402
    true_metre_major_step,
    true_metre_minor_step,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("span_m", "expected"),
    [
        (12.0, 1.0),  # short section: client's 1 m step
        (14.0, 1.0),
        (16.0, 2.0),  # P2 depth axis: client labels every 2 m
        (35.0, 5.0),
        (90.0, 10.0),
        (450.0, 50.0),
    ],
)
def test_major_step_is_round_and_capped(span_m: float, expected: float) -> None:
    step = true_metre_major_step(span_m, axis_height_pt=400.0, label_pt=8.0)
    assert step == expected
    assert int(span_m // step) + 1 <= 15


def test_major_step_thins_further_on_a_short_axes() -> None:
    # 100 pt tall axes at 8 pt labels fits ~6 labels, not 15.
    step = true_metre_major_step(12.0, axis_height_pt=100.0, label_pt=8.0)
    assert int(12.0 // step) + 1 <= 6


@pytest.mark.parametrize(("major", "minor"), [(1.0, 0.5), (2.0, 1.0), (5.0, 1.0), (10.0, 1.0), (50.0, 5.0)])
def test_minor_step_keeps_metre_grid(major: float, minor: float) -> None:
    assert true_metre_minor_step(major) == minor


def _capture_export(monkeypatch) -> dict:
    captured: dict = {}
    original = CrossSectionRenderer.export_figure_bytes

    def spy(self, figure, *args, **kwargs):
        result = original(self, figure, *args, **kwargs)
        captured["fig"] = figure
        captured["headers"] = list(self._header_labels)
        return result

    monkeypatch.setattr(CrossSectionRenderer, "export_figure_bytes", spy)
    return captured


def _drawn_y_labels(ax) -> list:
    low, high = sorted(ax.get_ylim())
    labels = []
    for tick in ax.yaxis.get_major_ticks():
        if low - 1e-9 <= tick.get_loc() <= high + 1e-9:
            labels.extend(
                lab for lab in (tick.label1, tick.label2) if lab.get_visible() and lab.get_text().strip()
            )
    return labels


def _axis_report(captured: dict):
    from renderer_water import _figure_renderer

    fig = captured["fig"]
    renderer = _figure_renderer(fig)
    fig.draw(renderer)
    sides = [_drawn_y_labels(ax) for ax in fig._css_main_axes if ax is not None]
    overlaps = 0
    for labels in sides:
        boxes = [label.get_window_extent(renderer) for label in labels]
        overlaps += sum(1 for a, b in itertools.combinations(boxes, 2) if a.overlaps(b))
    header_hits = [
        (header.get_text(), label.get_text())
        for header in captured["headers"]
        if header.get_visible()
        for labels in sides
        for label in labels
        if header.get_window_extent(renderer).overlaps(label.get_window_extent(renderer))
    ]
    minor = [ax.yaxis.get_minorticklocs() for ax in fig._css_main_axes if ax is not None]
    return fig, sides, overlaps, header_hits, minor


def _assert_clean(captured: dict, *, max_labels: int = 15) -> list:
    fig, sides, overlaps, header_hits, minor = _axis_report(captured)
    try:
        assert len(sides) == 2, "consulting sheet has left and right RL axes"
        for labels in sides:
            assert 2 <= len(labels) <= max_labels, [label.get_text() for label in labels]
        assert overlaps == 0
        assert header_hits == []
        # Unlabelled minor ticks between labels (matplotlib drops minors on majors).
        assert all(len(locs) >= len(sides[0]) - 1 for locs in minor), "unlabelled minor ticks kept"
        return [label.get_text() for label in sides[0]]
    finally:
        plt.close(fig)


def test_tall_section_labels_at_most_fifteen_per_side(monkeypatch) -> None:
    captured = _capture_export(monkeypatch)
    collars = [
        Collar(hole_id="MW-01", easting=0.0, northing=0.0, elevation=700.0, total_depth=90.0),
        Collar(hole_id="MW-02", easting=60.0, northing=0.0, elevation=698.0, total_depth=85.0),
    ]
    liths = [
        Lithology(hole_id=c.hole_id, from_depth=0.0, to_depth=c.total_depth, lithology_code="Clay")
        for c in collars
    ]
    build_cross_section(
        collars, liths, [(0.0, 0.0), (60.0, 0.0)], vertical_exaggeration=2.0,
        render_layout="consulting_section", export_formats=frozenset({"png"}),
    )
    texts = _assert_clean(captured)
    values = [float(t) for t in texts]
    steps = {round(b - a, 6) for a, b in zip(values, values[1:])}
    assert steps == {10.0}, texts  # true metres, not VE-exaggerated plot units


def test_depth_axis_labels_true_depth_with_exaggeration(monkeypatch) -> None:
    captured = _capture_export(monkeypatch)
    collars = [
        Collar(hole_id=h, easting=10.0 * i, northing=0.0, elevation=100.0, total_depth=30.0)
        for i, h in enumerate(("BH-1", "BH-2", "BH-3"))
    ]
    liths = [Lithology(hole_id=c.hole_id, from_depth=0.0, to_depth=30.0, lithology_code="Sand") for c in collars]
    build_cross_section(
        collars, liths, [(0.0, 0.0), (20.0, 0.0)], vertical_exaggeration=2.0,
        render_layout="consulting_section", elevation_mode="relative",
        interpretation_mode="borehole_only", export_formats=frozenset({"png"}),
    )
    texts = _assert_clean(captured)
    assert "0" in texts and "30" in texts, texts  # not 0..60 plot units
    assert max(float(t) for t in texts) <= 32.0


def test_sample_workbook_axis_is_thinned_and_clear(monkeypatch) -> None:
    captured = _capture_export(monkeypatch)
    parse_result = DataParser().parse_file(ROOT / "data" / "sample_boreholes.xlsx")
    points = [(c.easting, c.northing) for c in parse_result.collars]
    build_cross_section(
        parse_result.collars, parse_result.lithologies, points, vertical_exaggeration=5.0,
        render_layout="consulting_section", water_levels=parse_result.water_levels,
        export_formats=frozenset({"png"}),
    )
    texts = _assert_clean(captured)
    assert len(texts) < 20  # was 36 per side (every metre)


@pytest.mark.parametrize("transect_id", ["A_A", "B_B"])
def test_gwm_fixture_axis_is_thinned_and_clear(monkeypatch, transect_id: str) -> None:
    from gwm_reference import build_subset

    captured = _capture_export(monkeypatch)
    spec, subset = build_subset(transect_id)
    points = [(c.easting, c.northing) for c in subset.collars]
    build_cross_section(
        subset.collars, subset.lithologies, points, render_layout="consulting_section",
        vertical_exaggeration=spec.vertical_exaggeration, show_legend=False, show_hatches=True,
        water_levels=subset.water_levels, consulting_title_block=spec.title_block,
        screen_intervals=subset.screen_intervals, export_formats=frozenset({"png"}),
    )
    _assert_clean(captured)


def test_p2_depth_axis_matches_client_two_metre_step(monkeypatch) -> None:
    from advantage_p2_reference.fixtures import build_parse_result

    captured = _capture_export(monkeypatch)
    spec, pr = build_parse_result("A_A")
    points = [(c.easting, c.northing) for c in pr.collars]
    build_cross_section(
        pr.collars, pr.lithologies, points, vertical_exaggeration=spec.vertical_exaggeration,
        render_layout="consulting_section", consulting_title_block=spec.title_block,
        environmental_readings=pr.environmental_readings, environmental_parameters=("Chloride",),
        show_parameter_labels=True, parameter_interpolate_segments=spec.parameter_interpolate_segments,
        interpretation_mode=spec.interpretation_mode, elevation_mode=spec.elevation_mode,
        track_width_m=0.6, export_formats=frozenset({"png"}),
    )
    texts = _assert_clean(captured)
    assert texts[:3] == ["0", "2", "4"]


def test_header_over_top_tick_label_drops_that_label() -> None:
    from render_profiles import CONSULTING_SECTION_PROFILE
    from tests.conftest import run_pipeline

    collars = [
        Collar(hole_id="MW-01", easting=0.0, northing=0.0, elevation=665.0, total_depth=20.0),
        Collar(hole_id="MW-02", easting=50.0, northing=0.0, elevation=664.0, total_depth=18.0),
    ]
    liths = [
        Lithology(hole_id=c.hole_id, from_depth=0.0, to_depth=c.total_depth, lithology_code="Sand")
        for c in collars
    ]
    projected, polygons, _ = run_pipeline(collars, liths, [(0.0, 0.0), (50.0, 0.0)])
    renderer = CrossSectionRenderer(show_legend=False, render_profile=CONSULTING_SECTION_PROFILE)
    fig = renderer.render(polygons, projected, collar_depths={"MW-01": 20.0, "MW-02": 18.0})
    try:
        ax = fig._css_main_axes[0]
        fig.draw_without_rendering()
        top = _drawn_y_labels(ax)[-1]
        top_text = top.get_text()
        # A header parked right on the top left tick label (forced collision).
        x_disp, y_disp = top.get_window_extent().get_points().mean(axis=0)
        header = fig.text(
            x_disp / fig.bbox.width, y_disp / fig.bbox.height, "MW18-18",
            ha="center", va="center", fontsize=8,
        )
        renderer._header_labels.append(header)
        renderer.fit_consulting_page_margins(fig)
        fig.draw_without_rendering()
        remaining = [label.get_text() for label in _drawn_y_labels(ax)]
        assert top_text not in remaining
        assert len(remaining) >= 2  # only the colliding label is dropped
        header.set_visible(False)
        renderer.fit_consulting_page_margins(fig)  # re-run resets the suppression
        fig.draw_without_rendering()
        assert top_text in [label.get_text() for label in _drawn_y_labels(ax)]
    finally:
        plt.close(fig)
