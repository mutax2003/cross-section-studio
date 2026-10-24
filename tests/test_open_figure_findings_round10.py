"""Round 10 open figure findings.

1. More than four chemistry parameters (scripted / batch): the figure itself
   names the ones left off, under the chemistry key, and the plotted order is
   the order passed in (Configure order), not alphabetical.
2. Advantage P2 Fig 6: the sixth hole is "2017-BH10-01", as on the client sheet.
3. Line hatches (Shale ``\\\\``, ``\\``, ``x``) in figure legend swatches: the
   lines are phased into the swatch, so a key shows as many lines as it can
   hold (a single ``\\`` drew none in a section-sheet key, Shale one of two).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import hatch_patterns
from models import MAX_CHEMISTRY_PARAMETERS, Collar, EnvironmentalReading, Lithology
from pipeline import compute_section_geometry, render_cross_section_from_geometry
from render_profiles import (
    CONSULTING_SECTION_PROFILE,
    SECTION_SHEET_PROFILE,
    chemistry_parameter_cap_note,
)
from renderer import CrossSectionRenderer
from renderer_common import THRESHOLD_KEY_GID
from tests.conftest import run_pipeline

ROOT = Path(__file__).resolve().parents[1]
HOLES = ("BH-01", "BH-02", "BH-03")
POINTS = [(0.0, 0.0), (60.0, 0.0)]
# Deliberately not alphabetical: the order passed in must win.
SIX = ("Sulfate", "Chloride", "Benzene", "Arsenic", "Nitrate", "Lead")


def _collars():
    return [
        Collar(hole_id=h, easting=i * 30.0, northing=0.0, elevation=100.0, total_depth=12.0)
        for i, h in enumerate(HOLES)
    ]


def _lith(code: str = "Sandy Clay"):
    return [Lithology(hole_id=h, from_depth=0.0, to_depth=12.0, lithology_code=code) for h in HOLES]


def _readings(names):
    return [
        EnvironmentalReading(
            hole_id=HOLES[i % 3], parameter=name, value=10.0 * (i + 1), depth=1.0 + i, unit="mg/L"
        )
        for i, name in enumerate(names)
    ]


def _pipeline_render(names, **kwargs):
    geometry = compute_section_geometry(_collars(), _lith(), POINTS)
    return render_cross_section_from_geometry(
        geometry,
        POINTS,
        close_figure=False,
        environmental_readings=_readings(names),
        environmental_parameters=names,
        show_parameter_labels=True,
        **kwargs,
    )


def _omitted_note_artists(figure):
    return [
        t
        for t in figure.findobj()
        if getattr(t, "get_gid", lambda: None)() == "chemistry-omitted-parameters"
    ]


# --- 1. parameters beyond the cap are named on the figure ---------------------------


@pytest.mark.parametrize("layout", ["section_sheet", "chart", "consulting_section"])
def test_figure_names_parameters_beyond_the_cap(layout: str) -> None:
    result = _pipeline_render(SIX, render_layout=layout, export_formats=frozenset({"svg"}))
    renderer = result.retained_export_bundle["renderer"]
    figure = result.retained_export_bundle["figure"]
    try:
        assert MAX_CHEMISTRY_PARAMETERS == 4
        assert (
            renderer.chemistry_omitted_parameters_text
            == "Not shown: Nitrate, Lead (4-parameter limit)"
        )
        (note,) = _omitted_note_artists(figure)
        assert note.get_text() == "Not shown: Nitrate, Lead (4-parameter limit)"
        # Drawn under the plot (in the chemistry key's place), inside the page.
        figure.canvas.draw()
        r = figure.canvas.get_renderer()
        box = note.get_window_extent(r)
        main_ax = renderer._chemistry_threshold_key.axes
        assert box.y1 <= main_ax.get_window_extent(r).y0 + 1.0
        assert box.x0 >= 0.0 and box.x1 <= figure.bbox.width and box.y0 >= 0.0
        assert renderer._chemistry_threshold_key.get_gid() == THRESHOLD_KEY_GID
        # The QA note is kept.
        expected = chemistry_parameter_cap_note(SIX[:4], SIX[4:])
        assert expected in result.qa_notes
        assert b"chemistry-omitted-parameters" in result.svg_bytes
    finally:
        plt.close(figure)


def test_no_note_when_nothing_is_dropped() -> None:
    result = _pipeline_render(SIX[:4])
    renderer = result.retained_export_bundle["renderer"]
    figure = result.retained_export_bundle["figure"]
    try:
        assert renderer.chemistry_omitted_parameters_text is None
        assert not _omitted_note_artists(figure)
        assert not [n for n in result.qa_notes if "not plotted" in n]
    finally:
        plt.close(figure)


def test_threshold_mode_note_is_a_row_of_the_key() -> None:
    result = _pipeline_render(
        SIX, chemistry_color_mode="black_red", chemistry_threshold_green_max=100.0
    )
    renderer = result.retained_export_bundle["renderer"]
    figure = result.retained_export_bundle["figure"]
    try:
        key = renderer._chemistry_threshold_key
        assert renderer.chemistry_threshold_key_lines  # the key is drawn
        (note,) = _omitted_note_artists(figure)
        # One artist: the note is the key's last row, under the band rows.
        assert any(getattr(row, "_text", None) is note for row in key._css_rows)
        assert getattr(key._css_rows[-1], "_text", None) is note
    finally:
        plt.close(figure)


def test_direct_renderer_call_gets_the_note_too() -> None:
    projected, polygons, _ = run_pipeline(_collars(), _lith(), POINTS)
    renderer = CrossSectionRenderer(
        show_legend=True,
        render_profile=SECTION_SHEET_PROFILE.model_copy(
            update={"show_parameter_markers": True, "show_parameter_labels": True}
        ),
        environmental_readings=_readings(SIX),
        environmental_parameters=SIX,
    )
    figure = renderer.render(polygons, projected, collar_depths={h: 12.0 for h in HOLES})
    try:
        assert (
            renderer.chemistry_omitted_parameters_text
            == "Not shown: Nitrate, Lead (4-parameter limit)"
        )
        assert renderer.chemistry_cap_notes == [chemistry_parameter_cap_note(SIX[:4], SIX[4:])]
    finally:
        plt.close(figure)


def test_long_omitted_list_wraps_inside_the_frame() -> None:
    names = ("Chloride", "Sodium", "Sulfate", "Benzene") + tuple(
        f"Parameter number {i}" for i in range(12)
    )
    result = _pipeline_render(names, render_layout="consulting_section")
    figure = result.retained_export_bundle["figure"]
    try:
        (note,) = _omitted_note_artists(figure)
        assert "\n" in note.get_text()
        figure.canvas.draw()
        box = note.get_window_extent(figure.canvas.get_renderer())
        assert box.x0 >= 0.0 and box.x1 <= figure.bbox.width
    finally:
        plt.close(figure)


@pytest.mark.parametrize("profile", [SECTION_SHEET_PROFILE, CONSULTING_SECTION_PROFILE])
def test_configure_order_wins_not_alphabetical(profile) -> None:
    result = _pipeline_render(
        SIX,
        render_layout=profile.layout,
        chemistry_color_mode="threshold",
        chemistry_thresholds={name: (100.0, 200.0) for name in SIX},
    )
    renderer = result.retained_export_bundle["renderer"]
    figure = result.retained_export_bundle["figure"]
    try:
        # First four in the order passed in are plotted ...
        assert tuple(renderer.environmental_parameters) == SIX[:4]
        # ... and keyed in that order, in the legend and the threshold key.
        assert [entry["parameter"] for entry in renderer.parameter_series_legend] == list(SIX[:4])
        key_order = [
            line.split(":")[0].removeprefix("Label colour").strip()
            for line in renderer.chemistry_threshold_key_lines
        ]
        assert [name for name in key_order if name] == list(SIX[:4])
    finally:
        plt.close(figure)


# --- 2. P2 Fig 6 hole label ----------------------------------------------------------


def test_p2_fig6_sixth_hole_is_2017_bh10_01() -> None:
    from advantage_p2_reference.chlorides import load_chloride_readings
    from advantage_p2_reference.lithology_digitized import digitized_lithology
    from advantage_p2_reference.transects import ADVANTAGE_P2_TRANSECTS

    spec = ADVANTAGE_P2_TRANSECTS["A_A"]
    assert spec.hole_ids[5] == "2017-BH10-01"
    assert "2017-BH10" not in spec.hole_ids
    assert digitized_lithology("A_A", "2017-BH10-01")
    raw = json.loads(
        (ROOT / "advantage_p2_reference" / "chloride_readings.json").read_text(encoding="utf-8")
    )
    text = json.dumps(raw)
    assert '"2017-BH10"' not in text and '"2017-BH10-01"' in text
    holes = {reading.hole_id for reading in load_chloride_readings(transect_id="A_A")}
    assert "2017-BH10-01" in holes and "2017-BH10" not in holes


# --- 3. line hatches in figure legend swatches ----------------------------------------

# Figure legend swatch sizes (inches): section sheet / chart, consulting title block.
SECTION_SHEET_SWATCH = (0.2222, 0.0994)
CONSULTING_SWATCH = (0.3, 1.0 / 9.0)


def _distinct_lines(
    hatch: str, width: float, height: float, *, min_len: float = 0.02
) -> set[tuple[float, float]]:
    """(angle, offset) of every hatch line with at least ``min_len`` in inside the swatch."""
    _marks, lines = hatch_patterns.swatch_marks_path(hatch, width, height)
    if lines is None:
        return set()
    segs = lines.vertices.reshape(-1, 2, 2)
    length, _mid = hatch_patterns._clipped_lengths(segs, np.zeros(2), np.array([width, height]))
    keep = segs[length >= min_len]
    d = keep[:, 1] - keep[:, 0]
    angle = np.mod(np.arctan2(d[:, 1], d[:, 0]), np.pi)
    normal = np.stack([-np.sin(angle), np.cos(angle)], axis=1)
    offset = np.einsum("ij,ij->i", keep[:, 0], normal)
    return {(round(float(a), 3), round(float(o), 3)) for a, o in zip(angle, offset, strict=True)}


@pytest.mark.parametrize(
    ("hatch", "size", "at_least"),
    [
        ("\\\\", SECTION_SHEET_SWATCH, 2),  # Shale: was 1
        ("\\\\", CONSULTING_SWATCH, 3),  # was 2
        ("\\", SECTION_SHEET_SWATCH, 1),  # was 0: no mark at all
        ("x", SECTION_SHEET_SWATCH, 2),
        ("xxx", SECTION_SHEET_SWATCH, 6),  # Bedrock
        ("-", SECTION_SHEET_SWATCH, 1),
        ("|", SECTION_SHEET_SWATCH, 1),
    ],
)
def test_line_hatch_swatches_hold_their_lines(
    hatch: str, size: tuple[float, float], at_least: int
) -> None:
    hatch_patterns.install()
    assert len(_distinct_lines(hatch, *size)) >= at_least


def test_line_hatch_swatch_keeps_the_figure_spacing_and_is_centred() -> None:
    hatch_patterns.install()
    width, height = CONSULTING_SWATCH
    lines = sorted(o for _a, o in _distinct_lines("\\\\", width, height))
    # Same spacing as the fence: 12 lines per inch along x at 45 deg.
    gaps = np.diff(lines)
    assert np.allclose(gaps, gaps[0], atol=2e-3)
    assert gaps[0] == pytest.approx(np.sqrt(0.5) / 6.0, abs=2e-3)
    # Centred: the swatch centre's offset sits mid-way between the outer lines.
    angle = np.pi * 0.75
    normal = np.array([-np.sin(angle), np.cos(angle)])
    centre = float(np.dot((width / 2.0, height / 2.0), normal))
    assert (lines[0] + lines[-1]) / 2.0 == pytest.approx(centre, abs=0.02)


def test_gravel_swatch_is_not_rephased() -> None:
    hatch_patterns.install()
    hatch_patterns.swatch_marks_path.cache_clear()
    _marks, lines = hatch_patterns.swatch_marks_path("O", *SECTION_SHEET_SWATCH)
    assert lines is not None
    # Gravel outlines keep the cell phase: every vertex is a tiled cell vertex.
    import matplotlib.hatch as mh

    cell = mh.get_path("O", 6).vertices
    frac = np.mod(np.round(lines.vertices, 5), 1.0)
    cell_frac = {tuple(v) for v in np.mod(np.round(cell, 5), 1.0)}
    assert all(tuple(v) in cell_frac for v in frac[:50])


@pytest.mark.parametrize("layout", ["section_sheet", "chart", "consulting_section"])
def test_shale_figure_legend_swatch_draws_lines(layout: str) -> None:
    geometry = compute_section_geometry(_collars(), _lith("Shale"), POINTS)
    result = render_cross_section_from_geometry(
        geometry, POINTS, render_layout=layout, close_figure=False
    )
    figure = result.retained_export_bundle["figure"]
    try:
        hatches = [
            a
            for a in figure.findobj()
            if getattr(a, "get_gid", lambda: None)() == "legend-swatch-hatch"
        ]
        assert hatches, "Shale swatch has no hatch artist"
        figure.canvas.draw()
        r = figure.canvas.get_renderer()
        drawn = [a for a in hatches if a.get_window_extent(r).width > 0]
        assert drawn
    finally:
        plt.close(figure)
