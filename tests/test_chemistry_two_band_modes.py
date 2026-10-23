"""Two-band chemistry colour modes (black / red and green / red exceedance) and
the four-parameters-per-section cap.

Both modes are threshold-style: per-parameter limits, coloured dots, a key
under the distance label, workbook ``label_color`` wins — with ONE limit per
parameter (stored as ``green_max``; ``orange_max`` set equal, so no orange band).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex

from models import MAX_CHEMISTRY_PARAMETERS, Collar, EnvironmentalReading, Lithology
from pipeline import compute_section_geometry, render_cross_section_from_geometry
from render_profiles import (
    CONSULTING_SECTION_PROFILE,
    SECTION_SHEET_PROFILE,
    THRESHOLD_STYLE_MODES,
    TWO_BAND_MODES,
    ChemistryColorMode,
    cap_chemistry_parameters,
    chemistry_parameter_cap_note,
    is_threshold_style_mode,
    is_two_band_mode,
    resolved_chemistry_limits,
    resolved_chemistry_limits_for,
    resolved_chemistry_thresholds_for,
)
from render_theme import (
    CHEMISTRY_LABEL_BLACK,
    CHEMISTRY_LABEL_GREEN,
    CHEMISTRY_LABEL_INCONCLUSIVE,
    CHEMISTRY_LABEL_ORANGE,
    CHEMISTRY_LABEL_RED,
    chemistry_label_color,
    chemistry_threshold_bands,
    chemistry_threshold_key,
)
from renderer import CrossSectionRenderer
from renderer_chemistry import _shown_threshold_bands, _threshold_key_lines
from section_build_request import SectionBuildRequest
from tests.conftest import run_pipeline
from ui_helpers import export_metadata_payload

ROOT = Path(__file__).resolve().parents[1]
HOLES = ("BH-01", "BH-02", "BH-03")
LOW = {"black_red": CHEMISTRY_LABEL_BLACK, "green_red": CHEMISTRY_LABEL_GREEN}
NAN = float("nan")


# --- mode registry ----------------------------------------------------------------


def test_modes_are_registered_as_threshold_style() -> None:
    from typing import get_args

    assert set(get_args(ChemistryColorMode)) == {"black", "red", "black_red", "green_red", "threshold"}
    assert TWO_BAND_MODES == {"black_red", "green_red"}
    assert THRESHOLD_STYLE_MODES == {"black_red", "green_red", "threshold"}
    for mode in TWO_BAND_MODES:
        assert is_two_band_mode(mode) and is_threshold_style_mode(mode)
    assert is_threshold_style_mode("threshold") and not is_two_band_mode("threshold")
    assert not is_threshold_style_mode("black") and not is_threshold_style_mode(None)
    # The request DTO validates the literal.
    for mode in ("black_red", "green_red"):
        assert SectionBuildRequest(transect_points=((0.0, 0.0), (1.0, 0.0)), chemistry_color_mode=mode)
    with pytest.raises(ValueError):
        SectionBuildRequest(transect_points=((0.0, 0.0), (1.0, 0.0)), chemistry_color_mode="blue_red")


# --- colour rules -----------------------------------------------------------------


@pytest.mark.parametrize("mode", sorted(TWO_BAND_MODES))
def test_two_band_colour_rules(mode: str) -> None:
    low = LOW[mode]
    kw = {"green_max": 100.0, "yellow_max": 100.0}
    assert chemistry_label_color(50.0, mode, **kw) == low
    assert chemistry_label_color(100.0, mode, **kw) == low  # at the limit: not an exceedance
    assert chemistry_label_color(100.01, mode, **kw) == CHEMISTRY_LABEL_RED
    assert chemistry_label_color(999.0, mode, **kw) == CHEMISTRY_LABEL_RED
    # Never orange, whatever orange limit a caller passes.
    assert chemistry_label_color(150.0, mode, green_max=100.0, yellow_max=250.0) == CHEMISTRY_LABEL_RED
    # Non-detects: RDL within the limit -> at-or-below colour; above -> grey.
    assert chemistry_label_color(5.0, mode, detected=False, **kw) == low
    assert chemistry_label_color(100.0, mode, detected=False, **kw) == low
    assert chemistry_label_color(500.0, mode, detected=False, **kw) == CHEMISTRY_LABEL_INCONCLUSIVE
    # No limit set: black, detected or not.
    assert chemistry_label_color(500.0, mode, green_max=NAN, yellow_max=NAN) == CHEMISTRY_LABEL_BLACK
    assert chemistry_label_color(500.0, mode) == CHEMISTRY_LABEL_BLACK
    assert chemistry_label_color(500.0, mode, detected=False, green_max=NAN) == CHEMISTRY_LABEL_BLACK


def test_threshold_and_fixed_modes_are_unchanged() -> None:
    kw = {"green_max": 100.0, "yellow_max": 250.0}
    assert chemistry_label_color(50.0, "threshold", **kw) == CHEMISTRY_LABEL_GREEN
    assert chemistry_label_color(150.0, "threshold", **kw) == CHEMISTRY_LABEL_ORANGE
    assert chemistry_label_color(300.0, "threshold", **kw) == CHEMISTRY_LABEL_RED
    assert chemistry_label_color(150.0, "threshold", detected=False, **kw) == CHEMISTRY_LABEL_INCONCLUSIVE
    assert chemistry_label_color(150.0, "threshold", green_max=100.0, yellow_max=NAN) == CHEMISTRY_LABEL_BLACK
    assert chemistry_label_color(999.0, "black", **kw) == CHEMISTRY_LABEL_BLACK
    assert chemistry_label_color(999.0, "red", **kw) == CHEMISTRY_LABEL_RED


# --- limits resolver --------------------------------------------------------------


@pytest.mark.parametrize("mode", sorted(TWO_BAND_MODES))
def test_two_band_limits_are_one_limit_stored_as_an_equal_pair(mode: str) -> None:
    assert resolved_chemistry_limits(100.0, 250.0, mode) == (100.0, 100.0)  # orange ignored
    assert resolved_chemistry_limits(100.0, None, mode) == (100.0, 100.0)
    green, orange = resolved_chemistry_limits(None, 250.0, mode)
    assert math.isnan(green) and math.isnan(orange)  # orange alone is not a limit here
    thresholds = {"Chloride": (100.0, 250.0), "Benzene": (None, 5.0)}
    assert resolved_chemistry_limits_for("chloride", thresholds, 7.0, 9.0, mode=mode) == (100.0, 100.0)
    # No own green limit -> sheet-wide green limit.
    assert resolved_chemistry_limits_for("Benzene", thresholds, 7.0, 9.0, mode=mode) == (7.0, 7.0)
    assert resolved_chemistry_limits_for("TPH", thresholds, 7.0, 9.0, mode=mode) == (7.0, 7.0)
    # Nothing at all -> no limit set (no Configure default sneaks in).
    green, orange = resolved_chemistry_limits_for("TPH", thresholds, mode=mode)
    assert math.isnan(green) and math.isnan(orange)
    # Threshold mode keeps the old resolver, bit for bit.
    assert resolved_chemistry_limits_for("Benzene", thresholds, 7.0, 9.0, mode="threshold") == (
        resolved_chemistry_thresholds_for("Benzene", thresholds, 7.0, 9.0)
    )
    assert resolved_chemistry_limits(100.0, 250.0, "threshold") == (100.0, 250.0)


# --- bands / key text -------------------------------------------------------------


def test_two_band_bands_and_key_text() -> None:
    assert chemistry_threshold_bands(100.0, 100.0, "black_red") == (
        ("black", CHEMISTRY_LABEL_BLACK, "≤ 100"),
        ("red", CHEMISTRY_LABEL_RED, "> 100"),
    )
    assert chemistry_threshold_bands(0.5, 0.5, "green_red") == (
        ("green", CHEMISTRY_LABEL_GREEN, "≤ 0.5"),
        ("red", CHEMISTRY_LABEL_RED, "> 0.5"),
    )
    # The orange limit is ignored even when a caller passes a different one.
    assert [name for name, _h, _s in chemistry_threshold_bands(100.0, 250.0, "green_red")] == ["green", "red"]
    assert [name for name, _h, _s in _shown_threshold_bands(100.0, 100.0, "black_red")] == ["black", "red"]
    assert chemistry_threshold_bands(NAN, NAN, "black_red") == (("black", CHEMISTRY_LABEL_BLACK, "(no limit set)"),)
    assert chemistry_threshold_key(100.0, 100.0, "mg/L", "black_red") == "black ≤ 100 · red > 100 mg/L"
    assert chemistry_threshold_key(100.0, 100.0, "mg/L", "green_red") == "green ≤ 100 · red > 100 mg/L"
    # Threshold mode: unchanged (default mode argument).
    assert chemistry_threshold_key(100.0, 300.0, "mg/L") == "green ≤ 100 · orange 100–300 · red > 300 mg/L"
    assert [name for name, _h, _s in chemistry_threshold_bands(100.0, 250.0)] == ["green", "orange", "red"]
    # Per-parameter key lines: one parameter bare, several prefixed, no-limit wording.
    entries = [("Chloride", "mg/kg", 100.0, 100.0)]
    assert _threshold_key_lines(entries, mode="black_red") == ("black ≤ 100 · red > 100 mg/kg",)
    entries = [("Chloride", "mg/kg", 100.0, 100.0), ("Benzene", "mg/kg", NAN, NAN)]
    assert _threshold_key_lines(entries, {"Chloride"}, mode="green_red") == (
        "Chloride: green ≤ 100 · red > 100 · grey ND (RDL > 100) mg/kg",
        "Benzene (mg/kg): no limit set — black",
    )


# --- renderer ---------------------------------------------------------------------


def _reading(value: float, *, hole: str, parameter: str = "Chloride", **extra) -> EnvironmentalReading:
    return EnvironmentalReading(hole_id=hole, parameter=parameter, value=value, depth=2.0, unit="mg/L", **extra)


def _section():
    collars = [
        Collar(hole_id=h, easting=i * 30.0, northing=0.0, elevation=100.0, total_depth=12.0)
        for i, h in enumerate(HOLES)
    ]
    lith = [Lithology(hole_id=h, from_depth=0.0, to_depth=12.0, lithology_code="Sandy Clay") for h in HOLES]
    projected, polygons, _ = run_pipeline(collars, lith, [(0.0, 0.0), (60.0, 0.0)])
    return collars, lith, projected, polygons


def _render(readings, mode, *, parameters=("Chloride",), profile=SECTION_SHEET_PROFILE, **profile_updates):
    _collars, _lith, projected, polygons = _section()
    renderer = CrossSectionRenderer(
        show_legend=True,
        render_profile=profile.model_copy(
            update={
                "show_parameter_markers": True,
                "show_parameter_labels": True,
                "chemistry_color_mode": mode,
                **profile_updates,
            }
        ),
        environmental_readings=readings,
        environmental_parameters=tuple(parameters),
    )
    figure = renderer.render(polygons, projected, collar_depths={h: 12.0 for h in HOLES})
    return renderer, figure


def _labels(renderer) -> dict[str, str]:
    return {ann.get_text(): ann.get_color().lower() for kind, ann, _c in renderer._water_labels if kind == "chem"}


def _dot_colours(figure) -> set[str]:
    return {
        to_hex(c).upper()
        for ax in figure.axes
        for coll in ax.collections
        if hasattr(coll, "get_offsets") and len(coll.get_offsets()) > 1
        for c in coll.get_facecolors()
    }


READINGS = [
    _reading(50.0, hole="BH-01"),
    _reading(500.0, hole="BH-02"),
    _reading(5.0, hole="BH-03", detected=False),
]


@pytest.mark.parametrize("mode", sorted(TWO_BAND_MODES))
@pytest.mark.parametrize("profile", [SECTION_SHEET_PROFILE, CONSULTING_SECTION_PROFILE])
def test_figure_dots_labels_and_key_agree(mode: str, profile) -> None:
    low_name = "black" if mode == "black_red" else "green"
    renderer, figure = _render(READINGS, mode, profile=profile, chemistry_thresholds={"Chloride": (100.0, 100.0)})
    try:
        labels = _labels(renderer)
        assert labels == {"50": LOW[mode].lower(), "500": CHEMISTRY_LABEL_RED.lower(), "<5": LOW[mode].lower()}
        assert {LOW[mode].upper(), CHEMISTRY_LABEL_RED.upper()} <= _dot_colours(figure)
        assert CHEMISTRY_LABEL_ORANGE.upper() not in _dot_colours(figure)
        assert renderer.chemistry_threshold_key_text == f"{low_name} ≤ 100 · red > 100 mg/L"
        assert renderer.chemistry_rdl_notes == []
        # The legend sample shows the two band colours.
        (entry,) = renderer.parameter_series_legend
        assert entry["threshold_colors"] == (LOW[mode], CHEMISTRY_LABEL_RED)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("mode", sorted(TWO_BAND_MODES))
def test_sheet_wide_limit_inconclusive_non_detect_and_no_limit(mode: str) -> None:
    low_name = "black" if mode == "black_red" else "green"
    readings = READINGS + [
        _reading(0.4, hole="BH-01", parameter="Benzene"),
        _reading(2.0, hole="BH-02", parameter="Benzene", detected=False),
    ]
    # Sheet-wide limit 100 covers Chloride; Benzene has its own 1.0 (orange ignored).
    renderer, figure = _render(
        readings,
        mode,
        parameters=("Chloride", "Benzene"),
        chemistry_threshold_green_max=100.0,
        chemistry_threshold_yellow_max=100.0,
        chemistry_thresholds={"Benzene": (1.0, 50.0)},
    )
    try:
        labels = _labels(renderer)
        assert labels["<2"] == CHEMISTRY_LABEL_INCONCLUSIVE.lower()
        assert labels["0.4"] == LOW[mode].lower()
        assert labels["500"] == CHEMISTRY_LABEL_RED.lower()
        assert renderer.chemistry_threshold_key_lines == (
            f"Benzene: {low_name} ≤ 1 · red > 1 · grey ND (RDL > 1) mg/L",
            f"Chloride: {low_name} ≤ 100 · red > 100 mg/L",
        )
        assert renderer.chemistry_rdl_notes == [
            "BH-02 Benzene at 2 m: detection limit 2 mg/L exceeds guideline 1 mg/L — result inconclusive"
        ]
    finally:
        plt.close(figure)
    # No limit anywhere: black values, "no limit set" key.
    renderer, figure = _render(READINGS, mode)
    try:
        assert set(_labels(renderer).values()) == {CHEMISTRY_LABEL_BLACK.lower()}
        assert renderer.chemistry_threshold_key_text == "Chloride (mg/L): no limit set — black"
    finally:
        plt.close(figure)


def test_workbook_label_color_still_wins() -> None:
    readings = [_reading(500.0, hole="BH-02", label_color="green"), _reading(50.0, hole="BH-01")]
    renderer, figure = _render(readings, "black_red", chemistry_thresholds={"Chloride": (100.0, 100.0)})
    try:
        labels = _labels(renderer)
        assert labels["500"] == CHEMISTRY_LABEL_GREEN.lower()
        assert labels["50"] == CHEMISTRY_LABEL_BLACK.lower()
    finally:
        plt.close(figure)


def _render_from_geometry(collars, lith, **kwargs):
    points = [(0.0, 0.0), (60.0, 0.0)]
    geometry = compute_section_geometry(collars, lith, points)
    return render_cross_section_from_geometry(geometry, points, close_figure=False, **kwargs)


def test_pipeline_renders_both_modes_with_one_limit_and_ignores_orange() -> None:
    collars, lith, _projected, _polygons = _section()
    for mode in sorted(TWO_BAND_MODES):
        result = _render_from_geometry(
            collars,
            lith,
            environmental_readings=READINGS,
            environmental_parameters=("Chloride",),
            show_parameter_labels=True,
            chemistry_color_mode=mode,
            chemistry_threshold_green_max=100.0,
            chemistry_threshold_yellow_max=250.0,
        )
        renderer = result.retained_export_bundle["renderer"]
        try:
            low_name = "black" if mode == "black_red" else "green"
            assert renderer.chemistry_threshold_key_text == f"{low_name} ≤ 100 · red > 100 mg/L"
            assert renderer.profile.chemistry_threshold_yellow_max == 100.0
            assert b"<svg" in result.svg_bytes[:200]
        finally:
            plt.close(result.retained_export_bundle["figure"])


# --- four-parameter cap -----------------------------------------------------------


def test_cap_helper_keeps_configure_order() -> None:
    assert MAX_CHEMISTRY_PARAMETERS == 4
    plotted, dropped = cap_chemistry_parameters(["Zinc", " Chloride ", "Benzene", "Zinc", "Lead", "TPH", "Arsenic"])
    assert plotted == ("Zinc", "Chloride", "Benzene", "Lead")
    assert dropped == ("TPH", "Arsenic")
    assert cap_chemistry_parameters(None) == ((), ())
    note = chemistry_parameter_cap_note(plotted, dropped)
    assert "Up to 4 chemistry parameters" in note and "not plotted: TPH, Arsenic" in note


def test_engine_plots_first_four_and_notes_the_rest() -> None:
    names = ["P1", "P2", "P3", "P4", "P5", "P6"]
    readings = [_reading(10.0 * (i + 1), hole=HOLES[i % 3], parameter=name) for i, name in enumerate(names)]
    renderer, figure = _render(readings, "black", parameters=names)
    try:
        assert {entry["parameter"] for entry in renderer.parameter_series_legend} == {"P1", "P2", "P3", "P4"}
        assert renderer.chemistry_cap_notes == [chemistry_parameter_cap_note(names[:4], names[4:])]
    finally:
        plt.close(figure)
    collars, lith, _projected, _polygons = _section()
    result = _render_from_geometry(
        collars,
        lith,
        environmental_readings=readings,
        environmental_parameters=names,
        show_parameter_labels=True,
    )
    try:
        renderer = result.retained_export_bundle["renderer"]
        assert tuple(renderer.environmental_parameters) == ("P1", "P2", "P3", "P4")
        assert [n for n in result.overlap_warnings if "not plotted: P5, P6" in n]
        assert [n for n in result.qa_notes if "not plotted: P5, P6" in n]
        # The renderer saw a capped list, so it added no second note.
        assert renderer.chemistry_cap_notes == []
    finally:
        plt.close(result.retained_export_bundle["figure"])


# --- metadata ---------------------------------------------------------------------


def test_metadata_records_mode_and_limits() -> None:
    def payload(mode, limits):
        return export_metadata_payload(
            section_title="A-A'",
            preset_label=None,
            vertical_exaggeration=2.0,
            hole_count=3,
            transect_label="A-A'",
            overlap_warnings=(),
            chemistry_color_mode=mode,
            chemistry_thresholds=limits,
        )

    two = payload("black_red", {"Chloride": (100.0, 100.0), "Benzene": (0.5, 0.5)})
    assert two["chemistry_label_colour_mode"] == "black_red"
    assert two["chemistry_exceedance_limits"] == {"Chloride": 100.0, "Benzene": 0.5}
    assert two["chemistry_label_colour_rule"] == "black at or below the limit, red above"
    assert "chemistry_thresholds" not in two
    assert payload("green_red", {"Chloride": (100.0, 100.0)})["chemistry_label_colour_rule"].startswith("green")
    three = payload("threshold", {"Chloride": (100.0, 250.0)})
    assert three["chemistry_thresholds"] == {"Chloride": {"green_max": 100.0, "orange_max": 250.0}}
    assert "chemistry_exceedance_limits" not in three


# --- Configure (AppTest) ----------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "mode"),
    [("Black / red (exceedance)", "black_red"), ("Green / red (exceedance)", "green_red")],
)
def test_configure_offers_the_modes_and_generates(label: str, mode: str) -> None:
    from streamlit.testing.v1 import AppTest

    from workbook_template import build_input_template_bytes

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    at.file_uploader[0].upload("t.xlsx", build_input_template_bytes()).run()
    assert not at.exception
    radio = at.radio(key="chemistry_color_mode_radio")
    assert radio.options == [
        "All black",
        "All red (client P2 style)",
        "Black / red (exceedance)",
        "Green / red (exceedance)",
        "Green / orange / red thresholds",
    ]
    picker = at.multiselect(key="environmental_parameters_multiselect")
    assert picker.max_selections == MAX_CHEMISTRY_PARAMETERS
    assert len(picker.value) <= MAX_CHEMISTRY_PARAMETERS
    radio.set_value(label).run()
    assert not at.exception
    limit = at.number_input(key="chemistry_exceedance_limit")
    assert limit.label == "Exceedance limit ≤" and limit.value == 100.0
    # One "Exceedance ≤" column per parameter, no Orange ≤ inputs.
    labels = [n.label for n in at.number_input]
    assert any(lbl.startswith("Exceedance ≤ — ") for lbl in labels)
    assert not any(lbl.startswith("Orange ≤") for lbl in labels)
    hole_ids = list(at.session_state["hole_ids"])
    at.session_state["hole_sequence_multiselect"] = hole_ids[: min(4, len(hole_ids))]
    for box in at.checkbox:
        if "Generate even if data checks found warnings" in (box.label or ""):
            box.set_value(True)
            break
    at.run()
    [btn for btn in at.button if btn.label == "Generate section"][0].click().run()
    assert not at.exception
    assert at.session_state["svg_bytes"]
    assert at.radio(key="chemistry_color_mode_radio").value == label
