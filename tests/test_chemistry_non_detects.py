"""Lab non-detects (< reporting detection limit) on the Environmental sheet.

A non-detect keeps its detection limit (RDL) in ``value`` with
``detected=False``: it prints as "<5", never a bare "5"; in threshold mode it
is green when the limit is at or below the guideline (green limit) and grey
"inconclusive" when the limit is above it (never orange / red), with a QA note.
"""

from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from ai_quality import summarize_environmental_readings
from chemistry_qc import (
    mixed_unit_parameters,
    mixed_unit_warnings,
    rdl_above_guideline_notes,
    rdl_exceeds_guideline,
    unit_key,
)
from models import Collar, EnvironmentalReading, Lithology
from render_profiles import SECTION_SHEET_PROFILE
from render_theme import (
    CHEMISTRY_LABEL_BLACK,
    CHEMISTRY_LABEL_GREEN,
    CHEMISTRY_LABEL_INCONCLUSIVE,
    CHEMISTRY_LABEL_ORANGE,
    CHEMISTRY_LABEL_RED,
    chemistry_label_color,
)
from renderer import CrossSectionRenderer
from tests.conftest import run_pipeline

HOLES = ("BH-01", "BH-02", "BH-03")


def _reading(value: float, *, detected: bool = True, hole: str = "BH-01", **extra) -> EnvironmentalReading:
    payload = {"hole_id": hole, "parameter": "Benzene", "value": value, "depth": 2.0, "unit": "µg/L"}
    payload.update(extra)
    return EnvironmentalReading(detected=detected, **payload)


def _workbook(env_rows: list[dict]) -> BytesIO:
    collars = pd.DataFrame(
        [
            {"hole_id": h, "easting": i * 30.0, "northing": 0.0, "elevation": 100.0, "total_depth": 10.0}
            for i, h in enumerate(HOLES)
        ]
    )
    lith = pd.DataFrame(
        [{"hole_id": h, "from_depth": 0.0, "to_depth": 10.0, "lithology_code": "Clay"} for h in HOLES]
    )
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        collars.to_excel(writer, sheet_name="Collars", index=False)
        lith.to_excel(writer, sheet_name="Lithology", index=False)
        pd.DataFrame(env_rows).to_excel(writer, sheet_name="Environmental", index=False)
    buffer.seek(0)
    return buffer


def _ingest(env_rows: list[dict]):
    from ingestion import ingest_workbook

    result, _ = ingest_workbook(_workbook(env_rows))
    return result


def _env_row(value, **extra) -> dict:
    row = {"hole_id": "BH-01", "parameter": "Benzene", "value": value, "depth": 2.0, "unit": "µg/L"}
    row.update(extra)
    return row


# --- parsing -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "limit"),
    [
        ("<5", 5.0),
        ("< 5", 5.0),
        ("<RDL 5", 5.0),
        ("<RDL=0.5", 0.5),
        ("<MDL 0.02", 0.02),
        ("ND (0.5)", 0.5),
        ("ND<0.5", 0.5),
        ("n.d. 0.5", 0.5),
        ("BDL 2", 2.0),
        ("<0.5 µg/L", 0.5),
        ("<1e-3", 0.001),
    ],
)
def test_non_detect_text_in_value_keeps_the_detection_limit(text: str, limit: float) -> None:
    result = _ingest([_env_row(text)])
    assert not result.errors, result.errors
    (reading,) = result.environmental_readings
    assert reading.value == pytest.approx(limit)
    assert reading.detected is False
    assert reading.unit == "µg/L"


def test_measured_numbers_stay_detected() -> None:
    result = _ingest([_env_row(12.5), _env_row("7.25", depth=3.0)])
    assert not result.errors
    assert [(r.value, r.detected) for r in result.environmental_readings] == [(12.5, True), (7.25, True)]


def test_unit_in_value_text_fills_a_blank_unit_and_must_match_a_given_one() -> None:
    result = _ingest([_env_row("<0.5 mg/L", unit=None), _env_row("<2 ug/l", depth=3.0)])
    assert not result.errors, result.errors
    assert [r.unit for r in result.environmental_readings] == ["mg/L", "µg/L"]
    bad = _ingest([_env_row("<0.5 mg/L")])
    assert not bad.environmental_readings
    assert any("mg/L" in error and "one unit" in error for error in bad.errors)


@pytest.mark.parametrize("text", ["ND", "n.d.", "not detected", "<"])
def test_bare_non_detect_needs_its_limit(text: str) -> None:
    result = _ingest([_env_row(text)])
    assert not result.environmental_readings
    (error,) = [e for e in result.errors if e.startswith("Environmental row 2")]
    assert "detection limit" in error and "<0.5" in error


@pytest.mark.parametrize("text", ["abc", "five", "<0,5", "<5 5", "12 mg"])
def test_non_numeric_junk_is_still_rejected_clearly(text: str) -> None:
    result = _ingest([_env_row(text)])
    assert not result.environmental_readings
    errors = [e for e in result.errors if e.startswith("Environmental row 2")]
    assert errors, result.errors
    assert f"'{text}'" in errors[0]


@pytest.mark.parametrize("flag", ["<", "ND", "nd", "U", "n.d.", "not detected"])
def test_qualifier_column_marks_a_numeric_value_as_non_detect(flag: str) -> None:
    result = _ingest([_env_row(5.0, qualifier=flag), _env_row(3.0, depth=3.0, qualifier=None)])
    assert not result.errors, result.errors
    assert [(r.value, r.detected) for r in result.environmental_readings] == [(5.0, False), (3.0, True)]


def test_detected_column_accepts_yes_no_and_booleans() -> None:
    result = _ingest(
        [
            _env_row(5.0, detected="no"),
            _env_row(6.0, depth=3.0, detected="yes"),
            _env_row(7.0, depth=4.0, detected=False),
        ]
    )
    assert not result.errors, result.errors
    assert [r.detected for r in result.environmental_readings] == [False, True, False]


def test_unknown_qualifier_is_kept_and_contradicting_one_is_an_error() -> None:
    # Unrecognised qualifier text no longer loses the row: the value is kept
    # (Validate warns, see tests/test_hydro_qa_fixes.py); a contradiction is
    # still a row error.
    result = _ingest([_env_row(5.0, qualifier="maybe"), _env_row("<5", depth=3.0, qualifier="=")])
    assert [(r.value, r.detected, r.qualifier) for r in result.environmental_readings] == [
        (5.0, True, "maybe")
    ]
    assert not any("qualifier = 'maybe'" in e for e in result.errors)
    assert any("qualifier says detected" in e for e in result.errors)


def test_greater_than_text_keeps_the_value_label_advice() -> None:
    result = _ingest([_env_row(">1000")])
    assert not result.environmental_readings
    assert any("value_label" in e for e in result.errors)


# --- display ---------------------------------------------------------------------------


def test_non_detect_prints_with_less_than_sign() -> None:
    reading = _reading(5.0, detected=False)
    assert reading.value_text == "<5"
    assert reading.display_label == "<5 µg/L"
    assert _reading(5.0).value_text == "5"
    # A value_label still wins.
    assert _reading(5.0, detected=False, value_label="ND (5)").display_label == "ND (5)"


# --- colour rules ---------------------------------------------------------------------


def test_threshold_colour_rules_for_non_detects() -> None:
    def colour(value, detected):
        return chemistry_label_color(value, "threshold", green_max=1.0, yellow_max=5.0, detected=detected)

    # Limit at / below the guideline: below guideline -> green.
    assert colour(0.5, False) == CHEMISTRY_LABEL_GREEN
    assert colour(1.0, False) == CHEMISTRY_LABEL_GREEN
    # Limit above the guideline: inconclusive grey, never orange / red.
    assert colour(3.0, False) == CHEMISTRY_LABEL_INCONCLUSIVE
    assert colour(50.0, False) == CHEMISTRY_LABEL_INCONCLUSIVE
    # Detected values are unchanged.
    assert colour(3.0, True) == CHEMISTRY_LABEL_ORANGE
    assert colour(50.0, True) == CHEMISTRY_LABEL_RED
    # Single-colour modes ignore detection.
    assert chemistry_label_color(50.0, "black", detected=False) == CHEMISTRY_LABEL_BLACK
    assert chemistry_label_color(50.0, "red", detected=False) == CHEMISTRY_LABEL_RED


def _section_with(readings: list[EnvironmentalReading]):
    collars = [
        Collar(hole_id=h, easting=i * 30.0, northing=0.0, elevation=100.0, total_depth=12.0)
        for i, h in enumerate(HOLES)
    ]
    lith = [Lithology(hole_id=h, from_depth=0.0, to_depth=12.0, lithology_code="Sandy Clay") for h in HOLES]
    projected, polygons, _ = run_pipeline(collars, lith, [(0.0, 0.0), (60.0, 0.0)])
    return collars, lith, projected, polygons


def _render(readings, mode="threshold", **profile_updates):
    _collars, _lith, projected, polygons = _section_with(readings)
    renderer = CrossSectionRenderer(
        show_legend=True,
        render_profile=SECTION_SHEET_PROFILE.model_copy(
            update={
                "show_parameter_markers": True,
                "show_parameter_labels": True,
                "chemistry_color_mode": mode,
                "chemistry_thresholds": {"Benzene": (1.0, 5.0)},
                **profile_updates,
            }
        ),
        environmental_readings=readings,
        environmental_parameters=("Benzene",),
    )
    figure = renderer.render(polygons, projected, collar_depths={h: 12.0 for h in HOLES})
    return renderer, figure


def _labels(renderer) -> dict[str, str]:
    return {ann.get_text(): ann.get_color().lower() for kind, ann, _c in renderer._water_labels if kind == "chem"}


SECTION_READINGS = [
    _reading(0.5, detected=False, hole="BH-01"),
    _reading(5.0, detected=False, hole="BH-02"),
    _reading(8.0, hole="BH-03"),
]


def test_figure_prints_less_than_and_greys_inconclusive_non_detects() -> None:
    renderer, figure = _render(SECTION_READINGS)
    try:
        labels = _labels(renderer)
        assert set(labels) == {"<0.5", "<5", "8"}
        assert labels["<0.5"] == CHEMISTRY_LABEL_GREEN.lower()
        assert labels["<5"] == CHEMISTRY_LABEL_INCONCLUSIVE.lower()
        assert labels["8"] == CHEMISTRY_LABEL_RED.lower()
        assert renderer.chemistry_rdl_notes == [
            "BH-02 Benzene at 2 m: detection limit 5 µg/L exceeds guideline 1 µg/L — result inconclusive"
        ]
        assert renderer.chemistry_threshold_key_text == (
            "green ≤ 1 · orange 1–5 · red > 5 · grey ND (RDL > 1) µg/L"
        )
    finally:
        plt.close(figure)


def test_workbook_label_color_still_wins_over_inconclusive_grey() -> None:
    readings = [_reading(5.0, detected=False, hole="BH-02", label_color="orange")]
    renderer, figure = _render(readings)
    try:
        assert _labels(renderer)["<5"] == CHEMISTRY_LABEL_ORANGE.lower()
        # The QA note still names the inconclusive result.
        assert len(renderer.chemistry_rdl_notes) == 1
    finally:
        plt.close(figure)


@pytest.mark.parametrize("mode", ["black", "red"])
def test_single_colour_modes_just_print_the_less_than_text(mode: str) -> None:
    renderer, figure = _render(SECTION_READINGS, mode=mode)
    try:
        labels = _labels(renderer)
        assert {"<0.5", "<5", "8"} == set(labels)
        assert len(set(labels.values())) == 1
        assert renderer.chemistry_rdl_notes == []
    finally:
        plt.close(figure)


def test_pipeline_reports_the_inconclusive_note() -> None:
    from pipeline import build_cross_section

    collars, lith, _projected, _polygons = _section_with(SECTION_READINGS)
    result = build_cross_section(
        collars,
        lith,
        [(0.0, 0.0), (60.0, 0.0)],
        environmental_readings=SECTION_READINGS,
        environmental_parameters=("Benzene",),
        chemistry_color_mode="threshold",
        chemistry_thresholds={"Benzene": (1.0, 5.0)},
        export_formats=frozenset({"svg"}),
    )
    notes = [n for n in result.qa_notes if "result inconclusive" in n]
    assert notes and notes[0].startswith("BH-02 Benzene")
    assert notes[0] in result.overlap_warnings
    assert "&lt;5" in result.svg_bytes.decode("utf-8")


# --- Validate QA -----------------------------------------------------------------------


def test_unit_key_normalises_spelling_only() -> None:
    assert unit_key("µg/L") == unit_key("ug/l") == unit_key("μg / L")
    assert unit_key("mg/L") != unit_key("µg/L")


def test_mixed_units_within_one_parameter_warn_without_converting() -> None:
    readings = [
        _reading(5.0, unit="µg/L"),
        _reading(0.004, unit="mg/L", hole="BH-02"),
        _reading(3.0, unit="ug/l", hole="BH-03"),
        EnvironmentalReading(hole_id="BH-01", parameter="Chloride", value=1, depth=1, unit="mg/L"),
    ]
    assert mixed_unit_parameters(readings) == {"Benzene": ("µg/L", "mg/L")}
    (warning,) = mixed_unit_warnings(readings)
    assert "Benzene" in warning and "µg/L, mg/L" in warning and "not converted" in warning
    # Values are untouched.
    assert [r.value for r in readings[:2]] == [5.0, 0.004]


def test_rdl_above_guideline_notes_use_explicit_limits_only() -> None:
    nd_high = _reading(5.0, detected=False, hole="BH-03")
    nd_low = _reading(0.5, detected=False)
    measured = _reading(9.0, hole="BH-02")
    assert rdl_exceeds_guideline(nd_high, 1.0) and not rdl_exceeds_guideline(nd_low, 1.0)
    assert not rdl_exceeds_guideline(measured, 1.0)
    assert not rdl_exceeds_guideline(nd_high, None)
    notes = rdl_above_guideline_notes([nd_high, nd_low, measured], {"benzene": (1.0, None)})
    assert notes == ["BH-03 Benzene at 2 m: detection limit 5 µg/L exceeds guideline 1 µg/L — result inconclusive"]
    assert rdl_above_guideline_notes([nd_high], {}) == []


def test_validate_summary_lists_rdl_and_mixed_unit_warnings() -> None:
    collars = [
        Collar(hole_id=h, easting=i * 30.0, northing=0.0, elevation=100.0, total_depth=12.0)
        for i, h in enumerate(HOLES)
    ]
    readings = [
        # Workbook guideline seeded on one row.
        _reading(0.2, hole="BH-01", green_max=1.0),
        _reading(5.0, detected=False, hole="BH-03"),
        _reading(0.003, hole="BH-02", unit="mg/L"),
    ]
    summary = summarize_environmental_readings(collars, readings, HOLES)
    assert any("BH-03 Benzene" in w and "result inconclusive" in w for w in summary.warnings)
    assert any("more than one unit" in w for w in summary.warnings)
    # Explicit thresholds (Configure) override the workbook seed.
    relaxed = summarize_environmental_readings(collars, readings, HOLES, thresholds={"Benzene": (10.0, 20.0)})
    assert not any("result inconclusive" in w for w in relaxed.warnings)


# --- workbook template round trip -----------------------------------------------------


def test_template_has_qualifier_column_and_rules() -> None:
    from workbook_template import ENVIRONMENTAL_COLUMNS, _instructions_lines

    assert ENVIRONMENTAL_COLUMNS.index("qualifier") == ENVIRONMENTAL_COLUMNS.index("value") + 1
    text = "\n".join(_instructions_lines())
    assert "Non-detects" in text and "inconclusive" in text


def test_template_sample_and_cleaned_export_round_trip_non_detects() -> None:
    from ingestion import ingest_workbook
    from workbook_template import build_input_template_bytes, export_cleaned_workbook_bytes

    template, _ = ingest_workbook(BytesIO(build_input_template_bytes()))
    sample = [r for r in template.environmental_readings if not r.detected]
    assert sample and sample[0].value == 5.0

    result = _ingest([_env_row("<5"), _env_row(2.5, depth=3.0), _env_row(0.4, depth=4.0, qualifier="ND")])
    assert not result.errors
    again, _ = ingest_workbook(BytesIO(export_cleaned_workbook_bytes(result, project_metadata={})))
    assert [(r.value, r.detected) for r in again.environmental_readings] == [
        (5.0, False),
        (2.5, True),
        (0.4, False),
    ]
