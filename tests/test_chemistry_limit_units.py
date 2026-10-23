"""Workbook threshold limits and units.

1. Limits typed on a row in a minority unit are converted (with the row's
   value) to the parameter's main unit, both for the Configure / pipeline map
   (``chemistry_thresholds_from_readings``) and on the figure's harmonised
   copies (``harmonize_units``); a row whose unit cannot be converted has its
   limits ignored with a Validate warning and a figure QA note.
2. Two-band modes fall back own green -> own orange -> sheet green -> sheet
   orange, and Configure seeds the same limit the engine resolves.
3. NaN ("no limit set") is never an inverted pair, and cannot enter the
   sheet-wide limits through ``SectionBuildRequest``.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import pytest

from ai_quality import chemistry_threshold_inversion_warnings, summarize_environmental_readings
from chemistry_qc import (
    harmonize_units,
    ignored_limit_warnings,
    mixed_unit_warnings,
    workbook_threshold_limits,
)
from models import Collar, EnvironmentalReading, Lithology
from pipeline import (
    build_cross_section,
    compute_section_geometry,
    render_cross_section_from_geometry,
)
from render_profiles import (
    TWO_BAND_MODES,
    chemistry_thresholds_from_readings,
    orange_only_limit_notes,
    resolved_chemistry_limits,
    resolved_chemistry_limits_for,
    resolved_chemistry_thresholds_for,
    two_band_exceedance_limit,
)
from render_theme import CHEMISTRY_LABEL_GREEN, CHEMISTRY_LABEL_ORANGE
from section_build_request import SectionBuildRequest

HOLES = ("MW1", "MW2", "MW3")
POINTS = [(0.0, 0.0), (60.0, 0.0)]
NAN = float("nan")


def _reading(hole: str, value: float, unit: str = "mg/L", parameter: str = "Chloride", **extra):
    return EnvironmentalReading(
        hole_id=hole, parameter=parameter, value=value, depth=2.0, unit=unit, **extra
    )


def _collars():
    return [
        Collar(hole_id=h, easting=i * 30.0, northing=0.0, elevation=100.0, total_depth=12.0)
        for i, h in enumerate(HOLES)
    ]


def _lithology():
    return [
        Lithology(hole_id=h, from_depth=0.0, to_depth=12.0, lithology_code="Sandy Clay")
        for h in HOLES
    ]


# The review reproducer: the limit row is the one minority-unit row.
REPRODUCER = [
    _reading("MW1", 150000.0, unit="µg/L", green_max=100000.0, orange_max=250000.0),
    _reading("MW2", 50.0),
    _reading("MW3", 60.0),
]


def _chem_labels(renderer) -> dict[str, str]:
    return {
        ann.get_text(): ann.get_color().lower()
        for kind, ann, _c in renderer._water_labels
        if kind == "chem"
    }


# --- 1. limits follow the row's unit ------------------------------------------------


def test_minority_unit_limit_row_is_converted_to_the_main_unit() -> None:
    assert chemistry_thresholds_from_readings(REPRODUCER) == {"Chloride": (100.0, 250.0)}
    result = workbook_threshold_limits(REPRODUCER)
    assert result.thresholds == {"Chloride": (100.0, 250.0)} and result.warnings == []
    # First row giving each limit wins, after conversion: a later mg/L row
    # with a green limit does not override the converted one.
    later = REPRODUCER + [_reading("MW3", 70.0, green_max=999.0, orange_max=None)]
    assert chemistry_thresholds_from_readings(later) == {"Chloride": (100.0, 250.0)}
    # Blank-unit rows are read in the main unit; same-unit rows untouched.
    plain = [_reading("MW1", 1.0, unit="", green_max=3.0), _reading("MW2", 2.0), _reading("MW3", 2.5)]
    assert chemistry_thresholds_from_readings(plain) == {"Chloride": (3.0, None)}
    # ppm joins the aqueous family when the other units are all aqueous.
    ppm = [_reading("MW1", 0.5, unit="ppm", green_max=1.0), _reading("MW2", 2.0), _reading("MW3", 2.5)]
    assert chemistry_thresholds_from_readings(ppm) == {"Chloride": (1.0, None)}


def test_reproducer_colours_the_converted_value_orange() -> None:
    thresholds = chemistry_thresholds_from_readings(REPRODUCER)
    geometry = compute_section_geometry(_collars(), _lithology(), POINTS)
    result = render_cross_section_from_geometry(
        geometry,
        POINTS,
        environmental_readings=tuple(REPRODUCER),
        environmental_parameters=("Chloride",),
        chemistry_color_mode="threshold",
        chemistry_thresholds=thresholds,
        export_formats=frozenset({"svg"}),
        close_figure=False,
    )
    bundle = result.retained_export_bundle
    try:
        labels = _chem_labels(bundle["renderer"])
        # 150000 ug/L is 150 mg/L: above green 100, within orange 250.
        assert labels["150"] == CHEMISTRY_LABEL_ORANGE.lower()
        assert labels["50"] == labels["60"] == CHEMISTRY_LABEL_GREEN.lower()
    finally:
        plt.close(bundle["figure"])
    # The public entry point reports the conversion on the figure's QA notes.
    built = build_cross_section(
        _collars(),
        _lithology(),
        POINTS,
        environmental_readings=tuple(REPRODUCER),
        environmental_parameters=("Chloride",),
        chemistry_color_mode="threshold",
        chemistry_thresholds=thresholds,
        export_formats=frozenset({"svg"}),
    )
    assert any(
        note.startswith("MW1 Chloride 150000 µg/L shown as 150 mg/L") for note in built.qa_notes
    )
    assert CHEMISTRY_LABEL_ORANGE.lower() in built.svg_bytes.decode("utf-8").lower()


def test_harmonize_units_converts_the_copy_limits_and_drops_unconvertible_ones() -> None:
    readings = REPRODUCER + [_reading("MW3", 7.0, unit="µg/kg", green_max=1.0, orange_max=2.0)]
    harmonized = harmonize_units(readings)
    converted = harmonized.readings[0]
    assert (converted.value, converted.unit) == (150.0, "mg/L")
    assert (converted.green_max, converted.orange_max) == (100.0, 250.0)
    # mg/L rows (no limits) come back as the same objects.
    assert harmonized.readings[1] is readings[1] and harmonized.readings[2] is readings[2]
    # The soil row keeps its value and unit but loses its limits, with a note.
    stuck = harmonized.readings[3]
    assert (stuck.value, stuck.unit, stuck.green_max, stuck.orange_max) == (7.0, "µg/kg", None, None)
    assert harmonized.is_unconverted(stuck)
    assert "Chloride limits on MW3 are in µg/kg, not convertible to mg/L — ignored" in harmonized.notes
    # Inputs untouched.
    assert readings[0].green_max == 100000.0 and readings[3].green_max == 1.0


def test_unconvertible_limit_row_is_ignored_with_a_validate_warning() -> None:
    readings = [
        _reading("MW1", 7.0, unit="µg/kg", green_max=1.0, orange_max=2.0),
        _reading("MW2", 50.0),
        _reading("MW3", 60.0),
    ]
    expected = "Chloride limits on MW1 are in µg/kg, not convertible to mg/L — ignored"
    assert chemistry_thresholds_from_readings(readings) == {}
    assert ignored_limit_warnings(readings) == [expected]
    (mixed,) = mixed_unit_warnings(readings)
    assert "µg/kg cannot be converted to mg/L" in mixed
    summary = summarize_environmental_readings(_collars(), readings, HOLES)
    assert expected in summary.warnings
    # A second row of the same parameter in a convertible unit still seeds it.
    readings.append(_reading("MW2", 55000.0, unit="µg/L", green_max=100000.0))
    assert chemistry_thresholds_from_readings(readings) == {"Chloride": (100.0, None)}
    assert ignored_limit_warnings(readings) == [expected]


# --- 2. two-band fallback precedence ---------------------------------------------


@pytest.mark.parametrize(
    ("own", "sheet", "expected", "source"),
    [
        ((100.0, 250.0), (7.0, 9.0), 100.0, "own_green"),
        ((100.0, None), (7.0, 9.0), 100.0, "own_green"),
        ((None, 5.0), (7.0, 9.0), 5.0, "own_orange"),
        ((None, 5.0), (None, None), 5.0, "own_orange"),
        ((None, None), (7.0, 9.0), 7.0, "sheet_green"),
        (None, (7.0, 9.0), 7.0, "sheet_green"),
        (None, (None, 9.0), 9.0, "sheet_orange"),
        ((None, None), (None, 9.0), 9.0, "sheet_orange"),
        (None, (None, None), None, None),
        (None, (NAN, NAN), None, None),
        ((NAN, NAN), (NAN, 9.0), 9.0, "sheet_orange"),
    ],
)
@pytest.mark.parametrize("mode", sorted(TWO_BAND_MODES))
def test_two_band_fallback_precedence(mode, own, sheet, expected, source) -> None:
    thresholds = {} if own is None else {"Benzene": own}
    assert two_band_exceedance_limit("benzene", thresholds, *sheet) == (expected, source)
    green, orange = resolved_chemistry_limits_for("Benzene", thresholds, *sheet, mode=mode)
    if expected is None:
        assert math.isnan(green) and math.isnan(orange)
    else:
        assert (green, orange) == (expected, expected)


def test_two_band_sheet_wide_orange_only_pair_is_a_limit() -> None:
    for mode in TWO_BAND_MODES:
        assert resolved_chemistry_limits(None, 250.0, mode) == (250.0, 250.0)
        assert resolved_chemistry_limits(100.0, 250.0, mode) == (100.0, 100.0)
    # Threshold mode is untouched: an orange-only entry caps green at it.
    assert resolved_chemistry_thresholds_for("Benzene", {"Benzene": (None, 5.0)}, 7.0, 9.0) == (5.0, 5.0)


def test_two_band_figure_uses_an_orange_only_entry() -> None:
    readings = [_reading("MW1", 3.0), _reading("MW2", 8.0), _reading("MW3", 5.0)]
    geometry = compute_section_geometry(_collars(), _lithology(), POINTS)
    result = render_cross_section_from_geometry(
        geometry,
        POINTS,
        environmental_readings=tuple(readings),
        environmental_parameters=("Chloride",),
        chemistry_color_mode="green_red",
        chemistry_thresholds={"Chloride": (None, 5.0)},
        export_formats=frozenset({"svg"}),
        close_figure=False,
    )
    bundle = result.retained_export_bundle
    try:
        renderer = bundle["renderer"]
        labels = _chem_labels(renderer)
        assert labels["3"] == labels["5"] == CHEMISTRY_LABEL_GREEN.lower()
        assert labels["8"] != CHEMISTRY_LABEL_GREEN.lower()
        assert renderer.chemistry_threshold_key_text == "green ≤ 5 · red > 5 mg/L"
    finally:
        plt.close(bundle["figure"])


def test_configure_seeding_matches_the_engine_in_two_band_modes() -> None:
    from app_configure import seeded_parameter_thresholds

    workbook = {"Benzene": (None, 5.0), "Chloride": (None, 300.0)}
    for mode in TWO_BAND_MODES:
        # Orange-only workbook row: the engine's exceedance limit, not 100.
        assert seeded_parameter_thresholds("Benzene", workbook, default_green=7.0, default_orange=9.0, mode=mode) == (5.0, 5.0)
        assert seeded_parameter_thresholds("Chloride", workbook, default_green=7.0, default_orange=9.0, mode=mode) == (300.0, 300.0)
        # No own entry: sheet-wide green, then sheet-wide orange.
        assert seeded_parameter_thresholds("TPH", workbook, default_green=7.0, default_orange=9.0, mode=mode) == (7.0, 7.0)
        assert seeded_parameter_thresholds("TPH", workbook, default_green=None, default_orange=9.0, mode=mode) == (9.0, 9.0)
        # Nothing anywhere: the editor row needs a number (Configure default).
        assert seeded_parameter_thresholds("TPH", {}, default_green=None, default_orange=None, mode=mode) == (100.0, 100.0)
        assert seeded_parameter_thresholds("Chloride", {}, default_green=7.0, default_orange=9.0, mode=mode) == (100.0, 100.0)
        for parameter, limit in (("Benzene", 5.0), ("TPH", 7.0)):
            engine = resolved_chemistry_limits_for(parameter, workbook, 7.0, 9.0, mode=mode)
            assert engine == (limit, limit)
    # Threshold-mode seeding is unchanged.
    assert seeded_parameter_thresholds("Benzene", {"Benzene": (0.5, 5.0)}, default_green=7.0, default_orange=9.0) == (0.5, 5.0)
    assert seeded_parameter_thresholds("TPH", {}, default_green=None, default_orange=None) == (100.0, 250.0)


def test_orange_only_limits_are_noted() -> None:
    thresholds = {"Benzene": (None, 5.0), "Chloride": (100.0, 250.0)}
    (note,) = orange_only_limit_notes(thresholds)
    assert note.startswith("Benzene: only an orange limit (5) is set")
    assert "exceedance limit" in note
    assert orange_only_limit_notes({"Chloride": (100.0, 250.0)}) == []
    assert orange_only_limit_notes({"Chloride": (NAN, 7.0)}) != []
    # Sheet-wide orange-only pair: named for the plotted parameters it covers.
    notes = orange_only_limit_notes(thresholds, green_max=None, yellow_max=9.0, parameters=("Chloride", "TPH"))
    assert notes == [
        "TPH: no own limit — the sheet-wide pair has only an orange limit (9), which the Black / red "
        "and Green / red exceedance modes use as the exceedance limit (red above 9)"
    ]
    # Validate carries the per-parameter note as an Info line.
    readings = [_reading("MW1", 3.0, parameter="Benzene", orange_max=5.0), _reading("MW2", 4.0, parameter="Benzene")]
    summary = summarize_environmental_readings(_collars(), readings, HOLES)
    assert any(w.startswith("Info: Benzene: only an orange limit (5) is set") for w in summary.warnings)


# --- 3. NaN is never an inverted pair ----------------------------------------------


def test_nan_limits_are_not_inverted() -> None:
    assert chemistry_threshold_inversion_warnings({"Cl": (NAN, NAN)}) == []
    assert chemistry_threshold_inversion_warnings({"Cl": (NAN, 5.0), "Na": (5.0, NAN)}) == []
    assert chemistry_threshold_inversion_warnings({}, green_max=NAN, orange_max=NAN) == []
    assert chemistry_threshold_inversion_warnings({}, green_max=50.0, orange_max=10.0) != []


@pytest.mark.parametrize("field", ["chemistry_threshold_green_max", "chemistry_threshold_yellow_max"])
@pytest.mark.parametrize("bad", [NAN, float("inf"), float("-inf")])
def test_request_rejects_nan_and_inf_sheet_limits(field: str, bad: float) -> None:
    with pytest.raises(ValueError):
        SectionBuildRequest(transect_points=((0.0, 0.0), (1.0, 0.0)), **{field: bad})
    request = SectionBuildRequest(transect_points=((0.0, 0.0), (1.0, 0.0)), **{field: 5.0})
    assert getattr(request, field) == 5.0
    assert getattr(SectionBuildRequest(transect_points=((0.0, 0.0), (1.0, 0.0))), field) is None
