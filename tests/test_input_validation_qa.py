"""Regression tests for the adversarial input-validation QA round.

1. One mis-typed overlay depth (3200 m on a 12 m hole) no longer wrecks the
   figure silently: the reading is left off with a QA note naming it; model
   bounds reject non-finite / absurd depths and elevations.
2. Collar / Deviations inclination −90..90 (> 90 is an error, 90 warns);
   azimuth wraps to 0–360.
3. Environmental values / depths must be finite; negative concentrations warn.
4. Inverted colour bands warn; negative screen depths, orphan Deviations
   holes, row-numbered Deviations errors, absurd lithology depths.
5. Formula / Excel-error cells, TRUE/FALSE numbers, case-variant hole IDs,
   capped chemistry date notes, deterministic PDF bytes.
"""

from __future__ import annotations

import datetime
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import openpyxl
import pandas as pd
import pytest
from pydantic import ValidationError

from ai_quality import (
    analyze_parsed_data,
    chemistry_threshold_inversion_warnings,
    implausible_water_reading,
    summarize_environmental_readings,
    summarize_screen_intervals,
    summarize_water_levels,
)
from models import (
    Collar,
    DeviationReading,
    EnvironmentalReading,
    Lithology,
    ScreenInterval,
    WaterLevel,
)
from parsing import DataParser
from pipeline import (
    MAX_CHEMISTRY_DATE_NOTES,
    _latest_chemistry_samples,
    build_cross_section,
    compute_section_geometry,
    render_cross_section_from_geometry,
)

LINE = [(0.0, 0.0), (50.0, 0.0)]
SVG = frozenset({"svg"})


def _collars() -> list[Collar]:
    return [
        Collar(hole_id="MW-01", easting=0, northing=0, elevation=100, total_depth=12),
        Collar(hole_id="MW-02", easting=50, northing=0, elevation=101, total_depth=12),
    ]


def _lith() -> list[Lithology]:
    return [
        Lithology(hole_id=hole, from_depth=0, to_depth=12, lithology_code="SAND")
        for hole in ("MW-01", "MW-02")
    ]


def _figure_ylim(**kwargs) -> tuple[tuple[float, float], tuple[str, ...]]:
    """(y-limits of the section axes, QA notes) of a rendered figure."""
    from matplotlib import pyplot as plt

    geometry = compute_section_geometry(_collars(), _lith(), LINE)
    result = render_cross_section_from_geometry(
        geometry, LINE, export_formats=SVG, close_figure=False, **kwargs
    )
    figure = result.retained_export_bundle["figure"]
    try:
        axes = max(figure.axes, key=lambda ax: abs(ax.get_ylim()[1] - ax.get_ylim()[0]))
        return axes.get_ylim(), result.qa_notes
    finally:
        plt.close(figure)


# ------------------------------------------- 1. overlay readings outside the hole


def test_water_depth_in_mm_is_left_off_the_figure_with_a_qa_note() -> None:
    good_ylim, good_notes = _figure_ylim(
        water_levels=[WaterLevel(hole_id="MW-02", depth=3.0)]
    )
    ylim, notes = _figure_ylim(
        water_levels=[WaterLevel(hole_id="MW-01", depth=3200.0), WaterLevel(hole_id="MW-02", depth=3.0)]
    )
    # The axis is not stretched to −3100 m: same span as without the typo.
    assert min(ylim) == pytest.approx(min(good_ylim), abs=0.5)
    assert min(ylim) > 50
    assert "MW-01 water depth 3200 m is beyond the hole (TD 12 m) — not plotted; check units" in notes
    assert not any("MW-01" in note for note in good_notes)


def test_absurd_chemistry_depth_and_water_elevation_are_excluded_and_named() -> None:
    result = build_cross_section(
        _collars(),
        _lith(),
        LINE,
        water_levels=[WaterLevel(hole_id="MW-02", depth=0.0, elevation_masl=1e5)],
        environmental_readings=[
            EnvironmentalReading(hole_id="MW-02", parameter="Cl", value=5, depth=1e6),
            EnvironmentalReading(hole_id="MW-01", parameter="Cl", value=7, depth=2.0),
        ],
        environmental_parameters=["Cl"],
        export_formats=SVG,
    )
    joined = "\n".join(result.qa_notes)
    assert "MW-02 Cl sample depth 1e+06 m is beyond the hole (TD 12 m) — not plotted" in joined
    assert (
        "MW-02 water elevation 100000 m is 99899 m above ground (collar 101 m) — not plotted"
        in joined
    )
    # Surfaced with the other section notes the app shows after Generate.
    assert set(result.qa_notes) <= set(result.overlap_warnings)


def test_screen_far_below_td_is_excluded_from_figure_and_validate() -> None:
    screens = [ScreenInterval(hole_id="MW-01", from_depth=3.0, to_depth=900.0)]
    _ylim, notes = _figure_ylim(screen_intervals=screens)
    assert any("MW-01 screen 3–900 m is beyond the hole" in note for note in notes)
    summary = summarize_screen_intervals(_collars(), screens, ["MW-01", "MW-02"])
    assert any("not plotted; check units" in w for w in summary.warnings)
    assert summary.total_intervals == 0


def test_modest_overshoot_below_td_is_still_plotted() -> None:
    # 2 m below a 12 m hole: Validate warns, the figure keeps it.
    level = WaterLevel(hole_id="MW-01", depth=14.0)
    assert implausible_water_reading(level, collar_rl=100.0, total_depth=12.0) is None
    _ylim, notes = _figure_ylim(water_levels=[level])
    assert not any("not plotted" in note for note in notes)
    # Artesian head a few metres above ground is real.
    artesian = WaterLevel(hole_id="MW-01", depth=0.0, elevation_masl=104.0)
    assert implausible_water_reading(artesian, collar_rl=100.0, total_depth=12.0) is None


def test_validate_names_the_reading_the_figure_leaves_out() -> None:
    summary = summarize_water_levels(
        _collars(), [WaterLevel(hole_id="MW-01", depth=3200.0)], ["MW-01", "MW-02"]
    )
    assert any(
        w.startswith("MW-01 water depth 3200 m is beyond the hole") for w in summary.warnings
    )
    env = summarize_environmental_readings(
        _collars(),
        [EnvironmentalReading(hole_id="MW-01", parameter="Cl", value=1, depth=5000.0)],
        ["MW-01", "MW-02"],
    )
    assert any("MW-01 Cl sample depth 5000 m is beyond the hole" in w for w in env.warnings)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: WaterLevel(hole_id="A", depth=1e308),
        lambda: WaterLevel(hole_id="A", depth=1.0, elevation_masl=1e12),
        lambda: WaterLevel(hole_id="A", depth=float("inf")),
        lambda: ScreenInterval(hole_id="A", from_depth=0.0, to_depth=2e6),
        lambda: EnvironmentalReading(hole_id="A", parameter="Cl", value=1, depth=1e7),
        lambda: DeviationReading(hole_id="A", depth=5e6, inclination_deg=0, azimuth_deg=0),
        lambda: Lithology(hole_id="A", from_depth=0, to_depth=2e5, lithology_code="SAND"),
    ],
)
def test_model_bounds_reject_absurd_depths_and_elevations(factory) -> None:
    with pytest.raises(ValidationError, match="out of range|finite"):
        factory()


def test_water_sheet_elevation_masl_out_of_range_is_a_row_error() -> None:
    frame = pd.DataFrame([{"hole_id": "MW-01", "elevation_masl": 1e12}])
    levels, errors = DataParser()._parse_water_levels(frame, _collars())
    assert not levels
    assert errors == [
        "Water row 2 (MW-01): elevation_masl 1e+12 is out of range (limit ±1,000,000 m) — "
        "check the units"
    ]


# ------------------------------------------------------- 2. inclination / azimuth


@pytest.mark.parametrize("model", ["collar", "deviation"])
def test_inclination_past_horizontal_is_a_row_error_with_convention_hint(model: str) -> None:
    with pytest.raises(ValidationError, match="past horizontal.*negative mining dip"):
        if model == "collar":
            Collar(hole_id="A", easting=0, northing=0, elevation=1, total_depth=5, inclination_deg=120)
        else:
            DeviationReading(hole_id="A", depth=5, inclination_deg=120, azimuth_deg=0)
    with pytest.raises(ValidationError, match="beyond −90"):
        DeviationReading(hole_id="A", depth=5, inclination_deg=-95, azimuth_deg=0)


def test_azimuth_wraps_to_0_360_and_rejects_non_finite() -> None:
    assert DeviationReading(hole_id="A", depth=1, inclination_deg=10, azimuth_deg=-30).azimuth_deg == 330
    assert DeviationReading(hole_id="A", depth=1, inclination_deg=10, azimuth_deg=720).azimuth_deg == 0
    collar = Collar(hole_id="A", easting=0, northing=0, elevation=1, total_depth=5, azimuth_deg=370)
    assert collar.azimuth_deg == pytest.approx(10)
    with pytest.raises(ValidationError, match="finite"):
        DeviationReading(hole_id="A", depth=1, inclination_deg=10, azimuth_deg=float("inf"))


def test_horizontal_inclination_is_accepted_with_a_validate_warning() -> None:
    collars = [
        Collar(hole_id="H1", easting=0, northing=0, elevation=10, total_depth=5, inclination_deg=90),
        Collar(hole_id="H2", easting=9, northing=0, elevation=10, total_depth=5, inclination_deg=-90),
        Collar(hole_id="H3", easting=20, northing=0, elevation=10, total_depth=5),
    ]
    readings = [DeviationReading(hole_id="H3", depth=4, inclination_deg=90, azimuth_deg=0)]
    report = analyze_parsed_data(collars, [], deviation_readings=readings)
    horizontal = [issue for issue in report.issues if issue.code == "horizontal_hole"]
    assert {issue.hole_id for issue in horizontal} == {"H1", "H3"}  # −90 = straight down
    assert all("no vertical extent" in issue.message for issue in horizontal)
    assert all(issue.severity == "warning" for issue in horizontal)


# --------------------------------------------------- 3. environmental values


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_environmental_value_must_be_finite(bad: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        EnvironmentalReading(hole_id="A", parameter="Cl", value=bad, depth=1.0)


def test_environmental_infinite_depth_is_rejected_and_nan_depth_is_blank() -> None:
    with pytest.raises(ValidationError, match="finite"):
        EnvironmentalReading(hole_id="A", parameter="Cl", value=1, depth=float("inf"))
    with pytest.raises(ValidationError, match="depth or from_depth and to_depth are required"):
        EnvironmentalReading(hole_id="A", parameter="Cl", value=1, depth=float("nan"))


def test_environmental_sheet_inf_value_is_a_row_error_not_an_inf_label() -> None:
    frame = pd.DataFrame(
        [
            {"hole_id": "MW-01", "parameter": "Cl", "value": float("inf"), "depth": 2.0},
            {"hole_id": "MW-01", "parameter": "Cl", "value": 3.0, "depth": 4.0},
        ]
    )
    readings, errors = DataParser()._parse_environmental_dataframe(frame, _collars())
    assert [r.value for r in readings] == [3.0]
    assert len(errors) == 1 and errors[0].startswith("Environmental row 2 (MW-01)")
    assert all("inf" not in r.display_label and "nan" not in r.display_label for r in readings)


def test_negative_concentration_is_a_validate_warning_not_an_error() -> None:
    readings = [
        EnvironmentalReading(hole_id="MW-01", parameter="Benzene", value=-0.5, depth=2, unit="mg/L"),
        EnvironmentalReading(hole_id="MW-01", parameter="ORP", value=-120, depth=2, unit="mV"),
        EnvironmentalReading(hole_id="MW-01", parameter="Temperature", value=-2, depth=2, unit="°C"),
    ]
    summary = summarize_environmental_readings(_collars(), readings, ["MW-01"])
    negative = [w for w in summary.warnings if "negative" in w]
    assert negative == [
        "Benzene: negative value(s) (MW-01 -0.5 mg/L) — a concentration can't be negative; "
        "check the sign (a non-detect is written <RDL, e.g. <0.5)"
    ]


# ----------------------------------------------------------- 4. sheet checks


def test_inverted_colour_bands_warn_in_validate_and_on_the_figure() -> None:
    readings = [
        EnvironmentalReading(
            hole_id="MW-01", parameter="Chloride", value=200, depth=2, green_max=300, orange_max=250
        )
    ]
    summary = summarize_environmental_readings(_collars(), readings, ["MW-01"])
    assert any("Chloride: colour bands inverted (green_max 300 > orange_max 250)" in w for w in summary.warnings)
    # Configure-entered limits (per parameter and sheet-wide) on the figure.
    result = build_cross_section(
        _collars(),
        _lith(),
        LINE,
        environmental_readings=readings,
        environmental_parameters=["Chloride"],
        chemistry_color_mode="threshold",
        chemistry_thresholds={"Chloride": (300.0, 250.0)},
        export_formats=SVG,
    )
    assert any("Chloride: colour bands inverted" in note for note in result.qa_notes)
    assert chemistry_threshold_inversion_warnings({}, green_max=50, orange_max=10) == [
        "All parameters: colour bands inverted (green_max 50 > orange_max 10) — green is "
        "capped at 10, so there is no orange band; swap the limits"
    ]
    assert chemistry_threshold_inversion_warnings({"Cl": (10.0, 50.0), "Na": (None, 5.0)}) == []


def test_negative_screen_depth_is_a_row_error() -> None:
    frame = pd.DataFrame([{"hole_id": "MW-01", "from_depth": -1.0, "to_depth": 3.0}])
    screens, errors = DataParser()._parse_screens_dataframe(frame, _collars())
    assert not screens
    assert errors == ["Screens row 2 (MW-01): from_depth (-1) can't be negative — enter 0 or more."]


def test_absurd_lithology_depth_is_a_row_error() -> None:
    frame = pd.DataFrame(
        [{"hole_id": "MW-01", "from_depth": 0, "to_depth": 250000, "lithology_code": "SAND"}]
    )
    lith, errors = DataParser()._parse_lithologies(frame, _collars())
    assert not lith
    assert errors[0].startswith("Lithology row 2 (MW-01): to_depth 250000 is out of range")


def _workbook(sheets: dict[str, list[list[object]]]) -> BytesIO:
    book = openpyxl.Workbook()
    book.remove(book.active)
    for name, rows in sheets.items():
        sheet = book.create_sheet(name)
        for row in rows:
            sheet.append(row)
    buffer = BytesIO()
    book.save(buffer)
    buffer.seek(0)
    return buffer


_COLLAR_HEADER = ["hole_id", "easting", "northing", "elevation", "total_depth"]
_LITH_ROWS = [
    ["hole_id", "from_depth", "to_depth", "lithology_code"],
    ["MW-01", 0, 12, "SAND"],
    ["MW-02", 0, 12, "SAND"],
]


def _base_sheets() -> dict[str, list[list[object]]]:
    return {
        "Collars": [_COLLAR_HEADER, ["MW-01", 0, 0, 100, 12], ["MW-02", 50, 0, 101, 12]],
        "Lithology": [row[:] for row in _LITH_ROWS],
    }


def test_deviations_errors_carry_row_numbers_and_orphan_holes_are_listed() -> None:
    sheets = _base_sheets()
    sheets["Deviations"] = [
        ["hole_id", "depth", "inclination_deg", "azimuth_deg"],
        ["MW-01", 0, 10, 45],
        ["MW-01", 6, 130, 45],
        ["MW-9", 0, 5, 0],
        ["MW-9", 6, 5, 0],
        [None, None, None, None],
    ]
    result = DataParser().parse_file(_workbook(sheets))
    assert [r.hole_id for r in result.deviation_readings] == ["MW-01"]
    joined = "\n".join(result.errors)
    assert "Deviations row 3 (MW-01): inclination_deg 130° is past horizontal" in joined
    assert (
        "Deviations row 4: MW-9 is in Deviations but not in Collars (survey of 2 rows not used)"
        in joined
    )
    # Blank formatted rows are padding, not errors.
    assert "row 6" not in joined
    # API path (JSON bundle, no parser): Validate lists the orphan holes.
    report = analyze_parsed_data(
        _collars(),
        _lith(),
        deviation_readings=[DeviationReading(hole_id="MW-9", depth=0, inclination_deg=5, azimuth_deg=0)],
    )
    assert any(issue.code == "orphan_deviations" and "MW-9 (1)" in issue.message for issue in report.issues)


# ------------------------------------------------------------------- 5. LOW


def test_formula_without_cached_result_and_excel_errors_are_named() -> None:
    sheets = _base_sheets()
    sheets["Collars"] += [
        ["MW-03", "=1+1", 0, 100, 12],
        ["MW-04", "#N/A", 0, 100, 12],
        ["MW-05", "#DIV/0!", 0, 100, 12],
        ["MW-06", None, 0, 100, 12],
    ]
    result = DataParser().parse_file(_workbook(sheets))
    errors = {message.split(":")[0]: message for message in result.errors}
    assert "formula whose result was not cached" in errors["Collars row 4 (MW-03)"]
    assert "Excel error value #N/A" in errors["Collars row 5 (MW-04)"]
    assert "Excel error value #DIV/0!" in errors["Collars row 6 (MW-05)"]
    # A genuinely empty cell keeps the plain message.
    assert "easting is empty" in errors["Collars row 7 (MW-06)"]


def test_true_false_in_a_numeric_column_is_a_row_error() -> None:
    frame = pd.DataFrame(
        [
            {"hole_id": "MW-01", "easting": True, "northing": 0, "elevation": 100, "total_depth": 12},
            {"hole_id": "MW-02", "easting": 50, "northing": 0, "elevation": 101, "total_depth": 12},
        ]
    )
    collars, errors = DataParser()._parse_collars(frame)
    assert [c.hole_id for c in collars] == ["MW-02"]
    assert errors == [
        "Collars row 2 (MW-01), easting = 'True' is a TRUE/FALSE cell — enter a number."
    ]
    water = pd.DataFrame([{"hole_id": "MW-02", "depth": False}])
    levels, water_errors = DataParser()._parse_water_levels(water, collars)
    assert not levels and "TRUE/FALSE" in water_errors[0]
    with pytest.raises(ValidationError, match="TRUE/FALSE"):
        EnvironmentalReading(hole_id="A", parameter="Cl", value=True, depth=1.0)


def test_hole_ids_differing_only_by_case_or_spaces_warn() -> None:
    collars = _collars() + [
        Collar(hole_id="mw-01", easting=5, northing=0, elevation=100, total_depth=12),
        Collar(hole_id="MW -02", easting=7, northing=0, elevation=100, total_depth=12),
    ]
    report = analyze_parsed_data(collars, [])
    messages = [i.message for i in report.issues if i.code == "hole_id_case_variants"]
    assert len(messages) == 2
    assert messages[0].startswith("Hole IDs MW-01 and mw-01 differ only by case or spaces")
    # A child-sheet row with the wrong case names the Collars spelling.
    frame = pd.DataFrame([{"hole_id": "mw-01", "depth": 3.0}])
    _levels, errors = DataParser()._parse_water_levels(frame, _collars())
    assert errors == [
        "Water row 2: mw-01 is in Water but not in Collars — Collars has MW-01 — hole IDs "
        "must match exactly (case and spaces)."
    ]


def test_chemistry_date_notes_are_capped() -> None:
    readings = [
        EnvironmentalReading(
            hole_id="MW-01", parameter="Cl", value=1, depth=float(depth), date=datetime.date(2026, 1, day)
        )
        for depth in range(1, 41)
        for day in (1, 2)
    ]
    kept, notes = _latest_chemistry_samples(readings, ["MW-01"], ["Cl"])
    assert len(kept) == 40
    assert len(notes) == MAX_CHEMISTRY_DATE_NOTES + 1
    assert notes[-1].startswith("…and 30 more sample points with several sampling dates")


def test_identical_input_gives_identical_pdf_bytes() -> None:
    import time

    first = build_cross_section(_collars(), _lith(), LINE, export_formats=frozenset({"pdf"})).pdf_bytes
    time.sleep(1.1)  # CreationDate has one-second resolution
    second = build_cross_section(_collars(), _lith(), LINE, export_formats=frozenset({"pdf"})).pdf_bytes
    assert first == second
    assert b"/CreationDate" not in first
