"""Code-review findings (parsing / ingest / chemistry identity / limits) and their fixes.

F1  Parameter identity is ``chemistry_parameter_key`` end to end: the cap
    folded "Cl" / "cl" into one slot but the renderers matched readings by
    exact spelling, so every "cl" reading silently disappeared.
F2  TRUE/FALSE cells survive every data-sheet read (Data Entry, Field Data
    RL/TD + OVA/EC, import-profile lithology, mapped native sheets), not only
    the native tabs.
F3  A reader that cannot report booleans gives a user-visible import note
    (pandas' readers all put bools in the cell grid: openpyxl, xlrd
    XL_CELL_BOOLEAN, calamine, odf, pyxlsb).
F4  Sheet-wide and per-parameter chemistry limits share one parser: blank /
    NaN = no limit; TRUE/FALSE, negative and ±inf are errors.
F5  Colour mode / limits are validated before geometry is computed, and
    SectionBuildRequest normalises the mode like build_cross_section.
F6  One read per data sheet (no second openpyxl scan, probes reuse the frame).
F7  ``_read_sheet``'s unused ``**kwargs`` branch is gone.
F8  Lithology codes resolve case-insensitively against the scheme before aliases.
"""

from __future__ import annotations

import math
from io import BytesIO

import numpy as np
import openpyxl
import pandas as pd
import pytest
from pydantic import ValidationError

import parsing
import pipeline
from ai_quality import (
    normalize_lithology_code,
    read_mapped_sheets,
    summarize_environmental_readings,
)
from constants import CANONICAL_LITHOLOGY_CODES
from ingestion import ingest_workbook
from models import Collar, Lithology
from parsing import DataParser, bool_cell_detection_note, read_data_sheet
from render_profiles import (
    chemistry_parameter_spellings,
    normalise_chemistry_thresholds,
    parse_chemistry_limit,
)
from render_theme import CHEMISTRY_LABEL_RED
from section_build_request import SectionBuildRequest


def _xlsx(sheets: dict[str, list[list[object]]]) -> BytesIO:
    """A real .xlsx: openpyxl writes Python ``True`` as a boolean cell."""
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


def _table(rows: list[dict]) -> list[list[object]]:
    columns = list(rows[0])
    return [columns] + [[row.get(column) for column in columns] for row in rows]


_COLLARS = [
    {
        "hole_id": f"BH-{i}",
        "easting": 1000 + 50 * i,
        "northing": 5000,
        "elevation": 100,
        "total_depth": 12,
    }
    for i in (1, 2, 3)
]
_LITHOLOGY = [
    {"hole_id": f"BH-{i}", "from_depth": 0, "to_depth": 12, "lithology_code": "Clay"}
    for i in (1, 2, 3)
]
_LINE = [(1000.0, 5000.0), (1200.0, 5000.0)]


# --- F1: one parameter per chemistry_parameter_key --------------------------------


def _case_variant_workbook() -> BytesIO:
    environmental = [
        {"hole_id": "BH-1", "parameter": "Cl", "value": 7771, "depth": 2, "unit": "mg/L"},
        {"hole_id": "BH-2", "parameter": "cl", "value": 8882, "depth": 3, "unit": "mg/L"},
        {"hole_id": "BH-3", "parameter": " CL ", "value": 9993, "depth": 4, "unit": "mg/L"},
    ]
    return _xlsx(
        {
            "Collars": _table(_COLLARS),
            "Lithology": _table(_LITHOLOGY),
            "Environmental": _table(environmental),
        }
    )


@pytest.mark.parametrize("layout", ["section_sheet", "consulting_section", "chart"])
@pytest.mark.parametrize("picked", [["Cl"], ["cl"], ["Cl", "cl", "CL"]])
def test_case_variant_parameter_readings_all_plot(layout: str, picked: list[str]) -> None:
    result, _report = ingest_workbook(_case_variant_workbook())
    assert {r.parameter for r in result.environmental_readings} == {"Cl", "cl", "CL"}
    section = pipeline.build_cross_section(
        result.collars,
        result.lithologies,
        _LINE,
        environmental_readings=result.environmental_readings,
        environmental_parameters=picked,
        render_layout=layout,
        export_formats=frozenset({"svg"}),
    )
    svg = section.svg_bytes.decode()
    for value in ("7771", "8882", "9993"):
        assert value in svg, (layout, picked, value)
    # One parameter, so no "up to N parameters" cap note.
    assert not any("chemistry parameters per section" in note for note in section.qa_notes)


def test_renderer_matches_readings_by_key_without_pipeline_renaming(monkeypatch) -> None:
    """Direct renderer use: renderer_chemistry groups by key on its own."""
    monkeypatch.setattr(
        pipeline, "_canonical_parameter_spellings", lambda readings, _names: list(readings)
    )
    result, _report = ingest_workbook(_case_variant_workbook())
    section = pipeline.build_cross_section(
        result.collars,
        result.lithologies,
        _LINE,
        environmental_readings=result.environmental_readings,
        environmental_parameters=["Cl"],
        export_formats=frozenset({"svg"}),
    )
    svg = section.svg_bytes.decode()
    assert all(value in svg for value in ("7771", "8882", "9993"))


def test_per_parameter_limit_bands_every_spelling() -> None:
    """A limit typed for "cl" colours the "Cl" and "CL" readings too (same key)."""
    result, _report = ingest_workbook(_case_variant_workbook())
    kwargs = dict(
        environmental_readings=result.environmental_readings,
        environmental_parameters=["Cl"],
        chemistry_color_mode="black_red",
        parameter_draw_markers=False,
        export_formats=frozenset({"svg"}),
    )
    red = CHEMISTRY_LABEL_RED.lower().encode()

    def red_count(limit: float) -> int:
        return (
            pipeline.build_cross_section(
                result.collars,
                result.lithologies,
                _LINE,
                chemistry_thresholds={"cl": (limit, None)},
                **kwargs,
            )
            .svg_bytes.lower()
            .count(red)
        )

    # Same key either way; only the value labels change colour. 8882 ("cl")
    # and 9993 ("CL") print red above the "cl" limit, 7771 ("Cl") stays black.
    assert red_count(8000.0) - red_count(1e6) >= 2


def test_configure_options_and_validate_name_one_parameter() -> None:
    assert chemistry_parameter_spellings(["Cl", "cl", " CL ", "Na", ""]) == {
        "Cl": ("Cl", "cl", "CL"),
        "Na": ("Na",),
    }
    result, _report = ingest_workbook(_case_variant_workbook())
    summary = summarize_environmental_readings(
        result.collars, result.environmental_readings, ["BH-1", "BH-2", "BH-3"]
    )
    assert [(p.parameter, p.hole_count) for p in summary.parameters] == [("Cl", 3)]
    notes = [w for w in summary.warnings if "differ only in case" in w]
    assert len(notes) == 1 and notes[0].startswith("Info: ")
    assert "'Cl', 'cl', 'CL' → 'Cl'" in notes[0]


# --- F2: TRUE/FALSE cells on every data-sheet path --------------------------------


def test_data_entry_sheet_keeps_bool_after_an_equal_number() -> None:
    data_entry = [
        ["COLLARS"],
        ["hole_id", "easting", "northing", "elevation", "total_depth"],
        ["BH-1", 1, 5000, 100, 12],
        ["BH-2", True, 5000, 100, 12],  # 1 above TRUE: pandas folded it onto 1
        [],
        ["LITHOLOGY"],
        ["hole_id", "from_depth", "to_depth", "lithology_code"],
        ["BH-1", 0, 12, "Clay"],
        ["BH-2", 0, 12, "Clay"],
    ]
    result, _report = ingest_workbook(_xlsx({"Data Entry": data_entry}))
    assert [c.hole_id for c in result.collars] == ["BH-1"]
    assert any("easting = 'True' is a TRUE/FALSE cell" in e for e in result.errors), result.errors


def test_data_entry_frame_header_none_keeps_bools() -> None:
    workbook = pd.ExcelFile(_xlsx({"Data Entry": [["a", "b"], [1, 1], [True, False]]}))
    frame = read_data_sheet(workbook, "Data Entry", header=None)
    assert [type(v) for v in frame.iloc[2]] == [bool, bool]
    assert frame.iloc[1].tolist() == [1, 1] and frame.iloc[2].tolist() == [True, False]


def _field_export(lithology_rows: list[list[object]], field_rows=None) -> BytesIO:
    sheets = {"Lithology": [["Label", "Depth", "Lithology", "Lat", "Long"], *lithology_rows]}
    if field_rows is not None:
        sheets["Field Data"] = field_rows
    return _xlsx(sheets)


_FIELD_LITHOLOGY = [
    ["BH-01", "0-2m", "clay", 58.57, -119.19],
    ["BH-02", "0-3m", "clay", 58.571, -119.189],
]


def test_field_data_bool_cells_are_not_rl_td_or_readings() -> None:
    field = [
        ["Label", "Depth", "OVA", "EC", "Elevation", "Total_Depth"],
        ["BH-01", "0.0-0.15m", 1, 0.4, 101.0, 5],
        ["BH-02", "0.0-0.15m", True, 0.5, True, True],
    ]
    result, report = ingest_workbook(
        _field_export(_FIELD_LITHOLOGY, field), profile_id="field_export_v1"
    )
    collars = {c.hole_id: (c.elevation, c.total_depth) for c in result.collars}
    assert collars["BH-01"] == (101.0, 5.0)
    assert collars["BH-02"] == (100.0, 3.0)  # profile default RL, lithology TD — not 1.0
    readings = {(r.hole_id, r.parameter): r.value for r in result.environmental_readings}
    assert readings == {("BH-01", "OVA"): 1.0, ("BH-01", "EC"): 0.4, ("BH-02", "EC"): 0.5}
    assert any(
        e.startswith("Field Data row 3 (BH-02), ova = 'True' is a TRUE/FALSE cell")
        for e in result.errors
    ), result.errors
    notes = [w for w in report.warnings if "TRUE/FALSE" in w]
    assert any("Field Data row 3 (BH-02): Elevation = 'True'" in w for w in notes), notes
    assert any("Field Data row 3 (BH-02): Total_Depth = 'True'" in w for w in notes), notes


def test_import_profile_lithology_bool_coordinate_is_an_error() -> None:
    rows = [_FIELD_LITHOLOGY[0], ["BH-02", "0-3m", "clay", True, -119.189]]
    with pytest.raises(ValueError, match=r"Lithology row 3: Lat = 'True' is a TRUE/FALSE cell"):
        ingest_workbook(_field_export(rows), profile_id="field_export_v1")


def test_mapped_native_sheets_keep_bools() -> None:
    collars = [dict(row) for row in _COLLARS]
    collars[0]["easting"] = 1
    collars[1]["easting"] = True
    workbook = pd.ExcelFile(_xlsx({"Collars": _table(collars), "Lithology": _table(_LITHOLOGY)}))
    from ai_quality import propose_workbook_mapping

    collars_df, _lithology_df = read_mapped_sheets(
        None, propose_workbook_mapping(workbook), workbook
    )
    assert collars_df["easting"].tolist()[:2] == [1, True]
    assert isinstance(collars_df["easting"].iloc[1], bool)


# --- F3: readers that cannot report booleans --------------------------------------


def test_bool_marks_cover_every_engine_grid_including_numpy_bools() -> None:
    grid = [["a", "b"], [1, np.bool_(True)], [False, 2.5], ["", "x"]]
    assert parsing._mark_bool_cells(grid)
    assert grid[1][0] == 1 and grid[2][1] == 2.5 and grid[3] == ["", "x"]
    restored = parsing._restore_marked_bools(pd.DataFrame(grid[1:], columns=grid[0], dtype=object))
    assert restored["b"].tolist()[0] is True and restored["a"].tolist()[1] is False
    assert {"openpyxl", "xlrd", "calamine", "odf"} <= parsing.BOOL_AWARE_EXCEL_ENGINES


def test_reader_without_bool_cells_gives_an_import_warning(monkeypatch) -> None:
    monkeypatch.setattr(parsing, "BOOL_AWARE_EXCEL_ENGINES", frozenset())
    _result, report = ingest_workbook(
        _xlsx({"Collars": _table(_COLLARS), "Lithology": _table(_LITHOLOGY)})
    )
    notes = [w for w in report.warnings if w.startswith("TRUE/FALSE cells cannot be detected")]
    assert len(notes) == 1 and "openpyxl" in notes[0] and not notes[0].startswith("Info:")


def test_xlsx_read_has_no_bool_detection_note() -> None:
    workbook = pd.ExcelFile(_xlsx({"Collars": _table(_COLLARS)}))
    read_data_sheet(workbook, "Collars")
    assert bool_cell_detection_note(workbook) is None


# --- F4: one limit parser ----------------------------------------------------------


@pytest.mark.parametrize("blank", [None, "", "   ", math.nan, np.nan])
def test_blank_and_nan_limits_are_no_limit_everywhere(blank) -> None:
    assert parse_chemistry_limit(blank) is None
    assert pipeline._check_chemistry_limit(blank, "chemistry_threshold_green_max") is None
    assert normalise_chemistry_thresholds({"Cl": (blank, None)}) == {}
    request = SectionBuildRequest(
        transect_points=((0.0, 0.0), (1.0, 0.0)), chemistry_threshold_green_max=blank
    )
    assert request.chemistry_threshold_green_max is None


@pytest.mark.parametrize("bad", [True, False, np.bool_(True), -1.0, math.inf, -math.inf, "abc"])
def test_bool_negative_inf_limits_are_errors_everywhere(bad) -> None:
    with pytest.raises(ValueError):
        pipeline._check_chemistry_limit(bad, "chemistry_threshold_green_max")
    with pytest.raises(ValueError):
        normalise_chemistry_thresholds({"Cl": (bad, None)})
    with pytest.raises(ValidationError):
        SectionBuildRequest(
            transect_points=((0.0, 0.0), (1.0, 0.0)), chemistry_threshold_yellow_max=bad
        )


def test_numeric_text_and_zero_limits_agree() -> None:
    assert parse_chemistry_limit(" 12.5 ") == 12.5
    assert pipeline._check_chemistry_limit(0, "g") == 0.0
    assert normalise_chemistry_thresholds({"Cl": (" 12.5 ", 0)}) == {"Cl": (12.5, 0.0)}


# --- F5: mode / limits validated up front, request normalises the same way --------


def _two_holes():
    collars = [
        Collar(hole_id=f"B{i}", easting=30.0 * i, northing=0, elevation=100, total_depth=12)
        for i in range(2)
    ]
    lithology = [
        Lithology(hole_id=c.hole_id, from_depth=0, to_depth=12, lithology_code="Clay")
        for c in collars
    ]
    return collars, lithology


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chemistry_color_mode": "purple"},
        {"chemistry_color_mode": ""},
        {"chemistry_threshold_green_max": -1.0},
        {"chemistry_threshold_yellow_max": True},
        {"chemistry_thresholds": {"Cl": (math.inf, None)}},
    ],
)
def test_bad_chemistry_options_fail_before_geometry(monkeypatch, kwargs) -> None:
    calls: list[int] = []
    monkeypatch.setattr(pipeline, "compute_section_geometry", lambda *a, **k: calls.append(1))
    collars, lithology = _two_holes()
    with pytest.raises(ValueError):
        pipeline.build_cross_section(collars, lithology, [(0.0, 0.0), (30.0, 0.0)], **kwargs)
    assert calls == []


@pytest.mark.parametrize(
    ("given", "expected"),
    [("Threshold", "threshold"), (" BLACK_RED ", "black_red"), ("green_red", "green_red")],
)
def test_request_and_pipeline_normalise_mode_alike(given: str, expected: str) -> None:
    request = SectionBuildRequest(
        transect_points=((0.0, 0.0), (1.0, 0.0)), chemistry_color_mode=given
    )
    assert request.chemistry_color_mode == expected
    assert pipeline._check_chemistry_color_mode(given) == expected


@pytest.mark.parametrize("bad", ["purple", "", "   ", "black-red"])
def test_request_and_pipeline_reject_the_same_modes(bad: str) -> None:
    with pytest.raises(ValidationError):
        SectionBuildRequest(transect_points=((0.0, 0.0), (1.0, 0.0)), chemistry_color_mode=bad)
    with pytest.raises(ValueError):
        pipeline._check_chemistry_color_mode(bad)


# --- F6 / F7: one read per sheet --------------------------------------------------


def test_each_data_sheet_is_read_once_per_workbook(monkeypatch) -> None:
    workbook = pd.ExcelFile(
        _xlsx(
            {
                "Collars": _table(_COLLARS),
                "Lithology": _table(_LITHOLOGY),
                "Water": _table([{"hole_id": "BH-1", "depth": 2}]),
            }
        )
    )
    reads: list[str] = []
    original = workbook._reader.get_sheet_data

    def counting(sheet, *args, **kwargs):
        reads.append(sheet.title)
        return original(sheet, *args, **kwargs)

    monkeypatch.setattr(workbook._reader, "get_sheet_data", counting)
    # No second openpyxl scan of the sheet either.
    monkeypatch.setattr(
        type(workbook), "book", property(lambda _self: pytest.fail("bool scan re-read the sheet"))
    )
    parser = DataParser()
    assert parser._native_optional_sheet_has_rows(workbook, "Water")
    first = read_data_sheet(workbook, "Water")
    first.loc[0, "depth"] = 99  # callers get copies, the cache stays as read
    assert read_data_sheet(workbook, "Water").loc[0, "depth"] == 2
    assert reads.count("Water") == 1


def test_read_sheet_kwargs_branch_removed() -> None:
    assert not hasattr(parsing, "_read_sheet")
    assert not hasattr(parsing, "_restore_bool_cells")


# --- F8: lithology codes in any case ----------------------------------------------


@pytest.mark.parametrize("code", sorted(CANONICAL_LITHOLOGY_CODES))
def test_every_scheme_code_resolves_in_any_case_without_aliases(code: str) -> None:
    for variant in (
        code,
        code.upper(),
        code.lower(),
        code.swapcase(),
        code.title(),
        f"  {code.upper()}  ",
        code.replace(" ", "  "),
    ):
        assert normalize_lithology_code(variant, {}) == code, variant


def test_scheme_match_wins_over_alias_and_unknown_codes_fall_through() -> None:
    assert normalize_lithology_code("SAND AND GRAVEL", {"sand_and_gravel": "Gravel"}) == (
        "Sand and Gravel"
    )
    assert normalize_lithology_code("silty clay till", {"silty_clay_till": "Clay"}) == "Clay"
    assert normalize_lithology_code("  Mystery Unit ", {}) == "Mystery Unit"
