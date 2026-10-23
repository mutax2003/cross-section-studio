"""Adversarial QA round 2 — confirmed defects and their fixes.

D1  TRUE/FALSE in a numeric Excel column became 1 / 0 (pandas inferred an
    int/float column, so the row model's bool check never saw the bool).
D2  Unknown ``chemistry_color_mode`` strings were merged into the profile
    unchecked (``model_copy(update=...)`` skips the Literal).
D4  A negative / non-finite sheet-wide chemistry limit was accepted.
D7  The per-section parameter cap de-duplicated case-sensitively.
D6  rcParams written by another thread leaked into the exported bytes.
"""

from __future__ import annotations

import math
from io import BytesIO

import matplotlib as mpl
import openpyxl
import pytest
from pydantic import ValidationError

from ingestion import ingest_workbook
from models import Collar, EnvironmentalReading, Lithology
from pipeline import build_cross_section
from render_profiles import cap_chemistry_parameters
from section_build_request import SectionBuildRequest

# --- D1: TRUE/FALSE cells in numeric columns of a real workbook -------------------

_COLLARS = [
    {"hole_id": "BH-1", "easting": 1000, "northing": 5000, "elevation": 100, "total_depth": 12},
    {"hole_id": "BH-2", "easting": 1050, "northing": 5000, "elevation": 100, "total_depth": 12},
    {"hole_id": "BH-3", "easting": 1100, "northing": 5000, "elevation": 100, "total_depth": 12},
]
_LITHOLOGY = [
    {"hole_id": hole, "from_depth": 0, "to_depth": 12, "lithology_code": "Clay"}
    for hole in ("BH-1", "BH-2", "BH-3")
]


def _workbook(sheets: dict[str, list[dict]]) -> BytesIO:
    """A real .xlsx: openpyxl writes Python ``True`` as a boolean cell (``data_type == 'b'``)."""
    book = openpyxl.Workbook()
    book.remove(book.active)
    for name, rows in sheets.items():
        sheet = book.create_sheet(name)
        columns = list(rows[0].keys())
        sheet.append(columns)
        for row in rows:
            sheet.append([row.get(column) for column in columns])
    buffer = BytesIO()
    book.save(buffer)
    buffer.seek(0)
    return buffer


def _rows(base: list[dict], index: int, **overrides) -> list[dict]:
    rows = [dict(row) for row in base]
    rows[index].update(overrides)
    return rows


def test_bool_in_a_numeric_collar_column_is_a_row_error_not_1_or_0() -> None:
    sheets = {
        "Collars": _rows(_rows(_COLLARS, 1, easting=True), 2, total_depth=False),
        "Lithology": _LITHOLOGY,
    }
    result, _report = ingest_workbook(_workbook(sheets))
    assert [c.hole_id for c in result.collars] == ["BH-1"]
    assert {c.hole_id: (c.easting, c.total_depth) for c in result.collars} == {
        "BH-1": (1000.0, 12.0)
    }
    errors = [e for e in result.errors if "TRUE/FALSE" in e]
    assert any(e.startswith("Collars row 3 (BH-2), easting = 'True'") for e in errors), (
        result.errors
    )
    assert any(e.startswith("Collars row 4 (BH-3), total_depth = 'False'") for e in errors), (
        result.errors
    )


@pytest.mark.parametrize(
    ("sheet", "rows", "bad_index", "column", "expected_prefix"),
    [
        ("Lithology", _LITHOLOGY, 0, "to_depth", "Lithology row 2 (BH-1), to_depth = 'True'"),
        (
            "Deviations",
            [
                {"hole_id": "BH-1", "depth": 0, "inclination_deg": 5, "azimuth_deg": 0},
                {"hole_id": "BH-1", "depth": 5, "inclination_deg": 6, "azimuth_deg": 0},
            ],
            1,
            "inclination_deg",
            "Deviations row 3 (BH-1), inclination_deg = 'True'",
        ),
        (
            "Environmental",
            [
                {"hole_id": "BH-1", "parameter": "Cl", "value": 5, "depth": 2},
                {"hole_id": "BH-1", "parameter": "Cl", "value": 7, "depth": 3},
            ],
            1,
            "value",
            "Environmental row 3 (BH-1), value = 'True'",
        ),
        (
            "Water",
            [{"hole_id": "BH-1", "depth": 2}, {"hole_id": "BH-2", "depth": 3}],
            1,
            "depth",
            "Water row 3 (BH-2), depth = 'True'",
        ),
        (
            # 1 above TRUE: pandas folds the bool onto the equal int (1 == True)
            # even with dtype=object, so the cell is restored from openpyxl.
            "Screens",
            [
                {"hole_id": "BH-1", "from_depth": 1, "to_depth": 3},
                {"hole_id": "BH-2", "from_depth": 1, "to_depth": 3},
            ],
            1,
            "from_depth",
            "Screens row 3 (BH-2), from_depth = 'True'",
        ),
    ],
)
def test_bool_in_a_numeric_overlay_column_is_a_row_error(
    sheet: str, rows: list[dict], bad_index: int, column: str, expected_prefix: str
) -> None:
    sheets = {"Collars": _COLLARS, "Lithology": _LITHOLOGY}
    sheets[sheet] = _rows(rows, bad_index, **{column: True})
    result, _report = ingest_workbook(_workbook(sheets))
    assert any(e.startswith(expected_prefix) for e in result.errors), result.errors
    # The good rows on the same sheet survive; the bool row is dropped, not coerced.
    survivors = {
        "Lithology": result.lithologies,
        "Deviations": result.deviation_readings,
        "Environmental": result.environmental_readings,
        "Water": result.water_levels,
        "Screens": result.screen_intervals,
    }[sheet]
    good = [row for index, row in enumerate(rows) if index != bad_index]
    assert [getattr(item, column) for item in survivors] == [float(row[column]) for row in good]


def test_plain_numeric_sheets_still_parse_with_native_types() -> None:
    """The object-dtype read keeps ints / floats / blanks / numeric hole_ids working."""
    collars = [{**row, "hole_id": index + 1} for index, row in enumerate(_COLLARS)]
    lithology = [
        {"hole_id": 1, "from_depth": 0, "to_depth": 10.5, "lithology_code": "Clay"},
        {"hole_id": None, "from_depth": None, "to_depth": None, "lithology_code": None},
        {"hole_id": 2, "from_depth": 0, "to_depth": 12, "lithology_code": "Sand"},
        {"hole_id": 3, "from_depth": 0, "to_depth": 12, "lithology_code": "Sand"},
    ]
    result, _report = ingest_workbook(_workbook({"Collars": collars, "Lithology": lithology}))
    assert not result.errors
    assert [c.hole_id for c in result.collars] == ["1", "2", "3"]
    assert [lith.to_depth for lith in result.lithologies] == [10.5, 12.0, 12.0]


# --- D2 / D4 / D7: chemistry colour mode, sheet-wide limits, parameter cap --------


def _section():
    collars = [
        Collar(hole_id=f"BH-{i}", easting=30.0 * i, northing=0, elevation=100, total_depth=12)
        for i in range(3)
    ]
    lithology = []
    for collar in collars:
        lithology.append(
            Lithology(hole_id=collar.hole_id, from_depth=0, to_depth=5, lithology_code="Clay")
        )
        lithology.append(
            Lithology(hole_id=collar.hole_id, from_depth=5, to_depth=12, lithology_code="Sand")
        )
    readings = [
        EnvironmentalReading(hole_id="BH-1", parameter="Cl", value=120.0, depth=2.0, unit="mg/L")
    ]
    return collars, lithology, [(0.0, 0.0), (60.0, 0.0)], readings


def _build(**kwargs):
    collars, lithology, line, readings = _section()
    return build_cross_section(
        collars,
        lithology,
        line,
        environmental_readings=readings,
        environmental_parameters=["Cl"],
        **{"export_formats": frozenset({"svg"}), **kwargs},
    )


@pytest.mark.parametrize("mode", ["purple_red", "blue", "", "   ", "black-red"])
def test_unknown_chemistry_color_mode_is_rejected(mode: str) -> None:
    with pytest.raises(ValueError, match="chemistry_color_mode must be one of") as info:
        _build(chemistry_color_mode=mode, chemistry_threshold_green_max=100.0)
    assert "black_red" in str(info.value) and repr(mode) in str(info.value)


@pytest.mark.parametrize(
    ("given", "normalised"),
    [("BLACK_RED", "black_red"), (" threshold ", "threshold"), ("Green_Red", "green_red")],
)
def test_chemistry_color_mode_is_trimmed_and_case_folded(given: str, normalised: str) -> None:
    result = _build(chemistry_color_mode=given, chemistry_threshold_green_max=100.0)
    assert result.svg_bytes.startswith(b"<?xml")
    reference = _build(chemistry_color_mode=normalised, chemistry_threshold_green_max=100.0)
    assert result.svg_bytes == reference.svg_bytes


@pytest.mark.parametrize("limit", [-1.0, -0.0001, math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    "field", ["chemistry_threshold_green_max", "chemistry_threshold_yellow_max"]
)
def test_negative_or_non_finite_sheet_wide_limit_is_rejected(field: str, limit: float) -> None:
    with pytest.raises(ValueError, match=f"{field} must be a finite number >= 0"):
        _build(chemistry_color_mode="black_red", **{field: limit})
    with pytest.raises(ValidationError):
        SectionBuildRequest(
            transect_points=((0.0, 0.0), (1.0, 0.0)),
            chemistry_color_mode="black_red",
            **{field: limit},
        )


def test_zero_and_positive_sheet_wide_limits_are_accepted() -> None:
    request = SectionBuildRequest(
        transect_points=((0.0, 0.0), (1.0, 0.0)),
        chemistry_threshold_green_max=0,
        chemistry_threshold_yellow_max=250,
    )
    assert (request.chemistry_threshold_green_max, request.chemistry_threshold_yellow_max) == (
        0.0,
        250.0,
    )
    result = _build(
        chemistry_color_mode="threshold",
        chemistry_threshold_green_max=0,
        chemistry_threshold_yellow_max=250,
    )
    assert result.svg_bytes


def test_parameter_cap_dedupes_case_and_space_insensitively() -> None:
    plotted, dropped = cap_chemistry_parameters(
        ["Cl", " cl", "Na", "NA ", "SO4", "Benzene", "Toluene"]
    )
    assert plotted == ("Cl", "Na", "SO4", "Benzene")
    assert dropped == ("Toluene",)
    # First spelling wins; blanks are ignored.
    assert cap_chemistry_parameters(["", "  ", "Chloride", "CHLORIDE"]) == (("Chloride",), ())


# --- D6: export styling vs caller-owned rcParams ---------------------------------


def test_export_keeps_figure_styling_but_honours_caller_settings() -> None:
    """Font/hatch styling is the figure's own and survives stray rcParams writes;
    caller-owned output settings (pdf.fonttype) are honoured, not overridden."""
    baseline = _build(chemistry_color_mode="black_red", chemistry_threshold_green_max=100.0)
    with mpl.rc_context():
        mpl.rcParams["font.size"] = 20.0
        mpl.rcParams["hatch.linewidth"] = 3.0
        mpl.rcParams["svg.fonttype"] = "none"
        churned = _build(chemistry_color_mode="black_red", chemistry_threshold_green_max=100.0)
    assert churned.svg_bytes == baseline.svg_bytes
    with mpl.rc_context({"pdf.fonttype": 42}):
        truetype = _build(
            chemistry_color_mode="black_red",
            chemistry_threshold_green_max=100.0,
            export_formats=frozenset({"svg", "pdf"}),
        )
    assert b"/FontFile2" in truetype.pdf_bytes or b"/TrueType" in truetype.pdf_bytes


def test_editable_svg_names_the_export_font() -> None:
    from export_framing import ExportFramingConfig

    result = _build(
        chemistry_color_mode="black_red",
        chemistry_threshold_green_max=100.0,
        export_framing=ExportFramingConfig(svg_editable_text=True),
    )
    svg = result.svg_bytes.decode("utf-8")
    assert "Arial" in svg
