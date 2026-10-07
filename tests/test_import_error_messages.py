"""Plain-language import errors: bad value, Excel row, and the fix in one line."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from ai_quality import analyze_parsed_data
from app_upload import upload_headline
from app_validate import issue_display_message, lithology_skipped_holes
from models import Collar, Lithology
from parsing import DataParser, skipped_row_hole_ids

ROOT = Path(__file__).resolve().parents[1]

COLLARS = [
    {"hole_id": "BH-01", "easting": 500000, "northing": 4500000, "elevation": 100, "total_depth": 25},
    {"hole_id": "BH-02", "easting": 500050, "northing": 4500000, "elevation": 100, "total_depth": 26},
    {"hole_id": "BH-03", "easting": 500100, "northing": 4500000, "elevation": 100, "total_depth": 20},
]


def _workbook(lithology: list[dict], *, environmental: list[dict] | None = None) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(COLLARS).to_excel(writer, sheet_name="Collars", index=False)
        pd.DataFrame(lithology).to_excel(writer, sheet_name="Lithology", index=False)
        if environmental is not None:
            pd.DataFrame(environmental).to_excel(writer, sheet_name="Environmental", index=False)
    return buffer.getvalue()


def _lith(hole: str, top: object, base: object, code: str = "CLAY") -> dict:
    return {"hole_id": hole, "from_depth": top, "to_depth": base, "lithology_code": code}


BAD_LITHOLOGY = [
    _lith("BH-01", 0, 10),  # Excel row 2
    _lith("BH-01", "4 m", 25),  # row 3: text in a number column
    _lith("BH-02", 0, 4),  # row 4
    _lith("BH-02", 9, 5),  # row 5: from > to
    _lith("BH-09", 0, 5),  # row 6: not in Collars
    _lith("BH-03", 0, 20),  # row 7
]


@pytest.fixture(scope="module")
def bad_result():
    return DataParser().parse_file(BytesIO(_workbook(BAD_LITHOLOGY)))


def test_text_in_depth_column_names_row_value_and_fix(bad_result) -> None:
    assert (
        "Lithology row 3 (BH-01), from_depth = '4 m' — enter a number only (e.g. 4)."
        in bad_result.errors
    )
    assert not any("Input should be" in message for message in bad_result.errors)


def test_unknown_hole_says_which_sheet_and_how_to_fix(bad_result) -> None:
    assert (
        "Lithology row 6: BH-09 is in Lithology but not in Collars — "
        "add it to Collars or fix the spelling."
    ) in bad_result.errors


def test_from_deeper_than_to_shows_both_depths(bad_result) -> None:
    assert (
        "Lithology row 5 (BH-02): from_depth 9 is deeper than to_depth 5 — "
        "swap them or fix the typo."
    ) in bad_result.errors


def test_row_messages_stay_short(bad_result) -> None:
    assert all(len(message) <= 120 for message in bad_result.errors), bad_result.errors


def test_bad_label_colour_lists_allowed_values() -> None:
    result = DataParser().parse_file(
        BytesIO(
            _workbook(
                [_lith("BH-01", 0, 25)],
                environmental=[
                    {"hole_id": "BH-01", "parameter": "Cl", "value": 5, "depth": 2, "label_color": "purple"}
                ],
            )
        )
    )
    assert (
        "Environmental row 2 (BH-01), label_color = 'purple' — use green, red, black or orange, "
        "or leave it blank."
    ) in result.errors


def test_missing_required_column_names_sheet_and_column() -> None:
    lithology = [{"hole_id": "BH-01", "from_depth": 0, "lithology_code": "CLAY"}]
    with pytest.raises(ValueError) as excinfo:
        DataParser().parse_file(BytesIO(_workbook(lithology)))
    message = str(excinfo.value)
    assert message.startswith("The Lithology sheet is missing the to_depth column")
    assert "Help → Workbook & data entry" in message


def test_overlap_message_includes_both_depth_ranges() -> None:
    collars = [Collar(hole_id="BH-01", easting=0, northing=0, elevation=100, total_depth=25)]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0, to_depth=3, lithology_code="FILL"),
        Lithology(hole_id="BH-01", from_depth=3, to_depth=10, lithology_code="CLAY"),
        Lithology(hole_id="BH-01", from_depth=8, to_depth=25, lithology_code="SAND"),
    ]
    report = analyze_parsed_data(collars, lithologies)
    overlap = [issue.message for issue in report.issues if issue.code == "depth_overlap"]
    assert overlap == ["BH-01: 3–10 m overlaps 8–25 m — fix from_depth/to_depth in Lithology"]


def test_skipped_rows_map_back_to_holes(bad_result) -> None:
    holes = skipped_row_hole_ids(bad_result.errors)
    assert holes["Lithology"] == {"BH-01", "BH-02", "BH-09"}
    assert lithology_skipped_holes(bad_result.errors) == {"BH-01", "BH-02", "BH-09"}


def test_knock_on_advisory_says_rows_were_skipped() -> None:
    gap = SimpleNamespace(code="depth_gap", hole_id="BH-02", message="BH-02: gap")
    other = SimpleNamespace(code="depth_gap", hole_id="BH-03", message="BH-03: gap")
    assert "Lithology rows for this hole were skipped" in issue_display_message(gap, {"BH-02"})
    assert issue_display_message(other, {"BH-02"}) == "BH-03: gap"


def test_upload_headline_warns_when_rows_were_skipped() -> None:
    level, text = upload_headline(hole_count=3, interval_count=6, skipped_count=4, error_count=0)
    assert level == "warning"
    assert text.startswith("**Loaded with problems: 4 data errors (4 rows skipped) — see Validate.**")
    level, text = upload_headline(hole_count=3, interval_count=6, skipped_count=0, error_count=0)
    assert level == "success"


def test_upload_headline_counts_data_errors() -> None:
    level, text = upload_headline(hole_count=3, interval_count=6, skipped_count=1, error_count=2)
    assert level == "warning"
    assert "3 data errors (1 row skipped)" in text  # same total as Data health


def test_app_shows_problem_headline_and_counts_skipped_rows_as_errors() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    at.file_uploader[0].upload("bad.xlsx", _workbook(BAD_LITHOLOGY)).run()
    assert not at.exception
    warnings = [str(item.value) for item in at.warning]
    assert any("Loaded with problems: 3 data errors (3 rows skipped) — see Validate." in w for w in warnings)
    assert not any("Loaded **" in str(item.value) for item in at.success)
    # The skipped-row list appears once (on Validate), not per screen.
    listed = [str(md.value) for md in at.markdown if "BH-09 is in Lithology" in str(md.value)]
    assert len(listed) == 1
    status = [str(md.value) for md in at.markdown if str(md.value).startswith("**Needs fixes**")]
    assert status and "**3 errors**" in status[0]
    assert not [btn for btn in at.button if btn.label == "Load data health details"]


def test_missing_lithology_sheet_is_named_when_collars_exists() -> None:
    """A Collars-only workbook was told "No Collars sheet was found
    (sheets in this file: Collars)"."""
    from app_upload import _friendly_workbook_error

    text = _friendly_workbook_error(
        ValueError("Could not detect a supported workbook format. Sheets found: Collars, Notes")
    )
    assert "No **Lithology** sheet" in text and "No **Collars** sheet" not in text
    text = _friendly_workbook_error(
        ValueError("Could not detect a supported workbook format. Sheets found: Sheet1")
    )
    assert "No **Collars** sheet" in text


def test_non_detect_values_point_to_value_label() -> None:
    """'<10' / 'ND' in the value column were rejected with "enter a number
    only", with no hint that value_label exists for lab non-detects."""
    from io import BytesIO

    import openpyxl

    from models import DataParser
    from workbook_template import build_input_template_bytes

    book = openpyxl.load_workbook(BytesIO(build_input_template_bytes()))
    sheet = book["Environmental"]
    header = [cell.value for cell in sheet[1]]
    value_col = header.index("value") + 1
    sheet.cell(row=2, column=value_col, value="<10")
    buffer = BytesIO()
    book.save(buffer)
    buffer.seek(0)
    result = DataParser().parse_file(buffer)
    messages = [e for e in result.errors if "'<10'" in e]
    assert messages and "value_label" in messages[0], result.errors
