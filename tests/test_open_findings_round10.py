"""Regression tests for the round-10 open QA findings.

1. The batch ZIP mirrors the single-section report ZIP: each section ships
   ``<stem>.svg`` and ``<stem>_metadata.json`` built by the same payload
   builder (``ui_helpers.export_metadata_payload``: colour mode, rule and
   limits, VE drawn, lithology scheme id), and the binder summary page lists
   the colour rule and the limits per parameter.
2. Configure "Parameters to plot" offers no "Select all" past the
   4-parameter cap (``MAX_CHEMISTRY_PARAMETERS``).
3. The template's sample Environmental rows agree with their own limits: no
   label_color override, and the value above the orange limit prints red in
   threshold mode.
4. ``workbook_template.load_data_entry_sheets`` reads Data Entry through
   ``parsing.read_data_sheet``.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from batch_export import (
    BatchTransectSpec,
    binder_summary_lines,
    build_batch_zip,
    build_one_transect_package,
    clear_batch_geometry_memo,
    export_binder_pdf,
)
from constants import LITHOLOGY_SCHEME_ID
from models import (
    MAX_CHEMISTRY_PARAMETERS,
    Collar,
    EnvironmentalReading,
    Lithology,
    ParseResult,
)
from section_build_request import SectionBuildRequest
from ui_helpers import THRESHOLD_COLOUR_RULE, export_metadata_payload

ROOT = Path(__file__).resolve().parents[1]
LIMITS = {"Chloride": (100.0, 250.0)}


def _parse_with_chemistry() -> ParseResult:
    holes = ("BH-01", "BH-02", "BH-03", "BH-04")
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=101.0, total_depth=22.0),
        Collar(hole_id="BH-03", easting=100.0, northing=0.0, elevation=102.0, total_depth=24.0),
        Collar(hole_id="BH-04", easting=50.0, northing=40.0, elevation=103.0, total_depth=18.0),
    ]
    lithologies = [
        Lithology(hole_id=h, from_depth=0.0, to_depth=5.0, lithology_code="Sand") for h in holes
    ] + [Lithology(hole_id=h, from_depth=5.0, to_depth=15.0, lithology_code="Clay") for h in holes]
    readings = [
        EnvironmentalReading(hole_id=h, parameter="Chloride", value=v, depth=3.0, unit="mg/L")
        for h, v in zip(holes, (40.0, 180.0, 320.0, 90.0), strict=True)
    ]
    return ParseResult(
        collars=collars, lithologies=lithologies, environmental_readings=readings, errors=()
    )


def _threshold_request() -> SectionBuildRequest:
    return SectionBuildRequest(
        transect_points=((0.0, 0.0), (100.0, 0.0)),
        vertical_exaggeration=2.0,
        show_hatches=False,
        section_title="Site",
        allow_pinch_outs=True,
        environmental_parameters=("Chloride",),
        chemistry_color_mode="threshold",
        chemistry_thresholds=LIMITS,
    )


@pytest.fixture(scope="module")
def packages():
    clear_batch_geometry_memo()
    parse_result = _parse_with_chemistry()
    base = _threshold_request()
    specs = [
        BatchTransectSpec(label="A-A", hole_ids=("BH-01", "BH-02", "BH-03")),
        BatchTransectSpec(label="B-B", hole_ids=("BH-01", "BH-04", "BH-03")),
    ]
    built = [
        build_one_transect_package(parse_result, base, spec, preset_label="Section sheet")
        for spec in specs
    ]
    clear_batch_geometry_memo()
    return built


# --- 1. Batch ZIP mirrors the report ZIP -----------------------------------------


def test_batch_section_metadata_uses_the_single_section_payload(packages) -> None:
    package = packages[0]
    metadata = package.metadata
    assert metadata["chemistry_label_colour_mode"] == "threshold"
    assert metadata["chemistry_label_colour_rule"] == THRESHOLD_COLOUR_RULE
    assert metadata["chemistry_thresholds"] == {
        "Chloride": {"green_max": 100.0, "orange_max": 250.0}
    }
    assert metadata["lithology_scheme_id"] == LITHOLOGY_SCHEME_ID
    assert metadata["vertical_exaggeration"] == 2.0
    assert metadata.get("vertical_exaggeration_drawn"), metadata
    assert metadata["transect_label"] == "A-A"
    assert metadata["hole_count"] == 3
    assert metadata["preset"] == "Section sheet"
    # Same builder as the single-section report ZIP: the same fields.
    single = export_metadata_payload(
        section_title="Site",
        preset_label="Section sheet",
        vertical_exaggeration=2.0,
        hole_count=3,
        transect_label="A-A",
        overlap_warnings=(),
        consulting_fields={
            "figure_number": "",
            "project_number": "",
            "prepared_for": "",
            "prepared_by": "",
            "revised": "",
        },
        chemistry_color_mode="threshold",
        chemistry_thresholds=LIMITS,
        drawn_vertical_exaggeration=str(metadata["vertical_exaggeration_drawn"]),
    )
    assert set(metadata) == set(single)
    json.dumps(metadata)


def test_batch_zip_ships_svg_and_metadata_per_section(packages) -> None:
    entries = [package.entry() for package in packages]
    binder = export_binder_pdf([package.pdf_bytes for package in packages], cover_title="Site")
    with zipfile.ZipFile(io.BytesIO(build_batch_zip(entries, binder_pdf=binder))) as archive:
        names = set(archive.namelist())
        for stem in ("A-A", "B-B"):
            assert {f"{stem}.svg", f"{stem}.png", f"{stem}.pdf", f"{stem}_metadata.json"} <= names
            assert b"<svg" in archive.read(f"{stem}.svg")
            payload = json.loads(archive.read(f"{stem}_metadata.json"))
            assert payload["transect_label"] == stem
            assert payload["chemistry_label_colour_rule"] == THRESHOLD_COLOUR_RULE
            assert payload["chemistry_thresholds"]["Chloride"]["orange_max"] == 250.0
            assert payload["lithology_scheme_id"] == LITHOLOGY_SCHEME_ID
        readme = archive.read("README.txt").decode("utf-8")
    assert "metadata JSON" in readme and "SVG" in readme


def test_legacy_four_tuple_entries_still_zip_without_metadata() -> None:
    names = zipfile.ZipFile(io.BytesIO(build_batch_zip([("A", b"<svg/>", b"png", b"%PDF")]))).namelist()
    assert sorted(names) == ["A.pdf", "A.png", "A.svg", "README.txt"]


def test_binder_summary_page_lists_rule_and_limits(packages) -> None:
    pypdf = pytest.importorskip("pypdf")
    lines = binder_summary_lines(packages[0].metadata)
    assert lines[0].startswith("Chemistry label colours: Green / orange / red thresholds")
    assert THRESHOLD_COLOUR_RULE in lines[0]
    assert "Chloride: green ≤ 100, orange ≤ 250, red above 250" in lines
    assert f"Lithology colour scheme: {LITHOLOGY_SCHEME_ID}" in lines
    binder = export_binder_pdf(
        [package.pdf_bytes for package in packages], cover_title="Site", summary_lines=lines
    )
    cover_text = pypdf.PdfReader(io.BytesIO(binder)).pages[0].extract_text()
    assert "Chloride" in cover_text and "250" in cover_text
    assert "red above" in cover_text


def test_binder_summary_two_band_and_no_limit_lines() -> None:
    metadata = export_metadata_payload(
        section_title="S",
        preset_label=None,
        vertical_exaggeration=None,
        hole_count=2,
        transect_label=None,
        overlap_warnings=(),
        chemistry_color_mode="black_red",
        chemistry_thresholds={"Chloride": (100.0, 100.0), "Benzene": (None, None)},
    )
    lines = binder_summary_lines(metadata)
    assert lines[0] == (
        "Chemistry label colours: Black / red (exceedance) (black at or below the limit, red above)"
    )
    assert "Chloride: red above 100" in lines and "Benzene: no limit set" in lines
    # No lab data: the binder lists only the lithology scheme.
    assert binder_summary_lines(metadata, include_chemistry=False) == [
        f"Lithology colour scheme: {LITHOLOGY_SCHEME_ID}"
    ]


def test_metadata_always_names_a_rule_and_the_scheme() -> None:
    black = export_metadata_payload(
        section_title="S", preset_label=None, vertical_exaggeration=1.0, hole_count=1,
        transect_label=None, overlap_warnings=(),
    )
    assert black["chemistry_label_colour_rule"] == "every value black"
    assert black["lithology_scheme_id"] == LITHOLOGY_SCHEME_ID


# --- 2. Parameters to plot: no Select all past the cap ----------------------------


def _six_parameter_workbook() -> bytes:
    import openpyxl

    from workbook_template import build_input_template_bytes

    book = openpyxl.load_workbook(io.BytesIO(build_input_template_bytes()))
    sheet = book["Environmental"]
    header = [cell.value for cell in sheet[1]]
    column = {name: header.index(name) for name in ("hole_id", "parameter", "value", "depth", "unit")}
    for parameter in ("Sulfate", "Sodium", "Nitrate", "TPH"):
        for hole, value in (("MW-01", 10.0), ("MW-02", 20.0), ("MW-03", 30.0)):
            row = [None] * len(header)
            row[column["hole_id"]] = hole
            row[column["parameter"]] = parameter
            row[column["value"]] = value
            row[column["depth"]] = 2.0
            row[column["unit"]] = "mg/L"
            sheet.append(row)
    out = io.BytesIO()
    book.save(out)
    return out.getvalue()


def test_parameter_picker_has_no_select_all_past_the_cap() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    at.file_uploader[0].upload("six.xlsx", _six_parameter_workbook()).run()
    assert not at.exception
    picker = at.multiselect(key="environmental_parameters_multiselect")
    assert len(picker.options) > MAX_CHEMISTRY_PARAMETERS
    assert picker.max_selections == MAX_CHEMISTRY_PARAMETERS
    assert picker.proto.select_all == 0  # "Select all" never shown
    assert len(picker.value) <= MAX_CHEMISTRY_PARAMETERS
    captions = [caption.value for caption in at.caption]
    assert f"Up to {MAX_CHEMISTRY_PARAMETERS} parameters per section." in captions


def test_capped_multiselect_kwargs_match_the_installed_streamlit() -> None:
    import inspect

    import streamlit as st

    from app_configure import capped_multiselect_kwargs

    supported = "select_all" in inspect.signature(st.multiselect).parameters
    assert capped_multiselect_kwargs() == ({"select_all": False} if supported else {})


# --- 3. Template sample agrees with its own limits --------------------------------


def test_template_sample_has_no_contradicting_label_colors() -> None:
    from parsing import DataParser
    from render_profiles import resolved_chemistry_thresholds_for
    from render_theme import CHEMISTRY_LABEL_BLACK, CHEMISTRY_LABEL_RED, chemistry_label_color
    from workbook_template import _sample_environmental, build_input_template_bytes

    assert not any(row.get("label_color") for row in _sample_environmental())
    result = DataParser().parse_file(io.BytesIO(build_input_template_bytes()))
    chloride = [r for r in result.environmental_readings if r.parameter == "Chloride"]
    assert chloride and all(not r.label_color for r in chloride)
    green, orange = 100.0, 250.0
    above = [r for r in chloride if r.value is not None and r.value > orange]
    assert above, "the sample should show one value above its orange limit"
    for reading in above:
        limits = resolved_chemistry_thresholds_for(reading.parameter, LIMITS, green, orange)
        assert chemistry_label_color(
            reading.value, "threshold", green_max=limits[0], yellow_max=limits[1]
        ) == CHEMISTRY_LABEL_RED
    # Default mode: every sample value prints in the one fixed colour.
    assert {chemistry_label_color(r.value, "black") for r in chloride} == {CHEMISTRY_LABEL_BLACK}


# --- 4. Data Entry read path -------------------------------------------------------


def test_load_data_entry_sheets_reads_through_read_data_sheet(monkeypatch) -> None:
    import parsing
    from workbook_template import build_input_template_bytes, load_data_entry_sheets

    calls: list[tuple[str, object]] = []
    original = parsing.read_data_sheet

    def spy(workbook, sheet_name, *, header=0):
        calls.append((sheet_name, header))
        return original(workbook, sheet_name, header=header)

    monkeypatch.setattr(parsing, "read_data_sheet", spy)
    buffer = io.BytesIO(build_input_template_bytes())
    sheets = load_data_entry_sheets(buffer)
    assert calls == [("Data Entry", None)]
    assert sheets.project.get("section_title")
    assert not buffer.closed  # a caller's buffer stays usable (Project tab read next)
