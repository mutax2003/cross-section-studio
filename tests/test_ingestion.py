"""Tests for config-driven workbook ingestion."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion import (
    NATIVE_PROFILE_ID,
    DepthParser,
    FieldExportAdapter,
    FormatDetector,
    ImportProfile,
    load_override,
    load_profile,
    ingest_workbook,
    parse_depth_interval,
    export_platform_workbook,
)
from tests.conftest import make_workbook_bytes

SOURCE = Path(
    r"C:\Users\Andrew Liu\Downloads"
    r"\Boreholes_Advantage_2026_Phase_2_ESA_10011_24_110_08_W6M_062426.xlsx"
)
OUTPUT = ROOT / "data" / "advantage_phase2_platform.xlsx"
SAMPLE_WORKBOOK = ROOT / "data" / "sample_boreholes.xlsx"


def _field_export_bytes(
    rows: list[dict],
    *,
    columns: dict[str, str] | None = None,
    extra_sheets: dict[str, pd.DataFrame] | None = None,
) -> bytes:
    col_map = columns or {
        "hole_id": "Label",
        "depth_interval": "Depth",
        "lithology_code": "Lithology",
        "latitude": "Lat",
        "longitude": "Long",
    }
    sheet_rows = []
    for row in rows:
        sheet_rows.append(
            {
                col_map["hole_id"]: row["hole_id"],
                col_map["depth_interval"]: row["depth"],
                col_map["lithology_code"]: row["lithology"],
                col_map["latitude"]: row["lat"],
                col_map["longitude"]: row["long"],
            }
        )
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(sheet_rows).to_excel(writer, sheet_name="Lithology", index=False)
        if extra_sheets:
            for name, frame in extra_sheets.items():
                frame.to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()


@pytest.fixture
def simple_field_export() -> bytes:
    return _field_export_bytes(
        [
            {
                "hole_id": "BH-01",
                "depth": "0.00-2.00m",
                "lithology": "silty clay",
                "lat": 58.57,
                "long": -119.19,
            },
            {
                "hole_id": "BH-01",
                "depth": "2.00-4.00m",
                "lithology": "sand",
                "lat": 58.57,
                "long": -119.19,
            },
            {
                "hole_id": "BH-02",
                "depth": "0-3m",
                "lithology": "clay",
                "lat": 58.571,
                "long": -119.189,
            },
        ]
    )


def test_load_field_export_profile() -> None:
    profile = load_profile("field_export_v1")
    assert profile.id == "field_export_v1"
    assert profile.depth_format == "interval_string"
    assert profile.columns["hole_id"] == "Label"
    assert profile.coordinates.target_crs == "EPSG:32611"


def test_load_advantage_override() -> None:
    profile = load_override("advantage_phase2_2026")
    assert profile.id == "advantage_phase2_2026"
    assert profile.coordinate_offsets_m["BH26-15"] == [0.5, 0.0]
    assert profile.columns["hole_id"] == "Label"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("0.00-2.00m", (0.0, 2.0)),
        ("2.50-4.00m", (2.5, 4.0)),
        ("0-2", (0.0, 2.0)),
        ("0.00 - 2.00 m", (0.0, 2.0)),
    ],
)
def test_parse_depth_interval_variants(value: str, expected: tuple[float, float]) -> None:
    assert parse_depth_interval(value) == expected


def test_depth_parser_interval_string() -> None:
    parser = DepthParser("interval_string", {"depth_interval": "Depth Range"})
    row = pd.Series({"Depth Range": "1.0-3.5m"})
    assert parser.parse_row(row) == (1.0, 3.5)


def test_format_detector_native(make_workbook_bytes=make_workbook_bytes) -> None:
    data = make_workbook_bytes(
        [{"hole_id": "BH-01", "easting": 1.0, "northing": 2.0, "elevation": 100.0, "total_depth": 5.0}],
        [{"hole_id": "BH-01", "from_depth": 0.0, "to_depth": 5.0, "lithology_code": "Clay"}],
    )
    detection = FormatDetector().detect(BytesIO(data))
    assert detection.is_native
    assert detection.profile_id == NATIVE_PROFILE_ID
    assert detection.confidence >= 0.8


def test_format_detector_field_export(simple_field_export: bytes) -> None:
    detection = FormatDetector().detect(BytesIO(simple_field_export))
    assert not detection.is_native
    assert detection.profile_id == "field_export_v1"
    assert detection.confidence >= 0.9


def test_ingest_field_export_workbook(simple_field_export: bytes) -> None:
    result, report = ingest_workbook(BytesIO(simple_field_export), profile_id="field_export_v1")
    assert report.hole_count == 2
    assert report.lithology_interval_count == 3
    assert report.profile_id == "field_export_v1"
    assert len(result.collars) == 2
    assert all(collar.elevation == 100.0 for collar in result.collars)


def test_field_export_renamed_columns() -> None:
    data = _field_export_bytes(
        [
            {
                "hole_id": "BH-A",
                "depth": "0.00-1.00m",
                "lithology": "clay",
                "lat": 58.57,
                "long": -119.19,
            },
        ],
        columns={
            "hole_id": "BH",
            "depth_interval": "Depth Range",
            "lithology_code": "Lithology",
            "latitude": "Latitude",
            "longitude": "Longitude",
        },
    )
    profile = load_profile("field_export_v1").model_copy(
        update={
            "columns": {
                "hole_id": "BH",
                "depth_interval": "Depth Range",
                "lithology_code": "Lithology",
                "latitude": "Latitude",
                "longitude": "Longitude",
            }
        }
    )
    collars, lithology = FieldExportAdapter().adapt(BytesIO(data), profile)
    assert len(collars) == 1
    assert collars.iloc[0]["hole_id"] == "BH-A"
    assert lithology.iloc[0]["lithology_code"] == "Clay"


def test_field_data_sheet_warning(simple_field_export: bytes) -> None:
    data = _field_export_bytes(
        [
            {
                "hole_id": "BH-01",
                "depth": "0.00-2.00m",
                "lithology": "clay",
                "lat": 58.57,
                "long": -119.19,
            },
        ],
        extra_sheets={"Field Data": pd.DataFrame([{"sample": "OVA", "value": 1.2}])},
    )
    _, report = ingest_workbook(BytesIO(data), profile_id="field_export_v1")
    assert "Field Data" in report.optional_sheets_detected
    assert any("Field Data" in warning for warning in report.warnings)


@pytest.mark.skipif(not SAMPLE_WORKBOOK.exists(), reason="Run scripts/generate_sample_data.py first")
def test_ingest_native_sample_workbook() -> None:
    result, report = ingest_workbook(SAMPLE_WORKBOOK)
    assert report.profile_id == NATIVE_PROFILE_ID
    assert len(result.collars) == 4
    assert report.mapping_proposal is not None


@pytest.mark.skipif(not SOURCE.exists(), reason="Advantage source workbook not in Downloads")
def test_ingest_advantage_workbook() -> None:
    result, report = ingest_workbook(
        SOURCE,
        profile_id="field_export_v1",
        override_id="advantage_phase2_2026",
    )
    assert len(result.collars) == 23
    assert len(result.lithologies) == 70
    assert report.coordinate_offsets_applied.get("BH26-15") == (0.5, 0.0)

    c13 = next(c for c in result.collars if c.hole_id == "BH26-13")
    c15 = next(c for c in result.collars if c.hole_id == "BH26-15")
    assert (c13.easting, c13.northing) != (c15.easting, c15.northing)


@pytest.mark.skipif(not SOURCE.exists(), reason="Advantage source workbook not in Downloads")
def test_export_advantage_via_generic_cli_path(tmp_path: Path) -> None:
    out = tmp_path / "converted.xlsx"
    collars, lithology = export_platform_workbook(
        SOURCE,
        out,
        profile_id="field_export_v1",
        override_id="advantage_phase2_2026",
    )
    assert out.exists()
    assert len(collars) == 23
    assert len(lithology) == 70


@pytest.mark.skipif(not SOURCE.exists(), reason="Advantage source workbook not in Downloads")
def test_auto_detect_advantage_source() -> None:
    detection = FormatDetector().detect(SOURCE)
    assert detection.profile_id == "field_export_v1"


def test_unsupported_workbook_raises() -> None:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame([{"foo": 1}]).to_excel(writer, sheet_name="Random", index=False)
    with pytest.raises(ValueError, match="Could not detect"):
        FormatDetector().detect(BytesIO(buffer.getvalue()))


def test_field_data_total_depth_applied() -> None:
    raw = _field_export_bytes(
        [
            {
                "hole_id": "BH-01",
                "depth": "0.00-4.00m",
                "lithology": "clay",
                "lat": 58.57,
                "long": -119.19,
            },
        ],
        extra_sheets={
            "Field Data": pd.DataFrame([{"Label": "BH-01", "total_depth": 20.0}]),
        },
    )
    profile = load_profile("field_export_v1")
    collars_df, _ = FieldExportAdapter().adapt(BytesIO(raw), profile)
    td = float(collars_df.loc[collars_df["hole_id"] == "BH-01", "total_depth"].iloc[0])
    assert td == 20.0


def test_ingest_warns_on_placeholder_elevation(simple_field_export: bytes) -> None:
    _, report = ingest_workbook(
        BytesIO(simple_field_export),
        profile_id="field_export_v1",
    )
    assert any("elevation uses profile default" in warning.lower() for warning in report.warnings)

