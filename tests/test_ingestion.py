"""Tests for config-driven workbook ingestion."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion import (  # noqa: E402
    NATIVE_PROFILE_ID,
    DepthParser,
    FieldExportAdapter,
    FormatDetector,
    export_platform_workbook,
    ingest_workbook,
    load_override,
    load_profile,
    parse_depth_interval,
)
from models import DataParser  # noqa: E402
from paths import advantage_platform_workbook, advantage_source_workbook  # noqa: E402
from tests.conftest import make_workbook_bytes  # noqa: E402

SOURCE = advantage_source_workbook()
OUTPUT = advantage_platform_workbook()
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
    "bad_id",
    [
        "../etc/passwd",
        "..\\windows\\system32",
        "foo/bar",
        "foo\\bar",
        "",
        "a\x00b",
    ],
)
def test_load_profile_rejects_path_traversal(bad_id: str) -> None:
    with pytest.raises(ValueError, match="Invalid profile id"):
        load_profile(bad_id)


@pytest.mark.parametrize(
    "bad_id",
    ["../secret", "a/b", "x\\y"],
)
def test_load_override_rejects_path_traversal(bad_id: str) -> None:
    with pytest.raises(ValueError, match="Invalid override id"):
        load_override(bad_id)



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
    codes = {item.lithology_code for item in result.lithologies}
    assert "Sand" in codes
    assert "Sandstone" not in codes
    sand_intervals = [item for item in result.lithologies if item.lithology_code == "Sand"]
    assert len(sand_intervals) == 1
    assert sand_intervals[0].from_depth == 2.0


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
    collars, lithology, _ = FieldExportAdapter().adapt(BytesIO(data), profile)
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
    assert not any("future work" in warning.lower() for warning in report.warnings)


def test_field_data_ova_ec_readings_advantage_shape() -> None:
    """Advantage-style Field Data (Label, Depth interval, OVA, EC) → environmental readings."""
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
        extra_sheets={
            "Field Data": pd.DataFrame(
                [
                    {
                        "Label": "BH-01",
                        "Depth": "0.00-0.15m",
                        "OVA": 125.0,
                        "EC": 0.42,
                    },
                    {
                        "Label": "BH-01",
                        "Depth": "0.15-0.30m",
                        "OVA": 80.0,
                        "EC": None,
                    },
                ]
            )
        },
    )
    result, report = ingest_workbook(BytesIO(data), profile_id="field_export_v1")
    assert "Field Data" in report.optional_sheets_detected
    assert any("parsed 3 OVA/EC reading(s)" in warning for warning in report.warnings)
    assert not any("future work" in warning.lower() for warning in report.warnings)

    ova = [r for r in result.environmental_readings if r.parameter == "OVA"]
    ec = [r for r in result.environmental_readings if r.parameter == "EC"]
    assert len(ova) == 2
    assert len(ec) == 1
    assert ova[0].from_depth == pytest.approx(0.0)
    assert ova[0].to_depth == pytest.approx(0.15)
    assert ova[0].value == pytest.approx(125.0)
    assert ova[0].unit == "ppm"
    assert ec[0].value == pytest.approx(0.42)
    assert ec[0].unit == ""
    # Interval Depth must not become collar total_depth (lithology max remains).
    assert result.collars[0].total_depth == pytest.approx(2.0)


def test_field_export_ingest_keeps_water_overlay() -> None:
    """Adapted collars/lithology must not drop optional Water sheet rows."""
    data = _field_export_bytes(
        [
            {
                "hole_id": "BH-01",
                "depth": "0.00-2.00m",
                "lithology": "clay",
                "lat": 58.57,
                "long": -119.19,
            },
            {
                "hole_id": "BH-01",
                "depth": "2.00-5.00m",
                "lithology": "sand",
                "lat": 58.57,
                "long": -119.19,
            },
        ],
        extra_sheets={
            "Water": pd.DataFrame(
                [{"hole_id": "BH-01", "depth": 1.5, "series_id": "2025-06", "series_label": "June"}]
            )
        },
    )
    result, report = ingest_workbook(BytesIO(data), profile_id="field_export_v1")
    assert report.profile_id == "field_export_v1"
    assert len(result.collars) == 1
    assert len(result.water_levels) == 1
    assert result.water_levels[0].hole_id == "BH-01"
    assert abs(result.water_levels[0].depth - 1.5) < 1e-9


@pytest.mark.skipif(not SAMPLE_WORKBOOK.exists(), reason="Run scripts/generate_sample_data.py first")
def test_ingest_native_sample_workbook() -> None:
    result, report = ingest_workbook(SAMPLE_WORKBOOK)
    assert report.profile_id == NATIVE_PROFILE_ID
    assert len(result.collars) == 4
    assert report.mapping_proposal is not None


def test_ingest_advantage_workbook() -> None:
    assert SOURCE.exists(), "Commit data/fixtures/advantage_phase2_source.xlsx (synthetic CI fixture)"
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


def test_ingest_advantage_platform_workbook() -> None:
    assert OUTPUT.exists(), "Commit data/advantage_phase2_platform.xlsx (synthetic CI fixture)"
    result, report = ingest_workbook(OUTPUT)
    assert len(result.collars) >= 20
    assert len(result.lithologies) >= 50
    assert report.profile_id == NATIVE_PROFILE_ID


def test_export_advantage_via_generic_cli_path(tmp_path: Path) -> None:
    assert SOURCE.exists(), "Commit data/fixtures/advantage_phase2_source.xlsx (synthetic CI fixture)"
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


def test_auto_detect_advantage_source() -> None:
    assert SOURCE.exists(), "Commit data/fixtures/advantage_phase2_source.xlsx (synthetic CI fixture)"
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
    collars_df, _, _ = FieldExportAdapter().adapt(BytesIO(raw), profile)
    td = float(collars_df.loc[collars_df["hole_id"] == "BH-01", "total_depth"].iloc[0])
    assert td == 20.0


def test_ingest_warns_on_placeholder_elevation(simple_field_export: bytes) -> None:
    _, report = ingest_workbook(
        BytesIO(simple_field_export),
        profile_id="field_export_v1",
    )
    assert any("elevation uses profile default" in warning.lower() for warning in report.warnings)


def test_screens_and_gradients_sheets_parsed() -> None:
    from models import DataParser

    collars = [
        {
            "hole_id": "MW-01",
            "easting": 0.0,
            "northing": 0.0,
            "elevation": 100.0,
            "total_depth": 20.0,
        },
        {
            "hole_id": "MW-02",
            "easting": 50.0,
            "northing": 0.0,
            "elevation": 100.0,
            "total_depth": 18.0,
        },
    ]
    lithology = [
        {
            "hole_id": "MW-01",
            "from_depth": 0.0,
            "to_depth": 10.0,
            "lithology_code": "Sand",
        },
        {
            "hole_id": "MW-02",
            "from_depth": 0.0,
            "to_depth": 10.0,
            "lithology_code": "Clay",
        },
    ]
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(collars).to_excel(writer, sheet_name="Collars", index=False)
        pd.DataFrame(lithology).to_excel(writer, sheet_name="Lithology", index=False)
        pd.DataFrame(
            [{"hole_id": "MW-01", "from_depth": 4.0, "to_depth": 8.0}]
        ).to_excel(writer, sheet_name="Screens", index=False)
        pd.DataFrame([{"hole_id": "MW-02", "direction": "up"}]).to_excel(
            writer, sheet_name="Gradients", index=False
        )
    result = DataParser().parse_file(BytesIO(buffer.getvalue()))
    assert len(result.screen_intervals) == 1
    assert result.screen_intervals[0].hole_id == "MW-01"
    assert result.screen_intervals[0].from_depth == 4.0
    assert len(result.vertical_gradients) == 1
    assert result.vertical_gradients[0].hole_id == "MW-02"
    assert result.vertical_gradients[0].direction == "up"


def test_parse_file_reuses_open_excel_file() -> None:
    result = DataParser().parse_file(pd.ExcelFile(SAMPLE_WORKBOOK))
    assert result.collars
    assert result.lithologies


def test_screens_and_gradients_unknown_hole_surface_errors() -> None:
    from models import DataParser

    collars = [
        {
            "hole_id": "MW-01",
            "easting": 0.0,
            "northing": 0.0,
            "elevation": 100.0,
            "total_depth": 20.0,
        },
    ]
    lithology = [
        {
            "hole_id": "MW-01",
            "from_depth": 0.0,
            "to_depth": 10.0,
            "lithology_code": "Sand",
        },
    ]
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(collars).to_excel(writer, sheet_name="Collars", index=False)
        pd.DataFrame(lithology).to_excel(writer, sheet_name="Lithology", index=False)
        pd.DataFrame(
            [
                {"hole_id": "MW-01", "from_depth": 4.0, "to_depth": 8.0},
                {"hole_id": "MISSING", "from_depth": 1.0, "to_depth": 2.0},
            ]
        ).to_excel(writer, sheet_name="Screens", index=False)
        pd.DataFrame(
            [
                {"hole_id": "MW-01", "direction": "up"},
                {"hole_id": "GHOST", "direction": "down"},
            ]
        ).to_excel(writer, sheet_name="Gradients", index=False)
    result = DataParser().parse_file(BytesIO(buffer.getvalue()))
    assert len(result.screen_intervals) == 1
    assert len(result.vertical_gradients) == 1
    assert any("unknown hole_id 'MISSING'" in message for message in result.errors)
    assert any("unknown hole_id 'GHOST'" in message for message in result.errors)


def test_environmental_unknown_hole_surfaces_errors() -> None:
    from models import DataParser

    collars = [
        {
            "hole_id": "MW-01",
            "easting": 0.0,
            "northing": 0.0,
            "elevation": 100.0,
            "total_depth": 20.0,
        },
    ]
    lithology = [
        {
            "hole_id": "MW-01",
            "from_depth": 0.0,
            "to_depth": 10.0,
            "lithology_code": "Sand",
        },
    ]
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(collars).to_excel(writer, sheet_name="Collars", index=False)
        pd.DataFrame(lithology).to_excel(writer, sheet_name="Lithology", index=False)
        pd.DataFrame(
            [
                {
                    "hole_id": "MW-01",
                    "parameter": "Cl",
                    "value": 10.0,
                    "depth": 5.0,
                    "unit": "mg/L",
                },
                {
                    "hole_id": "ORPHAN",
                    "parameter": "Cl",
                    "value": 20.0,
                    "depth": 5.0,
                    "unit": "mg/L",
                },
            ]
        ).to_excel(writer, sheet_name="Environmental", index=False)
    result = DataParser().parse_file(BytesIO(buffer.getvalue()))
    assert len(result.environmental_readings) == 1
    assert any("unknown hole_id 'ORPHAN'" in message for message in result.errors)


def test_water_sheet_parses_gw_series_columns(tmp_path) -> None:
    workbook = tmp_path / "water_series.xlsx"
    collars = [
        {
            "hole_id": "MW-01",
            "easting": 0.0,
            "northing": 0.0,
            "elevation": 632.0,
            "total_depth": 30.0,
        },
    ]
    lithology = [
        {
            "hole_id": "MW-01",
            "from_depth": 0.0,
            "to_depth": 30.0,
            "lithology_code": "Clay",
        },
    ]
    water = [
        {
            "hole_id": "MW-01",
            "depth": 1.0,
            "series_id": "2024-05",
            "series_label": "May 2024",
        },
        {
            "hole_id": "MW-01",
            "depth": 1.2,
            "series_id": "2025-06",
            "series_label": "June 2025",
        },
    ]
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame(collars).to_excel(writer, sheet_name="Collars", index=False)
        pd.DataFrame(lithology).to_excel(writer, sheet_name="Lithology", index=False)
        pd.DataFrame(water).to_excel(writer, sheet_name="Water", index=False)
    result = DataParser().parse_file(workbook)
    assert len(result.water_levels) == 2
    series_ids = {level.series_id for level in result.water_levels}
    assert series_ids == {"2024-05", "2025-06"}
    assert result.water_levels[0].series_label in {"May 2024", "June 2025"}


def test_water_sheet_accepts_elevation_masl(tmp_path: Path) -> None:
    workbook = tmp_path / "water_masl.xlsx"
    collars = [
        {
            "hole_id": "MW-01",
            "easting": 0.0,
            "northing": 0.0,
            "elevation": 100.0,
            "total_depth": 30.0,
        },
    ]
    lithology = [
        {
            "hole_id": "MW-01",
            "from_depth": 0.0,
            "to_depth": 30.0,
            "lithology_code": "Clay",
        },
    ]
    water = [{"hole_id": "MW-01", "elevation_masl": 97.5, "series_id": "2024-06"}]
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame(collars).to_excel(writer, sheet_name="Collars", index=False)
        pd.DataFrame(lithology).to_excel(writer, sheet_name="Lithology", index=False)
        pd.DataFrame(water).to_excel(writer, sheet_name="Water", index=False)
    result = DataParser().parse_file(workbook)
    assert len(result.water_levels) == 1
    assert result.water_levels[0].depth == pytest.approx(2.5)
    assert result.water_levels[0].elevation_masl == pytest.approx(97.5)


def test_water_sheet_rejects_depth_and_masl_together(tmp_path: Path) -> None:
    workbook = tmp_path / "water_both.xlsx"
    collars = [
        {
            "hole_id": "MW-01",
            "easting": 0.0,
            "northing": 0.0,
            "elevation": 100.0,
            "total_depth": 30.0,
        },
    ]
    lithology = [
        {
            "hole_id": "MW-01",
            "from_depth": 0.0,
            "to_depth": 30.0,
            "lithology_code": "Clay",
        },
    ]
    water = [{"hole_id": "MW-01", "depth": 2.0, "elevation_masl": 98.0}]
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame(collars).to_excel(writer, sheet_name="Collars", index=False)
        pd.DataFrame(lithology).to_excel(writer, sheet_name="Lithology", index=False)
        pd.DataFrame(water).to_excel(writer, sheet_name="Water", index=False)
    result = DataParser().parse_file(workbook)
    assert any("not both" in error for error in result.errors)


def test_water_sheet_rejects_artesian_elevation_masl(tmp_path: Path) -> None:
    workbook = tmp_path / "water_artesian.xlsx"
    collars = [
        {
            "hole_id": "MW-01",
            "easting": 0.0,
            "northing": 0.0,
            "elevation": 100.0,
            "total_depth": 30.0,
        },
    ]
    lithology = [
        {
            "hole_id": "MW-01",
            "from_depth": 0.0,
            "to_depth": 30.0,
            "lithology_code": "Clay",
        },
    ]
    water = [{"hole_id": "MW-01", "elevation_masl": 101.5, "series_id": "2024-06"}]
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame(collars).to_excel(writer, sheet_name="Collars", index=False)
        pd.DataFrame(lithology).to_excel(writer, sheet_name="Lithology", index=False)
        pd.DataFrame(water).to_excel(writer, sheet_name="Water", index=False)
    result = DataParser().parse_file(workbook)
    assert len(result.water_levels) == 0
    assert any("artesian" in error.lower() or "above-collar" in error.lower() for error in result.errors)
    assert any("Water row" in error for error in result.errors)


def test_environmental_sheet_accepts_depth_column(tmp_path: Path) -> None:
    workbook = tmp_path / "environmental_depth.xlsx"
    collars = [
        {
            "hole_id": "MW-01",
            "easting": 0.0,
            "northing": 0.0,
            "elevation": 100.0,
            "total_depth": 30.0,
        },
        {
            "hole_id": "MW-02",
            "easting": 50.0,
            "northing": 0.0,
            "elevation": 100.0,
            "total_depth": 30.0,
        },
    ]
    lithology = [
        {
            "hole_id": "MW-01",
            "from_depth": 0.0,
            "to_depth": 30.0,
            "lithology_code": "Clay",
        },
        {
            "hole_id": "MW-02",
            "from_depth": 0.0,
            "to_depth": 30.0,
            "lithology_code": "Clay",
        },
    ]
    environmental = [
        {"hole_id": "MW-01", "depth": 3.5, "parameter": "Chloride", "value": 120.0, "unit": "mg/L"},
        {"hole_id": "MW-02", "depth": 3.5, "parameter": "Chloride", "value": 85.0, "unit": "mg/L"},
    ]
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame(collars).to_excel(writer, sheet_name="Collars", index=False)
        pd.DataFrame(lithology).to_excel(writer, sheet_name="Lithology", index=False)
        pd.DataFrame(environmental).to_excel(writer, sheet_name="Environmental", index=False)
    result = DataParser().parse_file(workbook)
    assert len(result.environmental_readings) == 2
    assert result.environmental_readings[0].sample_depth == pytest.approx(3.5)
    assert {reading.parameter for reading in result.environmental_readings} == {"Chloride"}


def test_data_entry_water_precedence_warns_when_native_ignored(tmp_path: Path) -> None:
    """Data Entry water wins; native Water rows are ignored with a parse warning."""
    workbook = tmp_path / "data_entry_water_precedence.xlsx"
    collars = [
        {
            "hole_id": "MW-01",
            "easting": 0.0,
            "northing": 0.0,
            "elevation": 100.0,
            "total_depth": 30.0,
        },
    ]
    lithology = [
        {
            "hole_id": "MW-01",
            "from_depth": 0.0,
            "to_depth": 30.0,
            "lithology_code": "Clay",
        },
    ]
    native_water = [{"hole_id": "MW-01", "depth": 9.9, "series_id": "native"}]
    data_entry_rows = [
        ["WATER (optional)", "", "", "", ""],
        ["hole_id", "depth", "elevation_masl", "series_id", "series_label"],
        ["MW-01", 1.5, "", "data-entry", "From Data Entry"],
    ]
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame(collars).to_excel(writer, sheet_name="Collars", index=False)
        pd.DataFrame(lithology).to_excel(writer, sheet_name="Lithology", index=False)
        pd.DataFrame(native_water).to_excel(writer, sheet_name="Water", index=False)
        pd.DataFrame(data_entry_rows).to_excel(
            writer, sheet_name="Data Entry", index=False, header=False
        )
    result = DataParser().parse_file(workbook)
    assert len(result.water_levels) == 1
    assert result.water_levels[0].series_id == "data-entry"
    assert result.water_levels[0].depth == pytest.approx(1.5)
    assert any("Native Water sheet was ignored" in message for message in result.errors)


def test_sections_sheet_parses_and_seeds_batch_lines(tmp_path: Path) -> None:
    from batch_export import parse_batch_transect_lines
    from parse_ops import format_section_specs_as_batch_text

    workbook = tmp_path / "sections.xlsx"
    collars = [
        {"hole_id": "MW-01", "easting": 0.0, "northing": 0.0, "elevation": 100.0, "total_depth": 10.0},
        {"hole_id": "MW-02", "easting": 10.0, "northing": 0.0, "elevation": 100.0, "total_depth": 10.0},
        {"hole_id": "MW-03", "easting": 20.0, "northing": 0.0, "elevation": 100.0, "total_depth": 10.0},
    ]
    lithology = [
        {"hole_id": hid, "from_depth": 0.0, "to_depth": 10.0, "lithology_code": "Clay"}
        for hid in ("MW-01", "MW-02", "MW-03")
    ]
    sections = [
        {"section_label": "A-A'", "hole_ids": "MW-01, MW-02, MW-03"},
        {"section_label": "B-B'", "hole_ids": "MW-01→MW-03"},
        {"section_label": "C-C'", "hole_ids": "MW-01; MW-02"},
        {"section_label": "Bad", "hole_ids": "MW-01"},  # <2 holes — skip
        {"section_label": "Orphan", "hole_ids": "MW-01, GHOST"},  # unknown collar — skip
        {"section_label": "Inject|Bad", "hole_ids": "MW-01, MW-02"},  # label injection — skip
    ]
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame(collars).to_excel(writer, sheet_name="Collars", index=False)
        pd.DataFrame(lithology).to_excel(writer, sheet_name="Lithology", index=False)
        pd.DataFrame(sections).to_excel(writer, sheet_name="Sections", index=False)

    result, report = ingest_workbook(workbook)
    assert "Sections" in report.optional_sheets_detected
    assert len(result.section_specs) == 3
    assert result.section_specs[0].hole_ids == ("MW-01", "MW-02", "MW-03")
    assert result.section_specs[1].hole_ids == ("MW-01", "MW-03")
    assert result.section_specs[2].hole_ids == ("MW-01", "MW-02")
    assert report.section_specs == list(result.section_specs)
    assert any("Sections sheet: loaded 3" in warning for warning in report.warnings)

    batch_text = format_section_specs_as_batch_text(result.section_specs)
    specs = parse_batch_transect_lines(batch_text)
    assert [spec.label for spec in specs] == ["A-A'", "B-B'", "C-C'"]
    assert specs[0].hole_ids == ("MW-01", "MW-02", "MW-03")



def test_converted_field_export_keeps_placeholder_elevation_flag(tmp_path: Path) -> None:
    """convert_workbook writes the profile placeholder RL; a re-import must
    still flag it (it used to come back as if surveyed, silently)."""
    from ingestion import export_platform_workbook

    source = ROOT / "data" / "fixtures" / "advantage_phase2_source.xlsx"
    first, first_report = ingest_workbook(source)
    assert first_report.uses_placeholder_elevation

    converted = tmp_path / "converted.xlsx"
    export_platform_workbook(source, converted)
    _again, again_report = ingest_workbook(converted)
    assert again_report.uses_placeholder_elevation
    assert any("placeholder" in warning.lower() for warning in again_report.warnings)

    surveyed = tmp_path / "surveyed.xlsx"
    export_platform_workbook(source, surveyed, elevation_m=612.5)
    _s, surveyed_report = ingest_workbook(surveyed)
    assert not surveyed_report.uses_placeholder_elevation


def _edit_converted_collars(converted: Path, out: Path, edit) -> Path:
    import openpyxl

    book = openpyxl.load_workbook(converted)
    sheet = book["Collars"]
    header = [cell.value for cell in sheet[1]]
    for row in range(2, sheet.max_row + 1):
        edit(sheet, row, header)
    book.save(out)
    return out


def test_placeholder_tag_edited_in_excel_never_drops_or_misflags_collars(tmp_path: Path) -> None:
    """Users fix converter placeholders by clearing the tag or typing surveyed
    RLs; neither may drop collars, keep a stale flag, or print a stale datum."""
    from io import BytesIO

    from ingestion import export_platform_workbook
    from workbook_template import export_cleaned_workbook_bytes

    source = ROOT / "data" / "fixtures" / "advantage_phase2_source.xlsx"
    converted = tmp_path / "converted.xlsx"
    export_platform_workbook(source, converted)
    base, _ = ingest_workbook(converted)

    def clear_tag(sheet, row, header):
        sheet.cell(row, header.index("elevation_datum") + 1).value = None

    cleared, cleared_report = ingest_workbook(
        _edit_converted_collars(converted, tmp_path / "cleared.xlsx", clear_tag)
    )
    assert len(cleared.collars) == len(base.collars)
    assert not cleared_report.uses_placeholder_elevation

    def flat_survey(sheet, row, header):
        sheet.cell(row, header.index("elevation") + 1).value = 612.5

    flat, flat_report = ingest_workbook(
        _edit_converted_collars(converted, tmp_path / "flat.xlsx", flat_survey)
    )
    assert not flat_report.uses_placeholder_elevation
    assert all(collar.elevation_datum is None for collar in flat.collars)

    def own_datum(sheet, row, header):
        sheet.cell(row, header.index("elevation_datum") + 1).value = "Placeholders removed - CGVD2013"
        sheet.cell(row, header.index("elevation") + 1).value = 100.0

    own, own_report = ingest_workbook(
        _edit_converted_collars(converted, tmp_path / "own.xlsx", own_datum)
    )
    assert not own_report.uses_placeholder_elevation
    assert own.collars[0].elevation_datum == "Placeholders removed - CGVD2013"

    # Validate's cleaned export must carry the tag through a re-import.
    _, base_report = ingest_workbook(converted)
    cleaned = export_cleaned_workbook_bytes(base, project_metadata=base_report.project_metadata)
    _again, again_report = ingest_workbook(BytesIO(cleaned))
    assert again_report.uses_placeholder_elevation


def _hostile_workbook(collars, lithology, extra: dict | None = None, **names) -> BytesIO:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(collars).to_excel(
            writer, sheet_name=names.get("collars_sheet", "Collars"), index=False
        )
        pd.DataFrame(lithology).to_excel(
            writer, sheet_name=names.get("lithology_sheet", "Lithology"), index=False
        )
        for sheet, rows in (extra or {}).items():
            pd.DataFrame(rows).to_excel(writer, sheet_name=sheet, index=False)
    buffer.seek(0)
    return buffer


_TWO_COLLARS = [
    {"hole_id": "BH1", "easting": 0, "northing": 0, "elevation": 100, "total_depth": 10},
    {"hole_id": "BH2", "easting": 50, "northing": 0, "elevation": 100, "total_depth": 10},
]
_TWO_LITH = [
    {"hole_id": hole, "from_depth": 0, "to_depth": 10, "lithology_code": "Clay"}
    for hole in ("BH1", "BH2")
]


def test_lowercase_tab_names_and_whitespace_headers_parse_like_the_canonical_ones() -> None:
    """Detection is case/whitespace-insensitive; the parser must read the same
    sheet and header it detected instead of failing on the literal name."""
    result, _ = ingest_workbook(
        _hostile_workbook(
            _TWO_COLLARS, _TWO_LITH, collars_sheet="collars", lithology_sheet="lithology"
        )
    )
    assert len(result.collars) == 2 and len(result.lithologies) == 2

    collars = [
        {**row, "Hole  ID": row.pop("hole_id"), "Total\tDepth": row.pop("total_depth")}
        for row in [dict(r) for r in _TWO_COLLARS]
    ]
    lith = [{**row, "To  Depth": row.pop("to_depth")} for row in [dict(r) for r in _TWO_LITH]]
    result, _ = ingest_workbook(_hostile_workbook(collars, lith))
    assert len(result.collars) == 2 and len(result.lithologies) == 2


def test_numeric_hole_ids_survive_a_blank_row_in_lithology() -> None:
    """pandas reads 1, <blank>, 2 as floats; "1.0" must not fail to match "1"."""
    collars = [{**row, "hole_id": index + 1} for index, row in enumerate(_TWO_COLLARS)]
    lith = [
        {"hole_id": 1, "from_depth": 0, "to_depth": 10, "lithology_code": "Clay"},
        {"hole_id": None, "from_depth": None, "to_depth": None, "lithology_code": None},
        {"hole_id": 2, "from_depth": 0, "to_depth": 10, "lithology_code": "Clay"},
    ]
    result, _ = ingest_workbook(_hostile_workbook(collars, lith))
    assert [c.hole_id for c in result.collars] == ["1", "2"]
    assert len(result.lithologies) == 2 and not result.errors


def test_spreadsheet_formulas_are_rejected_and_never_written_back_live() -> None:
    from openpyxl import load_workbook

    from workbook_template import export_cleaned_workbook_bytes

    hostile = [dict(_TWO_COLLARS[0], hole_id='=HYPERLINK("http://evil","BH1")'), _TWO_COLLARS[1]]
    result, _ = ingest_workbook(_hostile_workbook(hostile, _TWO_LITH))
    # Written by openpyxl as a formula with no cached value -> reads back blank:
    # the row must be reported, not silently dropped.
    assert len(result.collars) == 1
    assert any("Collars row 2" in error and "blank" in error for error in result.errors)

    # Text that merely starts with "=" (typed into Excel as a string).
    from models import Collar

    with pytest.raises(ValueError, match="formula"):
        Collar(hole_id="=1+1", easting=0, northing=0, elevation=1, total_depth=1)

    clean, report = ingest_workbook(_hostile_workbook(_TWO_COLLARS, _TWO_LITH))
    exported = export_cleaned_workbook_bytes(
        clean, project_metadata={"client_name": "=cmd|' /C calc'!A0"}
    )
    project = load_workbook(BytesIO(exported))["Project"]
    formula_cells = [c for row in project.iter_rows() for c in row if str(c.value).startswith("=")]
    assert formula_cells and all(c.data_type == "s" for c in formula_cells)


def test_structure_and_survey_row_problems_are_row_errors_not_aborts() -> None:
    extra = {
        "Faults": [
            {"name": "F1", "x_profile": None, "elevation": 5},
            {"name": "F1", "x_profile": 10, "elevation": 5},
        ],
        "Unconformities": [{"name": "U1", "x_profile": 1, "elevation": None}],
        "Deviations": [{"hole_id": "BH1", "depth": 5, "inclination_deg": None, "azimuth_deg": 0}],
    }
    result, _ = ingest_workbook(_hostile_workbook(_TWO_COLLARS, _TWO_LITH, extra))
    assert len(result.collars) == 2  # workbook still loads
    assert not result.deviation_readings  # NaN survey would NaN the hole geometry
    joined = "\n".join(result.errors)
    assert "Faults sheet" in joined and "Unconformities sheet" in joined
    assert "Deviations row for 'BH1': inclination_deg" in joined


def test_oversized_formatted_workbook_is_rejected_before_parsing(tmp_path: Path) -> None:
    """A million empty-but-styled rows used to tie the parser up for ~18 s."""
    from parsing import WorkbookTooLargeError, check_workbook_row_counts

    buffer = _hostile_workbook(_TWO_COLLARS, _TWO_LITH)
    check_workbook_row_counts(buffer)  # normal workbook passes
    assert buffer.tell() == 0  # stream rewound for the real reader

    with pytest.raises(WorkbookTooLargeError, match="rows \\(limit 10\\)"):
        check_workbook_row_counts(_hostile_workbook(_TWO_COLLARS * 6, _TWO_LITH), limit=10)


def test_sections_rows_naming_unknown_holes_are_reported_not_silently_dropped() -> None:
    """A Sections row with a misspelt hole used to vanish with only a log line,
    which read as 'the program did not read that tab'."""
    sections = [
        {"section_label": "A-A'", "hole_ids": "BH1, BH2"},
        {"section_label": "B-B'", "hole_ids": "BH1, BH-99"},
        {"section_label": "C-C'", "hole_ids": "BH1"},
        {"section_label": "D-D'", "hole_ids": "BH1, BH2, BH1"},
        {"section_label": "Custom", "hole_ids": "BH2, BH1"},
    ]
    result, report = ingest_workbook(_hostile_workbook(_TWO_COLLARS, _TWO_LITH, {"Sections": sections}))
    assert [spec.label for spec in result.section_specs] == ["A-A'", "Custom"]  # "Custom" is a valid label
    joined = "\n".join(result.errors)
    assert "Sections row 3 (B-B'): unknown collar(s) BH-99" in joined
    assert "Sections row 4" in joined
    assert "Sections row 5" in joined and "listed more than once: BH1" in joined


def test_cleaned_export_keeps_every_ingested_field() -> None:
    """Deviated-hole angles, hatch patterns, water styling and the structural
    sheets (Deviations, Faults, Unconformities) used to vanish on re-upload."""
    from workbook_template import export_cleaned_workbook_bytes

    collars = [
        {**_TWO_COLLARS[0], "inclination_deg": -85.0, "azimuth_deg": 45.0, "stick_up_m": 0.6},
        {**_TWO_COLLARS[1], "inclination_deg": -90.0, "azimuth_deg": 0.0},
    ]
    lith = [{**row, "hatch_pattern": "///"} for row in _TWO_LITH]
    extra = {
        "Water": [{"hole_id": "BH1", "depth": 2.5, "series_id": "s1", "color": "#ff0000", "marker": "v"}],
        "Deviations": [{"hole_id": "BH1", "depth": 5.0, "inclination_deg": -80.0, "azimuth_deg": 40.0}],
        "Faults": [{"name": "F1", "x_profile": 0.0, "elevation": 95.0}, {"name": "F1", "x_profile": 50.0, "elevation": 90.0}],
        "Unconformities": [{"name": "U1", "x_profile": 0.0, "elevation": 97.0}, {"name": "U1", "x_profile": 50.0, "elevation": 96.0}],
    }
    first, report = ingest_workbook(_hostile_workbook(collars, lith, extra))
    assert first.deviation_readings and first.faults and first.unconformities
    again, _ = ingest_workbook(BytesIO(export_cleaned_workbook_bytes(first, project_metadata=report.project_metadata)))
    assert [(c.inclination_deg, c.azimuth_deg, c.stick_up_m) for c in again.collars] == [
        (c.inclination_deg, c.azimuth_deg, c.stick_up_m) for c in first.collars
    ]
    assert [i.hatch_pattern for i in again.lithologies] == ["///", "///"]
    assert (again.water_levels[0].color, again.water_levels[0].marker) == ("#ff0000", "v")
    assert again.deviation_readings == first.deviation_readings
    assert again.faults == first.faults and again.unconformities == first.unconformities


def test_data_entry_project_number_row_is_a_field_not_a_block_header() -> None:
    from workbook_template import parse_data_entry_sheet

    sheets = parse_data_entry_sheet(
        pd.DataFrame([["PROJECT / CLIENT METADATA", "value"], ["project_number", "P-123"], ["client_name", "ACME"]])
    )
    assert sheets.project.get("project_number") == "P-123" and sheets.project.get("client_name") == "ACME"


def test_row_cap_still_applies_without_a_dimension_tag(tmp_path: Path) -> None:
    """Write-only workbooks carry no <dimension>; openpyxl then reports
    max_row=None and the cap used to be skipped."""
    import openpyxl

    from parsing import WorkbookTooLargeError, check_workbook_row_counts

    book = openpyxl.Workbook(write_only=True)
    sheet = book.create_sheet("Collars")
    for _ in range(30):
        sheet.append(["x"])
    path = tmp_path / "nodim.xlsx"
    book.save(path)
    assert openpyxl.load_workbook(path, read_only=True)["Collars"].max_row is None
    with pytest.raises(WorkbookTooLargeError):
        check_workbook_row_counts(path, limit=20)
    check_workbook_row_counts(path, limit=40)


def test_bad_collar_row_does_not_cascade_into_unknown_hole_errors() -> None:
    collars = [{**_TWO_COLLARS[0], "elevation": "1,110"}, _TWO_COLLARS[1]]
    lith = _TWO_LITH + [{"hole_id": "BH1", "from_depth": 10, "to_depth": 12, "lithology_code": "Sand"}]
    result, _ = ingest_workbook(_hostile_workbook(collars, lith, {"Water": [{"hole_id": "BH1", "depth": 2.0}]}))
    assert len(result.collars) == 1
    collar_errors = [e for e in result.errors if e.startswith("Collars row 2")]
    assert len(collar_errors) == 1 and "pydantic.dev" not in collar_errors[0]
    assert not any("unknown hole_id 'BH1'" in e for e in result.errors)  # the collar error explains them
