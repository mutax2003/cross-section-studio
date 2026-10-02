"""Field Data OVA/EC rows honour an optional ``label_color`` column."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion import ingest_workbook  # noqa: E402
from tests.conftest import make_field_export_bytes  # noqa: E402

_LITHO_ROWS = [
    {"hole_id": "BH-01", "depth": "0.00-2.00m", "lithology": "clay", "lat": 58.57, "long": -119.19},
]


def _ingest_with_field_data(field_rows: list[dict]):
    data = make_field_export_bytes(
        _LITHO_ROWS,
        extra_sheets={"Field Data": pd.DataFrame(field_rows)},
    )
    return ingest_workbook(BytesIO(data), profile_id="field_export_v1")


def _by_interval(readings, parameter: str, from_depth: float):
    return next(
        r
        for r in readings
        if r.parameter == parameter and abs((r.from_depth or 0.0) - from_depth) < 1e-9
    )


def test_field_data_label_color_column_is_applied_and_validated() -> None:
    result, report = _ingest_with_field_data(
        [
            {"Label": "BH-01", "Depth": "0.00-0.15m", "OVA": 125.0, "EC": 0.42, "label_color": "red"},
            {"Label": "BH-01", "Depth": "0.15-0.30m", "OVA": 80.0, "EC": 0.5, "label_color": ""},
            {"Label": "BH-01", "Depth": "0.30-0.45m", "OVA": 10.0, "EC": 0.1, "label_color": "blue"},
        ]
    )
    readings = result.environmental_readings

    assert _by_interval(readings, "OVA", 0.0).label_color == "red"
    assert _by_interval(readings, "EC", 0.0).label_color == "red"
    assert _by_interval(readings, "OVA", 0.15).label_color == ""
    assert _by_interval(readings, "EC", 0.15).label_color == ""

    # The reserved colour is rejected once for the row (not once per OVA/EC
    # reading) so the Validate "rows skipped" count matches the sheet.
    assert not any(abs((r.from_depth or 0.0) - 0.30) < 1e-9 for r in readings)
    blue_errors = [e for e in result.errors if "Field Data row 4" in e and "reserved" in e]
    assert len(blue_errors) == 1, result.errors
    assert "label_color" in blue_errors[0]
    assert len(readings) == 4


def test_field_data_label_colour_spelling_and_per_parameter_columns() -> None:
    result, _ = _ingest_with_field_data(
        [
            {
                "Label": "BH-01",
                "Depth": "0.00-0.15m",
                "OVA": 125.0,
                "EC": 0.42,
                "label_colour": "green",
                "ec_label_color": "orange",
            },
        ]
    )
    readings = result.environmental_readings
    assert _by_interval(readings, "OVA", 0.0).label_color == "green"
    assert _by_interval(readings, "EC", 0.0).label_color == "orange"
    assert not result.errors


def test_field_data_without_label_color_column_is_unchanged() -> None:
    result, _ = _ingest_with_field_data(
        [{"Label": "BH-01", "Depth": "0.00-0.15m", "OVA": 125.0, "EC": 0.42}],
    )
    assert [r.label_color for r in result.environmental_readings] == ["", ""]
    assert not result.errors
