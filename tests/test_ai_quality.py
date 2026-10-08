"""Tests for ai_quality.py."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest

from ai_quality import (
    MappingProposal,
    analyze_parsed_data,
    apply_column_mapping,
    load_lithology_aliases,
    normalize_lithology_code,
    propose_column_mappings,
    propose_workbook_mapping,
    read_mapped_sheets,
)
from models import Collar, DataParser, Lithology, Transect


def test_normalize_lithology_alias() -> None:
    aliases = load_lithology_aliases()
    assert normalize_lithology_code("CLY", aliases) == "Clay"
    assert normalize_lithology_code("sandstone", aliases) == "Sandstone"


def test_normalize_lithology_sand_not_remapped_to_sandstone() -> None:
    aliases = load_lithology_aliases()
    assert normalize_lithology_code("Sand", aliases) == "Sand"
    assert normalize_lithology_code("sand", aliases) == "Sand"
    assert normalize_lithology_code("SAND", aliases) == "Sand"
    assert normalize_lithology_code("Sandstone", aliases) == "Sandstone"
    assert normalize_lithology_code("snd", aliases) == "Sand"
    assert normalize_lithology_code("sst", aliases) == "Sandstone"


def test_fuzzy_column_mapping() -> None:
    mappings = propose_column_mappings(
        ["BH", "East", "North", "RL", "TD"],
        {"hole_id", "easting", "northing", "elevation", "total_depth"},
        {
            "hole_id": {"hole_id", "bh"},
            "easting": {"easting", "east"},
            "northing": {"northing", "north"},
            "elevation": {"elevation", "rl"},
            "total_depth": {"total_depth", "td"},
        },
    )
    by_canonical = {mapping.canonical_column: mapping for mapping in mappings}
    assert by_canonical["hole_id"].source_column == "BH"
    assert by_canonical["hole_id"].confidence >= 0.8


def test_apply_column_mapping() -> None:
    df = pd.DataFrame([{"BH": "BH-01", "East": 1.0, "North": 2.0, "RL": 100.0, "TD": 10.0}])
    mappings = propose_column_mappings(
        list(df.columns),
        {"hole_id", "easting", "northing", "elevation", "total_depth"},
        {
            "hole_id": {"hole_id", "bh"},
            "easting": {"easting", "east"},
            "northing": {"northing", "north"},
            "elevation": {"elevation", "rl"},
            "total_depth": {"total_depth", "td"},
        },
    )
    mapped = apply_column_mapping(df, mappings)
    assert "hole_id" in mapped.columns
    assert mapped.iloc[0]["hole_id"] == "BH-01"


def test_depth_gap_detection() -> None:
    collars = [Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0)]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Sandstone"),
        Lithology(hole_id="BH-01", from_depth=8.0, to_depth=15.0, lithology_code="Clay"),
    ]
    report = analyze_parsed_data(collars, lithologies)
    gap_issues = [issue for issue in report.issues if issue.code == "depth_gap"]
    assert len(gap_issues) >= 1


def test_overlap_detection() -> None:
    collars = [Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0)]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Sandstone"),
        Lithology(hole_id="BH-01", from_depth=8.0, to_depth=15.0, lithology_code="Clay"),
    ]
    report = analyze_parsed_data(collars, lithologies)
    assert any(issue.code == "depth_overlap" for issue in report.issues)
    assert report.has_blocking_errors


def test_below_total_depth() -> None:
    collars = [Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0)]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=12.0, lithology_code="Clay"),
    ]
    report = analyze_parsed_data(collars, lithologies)
    assert any(issue.code == "below_td" for issue in report.issues)


def test_workbook_mapping_from_bytes() -> None:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(
            [{"BH": "BH-01", "East": 1.0, "North": 2.0, "RL": 100.0, "TD": 10.0}]
        ).to_excel(writer, sheet_name="Collar", index=False)
        pd.DataFrame(
            [{"BH": "BH-01", "From": 0.0, "To": 5.0, "Lith": "CLY"}]
        ).to_excel(writer, sheet_name="Lith", index=False)
    buffer.seek(0)

    proposal = propose_workbook_mapping(pd.ExcelFile(buffer))
    assert proposal.collars_sheet == "Collar"
    assert proposal.lithology_sheet == "Lith"

    buffer.seek(0)
    collars_df, lithology_df = read_mapped_sheets(buffer, proposal)
    aliases = load_lithology_aliases()
    result = DataParser().parse_file(
        buffer,
        collars_df=collars_df,
        lithology_df=lithology_df,
        lithology_aliases=aliases,
    )
    assert len(result.collars) == 1
    assert result.lithologies[0].lithology_code == "Clay"


def test_off_transect_warning() -> None:
    collars = [Collar(hole_id="BH-01", easting=0.0, northing=100.0, elevation=50.0, total_depth=10.0)]
    lithologies = [Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Clay")]
    transect = Transect(points=[(0.0, 0.0), (100.0, 0.0)])
    report = analyze_parsed_data(collars, lithologies, transect=transect, offset_threshold_m=50.0)
    assert any(issue.code == "off_transect" for issue in report.issues)


def test_summarize_environmental_readings() -> None:
    from ai_quality import summarize_environmental_readings
    from models import EnvironmentalReading

    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    readings = [
        EnvironmentalReading(hole_id="BH-01", parameter="Chloride", value=120.0, depth=3.5, unit="mg/L"),
        EnvironmentalReading(hole_id="BH-02", parameter="Chloride", value=85.0, depth=4.0, unit="mg/L"),
        EnvironmentalReading(hole_id="BH-01", parameter="Benzene", value=0.5, depth=2.0, unit="mg/L"),
    ]
    summary = summarize_environmental_readings(collars, readings, ("BH-01", "BH-02"))
    assert summary.total_readings == 3
    assert tuple(param.parameter for param in summary.parameters) == ("Benzene", "Chloride")
    chloride = next(param for param in summary.parameters if param.parameter == "Chloride")
    assert chloride.hole_count == 2
    assert chloride.min_depth == pytest.approx(3.5)
    assert chloride.max_depth == pytest.approx(4.0)
    assert summary.holes_without_any_readings == ()


def _issues(report, code: str) -> list:
    return [issue for issue in report.issues if issue.code == code]


def test_unit_order_against_depth_warning() -> None:
    """Fuzz repro: A numbers Till 41 above Sand 5, so the fence un-matches it as a
    crossing — QA tells the user to check the numbering."""
    collars = [
        Collar(hole_id="A", easting=0, northing=0, elevation=354.0, total_depth=53),
        Collar(hole_id="B", easting=13, northing=0, elevation=280.6, total_depth=39),
    ]
    lithologies = [
        Lithology(hole_id="A", from_depth=2.37, to_depth=8.13, lithology_code="Till", unit_order=41),
        Lithology(hole_id="A", from_depth=25.0, to_depth=29.65, lithology_code="Sand", unit_order=5),
        Lithology(hole_id="B", from_depth=0.006, to_depth=6.46, lithology_code="Sand", unit_order=2),
        Lithology(
            hole_id="B", from_depth=11.28, to_depth=17.46, lithology_code="Till", unit_order=29
        ),
    ]
    issues = _issues(analyze_parsed_data(collars, lithologies), "unit_order_against_depth")
    assert len(issues) == 1
    assert issues[0].hole_id == "A"
    assert issues[0].severity == "warning"
    assert issues[0].message.startswith("unit_order runs against depth in A:")
    assert issues[0].message.endswith("check numbering")


def test_unit_order_increasing_with_depth_has_no_warning() -> None:
    collars = [Collar(hole_id="A", easting=0, northing=0, elevation=100.0, total_depth=20.0)]
    lithologies = [
        Lithology(hole_id="A", from_depth=0.0, to_depth=5.0, lithology_code="Clay", unit_order=1),
        Lithology(hole_id="A", from_depth=5.0, to_depth=8.0, lithology_code="Sand", unit_order=4),
        Lithology(hole_id="A", from_depth=8.0, to_depth=20.0, lithology_code="Till"),
    ]
    report = analyze_parsed_data(collars, lithologies)
    assert _issues(report, "unit_order_against_depth") == []


def test_zero_thickness_interval_warning() -> None:
    collars = [Collar(hole_id="B", easting=0, northing=0, elevation=0.0, total_depth=1268.9)]
    lithologies = [
        Lithology(
            hole_id="B", from_depth=4.94525, to_depth=5.48206, lithology_code="Sand", unit_order=55
        ),
        Lithology(
            hole_id="B", from_depth=5.48206, to_depth=5.48206, lithology_code="Gravel", unit_order=56
        ),
    ]
    issues = _issues(analyze_parsed_data(collars, lithologies), "zero_thickness_interval")
    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert "zero-thickness interval Gravel at 5.48 m" in issues[0].message
    assert "ignored in the fence" in issues[0].message


def test_thin_but_nonzero_interval_has_no_zero_thickness_warning() -> None:
    collars = [Collar(hole_id="B", easting=0, northing=0, elevation=0.0, total_depth=10.0)]
    lithologies = [
        Lithology(hole_id="B", from_depth=0.0, to_depth=5.0, lithology_code="Sand"),
        Lithology(hole_id="B", from_depth=5.0, to_depth=5.01, lithology_code="Gravel"),
        Lithology(hole_id="B", from_depth=5.01, to_depth=10.0, lithology_code="Clay"),
    ]
    report = analyze_parsed_data(collars, lithologies)
    assert _issues(report, "zero_thickness_interval") == []
