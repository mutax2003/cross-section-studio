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
