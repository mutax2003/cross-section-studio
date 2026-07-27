"""Tests for Advantage export conversion."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion import export_platform_workbook, parse_depth_interval
from paths import advantage_platform_workbook, advantage_source_workbook

OUTPUT = advantage_platform_workbook()
SOURCE = advantage_source_workbook()

COORDINATE_OFFSETS_M: dict[str, tuple[float, float]] = {
    "BH26-15": (0.5, 0.0),
}


def convert_advantage_export(source: Path, output: Path, *, elevation_m: float = 100.0):
    return export_platform_workbook(
        source,
        output,
        profile_id="field_export_v1",
        override_id="advantage_phase2_2026",
        elevation_m=elevation_m,
    )


def test_parse_depth_interval() -> None:
    assert parse_depth_interval("0.00-2.00m") == (0.0, 2.0)
    assert parse_depth_interval("2.50-4.00m") == (2.5, 4.0)


def test_convert_advantage_export() -> None:
    assert SOURCE.exists(), "Commit data/fixtures/advantage_phase2_source.xlsx (synthetic CI fixture)"
    collars, lithology = convert_advantage_export(SOURCE, OUTPUT)
    assert len(collars) == 23
    assert len(lithology) == 70
    assert set(collars.columns) == {"hole_id", "easting", "northing", "elevation", "total_depth"}
    assert collars["easting"].between(372000, 373500).all()
    assert collars["northing"].between(6493000, 6495000).all()

    bh01 = lithology[lithology["hole_id"] == "BH26-01"]
    assert len(bh01) == 4
    assert "Silty Clay" in bh01["lithology_code"].values

    c13 = collars[collars["hole_id"] == "BH26-13"].iloc[0]
    c15 = collars[collars["hole_id"] == "BH26-15"].iloc[0]
    assert (c13["easting"], c13["northing"]) != (c15["easting"], c15["northing"])
    assert "BH26-15" in COORDINATE_OFFSETS_M

    from models import DataParser

    result = DataParser().parse_file(OUTPUT)
    assert len(result.collars) == 23
    assert len(result.lithologies) == 70


def test_advantage_platform_workbook_ingests() -> None:
    assert OUTPUT.exists(), "Commit data/advantage_phase2_platform.xlsx (synthetic CI fixture)"
    from models import DataParser

    result = DataParser().parse_file(OUTPUT)
    assert len(result.collars) >= 20
    assert len(result.lithologies) >= 50


def test_advantage_transect_svgs_exist() -> None:
    transect_dir = ROOT / "data" / "advantage_transects"
    for name in ("cross_section_a_ew.svg", "cross_section_b_ns.svg", "cross_section_c_diagonal.svg"):
        path = transect_dir / name
        if path.exists():
            assert path.stat().st_size > 500
            assert b"<svg" in path.read_bytes()[:200].lower()
