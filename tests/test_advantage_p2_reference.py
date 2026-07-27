"""Tests for Advantage Phase 2 reference data (BH log palette + chlorides)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from advantage_p2_reference.chlorides import (
    load_chloride_readings,
    normalize_compound_hole_id,
    parse_chloride_value,
)
from advantage_p2_reference.fixtures import build_parse_result
from advantage_p2_reference.transects import ADVANTAGE_P2_TRANSECTS
from constants import (
    CONSULTING_LITHOLOGY_COLORS,
    USGS_LITHOLOGY_COLORS,
    get_lithology_style,
    parse_bh_log_legend_xlsx,
)
from paths import bh_log_lithology_legend_xlsx_path
from pipeline import build_cross_section
from tests.conftest import assert_valid_svg


def test_bh_log_legend_colors_loaded() -> None:
    assert CONSULTING_LITHOLOGY_COLORS["Clay"] == "#38220F"
    assert USGS_LITHOLOGY_COLORS["Sand"] == "#EAC086"
    assert get_lithology_style("Clay Loam", consulting_palette=True).color == "#DBC1AC"
    assert get_lithology_style("clay loam").color == "#DBC1AC"


def test_bh_log_legend_excel_is_runtime_source() -> None:
    xlsx = bh_log_lithology_legend_xlsx_path()
    if not xlsx.exists():
        pytest.skip("BH Log Lithology Legend.xlsx not present (CI uses JSON palette)")
    excel_colors = parse_bh_log_legend_xlsx(xlsx)
    assert excel_colors
    for code, colour in excel_colors.items():
        assert USGS_LITHOLOGY_COLORS[code] == colour
        assert get_lithology_style(code).color == colour


def test_normalize_compound_hole_id() -> None:
    assert normalize_compound_hole_id("2017-BH11/BH24-11") == "2017-BH11 / BH24-11"
    assert normalize_compound_hole_id("BH24-12 / HA25-02") == "BH24-12 / HA25-02"


def test_parse_chloride_value_handles_nd() -> None:
    value, label = parse_chloride_value("<5.0")
    assert value == 5.0
    assert label == "<5 mg/L"
    value, label = parse_chloride_value("<10 (ND)")
    assert value == 10.0
    assert label == "<10 mg/L"
    value, label = parse_chloride_value("<5 mg/L")
    assert value == 5.0
    assert label == "<5 mg/L"


def test_load_chloride_readings_for_transects() -> None:
    aa_readings = load_chloride_readings(transect_id="A_A")
    assert len(aa_readings) >= 10
    assert aa_readings[0].parameter == "Chloride"
    assert aa_readings[0].value_label
    hole_ids = {reading.hole_id for reading in aa_readings}
    assert "BH23-10" in hole_ids
    assert "2017-BH11 / BH24-11" in hole_ids


def test_build_parse_result_uses_bh_log_lithology_names() -> None:
    _, parse_result = build_parse_result("A_A")
    codes = {interval.lithology_code for interval in parse_result.lithologies}
    assert "Organics" in codes or "Loam" in codes
    assert "Sand" in codes or "Loamy Sand" in codes
    assert "Clay" in codes or "Clay Loam" in codes
    assert all(interval.unit_order is not None for interval in parse_result.lithologies)


def test_advantage_p2_chloride_transect_renders() -> None:
    spec, parse_result = build_parse_result("A_A")
    assert spec.transect_id == "A_A"
    transect_points = [(collar.easting, collar.northing) for collar in parse_result.collars]
    result = build_cross_section(
        parse_result.collars,
        parse_result.lithologies,
        transect_points,
        vertical_exaggeration=spec.vertical_exaggeration,
        render_layout="consulting_section",
        consulting_title_block=spec.title_block,
        environmental_readings=parse_result.environmental_readings,
        environmental_parameters=("Chloride",),
        show_parameter_labels=True,
    )
    assert_valid_svg(result.svg_bytes)
    text = result.svg_bytes.decode("utf-8", errors="ignore")
    assert "mg/L" in text
    assert len(ADVANTAGE_P2_TRANSECTS) == 2
