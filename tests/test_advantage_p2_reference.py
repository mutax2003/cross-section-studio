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
    assert label == "<5"
    value, label = parse_chloride_value("<10 (ND)")
    assert value == 10.0
    assert label == "<10"
    value, label = parse_chloride_value("<5 mg/L")
    assert value == 5.0
    assert label == "<5"
    value, label = parse_chloride_value("<5 mg/kg")
    assert value == 5.0
    assert label == "<5"


def test_load_chloride_readings_for_transects() -> None:
    aa_readings = load_chloride_readings(transect_id="A_A")
    assert len(aa_readings) >= 10
    assert aa_readings[0].parameter == "Chloride"
    assert aa_readings[0].value_label
    assert aa_readings[0].unit == "mg/kg"
    # Prefer From–To intervals when the chloride workbook is present.
    if any(reading.from_depth is not None for reading in aa_readings):
        assert all(
            reading.from_depth is not None and reading.to_depth is not None
            for reading in aa_readings
        )
    hole_ids = {reading.hole_id for reading in aa_readings}
    assert "BH23-10" in hole_ids
    assert "2017-BH11 / BH24-11" in hole_ids


def test_build_parse_result_uses_digitized_or_workbook_lithology() -> None:
    _, parse_result = build_parse_result("A_A")
    codes = {interval.lithology_code for interval in parse_result.lithologies}
    assert "Fill" in codes or "Clay" in codes
    assert "Sand" in codes or "Clay Loam" in codes
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
        parameter_interpolate_segments=spec.parameter_interpolate_segments,
        interpretation_mode=spec.interpretation_mode,  # type: ignore[arg-type]
        elevation_mode=spec.elevation_mode,
    )
    assert_valid_svg(result.svg_bytes)
    text = result.svg_bytes.decode("utf-8", errors="ignore")
    assert "mg/kg" in text
    assert "DEPTH (mbgs)" in text
    assert "WHITECAP" in text
    assert "GROUNDWATER LEVEL" not in text.upper()
    assert "NM" not in text.split("NOTES")[0]
    assert len(ADVANTAGE_P2_TRANSECTS) == 2
    # Text-only chlorides: compact numeric labels, no diamond scatter markers required.
    assert "<5" in text or "1110" in text or "12.2" in text
    # Borehole-only: no interpolated fence fill between sticks.
    assert "May 2024" not in text


def test_advantage_p2_hole_order_and_chainage() -> None:
    for transect_id, expected in ADVANTAGE_P2_TRANSECTS.items():
        spec, parse_result = build_parse_result(transect_id)
        assert tuple(c.hole_id for c in parse_result.collars) == expected.hole_ids
        assert tuple(c.easting for c in parse_result.collars) == expected.profile_eastings


def test_advantage_p2_no_water_series_in_legend() -> None:
    spec, parse_result = build_parse_result("B_B")
    assert parse_result.water_levels == ()
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
        parameter_interpolate_segments=False,
        interpretation_mode="borehole_only",
        elevation_mode="relative",
    )
    text = result.svg_bytes.decode("utf-8", errors="ignore").upper()
    assert "GROUNDWATER LEVEL" not in text
    assert "CHLORIDE" in text

