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
    assert CONSULTING_LITHOLOGY_COLORS["Clay"] == "#967259"
    assert USGS_LITHOLOGY_COLORS["Sand"] == "#FFE39F"
    assert get_lithology_style("Clay Loam", consulting_palette=True).color == "#C68642"
    assert get_lithology_style("clay loam").color == "#C68642"


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
    # Client Fig 6/7 contract: soil chloride in mg/kg, unit in the legend only.
    assert {reading.unit for reading in aa_readings} == {"mg/kg"}
    assert not any("mg" in reading.value_label for reading in aa_readings)
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
    upper = text.upper()
    # Wave A: compact numeric labels; unit belongs in legend when present.
    assert "CHLORIDE" in upper
    if parse_result.environmental_readings:
        unit = (parse_result.environmental_readings[0].unit or "").upper()
        if unit:
            assert unit in upper
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


# Stick centres read off the client sheets (data/pdf_extract_p2) against the
# 10 m grid lines, which sit 590.5 px apart (59.05 px/m) on both figures. The
# first stick of each section is centred on the 0 m grid line.
_SHEET_STATIONS: dict[str, tuple[float, ...]] = {
    "A_A": (0.0, 8.96, 10.96, 15.65, 25.44, 28.74, 32.09),
    "B_B": (0.0, 8.45, 17.41, 20.61, 25.84, 30.16, 33.02),
}
# Total depths read from the sheet sticks (same 59.05 px/m depth axis).
_SHEET_TOTAL_DEPTHS: dict[str, dict[str, float]] = {
    "A_A": {
        "BH23-10": 4.5,
        "2017-BH05": 4.5,
        "2017-BH18": 9.0,
        "2017-BH11 / BH24-11": 13.5,
        "2017-BH09": 3.0,
        "2017-BH10": 3.0,
        "BH23-07": 4.5,
    },
    "B_B": {
        "BH24-12 / HA25-02": 12.0,
        "BH23-04": 15.0,
        "2017-BH11 / BH24-11": 13.5,
        "BH23-03": 12.0,
        "2017-BH12": 7.5,
        "BH24-08": 6.0,
        "BH23-01": 6.0,
    },
}


@pytest.mark.parametrize("transect_id", sorted(_SHEET_STATIONS))
def test_advantage_p2_stations_match_client_sheet(transect_id: str) -> None:
    """Fixture chainage tracks the sheet stick centres to within 0.1 m."""
    _, parse_result = build_parse_result(transect_id)
    stations = _SHEET_STATIONS[transect_id]
    for collar, sheet_station in zip(parse_result.collars, stations, strict=True):
        assert collar.easting == pytest.approx(sheet_station, abs=0.1), collar.hole_id


@pytest.mark.parametrize("transect_id", sorted(_SHEET_TOTAL_DEPTHS))
def test_advantage_p2_total_depths_match_client_sheet(transect_id: str) -> None:
    """Collar TD equals the sheet stick bottom; sample depths never pad a digitised stick."""
    _, parse_result = build_parse_result(transect_id)
    for collar in parse_result.collars:
        assert collar.total_depth == pytest.approx(_SHEET_TOTAL_DEPTHS[transect_id][collar.hole_id])
        bottom = max(
            lith.to_depth for lith in parse_result.lithologies if lith.hole_id == collar.hole_id
        )
        assert bottom == pytest.approx(collar.total_depth), collar.hole_id
        samples = [
            r.sample_depth
            for r in parse_result.environmental_readings
            if r.hole_id == collar.hole_id
        ]
        assert max(samples) <= collar.total_depth, collar.hole_id


def _stack(parse_result, hole_id: str) -> tuple[tuple[float, float, str], ...]:
    rows = sorted(
        (lith for lith in parse_result.lithologies if lith.hole_id == hole_id),
        key=lambda lith: lith.from_depth,
    )
    return tuple((lith.from_depth, lith.to_depth, lith.lithology_code) for lith in rows)


def test_advantage_p2_fig7_logs_match_client_sheet() -> None:
    """Fig 7 sticks digitised from the sheet legend swatches (were Fill/Clay/... before)."""
    _, parse_result = build_parse_result("B_B")
    assert _stack(parse_result, "BH24-12 / HA25-02") == (
        (0.0, 3.0, "Sand"),
        (3.0, 7.0, "Sandy Clay"),
        (7.0, 7.5, "Sand"),
        (7.5, 12.0, "Sandy Clay"),
    )
    assert _stack(parse_result, "BH23-04") == (
        (0.0, 1.0, "Fill"),
        (1.0, 2.5, "Sandy Clay"),
        (2.5, 3.0, "Loamy Sand"),
        (3.0, 8.75, "Clay Loam"),
        (8.75, 10.5, "Sandy Clay Loam"),
        (10.5, 15.0, "Clay"),
    )
    assert _stack(parse_result, "2017-BH12") == ((0.0, 4.5, "Clay"), (4.5, 7.5, "Silty Clay"))
    assert _stack(parse_result, "BH23-01") == (
        (0.0, 0.3, "Fill"),
        (0.3, 1.5, "Clay"),
        (1.5, 2.5, "Sand"),
        (2.5, 5.5, "Clay"),
        (5.5, 6.0, "Sand"),
    )


def test_advantage_p2_fig6_logs_match_client_sheet() -> None:
    """Fig 6 sticks; 2017-BH11 / BH24-11 carries the identical log on both sheets."""
    _, fig6 = build_parse_result("A_A")
    _, fig7 = build_parse_result("B_B")
    assert _stack(fig6, "BH23-10") == ((0.0, 1.25, "Sandy Clay Loam"), (1.25, 4.5, "Clay Loam"))
    assert _stack(fig6, "2017-BH05") == ((0.0, 2.3, "Sandy Clay"), (2.3, 4.5, "Clay"))
    assert _stack(fig6, "2017-BH09") == ((0.0, 3.0, "Sandy Clay"),)
    assert _stack(fig6, "BH23-07") == ((0.0, 0.9, "Fill"), (0.9, 4.5, "Clay Loam"))
    shared = "2017-BH11 / BH24-11"
    assert (
        _stack(fig6, shared)
        == _stack(fig7, shared)
        == (
            (0.0, 2.0, "Sandy Clay"),
            (2.0, 3.0, "Sand"),
            (3.0, 10.5, "Sandy Clay"),
            (10.5, 13.5, "Silty Clay"),
        )
    )


def test_advantage_p2_digitised_sticks_are_contiguous() -> None:
    for transect_id in ADVANTAGE_P2_TRANSECTS:
        _, parse_result = build_parse_result(transect_id)
        for hole_id in {lith.hole_id for lith in parse_result.lithologies}:
            stack = _stack(parse_result, hole_id)
            assert stack[0][0] == 0.0, hole_id
            for (_, prev_to, _), (next_from, _, _) in zip(stack, stack[1:], strict=False):
                assert prev_to == next_from, hole_id


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
