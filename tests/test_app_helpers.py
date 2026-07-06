"""Unit tests for UI helper logic (no Streamlit runtime)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import Collar, Lithology
from tests.conftest import run_pipeline
from ui_helpers import (
    active_transect_selection,
    dedupe_messages,
    escape_html,
    legend_hatch_background,
    parse_coordinate_lines,
    sanitize_filename,
    svg_display_height,
    svg_is_valid,
    holes_missing_lithology,
    workflow_stage,
)


def test_svg_validation_accepts_matplotlib_output() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=5.0),
        Collar(hole_id="BH-02", easting=40.0, northing=0.0, elevation=100.0, total_depth=5.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=5.0, lithology_code="Clay"),
    ]
    _, _, svg_bytes = run_pipeline(collars, lithologies, [(0.0, 0.0), (40.0, 0.0)])
    assert svg_is_valid(svg_bytes)
    assert b"<svg" in svg_bytes.lower()


def test_svg_display_height_uses_viewbox() -> None:
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 480"></svg>'
    height = svg_display_height(svg, min_height=200, max_height=900)
    assert 200 <= height <= 900


def test_workflow_stage_progression() -> None:
    assert workflow_stage(has_upload=False, has_parse_result=False, has_profile=False) == 0
    assert workflow_stage(has_upload=True, has_parse_result=False, has_profile=False) == 1
    assert workflow_stage(has_upload=True, has_parse_result=True, has_profile=False) == 2
    assert workflow_stage(has_upload=True, has_parse_result=True, has_profile=True) == 3


def test_legend_hatch_background_returns_css() -> None:
    assert legend_hatch_background("---") != "none"
    assert legend_hatch_background("") == "none"


def test_escape_html_prevents_injection() -> None:
    assert escape_html('<script>alert("x")</script>') == "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;"


def test_parse_coordinate_lines_ignores_comments_and_blank_lines() -> None:
    text = "# header\n0 0\n\n50 0\n"
    assert parse_coordinate_lines(text) == [(0.0, 0.0), (50.0, 0.0)]


def test_parse_coordinate_lines_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        parse_coordinate_lines("0 0\nnan 0")


def test_dedupe_messages_preserves_order() -> None:
    assert dedupe_messages(["a", "b", "a", "c"]) == ("a", "b", "c")


def test_active_transect_coordinate_mode_filters_off_transect_holes() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-03", easting=25.0, northing=80.0, elevation=100.0, total_depth=10.0),
    ]
    selection = active_transect_selection(
        collars,
        "By coordinates",
        [],
        "0 0\n50 0",
        offset_warning_m=50.0,
    )
    assert selection is not None
    hole_ids, points = selection
    assert hole_ids == ("BH-01", "BH-02")
    assert points == ((0.0, 0.0), (50.0, 0.0))


def test_active_transect_coordinate_mode_requires_two_near_holes() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=0.0, northing=200.0, elevation=100.0, total_depth=10.0),
    ]
    assert active_transect_selection(
        collars,
        "By coordinates",
        [],
        "0 0\n50 0",
        offset_warning_m=50.0,
    ) is None


def test_active_transect_coordinate_mode_orders_holes_along_transect() -> None:
    collars = [
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-03", easting=25.0, northing=80.0, elevation=100.0, total_depth=10.0),
    ]
    selection = active_transect_selection(
        collars,
        "By coordinates",
        [],
        "0 0\n50 0",
        offset_warning_m=50.0,
    )
    assert selection is not None
    hole_ids, _ = selection
    assert hole_ids == ("BH-01", "BH-02")


def test_holes_missing_lithology_detects_gaps() -> None:
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Clay"),
    ]
    assert holes_missing_lithology(lithologies, ("BH-01", "BH-02")) == ("BH-02",)


def test_sanitize_filename_strips_unsafe_chars() -> None:
    assert sanitize_filename("Section A / Line 1") == "Section_A_Line_1"
    assert sanitize_filename("   ") == "cross_section"
