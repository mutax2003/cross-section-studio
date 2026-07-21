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
    # Parsed but no transect yet — stay on Validate
    assert (
        workflow_stage(
            has_upload=True,
            has_parse_result=True,
            has_profile=False,
            has_transect=False,
        )
        == 1
    )
    # Blocking QA keeps Validate active even with a transect
    assert (
        workflow_stage(
            has_upload=True,
            has_parse_result=True,
            has_profile=False,
            has_blocking_errors=True,
            has_transect=True,
        )
        == 1
    )
    assert (
        workflow_stage(
            has_upload=True,
            has_parse_result=True,
            has_profile=False,
            has_transect=True,
        )
        == 2
    )
    # Leftover SVG without a live parse must not show Generate
    assert (
        workflow_stage(
            has_upload=True,
            has_parse_result=False,
            has_profile=True,
        )
        == 1
    )
    # Blocking QA with SVG still keeps Validate active
    assert (
        workflow_stage(
            has_upload=True,
            has_parse_result=True,
            has_profile=True,
            has_blocking_errors=True,
            has_transect=True,
        )
        == 1
    )
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


def test_init_session_defaults_bumps_schema_and_clears_stale() -> None:
    from app_state import (
        SESSION_AI_KEYS,
        SESSION_PARSE_KEYS,
        SESSION_SCHEMA_VERSION,
        SESSION_SECTION_KEYS,
        init_session_defaults,
    )

    class _FakeSession(dict):
        pass

    session = _FakeSession()
    session["_schema_version"] = 1
    session["parse_result"] = object()
    session["svg_bytes"] = b"<svg/>"
    session["qa_narrative"] = "stale"
    session["file_bytes"] = b"xlsx"
    init_session_defaults(session)
    assert session["_schema_version"] == SESSION_SCHEMA_VERSION
    for key in SESSION_PARSE_KEYS:
        if key in session:
            assert session[key] is None or session[key] == [] or session[key] is False
    assert session.get("parse_result") is None
    assert session.get("svg_bytes") is None
    assert session.get("qa_narrative") is None
    for key in SESSION_SECTION_KEYS:
        if key in session:
            assert session[key] is None or session[key] == []
    for key in SESSION_AI_KEYS:
        if key in session:
            assert session[key] is None


def test_init_session_defaults_skips_clear_when_schema_current() -> None:
    from app_state import SESSION_SCHEMA_VERSION, init_session_defaults

    class _FakeSession(dict):
        pass

    session = _FakeSession()
    session["_schema_version"] = SESSION_SCHEMA_VERSION
    session["parse_result"] = "keep-me"
    session["svg_bytes"] = b"<svg/>"
    init_session_defaults(session)
    assert session["parse_result"] == "keep-me"
    assert session["svg_bytes"] == b"<svg/>"
