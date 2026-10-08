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
    holes_missing_lithology,
    legend_hatch_background,
    parse_coordinate_lines,
    sanitize_filename,
    svg_display_height,
    svg_is_valid,
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
    # An empty / unreadable upload does not tick Upload.
    assert (
        workflow_stage(has_upload=True, has_parse_result=False, has_profile=False, upload_failed=True)
        == 0
    )


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
            assert session[key] is None or session[key] == [] or session[key] is False or session[key] == ""
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


def test_plan_view_chart_fits_axes_to_utm_collars() -> None:
    """st.scatter_chart anchored axes at zero, collapsing UTM collars into a
    single dot; the Altair chart must bracket the data instead, label every
    collar and draw the section line through the chosen holes."""
    import pandas as pd

    from app_configure import IN_SECTION, NOT_IN_SECTION, _plan_view_chart

    frame = pd.DataFrame(
        {
            "hole_id": ["A", "B", "C"],
            "Easting": [500000.0, 500040.0, 500080.0],
            "Northing": [4500000.0, 4500010.0, 4500020.0],
            "section": [IN_SECTION, NOT_IN_SECTION, IN_SECTION],
        }
    )
    line = [(500000.0, 4500000.0), (500080.0, 4500020.0)]
    spec = _plan_view_chart(frame, "section", line).to_dict()
    line_layer, point_layer, label_layer = spec["layer"]
    x_domain = point_layer["encoding"]["x"]["scale"]["domain"]
    y_domain = point_layer["encoding"]["y"]["scale"]["domain"]
    assert 499990 < x_domain[0] < 500000 and 500080 < x_domain[1] < 500100
    assert 4499990 < y_domain[0] < 4500000 and 4500020 < y_domain[1] < 4500040
    # Equal scale: metres per pixel match on both axes.
    x_per_px = (x_domain[1] - x_domain[0]) / spec["width"]
    y_per_px = (y_domain[1] - y_domain[0]) / spec["height"]
    assert abs(x_per_px - y_per_px) < 1e-9
    assert point_layer["encoding"]["color"]["field"] == "section"
    assert label_layer["mark"]["type"] == "text"
    assert label_layer["encoding"]["text"]["field"] == "hole_id"
    assert line_layer["mark"]["type"] == "line"
    assert line_layer["encoding"]["order"]["field"] == "step"
    no_selection = _plan_view_chart(frame, None).to_dict()
    assert len(no_selection["layer"]) == 2
    assert "color" not in no_selection["layer"][0]["encoding"]


def test_equal_scale_domains_keep_tall_layouts_undistorted() -> None:
    from app_configure import equal_scale_domains

    x_domain, y_domain, height = equal_scale_domains([0.0, 10.0], [0.0, 1000.0])
    assert height == 480  # capped
    x_per_px = (x_domain[1] - x_domain[0]) / 640
    y_per_px = (y_domain[1] - y_domain[0]) / height
    assert abs(x_per_px - y_per_px) < 1e-9
    assert y_domain[0] < 0.0 and y_domain[1] > 1000.0


def test_suggested_lines_explain_themselves_and_rank_two_hole_lines_last() -> None:
    from app_configure import (
        describe_transect_candidates,
        max_offset_from_straight_m,
        recommended_batch_text,
    )
    from models import Collar
    from transect_planner import TransectCandidate

    collars = [
        Collar(hole_id=hole, easting=e, northing=n, elevation=100.0, total_depth=10.0)
        for hole, e, n in [
            ("MW-01", 0.0, 0.0),
            ("MW-02", 40.0, 6.0),
            ("MW-03", 80.0, 0.0),
            ("MW-05", 120.0, 0.0),
        ]
    ]
    assert max_offset_from_straight_m(collars, ["MW-01", "MW-02", "MW-05"]) == 6.0
    two = TransectCandidate(("MW-01", "MW-05"), 30.0, 0.0, 3, 120.0, 0)
    four = TransectCandidate(("MW-01", "MW-02", "MW-03", "MW-05"), 20.0, 0.0, 3, 121.2, 0)
    described = describe_transect_candidates([two, four], collars)
    assert [candidate for _label, candidate in described] == [four, two]
    assert described[0][0] == "MW-01 → MW-05 · 4 holes · 121 m long · max offset 6 m"
    assert "only 2 holes" in described[1][0]
    assert "score" not in described[0][0]
    assert recommended_batch_text([two, four]).splitlines() == [
        "A-A' | MW-01, MW-02, MW-03, MW-05",
        "B-B' | MW-01, MW-05",
    ]


def test_configure_gate_labels_and_reasons_use_plain_wording() -> None:
    from app_configure import ConfigureState, override_warnings_label

    assert override_warnings_label(0) == "Generate even if data checks found warnings"
    assert override_warnings_label(1).endswith("(1 warning)")
    assert override_warnings_label(3).endswith("(3 warnings)")
    base = dict(
        selected_holes=["A", "B"],
        coordinate_text="",
        transect_selection=None,
        can_generate=False,
        blocking=False,
        has_warnings=False,
        override_warnings=True,
        placeholder_blocks_interp=False,
        elevation_mode="absolute",
        fail_on_overlaps=True,
        has_overlap_warnings=True,
    )
    assert "Stop if matched layers overlap" in ConfigureState(**base).blocked_reason
    masl = ConfigureState(**{**base, "has_overlap_warnings": False}, placeholder_blocks_masl_water=True)
    assert "depth below ground" in masl.blocked_reason


def test_default_hole_sequence_takes_every_hole_of_a_small_workbook() -> None:
    """A 7-hole B-B' workbook opened on only its first 4 holes."""
    from app_configure import default_hole_sequence

    seven = [f"BH-{i}" for i in range(1, 8)]
    assert default_hole_sequence(seven) == seven
    many = [f"BH-{i}" for i in range(1, 24)]
    assert default_hole_sequence(many) == many[:4]
    assert default_hole_sequence(["A", "B"]) == ["A", "B"]


def test_coverage_gaps_do_not_gate_generate():
    from types import SimpleNamespace

    from ai_quality import QualityIssue
    from app_configure import gating_warning_count

    gap = QualityIssue(code="depth_gap", message="gap", severity="warning", hole_id="BH1")
    other = QualityIssue(code="no_lithology", message="none", severity="warning", hole_id="BH2")
    error = QualityIssue(code="depth_overlap", message="x", severity="error", hole_id="BH1")
    assert gating_warning_count(None) == 0
    assert gating_warning_count(SimpleNamespace(issues=[gap, gap])) == 0
    assert gating_warning_count(SimpleNamespace(issues=[gap, other, error])) == 1


def test_section_notes_only_prefix_real_overlaps() -> None:
    from app_common import _section_note_text

    assert _section_note_text("Clay / Silt between BH1–BH2") == "Polygon overlap: Clay / Silt between BH1–BH2"
    note = "Crossing correlation A–B: Clay, Sand logged in opposite order — drawn as pinch-outs"
    assert _section_note_text(note) == note
    assert _section_note_text("Map scale 1:1 000 doesn't fit a letter landscape page").startswith("Map scale")
