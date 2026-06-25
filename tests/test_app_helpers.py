"""Unit tests for UI helper logic (no Streamlit runtime)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import Collar, Lithology
from tests.conftest import run_pipeline
from ui_helpers import (
    legend_hatch_background,
    svg_display_height,
    svg_is_valid,
    transect_cache_key,
    workflow_stage,
)


def test_transect_cache_key_changes_with_uncertainty_thresholds() -> None:
    base = transect_cache_key(
        ("BH-01", "BH-02"),
        ((0.0, 0.0), (50.0, 0.0)),
        5.0,
        50.0,
        True,
        True,
        "Title",
        "interpolated",
        True,
        80.0,
        50.0,
    )
    changed = transect_cache_key(
        ("BH-01", "BH-02"),
        ((0.0, 0.0), (50.0, 0.0)),
        5.0,
        50.0,
        True,
        True,
        "Title",
        "interpolated",
        True,
        120.0,
        50.0,
    )
    assert base != changed


def test_transect_cache_key_changes_with_interpretation_mode() -> None:
    base = transect_cache_key(
        ("BH-01", "BH-02"),
        ((0.0, 0.0), (50.0, 0.0)),
        5.0,
        50.0,
        True,
        True,
        "Title",
        "interpolated",
        True,
    )
    changed = transect_cache_key(
        ("BH-01", "BH-02"),
        ((0.0, 0.0), (50.0, 0.0)),
        5.0,
        50.0,
        True,
        True,
        "Title",
        "borehole_only",
        True,
    )
    assert base != changed


def test_transect_cache_key_changes_with_hole_selection() -> None:
    base = transect_cache_key(
        ("BH-01", "BH-02"),
        ((0.0, 0.0), (50.0, 0.0)),
        5.0,
        50.0,
        True,
        True,
        "Title",
    )
    changed = transect_cache_key(
        ("BH-01", "BH-03"),
        ((0.0, 0.0), (50.0, 0.0)),
        5.0,
        50.0,
        True,
        True,
        "Title",
    )
    assert base != changed


def test_transect_cache_key_stable_for_same_inputs() -> None:
    args = (
        ("BH-01", "BH-02"),
        ((0.0, 0.0), (50.0, 0.0)),
        5.0,
        50.0,
        True,
        True,
        "Section A",
    )
    assert transect_cache_key(*args) == transect_cache_key(*args)


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
