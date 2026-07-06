"""Tests for geology QA and scoring improvements."""

from __future__ import annotations

from models import Collar, Lithology
from ai_quality import _hole_quality_issues, analyze_parsed_data, collars_use_placeholder_elevation
from ingestion import suggest_utm_crs
from stratigraphy import _correlation_sort_key, _LayerInterval
from transect_planner import score_transect


def test_duplicate_lithology_without_unit_order_is_error() -> None:
    collar = Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0)
    intervals = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=3.0, lithology_code="Clay"),
        Lithology(hole_id="BH-01", from_depth=3.0, to_depth=6.0, lithology_code="Sand"),
        Lithology(hole_id="BH-01", from_depth=6.0, to_depth=10.0, lithology_code="Clay"),
    ]
    issues = _hole_quality_issues(collar, intervals)
    assert any(issue.code == "duplicate_lithology_no_unit_order" for issue in issues)


def test_placeholder_elevation_warning() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=10.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
    ]
    report = analyze_parsed_data(collars, lithologies, placeholder_elevation_m=100.0)
    assert any(issue.code == "placeholder_elevation" for issue in report.issues)
    assert collars_use_placeholder_elevation(collars, 100.0)


def test_suggest_utm_crs_northern_hemisphere() -> None:
    assert suggest_utm_crs([53.0], [-114.0]) == "EPSG:32612"


def test_correlation_sort_key_orders_by_elevation() -> None:
    left = _LayerInterval("Sand", 0.0, 5.0, 100.0, 95.0, None)
    right = _LayerInterval("Clay", 5.0, 10.0, 95.0, 90.0, None)
    left_lookup = {("code", "Sand"): left}
    right_lookup = {("code", "Clay"): right}
    keys = sorted(
        set(left_lookup) | set(right_lookup),
        key=lambda key: _correlation_sort_key(key, left_lookup, right_lookup),
    )
    assert keys[0] == ("code", "Sand")


def test_transect_score_penalizes_pinch_out_mismatch() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    matched = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
    ]
    mismatched = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Sand"),
    ]
    matched_score = score_transect(collars, matched, ("BH-01", "BH-02")).score
    mismatch_score = score_transect(collars, mismatched, ("BH-01", "BH-02")).score
    assert matched_score > mismatch_score
