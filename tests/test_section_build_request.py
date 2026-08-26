"""Tests for SectionBuildRequest cache keys."""

from __future__ import annotations

from models import CorrelationOverride

from section_build_request import SectionBuildRequest


def _base_request(**updates) -> SectionBuildRequest:
    base = SectionBuildRequest(
        transect_points=((0.0, 0.0), (50.0, 0.0)),
        vertical_exaggeration=5.0,
        show_hatches=True,
        show_legend=True,
        section_title="Title",
        interpretation_mode="interpolated",
        allow_pinch_outs=True,
        offset_warning_m=50.0,
        uncertainty_spacing_m=80.0,
        uncertainty_offset_m=50.0,
    )
    if updates:
        return base.model_copy(update=updates)
    return base


def test_cache_key_changes_with_uncertainty_thresholds() -> None:
    base = _base_request().cache_key(("BH-01", "BH-02"))
    changed = _base_request(uncertainty_spacing_m=120.0).cache_key(("BH-01", "BH-02"))
    assert base != changed


def test_cache_key_changes_with_interpretation_mode() -> None:
    base = _base_request().cache_key(("BH-01", "BH-02"))
    changed = _base_request(interpretation_mode="borehole_only").cache_key(("BH-01", "BH-02"))
    assert base != changed


def test_cache_key_changes_with_hole_selection() -> None:
    request = _base_request()
    assert request.cache_key(("BH-01", "BH-02")) != request.cache_key(("BH-01", "BH-03"))


def test_cache_key_stable_for_same_inputs() -> None:
    request = _base_request(section_title="Section A")
    holes = ("BH-01", "BH-02")
    assert request.cache_key(holes) == request.cache_key(holes)


def test_geometry_cache_key_stable_for_cosmetic_changes() -> None:
    holes = ("BH-01", "BH-02")
    base = _base_request().geometry_cache_key(holes)
    assert base == _base_request(section_title="Other Title").geometry_cache_key(holes)
    assert base == _base_request(vertical_exaggeration=10.0).geometry_cache_key(holes)
    assert base == _base_request(show_hatches=False).geometry_cache_key(holes)
    assert base == _base_request(export_font_family="Arial").geometry_cache_key(holes)
    assert base == _base_request(export_font_size=12.0).geometry_cache_key(holes)
    assert base == _base_request(water_line_solid=True).geometry_cache_key(holes)
    assert base == _base_request(uncertainty_spacing_m=120.0).geometry_cache_key(holes)
    assert base == _base_request(uncertainty_offset_m=75.0).geometry_cache_key(holes)


def test_geometry_cache_key_changes_with_geometry_inputs() -> None:
    holes = ("BH-01", "BH-02")
    base = _base_request().geometry_cache_key(holes)
    assert base != _base_request(interpretation_mode="borehole_only").geometry_cache_key(holes)
    assert base != _base_request(
        transect_points=((0.0, 0.0), (100.0, 0.0))
    ).geometry_cache_key(holes)
    assert base != _base_request(offset_warning_m=25.0).geometry_cache_key(holes)
    assert base != _base_request(
        max_offset_for_interpolation_m=10.0
    ).geometry_cache_key(holes)
    assert base == _base_request(fail_on_overlaps=True).geometry_cache_key(holes)
    assert base == _base_request(warn_on_correlation_gaps=True).geometry_cache_key(holes)
    assert base != _base_request(fail_on_overlaps=True).cache_key(holes)
    assert base != _base_request(
        correlation_overrides=(
            CorrelationOverride(
                left_hole_id="BH-01",
                right_hole_id="BH-02",
                left_unit_order=1,
                right_unit_order=1,
            ),
        )
    ).geometry_cache_key(holes)
