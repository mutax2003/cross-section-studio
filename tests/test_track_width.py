"""Tests for adjustable borehole column width."""

from __future__ import annotations

import numpy as np

from models import Collar, Lithology
from pipeline import build_cross_section
from render_profiles import CONSULTING_SECTION_PROFILE, SECTION_SHEET_PROFILE
from renderer import CrossSectionRenderer, resolve_track_half_width
from section_build_request import SectionBuildRequest
from tests.conftest import assert_valid_svg


def test_resolve_track_half_width_honors_request() -> None:
    assert resolve_track_half_width(4.0, auto_fit=False) == 2.0


def test_resolve_track_half_width_auto_fits_close_holes() -> None:
    half = resolve_track_half_width(
        6.0,
        auto_fit=True,
        x_profiles=np.array([0.0, 5.0]),
    )
    # Full width capped at 40% of 5 m spacing → half = 1.0
    assert half == 1.0


def test_resolve_track_half_width_skips_resort_when_sorted() -> None:
    half = resolve_track_half_width(
        6.0,
        auto_fit=True,
        x_profiles=np.array([0.0, 5.0, 12.0]),
        x_sorted=True,
    )
    assert half == 1.0


def test_resolve_track_half_width_no_fit_when_spacing_wide() -> None:
    half = resolve_track_half_width(
        3.0,
        auto_fit=True,
        x_profiles=np.array([0.0, 50.0]),
    )
    assert half == 1.5


def test_renderer_uses_track_width_for_all_layouts() -> None:
    profile = SECTION_SHEET_PROFILE.model_copy(
        update={"track_width_m": 5.0, "auto_fit_track_width": False}
    )
    renderer = CrossSectionRenderer(render_profile=profile)
    assert renderer._track_half_width(np.array([0.0, 100.0])) == 2.5

    chart = CrossSectionRenderer(
        render_profile=profile.model_copy(update={"layout": "chart"})
    )
    assert chart._track_half_width(np.array([0.0, 100.0])) == 2.5


def test_consulting_layout_honors_track_width_override() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
    ]
    result = build_cross_section(
        collars,
        lithologies,
        [(0.0, 0.0), (50.0, 0.0)],
        render_layout="consulting_section",
        track_width_m=2.5,
        auto_fit_track_width=False,
    )
    assert_valid_svg(result.svg_bytes)


def test_geometry_cache_ignores_track_width() -> None:
    holes = ("BH-01", "BH-02")
    base = SectionBuildRequest(transect_points=((0.0, 0.0), (10.0, 0.0)))
    wide = base.model_copy(
        update={"track_width_m": 8.0, "auto_fit_track_width": False}
    )
    assert base.geometry_cache_key(holes) == wide.geometry_cache_key(holes)
    assert base.cache_key(holes) != wide.cache_key(holes)


def test_consulting_profile_default_narrower_than_section_sheet() -> None:
    assert CONSULTING_SECTION_PROFILE.track_width_m < SECTION_SHEET_PROFILE.track_width_m
