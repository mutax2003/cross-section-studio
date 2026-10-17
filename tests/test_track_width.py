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
        max_span_coverage=1.0,
    )
    # Full width capped at 40% of 5 m spacing → half = 1.0
    assert half == 1.0


def test_resolve_track_half_width_skips_resort_when_sorted() -> None:
    half = resolve_track_half_width(
        6.0,
        auto_fit=True,
        x_profiles=np.array([0.0, 5.0, 12.0]),
        x_sorted=True,
        max_span_coverage=1.0,
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


# Client B-B workbook layout: 7 holes over 32 m (min spacing 5 m). The 3 m
# default used to draw 2 m columns covering ~44% of the section.
_BB_X = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 26.0, 32.0])


def _coverage(half: float, xs: np.ndarray) -> float:
    return 2.0 * half * xs.size / float(xs.max() - xs.min())


def test_short_section_columns_capped_by_span_coverage() -> None:
    for width in (3.0, 8.0):
        half = resolve_track_half_width(width, x_profiles=_BB_X)
        assert _coverage(half, _BB_X) <= 0.20 + 1e-9
        assert 2.0 * half >= 0.004 * 32.0
        assert 2.0 * half <= 0.4 * 5.0 + 1e-9


def test_explicit_width_honoured_when_within_bounds() -> None:
    # P2 client figures request 0.6 m sticks on the same B-B layout (13% cover).
    assert resolve_track_half_width(0.6, x_profiles=_BB_X) == 0.3


def test_long_section_keeps_requested_width() -> None:
    xs = np.linspace(0.0, 480.0, 9)
    assert resolve_track_half_width(3.0, x_profiles=xs) == 1.5


def test_auto_fit_never_shrinks_below_visible_minimum() -> None:
    # 60 holes over 480 m: the coverage cap alone gives 0.6 m; the floor is 1.92 m.
    xs = np.linspace(0.0, 480.0, 60)
    half = resolve_track_half_width(3.0, x_profiles=xs)
    assert 2.0 * half >= 0.004 * 480.0 - 1e-9
    assert 2.0 * half <= 0.4 * float(np.diff(xs).min()) + 1e-9
    # The floor never widens a column beyond what the user asked for.
    assert resolve_track_half_width(0.5, x_profiles=xs) == 0.25


def test_auto_fit_off_draws_exact_width() -> None:
    assert resolve_track_half_width(3.0, auto_fit=False, x_profiles=_BB_X) == 1.5


def _bb_collars_and_lithologies() -> tuple[list[Collar], list[Lithology]]:
    collars: list[Collar] = []
    lithologies: list[Lithology] = []
    for i, x in enumerate(_BB_X):
        hid = f"BH-{i:02d}"
        collars.append(
            Collar(hole_id=hid, easting=float(x), northing=0.0, elevation=635.0, total_depth=8.0)
        )
        lithologies.append(Lithology(hole_id=hid, from_depth=0.0, to_depth=3.0, lithology_code="Clay"))
        lithologies.append(Lithology(hole_id=hid, from_depth=3.0, to_depth=8.0, lithology_code="Sand"))
    return collars, lithologies


def test_bb_layout_coverage_in_every_style(monkeypatch) -> None:
    captured: list[tuple[str, float, int, float]] = []
    original = CrossSectionRenderer._hole_context

    def spy(self, projected_df):
        ctx = original(self, projected_df)
        captured.append((self.profile.layout, ctx.track_half, len(ctx.x_by_hole), ctx.x_span))
        return ctx

    monkeypatch.setattr(CrossSectionRenderer, "_hole_context", spy)
    collars, lithologies = _bb_collars_and_lithologies()
    for layout in ("consulting_section", "section_sheet", "chart"):
        result = build_cross_section(
            collars,
            lithologies,
            [(0.0, 0.0), (32.0, 0.0)],
            render_layout=layout,
            track_width_m=3.0,
            export_formats=frozenset({"svg"}),
        )
        assert_valid_svg(result.svg_bytes)
    assert {c[0] for c in captured} == {"consulting_section", "section_sheet", "chart"}
    for _layout, half, n, span in captured:
        assert n == 7
        assert 2.0 * half * n / span <= 0.20 + 1e-9
        assert 2.0 * half >= 0.004 * span
