"""Tests for lithology styling and renderer visuals."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from constants import USGS_LITHOLOGY_COLORS, USGS_LITHOLOGY_HATCHES, get_lithology_style
from models import Collar, Lithology
from renderer import CrossSectionRenderer
from stratigraphy import build_stratigraphy
from tests.conftest import assert_valid_svg, run_pipeline


def test_every_canonical_lithology_has_color_and_hatch() -> None:
    for code in USGS_LITHOLOGY_COLORS:
        assert code in USGS_LITHOLOGY_HATCHES
        style = get_lithology_style(code)
        assert style.color.startswith("#")
        assert len(style.hatch) >= 2


def test_renderer_applies_hatches_to_svg() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Sandstone"),
        Lithology(hole_id="BH-01", from_depth=5.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=4.0, lithology_code="Silt"),
        Lithology(hole_id="BH-02", from_depth=4.0, to_depth=10.0, lithology_code="Organics"),
    ]
    _, _, svg_bytes = run_pipeline(collars, lithologies, [(0.0, 0.0), (50.0, 0.0)])
    assert_valid_svg(svg_bytes)
    lowered = svg_bytes.lower()
    assert b"sandstone" in lowered or b"clay" in lowered


def test_renderer_hatches_can_be_disabled() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=5.0),
        Collar(hole_id="BH-02", easting=40.0, northing=0.0, elevation=100.0, total_depth=5.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Gravel"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=5.0, lithology_code="Bedrock"),
    ]
    projected, polygons, _ = run_pipeline(collars, lithologies, [(0.0, 0.0), (40.0, 0.0)])
    renderer = CrossSectionRenderer(show_hatches=False, show_legend=False)
    figure = renderer.render(polygons, projected, collar_depths={"BH-01": 5.0, "BH-02": 5.0})
    svg_bytes = renderer.to_svg_bytes(figure)
    assert_valid_svg(svg_bytes)
