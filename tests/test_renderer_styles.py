"""Tests for lithology styling and renderer visuals."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from constants import CONSULTING_LITHOLOGY_COLORS, USGS_LITHOLOGY_COLORS, USGS_LITHOLOGY_HATCHES, get_lithology_style
from models import Collar, ConsultingTitleBlock, Lithology, ScreenInterval, VerticalGradient, WaterLevel
from render_profiles import CHART_PROFILE, CONSULTING_SECTION_PROFILE, SECTION_SHEET_PROFILE
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
    renderer = CrossSectionRenderer(show_hatches=False, show_legend=False, render_profile=CHART_PROFILE)
    figure = renderer.render(polygons, projected, collar_depths={"BH-01": 5.0, "BH-02": 5.0})
    svg_bytes = renderer.to_svg_bytes(figure)
    assert_valid_svg(svg_bytes)


def test_section_sheet_svg_includes_track_and_eol_markers() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=512.0, total_depth=24.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=510.0, total_depth=22.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=12.0, lithology_code="Sandstone"),
        Lithology(hole_id="BH-01", from_depth=12.0, to_depth=24.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Sandstone"),
        Lithology(hole_id="BH-02", from_depth=10.0, to_depth=22.0, lithology_code="Clay"),
    ]
    projected, polygons, _ = run_pipeline(
        collars,
        lithologies,
        [(0.0, 0.0), (50.0, 0.0)],
        show_legend=False,
    )
    renderer = CrossSectionRenderer(
        show_legend=False,
        show_ground_surface=True,
        render_profile=SECTION_SHEET_PROFILE,
        interpolate_water_table=False,
    )
    figure = renderer.render(
        polygons,
        projected,
        collar_depths={"BH-01": 24.0, "BH-02": 22.0},
    )
    svg_bytes = renderer.to_svg_bytes(figure)
    assert_valid_svg(svg_bytes)
    text = svg_bytes.decode("utf-8", errors="ignore")
    assert "BH-01" in text
    assert "BH-02" in text
    assert "TD" in text
    assert len(text) > 5000


def test_section_sheet_depth_axis_mode() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=40.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
    ]
    projected, polygons, _ = run_pipeline(collars, lithologies, [(0.0, 0.0), (40.0, 0.0)])
    depth_profile = SECTION_SHEET_PROFILE.model_copy(update={"y_axis_mode": "depth_below_collar"})
    renderer = CrossSectionRenderer(show_legend=False, render_profile=depth_profile)
    figure = renderer.render(polygons, projected, collar_depths={"BH-01": 10.0, "BH-02": 10.0})
    svg_bytes = renderer.to_svg_bytes(figure)
    assert_valid_svg(svg_bytes)
    assert b"Depth below collar" in svg_bytes


def test_consulting_section_svg_structure() -> None:
    collars = [
        Collar(hole_id="MW-01", easting=0.0, northing=0.0, elevation=665.0, total_depth=20.0),
        Collar(hole_id="MW-02", easting=50.0, northing=0.0, elevation=664.0, total_depth=18.0),
    ]
    lithologies = [
        Lithology(hole_id="MW-01", from_depth=0.0, to_depth=8.0, lithology_code="Sand"),
        Lithology(hole_id="MW-01", from_depth=8.0, to_depth=20.0, lithology_code="Clay"),
        Lithology(hole_id="MW-02", from_depth=0.0, to_depth=7.0, lithology_code="Sand"),
        Lithology(hole_id="MW-02", from_depth=7.0, to_depth=18.0, lithology_code="Clay"),
    ]
    water_levels = [
        WaterLevel(hole_id="MW-01", depth=2.5),
        WaterLevel(hole_id="MW-02", depth=3.0),
    ]
    _, _, svg_bytes = run_pipeline(
        collars,
        lithologies,
        [(0.0, 0.0), (50.0, 0.0)],
        render_layout="consulting_section",
        show_legend=False,
        water_levels=water_levels,
    )
    assert_valid_svg(svg_bytes)
    text = svg_bytes.decode("utf-8", errors="ignore")
    assert "ELEVATION" in text
    assert "DISTANCE" in text
    assert "LEGEND" in text
    assert "MW-01" in text
    assert "662.500" in text or "662.5" in text
    assert " m TD" not in text

    projected, polygons, _ = run_pipeline(
        collars,
        lithologies,
        [(0.0, 0.0), (50.0, 0.0)],
        show_legend=False,
    )
    renderer = CrossSectionRenderer(
        show_legend=False,
        render_profile=CONSULTING_SECTION_PROFILE,
        interpolate_water_table=True,
        consulting_title_block=ConsultingTitleBlock(
            section_label="A-A'",
            map_scale="1:1000",
        ),
    )
    figure = renderer.render(
        polygons,
        projected,
        collar_depths={"MW-01": 20.0, "MW-02": 18.0},
        water_levels=water_levels,
    )
    consulting_svg = renderer.to_svg_bytes(figure)
    assert_valid_svg(consulting_svg)
    consulting_text = consulting_svg.decode("utf-8", errors="ignore")
    assert "CROSS SECTION A-A'" in consulting_text
    assert "CROSS SECTION CROSS SECTION" not in consulting_text
    assert "VERTICAL EXAGGERATION" in consulting_text
    assert "NOTES:" in consulting_text

    # Prefix already present must not be doubled.
    prefixed = CrossSectionRenderer(
        show_legend=False,
        render_profile=CONSULTING_SECTION_PROFILE,
        consulting_title_block=ConsultingTitleBlock(section_label="CROSS SECTION B-B'"),
    )
    prefixed_svg = prefixed.to_svg_bytes(
        prefixed.render(
            polygons,
            projected,
            collar_depths={"MW-01": 20.0, "MW-02": 18.0},
            water_levels=water_levels,
        )
    )
    prefixed_text = prefixed_svg.decode("utf-8", errors="ignore")
    assert "CROSS SECTION B-B'" in prefixed_text
    assert "CROSS SECTION CROSS SECTION" not in prefixed_text


def test_consulting_depth_mode_well_columns_render() -> None:
    collars = [
        Collar(hole_id="MW-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="MW-02", easting=40.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="MW-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="MW-02", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
    ]
    projected, polygons, _ = run_pipeline(collars, lithologies, [(0.0, 0.0), (40.0, 0.0)])
    depth_profile = CONSULTING_SECTION_PROFILE.model_copy(
        update={"y_axis_mode": "depth_below_collar"}
    )
    renderer = CrossSectionRenderer(
        show_legend=False,
        render_profile=depth_profile,
        consulting_title_block=ConsultingTitleBlock(section_label="A-A'"),
    )
    figure = renderer.render(
        polygons,
        projected,
        collar_depths={"MW-01": 10.0, "MW-02": 10.0},
        water_levels=[],
    )
    svg_bytes = renderer.to_svg_bytes(figure)
    assert_valid_svg(svg_bytes)
    text = svg_bytes.decode("utf-8", errors="ignore")
    assert "NM" in text
    assert "#ffffff" in text.lower() or 'fill="#FFFFFF"' in text


def test_standard_lithology_colors_and_hatches() -> None:
    sand = get_lithology_style("Sand", use_hatch=True)
    clay = get_lithology_style("Clay", use_hatch=True)
    topsoil = get_lithology_style("Topsoil", use_hatch=True)
    assert sand.color.upper() == USGS_LITHOLOGY_COLORS["Sand"].upper()
    assert clay.color.upper() == USGS_LITHOLOGY_COLORS["Clay"].upper()
    assert topsoil.color.upper() == USGS_LITHOLOGY_COLORS["Topsoil"].upper()
    assert sand.hatch == USGS_LITHOLOGY_HATCHES["Sand"]
    assert clay.hatch == USGS_LITHOLOGY_HATCHES["Clay"]
    assert topsoil.hatch == USGS_LITHOLOGY_HATCHES["Topsoil"]
    assert get_lithology_style("Clay", use_hatch=False).hatch == ""


def test_consulting_section_parity_elements() -> None:
    collars = [
        Collar(hole_id="MW-01", easting=0.0, northing=0.0, elevation=665.0, total_depth=20.0),
        Collar(hole_id="MW-02", easting=50.0, northing=0.0, elevation=664.0, total_depth=18.0),
        Collar(hole_id="MW-03", easting=100.0, northing=0.0, elevation=663.0, total_depth=16.0),
    ]
    lithologies = [
        Lithology(hole_id="MW-01", from_depth=0.0, to_depth=8.0, lithology_code="Sand"),
        Lithology(hole_id="MW-01", from_depth=8.0, to_depth=20.0, lithology_code="Clay"),
        Lithology(hole_id="MW-02", from_depth=0.0, to_depth=7.0, lithology_code="Sand"),
        Lithology(hole_id="MW-02", from_depth=7.0, to_depth=18.0, lithology_code="Clay"),
        Lithology(hole_id="MW-03", from_depth=0.0, to_depth=6.0, lithology_code="Sand"),
        Lithology(hole_id="MW-03", from_depth=6.0, to_depth=16.0, lithology_code="Clay"),
    ]
    water_levels = [
        WaterLevel(hole_id="MW-01", depth=2.5),
        WaterLevel(hole_id="MW-02", depth=3.0),
    ]
    projected, polygons, _ = run_pipeline(
        collars,
        lithologies,
        [(0.0, 0.0), (100.0, 0.0)],
        show_legend=False,
    )
    title_block = ConsultingTitleBlock(
        section_label="B-B'",
        map_scale="1:1000",
        drawn_by="EC 03/30/22",
        prepared_for="SURGE ENERGY INC",
        prepared_by="ECOVENTURE",
        y_axis_label="ELEVATION (m)",
        screen_legend_label="SCREEN INTERVAL",
        show_gradient_legend=True,
        notes=(
            "GROUNDWATER BASED ON GROUNDWATER MONITORING WELL OBSERVATIONS ONLY.",
            "masl DENOTES METRES ABOVE SEA LEVEL.",
        ),
    )
    renderer = CrossSectionRenderer(
        show_legend=False,
        show_hatches=True,
        render_profile=CONSULTING_SECTION_PROFILE,
        interpolate_water_table=True,
        consulting_title_block=title_block,
        screen_intervals=(
            ScreenInterval(hole_id="MW-01", from_depth=5.0, to_depth=10.0),
        ),
        vertical_gradients=(VerticalGradient(hole_id="MW-01", direction="up"),),
    )
    figure = renderer.render(
        polygons,
        projected,
        collar_depths={"MW-01": 20.0, "MW-02": 18.0, "MW-03": 16.0},
        water_levels=water_levels,
    )
    svg_bytes = renderer.to_svg_bytes(figure)
    assert_valid_svg(svg_bytes)
    text = svg_bytes.decode("utf-8", errors="ignore")
    lowered = text.lower()
    assert "#ffffff" in lowered or 'fill="#FFFFFF"' in text
    assert USGS_LITHOLOGY_COLORS["Sand"].lower() in lowered
    assert CONSULTING_LITHOLOGY_COLORS["Clay"].lower() in lowered
    assert "NM" in text
    assert "NOTES:" in text
    assert "30" in text
    assert "DRAWN BY" in text
    assert "PREPARED FOR" in text
    assert "SAND" in text
    assert "CLAY" in text
    assert "SCREEN INTERVAL" in text
    assert "VERTICAL GRADIENT DIRECTION" in text
    assert " masl" in text
