"""Tests for lithology styling and renderer visuals."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from constants import (
    CONSULTING_LITHOLOGY_COLORS,
    USGS_LITHOLOGY_COLORS,
    USGS_LITHOLOGY_HATCHES,
    get_lithology_style,
)
from models import (
    Collar,
    ConsultingTitleBlock,
    EnvironmentalReading,
    Lithology,
    ScreenInterval,
    VerticalGradient,
    WaterLevel,
)
from pipeline import build_cross_section
from render_profiles import CHART_PROFILE, CONSULTING_SECTION_PROFILE, SECTION_SHEET_PROFILE
from render_theme import SCREEN_INTERVAL_HATCH, parameter_series_color
from renderer import CrossSectionRenderer, _resolve_parameter_label_offsets
from tests.conftest import assert_valid_svg, run_pipeline


def test_screen_interval_hatch_is_horizontal() -> None:
    """Screen bands use horizontal hatch so they do not read as lithology diagonals."""
    assert "-" in SCREEN_INTERVAL_HATCH
    assert "/" not in SCREEN_INTERVAL_HATCH
    assert "\\" not in SCREEN_INTERVAL_HATCH


def test_parameter_label_offsets_stagger_dense_stacks() -> None:
    fig, ax = matplotlib.pyplot.subplots(figsize=(12, 5))
    ax.set_position([0.1, 0.2, 0.8, 0.65])
    ax.set_xlim(0, 270)
    ax.set_ylim(618, 637)
    labels = [(90.0, 634.0 - index * 0.2, f"{index}") for index in range(6)]
    offsets = _resolve_parameter_label_offsets(ax, labels)
    assert len(offsets) == 6
    dys = [dy for _dx, dy, _leader in offsets]
    # Each successive label in a dense stack is nudged further down.
    assert dys[0] == 0.0
    assert dys[1] < -5.0
    assert dys[-1] < dys[1] - 30.0
    assert offsets[-1][2] is True
    matplotlib.pyplot.close(fig)


def test_every_canonical_lithology_has_color_and_hatch() -> None:
    for code in USGS_LITHOLOGY_COLORS:
        assert code in USGS_LITHOLOGY_HATCHES
        style = get_lithology_style(code)
        assert style.color.startswith("#")
        # Plain units (Clay, Silt, Topsoil, Coal...) carry no hatch by design;
        # hatches are sparse single marks, as in the CAD template.
        assert style.hatch == "" or (1 <= len(style.hatch) <= 3)


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
        # Water at MW-01 only: MW-02 is then a genuinely unmeasured hole -> NM.
        water_levels=[WaterLevel(hole_id="MW-01", depth=3.0)],
    )
    svg_bytes = renderer.to_svg_bytes(figure)
    assert_valid_svg(svg_bytes)
    text = svg_bytes.decode("utf-8", errors="ignore")
    assert "NM" in text
    assert "#d0d5dd" in text.lower() or "#ffffff" in text.lower()

    # Documented contract: no water data at all -> no NM labels anywhere.
    dry_renderer = CrossSectionRenderer(
        show_legend=False,
        render_profile=depth_profile,
        consulting_title_block=ConsultingTitleBlock(section_label="A-A'"),
    )
    dry_svg = dry_renderer.to_svg_bytes(
        dry_renderer.render(
            polygons,
            projected,
            collar_depths={"MW-01": 10.0, "MW-02": 10.0},
            water_levels=[],
        )
    ).decode("utf-8", errors="ignore")
    assert "NM" not in dry_svg


def test_consulting_relative_mode_water_labels_use_mbgs() -> None:
    collars = [
        Collar(hole_id="MW-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0),
        Collar(hole_id="MW-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=18.0),
    ]
    lithologies = [
        Lithology(hole_id="MW-01", from_depth=0.0, to_depth=20.0, lithology_code="Clay"),
        Lithology(hole_id="MW-02", from_depth=0.0, to_depth=18.0, lithology_code="Clay"),
    ]
    projected, polygons, _ = run_pipeline(collars, lithologies, [(0.0, 0.0), (50.0, 0.0)])
    depth_profile = CONSULTING_SECTION_PROFILE.model_copy(
        update={"y_axis_mode": "depth_below_collar"}
    )
    renderer = CrossSectionRenderer(
        show_legend=False,
        render_profile=depth_profile,
        interpolate_water_table=True,
        consulting_title_block=ConsultingTitleBlock(section_label="A-A'"),
    )
    figure = renderer.render(
        polygons,
        projected,
        collar_depths={"MW-01": 20.0, "MW-02": 18.0},
        water_levels=[
            WaterLevel(hole_id="MW-01", depth=2.5),
            WaterLevel(hole_id="MW-02", depth=3.0),
        ],
    )
    text = renderer.to_svg_bytes(figure).decode("utf-8", errors="ignore")
    assert "2.5 mbgs" in text  # 2 decimals max, trailing zeros stripped
    assert "3 mbgs" in text
    assert "97.500 masl" not in text
    assert "mbgs" in text


def test_consulting_omits_gw_legend_when_disabled() -> None:
    collars = [
        Collar(hole_id="MW-01", easting=0.0, northing=0.0, elevation=665.0, total_depth=20.0),
        Collar(hole_id="MW-02", easting=50.0, northing=0.0, elevation=664.0, total_depth=18.0),
    ]
    lithologies = [
        Lithology(hole_id="MW-01", from_depth=0.0, to_depth=20.0, lithology_code="Clay"),
        Lithology(hole_id="MW-02", from_depth=0.0, to_depth=18.0, lithology_code="Clay"),
    ]
    projected, polygons, _ = run_pipeline(collars, lithologies, [(0.0, 0.0), (50.0, 0.0)])
    profile = CONSULTING_SECTION_PROFILE.model_copy(update={"show_water_legend": False})
    renderer = CrossSectionRenderer(
        show_legend=False,
        render_profile=profile,
        interpolate_water_table=True,
        consulting_title_block=ConsultingTitleBlock(section_label="A-A'"),
    )
    figure = renderer.render(
        polygons,
        projected,
        collar_depths={"MW-01": 20.0, "MW-02": 18.0},
        water_levels=[
            WaterLevel(hole_id="MW-01", depth=2.5),
            WaterLevel(hole_id="MW-02", depth=3.0),
        ],
    )
    text = renderer.to_svg_bytes(figure).decode("utf-8", errors="ignore")
    assert "662.500" in text or "662.5" in text
    assert "GROUNDWATER ELEVATION masl" not in text
    assert "GROUNDWATER LEVEL (masl)" not in text


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
    assert CONSULTING_LITHOLOGY_COLORS["Sand"].lower() in lowered
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


def test_consulting_legend_stays_in_panel_with_many_entries() -> None:
    """Legend labels must clip inside the left panel, not bleed into the title block."""
    codes = [f"Lith-{index}" for index in range(10)]
    collars = [
        Collar(hole_id=f"MW-{index:02d}", easting=float(index * 40), northing=0.0, elevation=100.0, total_depth=20.0)
        for index in range(4)
    ]
    lithologies = [
        Lithology(
            hole_id=collar.hole_id,
            from_depth=0.0,
            to_depth=20.0,
            lithology_code=codes[index],
        )
        for index, collar in enumerate(collars)
    ]
    water = [
        WaterLevel(hole_id="MW-00", depth=2.0, series_id="2024-05", series_label="May 2024"),
        WaterLevel(hole_id="MW-01", depth=2.5, series_id="2024-06", series_label="June 2024"),
        WaterLevel(hole_id="MW-02", depth=3.0, series_id="2025-06", series_label="June 2025"),
    ]
    readings = [
        EnvironmentalReading(
            hole_id="MW-00",
            parameter="Chloride",
            value=120.0,
            depth=4.0,
            unit="mg/L",
        ),
    ]
    result = build_cross_section(
        collars,
        lithologies,
        [(0.0, 0.0), (120.0, 0.0)],
        water_levels=water,
        environmental_readings=readings,
        environmental_parameters=("Chloride",),
        render_layout="consulting_section",
        consulting_title_block=ConsultingTitleBlock(
            section_label="A-A'",
            project_number="TEST-001",
            prepared_for="CLIENT",
            prepared_by="ECOVENTURE",
        ),
        show_parameter_labels=True,
        legend_ncol=2,
    )
    assert_valid_svg(result.svg_bytes)
    text = result.svg_bytes.decode("utf-8", errors="ignore")
    assert "clip-path" in text.lower()
    assert "PROJECT" in text
    assert "LEGEND" in text


def test_correlation_lines_renders_pinch_out_wedges() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Sandstone"),
        Lithology(hole_id="BH-01", from_depth=5.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Sandstone"),
    ]
    _, _, svg_bytes = run_pipeline(
        collars,
        lithologies,
        [(0.0, 0.0), (50.0, 0.0)],
        interpretation_mode="correlation_lines",
        allow_pinch_outs=True,
    )
    assert_valid_svg(svg_bytes)
    assert b"path" in svg_bytes.lower()


def test_water_segment_mode_skips_unmeasured_gap() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-03", easting=100.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id=hole, from_depth=0.0, to_depth=10.0, lithology_code="Clay")
        for hole in ("BH-01", "BH-02", "BH-03")
    ]
    profile = SECTION_SHEET_PROFILE.model_copy(
        update={
            "water_interpolate_segments": True,
            "water_interpolate_across_gaps": False,
            "interpolate_water_table_default": True,
        }
    )
    projected, polygons, _ = run_pipeline(collars, lithologies, [(0.0, 0.0), (100.0, 0.0)])
    renderer = CrossSectionRenderer(
        show_legend=False,
        interpolate_water_table=True,
        render_profile=profile,
    )
    figure = renderer.render(
        polygons,
        projected,
        collar_depths={hole: 10.0 for hole in ("BH-01", "BH-02", "BH-03")},
        water_levels=[
            WaterLevel(hole_id="BH-01", depth=3.0),
            WaterLevel(hole_id="BH-03", depth=4.0),
        ],
    )
    svg_bytes = renderer.to_svg_bytes(figure)
    assert_valid_svg(svg_bytes)


def test_parameter_segment_mode_skips_unmeasured_gap() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-03", easting=100.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id=hole, from_depth=0.0, to_depth=10.0, lithology_code="Clay")
        for hole in ("BH-01", "BH-02", "BH-03")
    ]
    readings = [
        EnvironmentalReading(hole_id="BH-01", parameter="Chloride", value=120.0, depth=3.0, unit="mg/L"),
        EnvironmentalReading(hole_id="BH-03", parameter="Chloride", value=85.0, depth=4.0, unit="mg/L"),
    ]
    segment_profile = SECTION_SHEET_PROFILE.model_copy(
        update={
            "show_parameter_markers": True,
            "parameter_interpolate_segments": True,
            "parameter_interpolate_across_gaps": False,
        }
    )
    across_profile = segment_profile.model_copy(update={"parameter_interpolate_across_gaps": True})
    projected, polygons, _ = run_pipeline(collars, lithologies, [(0.0, 0.0), (100.0, 0.0)])
    segment_renderer = CrossSectionRenderer(
        show_legend=False,
        environmental_readings=readings,
        environmental_parameters=("Chloride",),
        render_profile=segment_profile,
    )
    segment_figure = segment_renderer.render(
        polygons,
        projected,
        collar_depths={hole: 10.0 for hole in ("BH-01", "BH-02", "BH-03")},
    )
    segment_svg = segment_renderer.to_svg_bytes(segment_figure)
    across_renderer = CrossSectionRenderer(
        show_legend=False,
        environmental_readings=readings,
        environmental_parameters=("Chloride",),
        render_profile=across_profile,
    )
    across_figure = across_renderer.render(
        polygons,
        projected,
        collar_depths={hole: 10.0 for hole in ("BH-01", "BH-02", "BH-03")},
    )
    across_svg = across_renderer.to_svg_bytes(across_figure)
    assert_valid_svg(segment_svg)
    assert_valid_svg(across_svg)
    # Series colour is fixed per parameter name, the same on every section.
    assert parameter_series_color("Chloride").lower() in segment_svg.decode("utf-8", errors="ignore").lower()
    assert len(across_svg) > len(segment_svg)


def test_parameter_interval_draws_vertical_bar() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
    ]
    readings = [
        EnvironmentalReading(
            hole_id="BH-01",
            parameter="Chloride",
            value=120.0,
            from_depth=2.0,
            to_depth=4.0,
            unit="mg/L",
        ),
        EnvironmentalReading(
            hole_id="BH-02",
            parameter="Chloride",
            value=85.0,
            from_depth=2.5,
            to_depth=3.5,
            unit="mg/L",
        ),
    ]
    profile = SECTION_SHEET_PROFILE.model_copy(
        update={
            "show_parameter_markers": True,
            "parameter_interpolate_segments": True,
            "parameter_interpolate_across_gaps": False,
        }
    )
    projected, polygons, _ = run_pipeline(collars, lithologies, [(0.0, 0.0), (50.0, 0.0)])
    renderer = CrossSectionRenderer(
        show_legend=False,
        environmental_readings=readings,
        environmental_parameters=("Chloride",),
        render_profile=profile,
    )
    figure = renderer.render(
        polygons,
        projected,
        collar_depths={"BH-01": 10.0, "BH-02": 10.0},
    )
    svg_bytes = renderer.to_svg_bytes(figure)
    assert_valid_svg(svg_bytes)
    assert "120" in svg_bytes.decode("utf-8", errors="ignore")
    assert "120 mg/L" not in svg_bytes.decode("utf-8", errors="ignore")


def test_wave_a_section_sheet_drafting_defaults() -> None:
    """GIS drafting defaults: ID-only headers, no scale/VE/Parameters, compact units."""
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
    ]
    readings = [
        EnvironmentalReading(hole_id="BH-01", parameter="Chloride", value=120.0, depth=3.0, unit="mg/L"),
        EnvironmentalReading(hole_id="BH-02", parameter="Chloride", value=85.0, depth=4.0, unit="mg/L"),
    ]
    projected, polygons, _ = run_pipeline(collars, lithologies, [(0.0, 0.0), (50.0, 0.0)])
    renderer = CrossSectionRenderer(
        show_legend=False,
        environmental_readings=readings,
        environmental_parameters=("Chloride",),
        render_profile=SECTION_SHEET_PROFILE,
    )
    figure = renderer.render(
        polygons,
        projected,
        collar_depths={"BH-01": 10.0, "BH-02": 10.0},
    )
    svg = renderer.to_svg_bytes(figure).decode("utf-8", errors="ignore")
    assert_valid_svg(svg.encode("utf-8"))
    assert "BH-01" in svg
    assert "RL 100" not in svg
    assert "TD 10" not in svg
    assert "Parameters:" not in svg
    assert "V.E." not in svg
    assert "120 mg/L" not in svg
    assert "120" in svg


def test_lithology_style_override_is_case_insensitive(tmp_path, monkeypatch) -> None:
    import constants as constants_mod
    from constants import get_lithology_style, save_lithology_style_override

    style_path = tmp_path / "lithology_styles.json"
    monkeypatch.setattr(constants_mod, "lithology_styles_path", lambda: style_path)
    constants_mod._load_lithology_style_overrides.cache_clear()
    get_lithology_style.cache_clear()
    save_lithology_style_override("Clay", "#112233", "..")
    constants_mod._load_lithology_style_overrides.cache_clear()
    get_lithology_style.cache_clear()
    style = get_lithology_style("clay")
    assert style.color == "#112233"
    assert style.hatch == ".."
    constants_mod._load_lithology_style_overrides.cache_clear()
    get_lithology_style.cache_clear()


def test_consulting_water_labels_do_not_overlap_and_series_get_own_colours() -> None:
    """Two series with near-identical heads at every hole: the label pass must
    separate every number and keep it inside the plot; each series keeps its
    own colour instead of one flat blue."""
    import itertools

    from matplotlib.text import Text

    from renderer_water import _figure_renderer

    collars = [
        Collar(hole_id=f"MW-0{i}", easting=40.0 * i, northing=0.0, elevation=100.0, total_depth=10.0)
        for i in range(1, 5)
    ]
    lithologies = [
        Lithology(hole_id=collar.hole_id, from_depth=0.0, to_depth=10.0, lithology_code="Clay")
        for collar in collars
    ]
    water = [
        WaterLevel(hole_id=collar.hole_id, depth=2.0 + 0.03 * i, series_id=series)
        for i, collar in enumerate(collars)
        for series in ("2024-05", "2025-06")
    ]
    projected, polygons, _ = run_pipeline(collars, lithologies, [(40.0, 0.0), (160.0, 0.0)])
    renderer = CrossSectionRenderer(
        show_legend=False,
        render_profile=CONSULTING_SECTION_PROFILE,
        consulting_title_block=ConsultingTitleBlock(section_label="A-A'"),
    )
    figure = renderer.render(
        polygons,
        projected,
        collar_depths={collar.hole_id: 10.0 for collar in collars},
        water_levels=water,
    )
    mpl_renderer = _figure_renderer(figure)
    labels = [annotation for _kind, annotation, _color in renderer._water_labels]
    assert len(labels) >= 8
    labels = [annotation for annotation in labels if annotation.get_visible()]
    boxes = []
    for annotation in labels:
        annotation.update_positions(mpl_renderer)
        boxes.append(Text.get_window_extent(annotation, mpl_renderer))
    frame = labels[0].axes.get_window_extent(mpl_renderer)
    for a, b in itertools.combinations(boxes, 2):
        width = min(a.x1, b.x1) - max(a.x0, b.x0)
        height = min(a.y1, b.y1) - max(a.y0, b.y0)
        assert not (width > 0 and height > 0), "water labels overlap"
    for box in boxes:
        assert frame.x0 - 1 <= box.x0 and box.x1 <= frame.x1 + 1, "water label clipped"
    series_colours = {entry["series_id"]: entry["color"] for entry in renderer.water_series_legend}
    assert len(set(series_colours.values())) == 2


def test_close_hole_id_headers_do_not_overlap() -> None:
    """Holes 2 m apart with long IDs (like BH18-03 / BH18-02 on GWM B-B'):
    the header pass must keep every column header readable."""
    import itertools

    from matplotlib.text import Text

    from renderer_water import _figure_renderer

    # Two close holes inside a long section, so they sit near each other on paper.
    eastings = {"2017-BH09-LONG": 0.0, "2017-BH10-LONG": 2.0, "2017-BH11-LONG": 200.0}
    ids = list(eastings)
    collars = [
        Collar(hole_id=hole, easting=x, northing=0.0, elevation=100.0, total_depth=8.0)
        for hole, x in eastings.items()
    ]
    lithologies = [
        Lithology(hole_id=hole, from_depth=0.0, to_depth=8.0, lithology_code="Clay") for hole in ids
    ]
    projected, polygons, _ = run_pipeline(collars, lithologies, [(0.0, 0.0), (200.0, 0.0)])
    for profile in (CONSULTING_SECTION_PROFILE, SECTION_SHEET_PROFILE):
        renderer = CrossSectionRenderer(show_legend=False, render_profile=profile)
        figure = renderer.render(polygons, projected, collar_depths={hole: 8.0 for hole in ids})
        mpl_renderer = _figure_renderer(figure)
        boxes = [Text.get_window_extent(text, mpl_renderer) for text in renderer._header_labels]
        assert len(boxes) == 3
        for a, b in itertools.combinations(boxes, 2):
            width = min(a.x1, b.x1) - max(a.x0, b.x0)
            height = min(a.y1, b.y1) - max(a.y0, b.y0)
            assert not (width > 0 and height > 0), f"{profile.layout}: hole-ID headers overlap"


def _dense_header_section(n: int, spacing_m: float):
    ids = [f"2017-BH{i:02d}-LONG" for i in range(n)]
    collars = [
        Collar(hole_id=hole, easting=i * spacing_m, northing=0.0, elevation=100.0, total_depth=8.0)
        for i, hole in enumerate(ids)
    ]
    lithologies = [
        Lithology(hole_id=hole, from_depth=0.0, to_depth=8.0, lithology_code="Clay") for hole in ids
    ]
    total = max(200.0, (n - 1) * spacing_m + 1.0)
    projected, polygons, _ = run_pipeline(collars, lithologies, [(0.0, 0.0), (total, 0.0)])
    return ids, projected, polygons


def _header_hits(figure, renderer) -> tuple[int, int]:
    """(header/header overlaps, header/tick-label overlaps) in display space."""
    import itertools

    from matplotlib.text import Text

    from renderer_water import _figure_renderer, _overlap_area

    mpl_renderer = _figure_renderer(figure)
    figure.draw_without_rendering()
    boxes = [Text.get_window_extent(text, mpl_renderer) for text in renderer._header_labels]
    ticks = [
        label.get_window_extent(mpl_renderer)
        for ax in figure.axes
        for label in ax.get_xticklabels() + ax.get_yticklabels()
        if label.get_visible() and label.get_text()
    ]
    pairs = sum(1 for a, b in itertools.combinations(boxes, 2) if _overlap_area(a, b) > 0)
    tick_hits = sum(1 for box in boxes for tick in ticks if _overlap_area(box, tick) > 0)
    return pairs, tick_hits


def test_dense_section_sheet_headers_clear_each_other_and_tick_labels() -> None:
    """30 long IDs 6 m apart: no horizontal stagger fits, so headers go
    vertical rather than fusing or sitting on the distance tick labels."""
    ids, projected, polygons = _dense_header_section(30, 6.0)
    renderer = CrossSectionRenderer(show_legend=False, render_profile=SECTION_SHEET_PROFILE)
    figure = renderer.render(polygons, projected, collar_depths={hole: 8.0 for hole in ids})
    assert _header_hits(figure, renderer) == (0, 0)


def test_header_and_water_passes_are_idempotent_and_rerun_for_export_pages() -> None:
    """Export resizes the page after render; the label passes must re-run from
    the drawn positions (not stack offsets) and leave the page clean."""
    import numpy as np

    from export_framing import ExportFramingConfig
    from renderer_water import resolve_header_collisions

    ids, projected, polygons = _dense_header_section(12, 8.0)
    renderer = CrossSectionRenderer(show_legend=False, render_profile=SECTION_SHEET_PROFILE)
    figure = renderer.render(polygons, projected, collar_depths={hole: 8.0 for hole in ids})

    def positions():
        from renderer_water import _figure_renderer

        mpl_renderer = _figure_renderer(figure)
        return np.array([t.get_window_extent(mpl_renderer).bounds for t in renderer._header_labels])

    first = positions()
    for _ in range(3):
        resolve_header_collisions(figure, renderer._header_labels)
    np.testing.assert_allclose(positions(), first, atol=0.5)

    renderer.export_framing = ExportFramingConfig(page_preset="letter_portrait", export_dpi=72)
    renderer.to_png_bytes(figure)
    np.testing.assert_allclose(figure.get_size_inches(), (8.5, 11.0))
    assert _header_hits(figure, renderer)[0] == 0


def test_depth_mode_title_sits_clear_of_hole_headers() -> None:
    """Depth (mbgs) sections draw hole headers above the axes, where the
    title also sits; they used to share one line ("BH-02 Title BH-03")."""
    from matplotlib.text import Text

    from renderer_water import _figure_renderer, _overlap_area

    ids = [f"BH26-{index:02d}" for index in range(10)]
    collars = [
        Collar(hole_id=hole, easting=index * 20.0, northing=0.0, elevation=100.0, total_depth=8.0)
        for index, hole in enumerate(ids)
    ]
    lithologies = [
        Lithology(hole_id=hole, from_depth=0.0, to_depth=8.0, lithology_code="Clay") for hole in ids
    ]
    projected, polygons, _ = run_pipeline(collars, lithologies, [(0.0, 0.0), (200.0, 0.0)])
    profile = SECTION_SHEET_PROFILE.model_copy(update={"y_axis_mode": "depth_below_collar"})
    renderer = CrossSectionRenderer(
        show_legend=True, render_profile=profile, title="Borehole Cross-Section"
    )
    figure = renderer.render(polygons, projected, collar_depths={hole: 8.0 for hole in ids})
    mpl_renderer = _figure_renderer(figure)
    figure.draw_without_rendering()
    title_box = figure.axes[0].title.get_window_extent(mpl_renderer)
    header_boxes = [Text.get_window_extent(t, mpl_renderer) for t in renderer._header_labels]
    assert header_boxes
    assert all(_overlap_area(box, title_box) == 0 for box in header_boxes)
    # With the title lifted clear, no header needs to stagger around it: one row.
    assert len({round(box.y0) for box in header_boxes}) == 1
    assert title_box.y0 > max(box.y1 for box in header_boxes)
    assert title_box.y1 <= figure.bbox.y1  # still on the page


def test_consulting_right_axis_label_stays_on_the_fixed_page() -> None:
    """The twin RL axis label sat 6 px past the letter page edge, so PNG and
    PDF exports silently dropped it."""
    ids, projected, polygons = _dense_header_section(3, 40.0)
    renderer = CrossSectionRenderer(show_legend=True, render_profile=CONSULTING_SECTION_PROFILE)
    figure = renderer.render(polygons, projected, collar_depths={hole: 8.0 for hole in ids})
    figure.draw_without_rendering()
    mpl_renderer = figure.canvas.get_renderer()
    right_labels = [
        ax.yaxis.label
        for ax in figure.axes
        if ax.yaxis.get_label_position() == "right" and ax.yaxis.label.get_text().strip()
    ]
    assert right_labels, "consulting profile draws a right-hand RL axis label"
    for label in right_labels:
        assert label.get_window_extent(mpl_renderer).x1 <= figure.bbox.x1


def test_very_long_titles_are_shortened_for_export() -> None:
    renderer = CrossSectionRenderer(title="T" * 300)
    assert len(renderer.title) <= 120 and renderer.title.endswith("…")
    assert CrossSectionRenderer(title="Section A-A'").title == "Section A-A'"


def test_consulting_groundwater_note_only_with_water_data() -> None:
    from render_theme import DEFAULT_CONSULTING_NOTES

    ids, projected, polygons = _dense_header_section(3, 40.0)

    def note_texts(water):
        renderer = CrossSectionRenderer(show_legend=True, render_profile=CONSULTING_SECTION_PROFILE)
        figure = renderer.render(
            polygons, projected, collar_depths={h: 8.0 for h in ids}, water_levels=water
        )
        return [t.get_text() for ax in figure.axes for t in ax.texts]

    dry = " ".join(note_texts(None))
    assert "GROUNDWATER BASED ON" not in dry
    assert "masl DENOTES" in dry
    wet = " ".join(note_texts([WaterLevel(hole_id=ids[0], depth=2.0)]))
    assert DEFAULT_CONSULTING_NOTES[0].split(" ")[0] in wet


def test_dense_consulting_headers_go_vertical_instead_of_overlapping() -> None:
    """30 long IDs 6 m apart on the consulting sheet: no horizontal layout
    fits, and the plot starts at the page top, so room is reserved for
    vertical headers."""
    ids, projected, polygons = _dense_header_section(30, 6.0)
    renderer = CrossSectionRenderer(show_legend=True, render_profile=CONSULTING_SECTION_PROFILE)
    figure = renderer.render(polygons, projected, collar_depths={hole: 8.0 for hole in ids})
    pairs, tick_hits = _header_hits(figure, renderer)
    assert pairs == 0 and tick_hits == 0
    assert all(t.get_rotation() == 90 for t in renderer._header_labels)
    assert all(t.get_window_extent(figure.canvas.get_renderer()).y1 <= figure.bbox.y1 for t in renderer._header_labels)


def test_agreed_lithology_scheme_groups_and_hatches() -> None:
    """Meeting 1 Oct 2026: one base colour per soil group, hatch marks the
    secondary component, Coal is the only black unit, Topsoil sits between
    clay/silt and organics in darkness with no hatch."""
    from constants import HATCH_GRAVEL, HATCH_PLUS, HATCH_SANDY, HATCH_SILTY

    def colour(code):
        return get_lithology_style(code).color.upper()

    def hatch(code):
        return get_lithology_style(code).hatch

    assert {colour(c) for c in ("Clay", "Sandy Clay", "Silty Clay", "Silty Clay Loam")} == {"#967259"}
    assert {colour(c) for c in ("Sandy Clay Loam", "Clay Loam", "Loam", "Silty Loam")} == {"#C68642"}
    assert {colour(c) for c in ("Sand", "Loamy Sand", "Silty Sand", "Sand and Gravel")} == {"#FFE39F"}
    assert {colour(c) for c in ("Siltstone", "Sandstone", "Mudstone")} == {"#4C516D"}
    assert hatch("Sandy Clay") == HATCH_SANDY and hatch("Silty Clay") == HATCH_SILTY
    assert hatch("Silty Clay Loam") == hatch("Clay Loam") == HATCH_PLUS
    assert hatch("Sand and Gravel") == hatch("Gravel") == HATCH_GRAVEL
    assert hatch("Silty Sand") == hatch("Loamy Sand") == HATCH_SILTY
    # Template 261002: Sand carries the same stipple as the other sandy units
    # (same dot density as Sandy Clay, Sandy Clay Loam and Sandstone).
    assert hatch("Sand") == hatch("Sandy Clay") == hatch("Sandy Clay Loam") == hatch("Sandstone") == HATCH_SANDY
    assert hatch("Clay") == hatch("Silt") == hatch("Loam") == hatch("Topsoil") == ""

    def luminance(code):
        r, g, b = (int(colour(code)[i : i + 2], 16) for i in (1, 3, 5))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    assert colour("Coal") == "#000000"
    # Only Coal is black; Organics is the darkest *brown* (#38220F), not black.
    assert all(colour(c) != "#000000" and luminance(c) > 25 for c in USGS_LITHOLOGY_COLORS if c != "Coal")
    assert luminance("Organics") < luminance("Topsoil") < min(luminance("Clay"), luminance("Silt"))


def test_consulting_first_borehole_is_drawn_in_full() -> None:
    """The first hole projects to x = 0; the axis used to start at 0 and clip
    the left half of its column. The axis now starts slightly negative, with
    no negative tick labels."""
    ids, projected, polygons = _dense_header_section(3, 40.0)
    renderer = CrossSectionRenderer(show_legend=True, render_profile=CONSULTING_SECTION_PROFILE)
    figure = renderer.render(polygons, projected, collar_depths={hole: 8.0 for hole in ids})
    figure.canvas.draw()
    ax = figure.axes[0]
    half = renderer._track_half_width(projected["x_profile"].to_numpy(dtype=float))
    assert ax.get_xlim()[0] <= -half  # whole first column inside the axes
    labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    assert labels and not any(label.lstrip().startswith(("-", "−")) for label in labels)
    assert "0" in labels


def test_old_excel_legend_is_fallback_only(tmp_path, monkeypatch, caplog) -> None:
    """Template 261002 supersedes the old BH Log hex sheet: a leftover Excel
    legend never recolours scheme codes; it only adds codes the scheme lacks."""
    import logging

    import openpyxl

    import constants
    import paths

    legend = tmp_path / "BH Log Lithology Legend.xlsx"
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["BH Log Lithology Legend"])  # title row; the parser reads header=1
    sheet.append(["Colour", "", "Lithology", "RGB"])
    sheet.append(["#38220F", "", "Clay", ""])
    sheet.append(["#123456", "", "Legacy Fill Unit", ""])
    book.save(legend)
    monkeypatch.setattr(paths, "bh_log_lithology_legend_xlsx_path", lambda: legend)
    monkeypatch.setattr(constants, "bh_log_lithology_legend_xlsx_path", lambda: legend)
    constants._load_bh_log_lithology_colors.cache_clear()
    try:
        with caplog.at_level(logging.WARNING):
            palette = constants._build_lithology_palette()
        assert palette["Clay"] == constants.USGS_LITHOLOGY_COLORS["Clay"] != "#38220F"
        assert palette["Coal"] == "#000000" and palette["Silty Sand"] == "#FFE39F"
        assert palette["Legacy Fill Unit"] == "#123456"  # old-only code falls back
        assert any("the scheme wins" in r.message for r in caplog.records)
    finally:
        constants._load_bh_log_lithology_colors.cache_clear()


def test_style_override_can_be_cleared(tmp_path, monkeypatch) -> None:
    import constants
    import paths

    override_file = tmp_path / "lithology_styles.json"
    monkeypatch.setattr(paths, "lithology_styles_path", lambda: override_file)
    monkeypatch.setattr(constants, "lithology_styles_path", lambda: override_file)
    constants._load_lithology_style_overrides.cache_clear()
    constants.get_lithology_style.cache_clear()
    try:
        constants.save_lithology_style_override("topsoil", "#8B6914", "..")
        assert constants.get_lithology_style("Topsoil").hatch == ".."
        assert constants.has_lithology_style_override("Topsoil")
        assert constants.clear_lithology_style_override("TOPSOIL")
        assert not constants.has_lithology_style_override("Topsoil")
        assert constants.get_lithology_style("Topsoil").hatch == ""  # back to the scheme
        assert not constants.clear_lithology_style_override("Topsoil")
    finally:
        constants._load_lithology_style_overrides.cache_clear()
        constants.get_lithology_style.cache_clear()


def test_consulting_legend_holds_many_units_without_overlap_or_silent_loss() -> None:
    """31 codes used to overrun the panel: the header overlapped the first row
    and everything past twelve entries vanished with no indication."""
    from matplotlib.text import Text

    from renderer_water import _overlap_area

    codes = sorted(USGS_LITHOLOGY_COLORS)
    ids = [f"BH-{i:02d}" for i in range(4)]
    collars = [
        Collar(hole_id=h, easting=i * 25.0, northing=0.0, elevation=100.0, total_depth=float(len(codes)))
        for i, h in enumerate(ids)
    ]
    lithologies = [
        Lithology(hole_id=h, from_depth=float(k), to_depth=float(k + 1), lithology_code=code)
        for h in ids
        for k, code in enumerate(codes)
    ]
    projected, polygons, _ = run_pipeline(collars, lithologies, [(0.0, 0.0), (75.0, 0.0)])
    renderer = CrossSectionRenderer(show_legend=True, render_profile=CONSULTING_SECTION_PROFILE)
    figure = renderer.render(polygons, projected, collar_depths={h: float(len(codes)) for h in ids}, lithology_codes=codes)
    figure.draw_without_rendering()
    mpl_renderer = figure.canvas.get_renderer()
    legend_ax = next(ax for ax in figure.axes if any(t.get_text() == "LEGEND" for t in ax.texts))
    texts = [t for t in legend_ax.texts if t.get_text().strip()]
    labels = {t.get_text() for t in texts}
    header = next(t for t in texts if t.get_text() == "LEGEND")
    header_box = Text.get_window_extent(header, mpl_renderer)
    entry_boxes = [Text.get_window_extent(t, mpl_renderer) for t in texts if t is not header]
    assert all(_overlap_area(header_box, box) == 0 for box in entry_boxes)
    note = next(label for label in labels if label.startswith("+") and "MORE UNITS" in label)
    shown_codes = [c for c in codes if c.upper() in labels]
    hidden = int(note.split()[0].lstrip("+"))
    # Every unit is either listed or counted in the note; the panel is full.
    assert len(shown_codes) + hidden == len(codes)
    assert len(shown_codes) >= 10
    panel = legend_ax.get_window_extent(mpl_renderer)
    assert all(box.y0 >= panel.y0 - 1 and box.y1 <= panel.y1 + 1 for box in entry_boxes)
    for a, b in itertools.combinations(entry_boxes, 2):
        assert _overlap_area(a, b) == 0


def _consulting_figure(n_holes: int = 3, spacing: float = 40.0, **profile_updates):
    ids, projected, polygons = _dense_header_section(n_holes, spacing)
    title_block = ConsultingTitleBlock(
        section_label="A-A'",
        transect_start_label="A",
        transect_start_secondary="WEST",
        transect_end_label="A'",
        transect_end_secondary="EAST",
        notes=(
            "GROUNDWATER BASED ON GROUNDWATER MONITORING WELL OBSERVATIONS ONLY, SEE TABLE 2.",
            "masl DENOTES METRES ABOVE SEA LEVEL.",
            "LITHOLOGY BETWEEN BOREHOLES IS INFERRED AND SCHEMATIC ONLY.",
        ),
    )
    renderer = CrossSectionRenderer(
        show_legend=True,
        render_profile=CONSULTING_SECTION_PROFILE.model_copy(update=profile_updates),
        consulting_title_block=title_block,
    )
    figure = renderer.render(polygons, projected, collar_depths={h: 8.0 for h in ids})
    figure.draw_without_rendering()
    return renderer, figure


def _texts_outside_page(figure) -> list[str]:
    mpl_renderer = figure.canvas.get_renderer()
    outside = []
    for ax in figure.axes:
        for text in ax.texts:
            if not text.get_text().strip() or not text.get_visible():
                continue
            box = text.get_window_extent(mpl_renderer)
            if box.x0 < figure.bbox.x0 - 0.5 or box.x1 > figure.bbox.x1 + 0.5 or box.y1 > figure.bbox.y1 + 0.5:
                outside.append(text.get_text())
    return outside


def test_consulting_end_labels_print_on_the_page() -> None:
    """'A / WEST' and 'A' / EAST' sat above the page edge on every letter
    sheet and were silently cut from PNG and PDF."""
    _, figure = _consulting_figure()
    labels = [t.get_text() for ax in figure.axes for t in ax.texts]
    assert any("WEST" in label for label in labels) and any("EAST" in label for label in labels)
    assert not [t for t in _texts_outside_page(figure) if "WEST" in t or "EAST" in t]
    # ...and they sit beside the hole-ID header strip, not on it.
    from matplotlib.text import Text

    from renderer_water import _overlap_area

    renderer, figure = _consulting_figure()
    mpl_renderer = figure.canvas.get_renderer()
    ends = [t for ax in figure.axes for t in ax.texts if "WEST" in t.get_text() or "EAST" in t.get_text()]
    end_boxes = [Text.get_window_extent(t, mpl_renderer) for t in ends]
    header_boxes = [Text.get_window_extent(t, mpl_renderer) for t in renderer._header_labels]
    assert all(_overlap_area(e, h) == 0 for e in end_boxes for h in header_boxes)


def test_consulting_notes_do_not_overlap_each_other() -> None:
    from matplotlib.text import Text

    from renderer_water import _overlap_area

    _, figure = _consulting_figure()
    mpl_renderer = figure.canvas.get_renderer()
    notes = [t for ax in figure.axes for t in ax.texts if t.get_text()[:2] in ("1.", "2.", "3.")]
    assert len(notes) >= 2
    boxes = [Text.get_window_extent(t, mpl_renderer) for t in notes]
    for a, b in itertools.combinations(boxes, 2):
        assert _overlap_area(a, b) == 0


def test_last_hole_value_labels_stay_on_the_page() -> None:
    """Right-only candidates pushed the last column's values past the page
    edge; a left-of-column fallback now catches them."""
    ids = [f"BH-{i:02d}" for i in range(6)]
    collars = [Collar(hole_id=h, easting=i * 40.0, northing=0.0, elevation=100.0, total_depth=8.0) for i, h in enumerate(ids)]
    lithologies = [Lithology(hole_id=h, from_depth=0.0, to_depth=8.0, lithology_code="Clay") for h in ids]
    readings = [
        EnvironmentalReading(hole_id=ids[-1], parameter="Chloride", value=1000.0 + k, depth=1.0 + k)
        for k in range(6)
    ]
    projected, polygons, _ = run_pipeline(collars, lithologies, [(0.0, 0.0), (200.0, 0.0)])
    renderer = CrossSectionRenderer(
        show_legend=True,
        render_profile=CONSULTING_SECTION_PROFILE.model_copy(
            update={"show_parameter_markers": True, "show_parameter_labels": True}
        ),
        environmental_readings=readings,
        environmental_parameters=("Chloride",),
    )
    figure = renderer.render(polygons, projected, collar_depths={h: 8.0 for h in ids})
    figure.draw_without_rendering()
    assert not [t for t in _texts_outside_page(figure) if t.startswith("100")]
    # ...and they do not sit on the twin RL axis tick numbers either.
    from matplotlib.text import Text

    from renderer_water import _drawn_tick_labels, _overlap_area

    mpl_renderer = figure.canvas.get_renderer()
    values = [Text.get_window_extent(a, mpl_renderer) for k, a, _c in renderer._water_labels if k == "chem"]
    ticks = [
        t.get_window_extent(mpl_renderer)
        for ax in figure.axes
        for t in _drawn_tick_labels(ax.yaxis, ax.get_ylim())
        if t.get_visible() and t.get_text().strip()
    ]
    assert values and not any(_overlap_area(v, t) > 0 for v in values for t in ticks)


def test_unknown_lithology_codes_get_distinct_fallback_styles() -> None:
    styles = {code: get_lithology_style(code) for code in ("CL", "SM", "TILL", "GP")}
    assert all(s.color.startswith("#") and len(s.hatch) >= 2 for s in styles.values())
    assert len({(s.color, s.hatch) for s in styles.values()}) >= 3  # not one grey for all
    assert get_lithology_style("CL") == get_lithology_style("cl")  # stable per code


# Fill colours read from the client CAD template Cross_Section_Litho_Legend_261002.
_CAD_TEMPLATE_261002_COLOURS = {
    "Clay": "#967259", "Silt": "#8D5524", "Loam": "#C68642", "Sand": "#FFE39F",
    "Topsoil": "#534230", "Organics": "#38220F", "Fill": "#854442", "Gravel": "#D9D9D9",
    "Mudstone": "#4C516D", "Drilling Waste": "#808080", "Other": "#4D5D53",
    "No Recovery": "#FFFFFF", "Bentonite": "#BFBFBF", "Coal": "#000000", "Refuse": "#8C973D",
    "Sandy Clay": "#967259", "Silty Clay": "#967259", "Silty Clay Loam": "#967259",
    "Sand and Gravel": "#FFE39F", "Loamy Sand": "#FFE39F", "Sandy Clay Loam": "#C68642",
    "Silty Loam": "#C68642", "Clay Loam": "#C68642", "Sandstone": "#4C516D", "Siltstone": "#4C516D",
}


def test_palette_matches_cad_template_261002() -> None:
    mismatched = {
        code: (expected, get_lithology_style(code).color.upper())
        for code, expected in _CAD_TEMPLATE_261002_COLOURS.items()
        if get_lithology_style(code).color.upper() != expected
    }
    assert not mismatched, mismatched


def test_section_sheet_legend_stays_on_the_page() -> None:
    """Long lithology names in a two-column legend ran off the right edge."""
    import matplotlib.pyplot as plt

    codes = ["Sandy Clay Loam", "Silty Clay Loam", "Sand and Gravel", "Loamy Sand", "Clay Loam", "Fill"]
    collars = [Collar(hole_id=f"BH-{i}", easting=10.0 * i, northing=0.0, elevation=100.0, total_depth=12.0) for i in range(3)]
    liths = [
        Lithology(hole_id=c.hole_id, from_depth=2.0 * k, to_depth=2.0 * (k + 1), lithology_code=code)
        for c in collars
        for k, code in enumerate(codes)
    ]
    projected, polygons, _ = run_pipeline(collars, liths, [(0.0, 0.0), (20.0, 0.0)])
    for page in ("letter_landscape", "letter_portrait"):
        from export_framing import ExportFramingConfig

        renderer = CrossSectionRenderer(
            show_legend=True,
            render_profile=SECTION_SHEET_PROFILE,
            export_framing=ExportFramingConfig(page_preset=page),
        )
        fig = renderer.render(polygons, projected, collar_depths={c.hole_id: 12.0 for c in collars})
        try:
            renderer._prepare_export_figure(fig)
            fig.draw_without_rendering()
            box = fig.axes[0].get_legend().get_window_extent(fig.canvas.get_renderer())
            assert box.x1 <= fig.bbox.x1 + 0.5, page
        finally:
            plt.close(fig)
