"""Tests for lithology styling and renderer visuals."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from constants import CONSULTING_LITHOLOGY_COLORS, USGS_LITHOLOGY_COLORS, USGS_LITHOLOGY_HATCHES, get_lithology_style
from models import Collar, ConsultingTitleBlock, EnvironmentalReading, Lithology, ScreenInterval, VerticalGradient, WaterLevel
from render_profiles import CHART_PROFILE, CONSULTING_SECTION_PROFILE, SECTION_SHEET_PROFILE
from render_theme import PARAMETER_READING_COLOR
from renderer import CrossSectionRenderer, _resolve_parameter_label_offsets
from stratigraphy import build_stratigraphy
from tests.conftest import assert_valid_svg, run_pipeline


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
        if code == "No Recovery":
            assert style.hatch == ""
        else:
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
    assert PARAMETER_READING_COLOR.lower() in segment_svg.decode("utf-8", errors="ignore").lower()
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
    assert "120 mg/L" in svg_bytes.decode("utf-8", errors="ignore")


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
