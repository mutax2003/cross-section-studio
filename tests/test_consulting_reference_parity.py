"""Structural parity checks against consulting reference sheet symbology."""

from __future__ import annotations

from models import (
    Collar,
    ConsultingTitleBlock,
    Lithology,
    ScreenInterval,
    VerticalGradient,
    WaterLevel,
)
from pipeline import build_cross_section
from ui_helpers import svg_is_valid


def _reference_fixture():
    """Five-well transect with topsoil/sand/clay, GW, screens, and gradients."""
    collars = [
        Collar(hole_id="MW11", easting=0.0, northing=0.0, elevation=670.0, total_depth=25.0),
        Collar(hole_id="MW10", easting=108.0, northing=0.0, elevation=668.5, total_depth=22.0),
        Collar(hole_id="MW08", easting=137.0, northing=0.0, elevation=667.0, total_depth=20.0),
        Collar(hole_id="MW06", easting=212.0, northing=0.0, elevation=665.5, total_depth=18.0),
        Collar(hole_id="MW02", easting=240.0, northing=0.0, elevation=664.0, total_depth=16.0),
    ]
    lithologies = []
    for hole_id, sand_bottom, clay_bottom in (
        ("MW11", 3.0, 25.0),
        ("MW10", 2.5, 22.0),
        ("MW08", 2.0, 20.0),
        ("MW06", 2.0, 18.0),
        ("MW02", 1.5, 16.0),
    ):
        lithologies.extend(
            [
                Lithology(hole_id=hole_id, from_depth=0.0, to_depth=0.5, lithology_code="Topsoil"),
                Lithology(hole_id=hole_id, from_depth=0.5, to_depth=sand_bottom, lithology_code="Sand"),
                Lithology(hole_id=hole_id, from_depth=sand_bottom, to_depth=clay_bottom, lithology_code="Clay"),
            ]
        )
    water = [
        WaterLevel(hole_id="MW11", depth=3.7),
        WaterLevel(hole_id="MW10", depth=3.5),
        WaterLevel(hole_id="MW08", depth=2.4),
        WaterLevel(hole_id="MW06", depth=1.1),
    ]
    screens = [
        ScreenInterval(hole_id="MW11", from_depth=8.0, to_depth=14.0),
        ScreenInterval(hole_id="MW08", from_depth=6.0, to_depth=12.0),
    ]
    gradients = [
        VerticalGradient(hole_id="MW11", direction="down"),
        VerticalGradient(hole_id="MW08", direction="up"),
    ]
    transect_points = [(c.easting, c.northing) for c in collars]
    title_block = ConsultingTitleBlock(
        section_label="B-B'",
        transect_start_primary="B",
        transect_start_secondary="SOUTHWEST",
        transect_end_primary="B'",
        transect_end_secondary="NORTHEAST",
        map_scale="1:1 000",
        project_number="12-30-036-04 W4M",
        source="ECOVENTURE 2025",
        date="11/05/25",
        drawn_by="EC",
        revised="SL",
        figure_number="6",
        prepared_for="SURGE ENERGY INC",
        prepared_by="ECOVENTURE",
        y_axis_label="ELEVATION (m)",
        screen_legend_label="SCREEN INTERVAL",
        show_gradient_legend=True,
    )
    return collars, lithologies, transect_points, water, screens, gradients, title_block


def test_consulting_reference_parity_structure() -> None:
    collars, lithologies, transect_points, water, screens, gradients, title_block = (
        _reference_fixture()
    )
    result = build_cross_section(
        collars,
        lithologies,
        transect_points,
        render_layout="consulting_section",
        vertical_exaggeration=5.0,
        show_legend=False,
        show_hatches=True,
        water_levels=water,
        consulting_title_block=title_block,
        screen_intervals=screens,
        vertical_gradients=gradients,
        export_formats=frozenset({"svg"}),
    )
    assert svg_is_valid(result.svg_bytes)
    text = result.svg_bytes.decode("utf-8", errors="ignore")
    assert "ELEVATION (m)" in text
    assert "DISTANCE (m)" in text
    assert "SCREEN INTERVAL" in text
    assert "VERTICAL GRADIENT DIRECTION" in text
    assert "GROUNDWATER ELEVATION" in text.upper()
    assert "GROUNDWATER LEVEL" in text.upper()
    assert " masl" in text
    assert "CROSS SECTION B-B'" in text
    assert "SOUTHWEST" in text
    assert "NORTHEAST" in text
    assert "Metres" in text
    assert "DRAWN BY" in text
    assert "FIGURE NO." in text
    assert text.count("ELEVATION (m)") >= 2
