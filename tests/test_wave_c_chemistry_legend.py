"""Wave C: chemistry colour modes, two-column legend, profile defaults."""

from __future__ import annotations

from models import Collar, EnvironmentalReading, Lithology
from pipeline import build_cross_section
from render_profiles import CONSULTING_SECTION_PROFILE, SECTION_SHEET_PROFILE
from render_theme import (
    CHEMISTRY_LABEL_BLACK,
    CHEMISTRY_LABEL_GREEN,
    CHEMISTRY_LABEL_RED,
    CHEMISTRY_LABEL_YELLOW,
    chemistry_label_color,
)
from tests.conftest import assert_valid_svg


def test_chemistry_label_color_black_default() -> None:
    assert chemistry_label_color(500.0, "black") == CHEMISTRY_LABEL_BLACK
    assert chemistry_label_color(500.0, "threshold", green_max=None, yellow_max=250.0) == (
        CHEMISTRY_LABEL_BLACK
    )


def test_chemistry_label_color_threshold_bands() -> None:
    assert chemistry_label_color(50.0, "threshold", green_max=100.0, yellow_max=250.0) == (
        CHEMISTRY_LABEL_GREEN
    )
    assert chemistry_label_color(150.0, "threshold", green_max=100.0, yellow_max=250.0) == (
        CHEMISTRY_LABEL_YELLOW
    )
    assert chemistry_label_color(300.0, "threshold", green_max=100.0, yellow_max=250.0) == (
        CHEMISTRY_LABEL_RED
    )


def test_section_sheet_profile_wave_c_defaults() -> None:
    assert SECTION_SHEET_PROFILE.chemistry_color_mode == "black"
    assert SECTION_SHEET_PROFILE.legend_ncol == 2
    assert CONSULTING_SECTION_PROFILE.legend_ncol == 2


def _minimal_chemistry_section(
    *,
    chemistry_color_mode: str = "black",
    green_max: float | None = None,
    yellow_max: float | None = None,
    legend_ncol: int | None = None,
):
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=20.0),
    ]
    lithologies = [
        Lithology(hole_id=hole, from_depth=0.0, to_depth=20.0, lithology_code="Clay")
        for hole in ("BH-01", "BH-02")
    ]
    readings = [
        EnvironmentalReading(hole_id="BH-01", parameter="Chloride", value=50.0, depth=4.0, unit="mg/L"),
        EnvironmentalReading(hole_id="BH-02", parameter="Chloride", value=300.0, depth=4.0, unit="mg/L"),
    ]
    kwargs: dict[str, object] = {
        "environmental_readings": readings,
        "environmental_parameters": ("Chloride",),
        "render_layout": "section_sheet",
        "parameter_draw_markers": False,
        "show_parameter_labels": True,
        "chemistry_color_mode": chemistry_color_mode,
    }
    if green_max is not None:
        kwargs["chemistry_threshold_green_max"] = green_max
    if yellow_max is not None:
        kwargs["chemistry_threshold_yellow_max"] = yellow_max
    if legend_ncol is not None:
        kwargs["legend_ncol"] = legend_ncol
    return build_cross_section(
        collars,
        lithologies,
        [(0.0, 0.0), (50.0, 0.0)],
        **kwargs,
    )


def test_threshold_colours_appear_in_svg() -> None:
    result = _minimal_chemistry_section(
        chemistry_color_mode="threshold",
        green_max=100.0,
        yellow_max=250.0,
    )
    assert_valid_svg(result.svg_bytes)
    text = result.svg_bytes.decode("utf-8", errors="ignore").lower()
    assert "50" in text
    assert "300" in text
    assert CHEMISTRY_LABEL_GREEN.lower() in text
    assert CHEMISTRY_LABEL_RED.lower() in text


def test_black_mode_uses_neutral_label_colour() -> None:
    result = _minimal_chemistry_section(chemistry_color_mode="black")
    assert_valid_svg(result.svg_bytes)
    text = result.svg_bytes.decode("utf-8", errors="ignore").lower()
    assert CHEMISTRY_LABEL_BLACK.lower() in text
    assert CHEMISTRY_LABEL_RED.lower() not in text
