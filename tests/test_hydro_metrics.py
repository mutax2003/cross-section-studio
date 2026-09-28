"""Tests for schematic horizontal hydraulic gradient metrics."""

from __future__ import annotations

import pytest

from hydro_metrics import (
    absurd_horizontal_gradient_warnings,
    format_gradient_label,
    horizontal_gradients_along_profile,
    is_measured_water_level,
    water_head_masl,
    water_status,
)
from models import WaterLevel


def test_water_head_prefers_elevation_masl() -> None:
    level = WaterLevel(hole_id="BH-01", depth=2.0, elevation_masl=98.5)
    assert water_head_masl(level, collar_rl=100.0) == pytest.approx(98.5)
    depth_only = WaterLevel(hole_id="BH-01", depth=2.0)
    assert water_head_masl(depth_only, collar_rl=100.0) == pytest.approx(98.0)


def test_water_status_normalization() -> None:
    assert water_status(WaterLevel(hole_id="A", depth=1.0)) == "measured"
    assert water_status(WaterLevel(hole_id="A", depth=0.0, status="DRY")) == "dry"
    assert water_status(WaterLevel(hole_id="A", depth=0.0, status="not measured")) == "nm"
    assert not is_measured_water_level(WaterLevel(hole_id="A", depth=0.0, status="dry"))


def test_horizontal_gradients_along_profile() -> None:
    levels = (
        WaterLevel(hole_id="BH-01", depth=2.0, elevation_masl=98.0),
        WaterLevel(hole_id="BH-02", depth=3.0, elevation_masl=97.0),
        WaterLevel(hole_id="BH-03", depth=0.0, status="dry"),
    )
    segments = horizontal_gradients_along_profile(
        levels,
        hole_order=("BH-01", "BH-02", "BH-03"),
        x_by_hole={"BH-01": 0.0, "BH-02": 10.0, "BH-03": 20.0},
        collar_rl_by_hole={"BH-01": 100.0, "BH-02": 100.0, "BH-03": 100.0},
    )
    assert len(segments) == 1
    assert segments[0].left_hole_id == "BH-01"
    assert segments[0].right_hole_id == "BH-02"
    assert segments[0].gradient == pytest.approx(-0.1)
    assert "i=" in format_gradient_label(segments[0])


def test_absurd_gradient_warnings() -> None:
    levels = (
        WaterLevel(hole_id="BH-01", depth=1.0, elevation_masl=99.0),
        WaterLevel(hole_id="BH-02", depth=20.0, elevation_masl=80.0),
    )
    segments = horizontal_gradients_along_profile(
        levels,
        hole_order=("BH-01", "BH-02"),
        x_by_hole={"BH-01": 0.0, "BH-02": 10.0},
        collar_rl_by_hole={"BH-01": 100.0, "BH-02": 100.0},
    )
    warnings = absurd_horizontal_gradient_warnings(segments, series_label="May")
    assert warnings
    assert "May" in warnings[0]
    assert "|i|" in warnings[0]
