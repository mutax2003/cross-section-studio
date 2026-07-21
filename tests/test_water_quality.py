"""Tests for groundwater QA summaries."""

from __future__ import annotations

from ai_quality import summarize_water_levels
from models import Collar, WaterLevel


def test_summarize_water_levels_reports_series_gaps() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=20.0),
    ]
    levels = (
        WaterLevel(hole_id="BH-01", depth=2.0, series_id="2024-05", series_label="May 2024"),
        WaterLevel(hole_id="BH-01", depth=2.5, series_id="2024-06", series_label="June 2024"),
    )
    summary = summarize_water_levels(collars, levels, ("BH-01", "BH-02"))
    assert summary.total_levels == 2
    assert summary.holes_without_any_water == ("BH-02",)
    june = next(item for item in summary.series if item.series_id == "2024-06")
    assert june.missing_hole_ids == ("BH-02",)


def test_summarize_water_levels_warns_when_all_off_transect() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=20.0),
        Collar(hole_id="BH-99", easting=200.0, northing=0.0, elevation=100.0, total_depth=20.0),
    ]
    levels = (WaterLevel(hole_id="BH-99", depth=2.0, series_id="2024-05", series_label="May 2024"),)
    summary = summarize_water_levels(collars, levels, ("BH-01", "BH-02"))
    assert summary.total_levels == 0
    assert summary.series == ()
    assert any("No groundwater readings on the selected transect" in warning for warning in summary.warnings)
    assert any("BH-99" in warning for warning in summary.warnings)


def test_summarize_screen_intervals_flags_beyond_td_and_overlap() -> None:
    from ai_quality import summarize_screen_intervals
    from models import ScreenInterval

    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    screens = (
        ScreenInterval(hole_id="BH-01", from_depth=2.0, to_depth=12.0),
        ScreenInterval(hole_id="BH-01", from_depth=8.0, to_depth=9.5),
    )
    summary = summarize_screen_intervals(collars, screens, ("BH-01",))
    assert any("exceeds total depth" in warning for warning in summary.warnings)
    assert any("overlapping screens" in warning for warning in summary.warnings)


def test_primary_water_depth_prefers_last_series_in_workbook_order() -> None:
    from render_theme import primary_water_depth_by_hole

    levels = (
        WaterLevel(hole_id="BH-01", depth=1.0, series_id="2024-05", series_label="May 2024"),
        WaterLevel(hole_id="BH-01", depth=2.0, series_id="2024-06", series_label="June 2024"),
        WaterLevel(hole_id="BH-02", depth=3.0, series_id="2024-06", series_label="June 2024"),
    )
    assert primary_water_depth_by_hole(levels) == {"BH-01": 2.0, "BH-02": 3.0}


def test_water_has_multiple_series_requires_distinct_ids() -> None:
    from render_theme import water_has_multiple_series

    single_named = (
        WaterLevel(hole_id="BH-01", depth=1.0, series_id="2024-05", series_label="May 2024"),
        WaterLevel(hole_id="BH-02", depth=2.0, series_id="2024-05", series_label="May 2024"),
    )
    assert water_has_multiple_series(single_named) is False

    multi = (
        WaterLevel(hole_id="BH-01", depth=1.0, series_id="2024-05", series_label="May 2024"),
        WaterLevel(hole_id="BH-01", depth=2.0, series_id="2024-06", series_label="June 2024"),
    )
    assert water_has_multiple_series(multi) is True
    assert water_has_multiple_series(()) is False
