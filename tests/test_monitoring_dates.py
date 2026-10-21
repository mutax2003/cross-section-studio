"""Monitoring-event dates on Water / Environmental rows.

Heads compared across time are not a potentiometric surface:
* a series whose dated readings span more than 7 days warns in Validate;
* duplicate readings of one hole × series × nest use the LATEST date (not row
  order) for the plot, Δh/Δx, nested iv and the planar flow fit;
* several dated lab samples at one hole/parameter/depth: the latest is drawn
  and a QA note says so;
* undated data behave exactly as before (figure parity).
"""

from __future__ import annotations

import datetime
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest
from pydantic import ValidationError

from ai_quality import summarize_environmental_readings, summarize_water_levels
from hydro_metrics import (
    fit_head_planes,
    format_vertical_gradient_note,
    groundwater_flow_notes,
    horizontal_gradients_along_profile,
    latest_environmental_readings,
    nested_vertical_gradients,
    prefer_water_reading,
    series_legend_date,
    synoptic_date_warnings,
)
from ingestion import ingest_workbook
from models import (
    Collar,
    EnvironmentalReading,
    Lithology,
    ScreenInterval,
    WaterLevel,
    parse_monitoring_date,
)
from pipeline import (
    build_cross_section,
    compute_section_geometry,
    render_cross_section_from_geometry,
)
from workbook_template import ENVIRONMENTAL_COLUMNS, WATER_COLUMNS, export_cleaned_workbook_bytes

D = datetime.date
_POINTS = [(-10.0, 0.0), (210.0, 0.0)]


# --- parsing -------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        D(2026, 4, 2),
        datetime.datetime(2026, 4, 2, 10, 30),
        pd.Timestamp("2026-04-02 10:30"),
        "2026-04-02",
        " 2026/04/02 ",
        "2026-04-02T10:30:00",
        "2026-4-2",
        46114,  # Excel serial for 2026-04-02
        46114.4375,
    ],
)
def test_parse_variants(raw: object) -> None:
    assert parse_monitoring_date(raw) == D(2026, 4, 2)


@pytest.mark.parametrize("raw", [None, float("nan"), pd.NaT, "", "   ", "nan"])
def test_blank_is_none(raw: object) -> None:
    assert parse_monitoring_date(raw) is None
    assert WaterLevel(hole_id="A", depth=1.0, date=raw).date is None


@pytest.mark.parametrize("raw", ["spring", "02/04/2026", "2026-13-40", 12, True])
def test_unparseable_rejected(raw: object) -> None:
    with pytest.raises((ValueError, ValidationError)):
        WaterLevel(hole_id="A", depth=1.0, date=raw)


def test_columns_appended_at_end() -> None:
    assert WATER_COLUMNS[-1] == "date"
    assert ENVIRONMENTAL_COLUMNS[-1] == "date"
    assert list(WaterLevel.model_fields)[-1] == "date"
    assert list(EnvironmentalReading.model_fields)[-1] == "date"


def _workbook(water_rows, env_rows=None) -> BytesIO:
    collars = pd.DataFrame(
        [
            {"hole_id": h, "easting": e, "northing": 0, "elevation": 100, "total_depth": 20}
            for h, e in (("A", 0), ("B", 100), ("C", 200))
        ]
    )
    lith = pd.DataFrame(
        [
            {"hole_id": h, "from_depth": 0, "to_depth": 20, "lithology_code": "Sand"}
            for h in ("A", "B", "C")
        ]
    )
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        collars.to_excel(writer, sheet_name="Collars", index=False)
        lith.to_excel(writer, sheet_name="Lithology", index=False)
        pd.DataFrame(water_rows).to_excel(writer, sheet_name="Water", index=False)
        if env_rows:
            pd.DataFrame(env_rows).to_excel(writer, sheet_name="Environmental", index=False)
    buffer.seek(0)
    return buffer


def test_workbook_parsing_and_row_error() -> None:
    buffer = _workbook(
        [
            {"hole_id": "A", "depth": 2.0, "date": datetime.datetime(2026, 4, 2)},
            {"hole_id": "B", "depth": 2.5, "date": "2026-04-03"},
            {"hole_id": "C", "depth": 3.0, "date": None},
            {"hole_id": "C", "depth": 3.1, "date": "next tuesday"},
        ],
        [
            {"hole_id": "A", "parameter": "Benzene", "value": 1.0, "depth": 3.0, "date": "2026-05-01"},
            {"hole_id": "B", "parameter": "Benzene", "value": "<0.5", "depth": 3.0, "date": "bad"},
        ],
    )
    result, _report = ingest_workbook(buffer)
    dates = [level.date for level in result.water_levels]
    assert dates == [D(2026, 4, 2), D(2026, 4, 3), None]
    water_error = next(e for e in result.errors if e.startswith("Water row 5"))
    assert "date = 'next tuesday'" in water_error and "YYYY-MM-DD" in water_error
    assert [r.date for r in result.environmental_readings] == [D(2026, 5, 1)]
    assert any(e.startswith("Environmental row 3") and "date = 'bad'" in e for e in result.errors)


def test_export_round_trip_keeps_dates_and_dated_duplicates() -> None:
    buffer = _workbook(
        [
            {"hole_id": "A", "depth": 2.0, "date": "2026-04-02"},
            # Same hole/series/depth on another date: must not be deduped away.
            {"hole_id": "A", "depth": 2.0, "date": "2026-05-20"},
            {"hole_id": "B", "depth": 2.5},
        ],
        [{"hole_id": "A", "parameter": "Benzene", "value": 1.0, "depth": 3.0, "date": "2026-05-01"}],
    )
    result, _ = ingest_workbook(buffer)
    exported = export_cleaned_workbook_bytes(result, project_metadata={})
    again, _ = ingest_workbook(BytesIO(exported))
    assert [lv.date for lv in again.water_levels] == [D(2026, 4, 2), D(2026, 5, 20), None]
    assert [r.date for r in again.environmental_readings] == [D(2026, 5, 1)]
    assert not again.errors


# --- synoptic check ---------------------------------------------------------------


def test_synoptic_warning_and_mixed_info() -> None:
    levels = [
        WaterLevel(hole_id="A", depth=1.0, series_id="S26", series_label="Spring 2026", date=D(2026, 4, 2)),
        WaterLevel(hole_id="B", depth=1.0, series_id="S26", series_label="Spring 2026", date=D(2026, 5, 20)),
        WaterLevel(hole_id="C", depth=1.0, series_id="S26", series_label="Spring 2026"),
        WaterLevel(hole_id="A", depth=1.0, series_id="tight", date=D(2026, 4, 2)),
        WaterLevel(hole_id="B", depth=1.0, series_id="tight", date=D(2026, 4, 9)),
    ]
    warnings, info = synoptic_date_warnings(levels)
    assert warnings == [
        "Series 'Spring 2026' mixes readings from 2026-04-02 to 2026-05-20 (48 days) — heads "
        "from different dates are not one potentiometric surface; split the series by "
        "monitoring event."
    ]
    assert len(info) == 1 and "dated (2026-04-02 to 05-20) and undated readings (C)" in info[0]
    # Configurable window.
    assert len(synoptic_date_warnings(levels, window_days=3)[0]) == 2
    # Undated data: nothing.
    assert synoptic_date_warnings([WaterLevel(hole_id="A", depth=1.0)]) == ([], [])


def test_synoptic_in_validate_summary() -> None:
    collars = [
        Collar(hole_id=h, easting=e, northing=0.0, elevation=100.0, total_depth=20.0)
        for h, e in (("A", 0.0), ("B", 100.0))
    ]
    levels = [
        WaterLevel(hole_id="A", depth=1.0, series_label="Spring 2026", date=D(2026, 4, 2)),
        WaterLevel(hole_id="B", depth=1.2, series_label="Spring 2026", date=D(2026, 5, 20)),
    ]
    summary = summarize_water_levels(collars, levels, ["A", "B"])
    assert any("(48 days)" in w for w in summary.warnings)
    loose = summarize_water_levels(collars, levels, ["A", "B"], synoptic_window_days=60)
    assert not any("days)" in w for w in loose.warnings)


# --- latest-wins duplicates ----------------------------------------------------------


def _dup_levels(order: str) -> list[WaterLevel]:
    old = WaterLevel(hole_id="B", depth=10.0, date=D(2026, 4, 1))  # head 90
    new = WaterLevel(hole_id="B", depth=5.0, date=D(2026, 4, 3))  # head 95
    first, second = (old, new) if order == "old-first" else (new, old)
    return [
        WaterLevel(hole_id="A", depth=4.0, date=D(2026, 4, 2)),
        first,
        WaterLevel(hole_id="C", depth=6.0, date=D(2026, 4, 2)),
        second,
    ]


def test_prefer_water_reading_rule() -> None:
    old = WaterLevel(hole_id="B", depth=10.0, date=D(2026, 4, 1))
    new = WaterLevel(hole_id="B", depth=5.0, date=D(2026, 4, 3))
    undated = WaterLevel(hole_id="B", depth=7.0)
    assert prefer_water_reading(old, new) is new
    assert prefer_water_reading(new, old) is new
    assert prefer_water_reading(new, undated) is new  # dated never displaced by undated
    assert prefer_water_reading(undated, old) is old
    plain = WaterLevel(hole_id="B", depth=8.0)
    assert prefer_water_reading(undated, plain) is plain  # undated: last row wins


@pytest.mark.parametrize("order", ["old-first", "new-first"])
def test_latest_wins_in_horizontal_gradients(order: str) -> None:
    segments = horizontal_gradients_along_profile(
        _dup_levels(order),
        hole_order=["A", "B", "C"],
        x_by_hole={"A": 0.0, "B": 100.0, "C": 200.0},
        collar_rl_by_hole={"A": 100.0, "B": 100.0, "C": 100.0},
    )
    assert [s.head_right_masl for s in segments][0] == pytest.approx(95.0)


@pytest.mark.parametrize("order", ["old-first", "new-first"])
def test_latest_wins_in_flow_fit_and_note(order: str) -> None:
    collars = [
        Collar(hole_id="A", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0),
        Collar(hole_id="B", easting=100.0, northing=0.0, elevation=100.0, total_depth=20.0),
        Collar(hole_id="C", easting=0.0, northing=100.0, elevation=100.0, total_depth=20.0),
    ]
    (fit,) = fit_head_planes(collars, _dup_levels(order))
    residual_free_head_b = 95.0
    # dh/dE = (h_B - h_A) / 100 with h_A = 96.
    assert fit.dh_de == pytest.approx((residual_free_head_b - 96.0) / 100.0)
    (note,) = groundwater_flow_notes(collars, _dup_levels(order), None)
    assert note.startswith("Groundwater flow (3-point, 3 wells, readings 2026-04-02 to 04-03): ")


def test_flow_note_unchanged_without_dates() -> None:
    collars = [
        Collar(hole_id="A", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0),
        Collar(hole_id="B", easting=100.0, northing=0.0, elevation=100.0, total_depth=20.0),
        Collar(hole_id="C", easting=0.0, northing=100.0, elevation=100.0, total_depth=20.0),
    ]
    levels = [WaterLevel(hole_id=h, depth=d) for h, d in (("A", 4.0), ("B", 5.0), ("C", 6.0))]
    (note,) = groundwater_flow_notes(collars, levels, None)
    assert note.startswith("Groundwater flow (3-point, 3 wells): ")


@pytest.mark.parametrize("order", ["old-first", "new-first"])
def test_latest_wins_in_nested_gradients(order: str) -> None:
    old = WaterLevel(hole_id="MW-1D", depth=0.0, elevation_masl=96.0, connect_group="D", date=D(2026, 4, 1))
    new = WaterLevel(hole_id="MW-1D", depth=0.0, elevation_masl=94.0, connect_group="D", date=D(2026, 4, 4))
    pair = [old, new] if order == "old-first" else [new, old]
    levels = [
        WaterLevel(hole_id="MW-1S", depth=0.0, elevation_masl=95.0, connect_group="S", date=D(2026, 4, 2)),
        *pair,
    ]
    screens = [
        ScreenInterval(hole_id="MW-1S", from_depth=5.0, to_depth=8.0),
        ScreenInterval(hole_id="MW-1D", from_depth=20.0, to_depth=23.0),
    ]
    analysis = nested_vertical_gradients(
        levels,
        screens,
        collar_rl_by_hole={"MW-1S": 100.0, "MW-1D": 100.0},
        plan_xy_by_hole={"MW-1S": (0.0, 0.0), "MW-1D": (3.0, 0.0)},
    )
    (gradient,) = analysis.gradients
    assert gradient.head_deep_masl == pytest.approx(94.0)
    note = format_vertical_gradient_note(gradient)
    assert note.endswith("between screen mid-points (readings 2026-04-02 to 04-04)")
    undated = nested_vertical_gradients(
        [lv.model_copy(update={"date": None}) for lv in levels[:2]],
        screens,
        collar_rl_by_hole={"MW-1S": 100.0, "MW-1D": 100.0},
        plan_xy_by_hole={"MW-1S": (0.0, 0.0), "MW-1D": (3.0, 0.0)},
    )
    assert format_vertical_gradient_note(undated.gradients[0]).endswith("between screen mid-points")


def test_duplicate_warning_names_latest_date() -> None:
    collars = [
        Collar(hole_id=h, easting=e, northing=0.0, elevation=100.0, total_depth=20.0)
        for h, e in (("A", 0.0), ("B", 100.0), ("C", 200.0))
    ]
    summary = summarize_water_levels(collars, _dup_levels("new-first"), ["A", "B", "C"])
    dup = [w for w in summary.warnings if "duplicate groundwater reading" in w]
    assert dup == [
        "B (default): duplicate groundwater reading — 2 readings dated 2026-04-01, 2026-04-03; "
        "the latest (2026-04-03) is used on the section, gradients and flow fit "
        "(by date, not row order)"
    ]
    undated = summarize_water_levels(
        collars,
        [WaterLevel(hole_id="B", depth=1.0), WaterLevel(hole_id="B", depth=2.0)],
        ["A", "B", "C"],
    )
    assert any(
        w == "B (default): duplicate groundwater reading — last value is used on the section"
        for w in undated.warnings
    )


def _section(levels, readings=(), **kwargs):
    collars = [
        Collar(hole_id=h, easting=e, northing=0.0, elevation=100.0, total_depth=20.0)
        for h, e in (("A", 0.0), ("B", 100.0), ("C", 200.0))
    ]
    lith = [Lithology(hole_id=c.hole_id, from_depth=0.0, to_depth=20.0, lithology_code="SAND") for c in collars]
    geometry = compute_section_geometry(collars, lith, _POINTS)
    result = render_cross_section_from_geometry(
        geometry,
        _POINTS,
        water_levels=levels,
        environmental_readings=tuple(readings),
        export_formats=frozenset({"svg"}),
        close_figure=False,
        **kwargs,
    )
    bundle = result.retained_export_bundle
    from matplotlib import pyplot as plt

    plt.close(bundle["figure"])
    return result, bundle["renderer"]


@pytest.mark.parametrize("order", ["old-first", "new-first"])
def test_latest_wins_on_plot(order: str) -> None:
    _result, renderer = _section(
        _dup_levels(order), show_water_elevation_labels=True, elevation_mode="absolute"
    )
    texts = [a.get_text() for kind, a, _c in renderer._water_labels if kind == "rl"]
    assert any(t.startswith("95") for t in texts), texts
    assert not any(t.startswith("90") for t in texts), texts


# --- chemistry -----------------------------------------------------------------------


def _benzene(value: float, date: D | None, hole: str = "B") -> EnvironmentalReading:
    return EnvironmentalReading(
        hole_id=hole, parameter="Benzene", value=value, depth=3.0, unit="mg/L", date=date
    )


def test_latest_environmental_readings() -> None:
    readings = [
        _benzene(5.0, D(2026, 3, 1)),
        _benzene(2.0, D(2026, 5, 1)),
        _benzene(9.0, D(2026, 1, 15)),
        _benzene(1.0, None, hole="A"),
        _benzene(1.5, None, hole="A"),
    ]
    kept, notes = latest_environmental_readings(readings)
    assert [r.value for r in kept] == [2.0, 1.0, 1.5]  # undated duplicates unchanged
    assert notes == ("B Benzene at 3 m: 3 sampling dates, latest (2026-05-01) shown",)


def test_chemistry_latest_on_figure_and_validate() -> None:
    readings = [_benzene(5.0, D(2026, 3, 1)), _benzene(2.0, D(2026, 5, 1)), _benzene(1.0, None, hole="A")]
    result, renderer = _section(
        [], readings, environmental_parameters=("Benzene",), show_parameter_labels=True
    )
    assert [r.value for r in renderer.environmental_readings] == [2.0, 1.0]
    note = "B Benzene at 3 m: 2 sampling dates, latest (2026-05-01) shown"
    assert note in result.qa_notes
    collars = [
        Collar(hole_id=h, easting=e, northing=0.0, elevation=100.0, total_depth=20.0)
        for h, e in (("A", 0.0), ("B", 100.0))
    ]
    summary = summarize_environmental_readings(collars, readings, ["A", "B"])
    assert f"Info: {note}" in summary.warnings


# --- legend ---------------------------------------------------------------------------


def test_series_legend_date_rules() -> None:
    dated = [WaterLevel(hole_id="A", depth=1.0, date=D(2026, 4, 2)), WaterLevel(hole_id="B", depth=1.0, date=D(2026, 4, 4))]
    assert series_legend_date(dated) == "2026-04-02 to 04-04"
    assert series_legend_date(dated[:1]) == "2026-04-02"
    labelled = [lv.model_copy(update={"series_label": "Spring"}) for lv in dated]
    assert series_legend_date(labelled) == ""
    wide = [*dated, WaterLevel(hole_id="C", depth=1.0, date=D(2026, 6, 1))]
    assert series_legend_date(wide) == ""
    assert series_legend_date([*dated, WaterLevel(hole_id="C", depth=1.0)]) == ""


def test_legend_shows_date_for_unlabelled_series() -> None:
    levels = [WaterLevel(hole_id=h, depth=d, date=D(2026, 4, 2)) for h, d in (("A", 4.0), ("B", 5.0), ("C", 6.0))]
    _result, renderer = _section(levels, elevation_mode="absolute")
    (entry,) = renderer.water_series_legend
    assert entry["level_label"] == "GROUNDWATER LEVEL (2026-04-02)"
    assert entry["elevation_label"] == "GROUNDWATER ELEVATION masl (2026-04-02)"


def test_no_dates_regression() -> None:
    levels = [WaterLevel(hole_id=h, depth=d) for h, d in (("A", 4.0), ("B", 5.0), ("C", 6.0))]
    result, renderer = _section(levels, elevation_mode="absolute")
    (entry,) = renderer.water_series_legend
    assert entry["level_label"] == "GROUNDWATER LEVEL (masl)"
    assert not any("readings 20" in n for n in result.qa_notes)
    assert not any("sampling dates" in n for n in result.qa_notes)
    # Identical figure with and without an explicit blank date.
    blank = [lv.model_copy(update={"date": None}) for lv in levels]
    collars = [
        Collar(hole_id=h, easting=e, northing=0.0, elevation=100.0, total_depth=20.0)
        for h, e in (("A", 0.0), ("B", 100.0), ("C", 200.0))
    ]
    lith = [Lithology(hole_id=c.hole_id, from_depth=0.0, to_depth=20.0, lithology_code="SAND") for c in collars]
    one = build_cross_section(collars, lith, _POINTS, water_levels=levels, export_formats=frozenset({"svg"}))
    two = build_cross_section(collars, lith, _POINTS, water_levels=blank, export_formats=frozenset({"svg"}))
    assert one.qa_notes == two.qa_notes
