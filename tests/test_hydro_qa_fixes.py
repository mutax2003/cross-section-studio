"""Regression tests for the hydrogeology QA fixes.

1. Latest reading chosen over ALL statuses before dry / NM holes are skipped
   (gradients, nested iv and the flow fit agree with the figure's DRY / NM).
2. LNAPL correction / water-line and vertical-gradient notes reach the PDF
   QA lines (Generate, Prepare from the retained figure, batch export).
3. Consulting vertical-gradient arrow anchors use the latest reading.
4. Assistant section facts: corrected depths, latest reading, dry / NM excluded.
5. Validate checks run on the readings used, not superseded ones.
6. Lab qualifiers (J, UJ, B, E, R, NDL ...) no longer lose the row.
Plus: numpy date serials, case-insensitive chemistry notes, product_density
default kept in ``model_fields_set``, Validate gradients on along-section Δx.
"""

from __future__ import annotations

import datetime
import math
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from ai_quality import summarize_environmental_readings, summarize_water_levels
from chemistry_qc import classify_lab_qualifier, lab_qualifier_warnings
from hydro_metrics import (
    fit_head_planes,
    horizontal_gradients_along_profile,
    latest_measured_water_by_hole,
    latest_water_readings,
    nested_vertical_gradients,
)
from models import (
    DEFAULT_LNAPL_SPECIFIC_GRAVITY,
    Collar,
    DeviationReading,
    EnvironmentalReading,
    Lithology,
    ParseResult,
    ScreenInterval,
    WaterLevel,
    parse_monitoring_date,
)
from pipeline import (
    _latest_chemistry_samples,
    compute_section_geometry,
    render_cross_section_from_geometry,
)
from render_theme import primary_water_level_by_hole

D1 = datetime.date(2026, 4, 1)
D2 = datetime.date(2026, 5, 1)


# ------------------------------------------------- 1. latest reading vs DRY / NM


def _three_holes() -> dict[str, float]:
    return {"A": 0.0, "B": 50.0, "C": 100.0}


def test_latest_measured_skips_hole_whose_latest_reading_is_dry() -> None:
    levels = [
        WaterLevel(hole_id="B", depth=3.0, date=D1),
        WaterLevel(hole_id="B", depth=0.0, status="dry", date=D2),
        WaterLevel(hole_id="A", depth=2.0, date=D2),
    ]
    assert set(latest_measured_water_by_hole(levels)) == {"A"}
    # Row order does not matter (latest date wins first).
    assert set(latest_measured_water_by_hole(levels[::-1])) == {"A"}


def test_horizontal_gradient_does_not_fall_back_to_older_measured_head() -> None:
    levels = [
        WaterLevel(hole_id="A", depth=2.0, date=D2),
        WaterLevel(hole_id="B", depth=9.0, date=D1),  # superseded measured head
        WaterLevel(hole_id="B", depth=0.0, status="nm", date=D2),  # plotted: NM
        WaterLevel(hole_id="C", depth=3.0, date=D2),
    ]
    segments = horizontal_gradients_along_profile(
        levels,
        hole_order=["A", "B", "C"],
        x_by_hole=_three_holes(),
        collar_rl_by_hole={"A": 100.0, "B": 100.0, "C": 100.0},
    )
    assert [(s.left_hole_id, s.right_hole_id) for s in segments] == [("A", "C")]


def test_nested_gradient_skips_nest_whose_latest_reading_is_dry() -> None:
    screens = [
        ScreenInterval(hole_id="S", from_depth=4.0, to_depth=8.0),
        ScreenInterval(hole_id="D", from_depth=14.0, to_depth=18.0),
    ]
    levels = [
        WaterLevel(hole_id="S", depth=5.0, date=D1),
        WaterLevel(hole_id="S", depth=0.0, status="dry", date=D2),
        WaterLevel(hole_id="D", depth=5.5, date=D2),
    ]
    analysis = nested_vertical_gradients(
        levels,
        screens,
        collar_rl_by_hole={"S": 100.0, "D": 100.0},
        plan_xy_by_hole={"S": (0.0, 0.0), "D": (1.0, 0.0)},
    )
    assert analysis.gradients == ()


def test_flow_fit_drops_well_whose_latest_reading_is_dry() -> None:
    collars = [
        Collar(hole_id="A", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0),
        Collar(hole_id="B", easting=100.0, northing=0.0, elevation=100.0, total_depth=20.0),
        Collar(hole_id="C", easting=0.0, northing=100.0, elevation=100.0, total_depth=20.0),
    ]
    base = [
        WaterLevel(hole_id="A", depth=5.0, date=D2),
        WaterLevel(hole_id="B", depth=5.3, date=D2),
        WaterLevel(hole_id="C", depth=5.0, date=D1),
    ]
    assert len(fit_head_planes(collars, base)) == 1
    dry_latest = [*base, WaterLevel(hole_id="C", depth=0.0, status="dry", date=D2)]
    assert fit_head_planes(collars, dry_latest) == ()


# ------------------------------------------------- 2. notes reach the PDF


def _pdf_text(payload: bytes) -> str:
    from pypdf import PdfReader

    return "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(payload)).pages)


def _collars() -> list[Collar]:
    return [
        Collar(hole_id="MW-1", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0),
        Collar(hole_id="MW-3", easting=100.0, northing=0.0, elevation=100.5, total_depth=20.0),
        Collar(hole_id="MW-4", easting=200.0, northing=0.0, elevation=101.0, total_depth=20.0),
    ]


def _lithologies() -> list[Lithology]:
    return [
        Lithology(hole_id=hole, from_depth=0.0, to_depth=20.0, lithology_code="SAND")
        for hole in ("MW-1", "MW-3", "MW-4")
    ]


def _lnapl_nested_water() -> list[WaterLevel]:
    return [
        WaterLevel(hole_id="MW-1", depth=4.0),
        WaterLevel(hole_id="MW-3", depth=5.0, product_thickness_m=0.32, connect_group="shallow"),
        WaterLevel(hole_id="MW-3", depth=5.6, connect_group="deep"),
        WaterLevel(hole_id="MW-4", depth=5.5),
    ]


def _screens() -> list[ScreenInterval]:
    return [
        ScreenInterval(hole_id="MW-3", from_depth=4.0, to_depth=7.0),
        ScreenInterval(hole_id="MW-3", from_depth=14.0, to_depth=17.0),
    ]


POINTS = [(-20.0, 0.0), (220.0, 0.0)]


def _render(formats: frozenset[str], *, close_figure: bool = True):
    geometry = compute_section_geometry(
        _collars(), _lithologies(), POINTS, interpretation_mode="borehole_only"
    )
    return render_cross_section_from_geometry(
        geometry,
        POINTS,
        interpretation_mode="borehole_only",
        water_levels=_lnapl_nested_water(),
        screen_intervals=_screens(),
        render_layout="consulting_section",
        export_formats=formats,
        close_figure=close_figure,
    )


def test_generate_pdf_qa_lines_include_lnapl_and_vertical_gradient_notes() -> None:
    pytest.importorskip("pypdf")
    result = _render(frozenset({"pdf"}))
    text = " ".join(_pdf_text(result.pdf_bytes).split())
    assert "corrected for LNAPL" in text
    assert "Vertical gradient" in text


def test_prepare_from_retained_figure_pdf_includes_notes() -> None:
    pytest.importorskip("pypdf")
    from app_services import _try_export_from_retained_figure

    result = _render(frozenset({"svg"}), close_figure=False)
    retained = result.retained_export_bundle
    assert retained is not None
    exported = _try_export_from_retained_figure(retained, frozenset({"png", "pdf"}))
    assert exported is not None
    text = " ".join(_pdf_text(exported[2]).split())
    assert "corrected for LNAPL" in text
    assert "Vertical gradient" in text


def test_batch_export_pdf_includes_notes() -> None:
    pytest.importorskip("pypdf")
    from batch_export import (
        BatchTransectSpec,
        build_one_transect_exports,
        clear_batch_geometry_memo,
    )
    from section_build_request import SectionBuildRequest

    clear_batch_geometry_memo()
    parse_result = ParseResult(
        collars=tuple(_collars()),
        lithologies=tuple(_lithologies()),
        errors=(),
        water_levels=tuple(_lnapl_nested_water()),
        screen_intervals=tuple(_screens()),
    )
    base = SectionBuildRequest(
        transect_points=tuple(POINTS),
        section_title="Site",
        interpretation_mode="borehole_only",
        render_layout="consulting_section",
    )
    spec = BatchTransectSpec(label="A-A", hole_ids=("MW-1", "MW-3", "MW-4"))
    _label, _svg, _png, pdf = build_one_transect_exports(
        parse_result, base, spec, export_formats=frozenset({"pdf"})
    )
    text = " ".join(_pdf_text(pdf).split())
    assert "corrected for LNAPL" in text
    assert "Vertical gradient" in text


# ------------------------------------------------- 3. arrow anchors


def test_primary_water_level_uses_latest_date_not_last_row() -> None:
    levels = (
        WaterLevel(hole_id="BH-01", depth=2.0, series_id="s", date=D2),
        WaterLevel(hole_id="BH-01", depth=7.0, series_id="s", date=D1),  # older, last row
    )
    assert primary_water_level_by_hole(levels)["BH-01"].depth == 2.0


def test_primary_water_level_has_no_anchor_when_latest_is_dry() -> None:
    levels = (
        WaterLevel(hole_id="BH-01", depth=2.0, series_id="s", date=D1),
        WaterLevel(hole_id="BH-01", depth=0.0, series_id="s", status="dry", date=D2),
        WaterLevel(hole_id="BH-02", depth=3.0, series_id="s", date=D2),
    )
    assert set(primary_water_level_by_hole(levels)) == {"BH-02"}


# ------------------------------------------------- 4. assistant facts


def test_assistant_water_facts_use_corrected_latest_depths() -> None:
    from app_common import _water_and_nm

    collars = (
        Collar(hole_id="A", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0,
               stick_up_m=0.8),
        Collar(hole_id="B", easting=50.0, northing=0.0, elevation=100.0, total_depth=20.0),
        Collar(hole_id="C", easting=100.0, northing=0.0, elevation=100.0, total_depth=30.0),
        Collar(hole_id="D", easting=150.0, northing=0.0, elevation=100.0, total_depth=20.0),
    )
    levels = (
        WaterLevel(hole_id="A", depth=3.0, depth_reference="btoc"),  # 2.2 m bgs
        WaterLevel(hole_id="B", depth=4.0, date=D2),
        WaterLevel(hole_id="B", depth=9.0, date=D1),  # older, last row
        WaterLevel(hole_id="C", depth=10.0),  # inclined hole: TVD < 10 m
        WaterLevel(hole_id="D", depth=2.0, date=D1),
        WaterLevel(hole_id="D", depth=0.0, status="dry", date=D2),  # latest: dry
    )
    deviations = (
        DeviationReading(hole_id="C", depth=0.0, inclination_deg=60.0, azimuth_deg=90.0),
        DeviationReading(hole_id="C", depth=30.0, inclination_deg=60.0, azimuth_deg=90.0),
    )
    parse_result = ParseResult(
        collars=collars,
        lithologies=(),
        errors=(),
        water_levels=levels,
        deviation_readings=deviations,
    )
    water, nm = _water_and_nm(parse_result, ["A", "B", "C", "D"])
    assert water["A"]["default"] == pytest.approx(2.2)
    assert water["B"]["default"] == pytest.approx(4.0)
    # Inclined hole: true vertical depth from the survey (as the figure), not 10 m.
    from hydro_metrics import water_depth_below_ground_m
    from projection import build_along_hole_tvd

    table = build_along_hole_tvd(collars, deviations)["C"]
    expected = water_depth_below_ground_m(
        levels[3], 100.0, md_to_elevation=lambda md: table.elevation_at(100.0, md)
    )
    assert expected < 9.0
    assert water["C"]["default"] == pytest.approx(expected, abs=1e-3)
    assert "D" not in water
    assert nm == ["D"]


# ------------------------------------------------- 5. Validate on used readings


def _validate_collars() -> list[Collar]:
    return [
        Collar(hole_id="MW-1", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0),
        Collar(hole_id="MW-2", easting=50.0, northing=0.0, elevation=100.0, total_depth=20.0),
    ]


def test_superseded_reading_does_not_trigger_synoptic_or_artesian_or_lnapl() -> None:
    levels = [
        # Superseded (older) readings at MW-1: artesian head + product.
        WaterLevel(hole_id="MW-1", depth=0.0, elevation_masl=100.5,
                   date=datetime.date(2025, 1, 1), product_thickness_m=0.3),
        WaterLevel(hole_id="MW-1", depth=3.0, date=D2),
        WaterLevel(hole_id="MW-2", depth=3.2, date=D2),
    ]
    summary = summarize_water_levels(_validate_collars(), levels, ["MW-1", "MW-2"])
    text = " ".join((*summary.warnings, *summary.notes))
    assert "potentiometric surface" not in text
    assert "LNAPL" not in text
    assert "artesian" not in text.lower()
    # The duplicate is still reported.
    assert any("duplicate groundwater reading" in w for w in summary.warnings)


def test_synoptic_warning_still_fires_for_readings_actually_used() -> None:
    levels = [
        WaterLevel(hole_id="MW-1", depth=3.0, date=datetime.date(2025, 1, 1)),
        WaterLevel(hole_id="MW-2", depth=3.2, date=D2),
    ]
    summary = summarize_water_levels(_validate_collars(), levels, ["MW-1", "MW-2"])
    assert any("potentiometric surface" in w for w in summary.warnings)


def test_latest_water_readings_keeps_one_per_hole_series_nest() -> None:
    levels = [
        WaterLevel(hole_id="A", depth=1.0, date=D1),
        WaterLevel(hole_id="A", depth=2.0, date=D2),
        WaterLevel(hole_id="A", depth=3.0, connect_group="deep"),
        WaterLevel(hole_id="A", depth=4.0, series_id="other"),
    ]
    assert sorted(level.depth for level in latest_water_readings(levels)) == [2.0, 3.0, 4.0]


def test_rdl_note_only_for_sample_actually_plotted() -> None:
    collars = [Collar(hole_id="BH-1", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0)]
    readings = [
        # Old non-detect with a high limit, superseded by a later detect.
        EnvironmentalReading(hole_id="BH-1", parameter="Benzene", value=50.0, depth=2.0,
                             detected=False, green_max=5.0, date=D1),
        EnvironmentalReading(hole_id="BH-1", parameter="Benzene", value=1.0, depth=2.0,
                             green_max=5.0, date=D2),
    ]
    summary = summarize_environmental_readings(collars, readings, ["BH-1"])
    assert not any("detection limit" in w for w in summary.warnings)


def test_series_legend_ignores_superseded_dates() -> None:
    from hydro_metrics import series_legend_date

    levels = [
        WaterLevel(hole_id="A", depth=1.0, date=datetime.date(2025, 1, 1)),
        WaterLevel(hole_id="A", depth=2.0, date=D2),
        WaterLevel(hole_id="B", depth=2.0, date=D2),
    ]
    assert series_legend_date(levels) == ""
    assert series_legend_date(latest_water_readings(levels)) == D2.isoformat()


# ------------------------------------------------- 6. lab qualifiers


@pytest.mark.parametrize(
    ("flag", "kind", "rejected"),
    [
        ("U", "non_detect", False),
        ("UJ", "non_detect", False),
        ("ub", "non_detect", False),
        ("<", "non_detect", False),
        ("ND", "non_detect", False),
        ("NDL", "non_detect", False),
        ("BDL", "non_detect", False),
        ("not detected", "non_detect", False),
        ("J", "detected", False),
        ("J+", "detected", False),
        ("B", "detected", False),
        ("E", "detected", False),
        ("JB", "detected", False),
        ("NJ", "detected", False),
        ("*", "detected", False),
        ("R", "detected", True),
        ("J, R", "detected", True),
        ("UR", "non_detect", True),
        ("maybe", "unknown", False),
        ("", None, False),
    ],
)
def test_classify_lab_qualifier(flag: str, kind: str | None, rejected: bool) -> None:
    assert classify_lab_qualifier(flag) == (kind, rejected)


def _env_workbook(rows: list[dict]) -> BytesIO:
    collars = pd.DataFrame(
        [{"hole_id": "BH-01", "easting": 0.0, "northing": 0.0, "elevation": 100.0,
          "total_depth": 10.0},
         {"hole_id": "BH-02", "easting": 30.0, "northing": 0.0, "elevation": 100.0,
          "total_depth": 10.0}]
    )
    lith = pd.DataFrame(
        [{"hole_id": h, "from_depth": 0.0, "to_depth": 10.0, "lithology_code": "Clay"}
         for h in ("BH-01", "BH-02")]
    )
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        collars.to_excel(writer, sheet_name="Collars", index=False)
        lith.to_excel(writer, sheet_name="Lithology", index=False)
        pd.DataFrame(rows).to_excel(writer, sheet_name="Environmental", index=False)
    buffer.seek(0)
    return buffer


def _env_row(value, *, depth: float, qualifier=None) -> dict:
    return {"hole_id": "BH-01", "parameter": "Benzene", "depth": depth, "value": value,
            "unit": "µg/L", "qualifier": qualifier}


def test_lab_qualifiers_keep_rows_and_validate_warns_on_r_and_unknown() -> None:
    from ingestion import ingest_workbook

    result, _report = ingest_workbook(
        _env_workbook(
            [
                _env_row(5.0, depth=1.0, qualifier="J"),
                _env_row(0.5, depth=2.0, qualifier="UJ"),
                _env_row(3.0, depth=3.0, qualifier="B"),
                _env_row(7.0, depth=4.0, qualifier="R"),
                _env_row(2.0, depth=5.0, qualifier="weird"),
                _env_row("NDL 5", depth=6.0),
            ]
        )
    )
    assert not [e for e in result.errors if e.startswith("Environmental")], result.errors
    got = [(r.depth, r.value, r.detected) for r in result.environmental_readings]
    assert got == [
        (1.0, 5.0, True),
        (2.0, 0.5, False),
        (3.0, 3.0, True),
        (4.0, 7.0, True),
        (5.0, 2.0, True),
        (6.0, 5.0, False),
    ]
    warnings = lab_qualifier_warnings(result.environmental_readings)
    assert any("R-qualified (rejected) result plotted" in w and "at 4 m" in w for w in warnings)
    assert any("qualifier 'weird' not recognised" in w for w in warnings)
    assert len(warnings) == 2
    summary = summarize_environmental_readings(
        result.collars, result.environmental_readings, ["BH-01", "BH-02"]
    )
    assert any("R-qualified (rejected) result plotted" in w for w in summary.warnings)


# ------------------------------------------------- small items


@pytest.mark.parametrize("serial", [np.int64(46113), np.float64(46113.0), np.int32(46113)])
def test_parse_monitoring_date_accepts_numpy_serials(serial) -> None:
    assert parse_monitoring_date(serial) == datetime.date(2026, 4, 1)


def test_parse_monitoring_date_numpy_nan_and_datetime64() -> None:
    assert parse_monitoring_date(np.float64("nan")) is None
    assert parse_monitoring_date(np.datetime64("2026-04-02")) == datetime.date(2026, 4, 2)
    assert parse_monitoring_date(np.datetime64("NaT")) is None


def test_chemistry_date_notes_match_parameters_case_insensitively() -> None:
    readings = [
        EnvironmentalReading(hole_id="BH-1", parameter="benzene", value=1.0, depth=2.0, date=D1),
        EnvironmentalReading(hole_id="BH-1", parameter="Benzene", value=2.0, depth=2.0, date=D2),
    ]
    _kept, notes = _latest_chemistry_samples(readings, ["BH-1"], ["BENZENE"])
    assert len(notes) == 1 and "2 sampling dates" in notes[0]


def test_product_density_default_is_a_set_field() -> None:
    level = WaterLevel(hole_id="MW-1", depth=5.0, product_thickness_m=0.3)
    assert level.product_density == DEFAULT_LNAPL_SPECIFIC_GRAVITY
    assert "product_density" in level.model_fields_set
    dumped = level.model_dump(exclude_unset=True)
    assert dumped["product_density"] == DEFAULT_LNAPL_SPECIFIC_GRAVITY
    assert WaterLevel.model_validate(dumped).product_density == DEFAULT_LNAPL_SPECIFIC_GRAVITY
    # Explicit density and no product are unchanged.
    assert WaterLevel(hole_id="A", depth=1.0, product_thickness_m=0.3,
                      product_density=0.9).product_density == 0.9
    assert WaterLevel(hole_id="A", depth=1.0).product_density is None


# ------------------------------------------------- Validate Δx = figure Δx


def test_validate_horizontal_gradient_uses_along_section_distance() -> None:
    # MW-2 sits 40 m off a line along the x axis: plan distance 64 m
    # (|i| 0.109), along-section distance 50 m (|i| 0.140, as drawn).
    collars = [
        Collar(hole_id="MW-1", easting=0.0, northing=0.0, elevation=100.0, total_depth=30.0),
        Collar(hole_id="MW-2", easting=50.0, northing=40.0, elevation=100.0, total_depth=30.0),
    ]
    levels = [
        WaterLevel(hole_id="MW-1", depth=2.0),
        WaterLevel(hole_id="MW-2", depth=9.0),  # Δh = 7 m
    ]
    along = summarize_water_levels(
        collars, levels, ["MW-1", "MW-2"], transect_points=[(0.0, 0.0), (100.0, 0.0)]
    )
    gradient_warnings = [w for w in along.warnings if "Horizontal hydraulic gradient" in w]
    assert len(gradient_warnings) == 1
    assert "|i|=0.140" in gradient_warnings[0]
    assert "Δx=50.0 m" in gradient_warnings[0]
    assert not any("plan distance between consecutive collars" in w for w in along.warnings)

    plan = summarize_water_levels(collars, levels, ["MW-1", "MW-2"])
    assert any("Δx=64.0 m" in w for w in plan.warnings)
    assert any("plan distance between consecutive collars" in w for w in plan.warnings)


def test_lab_flag_n_is_detected_in_qualifier_column_but_no_in_detected_column() -> None:
    from chemistry_qc import classify_lab_qualifier
    from parsing import _detect_flag

    assert classify_lab_qualifier("N") == ("detected", False)
    assert _detect_flag("N", "qualifier") is True
    assert _detect_flag("N", "detected") is False
    assert _detect_flag("no", "qualifier") is False


def test_mining_style_dip_on_deviation_rows_is_converted() -> None:
    from models import Collar, DeviationReading
    from projection import build_along_hole_tvd

    collar = Collar(
        hole_id="M", easting=0.0, northing=0.0, elevation=100.0, total_depth=60.0,
        inclination_deg=-90.0, azimuth_deg=0.0,
    )
    rows = [DeviationReading(hole_id="M", depth=d, inclination_deg=-90.0, azimuth_deg=0.0)
            for d in (0.0, 50.0)]
    tvd = build_along_hole_tvd([collar], rows).get("M")
    assert tvd is None or abs(tvd.tvd_at(50.0) - 50.0) < 1e-6

    dipping = [DeviationReading(hole_id="N", depth=d, inclination_deg=-60.0, azimuth_deg=90.0)
               for d in (0.0, 50.0)]
    collar_n = collar.model_copy(update={"hole_id": "N", "inclination_deg": -60.0, "azimuth_deg": 90.0})
    got = build_along_hole_tvd([collar_n], dipping)["N"].tvd_at(40.0)
    assert abs(got - 40.0 * math.cos(math.radians(30.0))) < 1e-6


def test_validate_table_cells_escape_pipes() -> None:
    from app_validate import _md_cell, _md_code_cell

    assert _md_cell("BH|1") == "BH\\|1"
    assert _md_code_cell("a`b|c") == "`a'b\\|c`"


def test_manifest_with_bom_still_parses(tmp_path, monkeypatch) -> None:
    import json

    from app_version import fetch_release_manifest

    monkeypatch.setenv("CROSS_SECTION_ALLOW_DEV_UPDATE", "1")
    path = tmp_path / "release-manifest.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"version": "9.9.9"}).encode("utf-8"))
    assert fetch_release_manifest(path.as_uri())["version"] == "9.9.9"
