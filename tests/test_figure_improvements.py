"""Regression tests for cross-section figure improvement plan."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from gwm_reference import GWM_TRANSECTS, build_subset
from pipeline import build_cross_section
from ui_helpers import svg_is_valid

ROOT = Path(__file__).resolve().parents[1]


def _svg_label_x_positions(svg: str, *labels: str) -> dict[str, float]:
    """Return approximate SVG x coordinates for matplotlib path-text labels."""
    positions: dict[str, float] = {}
    for label in labels:
        marker = f"<!-- {label} -->"
        idx = svg.find(marker)
        if idx == -1:
            continue
        chunk = svg[idx : idx + 400]
        match = re.search(r'transform="translate\(([\d.]+)', chunk)
        if match:
            positions[label] = float(match.group(1))
    return positions


@pytest.mark.parametrize(
    ("transect_id", "start_label", "end_label"),
    (
        ("A_A", "NORTHWEST", "SOUTHEAST"),
        ("D_D", "NORTHWEST", "SOUTHEAST"),
        ("B_B", "NORTH", "SOUTH"),
        ("C_C", "NORTH", "SOUTH"),
    ),
)
def test_consulting_transect_end_labels_left_to_right(
    transect_id: str, start_label: str, end_label: str
) -> None:
    spec, subset = build_subset(transect_id)
    transect_points = [(collar.easting, collar.northing) for collar in subset.collars]
    result = build_cross_section(
        subset.collars,
        subset.lithologies,
        transect_points,
        render_layout="consulting_section",
        vertical_exaggeration=spec.vertical_exaggeration,
        show_legend=False,
        water_levels=subset.water_levels,
        consulting_title_block=spec.title_block,
        screen_intervals=subset.screen_intervals,
        export_formats=frozenset({"svg"}),
    )
    text = result.svg_bytes.decode("utf-8", errors="ignore")
    assert start_label in text
    assert end_label in text
    positions = _svg_label_x_positions(text, start_label, end_label)
    assert start_label in positions, f"missing SVG x for {start_label}"
    assert end_label in positions, f"missing SVG x for {end_label}"
    assert positions[start_label] < positions[end_label], (
        f"{start_label} should be left of {end_label} "
        f"({positions[start_label]:.1f} vs {positions[end_label]:.1f})"
    )


def test_gwm_transect_profile_eastings_in_fixtures() -> None:
    for transect_id in GWM_TRANSECTS:
        spec, subset = build_subset(transect_id)
        for collar, expected_easting in zip(subset.collars, spec.profile_eastings, strict=True):
            assert collar.easting == expected_easting


def test_section_build_request_fail_on_overlaps_in_cache_key() -> None:
    from section_build_request import SectionBuildRequest

    base = SectionBuildRequest(transect_points=((0.0, 0.0), (50.0, 0.0)))
    blocked = base.model_copy(update={"fail_on_overlaps": True})
    assert base.cache_key(("BH-01", "BH-02")) != blocked.cache_key(("BH-01", "BH-02"))


def test_ops_audit_writes_json_line(tmp_path, monkeypatch) -> None:
    from ops_audit import audit_event

    # Audit path must resolve under the writable user data directory.
    monkeypatch.setattr("ops_audit.user_data_dir", lambda: tmp_path)
    monkeypatch.setattr("paths.user_data_dir", lambda: tmp_path)
    monkeypatch.setenv("CROSS_SECTION_AUDIT_LOG", "audit.log")
    audit_event("test_event", detail="ok")
    log_path = tmp_path / "audit.log"
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert '"event": "test_event"' in lines[0]


def test_gwm_c_c_transect_hole_membership_matches_reference() -> None:
    spec = GWM_TRANSECTS["C_C"]
    assert spec.hole_ids == (
        "BH18-07",
        "MW18-19",
        "BH18-04",
        "BH18-05",
        "MW18-20",
        "MW18-17",
    )
    assert spec.profile_eastings == (0.0, 245.0, 260.0, 360.0, 410.0, 570.0)


def test_consulting_pinch_out_legend_when_pinch_present() -> None:
    spec, subset = build_subset("A_A")
    transect_points = [(collar.easting, collar.northing) for collar in subset.collars]
    result = build_cross_section(
        subset.collars,
        subset.lithologies,
        transect_points,
        render_layout="consulting_section",
        vertical_exaggeration=spec.vertical_exaggeration,
        show_legend=False,
        allow_pinch_outs=True,
        water_levels=subset.water_levels,
        consulting_title_block=spec.title_block,
        screen_intervals=subset.screen_intervals,
        export_formats=frozenset({"svg"}),
    )
    text = result.svg_bytes.decode("utf-8", errors="ignore")
    assert any(p.is_pinch_out for p in result.polygons)


def test_consulting_layout_exports_single_page_pdf() -> None:
    spec, subset = build_subset("D_D")
    transect_points = [(collar.easting, collar.northing) for collar in subset.collars]
    result = build_cross_section(
        subset.collars,
        subset.lithologies,
        transect_points,
        render_layout="consulting_section",
        vertical_exaggeration=spec.vertical_exaggeration,
        consulting_title_block=spec.title_block,
        screen_intervals=subset.screen_intervals,
        export_formats=frozenset({"pdf"}),
    )
    assert result.pdf_bytes.startswith(b"%PDF")
    assert len(result.pdf_bytes) > 500


def test_fail_on_overlaps_blocks_export(monkeypatch: pytest.MonkeyPatch) -> None:
    from models import Collar, Lithology
    from stratigraphy import PolygonOverlap

    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=12.0),
        Collar(hole_id="BH-02", easting=30.0, northing=0.0, elevation=100.0, total_depth=12.0),
        Collar(hole_id="BH-03", easting=60.0, northing=0.0, elevation=100.0, total_depth=12.0),
    ]
    lithologies = []
    for hole_id in ("BH-01", "BH-02", "BH-03"):
        lithologies.extend(
            [
                Lithology(hole_id=hole_id, from_depth=0.0, to_depth=4.0, lithology_code="Sand"),
                Lithology(hole_id=hole_id, from_depth=4.0, to_depth=8.0, lithology_code="Clay"),
                Lithology(hole_id=hole_id, from_depth=8.0, to_depth=12.0, lithology_code="Silt"),
            ]
        )
    fake_overlap = PolygonOverlap(
        left_lithology_code="Clay",
        right_lithology_code="Silt",
        hole_pair=("BH-01", "BH-02"),
        centroid_x=25.0,
        centroid_y=95.0,
    )

    def _fake_detect(_polygons):
        return (fake_overlap,)

    monkeypatch.setattr("pipeline.detect_polygon_overlaps", _fake_detect)

    with pytest.raises(ValueError, match="Polygon overlap"):
        build_cross_section(
            collars,
            lithologies,
            [(0.0, 0.0), (60.0, 0.0)],
            render_layout="consulting_section",
            fail_on_overlaps=True,
            export_formats=frozenset({"svg"}),
        )


def test_preflight_polygon_overlap_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    from app_services import preflight_polygon_overlap_warnings
    from models import Collar, Lithology, ParseResult
    from stratigraphy import PolygonOverlap

    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=12.0),
        Collar(hole_id="BH-02", easting=30.0, northing=0.0, elevation=100.0, total_depth=12.0),
        Collar(hole_id="BH-03", easting=60.0, northing=0.0, elevation=100.0, total_depth=12.0),
    ]
    lithologies = []
    for hole_id in ("BH-01", "BH-02", "BH-03"):
        lithologies.extend(
            [
                Lithology(hole_id=hole_id, from_depth=0.0, to_depth=4.0, lithology_code="Sand"),
                Lithology(hole_id=hole_id, from_depth=4.0, to_depth=8.0, lithology_code="Clay"),
                Lithology(hole_id=hole_id, from_depth=8.0, to_depth=12.0, lithology_code="Silt"),
            ]
        )
    subset = ParseResult(collars=tuple(collars), lithologies=tuple(lithologies), errors=())
    fake_overlap = PolygonOverlap(
        left_lithology_code="Clay",
        right_lithology_code="Silt",
        hole_pair=("BH-01", "BH-02"),
        centroid_x=25.0,
        centroid_y=95.0,
    )

    def _fake_detect(_polygons):
        return (fake_overlap,)

    monkeypatch.setattr("app_services.detect_polygon_overlaps", _fake_detect)

    warnings = preflight_polygon_overlap_warnings(
        subset,
        ((0.0, 0.0), (60.0, 0.0)),
        interpretation_mode="interpolated",
        allow_pinch_outs=True,
        correlation_overrides=(),
    )
    assert len(warnings) == 1
    assert "Polygon overlap" in warnings[0]


def test_gwm_dual_gw_series_uses_june_2024() -> None:
    spec, subset = build_subset("A_A")
    series_ids = {level.series_id for level in subset.water_levels}
    assert "2024-05" in series_ids
    assert "2024-06" in series_ids
    transect_points = [(collar.easting, collar.northing) for collar in subset.collars]
    result = build_cross_section(
        subset.collars,
        subset.lithologies,
        transect_points,
        render_layout="consulting_section",
        vertical_exaggeration=spec.vertical_exaggeration,
        water_levels=subset.water_levels,
        consulting_title_block=spec.title_block,
        screen_intervals=subset.screen_intervals,
        export_formats=frozenset({"svg"}),
    )
    assert svg_is_valid(result.svg_bytes)
    text = result.svg_bytes.decode("utf-8", errors="ignore").upper()
    assert "MAY 2024" in text
    assert "JUNE 2024" in text
