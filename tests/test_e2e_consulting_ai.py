"""Consulting layout, optional sheets, and AI local-path edge cases (CI)."""

from __future__ import annotations

from io import BytesIO

import pandas as pd

from ai_assistant import AIAssistant
from ai_quality import analyze_parsed_data
from models import (
    Collar,
    ConsultingTitleBlock,
    DataParser,
    Lithology,
    ParseResult,
    ScreenInterval,
    VerticalGradient,
    WaterLevel,
    apply_unit_order_fix,
    subset_parse_result,
)
from pipeline import build_cross_section
from ui_helpers import svg_is_valid


def test_consulting_section_nm_notes_and_exports() -> None:
    collars = [
        Collar(hole_id="MW-01", easting=0, northing=0, elevation=665, total_depth=20),
        Collar(hole_id="MW-02", easting=50, northing=0, elevation=664, total_depth=18),
        Collar(hole_id="MW-03", easting=100, northing=0, elevation=663, total_depth=16),
    ]
    lith = [
        Lithology(hole_id="MW-01", from_depth=0, to_depth=8, lithology_code="Sand"),
        Lithology(hole_id="MW-01", from_depth=8, to_depth=20, lithology_code="Clay"),
        Lithology(hole_id="MW-02", from_depth=0, to_depth=7, lithology_code="Sand"),
        Lithology(hole_id="MW-02", from_depth=7, to_depth=18, lithology_code="Clay"),
        Lithology(hole_id="MW-03", from_depth=0, to_depth=6, lithology_code="Sand"),
        Lithology(hole_id="MW-03", from_depth=6, to_depth=16, lithology_code="Clay"),
    ]
    water = [
        WaterLevel(hole_id="MW-01", depth=2.5),
        WaterLevel(hole_id="MW-02", depth=3.0),
    ]
    result = build_cross_section(
        collars,
        lith,
        [(0, 0), (100, 0)],
        render_layout="consulting_section",
        show_legend=False,
        show_hatches=True,
        water_levels=water,
        consulting_title_block=ConsultingTitleBlock(
            section_label="B-B'",
            map_scale="1:1000",
            drawn_by="EC",
            prepared_for="CLIENT",
            notes=("NOTE ONE", "NOTE TWO"),
        ),
        screen_intervals=[ScreenInterval(hole_id="MW-01", from_depth=5, to_depth=10)],
        vertical_gradients=[VerticalGradient(hole_id="MW-01", direction="up")],
        export_formats=frozenset({"svg", "png"}),
    )
    text = result.svg_bytes.decode("utf-8", errors="ignore")
    assert svg_is_valid(result.svg_bytes)
    assert len(result.png_bytes) > 100
    assert "NM" in text
    assert "NOTES" in text
    assert "DRAWN BY" in text
    assert "Sand" in result.lithology_codes or "Clay" in result.lithology_codes


def test_screens_gradients_ingest_and_subset() -> None:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                {
                    "hole_id": "MW-01",
                    "easting": 0,
                    "northing": 0,
                    "elevation": 100,
                    "total_depth": 20,
                },
                {
                    "hole_id": "MW-02",
                    "easting": 50,
                    "northing": 0,
                    "elevation": 100,
                    "total_depth": 18,
                },
            ]
        ).to_excel(writer, sheet_name="Collars", index=False)
        pd.DataFrame(
            [
                {
                    "hole_id": "MW-01",
                    "from_depth": 0,
                    "to_depth": 10,
                    "lithology_code": "Sand",
                },
                {
                    "hole_id": "MW-02",
                    "from_depth": 0,
                    "to_depth": 10,
                    "lithology_code": "Clay",
                },
            ]
        ).to_excel(writer, sheet_name="Lithology", index=False)
        pd.DataFrame([{"hole_id": "MW-01", "from_depth": 4, "to_depth": 8}]).to_excel(
            writer, sheet_name="Screens", index=False
        )
        pd.DataFrame([{"hole_id": "MW-02", "direction": "down"}]).to_excel(
            writer, sheet_name="Gradients", index=False
        )
    parsed = DataParser().parse_file(BytesIO(buffer.getvalue()))
    assert len(parsed.screen_intervals) == 1
    assert parsed.vertical_gradients[0].direction == "down"
    subset = subset_parse_result(parsed, ["MW-01"])
    assert len(subset.vertical_gradients) == 0
    assert len(subset.screen_intervals) == 1


def test_ai_local_paths_and_unit_order_fix() -> None:
    assistant = AIAssistant(None)
    assert assistant.suggest_fix_plan(()) == ()
    assert assistant.parse_transect_request("through XX-99", ["MW-01", "MW-02"]) is None
    parsed = assistant.parse_transect_request("B-B' MW-01 MW-02", ["MW-01", "MW-02"])
    assert parsed is not None
    assert parsed.hole_ids == ("MW-01", "MW-02")
    assert (
        assistant.suggest_lithology_mappings(["Fat Clay"])[0].canonical_code == "Clay"
    )

    collars = [
        Collar(hole_id="MW-01", easting=0, northing=0, elevation=100, total_depth=10),
    ]
    dup_lith = [
        Lithology(hole_id="MW-01", from_depth=0, to_depth=5, lithology_code="Clay"),
        Lithology(hole_id="MW-01", from_depth=5, to_depth=10, lithology_code="Clay"),
    ]
    qr = analyze_parsed_data(collars, dup_lith)
    assert any(issue.code == "duplicate_lithology_no_unit_order" for issue in qr.issues)
    fixed = apply_unit_order_fix(
        ParseResult(collars=tuple(collars), lithologies=tuple(dup_lith), errors=())
    )
    assert all(item.unit_order is not None for item in fixed.lithologies)
