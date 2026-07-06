"""Extra end-to-end edge-case checks (consulting layout, AI, ingest)."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

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

checks: list[tuple[str, bool, str]] = []


def ok(name: str, cond: bool, detail: str = "") -> None:
    checks.append((name, bool(cond), detail))
    print(("PASS" if cond else "FAIL"), name, detail)


def main() -> int:
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
    title_block = ConsultingTitleBlock(
        section_label="B-B'",
        map_scale="1:1000",
        drawn_by="EC",
        prepared_for="CLIENT",
        notes=("NOTE ONE", "NOTE TWO"),
    )
    _, _, svg, png, _, codes, _ = build_cross_section(
        collars,
        lith,
        [(0, 0), (100, 0)],
        render_layout="consulting_section",
        show_legend=False,
        show_hatches=True,
        water_levels=water,
        consulting_title_block=title_block,
        screen_intervals=[ScreenInterval(hole_id="MW-01", from_depth=5, to_depth=10)],
        vertical_gradients=[VerticalGradient(hole_id="MW-01", direction="up")],
        export_formats=frozenset({"svg", "png"}),
    )
    text = svg.decode("utf-8", errors="ignore")
    ok("consulting_svg_valid", svg_is_valid(svg))
    ok("consulting_png", len(png) > 100)
    ok("consulting_nm", "NM" in text)
    ok("consulting_notes", "NOTES" in text)
    ok("consulting_drawn_by", "DRAWN BY" in text)
    ok("consulting_codes", "Sand" in codes or "Clay" in codes)

    try:
        _, _, svg1, _, _, _, _ = build_cross_section(
            collars[:1],
            lith[:2],
            [(0, 0), (10, 0)],
            interpretation_mode="borehole_only",
            show_legend=False,
            export_formats=frozenset({"svg"}),
        )
        ok("single_hole_borehole_only", len(svg1) > 100)
    except Exception as exc:
        ok("single_hole_borehole_only", False, str(exc))

    dup_lith = [
        Lithology(hole_id="MW-01", from_depth=0, to_depth=5, lithology_code="Clay"),
        Lithology(hole_id="MW-01", from_depth=5, to_depth=10, lithology_code="Clay"),
    ]
    qr = analyze_parsed_data(collars[:1], dup_lith)
    ok(
        "qa_dup_unit_order_error",
        any(issue.code == "duplicate_lithology_no_unit_order" for issue in qr.issues),
    )
    pr = ParseResult(collars=tuple(collars[:1]), lithologies=tuple(dup_lith), errors=())
    fixed = apply_unit_order_fix(pr)
    ok("auto_unit_order_fix", all(item.unit_order is not None for item in fixed.lithologies))

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
    ok("ingest_screens", len(parsed.screen_intervals) == 1)
    ok("ingest_gradients_down", parsed.vertical_gradients[0].direction == "down")
    subset = subset_parse_result(parsed, ["MW-01"])
    ok(
        "subset_drops_other_gradient",
        len(subset.vertical_gradients) == 0 and len(subset.screen_intervals) == 1,
    )

    assistant = AIAssistant(None)
    ok("ai_empty_fix_plan", assistant.suggest_fix_plan(()) == ())
    ok(
        "ai_unknown_hole_transect",
        assistant.parse_transect_request("through XX-99", ["MW-01", "MW-02"]) is None,
    )
    ok(
        "ai_one_hole_transect",
        assistant.parse_transect_request("MW-01 only", ["MW-01", "MW-02"]) is None,
    )
    parsed_transect = assistant.parse_transect_request(
        "B-B' MW-01 MW-02",
        ["MW-01", "MW-02"],
    )
    ok(
        "ai_nl_transect",
        parsed_transect is not None and parsed_transect.hole_ids == ("MW-01", "MW-02"),
    )
    ok(
        "ai_nm_answer",
        "MW-03"
        in assistant.answer_section_question(
            "NM wells?",
            {
                "hole_ids": ["MW-01", "MW-03"],
                "water_levels": {"MW-01": 1.0},
                "nm_hole_ids": ["MW-03"],
                "lithology_thicknesses": {},
                "offsets_m": {},
                "overlap_warnings": [],
            },
        ),
    )
    ok("ai_empty_question", "Ask a question" in assistant.answer_section_question("  ", {}))
    ok(
        "ai_fat_clay",
        assistant.suggest_lithology_mappings(["Fat Clay"])[0].canonical_code == "Clay",
    )

    wide = [
        Collar(hole_id="A", easting=0, northing=0, elevation=100, total_depth=10),
        Collar(hole_id="B", easting=240, northing=0, elevation=100, total_depth=10),
    ]
    wide_lith = [
        Lithology(hole_id="A", from_depth=0, to_depth=10, lithology_code="Sand"),
        Lithology(hole_id="B", from_depth=0, to_depth=10, lithology_code="Clay"),
    ]
    _, _, svg_wide, _, _, _, _ = build_cross_section(
        wide,
        wide_lith,
        [(0, 0), (240, 0)],
        render_layout="consulting_section",
        show_legend=False,
        export_formats=frozenset({"svg"}),
    )
    ok("wide_transect_consulting", len(svg_wide) > 1000)

    overlap_lith = [
        Lithology(hole_id="MW-01", from_depth=0, to_depth=6, lithology_code="Sand"),
        Lithology(hole_id="MW-01", from_depth=5, to_depth=10, lithology_code="Clay"),
    ]
    qr_overlap = analyze_parsed_data(collars[:1], overlap_lith)
    ok("qa_depth_overlap", any(issue.code == "depth_overlap" for issue in qr_overlap.issues))
    plan = assistant.suggest_fix_plan(qr_overlap.issues)
    ok("fix_plan_overlap", any(step.action_id == "manual_intervals" for step in plan))

    # Empty water list still labels all wells NM in consulting layout
    _, _, svg_nm, _, _, _, _ = build_cross_section(
        collars[:2],
        lith[:4],
        [(0, 0), (50, 0)],
        render_layout="consulting_section",
        show_legend=False,
        water_levels=[],
        export_formats=frozenset({"svg"}),
    )
    ok("all_wells_nm", svg_nm.decode("utf-8", errors="ignore").count("NM") >= 2)

    failed = [name for name, passed, _ in checks if not passed]
    print("---")
    print(f"{len(checks) - len(failed)}/{len(checks)} edge checks passed")
    if failed:
        for name, passed, detail in checks:
            if not passed:
                print("FAILED", name, detail)
        return 1
    print("ALL EDGE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
