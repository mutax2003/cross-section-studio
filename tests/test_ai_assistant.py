"""Tests for ai_assistant.py and transect_planner.py."""

from __future__ import annotations

from ai_assistant import AIAssistant, MockLLMProvider
from ai_quality import ColumnMapping, MappingProposal, QualityIssue
from models import Collar, Lithology
from stratigraphy import CorrelationPairSummary
from transect_planner import recommend_transects, score_transect


def test_mock_llm_lithology_suggestions() -> None:
    assistant = AIAssistant(MockLLMProvider())
    suggestions = assistant.suggest_lithology_mappings(["Fat Clay"])
    assert len(suggestions) == 1
    assert suggestions[0].canonical_code == "Clay"


def test_local_lithology_suggestions_without_provider() -> None:
    assistant = AIAssistant(None)
    suggestions = assistant.suggest_lithology_mappings(["silty clay with organics", "top soil"])
    codes = {item.source_code: item.canonical_code for item in suggestions}
    assert codes["silty clay with organics"] == "Silty Clay"
    assert codes["top soil"] == "Topsoil"


def test_local_narrative_without_provider() -> None:
    assistant = AIAssistant(None)
    text = assistant.explain_quality_issues(
        [
            QualityIssue(
                code="depth_gap",
                message="gap",
                severity="warning",
                hole_id="BH-01",
            )
        ]
    )
    assert "warning" in text


def test_local_fix_plan_has_actionable_steps() -> None:
    assistant = AIAssistant(None)
    steps = assistant.suggest_fix_plan(
        [
            QualityIssue(
                code="duplicate_lithology_no_unit_order",
                message="dup",
                severity="error",
                hole_id="BH-01",
            ),
            QualityIssue(
                code="placeholder_elevation",
                message="placeholder",
                severity="warning",
            ),
        ]
    )
    assert len(steps) == 2
    by_code = {step.issue_code: step for step in steps}
    assert by_code["duplicate_lithology_no_unit_order"].action_id == "auto_unit_order"
    assert by_code["duplicate_lithology_no_unit_order"].blocks_generate is True
    assert by_code["placeholder_elevation"].action_id == "relative_elevation"


def test_mock_llm_fix_plan() -> None:
    assistant = AIAssistant(MockLLMProvider())
    steps = assistant.suggest_fix_plan(
        [
            QualityIssue(
                code="duplicate_lithology_no_unit_order",
                message="dup",
                severity="error",
                hole_id="BH-01",
            )
        ]
    )
    assert steps[0].action_id == "auto_unit_order"


def test_local_report_metadata() -> None:
    assistant = AIAssistant(None)
    suggestion = assistant.suggest_report_metadata(
        {
            "hole_ids": ["MW-01", "MW-02", "MW-03"],
            "water_measurement_count": 2,
            "nm_hole_ids": ["MW-03"],
            "vertical_exaggeration": 5.0,
            "map_scale": "1:1000",
            "section_label": "B-B'",
            "workbook_name": "site.xlsx",
        }
    )
    assert suggestion.section_label == "B-B'"
    assert any("MW-03" in note for note in suggestion.notes)
    assert "MW-01" in suggestion.figure_caption
    assert suggestion.source == "site.xlsx"

    prefixed = assistant.suggest_report_metadata(
        {"section_label": "CROSS SECTION C-C'", "hole_ids": ["MW-01", "MW-02"]}
    )
    assert prefixed.section_label == "C-C'"
    assert "CROSS SECTION CROSS SECTION" not in prefixed.figure_caption.upper()


def test_local_correlation_suggestions() -> None:
    assistant = AIAssistant(None)
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Sand", unit_order=1),
        Lithology(hole_id="BH-01", from_depth=5.0, to_depth=10.0, lithology_code="Clay", unit_order=2),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=4.0, lithology_code="Sand", unit_order=2),
        Lithology(hole_id="BH-02", from_depth=4.0, to_depth=9.0, lithology_code="Clay", unit_order=3),
    ]
    summaries = [
        CorrelationPairSummary(
            left_hole_id="BH-01",
            right_hole_id="BH-02",
            matched_count=0,
            left_only_codes=("Sand", "Clay"),
            right_only_codes=("Sand", "Clay"),
            pinch_out_candidates=2,
        )
    ]
    suggestions = assistant.suggest_correlation_overrides(
        summaries,
        lithologies,
        ("BH-01", "BH-02"),
    )
    assert suggestions
    override = suggestions[0].to_override()
    assert override.left_hole_id == "BH-01"
    assert override.right_hole_id == "BH-02"
    assert override.left_unit_order == 1
    assert override.right_unit_order == 2


def test_local_section_qa() -> None:
    assistant = AIAssistant(None)
    facts = {
        "hole_ids": ["MW-01", "MW-02", "MW-03"],
        "water_levels": {"MW-01": 2.5, "MW-02": 3.0},
        "nm_hole_ids": ["MW-03"],
        "lithology_thicknesses": {"Clay": {"MW-01": 8.0, "MW-02": 7.0}},
        "offsets_m": {"MW-01": 1.2},
        "overlap_warnings": [],
    }
    assert "MW-03" in assistant.answer_section_question("Which wells are NM?", facts)
    assert "2.5" in assistant.answer_section_question("What are the water levels?", facts)
    assert "8.0" in assistant.answer_section_question("Clay thickness?", facts)


def test_local_sheet_roles() -> None:
    assistant = AIAssistant(None)
    suggestions = assistant.suggest_sheet_roles(
        ["BH_Collars", "Intervals", "Screens"],
        {
            "BH_Collars": ["hole_id", "easting", "northing", "elevation", "total_depth"],
            "Intervals": ["hole_id", "from_depth", "to_depth", "lithology_code"],
            "Screens": ["hole_id", "from_depth", "to_depth"],
        },
    )
    by_sheet = {item.sheet_name: item.role for item in suggestions}
    assert by_sheet["BH_Collars"] == "collars"
    assert by_sheet["Intervals"] == "lithology"
    assert by_sheet["Screens"] == "screens"


def test_local_transect_parse() -> None:
    assistant = AIAssistant(None)
    parsed = assistant.parse_transect_request(
        "Section B-B' through MW-01, MW-03, and MW-07",
        ["MW-01", "MW-02", "MW-03", "MW-07"],
    )
    assert parsed is not None
    assert parsed.hole_ids == ("MW-01", "MW-03", "MW-07")
    assert "B" in parsed.section_label.upper()


def test_llm_column_mapping_fallback() -> None:
    proposal = MappingProposal(
        collars_sheet="Collars",
        lithology_sheet="Lithology",
        collar_column_mappings=(
            ColumnMapping("Unknown", "hole_id", 0.4),
        ),
        lithology_column_mappings=(),
    )
    assistant = AIAssistant(MockLLMProvider())
    suggestions = assistant.suggest_column_mappings(proposal, sheet="collars")
    assert suggestions[0].canonical_column == "hole_id"


def test_transect_recommender_orders_candidates() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=101.0, total_depth=20.0),
        Collar(hole_id="BH-03", easting=100.0, northing=0.0, elevation=102.0, total_depth=20.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Sandstone"),
        Lithology(hole_id="BH-01", from_depth=5.0, to_depth=15.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=5.0, lithology_code="Sandstone"),
        Lithology(hole_id="BH-02", from_depth=5.0, to_depth=15.0, lithology_code="Silt"),
        Lithology(hole_id="BH-03", from_depth=0.0, to_depth=15.0, lithology_code="Bedrock"),
    ]
    candidates = recommend_transects(collars, lithologies, top_n=2)
    assert len(candidates) >= 1
    assert candidates[0].score >= candidates[-1].score

    scored = score_transect(collars, lithologies, ("BH-01", "BH-02", "BH-03"))
    assert scored.length_m == 100.0
