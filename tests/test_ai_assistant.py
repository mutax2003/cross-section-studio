"""Tests for ai_assistant.py and transect_planner.py."""

from __future__ import annotations

from ai_assistant import AIAssistant, MockLLMProvider
from ai_quality import ColumnMapping, MappingProposal, QualityIssue
from models import Collar, Lithology
from transect_planner import recommend_transects, score_transect


def test_mock_llm_lithology_suggestions() -> None:
    assistant = AIAssistant(MockLLMProvider())
    suggestions = assistant.suggest_lithology_mappings(["Fat Clay"])
    assert len(suggestions) == 1
    assert suggestions[0].canonical_code == "Clay"


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
