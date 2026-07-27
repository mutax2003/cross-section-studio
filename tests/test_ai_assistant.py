"""Tests for ai_assistant.py and transect_planner.py."""

from __future__ import annotations

import json

from ai_assistant import (
    AIAssistant,
    GeminiProvider,
    GroqProvider,
    MockLLMProvider,
    OpenAIProvider,
    _load_json_content,
    build_llm_provider,
    resolve_llm_api_key,
)
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


def test_load_json_content_strips_fences() -> None:
    payload = _load_json_content('```json\n{"summary": "ok"}\n```')
    assert payload["summary"] == "ok"


def test_build_llm_provider_returns_none_without_key() -> None:
    assert build_llm_provider("groq", "") is None


def test_build_llm_provider_groq_and_openai() -> None:
    groq = build_llm_provider("groq", "gsk-test")
    openai = build_llm_provider("openai", "sk-test")
    assert isinstance(groq, GroqProvider)
    assert isinstance(openai, OpenAIProvider)


def test_resolve_llm_api_key_from_env(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "env-groq-key")
    assert resolve_llm_api_key("groq", "") == "env-groq-key"
    assert resolve_llm_api_key("groq", "inline") == "inline"


def test_gemini_provider_complete_json(monkeypatch) -> None:
    response_body = json.dumps(
        {
            "candidates": [
                {"content": {"parts": [{"text": '{"summary": "Gemini QA text"}'}]}}
            ]
        }
    ).encode("utf-8")

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return response_body

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: _FakeResponse())
    provider = GeminiProvider("gemini-test-key")
    payload = provider.complete_json("system", "user")
    assert payload["summary"] == "Gemini QA text"


def test_preferred_free_llm_provider_from_env(monkeypatch) -> None:
    from ai_assistant import (
        is_free_llm_provider,
        preferred_llm_provider_from_env,
    )

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert preferred_llm_provider_from_env() is None

    monkeypatch.setenv("OPENAI_API_KEY", "sk-paid")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-free")
    assert preferred_llm_provider_from_env() == "gemini"
    assert is_free_llm_provider("gemini")

    monkeypatch.setenv("GROQ_API_KEY", "gsk-free")
    assert preferred_llm_provider_from_env() == "groq"
    assert is_free_llm_provider("groq")
    assert not is_free_llm_provider("openai")


def test_seed_free_llm_defaults_enables_groq(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-free")
    monkeypatch.delenv("CROSS_SECTION_DISABLE_LLM", raising=False)

    class _FakeSession(dict):
        def get(self, key, default=None):
            return super().get(key, default)

        def __getattr__(self, name: str):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name: str, value) -> None:
            self[name] = value

    from app_sidebar import _seed_free_llm_defaults

    session = _FakeSession()
    monkeypatch.setattr("app_sidebar.st.session_state", session, raising=False)
    monkeypatch.setattr("app_sidebar.llm_disabled_by_deployment", lambda: False)
    _seed_free_llm_defaults()
    assert session["llm_provider"] == "groq"
    assert session["enable_ai_suggestions"] is True


def test_build_assistant_uses_env_groq_key(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "env-groq-key")
    monkeypatch.delenv("CROSS_SECTION_DISABLE_LLM", raising=False)

    class _FakeSession(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    from app_common import _build_assistant

    session = _FakeSession({"enable_ai_suggestions": True, "llm_provider": "groq"})
    monkeypatch.setattr("app_common.st.session_state", session, raising=False)
    assistant = _build_assistant()
    assert assistant.enabled
    assert isinstance(assistant.provider, GroqProvider)


def test_build_assistant_respects_disable_llm(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "env-groq-key")
    monkeypatch.setenv("CROSS_SECTION_DISABLE_LLM", "1")

    class _FakeSession(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    from app_common import _build_assistant, llm_disabled_by_deployment, llm_suggestions_available

    assert llm_disabled_by_deployment() is True
    session = _FakeSession({"enable_ai_suggestions": True, "llm_provider": "groq"})
    monkeypatch.setattr("app_common.st.session_state", session, raising=False)
    assistant = _build_assistant()
    assert not assistant.enabled
    assert llm_suggestions_available() is False


def test_clear_ai_session_state_resets_keys() -> None:
    from app_state import SESSION_AI_KEYS, clear_ai_session_state, init_session_defaults

    class _FakeSession(dict):
        pass

    session = _FakeSession()
    init_session_defaults(session)
    session["qa_narrative"] = "stale"
    session["ai_column_suggestions"] = {"collars": ()}
    session["ai_figure_caption"] = "caption"
    clear_ai_session_state(session)
    for key in SESSION_AI_KEYS:
        assert session[key] is None


def test_apply_report_suggestion_merges_figure_caption(monkeypatch) -> None:
    from ai_assistant import ReportMetadataSuggestion
    from app_common import _apply_report_suggestion

    class _FakeSession(dict):
        def get(self, key, default=None):
            return super().get(key, default)

        def __getattr__(self, name: str):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name: str, value) -> None:
            self[name] = value

    session = _FakeSession()
    monkeypatch.setattr("app_common.st.session_state", session, raising=False)
    suggestion = ReportMetadataSuggestion(
        section_label="A-A'",
        map_scale="1:1000",
        figure_caption="Cross section A-A' through MW-01.",
        notes=("NM at MW-03.",),
        prepared_for="Client",
        prepared_by="Firm",
        source="site.xlsx",
        project_number="P-1",
        transect_start_label="MW-01",
        transect_end_label="MW-03",
    )
    _apply_report_suggestion(suggestion)
    pending = session["_pending_project_seed"]
    notes = pending["consulting_notes"]
    assert "Cross section A-A' through MW-01." in notes
    assert "NM at MW-03." in notes
    assert session["ai_figure_caption"] == suggestion.figure_caption


def test_column_rename_checklist_format() -> None:
    from app_validate import _column_rename_checklist

    text = _column_rename_checklist(
        {
            "collars": (ColumnMapping("Unknown", "hole_id", 0.9),),
            "lithology": (),
        }
    )
    assert "`Unknown` → `hole_id`" in text
    assert "Column rename checklist" in text


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
        "water_levels": {"MW-01": {"default": 2.5}, "MW-02": {"default": 3.0}},
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
