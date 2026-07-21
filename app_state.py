"""Streamlit session defaults and domain clear helpers."""

from __future__ import annotations

from typing import Any

# Bump when session key shapes change so Cloud reconnects drop stale state.
SESSION_SCHEMA_VERSION = 2

# Domain key groups for targeted resets
SESSION_PARSE_KEYS = (
    "file_bytes",
    "parse_result",
    "import_report",
    "quality_report",
    "mapping_proposal",
    "detection_result",
    "hole_ids",
    "unique_lithology_codes",
    "lithology_index",
    "parse_signature",
    "file_hash",
    "lithology_aliases",
)

SESSION_SECTION_KEYS = (
    "render_cache_key",
    "polygon_overlap_warnings",
    "section_lithology_codes",
    "section_polygon_count",
    "section_hole_count",
    "transect_selection_key",
    "transect_selection",
    "svg_display_meta",
    "transect_candidates",
    "svg_bytes",
    "png_bytes",
    "pdf_bytes",
    "section_build_subset_json",
    "section_build_request_json",
)

SESSION_AI_KEYS = (
    "qa_narrative",
    "qa_fix_plan",
    "ai_report_suggestion",
    "ai_lithology_suggestions",
    "ai_correlation_suggestions",
    "ai_sheet_roles",
    "ai_column_suggestions",
    "section_qa_answer",
    "ai_figure_caption",
)

DEFAULT_SESSION: dict[str, object] = {
    "file_bytes": None,
    "parse_result": None,
    "import_report": None,
    "quality_report": None,
    "mapping_proposal": None,
    "detection_result": None,
    "section_lithology_codes": None,
    "hole_ids": [],
    "unique_lithology_codes": [],
    "render_cache_key": None,
    "polygon_overlap_warnings": [],
    "suggested_offset_m": 50.0,
    "section_polygon_count": None,
    "section_hole_count": None,
    "lithology_index": None,
    "parse_signature": None,
    "file_hash": None,
    "transect_selection_key": None,
    "transect_selection": None,
    "svg_display_meta": None,
    "offset_warning_m": 50.0,
    "uncertainty_offset_m": 50.0,
    "transect_candidates": None,
    "svg_bytes": None,
    "png_bytes": None,
    "pdf_bytes": None,
    "section_build_subset_json": None,
    "section_build_request_json": None,
    "qa_narrative": None,
    "qa_fix_plan": None,
    "ai_report_suggestion": None,
    "ai_lithology_suggestions": None,
    "ai_correlation_suggestions": None,
    "ai_sheet_roles": None,
    "ai_column_suggestions": None,
    "section_qa_answer": None,
    "nl_transect_text": "",
    "lithology_aliases": None,
    "session_correlation_overrides": None,
    "elevation_mode": "absolute",
    "auto_assign_unit_order": True,
    "ai_figure_caption": None,
    "uploaded_name": None,
    "workbook_uploader_key": 0,
    "output_preset": "section_sheet",
    "allow_pinch_outs": False,
    "show_ground_surface": True,
    "show_hatches": True,
    "show_legend": True,
    "enable_ai_suggestions": False,
    "fail_on_overlaps": False,
    "llm_provider": "groq",
}


def _clear_keys(session: Any, keys: tuple[str, ...]) -> None:
    for key in keys:
        if key in DEFAULT_SESSION:
            session[key] = DEFAULT_SESSION[key]
        elif key in session:
            session[key] = None


def clear_parse_session_state(session: Any | None = None) -> None:
    """Clear workbook parse / detection state."""
    import streamlit as st

    target = session if session is not None else st.session_state
    _clear_keys(target, SESSION_PARSE_KEYS)


def clear_section_output_state(session: Any | None = None) -> None:
    """Clear generated section artifacts (SVG/PNG/PDF and build cache)."""
    import streamlit as st

    target = session if session is not None else st.session_state
    _clear_keys(target, SESSION_SECTION_KEYS)


def clear_ai_session_state(session: Any | None = None) -> None:
    """Clear advisory AI suggestions and narratives."""
    import streamlit as st

    target = session if session is not None else st.session_state
    _clear_keys(target, SESSION_AI_KEYS)


def init_session_defaults(session: Any | None = None) -> None:
    import streamlit as st

    target = session if session is not None else st.session_state
    current_version = target.get("_schema_version")
    if current_version != SESSION_SCHEMA_VERSION:
        clear_parse_session_state(target)
        clear_section_output_state(target)
        clear_ai_session_state(target)
        target["_schema_version"] = SESSION_SCHEMA_VERSION
    for key, value in DEFAULT_SESSION.items():
        if key not in target:
            target[key] = value
