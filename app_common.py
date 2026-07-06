"""Shared Streamlit helpers (no page layout)."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Sequence

from dataclasses import dataclass, replace

import pandas as pd
import streamlit as st

from ai_assistant import AIAssistant, OpenAIProvider

if TYPE_CHECKING:
    from ai_assistant import ReportMetadataSuggestion

from ai_quality import analyze_parsed_data, load_lithology_aliases
from app_services import cached_build_section, cached_ingest_workbook
from app_state import clear_section_output_state
from constants import USGS_LITHOLOGY_HATCHES, get_lithology_style
from ingestion import ImportReport
from models import (
    ConsultingTitleBlock,
    CorrelationOverride,
    Lithology,
    ParseResult,
    Transect,
    apply_unit_order_fix,
    lithologies_by_hole,
    subset_parse_result,
)
from projection import off_transect_warnings
from section_build_request import SectionBuildRequest
from ui_helpers import (
    active_transect_selection,
    dedupe_messages,
    escape_html,
    holes_missing_lithology,
    legend_hatch_background,
    svg_display_meta,
)

logger = logging.getLogger(__name__)

_WORKFLOW_LABELS = ("Upload", "Validate", "Configure", "Generate")


def _render_workflow_stepper(stage: int) -> None:
    steps_html = []
    for index, label in enumerate(_WORKFLOW_LABELS):
        if index < stage:
            css_class = "workflow-step done"
        elif index == stage:
            css_class = "workflow-step active"
        else:
            css_class = "workflow-step"
        steps_html.append(
            f'<div class="{css_class}" role="listitem" aria-current="{"step" if index == stage else "false"}">'
            f"{index + 1}. {escape_html(label)}</div>"
        )
    st.markdown(f'<div class="workflow" role="list">{"".join(steps_html)}</div>', unsafe_allow_html=True)


def _render_hero(stage: int) -> None:
    st.markdown(
        """
<div class="app-hero">
  <h1>Cross Section Studio</h1>
  <p>Upload borehole data, validate stratigraphy, and generate publication-ready profiles with USGS-style fills and hatch patterns.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    _render_workflow_stepper(stage)


def _metric_tone(error_count: int, warning_count: int, *, errors_only: bool = False) -> str:
    if error_count > 0:
        return "error"
    if errors_only:
        return "ok"
    if warning_count > 0:
        return "warn"
    return "ok"


def _render_metric_card(value: str | int, label: str, tone: str = "ok") -> None:
    st.markdown(
        f'<div class="metric-card {tone}" role="status">'
        f'<div class="value">{escape_html(value)}</div>'
        f'<div class="label">{escape_html(label)}</div></div>',
        unsafe_allow_html=True,
    )


def _render_profile_chips(
    *,
    interpretation_mode: str,
    vertical_exaggeration: float,
    hole_count: int | None,
    polygon_count: int | None,
    is_stale: bool,
) -> None:
    mode_label = (
        "Observed only"
        if interpretation_mode == "borehole_only"
        else "Interpolated fence"
    )
    chips = [
        f'<span class="chip brand">{escape_html(mode_label)}</span>',
        f'<span class="chip">VE {escape_html(vertical_exaggeration)}×</span>',
    ]
    if hole_count is not None:
        chips.append(f'<span class="chip">{escape_html(hole_count)} boreholes</span>')
    if polygon_count is not None and interpretation_mode == "interpolated":
        chips.append(f'<span class="chip">{escape_html(polygon_count)} polygons</span>')
    if is_stale:
        chips.append('<span class="chip warn">Settings changed</span>')
    st.markdown(f'<div class="profile-header">{"".join(chips)}</div>', unsafe_allow_html=True)


def _sidebar_heading(title: str) -> None:
    st.markdown(f'<div class="sidebar-section-title">{title}</div>', unsafe_allow_html=True)



def _get_aliases() -> dict[str, str]:
    if st.session_state.lithology_aliases is None:
        st.session_state.lithology_aliases = load_lithology_aliases()
    return st.session_state.lithology_aliases


def _render_lithology_legend(codes: list[str]) -> None:
    if not codes:
        st.caption("Legend appears after you generate a cross-section.")
        return
    rows = []
    for code in sorted(codes):
        style = get_lithology_style(code)
        hatch = USGS_LITHOLOGY_HATCHES.get(code, "..")
        hatch_bg = legend_hatch_background(hatch)
        rows.append(
            f'<div class="legend-row">'
            f'<span class="legend-swatch" style="background-color:{style.color};'
            f'background-image:{hatch_bg};"></span>'
            f"<strong>{escape_html(code)}</strong>"
            f"</div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def _display_svg(svg_bytes: bytes) -> None:
    """Render SVG in Streamlit (st.image does not support SVG via PIL)."""
    cached = st.session_state.get("svg_display_meta")
    if cached is None or st.session_state.svg_bytes != svg_bytes:
        cached = svg_display_meta(svg_bytes)
        st.session_state.svg_display_meta = cached
    if not cached.valid:
        st.error("Renderer produced invalid or empty SVG output.")
        return
    st.markdown(
        f'<div class="svg-frame" role="img" aria-label="Cross-section profile" '
        f'style="min-height:{cached.height}px;">',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<img src="data:image/svg+xml;base64,{cached.encoded}" '
        'style="width:100%;height:auto;display:block;" alt="Cross-section profile" />',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_overlap_warnings(warnings: Sequence[str]) -> None:
    unique = dedupe_messages(warnings)
    if not unique:
        return
    if len(unique) == 1:
        st.warning(f"Polygon overlap: {unique[0]}")
        return
    with st.expander(
        f"{len(unique)} polygon overlaps detected on section",
        expanded=True,
    ):
        for message in unique:
            st.write(message)


def _active_transect_selection(
    parse_result: ParseResult,
    transect_mode: str,
    selected_holes: list[str],
    coordinate_text: str,
    offset_warning_m: float,
) -> tuple[tuple[str, ...], tuple[tuple[float, float], ...]] | None:
    selection_key = (
        transect_mode,
        tuple(selected_holes),
        coordinate_text,
        offset_warning_m,
        len(parse_result.collars),
    )
    if (
        st.session_state.get("transect_selection_key") == selection_key
        and st.session_state.get("transect_selection") is not None
    ):
        return st.session_state.transect_selection
    selection = active_transect_selection(
        parse_result.collars,
        transect_mode,
        selected_holes,
        coordinate_text,
        offset_warning_m,
    )
    st.session_state.transect_selection_key = selection_key
    st.session_state.transect_selection = selection
    return selection


def _reanalyze_quality(
    parse_result: ParseResult,
    import_report: ImportReport | None,
    *,
    placeholder_elevation_m: float | None = None,
) -> None:
    qa = analyze_parsed_data(
        parse_result.collars,
        parse_result.lithologies,
        placeholder_elevation_m=placeholder_elevation_m,
    )
    st.session_state.quality_report = qa
    st.session_state.parse_result = parse_result
    st.session_state.lithology_index = lithologies_by_hole(parse_result.lithologies)
    st.session_state.hole_ids = [collar.hole_id for collar in parse_result.collars]
    st.session_state.unique_lithology_codes = sorted(
        {lithology.lithology_code for lithology in parse_result.lithologies}
    )
    st.session_state.qa_fix_plan = None
    st.session_state.qa_narrative = None
    if import_report is not None:
        st.session_state.import_report = replace(import_report, quality_report=qa)


def _apply_auto_unit_order_fix(
    parse_result: ParseResult,
    import_report: ImportReport | None,
    *,
    success_message: str,
) -> None:
    fixed = apply_unit_order_fix(parse_result)
    placeholder_m = (
        import_report.profile_default_elevation_m
        if import_report and import_report.uses_placeholder_elevation
        else None
    )
    _reanalyze_quality(fixed, import_report, placeholder_elevation_m=placeholder_m)
    st.session_state.transect_candidates = None
    st.session_state.ai_correlation_suggestions = None
    clear_section_output_state()
    st.success(success_message)
    st.rerun()


def _session_correlation_overrides() -> tuple[CorrelationOverride, ...]:
    overrides = st.session_state.get("session_correlation_overrides")
    if not overrides:
        return ()
    return tuple(overrides)


def _build_consulting_title_block(
    section_title: str,
    *,
    section_label: str,
    transect_start_label: str,
    transect_end_label: str,
    transect_start_primary: str,
    transect_start_secondary: str,
    transect_end_primary: str,
    transect_end_secondary: str,
    map_scale: str,
    figure_number: str,
    project_number: str,
    source: str,
    date: str,
    notes_text: str,
    drawn_by: str,
    revised: str,
    prepared_for: str,
    prepared_by: str,
    logo_prepared_for_bytes: bytes | None,
    logo_prepared_by_bytes: bytes | None,
) -> ConsultingTitleBlock:
    notes = tuple(line.strip() for line in notes_text.splitlines() if line.strip())
    return ConsultingTitleBlock(
        section_label=section_label or section_title,
        transect_start_label=transect_start_label.strip(),
        transect_end_label=transect_end_label.strip(),
        transect_start_primary=transect_start_primary.strip(),
        transect_start_secondary=transect_start_secondary.strip(),
        transect_end_primary=transect_end_primary.strip(),
        transect_end_secondary=transect_end_secondary.strip(),
        map_scale=map_scale.strip() or "1:1000",
        figure_number=figure_number.strip(),
        project_number=project_number.strip(),
        source=source.strip(),
        date=date.strip(),
        notes=notes or _default_consulting_notes(),
        drawn_by=drawn_by.strip(),
        revised=revised.strip(),
        prepared_for=prepared_for.strip(),
        prepared_by=prepared_by.strip(),
        logo_prepared_for_bytes=logo_prepared_for_bytes,
        logo_prepared_by_bytes=logo_prepared_by_bytes,
    )


@dataclass(frozen=True)
class _EffectiveRenderOptions:
    layout: str
    show_ground_surface: bool
    track_width_m: float
    show_legend: bool
    interpolate_water_table: bool
    allow_pinch_outs: bool
    consulting_title_block: ConsultingTitleBlock | None


def _effective_render_options(**kwargs) -> _EffectiveRenderOptions:
    from app_build import EffectiveRenderOptions, effective_render_options

    result = effective_render_options(**kwargs)
    return _EffectiveRenderOptions(
        layout=result.layout,
        show_ground_surface=result.show_ground_surface,
        track_width_m=result.track_width_m,
        show_legend=result.show_legend,
        interpolate_water_table=result.interpolate_water_table,
        allow_pinch_outs=result.allow_pinch_outs,
        consulting_title_block=result.consulting_title_block,
    )


def _build_section_request(**kwargs) -> SectionBuildRequest:
    from app_build import build_section_request

    return build_section_request(**kwargs)


def _sidebar_render_cache_key(*args, **kwargs) -> str | None:
    from app_build import sidebar_render_cache_key

    return sidebar_render_cache_key(*args, **kwargs)


def _resolve_section_request_and_cache_key(*args, **kwargs):
    from app_build import collect_section_build_request

    return collect_section_build_request(*args, **kwargs)


def _parse_signature_key(
    *,
    profile_id: str | None,
    override_id: str | None,
    elevation_m: float | None,
    target_crs: str | None,
    file_hash: str | None = None,
) -> str:
    payload: dict[str, Any] = {
        "profile": profile_id,
        "override": override_id,
        "elevation": elevation_m,
        "crs": target_crs,
        "aliases": _get_aliases(),
    }
    if file_hash:
        payload["file"] = file_hash
    return json.dumps(payload, sort_keys=True, default=str)


def _health_emoji(error_count: int, warning_count: int) -> str:
    if error_count > 0:
        return "🔴"
    if warning_count > 0:
        return "🟡"
    return "🟢"


def _build_assistant() -> AIAssistant:
    """Return LLM-backed assistant only when toggle is on and an API key is set."""
    if not st.session_state.get("enable_ai_suggestions"):
        return AIAssistant(None)
    api_key = st.session_state.get("openai_api_key", "").strip()
    if api_key:
        return AIAssistant(OpenAIProvider(api_key=api_key))
    return AIAssistant(None)


def _default_consulting_notes() -> tuple[str, ...]:
    """Lazy import avoids Streamlit hot-reload ImportError on partial modules."""
    from render_theme import DEFAULT_CONSULTING_NOTES

    return DEFAULT_CONSULTING_NOTES


def _apply_report_suggestion(suggestion: ReportMetadataSuggestion) -> None:
    st.session_state.consulting_section_label = suggestion.section_label
    st.session_state.consulting_map_scale = suggestion.map_scale
    st.session_state.consulting_notes = "\n".join(suggestion.notes)
    st.session_state.consulting_prepared_for = suggestion.prepared_for
    st.session_state.consulting_prepared_by = suggestion.prepared_by
    st.session_state.consulting_source = suggestion.source
    st.session_state.consulting_project_number = suggestion.project_number
    st.session_state.consulting_start_label = suggestion.transect_start_label
    st.session_state.consulting_end_label = suggestion.transect_end_label
    st.session_state.ai_figure_caption = suggestion.figure_caption


def _water_and_nm(
    parse_result: ParseResult,
    hole_ids: Sequence[str],
) -> tuple[dict[str, float], list[str]]:
    hole_set = set(hole_ids)
    water = {
        level.hole_id: level.depth
        for level in parse_result.water_levels
        if level.hole_id in hole_set
    }
    nm_holes = [hole_id for hole_id in hole_ids if hole_id not in water]
    return water, nm_holes


def _report_context_from_selection(
    parse_result: ParseResult,
    hole_ids: Sequence[str],
    *,
    vertical_exaggeration: float,
    map_scale: str,
    section_title: str,
) -> dict[str, object]:
    water, nm_holes = _water_and_nm(parse_result, hole_ids)
    return {
        "hole_ids": list(hole_ids),
        "water_measurement_count": len(water),
        "nm_hole_ids": nm_holes,
        "vertical_exaggeration": vertical_exaggeration,
        "map_scale": map_scale,
        "section_label": section_title,
        "workbook_name": str(st.session_state.get("uploaded_name") or ""),
    }


def _section_facts(
    parse_result: ParseResult,
    hole_ids: Sequence[str],
    *,
    offsets_m: dict[str, float] | None = None,
) -> dict[str, object]:
    hole_set = set(hole_ids)
    water, nm_holes = _water_and_nm(parse_result, hole_ids)
    thicknesses: dict[str, dict[str, float]] = {}
    for lith in parse_result.lithologies:
        if lith.hole_id not in hole_set:
            continue
        thick = max(0.0, lith.to_depth - lith.from_depth)
        by_hole = thicknesses.setdefault(lith.lithology_code, {})
        by_hole[lith.hole_id] = by_hole.get(lith.hole_id, 0.0) + thick
    return {
        "hole_ids": list(hole_ids),
        "water_levels": water,
        "nm_hole_ids": nm_holes,
        "lithology_thicknesses": thicknesses,
        "offsets_m": offsets_m or {},
        "overlap_warnings": list(st.session_state.get("polygon_overlap_warnings") or []),
    }


def _mapping_rows(mappings: tuple) -> list[dict[str, str]]:
    return [
        {
            "source": mapping.source_column,
            "canonical": mapping.canonical_column,
            "confidence": f"{mapping.confidence:.0%}",
        }
        for mapping in mappings
    ]



def _parse_uploaded_workbook(
    file_bytes: bytes,
    *,
    profile_id: str | None,
    override_id: str | None,
    elevation_m: float | None,
    target_crs: str | None,
    auto_assign_unit_order: bool,
) -> tuple[ParseResult, ImportReport]:
    return cached_ingest_workbook(
        file_bytes,
        profile_id,
        override_id,
        elevation_m,
        target_crs,
        json.dumps(_get_aliases(), sort_keys=True),
        auto_assign_unit_order,
    )


def _generate_cross_section(
    parse_result: ParseResult,
    transect_points: list[tuple[float, float]],
    hole_ids: tuple[str, ...],
    request: SectionBuildRequest,
    offset_warning_m: float,
    *,
    lithology_index: dict[str, tuple[Lithology, ...]] | None = None,
) -> tuple[bytes, bytes, bytes, int, list[str], list[str]]:
    from app_build import generate_cross_section

    return generate_cross_section(
        parse_result,
        transect_points,
        hole_ids,
        request,
        offset_warning_m,
        lithology_index=lithology_index,
    )


def _apply_pending_offset_thresholds() -> None:
    """Apply suggested offset thresholds before sidebar widgets bind session keys."""
    if not st.session_state.pop("_apply_suggested_offset", False):
        return
    suggested = float(st.session_state.suggested_offset_m)
    st.session_state.offset_warning_m = suggested
    st.session_state.uncertainty_offset_m = suggested


