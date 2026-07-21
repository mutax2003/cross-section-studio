"""Shared Streamlit helpers (no page layout)."""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, Sequence

from dataclasses import dataclass, replace

import streamlit as st

from ai_assistant import AIAssistant, build_llm_provider

if TYPE_CHECKING:
    from ai_assistant import ReportMetadataSuggestion

from ai_quality import analyze_parsed_data, load_lithology_aliases
from app_services import cached_ingest_workbook
from app_state import clear_section_output_state
from constants import DEFAULT_LITHOLOGY_COLOR, USGS_LITHOLOGY_HATCHES, get_lithology_style, normalize_hex_colour
from ingestion import ImportReport
from models import (
    ConsultingTitleBlock,
    CorrelationOverride,
    Lithology,
    ParseResult,
    apply_unit_order_fix,
    lithologies_by_hole,
)
from section_build_request import SectionBuildRequest
from ui_helpers import (
    active_transect_selection,
    dedupe_messages,
    escape_html,
    legend_hatch_background,
    svg_display_meta,
)

logger = logging.getLogger(__name__)

_WORKFLOW_LABELS = ("Upload", "Validate", "Configure", "Generate")
_LLM_DISABLE_VALUES = frozenset({"1", "true", "yes", "on"})


def llm_disabled_by_deployment() -> bool:
    """True when CROSS_SECTION_DISABLE_LLM blocks third-party LLM calls."""
    return os.environ.get("CROSS_SECTION_DISABLE_LLM", "").strip().lower() in _LLM_DISABLE_VALUES


def llm_assist_status_caption() -> str:
    """One-line status for Validate / Configure AI actions."""
    if llm_disabled_by_deployment():
        return "Assist mode: **LLM disabled by deployment** — local rules only."
    if st.session_state.get("enable_ai_suggestions"):
        provider = str(st.session_state.get("llm_provider", "groq"))
        tier = "free" if provider in {"groq", "gemini"} else "paid"
        return (
            f"Assist mode: **LLM enabled** (`{provider}`, {tier}) — "
            "local rules + provider polish."
        )
    return (
        "Assist mode: **Local rules** — set `GROQ_API_KEY` or `GEMINI_API_KEY` "
        "(or enable LLM in the sidebar) for free-tier polish."
    )


def llm_suggestions_available() -> bool:
    """True when column-mapping LLM suggestions can run (toggle + key + not disabled)."""
    if llm_disabled_by_deployment():
        return False
    if not st.session_state.get("enable_ai_suggestions"):
        return False
    provider_kind = str(st.session_state.get("llm_provider", "groq"))
    return bool(_llm_api_key_for_provider(provider_kind))


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
    compact = stage >= 1
    hero_class = "app-hero compact" if compact else "app-hero"
    tagline = (
        ""
        if compact
        else "<p>Upload · validate · configure · export publication-ready borehole profiles.</p>"
    )
    st.markdown(
        f"""
<div class="{hero_class}">
  <h1>Cross Section Studio</h1>
  {tagline}
</div>
""",
        unsafe_allow_html=True,
    )
    _render_workflow_stepper(stage)


def _render_sticky_generate_strip(
    *,
    has_svg: bool,
    can_generate: bool,
    is_stale: bool,
    section_title: str,
) -> None:
    """Thin action bar under hero when a section exists or parse is ready."""
    if has_svg:
        status = (
            f"<strong>{escape_html(section_title)}</strong> — "
            + ("settings changed — regenerate before export" if is_stale else "profile ready")
        )
    else:
        status = f"<strong>{escape_html(section_title)}</strong> — configure transect, then generate"
    col_status, col_action = st.columns([4, 1])
    with col_status:
        st.markdown(
            f'<div class="generate-strip"><span class="strip-status">{status}</span></div>',
            unsafe_allow_html=True,
        )
    with col_action:
        if has_svg:
            if st.button(
                "Regenerate",
                type="primary",
                disabled=not can_generate,
                key="sticky_regenerate",
                width="stretch",
            ):
                st.session_state["_regenerate_requested"] = True
                st.rerun()
    if has_svg and not can_generate:
        st.caption("Open **Setup — Validate & Configure** to resolve blocking issues.")


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
    preset_label: str | None = None,
    render_layout: str | None = None,
    transect_label: str | None = None,
    png_ready: bool = False,
    pdf_ready: bool = False,
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
    if preset_label:
        chips.append(f'<span class="chip">{escape_html(preset_label)}</span>')
    if render_layout and render_layout != "section_sheet":
        layout_short = {
            "consulting_section": "Consulting sheet",
            "chart": "Chart",
        }.get(render_layout, render_layout.replace("_", " "))
        chips.append(f'<span class="chip">{escape_html(layout_short)}</span>')
    if hole_count is not None:
        chips.append(f'<span class="chip">{escape_html(hole_count)} boreholes</span>')
    if transect_label:
        chips.append(f'<span class="chip">{escape_html(transect_label)}</span>')
    if polygon_count is not None and interpretation_mode == "interpolated":
        chips.append(f'<span class="chip">{escape_html(polygon_count)} polygons</span>')
    freshness = "Stale" if is_stale else "Fresh"
    chips.append(
        f'<span class="chip {"warn" if is_stale else ""}">{freshness}</span>'
    )
    export_bits = [f"PNG {'✓' if png_ready else '—'}", f"PDF {'✓' if pdf_ready else '—'}"]
    chips.append(f'<span class="chip">{" · ".join(export_bits)}</span>')
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
        hatch = style.hatch or USGS_LITHOLOGY_HATCHES.get(code, "..")
        hatch_bg = legend_hatch_background(hatch)
        color = normalize_hex_colour(style.color) or DEFAULT_LITHOLOGY_COLOR
        rows.append(
            f'<div class="legend-row">'
            f'<span class="legend-swatch" style="background-color:{color};'
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


def safe_lithology_index(
    parse_result: ParseResult | None = None,
) -> dict[str, tuple[Lithology, ...]] | None:
    """Return a usable lithology index from session, or rebuild from parse_result.

    Streamlit may rehydrate index values as plain dicts after deploy; prefer rebuilding.
    """
    index = st.session_state.get("lithology_index")
    if isinstance(index, dict) and index:
        sample = next(iter(index.values()), ())
        first = sample[0] if sample else None
        if first is None or isinstance(first, Lithology):
            return index  # type: ignore[return-value]
    if parse_result is not None:
        rebuilt = lithologies_by_hole(parse_result.lithologies)
        st.session_state.lithology_index = rebuilt
        return rebuilt
    return None


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


def _health_status_label(error_count: int, warning_count: int) -> str:
    if error_count > 0:
        return "Needs fixes"
    if warning_count > 0:
        return "Review warnings"
    return "Ready"


def _llm_api_key_for_provider(provider_kind: str) -> str:
    """Resolve API key from ephemeral UI input, Streamlit secrets, or environment.

    Never reads durable ``llm_api_key`` / ``openai_api_key`` session keys.
    """
    runtime = str(
        st.session_state.get(f"_llm_api_key_runtime_{provider_kind}")
        or st.session_state.get("_llm_api_key_runtime")
        or ""
    ).strip()
    if runtime:
        return runtime
    try:
        secrets = getattr(st, "secrets", None)
        if secrets is not None:
            for secret_key in (
                f"{provider_kind}_api_key",
                f"{str(provider_kind).upper()}_API_KEY",
                "GROQ_API_KEY",
                "GEMINI_API_KEY",
                "OPENAI_API_KEY",
            ):
                try:
                    value = str(secrets.get(secret_key, "") or "").strip()
                except FileNotFoundError:
                    value = ""
                except (AttributeError, KeyError, TypeError) as exc:
                    logger.warning("LLM secret %s unreadable: %s", secret_key, exc)
                    value = ""
                if value:
                    return value
    except FileNotFoundError:
        pass
    except (AttributeError, KeyError, TypeError) as exc:
        logger.warning("Streamlit secrets unavailable for LLM API key: %s", exc)
    from ai_assistant import resolve_llm_api_key

    return resolve_llm_api_key(provider_kind, None)


def _build_assistant() -> AIAssistant:
    """Return LLM-backed assistant when enabled and an API key is set."""
    if llm_disabled_by_deployment() or not st.session_state.get("enable_ai_suggestions"):
        return AIAssistant(None)
    provider_kind = st.session_state.get("llm_provider", "groq")
    api_key = _llm_api_key_for_provider(str(provider_kind))
    provider = build_llm_provider(provider_kind, api_key or None)
    return AIAssistant(provider)


def _default_consulting_notes() -> tuple[str, ...]:
    """Lazy import avoids Streamlit hot-reload ImportError on partial modules."""
    from render_theme import DEFAULT_CONSULTING_NOTES

    return DEFAULT_CONSULTING_NOTES


def _apply_report_suggestion(suggestion: ReportMetadataSuggestion) -> None:
    """Queue AI report fields for the next sidebar render (widget-safe)."""
    notes_lines = [str(note).strip() for note in suggestion.notes if str(note).strip()]
    caption = (suggestion.figure_caption or "").strip()
    if caption and not any(
        caption.casefold() == note.casefold() or caption.casefold() in note.casefold()
        for note in notes_lines
    ):
        notes_lines = [caption, *notes_lines]
    pending = dict(st.session_state.get("_pending_project_seed") or {})
    pending.update(
        {
            "consulting_section_label": suggestion.section_label,
            "consulting_map_scale": suggestion.map_scale,
            "consulting_notes": "\n".join(notes_lines),
            "consulting_prepared_for": suggestion.prepared_for,
            "consulting_prepared_by": suggestion.prepared_by,
            "consulting_source": suggestion.source,
            "consulting_project_number": suggestion.project_number,
            "consulting_start_label": suggestion.transect_start_label,
            "consulting_end_label": suggestion.transect_end_label,
            "section_title": suggestion.section_label,
        }
    )
    st.session_state["_pending_project_seed"] = pending
    st.session_state.ai_figure_caption = suggestion.figure_caption or None


def _water_and_nm(
    parse_result: ParseResult,
    hole_ids: Sequence[str],
) -> tuple[dict[str, dict[str, float]], list[str]]:
    hole_set = set(hole_ids)
    water: dict[str, dict[str, float]] = {}
    for level in parse_result.water_levels:
        if level.hole_id not in hole_set:
            continue
        series_id = level.series_id or "default"
        water.setdefault(level.hole_id, {})[series_id] = level.depth
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
        "water_measurement_count": sum(len(series) for series in water.values()),
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


