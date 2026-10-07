"""Shared Streamlit helpers (no page layout)."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import streamlit as st

from ai_assistant import AIAssistant, build_llm_provider

if TYPE_CHECKING:
    from ai_assistant import ReportMetadataSuggestion

from ai_quality import analyze_parsed_data, load_lithology_aliases
from app_services import cached_ingest_workbook
from app_state import clear_section_output_state
from constants import (
    DEFAULT_LITHOLOGY_COLOR,
    get_lithology_style,
    normalize_hex_colour,
)
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
    PREVIEW_ZOOM_OPTIONS,
    active_transect_selection,
    dedupe_messages,
    escape_html,
    legend_hatch_background,
    preview_img_style,
    svg_display_meta,
)

logger = logging.getLogger(__name__)

_WORKFLOW_LABELS = ("Upload", "Validate", "Configure", "Generate")
_LLM_DISABLE_VALUES = frozenset({"1", "true", "yes", "on"})


def llm_disabled_by_deployment() -> bool:
    """True when CROSS_SECTION_DISABLE_LLM blocks third-party LLM calls."""
    return os.environ.get("CROSS_SECTION_DISABLE_LLM", "").strip().lower() in _LLM_DISABLE_VALUES


def llm_assist_status_caption() -> str:
    """One-line status for Validate / Configure AI actions.

    Empty unless an AI provider is switched on: users on local rules do not
    need to read about API keys they have not asked for.
    """
    if llm_disabled_by_deployment():
        return ""
    if st.session_state.get("enable_ai_suggestions"):
        provider = str(st.session_state.get("llm_provider", "groq"))
        provider_name = {"groq": "Groq", "gemini": "Gemini", "openai": "OpenAI"}.get(
            provider, provider.title()
        )
        tier = "free tier" if provider in {"groq", "gemini"} else "paid"
        return f"AI suggestions on ({provider_name}, {tier}) — local checks plus AI wording."
    return ""


def llm_suggestions_available() -> bool:
    """True when column-mapping LLM suggestions can run (toggle + key + not disabled)."""
    if llm_disabled_by_deployment():
        return False
    if not st.session_state.get("enable_ai_suggestions"):
        return False
    provider_kind = str(st.session_state.get("llm_provider", "groq"))
    return bool(_llm_api_key_for_provider(provider_kind))


def _workflow_stepper_html(stage: int) -> str:
    steps_html = []
    for index, label in enumerate(_WORKFLOW_LABELS):
        if index < stage:
            css_class = "workflow-step done"
            prefix = "✓ "
        elif index == stage:
            css_class = "workflow-step active"
            prefix = f"{index + 1}. "
        else:
            css_class = "workflow-step"
            prefix = f"{index + 1}. "
        steps_html.append(
            f'<div class="{css_class}" role="listitem" aria-current="{"step" if index == stage else "false"}">'
            f"{prefix}{escape_html(label)}</div>"
        )
    return f'<div class="workflow" role="list">{"".join(steps_html)}</div>'


def _render_workflow_stepper(stage: int) -> None:
    st.markdown(_workflow_stepper_html(stage), unsafe_allow_html=True)


def _render_hero(stage: int) -> None:
    """Brand header with the workflow stepper.

    Once a section exists (stage 3) the hero collapses to a single line so the
    figure sits above the fold at 1280×720.
    """
    if stage >= 3:
        hero_class = "app-hero compact oneline"
    elif stage >= 1:
        hero_class = "app-hero compact"
    else:
        hero_class = "app-hero"
    tagline = (
        ""
        if stage >= 1
        else "<p>Upload · validate · configure · export publication-ready borehole profiles.</p>"
    )
    st.markdown(
        f"""
<div class="{hero_class}">
  <h1>Cross Section Studio</h1>
  {tagline}
  {_workflow_stepper_html(stage)}
</div>
""",
        unsafe_allow_html=True,
    )


def generate_strip_status_html(
    *,
    section_title: str,
    has_svg: bool,
    can_generate: bool,
    is_stale: bool,
    blocked_reason: str | None = None,
) -> str:
    """Figure heading plus a one-line, plain-language status for the action bar."""
    reason = (blocked_reason or "").strip().rstrip(".")
    if not has_svg:
        status = "Choose a section line, then generate."
    elif is_stale and not can_generate:
        status = (
            f"Out of date — to update the figure, {escape_html(reason or 'fix the issues under Setup')}."
        )
    elif is_stale:
        status = "Out of date — settings changed. Generate section to update."
    elif not can_generate and reason:
        status = f"Up to date. To generate again, {escape_html(reason)}."
    else:
        status = "Up to date."
    tone = " is-stale" if has_svg and is_stale else ""
    return (
        f'<div class="generate-strip{tone}" role="status">'
        f'<h2 class="strip-title">{escape_html(section_title)}</h2>'
        f'<span class="strip-status">{status}</span></div>'
    )


def _render_sticky_generate_strip(
    *,
    has_svg: bool,
    can_generate: bool,
    is_stale: bool,
    section_title: str,
    blocked_reason: str | None = None,
) -> None:
    """Action bar under the hero: figure title, status and the one Generate button."""
    col_status, col_action = st.columns([3, 1.2], vertical_alignment="center")
    with col_status:
        st.markdown(
            generate_strip_status_html(
                section_title=section_title,
                has_svg=has_svg,
                can_generate=can_generate,
                is_stale=is_stale,
                blocked_reason=blocked_reason,
            ),
            unsafe_allow_html=True,
        )
    with col_action:
        if has_svg:
            # The only Generate button once a figure exists: primary when the
            # figure is out of date, disabled (reason in the status) when blocked.
            if st.button(
                "Generate section",
                type="primary" if is_stale else "secondary",
                disabled=not can_generate,
                key="generate_section_strip",
                width="stretch",
                help=None if can_generate else (blocked_reason or None),
            ):
                st.session_state["_regenerate_requested"] = True
                st.rerun()


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
        f'<div class="metric-card {tone}">'
        f'<div class="value">{escape_html(value)}</div>'
        f'<div class="label">{escape_html(label)}</div></div>',
        unsafe_allow_html=True,
    )


def profile_chips_html(
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
) -> str:
    """At most three chips (style, VE, holes); the rest sit in a "+N more" tooltip."""
    mode_label = {
        "borehole_only": "Observed only",
        "correlation_lines": "Contact lines only",
    }.get(interpretation_mode, "Interpolated fence")
    chips = [
        f'<span class="chip brand">{escape_html(preset_label or mode_label)}</span>',
        f'<span class="chip">VE {escape_html(f"{vertical_exaggeration:g}")}×</span>',
    ]
    if hole_count is not None:
        chips.append(f'<span class="chip">{escape_html(hole_count)} holes</span>')
    extra: list[str] = []
    if preset_label:
        extra.append(mode_label)
    if render_layout and render_layout != "section_sheet":
        extra.append(
            {"consulting_section": "Consulting sheet", "chart": "Chart"}.get(
                render_layout, render_layout.replace("_", " ")
            )
        )
    if transect_label:
        extra.append(f"Section line {transect_label}")
    if polygon_count is not None and interpretation_mode == "interpolated":
        extra.append(f"{polygon_count} polygons")
    extra.append("Out of date" if is_stale else "Up to date")
    extra.append("PNG and PDF ready" if png_ready and pdf_ready else "PNG and PDF not prepared yet")
    details = escape_html(" · ".join(extra))
    chips.append(
        f'<span class="chip more" tabindex="0" title="{details}" '
        f'aria-label="More details: {details}">+{len(extra)} more</span>'
    )
    return f'<div class="profile-header">{"".join(chips)}</div>'


def _render_profile_chips(**kwargs: Any) -> None:
    st.markdown(profile_chips_html(**kwargs), unsafe_allow_html=True)


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
    show_hatches = bool(st.session_state.get("show_hatches", True))
    for code in sorted(codes):
        style = get_lithology_style(code)
        # Match the figure: hatches off in the sidebar means plain swatches here too.
        hatch_bg = legend_hatch_background(style.hatch if show_hatches else "")
        color = normalize_hex_colour(style.color) or DEFAULT_LITHOLOGY_COLOR
        rows.append(
            f'<div class="legend-row">'
            f'<span class="legend-swatch" style="background-color:{color};'
            f'background-image:{hatch_bg};"></span>'
            f"<strong>{escape_html(code)}</strong>"
            f"</div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)



@st.cache_data(show_spinner=False)
def _full_legend_png() -> bytes:
    from lithology_legend_sheet import build_legend_sheet_png

    return build_legend_sheet_png()


def render_full_lithology_legend() -> None:
    """Every palette code with its colour and pattern, as drawn in figures."""
    with st.expander("Full lithology legend (all codes)", expanded=False):
        png = _full_legend_png()
        st.image(
            png,
            caption="Colours and patterns follow the client legend template (261002).",
            width="stretch",
        )
        st.download_button(
            "Download legend (PNG)",
            data=png,
            file_name="lithology_legend.png",
            mime="image/png",
            key="download_full_lithology_legend",
        )

def _display_svg(
    svg_bytes: bytes,
    alt_text: str = "Cross-section profile",
    *,
    zoom_slot: Any = None,
) -> None:
    """Render SVG in Streamlit (st.image does not support SVG via PIL).

    ``zoom_slot`` (a container/column) lets the caller place the compact zoom
    control beside other controls instead of on its own row.
    """
    cached = st.session_state.get("svg_display_meta")
    if cached is None or st.session_state.svg_bytes != svg_bytes:
        cached = svg_display_meta(svg_bytes)
        st.session_state.svg_display_meta = cached
    if not cached.valid:
        st.error(
            "The figure came out empty. Check the section line has at least two holes "
            "with logs, then generate again."
        )
        return
    # Without a readable natural width every choice would render fit-to-width.
    zoom = None
    if cached.natural_width_px:
        with zoom_slot if zoom_slot is not None else st.container():
            zoom = st.segmented_control(
                "Preview size",
                list(PREVIEW_ZOOM_OPTIONS),
                default="Fit width",
                key="svg_preview_zoom",
                label_visibility="collapsed",
                help="Fit width shows the whole sheet; 100% / 150% show the figure at "
                "native size or larger so small labels can be checked before export.",
            )
    img_style, zoomed = preview_img_style(zoom, cached.natural_width_px)
    frame_attrs = (
        'class="svg-frame svg-frame--zoomed" tabindex="0" '
        'role="region" aria-label="Zoomed figure preview, scrollable"'
        if zoomed
        else 'class="svg-frame"'
    )
    # One markdown block: Streamlit auto-closes a lone <div>, so splitting this
    # across calls renders an empty bordered frame with the image outside it.
    st.markdown(
        f"<div {frame_attrs}>"
        f'<img src="data:image/svg+xml;base64,{cached.encoded}" '
        f'style="{img_style}" alt="{escape_html(alt_text)}" />'
        "</div>",
        unsafe_allow_html=True,
    )


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
    # Only a scale the user entered is passed on; left blank, the title block
    # prints "AS SHOWN" and defers to the true-scale bar.
    scale_kwargs = {"map_scale": map_scale.strip()} if map_scale.strip() else {}
    return ConsultingTitleBlock(
        **scale_kwargs,
        section_label=section_label or section_title,
        transect_start_label=transect_start_label.strip(),
        transect_end_label=transect_end_label.strip(),
        transect_start_primary=transect_start_primary.strip(),
        transect_start_secondary=transect_start_secondary.strip(),
        transect_end_primary=transect_end_primary.strip(),
        transect_end_secondary=transect_end_secondary.strip(),
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
            # Only this provider's key: a generic fallback once sent an
            # OPENAI_API_KEY from secrets to Groq as its bearer token.
            for secret_key in (
                f"{provider_kind}_api_key",
                f"{str(provider_kind).upper()}_API_KEY",
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


