"""Sidebar widgets and style controls."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from app_common import (
    _apply_pending_offset_thresholds,
    _apply_report_suggestion,
    _build_assistant,
    _build_consulting_title_block,
    _default_consulting_notes,
    _llm_api_key_for_provider,
    _report_context_from_selection,
    llm_disabled_by_deployment,
)
from ai_assistant import (
    DEFAULT_LLM_PROVIDER,
    is_free_llm_provider,
    preferred_llm_provider_from_env,
    resolve_llm_api_key,
)
from app_upload import (
    apply_pending_project_seed,
    clear_workbook_session,
    load_sample_workbook,
    render_input_template_download,
)
from constants import USGS_LITHOLOGY_HATCHES, get_lithology_style, save_lithology_style_override
from ingestion import DATA_ENTRY_PROFILE_ID, NATIVE_PROFILE_ID, list_profiles
from models import ConsultingTitleBlock
from pipeline import DEFAULT_UNCERTAINTY_SPACING_M
from ui_output_presets import OUTPUT_PRESET_LABELS, resolve_output_preset


@dataclass(frozen=True)
class SidebarState:
    uploaded: object | None
    interpretation_mode: str
    report_preset: bool
    render_layout: str
    is_consulting_layout: bool
    allow_pinch_outs: bool
    show_ground_surface: bool
    track_width_m: float
    interpolate_water_table: bool
    show_water_elevation_labels: bool
    show_water_legend: bool
    show_dry_well_nm: bool
    water_interpolate_across_gaps: bool
    parameter_interpolate_across_gaps: bool
    warn_on_correlation_gaps: bool
    show_hatches: bool
    show_legend: bool
    section_title: str
    consulting_title_block: ConsultingTitleBlock | None
    vertical_exaggeration: float
    transect_mode: str
    offset_warning_m: float
    max_offset_for_interpolation_m: float
    uncertainty_spacing_m: float
    uncertainty_offset_m: float
    selected_profile_key: str
    override_id: str | None
    default_elevation_m: float
    target_crs: str | None


def render_sidebar() -> SidebarState:
    apply_pending_project_seed()
    has_parsed = st.session_state.get("parse_result") is not None

    with st.expander("Data source", expanded=True):
        st.caption(
            "Enter geology in Excel (download the template), then upload here. "
            "Native Collars/Lithology workbooks and field exports with Lat/Long also work."
        )
        render_input_template_download(
            key="sidebar_download_input_template",
            help="Fill Collars + Lithology in Excel, then upload with the control below.",
        )
        uploaded_name = st.session_state.get("uploaded_name")
        if uploaded_name or st.session_state.get("file_bytes"):
            st.caption(f"Loaded: {uploaded_name or 'workbook.xlsx'}")
        action_cols = st.columns(2)
        with action_cols[0]:
            if st.button("Clear workbook", key="sidebar_clear_workbook"):
                clear_workbook_session()
                st.rerun()
        with action_cols[1]:
            if st.button("Try sample project", key="sidebar_try_sample"):
                try:
                    load_sample_workbook()
                    st.rerun()
                except FileNotFoundError as exc:
                    st.error(str(exc))
        uploaded = st.file_uploader(
            "Upload Excel workbook",
            type=["xlsx"],
            key=f"workbook_uploader_{st.session_state.get('workbook_uploader_key', 0)}",
            help=(
                "Upload a filled template, native Collars/Lithology workbook, "
                "or field export with Lat/Long."
            ),
        )
        if uploaded is not None:
            st.session_state.uploaded_name = uploaded.name

        selected_profile_key, override_id, default_elevation_m, target_crs = _render_import_settings(
            expanded=bool(uploaded) or bool(st.session_state.get("file_bytes")),
        )

    with st.expander("Section output", expanded=has_parsed):
        output_preset = st.selectbox(
            "Output style",
            options=tuple(OUTPUT_PRESET_LABELS.keys()),
            format_func=lambda key: OUTPUT_PRESET_LABELS[key],
            key="output_preset",
            help="Consulting report includes footer title block and groundwater legend.",
        )
        preset_config = resolve_output_preset(output_preset)
        render_layout = preset_config.render_layout
        report_preset = preset_config.report_preset
        is_consulting_layout = render_layout == "consulting_section"
        if st.session_state.get("_synced_output_preset") != output_preset:
            st.session_state.allow_pinch_outs = preset_config.allow_pinch_outs
            st.session_state.show_ground_surface = preset_config.show_ground_surface
            st.session_state.show_legend = preset_config.show_legend
            st.session_state._synced_output_preset = output_preset

        interpretation_mode = st.radio(
            "Interpretation",
            options=["interpolated", "correlation_lines", "borehole_only"],
            format_func=lambda value: {
                "interpolated": "Connect layers between holes",
                "correlation_lines": "Contact lines only (no shading)",
                "borehole_only": "Observed logs only (no inter-hole fill)",
            }[value],
            help=(
                "Interpolated: correlate lithology between boreholes. "
                "Contact lines: fence contacts without fill. "
                "Observed only: stick logs without correlation."
            ),
        )
        allow_pinch_outs = st.toggle(
            "Show layers that thin out between holes",
            key="allow_pinch_outs",
            disabled=interpretation_mode == "borehole_only",
            help="When off, units logged in only one hole are not inferred across the section (pinch-outs).",
        )
        show_ground_surface = st.toggle(
            "Show ground surface (collar RL)",
            key="show_ground_surface",
            disabled=report_preset,
            help="Linear interpolation between collar elevations — not a DEM.",
        )
        show_hatches = st.toggle(
            "Hatch patterns",
            key="show_hatches",
            help="USGS-style hatch patterns on lithology fills.",
        )
        if "section_title" not in st.session_state:
            st.session_state.section_title = "Borehole Cross-Section"
        section_title = st.text_input("Section title", key="section_title")
        vertical_exaggeration = st.slider(
            "Vertical exaggeration",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            key="vertical_exaggeration",
        )

        st.markdown("**Transect thresholds**")
        if st.session_state.pop("pending_transect_mode", None):
            st.session_state.transect_definition_mode = "By hole sequence"
        transect_mode = st.radio(
            "Transect definition mode",
            options=["By hole sequence", "By coordinates", "Recommended"],
            key="transect_definition_mode",
        )
        _apply_pending_offset_thresholds()
        offset_warning_m = st.number_input(
            "Transect offset warning (m)",
            min_value=1.0,
            step=5.0,
            key="offset_warning_m",
            help="Warn when a selected borehole is farther than this from the transect line",
        )

    track_width_m = 3.0
    interpolate_water_table = preset_config.interpolate_water_table
    show_water_elevation_labels = is_consulting_layout
    show_water_legend = is_consulting_layout
    show_dry_well_nm = is_consulting_layout
    water_interpolate_across_gaps = False
    parameter_interpolate_across_gaps = False
    warn_on_correlation_gaps = False
    show_legend = preset_config.show_legend
    max_offset_for_interpolation_m = float(st.session_state.offset_warning_m)
    uncertainty_spacing_m = float(DEFAULT_UNCERTAINTY_SPACING_M)
    uncertainty_offset_m = float(st.session_state.get("uncertainty_offset_m", 50.0))
    selected_profile_key = "auto"
    override_id: str | None = None
    default_elevation_m = 100.0
    target_crs: str | None = "EPSG:32611"
    consulting_title_block: ConsultingTitleBlock | None = None

    with st.expander("Advanced", expanded=False):
        track_width_m = st.slider(
            "Track width (m)",
            min_value=1.5,
            max_value=6.0,
            value=3.0,
            step=0.5,
            disabled=report_preset or is_consulting_layout,
            help="Width of each borehole column on the section profile.",
        )
        interpolate_water_table = st.toggle(
            "Interpolate water table between holes",
            value=interpolate_water_table,
            disabled=is_consulting_layout,
            help="When off, only measured water levels are shown as points.",
        )
        show_water_elevation_labels = st.toggle(
            "Show water elevation labels",
            value=show_water_elevation_labels,
            disabled=is_consulting_layout,
        )
        show_water_legend = st.toggle(
            "Show groundwater legend",
            value=show_water_legend,
            disabled=is_consulting_layout,
        )
        show_dry_well_nm = st.toggle(
            "Show dry-well NM markers",
            value=show_dry_well_nm,
            disabled=is_consulting_layout,
        )
        water_interpolate_across_gaps = st.toggle(
            "Interpolate water across gaps",
            value=water_interpolate_across_gaps,
            disabled=is_consulting_layout,
            help="When off, dashed lines connect only consecutive measured holes.",
        )
        parameter_interpolate_across_gaps = st.toggle(
            "Interpolate parameters across gaps",
            value=parameter_interpolate_across_gaps,
            help="When off, parameter fence lines connect only consecutive holes with readings.",
        )
        warn_on_correlation_gaps = st.toggle(
            "Warn on correlation gaps",
            value=warn_on_correlation_gaps,
            disabled=interpretation_mode == "borehole_only",
            help="Add correlation gap notes to QA warnings when units do not match between holes.",
        )
        show_legend = st.toggle(
            "Legend on chart",
            key="show_legend",
            disabled=is_consulting_layout,
            help="Consulting layout places the legend in the footer title block.",
        )
        max_offset_for_interpolation_m = st.number_input(
            "Max offset for interpolation (m)",
            min_value=1.0,
            step=5.0,
            value=float(st.session_state.offset_warning_m),
            help="Holes farther than this are excluded from inter-hole correlation polygons.",
        )
        uncertainty_spacing_m = st.number_input(
            "Uncertain zone spacing (m)",
            min_value=10.0,
            value=float(DEFAULT_UNCERTAINTY_SPACING_M),
            step=10.0,
            disabled=interpretation_mode == "borehole_only",
            help="Shade inter-hole zones when adjacent boreholes are farther apart than this.",
        )
        uncertainty_offset_m = st.number_input(
            "Uncertain zone offset (m)",
            min_value=1.0,
            step=5.0,
            key="uncertainty_offset_m",
            disabled=interpretation_mode == "borehole_only",
            help="Shade inter-hole zones when either borehole exceeds this transect offset.",
        )
        if st.session_state.get("parse_result"):
            _render_fill_style_editor()

    with st.expander("AI Assist", expanded=False):
        _render_ai_assist()

    if is_consulting_layout:
        with st.expander("Consulting report sheet", expanded=True):
            consulting_title_block = _render_consulting_report_sheet(section_title)
    else:
        consulting_title_block = None

    return SidebarState(
        uploaded=uploaded,
        interpretation_mode=interpretation_mode,
        report_preset=report_preset,
        render_layout=render_layout,
        is_consulting_layout=is_consulting_layout,
        allow_pinch_outs=allow_pinch_outs,
        show_ground_surface=show_ground_surface,
        track_width_m=track_width_m,
        interpolate_water_table=interpolate_water_table,
        show_water_elevation_labels=show_water_elevation_labels,
        show_water_legend=show_water_legend,
        show_dry_well_nm=show_dry_well_nm,
        water_interpolate_across_gaps=water_interpolate_across_gaps,
        parameter_interpolate_across_gaps=parameter_interpolate_across_gaps,
        warn_on_correlation_gaps=warn_on_correlation_gaps,
        show_hatches=show_hatches,
        show_legend=show_legend,
        section_title=section_title,
        consulting_title_block=consulting_title_block,
        vertical_exaggeration=vertical_exaggeration,
        transect_mode=transect_mode,
        offset_warning_m=offset_warning_m,
        max_offset_for_interpolation_m=max_offset_for_interpolation_m,
        uncertainty_spacing_m=uncertainty_spacing_m,
        uncertainty_offset_m=uncertainty_offset_m,
        selected_profile_key=selected_profile_key,
        override_id=override_id,
        default_elevation_m=default_elevation_m,
        target_crs=target_crs,
    )


def _render_fill_style_editor() -> None:
    st.caption("Override USGS fill color and hatch for a lithology code (saved locally).")
    style_codes = sorted(st.session_state.get("unique_lithology_codes") or [])
    if not style_codes:
        st.info("Parse data to edit lithology fill styles.")
        return
    style_code = st.selectbox("Lithology code", options=style_codes, key="style_editor_code")
    current_style = get_lithology_style(style_code)
    style_color = st.color_picker("Fill color", value=current_style.color, key="style_editor_color")
    style_hatch_options = sorted(set(USGS_LITHOLOGY_HATCHES.values()))
    style_hatch = st.selectbox(
        "Hatch pattern",
        options=style_hatch_options,
        index=style_hatch_options.index(current_style.hatch)
        if current_style.hatch in style_hatch_options
        else 0,
        key="style_editor_hatch",
    )
    if st.button("Save fill style", key="save_fill_style"):
        save_lithology_style_override(style_code, style_color, style_hatch)
        st.success(f"Saved style for {style_code}. Use Generate Cross-Section to preview.")


def _seed_free_llm_defaults() -> None:
    """Prefer free-tier providers and auto-enable when an env key is present."""
    if llm_disabled_by_deployment():
        return
    if not st.session_state.get("_llm_provider_seeded"):
        detected = preferred_llm_provider_from_env()
        if detected is not None:
            st.session_state.llm_provider = detected
        elif "llm_provider" not in st.session_state:
            st.session_state.llm_provider = DEFAULT_LLM_PROVIDER
        st.session_state._llm_provider_seeded = True
    if not st.session_state.get("_llm_enable_seeded"):
        provider = str(st.session_state.get("llm_provider", DEFAULT_LLM_PROVIDER))
        if is_free_llm_provider(provider) and resolve_llm_api_key(provider, None):  # type: ignore[arg-type]
            st.session_state.enable_ai_suggestions = True
        st.session_state._llm_enable_seeded = True


def _render_ai_assist() -> None:
    if llm_disabled_by_deployment():
        st.caption("Third-party LLM assist is disabled for this deployment (`CROSS_SECTION_DISABLE_LLM`). Local rules still run.")
        st.session_state["enable_ai_suggestions"] = False
        return
    _seed_free_llm_defaults()
    # Drop any legacy durable session secrets from older builds.
    st.session_state.pop("llm_api_key", None)
    st.session_state.pop("openai_api_key", None)
    st.caption(
        "Free tier: **Groq** (console.groq.com) or **Gemini** (aistudio.google.com/apikey). "
        "Set `GROQ_API_KEY` or `GEMINI_API_KEY` to auto-enable."
    )
    st.selectbox(
        "LLM provider",
        options=("groq", "gemini", "openai"),
        format_func=lambda value: {
            "groq": "Groq — free tier (recommended)",
            "gemini": "Google Gemini — free tier",
            "openai": "OpenAI — paid",
        }[value],
        key="llm_provider",
        help="Groq and Gemini free API keys power QA narratives and mapping assist. OpenAI is optional/paid.",
    )
    provider = str(st.session_state.get("llm_provider", DEFAULT_LLM_PROVIDER))
    # Prefer env/secrets; only show the password widget when neither is set.
    resolved = _llm_api_key_for_provider(provider)
    runtime_key = f"_llm_api_key_runtime_{provider}"
    # Clear other providers' ephemeral keys so a switch does not reuse the wrong secret.
    for other in ("groq", "gemini", "openai"):
        if other != provider:
            st.session_state.pop(f"_llm_api_key_runtime_{other}", None)
    st.session_state.pop("_llm_api_key_runtime", None)

    env_only = resolve_llm_api_key(provider, None)  # type: ignore[arg-type]
    if env_only or (resolved and not st.session_state.get(runtime_key)):
        label = "free-tier" if is_free_llm_provider(provider) else "provider"
        st.caption(
            f"API key loaded from environment or Streamlit secrets ({label}). Not stored in session."
        )
        st.session_state.pop(runtime_key, None)
    else:
        entered = st.text_input(
            "API key",
            type="password",
            value="",
            help=(
                "Free keys: Groq at console.groq.com, Gemini at aistudio.google.com/apikey. "
                "Prefer GROQ_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY in the environment — "
                "widget values are kept only for this browser session run, not as durable session keys."
            ),
        )
        if entered.strip():
            st.session_state[runtime_key] = entered.strip()
            if is_free_llm_provider(provider) and not st.session_state.get("enable_ai_suggestions"):
                st.session_state.enable_ai_suggestions = True
        if st.session_state.get(runtime_key):
            st.caption("Key held in memory for this session only (not a durable Streamlit widget key).")
            if st.button("Clear API key", key="clear_llm_api_key_runtime"):
                st.session_state.pop(runtime_key, None)
                st.rerun()
    st.checkbox(
        "Enable LLM suggestions",
        value=False,
        key="enable_ai_suggestions",
        help="Uses the selected provider when an API key is set. Local checks always work. Free keys auto-enable once.",
    )


def _render_import_settings(*, expanded: bool = False) -> tuple[str, str | None, float, str | None]:
    with st.expander("Workbook import", expanded=expanded):
        profile_options = {profile.id: profile.label for profile in list_profiles()}
        profile_options[NATIVE_PROFILE_ID] = "Native platform (Collars + Lithology)"
        profile_options[DATA_ENTRY_PROFILE_ID] = "Cross Section input template (Data Entry)"
        auto_profile = st.session_state.get("detection_result")
        default_profile_index = 0
        profile_ids = ["auto"] + list(profile_options.keys())
        if auto_profile is not None and auto_profile.profile_id in profile_options:
            default_profile_index = profile_ids.index(auto_profile.profile_id)
        selected_profile_key = st.selectbox(
            "Import profile",
            options=profile_ids,
            format_func=lambda key: "Auto-detect" if key == "auto" else profile_options[key],
            index=min(default_profile_index, len(profile_ids) - 1),
        )
        override_id = st.text_input(
            "Profile override (optional)",
            value="advantage_phase2_2026" if auto_profile and not auto_profile.is_native else "",
            help="e.g. advantage_phase2_2026 for per-site coordinate offsets",
        ).strip() or None
        default_elevation_m = st.number_input(
            "Default collar elevation (m)",
            min_value=0.0,
            value=100.0,
            step=1.0,
            help="Used for field exports without RL/elevation column",
        )
        target_crs = st.text_input(
            "Target CRS (EPSG)",
            value="EPSG:32611",
            help="UTM zone for Lat/Long field exports (Alberta default: 32611)",
        ).strip() or None
        st.session_state.auto_assign_unit_order = st.checkbox(
            "Auto-assign layer order from depth on import",
            value=bool(st.session_state.get("auto_assign_unit_order", True)),
            help="Assigns stratigraphic order 1..n per hole when duplicate lithology codes lack unit_order.",
        )
        if st.session_state.get("parse_result") is not None:
            if st.button("Re-parse workbook", key="reparse_workbook"):
                st.session_state.parse_signature = None
                st.rerun()
    return selected_profile_key, override_id, default_elevation_m, target_crs


def _render_consulting_report_sheet(section_title: str) -> ConsultingTitleBlock:
    st.markdown("**Report sheet (consulting)**")
    # Init keyed widgets only when absent so Project metadata seeding is not overwritten.
    if "consulting_section_label" not in st.session_state:
        st.session_state.consulting_section_label = section_title or "Borehole Cross-Section"
    if "consulting_map_scale" not in st.session_state:
        st.session_state.consulting_map_scale = "1:1000"
    if "consulting_notes" not in st.session_state:
        st.session_state.consulting_notes = "\n".join(_default_consulting_notes())
    for key in (
        "consulting_start_label",
        "consulting_end_label",
        "consulting_start_primary",
        "consulting_start_secondary",
        "consulting_end_primary",
        "consulting_end_secondary",
        "consulting_figure_number",
        "consulting_project_number",
        "consulting_source",
        "consulting_date",
        "consulting_drawn_by",
        "consulting_revised",
        "consulting_prepared_for",
        "consulting_prepared_by",
    ):
        if key not in st.session_state:
            st.session_state[key] = ""

    consulting_section_label = st.text_input("Section label", key="consulting_section_label")
    transect_start_label = st.text_input("Transect start label", key="consulting_start_label")
    transect_end_label = st.text_input("Transect end label", key="consulting_end_label")
    transect_cols = st.columns(2)
    with transect_cols[0]:
        transect_start_primary = st.text_input("Start primary (e.g. B)", key="consulting_start_primary")
        transect_start_secondary = st.text_input(
            "Start secondary (e.g. SOUTHWEST)",
            key="consulting_start_secondary",
        )
    with transect_cols[1]:
        transect_end_primary = st.text_input("End primary (e.g. B')", key="consulting_end_primary")
        transect_end_secondary = st.text_input(
            "End secondary (e.g. NORTHEAST)",
            key="consulting_end_secondary",
        )
    map_scale = st.text_input("Map scale", key="consulting_map_scale")
    figure_number = st.text_input("Figure number", key="consulting_figure_number")
    project_number = st.text_input("Project number", key="consulting_project_number")
    source = st.text_input("Source", key="consulting_source")
    report_date = st.text_input("Date", key="consulting_date")
    drawn_by = st.text_input("Drawn by", key="consulting_drawn_by")
    revised = st.text_input("Revised", key="consulting_revised")
    prepared_for = st.text_input("Prepared for", key="consulting_prepared_for")
    prepared_by = st.text_input("Prepared by", key="consulting_prepared_by")
    notes_text = st.text_area("Notes", key="consulting_notes")
    logo_for = st.file_uploader(
        "Logo — prepared for (PNG)",
        type=["png"],
        key="consulting_logo_for",
    )
    logo_by = st.file_uploader(
        "Logo — prepared by (PNG)",
        type=["png"],
        key="consulting_logo_by",
    )
    st.caption(
        "Optional Excel sheets: **Screens** (hole_id, from_depth, to_depth) and "
        "**Gradients** (hole_id, direction)."
    )
    report_ai_cols = st.columns(2)
    with report_ai_cols[0]:
        if st.button("Suggest report fields", key="suggest_report_fields"):
            report_holes = list(st.session_state.get("hole_ids") or [])
            label_for_context = consulting_section_label or section_title
            if st.session_state.parse_result is not None:
                context = _report_context_from_selection(
                    st.session_state.parse_result,
                    report_holes,
                    vertical_exaggeration=float(
                        st.session_state.get("vertical_exaggeration", 5.0)
                    ),
                    map_scale=map_scale,
                    section_title=label_for_context,
                )
            else:
                context = {
                    "hole_ids": report_holes,
                    "map_scale": map_scale,
                    "section_label": label_for_context,
                    "workbook_name": st.session_state.get("uploaded_name", ""),
                    "vertical_exaggeration": 5.0,
                }
            st.session_state.ai_report_suggestion = (
                _build_assistant().suggest_report_metadata(context)
            )
    with report_ai_cols[1]:
        if st.session_state.get("ai_report_suggestion") and st.button(
            "Accept report suggestion",
            key="accept_report_fields",
        ):
            _apply_report_suggestion(st.session_state.ai_report_suggestion)
            st.rerun()
    if st.session_state.get("ai_report_suggestion"):
        preview = st.session_state.ai_report_suggestion
        st.caption(
            f"Suggested: **{preview.section_label}** · "
            f"{len(preview.notes)} note(s) · {preview.figure_caption}"
        )
    if st.session_state.get("ai_figure_caption"):
        st.caption(f"Figure caption: {st.session_state.ai_figure_caption}")
    return _build_consulting_title_block(
        section_title,
        section_label=consulting_section_label,
        transect_start_label=transect_start_label,
        transect_end_label=transect_end_label,
        transect_start_primary=transect_start_primary,
        transect_start_secondary=transect_start_secondary,
        transect_end_primary=transect_end_primary,
        transect_end_secondary=transect_end_secondary,
        map_scale=map_scale,
        figure_number=figure_number,
        project_number=project_number,
        source=source,
        date=report_date,
        notes_text=notes_text,
        drawn_by=drawn_by,
        revised=revised,
        prepared_for=prepared_for,
        prepared_by=prepared_by,
        logo_prepared_for_bytes=logo_for.getvalue() if logo_for else None,
        logo_prepared_by_bytes=logo_by.getvalue() if logo_by else None,
    )
