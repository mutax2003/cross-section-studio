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
    _report_context_from_selection,
    _sidebar_heading,
)
from constants import USGS_LITHOLOGY_HATCHES, get_lithology_style, save_lithology_style_override
from ingestion import NATIVE_PROFILE_ID, list_profiles
from models import ConsultingTitleBlock
from pipeline import DEFAULT_UNCERTAINTY_SPACING_M


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
    _sidebar_heading("Data source")
    uploaded = st.file_uploader(
        "Borehole database (.xlsx)",
        type=["xlsx"],
        help="Native Collars/Lithology workbook or field export with Lat/Long",
    )
    if uploaded is not None:
        st.session_state.uploaded_name = uploaded.name

    st.divider()
    _sidebar_heading("Cross-section style")
    interpretation_mode = st.radio(
        "Interpretation",
        options=["interpolated", "correlation_lines", "borehole_only"],
        format_func=lambda value: {
            "interpolated": "Interpolated fence diagram",
            "correlation_lines": "Contact lines only (no fill)",
            "borehole_only": "Observed only (no inter-hole fill)",
        }[value],
        help="Observed only shows logged intervals on borehole sticks without correlation.",
    )
    report_preset = st.toggle(
        "Consulting report preset",
        value=False,
        help="Section-sheet layout with sky fill, ground surface, wide tracks, and no pinch-outs.",
    )
    render_layout = st.radio(
        "Layout",
        options=["section_sheet", "consulting_section", "chart"],
        format_func=lambda value: {
            "section_sheet": "Section sheet (Strater-style)",
            "consulting_section": "Consulting section (fence + title block)",
            "chart": "Chart (legacy debug)",
        }[value],
        index=0,
        disabled=report_preset,
        help="Consulting section uses fence fills, report grid, and a footer title block.",
    )
    is_consulting_layout = render_layout == "consulting_section" and not report_preset
    allow_pinch_outs = st.toggle(
        "Allow pinch-out wedges",
        value=not report_preset,
        disabled=interpretation_mode == "borehole_only",
        help="When off, units present in only one hole are not inferred between holes.",
    )
    show_ground_surface = st.toggle(
        "Show ground surface (collar RL)",
        value=True,
        disabled=report_preset,
        help="Linear interpolation between collar elevations — not a DEM.",
    )
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
        value=is_consulting_layout,
        disabled=is_consulting_layout,
        help="When off, only measured water levels are shown as points.",
    )
    show_hatches = st.toggle(
        "Hatch patterns",
        value=True,
        help="USGS-style hatch patterns on lithology fills (all layouts).",
    )
    show_legend = st.toggle(
        "Legend on chart",
        value=not is_consulting_layout,
        disabled=is_consulting_layout,
        help="Consulting layout places the legend in the footer title block.",
    )
    with st.expander("Fill style editor", expanded=False):
        st.caption("Override USGS fill color and hatch for a lithology code (saved locally).")
        style_codes = sorted(st.session_state.get("unique_lithology_codes") or [])
        if style_codes:
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
                st.success(f"Saved style for {style_code}. Regenerate to preview.")
        else:
            st.info("Upload and validate data to edit lithology fill styles.")
    section_title = st.text_input("Section title", value="Borehole Cross-Section")
    consulting_title_block: ConsultingTitleBlock | None = None
    if is_consulting_layout:
        consulting_title_block = _render_consulting_report_sheet(section_title)
    vertical_exaggeration = st.slider(
        "Vertical exaggeration",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        key="vertical_exaggeration",
    )

    st.divider()
    _sidebar_heading("Transect thresholds")
    if st.session_state.pop("pending_transect_mode", None):
        st.session_state.transect_definition_mode = "By hole sequence"
    transect_mode = st.radio(
        "Definition mode",
        options=["By hole sequence", "By coordinates", "Recommended"],
        label_visibility="collapsed",
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
    max_offset_for_interpolation_m = st.number_input(
        "Max offset for interpolation (m)",
        min_value=1.0,
        step=5.0,
        value=float(st.session_state.offset_warning_m),
        help="Holes farther than this are excluded from inter-hole correlation polygons.",
    )

    st.divider()
    with st.expander("Interpretation uncertainty", expanded=False):
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

    selected_profile_key, override_id, default_elevation_m, target_crs = _render_import_settings()

    with st.expander("AI Assist (optional)", expanded=False):
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API key",
            type="password",
            help="Headers-only column mapping and QA narratives. Leave blank for local checks only.",
        )
        st.checkbox(
            "Enable LLM suggestions",
            value=False,
            key="enable_ai_suggestions",
            help="Uses OpenAI when an API key is set. Local rule-based suggestions always work.",
        )

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


def _render_import_settings() -> tuple[str, str | None, float, str | None]:
    with st.expander("Workbook import", expanded=False):
        profile_options = {profile.id: profile.label for profile in list_profiles()}
        profile_options[NATIVE_PROFILE_ID] = "Native platform (Collars + Lithology)"
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
            "Auto-assign unit_order from depth on import",
            value=bool(st.session_state.get("auto_assign_unit_order", True)),
            help="Assigns stratigraphic order 1..n per hole when duplicate lithology codes lack unit_order.",
        )
    return selected_profile_key, override_id, default_elevation_m, target_crs


def _render_consulting_report_sheet(section_title: str) -> ConsultingTitleBlock:
    with st.expander("Report sheet (consulting)", expanded=True):
        consulting_section_label = st.text_input(
            "Section label",
            value=section_title,
            key="consulting_section_label",
        )
        transect_start_label = st.text_input("Transect start label", value="", key="consulting_start_label")
        transect_end_label = st.text_input("Transect end label", value="", key="consulting_end_label")
        transect_cols = st.columns(2)
        with transect_cols[0]:
            transect_start_primary = st.text_input(
                "Start primary (e.g. B)",
                value="",
                key="consulting_start_primary",
            )
            transect_start_secondary = st.text_input(
                "Start secondary (e.g. SOUTHWEST)",
                value="",
                key="consulting_start_secondary",
            )
        with transect_cols[1]:
            transect_end_primary = st.text_input(
                "End primary (e.g. B')",
                value="",
                key="consulting_end_primary",
            )
            transect_end_secondary = st.text_input(
                "End secondary (e.g. NORTHEAST)",
                value="",
                key="consulting_end_secondary",
            )
        map_scale = st.text_input("Map scale", value="1:1000", key="consulting_map_scale")
        figure_number = st.text_input("Figure number", value="", key="consulting_figure_number")
        project_number = st.text_input("Project number", value="", key="consulting_project_number")
        source = st.text_input("Source", value="", key="consulting_source")
        report_date = st.text_input("Date", value="", key="consulting_date")
        drawn_by = st.text_input("Drawn by", value="", key="consulting_drawn_by")
        revised = st.text_input("Revised", value="", key="consulting_revised")
        prepared_for = st.text_input("Prepared for", value="", key="consulting_prepared_for")
        prepared_by = st.text_input("Prepared by", value="", key="consulting_prepared_by")
        default_notes = "\n".join(_default_consulting_notes())
        notes_text = st.text_area("Notes", value=default_notes, key="consulting_notes")
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
                if st.session_state.parse_result is not None:
                    context = _report_context_from_selection(
                        st.session_state.parse_result,
                        report_holes,
                        vertical_exaggeration=float(
                            st.session_state.get("vertical_exaggeration", 5.0)
                        ),
                        map_scale=map_scale,
                        section_title=section_title,
                    )
                else:
                    context = {
                        "hole_ids": report_holes,
                        "map_scale": map_scale,
                        "section_label": consulting_section_label or section_title,
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
