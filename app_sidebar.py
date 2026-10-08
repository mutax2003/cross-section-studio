"""Sidebar widgets and style controls."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from ai_assistant import (
    DEFAULT_LLM_PROVIDER,
    is_free_llm_provider,
    preferred_llm_provider_from_env,
    resolve_llm_api_key,
)
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
from app_identity import COPYRIGHT_SHORT, CREATED_BY
from app_upload import (
    DESTRUCTIVE_PROMPTS,
    apply_pending_project_seed,
    render_input_template_download,
    request_destructive,
    run_destructive,
)
from constants import (
    DEFAULT_PROFILE_ELEVATION_M,
    USGS_LITHOLOGY_HATCHES,
    clear_lithology_style_override,
    get_lithology_style,
    has_lithology_style_override,
    save_lithology_style_override,
)
from export_framing import ExportFramingConfig
from ingestion import DATA_ENTRY_PROFILE_ID, NATIVE_PROFILE_ID, list_profiles
from models import ConsultingTitleBlock
from pipeline import DEFAULT_UNCERTAINTY_SPACING_M
from render_theme import water_has_multiple_series
from ui_output_presets import (
    DEFAULT_VE_CHOICE,
    INTERPRETATION_LABELS,
    OUTPUT_PRESET_LABELS,
    locked_figure_summary,
    locked_groundwater_summary,
    normalize_ve_choice,
    output_preset_short_name,
    resolve_output_preset,
    sidebar_visibility,
    ve_choice_label,
    ve_choice_options,
    ve_choice_to_request,
)


def _render_pending_destructive() -> None:
    action = st.session_state.get("_pending_destructive")
    if action not in DESTRUCTIVE_PROMPTS:
        return
    if st.session_state.get("svg_bytes") is None:
        # The section the prompt protected is gone (new upload, clear, ...):
        # a leftover Confirm must never wipe whatever is generated next.
        st.session_state.pop("_pending_destructive", None)
        return
    confirm_label, question = DESTRUCTIVE_PROMPTS[action]
    st.warning(question + " Download anything you need first.")
    confirm_col, cancel_col = st.columns(2)
    with confirm_col:
        if st.button(confirm_label, key="confirm_destructive", type="primary", width="stretch"):
            st.session_state.pop("_pending_destructive", None)
            run_destructive(action)
    with cancel_col:
        if st.button("Cancel", key="cancel_destructive", width="stretch"):
            st.session_state.pop("_pending_destructive", None)
            st.rerun()

_OUTPUT_STYLE_HELP: dict[str, str] = {
    "section_sheet": "General-purpose sheet with an elevation axis, hole headers and a side legend. Good default.",
    "consulting_report": "Client figure with a title block, scale bar and notes along the bottom; water levels labelled.",
    "gwm_fence": "Elevation section with water levels, matching groundwater monitoring figures.",
    "p2_chemistry_sticks": "Hole columns by depth below ground with lab values (e.g. chloride) beside each hole.",
    "chemistry_gw": "Lab values and groundwater levels on one elevation section.",
    "quick_preview": "Fast chart for checking data; not for reports.",
}

_TRANSECT_MODE_LABELS: dict[str, str] = {
    "By hole sequence": "Pick holes in order",
    "By coordinates": "Type map coordinates",
    "Recommended": "Use a suggested line",
}

_PAGE_PRESET_LABELS: dict[str, str] = {
    "auto": "Automatic (matches the output style)",
    "tight_fence": "Section only (tight crop)",
    "title_block": "Full sheet with title block",
    "letter_portrait": "Letter, portrait",
    "letter_landscape": "Letter, landscape",
    "tabloid_landscape": "Tabloid (11 × 17), landscape",
}

_FILENAME_PATTERN_LABELS: dict[str, str] = {
    "section_title": "Section title (e.g. Section_A-A)",
    "project_figure_transect_rev": "Project_Figure_Section_Rev (e.g. 12345_Fig3_A-A_RevA)",
}

# Title block fields: shown up front vs. under "More title block fields".
_TITLE_BLOCK_MAIN_KEYS: tuple[str, ...] = (
    "consulting_figure_number",
    "consulting_project_number",
    "consulting_date",
    "consulting_prepared_by",
)
_TITLE_BLOCK_MORE_KEYS: tuple[str, ...] = (
    "consulting_start_primary",
    "consulting_start_secondary",
    "consulting_end_primary",
    "consulting_end_secondary",
    "consulting_start_label",
    "consulting_end_label",
    "consulting_map_scale",
    "consulting_source",
    "consulting_drawn_by",
    "consulting_revised",
    "consulting_prepared_for",
    "consulting_notes",
    "consulting_logo_for",
    "consulting_logo_by",
)

# Export widgets live in the Export section (and margins / crop / CAD layers in
# Advanced); Quick preview hides them, so their values are kept alive.
_EXPORT_WIDGET_KEYS: tuple[str, ...] = (
    "export_page_preset",
    "export_filename_pattern",
    "export_revision",
    "export_dpi",
    "export_margin_top_in",
    "export_margin_bottom_in",
    "export_margin_left_in",
    "export_margin_right_in",
    "export_fence_only",
    "export_show_draft_watermark",
    "export_cad_svg_layers",
    "export_include_title_block",
    "export_include_legend",
    "export_include_water_table",
    "export_include_qa_footer",
    "export_viewport_xmin",
    "export_viewport_ymin",
    "export_viewport_xmax",
    "export_viewport_ymax",
    "export_output_dir",
)


def _keep_widget_state(*keys: str) -> None:
    """Keep values of keyed widgets that are hidden this run.

    Streamlit drops a widget's session value when the widget is not rendered;
    re-assigning it keeps the user's choice for when the control comes back.
    """
    for key in keys:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def _is_filled(key: str) -> bool:
    value = st.session_state.get(key)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _filled_count(keys: tuple[str, ...]) -> str:
    return f"{sum(_is_filled(key) for key in keys)} of {len(keys)} filled"


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
    auto_fit_track_width: bool
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
    # None = auto (fit page, caption prints the measured VE); else exact VE.
    vertical_exaggeration: float | None
    transect_mode: str
    offset_warning_m: float
    max_offset_for_interpolation_m: float
    uncertainty_spacing_m: float
    uncertainty_offset_m: float
    selected_profile_key: str
    override_id: str | None
    default_elevation_m: float | None
    target_crs: str | None
    output_preset: str
    sample_figure_profile: bool
    prefer_chemistry: bool
    show_parameter_labels_default: bool | None
    parameter_interpolate_segments_default: bool | None
    parameter_draw_markers_default: bool | None
    elevation_mode_default: str | None
    column_header_detail: str
    show_scale_bar: bool
    show_ve_annotation: bool
    show_parameter_legend_text: bool
    export_font_family: str
    export_font_size: float
    parameter_marker_size: float
    connect_chemistry_values: bool
    water_line_solid_default: bool | None
    legend_ncol: int
    export_framing: ExportFramingConfig


def _render_borehole_column_controls() -> tuple[float, bool]:
    """Schematic column width + optional auto-fit to hole spacing (always editable)."""
    track_width_m = st.slider(
        "Borehole column width (m)",
        min_value=0.5,
        max_value=8.0,
        step=0.25,
        key="track_width_m",
        disabled=False,
        help=(
            "Drawn width of each hole's column on the cross-section "
            "(not the casing diameter). Typical: section sheet ~3 m, consulting report ~1.2 m."
        ),
    )
    auto_fit_track_width = st.toggle(
        "Auto-fit column width to hole spacing",
        key="auto_fit_track_width",
        disabled=False,
        help="Narrows the columns when holes are close together so they never overlap.",
    )
    return float(track_width_m), bool(auto_fit_track_width)


def _render_export_framing_panel() -> None:
    """Export section: page, file names and what goes on the exported sheet."""
    st.selectbox(
        "Page size and crop",
        options=list(_PAGE_PRESET_LABELS),
        format_func=lambda value: _PAGE_PRESET_LABELS.get(value, value),
        key="export_page_preset",
        help="Page size and cropping for PNG and PDF files.",
    )
    st.selectbox(
        "File names",
        options=list(_FILENAME_PATTERN_LABELS),
        format_func=lambda value: _FILENAME_PATTERN_LABELS.get(value, value),
        key="export_filename_pattern",
        help="How downloaded files are named, e.g. from the section title or as project_figure_section_rev.",
    )
    st.text_input(
        "Revision / draft tag",
        key="export_revision",
        placeholder="Rev A or DRAFT",
        help="Added to file names (and the title block REVISED field when blank).",
    )
    st.number_input(
        "Image resolution (DPI)",
        min_value=150,
        max_value=600,
        step=50,
        key="export_dpi",
        help="PNG resolution. 300 DPI suits reports; higher values make larger, slower files.",
    )
    st.toggle(
        "Section only (no title block or legend)",
        key="export_fence_only",
        help="Exports just the cross-section drawing, e.g. to paste into a CAD sheet.",
    )
    st.toggle(
        "DRAFT watermark on PNG and PDF",
        key="export_show_draft_watermark",
        help="Prints a light DRAFT watermark and adds DRAFT to file names.",
    )
    layer_cols = st.columns(2)
    with layer_cols[0]:
        st.toggle("Include title block", key="export_include_title_block", help="Project / figure details box on the sheet.")
        st.toggle("Include lithology legend", key="export_include_legend", help="Key of soil and rock patterns shown.")
    with layer_cols[1]:
        st.toggle("Include water table", key="export_include_water_table", help="Groundwater lines and labels.")
        st.toggle(
            "Include QA notes (PDF)",
            key="export_include_qa_footer",
            help="Adds the data-check notes (overlaps, gaps, crossing layers) to the PDF.",
        )
    st.text_input(
        "Save exports to folder (optional)",
        key="export_output_dir",
        placeholder=r"P:\Projects\Job\Figures",
    )


def _render_export_layout_advanced() -> None:
    """Margins, viewport crop and CAD layers (Advanced section)."""
    st.markdown("**Page margins and crop**")
    margin_cols = st.columns(2)
    with margin_cols[0]:
        st.number_input("Top margin (in)", min_value=0.0, max_value=2.0, step=0.05, key="export_margin_top_in", help="White space around the drawing on the exported page.")
        st.number_input("Left margin (in)", min_value=0.0, max_value=2.0, step=0.05, key="export_margin_left_in", help="White space around the drawing on the exported page.")
    with margin_cols[1]:
        st.number_input("Bottom margin (in)", min_value=0.0, max_value=2.0, step=0.05, key="export_margin_bottom_in", help="White space around the drawing on the exported page.")
        st.number_input("Right margin (in)", min_value=0.0, max_value=2.0, step=0.05, key="export_margin_right_in", help="White space around the drawing on the exported page.")
    with st.expander("Crop to an area (section distance / elevation)", expanded=False):
        crop_cols = st.columns(2)
        with crop_cols[0]:
            st.text_input("Distance from", key="export_viewport_xmin", placeholder="optional")
            st.text_input("Elevation from", key="export_viewport_ymin", placeholder="optional")
        with crop_cols[1]:
            st.text_input("Distance to", key="export_viewport_xmax", placeholder="optional")
            st.text_input("Elevation to", key="export_viewport_ymax", placeholder="optional")
    st.toggle(
        "Layered SVG for CAD",
        key="export_cad_svg_layers",
        help="Groups the SVG into named layers (geology, water, labels) for CAD editing.",
    )


def render_sidebar() -> SidebarState:
    from ui_helpers import build_export_framing_from_mapping

    apply_pending_project_seed()
    has_parsed = st.session_state.get("parse_result") is not None

    # ------------------------------------------------------------------ Data
    with st.expander("Data", expanded=True):
        st.caption(
            "Fill in the Excel template, then upload it here. Workbooks with Collars and "
            "Lithology sheets, and field exports with latitude/longitude, also work."
        )
        render_input_template_download(
            key="sidebar_download_input_template",
            help="Fill in Collars and Lithology in Excel, then upload it below.",
        )
        uploaded_name = st.session_state.get("uploaded_name")
        if uploaded_name or st.session_state.get("file_bytes"):
            st.caption(f"Loaded: {uploaded_name or 'workbook.xlsx'}")
        # Stacked full-width: two columns in the 300px sidebar truncated both
        # labels ('Clear wor…'), which is risky for a destructive action.
        if st.button(
            "Try sample project",
            key="sidebar_try_sample",
            width="stretch",
            icon=":material/science:",
        ):
            request_destructive("sample")
        if st.button(
            "Clear workbook",
            key="sidebar_clear_workbook",
            width="stretch",
            icon=":material/delete:",
        ):
            request_destructive("clear")
        _render_pending_destructive()
        uploaded = st.file_uploader(
            "Upload Excel workbook",
            type=["xlsx"],
            key=f"workbook_uploader_{st.session_state.get('workbook_uploader_key', 0)}",
            help=(
                "A filled-in template, a workbook with Collars and Lithology sheets, "
                "or a field export with latitude/longitude."
            ),
        )
        if uploaded is not None:
            st.session_state.uploaded_name = uploaded.name

        # Open the format settings only when they need attention (failed parse or
        # a non-standard workbook); a clean workbook keeps the sidebar short.
        detection = st.session_state.get("detection_result")
        needs_import_attention = bool(st.session_state.get("file_bytes")) and (
            st.session_state.get("parse_result") is None
            or (detection is not None and detection.profile_id != NATIVE_PROFILE_ID)
        )
        selected_profile_key, override_id, default_elevation_m, target_crs = _render_import_settings(
            expanded=needs_import_attention,
        )

    # ---------------------------------------------------------- Figure style
    with st.expander("Figure style", expanded=has_parsed):
        output_preset = st.selectbox(
            "Output style",
            options=tuple(OUTPUT_PRESET_LABELS.keys()),
            # Short names fit the sidebar; the caption below explains the style.
            format_func=output_preset_short_name,
            key="output_preset",
            help="  \n".join(
                f"**{OUTPUT_PRESET_LABELS[key]}**: {text}" for key, text in _OUTPUT_STYLE_HELP.items()
            ),
        )
        # Visible, not hidden behind (?): what this style produces.
        st.caption(_OUTPUT_STYLE_HELP.get(output_preset, ""))
        preset_config = resolve_output_preset(output_preset)
        parsed = st.session_state.get("parse_result")
        if (
            preset_config.prefer_chemistry
            and parsed is not None
            and not getattr(parsed, "environmental_readings", None)
        ):
            st.caption(
                "This workbook has no lab values, so no chemistry labels will be drawn. "
                "Add an Environmental sheet (see Help → Workbook and data entry)."
            )
        visibility = sidebar_visibility(output_preset)
        style_name = output_preset_short_name(output_preset)
        render_layout = preset_config.render_layout
        report_preset = preset_config.report_preset
        is_consulting_layout = render_layout == "consulting_section"
        sample_figure = preset_config.sample_figure_profile
        # Generic consulting forces GW chrome; sample presets use preset flags.
        force_gw_chrome = is_consulting_layout and not sample_figure
        interpolate_water_table = (
            True if force_gw_chrome else preset_config.interpolate_water_table
        )
        show_water_elevation_labels = (
            preset_config.show_water_elevation_labels
            if preset_config.show_water_elevation_labels is not None
            else force_gw_chrome
        )
        parsed_for_water = st.session_state.get("parse_result")
        several_water_series = water_has_multiple_series(
            getattr(parsed_for_water, "water_levels", None)
        )
        show_water_legend = (
            preset_config.show_water_legend
            if preset_config.show_water_legend is not None
            # Several blue series are unreadable without a key, so start on.
            else (force_gw_chrome or several_water_series)
        )
        show_dry_well_nm = (
            preset_config.show_dry_well_nm
            if preset_config.show_dry_well_nm is not None
            else force_gw_chrome
        )
        water_interpolate_across_gaps = bool(
            preset_config.water_interpolate_across_gaps
            if preset_config.water_interpolate_across_gaps is not None
            else False
        )
        if st.session_state.get("_synced_output_preset") != output_preset:
            st.session_state.allow_pinch_outs = preset_config.allow_pinch_outs
            st.session_state.show_ground_surface = preset_config.show_ground_surface
            st.session_state.show_legend = preset_config.show_legend
            if preset_config.show_scale_bar is not None:
                st.session_state.show_scale_bar = preset_config.show_scale_bar
            if preset_config.show_ve_annotation is not None:
                st.session_state.show_ve_annotation = preset_config.show_ve_annotation
            if preset_config.show_parameter_legend_text is not None:
                st.session_state.show_parameter_legend_text = (
                    preset_config.show_parameter_legend_text
                )
            if preset_config.interpretation_mode is not None:
                st.session_state.interpretation_mode = preset_config.interpretation_mode
            if preset_config.elevation_mode is not None:
                st.session_state.elevation_mode = preset_config.elevation_mode
            if preset_config.vertical_exaggeration is not None:
                st.session_state.vertical_exaggeration = normalize_ve_choice(
                    preset_config.vertical_exaggeration
                )
            st.session_state._synced_output_preset = output_preset

        if visibility.interpretation_editable:
            interpretation_mode = st.radio(
                "Layers between holes",
                options=list(INTERPRETATION_LABELS),
                format_func=lambda value: INTERPRETATION_LABELS[value],
                key="interpretation_mode",
                help=(
                    "Connect layers: shade matching layers from hole to hole. "
                    "Contact lines: draw the boundaries only. "
                    "Observed logs: show each hole's log with nothing in between."
                ),
            )
        else:
            _keep_widget_state("interpretation_mode")
            interpretation_mode = str(preset_config.interpretation_mode)

        # Hidden (not disabled) when the style fixes these: one caption instead.
        if visibility.pinch_outs_editable:
            allow_pinch_outs = st.toggle(
                "Show layers that end between holes",
                key="allow_pinch_outs",
                disabled=interpretation_mode == "borehole_only",
                help="When off, a layer logged in only one hole is not drawn toward its neighbours.",
            )
        else:
            if force_gw_chrome:
                # Generic consulting builds always draw without them (app_build).
                st.session_state.allow_pinch_outs = False
            _keep_widget_state("allow_pinch_outs")
            allow_pinch_outs = bool(preset_config.allow_pinch_outs and not force_gw_chrome)

        if visibility.ground_surface_editable:
            show_ground_surface = st.toggle(
                "Show ground surface",
                key="show_ground_surface",
                help="A straight line joining the hole collar elevations (not a surveyed surface).",
            )
        else:
            _keep_widget_state("show_ground_surface")
            show_ground_surface = True

        ve_default = normalize_ve_choice(
            preset_config.vertical_exaggeration
            if preset_config.vertical_exaggeration is not None
            else DEFAULT_VE_CHOICE
        )
        if "vertical_exaggeration" not in st.session_state:
            st.session_state.vertical_exaggeration = ve_default
        else:
            # Older sessions / workbook seeds hold "5" or 5.0: keep a valid option.
            st.session_state.vertical_exaggeration = normalize_ve_choice(
                st.session_state.vertical_exaggeration
            )
        if visibility.vertical_exaggeration_editable:
            ve_choice = st.selectbox(
                "Vertical exaggeration",
                options=ve_choice_options(st.session_state.vertical_exaggeration),
                format_func=ve_choice_label,
                key="vertical_exaggeration",
                help=(
                    "Auto fits the section to the page and prints the vertical exaggeration "
                    "it really ends up with. A number draws exactly that exaggeration "
                    "(1× = true scale); the plot shrinks inside its frame to keep it."
                ),
            )
            vertical_exaggeration = ve_choice_to_request(ve_choice)
        else:
            _keep_widget_state("vertical_exaggeration")
            vertical_exaggeration = ve_choice_to_request(ve_default)

        figure_summary = locked_figure_summary(output_preset)
        if figure_summary:
            st.caption(figure_summary)

        show_hatches = st.toggle(
            "Hatch patterns",
            key="show_hatches",
            help=(
                "Lithology patterns from the legend template (dots = sandy, lines = silty, "
                "+ = clay loam, cobbles = gravel). Off = solid colours only, which merges "
                "units that share a colour."
            ),
        )

        # One title field per style. Consulting layouts print the title block's
        # section label as the figure title, so that field is shown here as
        # "Section title" and section_title (file names, metadata) follows it.
        if "section_title" not in st.session_state:
            st.session_state.section_title = "Borehole Cross-Section"
        if is_consulting_layout:
            if (
                st.session_state.pop("_reset_consulting_section_label", False)
                or "consulting_section_label" not in st.session_state
            ):
                st.session_state.consulting_section_label = (
                    st.session_state.get("section_title") or "Borehole Cross-Section"
                )
            consulting_section_label = st.text_input(
                "Section title",
                key="consulting_section_label",
                help="Printed as the figure title and used in file names.",
            )
            section_title = consulting_section_label or str(
                st.session_state.get("section_title") or "Borehole Cross-Section"
            )
            st.session_state["section_title"] = section_title
        else:
            section_title = st.text_input(
                "Section title",
                key="section_title",
                help="Printed on the figure and used in file names.",
            )
            # Same value for the consulting styles' title block (one field).
            st.session_state["consulting_section_label"] = section_title

        # Groundwater
        if visibility.groundwater_editable:
            with st.expander("Groundwater", expanded=False):
                interpolate_water_table = st.toggle(
                    "Join water table between holes",
                    value=interpolate_water_table,
                    help="When off, measured water levels are shown at each hole only.",
                )
                show_water_elevation_labels = st.toggle(
                    "Show water level labels",
                    value=show_water_elevation_labels,
                )
                show_water_legend = st.toggle(
                    "Show groundwater legend",
                    value=show_water_legend,
                )
                show_dry_well_nm = st.toggle(
                    "Show 'not measured' markers for dry wells",
                    value=show_dry_well_nm,
                )
                water_interpolate_across_gaps = st.toggle(
                    "Join water levels across holes with no reading",
                    value=water_interpolate_across_gaps,
                    help="When off, the water line only joins neighbouring holes that both have a reading.",
                )
        else:
            gw_summary = locked_groundwater_summary(output_preset)
            if gw_summary:
                st.caption(gw_summary)

        # Labels and legend
        with st.expander("Labels and legend", expanded=False):
            if visibility.label_detail_editable:
                st.selectbox(
                    "Borehole label detail",
                    options=["id_only", "id_rl_td"],
                    format_func=lambda value: (
                        "Hole ID only" if value == "id_only" else "Hole ID + collar elevation + total depth"
                    ),
                    key="column_header_detail",
                    help="Text above each hole column.",
                )
            else:
                _keep_widget_state("column_header_detail")
            if visibility.chart_legend_editable:
                show_legend = st.toggle(
                    "Legend on chart",
                    key="show_legend",
                    help="Show the lithology legend beside the cross-section.",
                )
            else:
                _keep_widget_state("show_legend")
                show_legend = False
            if not (visibility.label_detail_editable and visibility.chart_legend_editable):
                st.caption(
                    f"Set by {style_name}: hole ID only above each column; "
                    "legend in the title block."
                )
            st.toggle(
                "Two-column lithology legend",
                key="legend_two_columns",
                value=True,
                help="Wrap long lithology lists into two columns.",
            )
            st.toggle(
                "Show scale bar",
                key="show_scale_bar",
                help="Horizontal scale bar on the figure.",
            )
            st.toggle(
                "Show vertical exaggeration note",
                key="show_ve_annotation",
                help=(
                    "Prints the vertical exaggeration on the figure: the chosen value, "
                    "or with Auto the value measured on the page (e.g. 'V.E. ≈2.2×')."
                ),
            )
            if visibility.parameter_text_block_editable:
                st.toggle(
                    "Show lab parameters note",
                    key="show_parameter_legend_text",
                    help="Short 'Parameters: …' note in the lower-left corner listing the plotted lab values.",
                )
            else:
                _keep_widget_state("show_parameter_legend_text")

        # Chemistry: only when the workbook has lab values to plot.
        parse_result = st.session_state.get("parse_result")
        has_chemistry = bool(getattr(parse_result, "environmental_readings", None))
        parameter_interpolate_across_gaps = False
        if has_chemistry:
            with st.expander("Chemistry", expanded=False):
                if visibility.chemistry_marker_size_editable:
                    st.number_input(
                        "Chemistry dot size",
                        min_value=4.0,
                        max_value=64.0,
                        step=2.0,
                        key="parameter_marker_size",
                        help="Size of chemistry sample dots.",
                    )
                else:
                    _keep_widget_state("parameter_marker_size")
                    st.caption(f"Set by {style_name}: lab values shown as labels without dots.")
                # Joining values between holes is set once, in Configure
                # ("Interpolate parameter between adjacent holes").
                parameter_interpolate_across_gaps = st.toggle(
                    "Join lab values across holes with no reading",
                    value=False,
                    help="When off, lines only join neighbouring holes that both have a reading.",
                )
        else:
            _keep_widget_state("parameter_marker_size", "connect_chemistry_values")

    # ---------------------------------------------------------- Section line
    with st.expander("Section line", expanded=has_parsed):
        if st.session_state.pop("pending_transect_mode", None):
            st.session_state.transect_definition_mode = "By hole sequence"
        transect_mode = st.radio(
            "How to set the section line",
            options=list(_TRANSECT_MODE_LABELS),
            format_func=lambda value: _TRANSECT_MODE_LABELS.get(value, value),
            key="transect_definition_mode",
            help="Pick holes in order, use a suggested line through the holes, or type map coordinates.",
        )
        _apply_pending_offset_thresholds()
        offset_warning_m = st.number_input(
            "Warn when a hole is this far off the line (m)",
            min_value=1.0,
            step=5.0,
            key="offset_warning_m",
            help="Flags selected holes that sit more than this many metres from the section line.",
        )

    # ----------------------------------------------- Title block (consulting)
    consulting_title_block: ConsultingTitleBlock | None = None
    if not visibility.title_block_shown:
        # Keep typed / Project-sheet values for when a consulting style returns
        # (file uploaders cannot be re-assigned, so logos are not kept).
        _keep_widget_state(
            *(key for key in _TITLE_BLOCK_MAIN_KEYS + _TITLE_BLOCK_MORE_KEYS if "logo" not in key),
        )
    if visibility.title_block_shown:
        _seed_title_block_defaults()
        with st.expander(
            f"Title block ({_filled_count(_TITLE_BLOCK_MAIN_KEYS + _TITLE_BLOCK_MORE_KEYS)})",
            expanded=False,
            key="sidebar_title_block_expander",
        ):
            consulting_title_block = _render_consulting_report_sheet(section_title)

    # ---------------------------------------------------------------- Export
    if visibility.export_shown:
        with st.expander("Export", expanded=False):
            _render_export_framing_panel()
    else:
        _keep_widget_state(*_EXPORT_WIDGET_KEYS)
        st.caption(
            f"{style_name} is for checking data. Pick another output style for "
            "report files and export settings."
        )

    # -------------------------------------------------------------- Advanced
    with st.expander("Advanced", expanded=False):
        st.markdown("**Borehole columns**")
        track_width_m, auto_fit_track_width = _render_borehole_column_controls()

        st.markdown("**Fonts**")
        st.selectbox(
            "Figure font",
            options=["Arial", "Calibri", "DejaVu Sans"],
            key="export_font_family",
            help="Arial matches most drafting templates when the PDF is edited later.",
        )
        st.number_input(
            "Font size (pt)",
            min_value=6.0,
            max_value=14.0,
            step=0.5,
            key="export_font_size",
        )

        st.markdown("**Uncertainty shading**")
        borehole_only = interpretation_mode == "borehole_only"
        uncertainty_spacing_m = st.number_input(
            "Shade as uncertain when holes are more than this far apart (m)",
            min_value=10.0,
            value=float(DEFAULT_UNCERTAINTY_SPACING_M),
            step=10.0,
            disabled=borehole_only,
        )
        uncertainty_offset_m = st.number_input(
            "Shade as uncertain when a hole is more than this far off the line (m)",
            min_value=1.0,
            step=5.0,
            key="uncertainty_offset_m",
            disabled=borehole_only,
        )
        max_offset_for_interpolation_m = st.number_input(
            "Don't connect layers to holes more than this far off the line (m)",
            min_value=1.0,
            step=5.0,
            value=float(st.session_state.offset_warning_m),
        )
        warn_on_correlation_gaps = st.toggle(
            "Warn when layers don't match between holes",
            value=False,
            disabled=borehole_only,
            help="Adds a note to the checks when a layer in one hole has no match in the next.",
        )

        if visibility.export_shown:
            _render_export_layout_advanced()

        if st.session_state.get("parse_result"):
            st.markdown("**Lithology colours and patterns**")
            _render_fill_style_editor()

        st.markdown("**AI assist**")
        _render_ai_assist()

    export_framing = build_export_framing_from_mapping(dict(st.session_state))

    st.caption(f"{CREATED_BY} · {COPYRIGHT_SHORT}")

    return SidebarState(
        uploaded=uploaded,
        interpretation_mode=interpretation_mode,
        report_preset=report_preset,
        render_layout=render_layout,
        is_consulting_layout=is_consulting_layout,
        allow_pinch_outs=allow_pinch_outs,
        show_ground_surface=show_ground_surface,
        track_width_m=track_width_m,
        auto_fit_track_width=auto_fit_track_width,
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
        output_preset=output_preset,
        sample_figure_profile=sample_figure,
        prefer_chemistry=preset_config.prefer_chemistry,
        show_parameter_labels_default=preset_config.show_parameter_labels,
        parameter_interpolate_segments_default=preset_config.parameter_interpolate_segments,
        parameter_draw_markers_default=preset_config.parameter_draw_markers,
        elevation_mode_default=preset_config.elevation_mode,
        column_header_detail=str(st.session_state.get("column_header_detail", "id_only")),
        show_scale_bar=bool(st.session_state.get("show_scale_bar", False)),
        show_ve_annotation=bool(st.session_state.get("show_ve_annotation", False)),
        show_parameter_legend_text=bool(
            st.session_state.get("show_parameter_legend_text", False)
        ),
        export_font_family=str(st.session_state.get("export_font_family", "Arial")),
        export_font_size=float(st.session_state.get("export_font_size", 8.0)),
        parameter_marker_size=float(st.session_state.get("parameter_marker_size", 16.0)),
        connect_chemistry_values=bool(
            st.session_state.get("connect_chemistry_values", False)
        ),
        water_line_solid_default=preset_config.water_line_solid,
        legend_ncol=2 if bool(st.session_state.get("legend_two_columns", True)) else 1,
        export_framing=export_framing,
    )


def _render_fill_style_editor() -> None:
    st.caption("Change the fill colour and pattern for a lithology code (saved on this computer).")
    style_codes = sorted(st.session_state.get("unique_lithology_codes") or [])
    if not style_codes:
        st.info("Load a workbook to edit lithology colours.")
        return
    style_code = st.selectbox(
        "Lithology code",
        options=style_codes,
        key="style_editor_code",
        help="Soil / rock code from your workbook whose fill you want to change.",
    )
    current_style = get_lithology_style(style_code)
    style_color = st.color_picker("Fill colour", value=current_style.color, key="style_editor_color")
    style_hatch_options = sorted(set(USGS_LITHOLOGY_HATCHES.values()))
    style_hatch = st.selectbox(
        "Hatch pattern",
        options=style_hatch_options,
        format_func=lambda hatch: hatch or "None (solid fill)",
        help="Pattern drawn over the fill colour (template 261002 marks by default).",
        index=style_hatch_options.index(current_style.hatch)
        if current_style.hatch in style_hatch_options
        else 0,
        key="style_editor_hatch",
    )
    save_col, reset_col = st.columns(2)
    with save_col:
        if st.button("Save fill style", key="save_fill_style", width="stretch"):
            save_lithology_style_override(style_code, style_color, style_hatch)
            st.success(f"Saved style for {style_code}. Use Generate section to preview.")
    with reset_col:
        if st.button(
            "Reset to scheme",
            key="reset_fill_style",
            width="stretch",
            disabled=not has_lithology_style_override(style_code),
            help="Remove the saved override so this code uses the agreed colour and hatch.",
        ):
            clear_lithology_style_override(style_code)
            st.success(f"{style_code} uses the agreed scheme again.")
            st.rerun()


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
        st.caption("AI assist is turned off for this installation. Local checks still run.")
        st.session_state["enable_ai_suggestions"] = False
        return
    _seed_free_llm_defaults()
    # Drop any legacy durable session secrets from older builds.
    st.session_state.pop("llm_api_key", None)
    st.session_state.pop("openai_api_key", None)
    st.caption(
        "Optional. A free Groq or Google Gemini key adds written QA summaries and "
        "column-matching help. Local checks always run."
    )
    st.selectbox(
        "AI provider",
        options=("groq", "gemini", "openai"),
        format_func=lambda value: {
            "groq": "Groq — free tier (recommended)",
            "gemini": "Google Gemini — free tier",
            "openai": "OpenAI — paid",
        }[value],
        key="llm_provider",
        help="Free keys: Groq at console.groq.com, Gemini at aistudio.google.com/apikey. OpenAI is paid.",
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
        st.caption("Using the key set up for this installation.")
        st.session_state.pop(runtime_key, None)
    else:
        entered = st.text_input(
            "API key",
            type="password",
            value="",
            help="Get a free key at console.groq.com (Groq) or aistudio.google.com/apikey (Gemini).",
        )
        if entered.strip():
            st.session_state[runtime_key] = entered.strip()
            if is_free_llm_provider(provider) and not st.session_state.get("enable_ai_suggestions"):
                st.session_state.enable_ai_suggestions = True
        if st.session_state.get(runtime_key):
            st.caption("Key is used for this session only and isn't saved.")
            if st.button("Clear API key", key="clear_llm_api_key_runtime"):
                st.session_state.pop(runtime_key, None)
                st.rerun()
    st.checkbox(
        "Use AI suggestions",
        value=False,
        key="enable_ai_suggestions",
        help="Uses the selected provider once a key is set. Turns on by itself the first time a free key is found.",
    )


def _render_import_settings(*, expanded: bool = False) -> tuple[str, str | None, float, str | None]:
    with st.expander("Workbook format", expanded=expanded):
        profile_options = {profile.id: profile.label for profile in list_profiles()}
        profile_options[NATIVE_PROFILE_ID] = "Standard workbook (Collars + Lithology)"
        profile_options[DATA_ENTRY_PROFILE_ID] = "Cross Section Studio template"
        auto_profile = st.session_state.get("detection_result")
        default_profile_index = 0
        profile_ids = ["auto"] + list(profile_options.keys())
        if auto_profile is not None and auto_profile.profile_id in profile_options:
            default_profile_index = profile_ids.index(auto_profile.profile_id)
        selected_profile_key = st.selectbox(
            "Workbook format",
            options=profile_ids,
            format_func=lambda key: "Auto-detect" if key == "auto" else profile_options[key],
            index=min(default_profile_index, len(profile_ids) - 1),
        )
        default_elevation_m = st.number_input(
            "Default collar elevation (m)",
            min_value=0.0,
            value=None,
            step=1.0,
            placeholder=f"Blank = {DEFAULT_PROFILE_ELEVATION_M:.0f} m, flagged as unsurveyed",
            help=(
                "For field exports without a collar elevation column. Leave blank to use a "
                "placeholder (flagged as unsurveyed); enter the surveyed site elevation for "
                "sections in metres above sea level."
            ),
        )
        st.session_state.auto_assign_unit_order = st.checkbox(
            "Number layers down each hole automatically",
            value=bool(st.session_state.get("auto_assign_unit_order", True)),
            help="Orders layers 1, 2, 3… from the top of each hole when the workbook gives no layer order.",
        )
        # Rarely needed; kept out of the way so a client-specific value is never
        # applied by default (blank = the detected format's own settings).
        with st.expander("Advanced import options", expanded=False):
            target_crs = st.text_input(
                "Map coordinate system (EPSG code)",
                value="EPSG:32611",
                help=(
                    "Used to convert latitude/longitude in field exports to metres. "
                    "Default: UTM zone 11N (Alberta)."
                ),
            ).strip() or None
            override_id = st.text_input(
                "Site-specific import settings (optional)",
                value="",
                placeholder="Leave blank",
                help="Name of a saved site adjustment (e.g. per-hole coordinate corrections) you were given.",
            ).strip() or None
        if st.session_state.get("parse_result") is not None:
            if st.button("Re-read workbook", key="reparse_workbook"):
                st.session_state.parse_signature = None
                st.rerun()
    return selected_profile_key, override_id, default_elevation_m, target_crs


def _seed_title_block_defaults() -> None:
    """Init keyed title-block widgets only when absent (keeps Project-sheet seeding)."""
    if "consulting_map_scale" not in st.session_state:
        # Blank = "AS SHOWN": the drawn scale bar is true to scale.
        st.session_state.consulting_map_scale = ""
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


def _render_consulting_report_sheet(section_title: str) -> ConsultingTitleBlock:
    """Title block fields. The section label is edited as "Section title" under
    Figure style (one field); ``section_title`` follows it."""
    _seed_title_block_defaults()
    consulting_section_label = str(st.session_state.get("consulting_section_label") or "")
    figure_number = st.text_input("Figure no.", key="consulting_figure_number")
    project_number = st.text_input("Project no.", key="consulting_project_number")
    report_date = st.text_input("Date", key="consulting_date")
    prepared_by = st.text_input("Prepared by", key="consulting_prepared_by")
    with st.expander(
        f"More title block fields ({_filled_count(_TITLE_BLOCK_MORE_KEYS)})",
        expanded=False,
        key="sidebar_title_block_more_expander",
    ):
        st.caption("Section line ends (printed at each end of the cross-section)")
        transect_cols = st.columns(2)
        with transect_cols[0]:
            transect_start_primary = st.text_input("Start letter (e.g. B)", key="consulting_start_primary")
            transect_start_secondary = st.text_input(
                "Start direction (e.g. SOUTHWEST)",
                key="consulting_start_secondary",
            )
        with transect_cols[1]:
            transect_end_primary = st.text_input("End letter (e.g. B')", key="consulting_end_primary")
            transect_end_secondary = st.text_input(
                "End direction (e.g. NORTHEAST)",
                key="consulting_end_secondary",
            )
        transect_start_label = st.text_input(
            "Start label (used if no start letter)", key="consulting_start_label"
        )
        transect_end_label = st.text_input("End label (used if no end letter)", key="consulting_end_label")
        map_scale = st.text_input(
            "Map scale",
            key="consulting_map_scale",
            placeholder="As shown (scale bar)",
            help="Leave blank to print AS SHOWN; the scale bar is drawn true to scale. "
            "A value you enter (e.g. 1:1000) is printed as written.",
        )
        source = st.text_input("Source", key="consulting_source")
        drawn_by = st.text_input("Drawn by", key="consulting_drawn_by")
        revised = st.text_input("Revised", key="consulting_revised")
        prepared_for = st.text_input("Prepared for", key="consulting_prepared_for")
        notes_text = st.text_area("Notes", key="consulting_notes", help="One note per line.")
        logo_for = st.file_uploader(
            "Client logo (PNG)",
            type=["png"],
            key="consulting_logo_for",
        )
        logo_by = st.file_uploader(
            "Your company logo (PNG)",
            type=["png"],
            key="consulting_logo_by",
        )
        st.caption(
            "Optional workbook sheets: **Screens** (hole_id, from_depth, to_depth) and "
            "**Gradients** (hole_id, direction)."
        )
    report_ai_cols = st.columns(2)
    with report_ai_cols[0]:
        if st.button("Suggest title block", key="suggest_report_fields"):
            report_holes = list(st.session_state.get("hole_ids") or [])
            label_for_context = consulting_section_label or section_title
            if st.session_state.parse_result is not None:
                context = _report_context_from_selection(
                    st.session_state.parse_result,
                    report_holes,
                    vertical_exaggeration=ve_choice_to_request(
                        st.session_state.get("vertical_exaggeration", DEFAULT_VE_CHOICE)
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
                    "vertical_exaggeration": ve_choice_to_request(
                        st.session_state.get("vertical_exaggeration", DEFAULT_VE_CHOICE)
                    ),
                }
            st.session_state.ai_report_suggestion = (
                _build_assistant().suggest_report_metadata(context)
            )
    with report_ai_cols[1]:
        if st.session_state.get("ai_report_suggestion") and st.button(
            "Use suggestion",
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
