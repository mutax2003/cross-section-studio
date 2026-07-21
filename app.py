"""Streamlit application for borehole cross-section generation."""

from __future__ import annotations

import logging
import os
import traceback

import streamlit as st

from app_build import collect_section_build_request, generate_cross_section
from app_common import (
    _active_transect_selection,
    _render_hero,
    _render_sticky_generate_strip,
    safe_lithology_index,
)
from app_configure import render_configure_step, render_transect_sidebar
from app_generate import render_profile_and_downloads
from app_menubar import render_menubar
from app_sidebar import render_sidebar
from app_state import init_session_defaults
from app_styles import APP_CSS
from app_upload import (
    _BytesUpload,
    handle_workbook_upload,
    render_welcome_card,
    render_workbook_recovery,
)
from app_validate import render_validate_step
from models import ParseResult
from ops_apm import init_apm
from ops_audit import audit_event
from ops_auth import render_logout_control, require_auth
from ops_logging import configure_logging
from ui_helpers import parse_coordinate_lines, svg_display_meta, workflow_stage
from ui_output_presets import OUTPUT_PRESET_LABELS

configure_logging()
init_apm()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Cross Section Studio",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(APP_CSS, unsafe_allow_html=True)
init_session_defaults()
require_auth()
render_logout_control()
render_menubar()

with st.sidebar:
    sidebar = render_sidebar()

_render_hero(
    workflow_stage(
        has_upload=sidebar.uploaded is not None or bool(st.session_state.get("file_bytes")),
        has_parse_result=st.session_state.parse_result is not None,
        has_profile=st.session_state.svg_bytes is not None,
        has_blocking_errors=bool(
            getattr(st.session_state.get("quality_report"), "has_blocking_errors", False)
        ),
        has_transect=st.session_state.get("transect_selection") is not None,
    )
)

flash_success = st.session_state.pop("_flash_success", None)
if flash_success:
    st.success(flash_success)
flash_error = st.session_state.pop("_flash_error", None)
if flash_error:
    st.error(flash_error)

parse_result: ParseResult | None = st.session_state.parse_result
# Prefer session bytes when the uploader is empty (clear/sample remount) so sample
# loads and Cloud reconnects keep working without a re-upload widget value.
upload_source = sidebar.uploaded
if upload_source is None and st.session_state.get("file_bytes"):
    upload_source = _BytesUpload(
        st.session_state.file_bytes,
        str(st.session_state.get("uploaded_name") or "workbook.xlsx"),
    )

configure_state = None
has_svg = st.session_state.svg_bytes is not None

if upload_source is None:
    render_welcome_card()
else:
    parse_result = handle_workbook_upload(
        upload_source,
        selected_profile_key=sidebar.selected_profile_key,
        override_id=sidebar.override_id,
        default_elevation_m=sidebar.default_elevation_m,
        target_crs=sidebar.target_crs,
    )

    if parse_result is None and st.session_state.get("file_bytes"):
        render_workbook_recovery(key_prefix="main")
    elif parse_result is not None:
        hole_ids = list(st.session_state.hole_ids)
        with st.sidebar:
            selected_holes, coordinate_text = render_transect_sidebar(
                parse_result,
                hole_ids,
                sidebar.transect_mode,
            )

        if has_svg:
            with st.expander("Setup — Validate & Configure", expanded=False):
                render_validate_step()
                configure_state = render_configure_step(
                    parse_result,
                    transect_mode=sidebar.transect_mode,
                    selected_holes=selected_holes,
                    coordinate_text=coordinate_text,
                    offset_warning_m=sidebar.offset_warning_m,
                    interpretation_mode=sidebar.interpretation_mode,
                    allow_pinch_outs=sidebar.allow_pinch_outs,
                    quality_report=st.session_state.quality_report,
                    import_report=st.session_state.import_report,
                    is_consulting_layout=sidebar.is_consulting_layout,
                    max_offset_for_interpolation_m=sidebar.max_offset_for_interpolation_m,
                )

            is_stale = True
            transect_label: str | None = None
            if configure_state and configure_state.transect_selection is not None:
                active_ids, _ = configure_state.transect_selection
                if len(active_ids) >= 2:
                    transect_label = f"A–A′ {active_ids[0]}→{active_ids[-1]}"
                    import_report = st.session_state.import_report
                    coordinate_reference = sidebar.target_crs or (
                        import_report.suggested_utm_crs if import_report else ""
                    ) or ""
                    _, current_key = collect_section_build_request(
                        parse_result,
                        transect_mode=sidebar.transect_mode,
                        selected_holes=configure_state.selected_holes,
                        coordinate_text=configure_state.coordinate_text,
                        offset_warning_m=sidebar.offset_warning_m,
                        vertical_exaggeration=sidebar.vertical_exaggeration,
                        show_hatches=sidebar.show_hatches,
                        show_legend=sidebar.show_legend,
                        section_title=sidebar.section_title,
                        interpretation_mode=sidebar.interpretation_mode,
                        allow_pinch_outs=sidebar.allow_pinch_outs,
                        uncertainty_spacing_m=sidebar.uncertainty_spacing_m,
                        uncertainty_offset_m=sidebar.uncertainty_offset_m,
                        max_offset_for_interpolation_m=sidebar.max_offset_for_interpolation_m,
                        show_ground_surface=sidebar.show_ground_surface,
                        interpolate_water_table=sidebar.interpolate_water_table,
                        warn_on_correlation_gaps=sidebar.warn_on_correlation_gaps,
                        show_water_elevation_labels=sidebar.show_water_elevation_labels,
                        show_water_legend=sidebar.show_water_legend,
                        show_dry_well_nm=sidebar.show_dry_well_nm,
                        water_interpolate_across_gaps=sidebar.water_interpolate_across_gaps,
                        environmental_parameters=configure_state.environmental_parameters,
                        show_parameter_labels=configure_state.show_parameter_labels,
                        parameter_interpolate_segments=configure_state.parameter_interpolate_segments,
                        parameter_interpolate_across_gaps=sidebar.parameter_interpolate_across_gaps,
                        render_layout=sidebar.render_layout,
                        track_width_m=sidebar.track_width_m,
                        coordinate_reference=coordinate_reference,
                        uses_placeholder_elevation=bool(
                            import_report and import_report.uses_placeholder_elevation
                        ),
                        elevation_mode=configure_state.elevation_mode,
                        report_preset=sidebar.report_preset,
                        consulting_title_block=sidebar.consulting_title_block,
                        selection=configure_state.transect_selection,
                        fail_on_overlaps=configure_state.fail_on_overlaps,
                    )
                    is_stale = (
                        st.session_state.render_cache_key is None
                        or current_key is None
                        or current_key != st.session_state.render_cache_key
                    )

            preset_key = str(st.session_state.get("output_preset", "section_sheet"))
            _render_sticky_generate_strip(
                has_svg=True,
                can_generate=bool(configure_state and configure_state.can_generate),
                is_stale=is_stale,
                section_title=sidebar.section_title,
            )
            render_profile_and_downloads(
                section_title=sidebar.section_title,
                interpretation_mode=sidebar.interpretation_mode,
                vertical_exaggeration=sidebar.vertical_exaggeration,
                is_stale=is_stale,
                parse_result_available=True,
                preset_label=OUTPUT_PRESET_LABELS.get(preset_key),
                render_layout=sidebar.render_layout,
                transect_label=transect_label,
            )
        else:
            render_validate_step()
            configure_state = render_configure_step(
                parse_result,
                transect_mode=sidebar.transect_mode,
                selected_holes=selected_holes,
                coordinate_text=coordinate_text,
                offset_warning_m=sidebar.offset_warning_m,
                interpretation_mode=sidebar.interpretation_mode,
                allow_pinch_outs=sidebar.allow_pinch_outs,
                quality_report=st.session_state.quality_report,
                import_report=st.session_state.import_report,
                is_consulting_layout=sidebar.is_consulting_layout,
                max_offset_for_interpolation_m=sidebar.max_offset_for_interpolation_m,
            )

        generate_clicked = False
        if not has_svg:
            regenerate_requested = bool(st.session_state.pop("_regenerate_requested", False))
            gen_col1, gen_col2 = st.columns([1, 3])
            with gen_col1:
                generate_clicked = st.button(
                    "Generate Cross-Section",
                    type="primary",
                    disabled=not configure_state.can_generate,
                    width="stretch",
                    key="generate_cross_section",
                )
                if regenerate_requested and configure_state.can_generate:
                    generate_clicked = True
                elif regenerate_requested and not configure_state.can_generate:
                    st.caption("Generate shortcut ignored — resolve Configure / Validate first.")
            with gen_col2:
                if configure_state.can_generate:
                    st.caption("Builds the profile from sidebar style and transect settings.")
                elif configure_state.blocking:
                    st.caption("Fix blocking QA errors in Validate before generating.")
                elif configure_state.placeholder_blocks_interp:
                    st.caption(
                        "Placeholder collar elevations block interpolated geology — "
                        "switch to relative depth or borehole-only mode."
                    )
                elif configure_state.fail_on_overlaps and configure_state.has_overlap_warnings:
                    st.caption(
                        "Polygon overlaps block export — resolve correlation or disable "
                        "'Block export on polygon overlaps'."
                    )
                elif configure_state.transect_selection is None:
                    st.caption("Select a transect (holes, coordinates, or recommended) before generating.")
                elif configure_state.has_warnings and not configure_state.override_warnings:
                    st.caption(
                        "QA warnings are present — enable 'Allow generate with warnings' in Configure, "
                        "or resolve the warnings in Validate."
                    )
                else:
                    st.caption("Resolve Configure / Validate issues before generating.")
        else:
            regenerate_requested = bool(st.session_state.pop("_regenerate_requested", False))
            generate_clicked = regenerate_requested and configure_state is not None and configure_state.can_generate
            if regenerate_requested and configure_state and not configure_state.can_generate:
                st.caption("Regenerate ignored — open **Setup — Validate & Configure** to resolve issues.")

        if (not has_svg and generate_clicked) or (has_svg and generate_clicked):
            try:
                selection = configure_state.transect_selection or _active_transect_selection(
                    parse_result,
                    sidebar.transect_mode,
                    configure_state.selected_holes,
                    configure_state.coordinate_text,
                    sidebar.offset_warning_m,
                )
                if selection is None:
                    if sidebar.transect_mode == "By coordinates":
                        try:
                            parse_coordinate_lines(configure_state.coordinate_text)
                        except ValueError as exc:
                            raise ValueError(str(exc)) from exc
                        raise ValueError(
                            f"Need at least two boreholes within {sidebar.offset_warning_m:.0f} m of the transect line"
                        )
                    raise ValueError("Select at least two holes for the transect")
                active_hole_ids, transect_points = selection
                import_report = st.session_state.import_report
                coordinate_reference = sidebar.target_crs or (
                    import_report.suggested_utm_crs if import_report else ""
                ) or ""
                uses_placeholder = bool(
                    import_report and import_report.uses_placeholder_elevation
                )
                build_request, cache_key = collect_section_build_request(
                    parse_result,
                    transect_mode=sidebar.transect_mode,
                    selected_holes=configure_state.selected_holes,
                    coordinate_text=configure_state.coordinate_text,
                    offset_warning_m=sidebar.offset_warning_m,
                    vertical_exaggeration=sidebar.vertical_exaggeration,
                    show_hatches=sidebar.show_hatches,
                    show_legend=sidebar.show_legend,
                    section_title=sidebar.section_title,
                    interpretation_mode=sidebar.interpretation_mode,
                    allow_pinch_outs=sidebar.allow_pinch_outs,
                    uncertainty_spacing_m=sidebar.uncertainty_spacing_m,
                    uncertainty_offset_m=sidebar.uncertainty_offset_m,
                    max_offset_for_interpolation_m=sidebar.max_offset_for_interpolation_m,
                    show_ground_surface=sidebar.show_ground_surface,
                    interpolate_water_table=sidebar.interpolate_water_table,
                    warn_on_correlation_gaps=sidebar.warn_on_correlation_gaps,
                    show_water_elevation_labels=sidebar.show_water_elevation_labels,
                    show_water_legend=sidebar.show_water_legend,
                    show_dry_well_nm=sidebar.show_dry_well_nm,
                    water_interpolate_across_gaps=sidebar.water_interpolate_across_gaps,
                    environmental_parameters=configure_state.environmental_parameters,
                    show_parameter_labels=configure_state.show_parameter_labels,
                    parameter_interpolate_segments=configure_state.parameter_interpolate_segments,
                    parameter_interpolate_across_gaps=sidebar.parameter_interpolate_across_gaps,
                    render_layout=sidebar.render_layout,
                    track_width_m=sidebar.track_width_m,
                    coordinate_reference=coordinate_reference,
                    uses_placeholder_elevation=uses_placeholder,
                    elevation_mode=configure_state.elevation_mode,
                    report_preset=sidebar.report_preset,
                    consulting_title_block=sidebar.consulting_title_block,
                    selection=selection,
                    fail_on_overlaps=configure_state.fail_on_overlaps,
                )
                if build_request is None or cache_key is None:
                    raise ValueError("Select at least two holes for the transect")

                svg_bytes, png_bytes, pdf_bytes, polygon_count, lithology_codes, overlap_warnings = (
                    generate_cross_section(
                        parse_result,
                        list(transect_points),
                        active_hole_ids,
                        build_request,
                        sidebar.offset_warning_m,
                        lithology_index=safe_lithology_index(parse_result),
                    )
                )
                st.session_state.svg_bytes = svg_bytes
                st.session_state.png_bytes = png_bytes
                st.session_state.pdf_bytes = pdf_bytes
                st.session_state.svg_display_meta = svg_display_meta(svg_bytes)
                st.session_state.section_lithology_codes = lithology_codes
                st.session_state.section_polygon_count = polygon_count
                st.session_state.section_hole_count = len(active_hole_ids)
                audit_event(
                    "section_generated",
                    section_title=sidebar.section_title,
                    hole_count=len(active_hole_ids),
                    polygon_count=polygon_count,
                    layout=build_request.render_layout,
                )
                st.session_state.polygon_overlap_warnings = overlap_warnings
                st.session_state.render_cache_key = cache_key
                if sidebar.interpretation_mode == "borehole_only":
                    st.session_state["_flash_success"] = (
                        f"Generated observed-data cross-section across {len(active_hole_ids)} boreholes."
                    )
                else:
                    st.session_state["_flash_success"] = (
                        f"Generated cross-section with {polygon_count} geological polygons "
                        f"across {len(active_hole_ids)} boreholes."
                    )
                st.rerun()
            except Exception as exc:
                logger.exception("Cross-section generation failed")
                error_msg = "Cross-section generation failed. See server logs for details."
                st.session_state["_flash_error"] = error_msg
                # Keep SVG if present but mark stale so downloads stay gated.
                st.session_state.render_cache_key = None
                st.error(st.session_state.pop("_flash_error", error_msg))
                if os.environ.get("CROSS_SECTION_DEBUG_UI", "").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }:
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())
                else:
                    st.caption(str(exc)[:240])

with st.expander("Excel format", expanded=False):
    st.caption("Quick reference — full help is under Help → Workbook quick reference.")
    st.markdown(
        """
**Native platform:** `Collars` + `Lithology` sheets with `hole_id`, `easting`, `northing`, etc.
Optional `unit_order` column (1 = shallowest) for repeated lithology codes.

**Optional sheets:** `Water`, `Screens`, `Gradients`, `Correlations`, `Deviations`,
`Environmental`, `Faults`, `Unconformities`.

Use **Help → Workbook quick reference** (or `docs/help/workbook-quick.md`) for a short guide,
and `docs/workbook-format.md` for the full schema.

**Field export:** single `Lithology` sheet with `Label`, `Depth` (e.g. `0.00-2.00m`),
`Lithology`, `Lat`, `Long` — auto-converted to UTM on import.

Column names are fuzzy-matched for native workbooks (e.g. `BH`, `RL`, `TD`, `CLY`).
        """
    )
