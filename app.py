"""Streamlit application for borehole cross-section generation."""

from __future__ import annotations

import logging
import traceback

import streamlit as st

from app_build import collect_section_build_request, generate_cross_section
from app_common import _active_transect_selection, _render_hero
from app_configure import render_configure_step, render_transect_sidebar
from app_generate import render_profile_and_downloads
from app_sidebar import render_sidebar
from app_state import init_session_defaults
from app_styles import APP_CSS
from app_upload import handle_workbook_upload, render_welcome_card
from app_validate import render_validate_step
from models import ParseResult
from ui_helpers import parse_coordinate_lines, svg_display_meta, workflow_stage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Cross Section Studio",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(APP_CSS, unsafe_allow_html=True)
init_session_defaults()

with st.sidebar:
    sidebar = render_sidebar()

_render_hero(
    workflow_stage(
        has_upload=sidebar.uploaded is not None,
        has_parse_result=st.session_state.parse_result is not None,
        has_profile=st.session_state.svg_bytes is not None,
    )
)

parse_result: ParseResult | None = st.session_state.parse_result

if sidebar.uploaded is None:
    render_welcome_card()
else:
    parse_result = handle_workbook_upload(
        sidebar.uploaded,
        selected_profile_key=sidebar.selected_profile_key,
        override_id=sidebar.override_id,
        default_elevation_m=sidebar.default_elevation_m,
        target_crs=sidebar.target_crs,
    )
    render_validate_step()

configure_state = None
if parse_result is not None:
    hole_ids = list(st.session_state.hole_ids)
    with st.sidebar:
        selected_holes, coordinate_text = render_transect_sidebar(
            parse_result,
            hole_ids,
            sidebar.transect_mode,
        )
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
    )

    regenerate_requested = st.session_state.get("_regenerate_requested", False)
    gen_col1, gen_col2 = st.columns([1, 3])
    with gen_col1:
        generate_clicked = st.button(
            "Generate Cross-Section",
            type="primary",
            disabled=not configure_state.can_generate,
            width="stretch",
        )
        if regenerate_requested and configure_state.can_generate:
            generate_clicked = True
            st.session_state.pop("_regenerate_requested", None)
    with gen_col2:
        if configure_state.can_generate:
            st.caption("Builds the profile from sidebar style and transect settings.")
        elif configure_state.blocking:
            st.caption("Fix blocking QA errors before generating.")
        else:
            st.caption("Enable warnings override above or resolve QA issues.")

    if generate_clicked:
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
                render_layout=sidebar.render_layout,
                track_width_m=sidebar.track_width_m,
                coordinate_reference=coordinate_reference,
                uses_placeholder_elevation=uses_placeholder,
                elevation_mode=configure_state.elevation_mode,
                report_preset=sidebar.report_preset,
                consulting_title_block=sidebar.consulting_title_block,
                selection=selection,
            )
            if build_request is None or cache_key is None:
                raise ValueError("Select at least two holes for the transect")

            svg_bytes, png_bytes, _, polygon_count, lithology_codes, overlap_warnings = (
                generate_cross_section(
                    parse_result,
                    list(transect_points),
                    active_hole_ids,
                    build_request,
                    sidebar.offset_warning_m,
                    lithology_index=st.session_state.lithology_index,
                )
            )
            st.session_state.svg_bytes = svg_bytes
            st.session_state.png_bytes = png_bytes
            st.session_state.pdf_bytes = None
            st.session_state.svg_display_meta = svg_display_meta(svg_bytes)
            st.session_state.section_lithology_codes = lithology_codes
            st.session_state.section_polygon_count = polygon_count
            st.session_state.section_hole_count = len(active_hole_ids)
            st.session_state.polygon_overlap_warnings = overlap_warnings
            st.session_state.render_cache_key = cache_key
            if sidebar.interpretation_mode == "borehole_only":
                st.success(
                    f"Generated observed-data cross-section across {len(active_hole_ids)} boreholes."
                )
            else:
                st.success(
                    f"Generated cross-section with {polygon_count} geological polygons "
                    f"across {len(active_hole_ids)} boreholes."
                )
        except Exception as exc:
            logger.exception("Cross-section generation failed")
            st.error(str(exc))
            with st.expander("Error details"):
                st.code(traceback.format_exc())

if st.session_state.svg_bytes is not None:
    is_stale = False
    if st.session_state.render_cache_key is not None and parse_result is not None and configure_state:
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
        )
        is_stale = current_key is not None and current_key != st.session_state.render_cache_key

    render_profile_and_downloads(
        section_title=sidebar.section_title,
        interpretation_mode=sidebar.interpretation_mode,
        vertical_exaggeration=sidebar.vertical_exaggeration,
        is_stale=is_stale,
        parse_result_available=parse_result is not None,
    )

with st.expander("Excel format"):
    st.markdown(
        """
**Native platform:** `Collars` + `Lithology` sheets with `hole_id`, `easting`, `northing`, etc.
Optional `unit_order` column (1 = shallowest) for repeated lithology codes.

**Optional sheets:** `Water`, `Screens`, `Gradients`, `Correlations`, `Deviations`,
`Environmental`, `Faults`, `Unconformities`. See `docs/workbook-format.md`.

**Field export:** single `Lithology` sheet with `Label`, `Depth` (e.g. `0.00-2.00m`),
`Lithology`, `Lat`, `Long` — auto-converted to UTM on import.

Column names are fuzzy-matched for native workbooks (e.g. `BH`, `RL`, `TD`, `CLY`).
        """
    )
