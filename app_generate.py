"""Generate step: profile display and downloads."""

from __future__ import annotations

import streamlit as st

from app_common import _display_svg, _render_overlap_warnings, _render_profile_chips
from app_services import cached_build_section_pdf
from ui_helpers import sanitize_filename


def render_profile_and_downloads(
    *,
    section_title: str,
    interpretation_mode: str,
    vertical_exaggeration: float,
    is_stale: bool,
    parse_result_available: bool,
) -> None:
    """Render profile chips, SVG, and SVG/PNG/PDF downloads."""
    if st.session_state.svg_bytes is None:
        return

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Cross-Section Profile")
    _render_profile_chips(
        interpretation_mode=interpretation_mode,
        vertical_exaggeration=vertical_exaggeration,
        hole_count=st.session_state.section_hole_count,
        polygon_count=st.session_state.section_polygon_count,
        is_stale=is_stale,
    )
    if is_stale:
        st.markdown(
            '<div class="stale-banner" tabindex="0">Settings changed — click <strong>Regenerate</strong> '
            "or <strong>Generate Cross-Section</strong> to refresh.</div>",
            unsafe_allow_html=True,
        )
        if parse_result_available and st.button("Regenerate", type="primary", key="regenerate_stale"):
            st.session_state["_regenerate_requested"] = True
            st.rerun()
    _render_overlap_warnings(st.session_state.polygon_overlap_warnings)
    _display_svg(st.session_state.svg_bytes)
    st.markdown("</div>", unsafe_allow_html=True)

    base = sanitize_filename(section_title)
    dl_col1, dl_col2, dl_col3, dl_col4 = st.columns([1, 1, 1, 2])
    with dl_col1:
        st.download_button(
            label="Download SVG" + (" (stale)" if is_stale else ""),
            data=st.session_state.svg_bytes,
            file_name=f"{base}.svg",
            mime="image/svg+xml",
            type="primary",
            width="stretch",
            disabled=is_stale,
        )
    with dl_col2:
        png_data = st.session_state.get("png_bytes")
        st.download_button(
            label="Download PNG" + (" (stale)" if is_stale else ""),
            data=png_data or b"",
            file_name=f"{base}.png",
            mime="image/png",
            width="stretch",
            disabled=is_stale or not png_data,
        )
    with dl_col3:
        pdf_data = st.session_state.get("pdf_bytes")
        if (
            not pdf_data
            and not is_stale
            and st.session_state.get("section_build_subset_json")
            and st.session_state.get("section_build_request_json")
        ):
            if st.button("Prepare PDF report", key="prepare_pdf_report", disabled=is_stale):
                pdf_data = cached_build_section_pdf(
                    st.session_state.section_build_subset_json,
                    st.session_state.section_build_request_json,
                )
                st.session_state.pdf_bytes = pdf_data
                st.rerun()
        st.download_button(
            label="Download PDF" + (" (stale)" if is_stale else ""),
            data=pdf_data or b"",
            file_name=f"{base}.pdf",
            mime="application/pdf",
            width="stretch",
            disabled=is_stale or not pdf_data,
        )
    with dl_col4:
        st.caption("SVG for editing; PNG (300 DPI) and PDF report bundle for deliverables.")
