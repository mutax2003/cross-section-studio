"""Generate step: profile display and downloads."""

from __future__ import annotations

import streamlit as st

from app_common import _display_svg, _render_overlap_warnings, _render_profile_chips
from app_services import cached_build_section_exports
from ui_helpers import sanitize_filename

try:
    from ops_audit import audit_event as _audit_event
except ImportError:  # pragma: no cover - ops optional until landed
    def _audit_event(event: str, **fields: object) -> None:
        return None


def _ensure_raster_exports() -> None:
    """Build PNG/PDF on demand from the last Generate cache keys."""
    subset_json = st.session_state.get("section_build_subset_json")
    request_json = st.session_state.get("section_build_request_json")
    if not subset_json or not request_json:
        return
    if st.session_state.get("png_bytes") and st.session_state.get("pdf_bytes"):
        return
    png_bytes, pdf_bytes = cached_build_section_exports(subset_json, request_json)
    st.session_state.png_bytes = png_bytes
    st.session_state.pdf_bytes = pdf_bytes


def _audit_section_export(fmt: str, section_title: str) -> None:
    _audit_event(
        "section_exported",
        format=fmt,
        section_title=section_title,
        workbook=st.session_state.get("uploaded_name"),
    )


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
            '<div class="stale-banner" tabindex="0">Settings changed since the last build — '
            "click <strong>Generate Cross-Section</strong> to refresh before download.</div>",
            unsafe_allow_html=True,
        )
        if parse_result_available and st.button(
            "Generate Cross-Section",
            type="primary",
            key="regenerate_stale",
        ):
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
            on_click=_audit_section_export,
            kwargs={"fmt": "svg", "section_title": section_title},
        )
    with dl_col2:
        png_data = st.session_state.get("png_bytes")
        if not png_data and not is_stale and parse_result_available:
            if st.button("Prepare PNG", key="prepare_png_export", width="stretch"):
                _ensure_raster_exports()
                st.rerun()
        else:
            st.download_button(
                label="Download PNG" + (" (stale)" if is_stale else ""),
                data=png_data or b"",
                file_name=f"{base}.png",
                mime="image/png",
                width="stretch",
                disabled=is_stale or not png_data,
                on_click=_audit_section_export,
                kwargs={"fmt": "png", "section_title": section_title},
            )
    with dl_col3:
        pdf_data = st.session_state.get("pdf_bytes")
        if not pdf_data and not is_stale and parse_result_available:
            if st.button("Prepare PDF", key="prepare_pdf_export", width="stretch"):
                _ensure_raster_exports()
                st.rerun()
        else:
            st.download_button(
                label="Download PDF" + (" (stale)" if is_stale else ""),
                data=pdf_data or b"",
                file_name=f"{base}.pdf",
                mime="application/pdf",
                width="stretch",
                disabled=is_stale or not pdf_data,
                on_click=_audit_section_export,
                kwargs={"fmt": "pdf", "section_title": section_title},
            )
    with dl_col4:
        st.caption(
            "SVG is ready after Generate. Prepare PNG/PDF once for deliverables "
            "(skips raster work until you need them)."
        )
