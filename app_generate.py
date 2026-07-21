"""Generate step: profile display and downloads."""

from __future__ import annotations

import streamlit as st

from app_common import _display_svg, _render_overlap_warnings, _render_profile_chips
from app_services import (
    cached_build_section_exports,
    cached_build_section_pdf,
    cached_build_section_png,
)
from ui_helpers import sanitize_filename

try:
    from ops_audit import audit_event as _audit_event
except ImportError:  # pragma: no cover - ops optional until landed
    def _audit_event(event: str, **fields: object) -> None:
        return None


def _ensure_png_export() -> None:
    """Build PNG on demand from the last Generate cache keys."""
    subset_json = st.session_state.get("section_build_subset_json")
    request_json = st.session_state.get("section_build_request_json")
    if not subset_json or not request_json:
        return
    if st.session_state.get("png_bytes"):
        return
    st.session_state.png_bytes = cached_build_section_png(subset_json, request_json)


def _ensure_pdf_export() -> None:
    """Build PDF on demand from the last Generate cache keys."""
    subset_json = st.session_state.get("section_build_subset_json")
    request_json = st.session_state.get("section_build_request_json")
    if not subset_json or not request_json:
        return
    if st.session_state.get("pdf_bytes"):
        return
    st.session_state.pdf_bytes = cached_build_section_pdf(subset_json, request_json)


def _ensure_both_exports() -> None:
    """Build PNG and PDF together from the last Generate cache keys."""
    subset_json = st.session_state.get("section_build_subset_json")
    request_json = st.session_state.get("section_build_request_json")
    if not subset_json or not request_json:
        return
    png_data = st.session_state.get("png_bytes")
    pdf_data = st.session_state.get("pdf_bytes")
    if png_data and pdf_data:
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
    preset_label: str | None = None,
    render_layout: str | None = None,
    transect_label: str | None = None,
) -> None:
    """Render profile chips, SVG, and SVG/PNG/PDF downloads."""
    if st.session_state.svg_bytes is None:
        return

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Cross-Section Profile")
    png_ready = bool(st.session_state.get("png_bytes"))
    pdf_ready = bool(st.session_state.get("pdf_bytes"))
    _render_profile_chips(
        interpretation_mode=interpretation_mode,
        vertical_exaggeration=vertical_exaggeration,
        hole_count=st.session_state.section_hole_count,
        polygon_count=st.session_state.section_polygon_count,
        is_stale=is_stale,
        preset_label=preset_label,
        render_layout=render_layout,
        transect_label=transect_label,
        png_ready=png_ready,
        pdf_ready=pdf_ready,
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
    png_data = st.session_state.get("png_bytes")
    pdf_data = st.session_state.get("pdf_bytes")
    rasters_ready = bool(png_data and pdf_data)
    dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 1])
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
        if rasters_ready:
            st.download_button(
                label="Download PNG" + (" (stale)" if is_stale else ""),
                data=png_data or b"",
                file_name=f"{base}.png",
                mime="image/png",
                width="stretch",
                disabled=is_stale,
                on_click=_audit_section_export,
                kwargs={"fmt": "png", "section_title": section_title},
            )
        elif not is_stale and parse_result_available:
            if st.button("Prepare PNG & PDF", key="prepare_both_exports", width="stretch"):
                _ensure_both_exports()
                st.rerun()
        else:
            st.download_button(
                label="Download PNG" + (" (stale)" if is_stale else ""),
                data=b"",
                file_name=f"{base}.png",
                mime="image/png",
                width="stretch",
                disabled=True,
            )
    with dl_col3:
        if rasters_ready:
            st.download_button(
                label="Download PDF" + (" (stale)" if is_stale else ""),
                data=pdf_data or b"",
                file_name=f"{base}.pdf",
                mime="application/pdf",
                width="stretch",
                disabled=is_stale,
                on_click=_audit_section_export,
                kwargs={"fmt": "pdf", "section_title": section_title},
            )
        elif not is_stale and parse_result_available:
            st.caption("PNG and PDF build together.")
        else:
            st.download_button(
                label="Download PDF" + (" (stale)" if is_stale else ""),
                data=b"",
                file_name=f"{base}.pdf",
                mime="application/pdf",
                width="stretch",
                disabled=True,
            )
    if is_stale:
        st.caption(
            "SVG may be stale — regenerate first. Prepare again after Generate for PNG/PDF deliverables."
        )
    else:
        st.caption(
            "SVG is ready after Generate. Use **Prepare PNG & PDF** for deliverables "
            "(raster exports are skipped until you need them)."
        )
    if not is_stale and parse_result_available:
        with st.expander("Prepare formats separately", expanded=False):
            sep1, sep2 = st.columns(2)
            with sep1:
                if png_data:
                    st.download_button(
                        label="Download PNG",
                        data=png_data,
                        file_name=f"{base}.png",
                        mime="image/png",
                        width="stretch",
                        on_click=_audit_section_export,
                        kwargs={"fmt": "png", "section_title": section_title},
                    )
                elif st.button("Prepare PNG only", key="prepare_png_export", width="stretch"):
                    _ensure_png_export()
                    st.rerun()
            with sep2:
                if pdf_data:
                    st.download_button(
                        label="Download PDF",
                        data=pdf_data,
                        file_name=f"{base}.pdf",
                        mime="application/pdf",
                        width="stretch",
                        on_click=_audit_section_export,
                        kwargs={"fmt": "pdf", "section_title": section_title},
                    )
                elif st.button("Prepare PDF only", key="prepare_pdf_export", width="stretch"):
                    _ensure_pdf_export()
                    st.rerun()
