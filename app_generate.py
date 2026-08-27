"""Generate step: profile display and downloads."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from app_common import _display_svg, _render_overlap_warnings, _render_profile_chips
from app_services import cached_build_section_bundle, cached_build_section_exports
from batch_export import build_batch_zip, export_binder_pdf
from docx_export import build_figure_docx_bytes
from export_framing import ExportFramingConfig, build_export_filename, build_report_package_bytes, png_clipboard_html, save_exports_to_directory
from models import ConsultingTitleBlock
from ui_helpers import export_metadata_payload, sanitize_filename

try:
    from ops_audit import audit_event as _audit_event
except ImportError:  # pragma: no cover - ops optional until landed
    def _audit_event(event: str, **fields: object) -> None:
        return None


def _ensure_png_export() -> bool:
    return _ensure_both_exports()


def _ensure_pdf_export() -> bool:
    return _ensure_both_exports()


def _ensure_both_exports() -> bool:
    subset_json = st.session_state.get("section_build_subset_json")
    request_json = st.session_state.get("section_build_request_json")
    if not subset_json or not request_json:
        st.error("Generate the section first, then Prepare.")
        return False
    png_data = st.session_state.get("png_bytes")
    pdf_data = st.session_state.get("pdf_bytes")
    if png_data and pdf_data:
        return False
    png_bytes, pdf_bytes = cached_build_section_exports(subset_json, request_json)
    st.session_state.png_bytes = png_bytes
    st.session_state.pdf_bytes = pdf_bytes
    return True


def _ensure_all_exports() -> tuple[bytes, bytes, bytes]:
    subset_json = st.session_state.get("section_build_subset_json")
    request_json = st.session_state.get("section_build_request_json")
    if not subset_json or not request_json:
        st.error("Generate the section first, then Prepare.")
        return b"", b"", b""
    svg_bytes = st.session_state.get("svg_bytes") or b""
    png_bytes = st.session_state.get("png_bytes") or b""
    pdf_bytes = st.session_state.get("pdf_bytes") or b""
    if png_bytes and pdf_bytes:
        return svg_bytes, png_bytes, pdf_bytes
    bundle = cached_build_section_bundle(subset_json, request_json)
    svg_bytes = bundle[0] or svg_bytes
    png_bytes = bundle[1]
    pdf_bytes = bundle[2]
    st.session_state.png_bytes = png_bytes
    st.session_state.pdf_bytes = pdf_bytes
    if svg_bytes:
        st.session_state.svg_bytes = svg_bytes
    return svg_bytes, png_bytes, pdf_bytes


def _audit_section_export(fmt: str, section_title: str) -> None:
    _audit_event(
        "section_exported",
        format=fmt,
        section_title=section_title,
        workbook=st.session_state.get("uploaded_name"),
    )


def _export_stem(
    *,
    section_title: str,
    export_framing: ExportFramingConfig | None,
    consulting_title_block: ConsultingTitleBlock | None,
    transect_label: str | None,
) -> str:
    framing = export_framing or ExportFramingConfig()
    figure_number = consulting_title_block.figure_number if consulting_title_block else ""
    project_number = consulting_title_block.project_number if consulting_title_block else ""
    return build_export_filename(
        pattern=framing.filename_pattern,
        section_title=section_title,
        figure_number=figure_number,
        project_number=project_number,
        transect_label=transect_label or section_title,
        revision=framing.export_revision,
        draft=framing.show_draft_watermark,
    )


def _consulting_field_map(
    consulting_title_block: ConsultingTitleBlock | None,
) -> dict[str, str]:
    if consulting_title_block is None:
        return {}
    return {
        "figure_number": consulting_title_block.figure_number,
        "project_number": consulting_title_block.project_number,
        "prepared_for": consulting_title_block.prepared_for,
        "prepared_by": consulting_title_block.prepared_by,
        "revised": consulting_title_block.revised,
    }


def _render_batch_export(
    *,
    section_title: str,
    export_framing: ExportFramingConfig | None,
    is_stale: bool,
) -> None:
    labels_raw = str(st.session_state.get("batch_transect_labels", "")).strip()
    if not labels_raw:
        return
    labels = [line.strip() for line in labels_raw.splitlines() if line.strip()]
    if not labels:
        return
    st.caption(f"Batch transects configured: {len(labels)}")
    if is_stale:
        st.info("Regenerate the current section before batch export.")
        return
    if st.button("Prepare batch ZIP (current section × labels)", key="prepare_batch_zip"):
        svg_bytes, png_bytes, pdf_bytes = _ensure_all_exports()
        entries = []
        for label in labels:
            stem = sanitize_filename(
                f"{_export_stem(section_title=section_title, export_framing=export_framing, consulting_title_block=None, transect_label=label)}_{label}"
            )
            entries.append((stem, svg_bytes, png_bytes, pdf_bytes))
        binder = export_binder_pdf([pdf for _, _, _, pdf in entries if pdf])
        zip_bytes = build_batch_zip(entries, binder_pdf=binder or None)
        st.download_button(
            "Download batch ZIP",
            data=zip_bytes,
            file_name=f"{sanitize_filename(section_title)}_batch.zip",
            mime="application/zip",
            key="download_batch_zip",
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
    export_framing: ExportFramingConfig | None = None,
    consulting_title_block: ConsultingTitleBlock | None = None,
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

    base = _export_stem(
        section_title=section_title,
        export_framing=export_framing,
        consulting_title_block=consulting_title_block,
        transect_label=transect_label,
    )
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
                if _ensure_both_exports():
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

    if not is_stale and rasters_ready and png_data:
        components.html(png_clipboard_html(png_data), height=48)

    package_col1, package_col2 = st.columns(2)
    metadata = export_metadata_payload(
        section_title=section_title,
        preset_label=preset_label,
        vertical_exaggeration=vertical_exaggeration,
        hole_count=st.session_state.get("section_hole_count"),
        transect_label=transect_label,
        overlap_warnings=st.session_state.get("polygon_overlap_warnings") or [],
        consulting_fields=_consulting_field_map(consulting_title_block),
    )
    with package_col1:
        if not is_stale and parse_result_available:
            if st.button("Build report package (ZIP)", key="build_report_package", width="stretch"):
                svg_bytes, png_bytes, pdf_bytes = _ensure_all_exports()
                caption = st.session_state.get("ai_figure_caption") or section_title
                docx_bytes = b""
                if png_bytes:
                    try:
                        docx_bytes = build_figure_docx_bytes(
                            png_bytes=png_bytes,
                            caption=str(caption),
                            title=section_title,
                            metadata=metadata,
                        )
                    except RuntimeError as exc:
                        st.warning(str(exc))
                zip_bytes = build_report_package_bytes(
                    stem=base,
                    svg_bytes=svg_bytes,
                    png_bytes=png_bytes,
                    pdf_bytes=pdf_bytes,
                    metadata=metadata,
                    docx_bytes=docx_bytes or None,
                )
                st.session_state["report_package_bytes"] = zip_bytes
            zip_payload = st.session_state.get("report_package_bytes")
            if zip_payload:
                st.download_button(
                    "Download report package",
                    data=zip_payload,
                    file_name=f"{base}_package.zip",
                    mime="application/zip",
                    key="download_report_package",
                    width="stretch",
                )
    with package_col2:
        output_dir = str(st.session_state.get("export_output_dir", "")).strip()
        if output_dir and not is_stale and parse_result_available:
            if st.button("Save exports to folder", key="save_exports_folder", width="stretch"):
                svg_bytes, png_bytes, pdf_bytes = _ensure_all_exports()
                written = save_exports_to_directory(
                    output_dir,
                    stem=base,
                    svg_bytes=svg_bytes,
                    png_bytes=png_bytes,
                    pdf_bytes=pdf_bytes,
                    metadata=metadata,
                )
                st.success(f"Saved {len(written)} file(s) to {output_dir}")

    _render_batch_export(
        section_title=section_title,
        export_framing=export_framing,
        is_stale=is_stale,
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
                    if _ensure_png_export():
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
                    if _ensure_pdf_export():
                        st.rerun()
            if rasters_ready and png_data:
                try:
                    docx_bytes = build_figure_docx_bytes(
                        png_bytes=png_data,
                        caption=str(st.session_state.get("ai_figure_caption") or section_title),
                        title=section_title,
                        metadata=metadata,
                    )
                    st.download_button(
                        "Download Word figure pack (.docx)",
                        data=docx_bytes,
                        file_name=f"{base}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key="download_docx_pack",
                    )
                except RuntimeError as exc:
                    st.caption(str(exc))
