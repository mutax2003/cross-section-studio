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


def _build_request_json() -> tuple[object, object]:
    return (
        st.session_state.get("section_build_subset_json"),
        st.session_state.get("section_build_request_json"),
    )


def _session_export_triple() -> tuple[bytes, bytes, bytes]:
    return (
        st.session_state.get("svg_bytes") or b"",
        st.session_state.get("png_bytes") or b"",
        st.session_state.get("pdf_bytes") or b"",
    )


def _ensure_both_exports() -> bool:
    """Prepare PNG+PDF when missing. Returns True if newly built."""
    subset_json, request_json = _build_request_json()
    if not subset_json or not request_json:
        st.error("Generate the section first, then Prepare.")
        return False
    _, png_data, pdf_data = _session_export_triple()
    if png_data and pdf_data:
        return False
    png_bytes, pdf_bytes = cached_build_section_exports(subset_json, request_json)
    st.session_state.png_bytes = png_bytes
    st.session_state.pdf_bytes = pdf_bytes
    st.session_state.pop("figure_docx_bytes", None)
    st.session_state.pop("_figure_docx_cache_token", None)
    st.session_state.pop("report_package_bytes", None)
    return True


def _ensure_all_exports() -> tuple[bytes, bytes, bytes]:
    svg_bytes, png_bytes, pdf_bytes = _session_export_triple()
    if png_bytes and pdf_bytes:
        return svg_bytes, png_bytes, pdf_bytes
    subset_json, request_json = _build_request_json()
    if not subset_json or not request_json:
        st.error("Generate the section first, then Prepare.")
        return svg_bytes, b"", b""
    if svg_bytes:
        _ensure_both_exports()
        return _session_export_triple()
    bundle = cached_build_section_bundle(subset_json, request_json)
    svg_bytes = bundle[0] or svg_bytes
    st.session_state.png_bytes = bundle[1]
    st.session_state.pdf_bytes = bundle[2]
    if svg_bytes:
        st.session_state.svg_bytes = svg_bytes
    return _session_export_triple()


def _cached_docx_bytes(
    png_bytes: bytes,
    *,
    section_title: str,
    metadata: dict[str, object],
) -> bytes:
    cache_token = st.session_state.get("render_cache_key")
    if st.session_state.get("_figure_docx_cache_token") == cache_token:
        return st.session_state.get("figure_docx_bytes") or b""
    docx_bytes = _build_docx_if_ready(
        png_bytes,
        section_title=section_title,
        metadata=metadata,
    )
    st.session_state["figure_docx_bytes"] = docx_bytes
    st.session_state["_figure_docx_cache_token"] = cache_token
    return docx_bytes


def _build_docx_if_ready(
    png_bytes: bytes,
    *,
    section_title: str,
    metadata: dict[str, object],
) -> bytes:
    if not png_bytes:
        return b""
    try:
        return build_figure_docx_bytes(
            png_bytes=png_bytes,
            caption=str(st.session_state.get("ai_figure_caption") or section_title),
            title=section_title,
            metadata=metadata,
        )
    except RuntimeError as exc:
        st.warning(str(exc))
        return b""


def _audit_section_export(fmt: str, section_title: str) -> None:
    _audit_event(
        "section_exported",
        format=fmt,
        section_title=section_title,
        workbook=st.session_state.get("uploaded_name"),
    )


def _format_download(
    *,
    label: str,
    data: bytes,
    file_name: str,
    mime: str,
    fmt: str,
    section_title: str,
    is_stale: bool,
    ready: bool,
    primary: bool = False,
    key: str | None = None,
) -> None:
    stale_suffix = " (stale)" if is_stale and ready else ""
    kwargs: dict[str, object] = {
        "label": label + stale_suffix,
        "data": data if ready else b"",
        "file_name": file_name,
        "mime": mime,
        "width": "stretch",
        "disabled": (not ready) or is_stale,
    }
    if primary:
        kwargs["type"] = "primary"
    if key:
        kwargs["key"] = key
    if ready:
        kwargs["on_click"] = _audit_section_export
        kwargs["kwargs"] = {"fmt": fmt, "section_title": section_title}
    st.download_button(**kwargs)


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
    st.markdown("**Filename copies (batch ZIP)**")
    st.caption(
        f"{len(labels)} label(s) — packages the **current** section figure under each "
        "filename stem (does not rebuild separate transects)."
    )
    if is_stale:
        st.info("Regenerate the current section before batch export.")
        return
    if st.button("Build filename-copy ZIP", key="prepare_batch_zip"):
        svg_bytes, png_bytes, pdf_bytes = _ensure_all_exports()
        entries = [
            (
                sanitize_filename(
                    _export_stem(
                        section_title=section_title,
                        export_framing=export_framing,
                        consulting_title_block=None,
                        transect_label=label,
                    )
                ),
                svg_bytes,
                png_bytes,
                pdf_bytes,
            )
            for label in labels
        ]
        st.session_state["batch_package_bytes"] = build_batch_zip(
            entries,
            binder_pdf=export_binder_pdf([pdf_bytes] if pdf_bytes else []) or None,
        )
    batch_payload = st.session_state.get("batch_package_bytes")
    if batch_payload:
        st.download_button(
            "Download filename-copy ZIP",
            data=batch_payload,
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
    rasters_ready = png_ready and pdf_ready
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
    svg_bytes, png_data, pdf_data = _session_export_triple()
    metadata = export_metadata_payload(
        section_title=section_title,
        preset_label=preset_label,
        vertical_exaggeration=vertical_exaggeration,
        hole_count=st.session_state.get("section_hole_count"),
        transect_label=transect_label,
        overlap_warnings=st.session_state.get("polygon_overlap_warnings") or [],
        consulting_fields=_consulting_field_map(consulting_title_block),
    )

    st.markdown("**Quick downloads**")
    dl_col1, dl_col2, dl_col3 = st.columns(3)
    with dl_col1:
        _format_download(
            label="SVG (CAD / review)",
            data=st.session_state.svg_bytes or b"",
            file_name=f"{base}.svg",
            mime="image/svg+xml",
            fmt="svg",
            section_title=section_title,
            is_stale=is_stale,
            ready=True,
            primary=True,
        )
    with dl_col2:
        _format_download(
            label="PNG (Word / slides)",
            data=png_data or b"",
            file_name=f"{base}.png",
            mime="image/png",
            fmt="png",
            section_title=section_title,
            is_stale=is_stale,
            ready=rasters_ready,
        )
    with dl_col3:
        _format_download(
            label="PDF (print)",
            data=pdf_data or b"",
            file_name=f"{base}.pdf",
            mime="application/pdf",
            fmt="pdf",
            section_title=section_title,
            is_stale=is_stale,
            ready=rasters_ready,
        )

    if not is_stale and parse_result_available and not rasters_ready:
        st.info(
            "SVG is ready. Click **Prepare deliverables** once to build PNG, PDF, "
            "Word, clipboard, and package options (one draw)."
        )
        if st.button(
            "Prepare deliverables (PNG · PDF · Word · package)",
            type="primary",
            key="prepare_both_exports",
            width="stretch",
        ):
            if _ensure_both_exports():
                st.rerun()
    elif is_stale:
        st.caption("Regenerate before preparing or downloading deliverables.")
    else:
        st.caption(
            "SVG after Generate · PNG/PDF for reports · package ZIP for handoff. "
            "Framing (page size, DPI, DRAFT, CAD SVG tag) is in the sidebar."
        )

    if not is_stale and rasters_ready and parse_result_available:
        st.markdown("**Drafter package**")
        pack1, pack2, pack3 = st.columns(3)
        docx_bytes = _cached_docx_bytes(
            png_data or b"",
            section_title=section_title,
            metadata=metadata,
        )
        with pack1:
            if docx_bytes:
                st.download_button(
                    "Word figure (.docx)",
                    data=docx_bytes,
                    file_name=f"{base}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="download_docx_pack",
                    width="stretch",
                )
            else:
                st.caption("Word pack needs python-docx.")
            if png_data:
                components.html(png_clipboard_html(png_data), height=48)
        with pack2:
            if st.button("Build report ZIP", key="build_report_package", width="stretch"):
                svg_bytes, png_bytes, pdf_bytes = _session_export_triple()
                st.session_state["report_package_bytes"] = build_report_package_bytes(
                    stem=base,
                    svg_bytes=svg_bytes,
                    png_bytes=png_bytes,
                    pdf_bytes=pdf_bytes,
                    metadata=metadata,
                    docx_bytes=docx_bytes or None,
                )
            zip_payload = st.session_state.get("report_package_bytes")
            if zip_payload:
                st.download_button(
                    "Download report ZIP",
                    data=zip_payload,
                    file_name=f"{base}_package.zip",
                    mime="application/zip",
                    key="download_report_package",
                    width="stretch",
                )
            else:
                st.caption("ZIP = SVG + PNG + PDF + metadata (+ Word).")
        with pack3:
            output_dir = str(st.session_state.get("export_output_dir", "")).strip()
            if output_dir:
                if st.button("Save to project folder", key="save_exports_folder", width="stretch"):
                    svg_bytes, png_bytes, pdf_bytes = _session_export_triple()
                    written = save_exports_to_directory(
                        output_dir,
                        stem=base,
                        svg_bytes=svg_bytes,
                        png_bytes=png_bytes,
                        pdf_bytes=pdf_bytes,
                        metadata=metadata,
                        docx_bytes=docx_bytes or None,
                    )
                    st.success(f"Saved {len(written)} file(s) to {output_dir}")
            else:
                st.caption("Set **Export output folder** in sidebar framing to save files.")

    _render_batch_export(
        section_title=section_title,
        export_framing=export_framing,
        is_stale=is_stale,
    )
