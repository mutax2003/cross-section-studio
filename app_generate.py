"""Generate step: profile display and downloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import streamlit as st

from app_common import _display_svg, _render_overlap_warnings, _render_profile_chips
from app_services import (
    cached_build_section_bundle,
    cached_build_section_exports,
    cached_parse_request,
)
from batch_export import (
    BATCH_DEFAULT_EXPORT_FORMATS,
    build_batch_zip,
    build_one_transect_exports,
    export_binder_pdf,
    split_batch_transect_lines,
)
from docx_export import build_figure_docx_bytes
from export_framing import (
    ExportFramingConfig,
    build_export_filename,
    build_report_package_bytes,
    png_clipboard_html,
    save_exports_to_directory,
)
from models import ConsultingTitleBlock
from pipeline import ALL_EXPORT_FORMATS
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
        st.error("Generate the section first, then prepare deliverables.")
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
        st.error("Generate the section first, then prepare deliverables.")
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
    # The Word pack embeds the caption and metadata, so they are part of the key.
    cache_token = (
        st.session_state.get("render_cache_key"),
        str(st.session_state.get("ai_figure_caption") or ""),
        json.dumps(metadata, sort_keys=True, default=str),
    )
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
    help: str | None = None,
) -> None:
    kwargs: dict[str, object] = {
        "label": label,
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
    if help:
        kwargs["help"] = help
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
    include_transect_label: bool = False,
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
        include_transect_label=include_transect_label,
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
    consulting_title_block: ConsultingTitleBlock | None,
    is_stale: bool,
) -> None:
    specs_raw = str(st.session_state.get("batch_transect_specs", "")).strip()
    if not specs_raw:
        return
    st.markdown("**Batch ZIP (several section lines)**")
    parse_result = st.session_state.get("parse_result")
    known_holes = [c.hole_id for c in parse_result.collars] if parse_result is not None else None
    # Check each line on its own: build the good ones, list the rest.
    specs, skipped_lines = split_batch_transect_lines(specs_raw, known_holes)
    for status in skipped_lines:
        st.caption(f"Skipped — {status.message}")
    if not specs:
        st.warning("No section line in the batch list is ready yet — fix the lines above.")
        return
    st.caption(
        f"{len(specs)} section line(s) — each one is drawn afresh as PNG and PDF "
        "(SVG optional), not copied from the current figure."
    )
    include_svg = st.checkbox(
        "Include SVG in batch ZIP",
        value=False,
        key="batch_include_svg",
        help="SVG files are the slowest to build; leave off if you only need PNG and PDF.",
    )
    if is_stale:
        st.info("Click Generate section first so the batch uses your current figure style.")
        return
    request_json = st.session_state.get("section_build_request_json")
    if parse_result is None or not request_json:
        st.info("Generate a section first, then prepare the batch ZIP.")
        return
    batch_token = hashlib.sha256(
        f"{request_json}\x00{specs_raw}\x00{include_svg}\x00{section_title}".encode()
    ).hexdigest()
    if st.button("Prepare batch ZIP", key="prepare_batch_zip"):
        try:
            base_request = cached_parse_request(request_json)
            formats = ALL_EXPORT_FORMATS if include_svg else BATCH_DEFAULT_EXPORT_FORMATS
            raw_entries = []
            failed: list[str] = []
            with st.spinner(f"Drawing {len(specs)} section line(s)…"):
                for spec in specs:
                    try:
                        raw_entries.append(
                            build_one_transect_exports(
                                parse_result, base_request, spec, export_formats=formats
                            )
                        )
                    except ValueError as exc:  # e.g. too few holes with logs
                        failed.append(f"{spec.label}: {exc}")
            for reason in failed:
                st.warning(f"Left out of the ZIP — {reason}")
            if not raw_entries:
                st.error("None of the section lines could be drawn, so no ZIP was made.")
                return
            entries = [
                (
                    sanitize_filename(
                        _export_stem(
                            section_title=section_title,
                            export_framing=export_framing,
                            consulting_title_block=consulting_title_block,
                            transect_label=label,
                            include_transect_label=True,
                        )
                    ),
                    svg_bytes,
                    png_bytes,
                    pdf_bytes,
                )
                for label, svg_bytes, png_bytes, pdf_bytes in raw_entries
            ]
            pdfs = [pdf for _stem, _svg, _png, pdf in entries if pdf]
            st.session_state["batch_package_bytes"] = build_batch_zip(
                entries,
                binder_pdf=export_binder_pdf(pdfs, cover_title=section_title) or None,
            )
            st.session_state["_batch_package_token"] = batch_token
            left_out = [status.label for status in skipped_lines] + [
                reason.split(":", 1)[0] for reason in failed
            ]
            note = f" Left out: {', '.join(left_out)}." if left_out else ""
            st.success(f"Batch ZIP ready: {len(entries)} section line(s).{note}")
        except Exception as exc:  # noqa: BLE001 — surface any rebuild failure in UI
            st.error(f"Couldn't prepare the batch ZIP: {exc}")
            return
    batch_payload = st.session_state.get("batch_package_bytes")
    if batch_payload and st.session_state.get("_batch_package_token") != batch_token:
        # Specs, output style, or title changed since this ZIP was built.
        st.session_state.pop("batch_package_bytes", None)
        st.session_state.pop("_batch_package_token", None)
        batch_payload = None
    if batch_payload:
        st.download_button(
            "Download batch ZIP",
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
    can_generate: bool = True,
    blocked_reason: str | None = None,
) -> None:
    """Render profile chips, SVG, and SVG/PNG/PDF downloads."""
    if st.session_state.svg_bytes is None:
        return

    png_ready = bool(st.session_state.get("png_bytes"))
    pdf_ready = bool(st.session_state.get("pdf_bytes"))
    rasters_ready = png_ready and pdf_ready
    base = _export_stem(
        section_title=section_title,
        export_framing=export_framing,
        consulting_title_block=consulting_title_block,
        transect_label=transect_label,
        include_transect_label=True,
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
    if is_stale:
        raster_help = "Out of date — click Generate section first."
    elif not rasters_ready:
        raster_help = "Click Prepare deliverables first (builds PNG, PDF and Word in one go)."
    else:
        raster_help = None
    docx_bytes = b""
    if not is_stale and rasters_ready and parse_result_available:
        docx_bytes = _cached_docx_bytes(
            png_data or b"",
            section_title=section_title,
            metadata=metadata,
        )
    prepared_toast = st.session_state.pop("_prepared_toast", False)
    if prepared_toast and rasters_ready:
        st.toast(
            "PDF, PNG and Word file ready" if docx_bytes else "PDF and PNG ready",
            icon="✅",
        )

    # Keyed container: a raw <div> via st.markdown is auto-closed and wraps
    # nothing, so the .section-card chrome never applied to this section.
    with st.container(key="section_card"):
        # Chips and the compact zoom control share one row; the stale/blocked
        # status and the single Generate button live in the strip above.
        chips_col, zoom_col = st.columns([3, 2], vertical_alignment="center")
        with chips_col:
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

        # One toolbar row: Prepare deliverables (until rasters exist) and the
        # downloads, PDF first — it is what goes into client binders. Short
        # labels so nothing truncates at 1024 px; the purpose is in the tooltip.
        show_prepare = not is_stale and parse_result_available and not rasters_ready
        show_word = bool(docx_bytes)
        weights = ([2] if show_prepare else []) + [1, 1, 1] + ([1] if show_word else [])
        columns = list(st.columns(weights, vertical_alignment="center"))
        if show_prepare:
            with columns.pop(0):
                if st.button(
                    "Prepare deliverables",
                    type="primary",
                    key="prepare_both_exports",
                    width="stretch",
                    help="Builds the PDF, PNG and Word file in one go, then unlocks "
                    "the report ZIP and saving to a project folder.",
                ):
                    with st.spinner("Building PDF, PNG and Word file…"):
                        prepared = _ensure_both_exports()
                    if prepared:
                        st.session_state["_prepared_toast"] = True
                        st.rerun()
        with columns.pop(0):
            _format_download(
                label="PDF",
                data=pdf_data or b"",
                file_name=f"{base}.pdf",
                mime="application/pdf",
                fmt="pdf",
                section_title=section_title,
                is_stale=is_stale,
                ready=rasters_ready,
                primary=rasters_ready and not is_stale,
                help=raster_help or "For print",
            )
        with columns.pop(0):
            _format_download(
                label="PNG",
                data=png_data or b"",
                file_name=f"{base}.png",
                mime="image/png",
                fmt="png",
                section_title=section_title,
                is_stale=is_stale,
                ready=rasters_ready,
                help=raster_help or "For Word/slides",
            )
        with columns.pop(0):
            _format_download(
                label="SVG",
                data=st.session_state.svg_bytes or b"",
                file_name=f"{base}.svg",
                mime="image/svg+xml",
                fmt="svg",
                section_title=section_title,
                is_stale=is_stale,
                ready=True,
                help="Out of date — click Generate section first." if is_stale else "Editable in CAD",
            )
        if show_word:
            with columns.pop(0):
                _format_download(
                    label="Word",
                    data=docx_bytes,
                    file_name=f"{base}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    fmt="docx",
                    section_title=section_title,
                    is_stale=is_stale,
                    ready=True,
                    key="download_docx_pack",
                    help="Figure with caption, ready to paste into a report",
                )

        _render_overlap_warnings(st.session_state.polygon_overlap_warnings)
        _display_svg(
            st.session_state.svg_bytes,
            alt_text=(
                f"{section_title}: {transect_label or 'cross-section'}, "
                f"{st.session_state.section_hole_count} boreholes, "
                f"vertical exaggeration {vertical_exaggeration:g}×"
            ),
            zoom_slot=zoom_col,
        )
        if not is_stale and rasters_ready:
            st.caption(
                "PDF and PNG for reports · SVG for CAD · report ZIP for handoff. "
                "Page size, resolution and DRAFT stamp are under Export framing in the sidebar."
            )

    if not is_stale and rasters_ready and parse_result_available:
        st.markdown("**More deliverables**")
        pack1, pack2, pack3 = st.columns(3)
        with pack1:
            if not docx_bytes:
                st.caption("The Word file needs the python-docx add-on on this computer.")
            if png_data:
                st.iframe(png_clipboard_html(png_data), height=48)
        with pack2:
            if st.button("Prepare report ZIP", key="build_report_package", width="stretch"):
                svg_bytes, png_bytes, pdf_bytes = _session_export_triple()
                with st.spinner("Packaging report ZIP…"):
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
                st.caption("ZIP = SVG + PNG + PDF + figure details (+ Word).")
        with pack3:
            output_dir = str(st.session_state.get("export_output_dir", "")).strip()
            target = Path(output_dir).expanduser() if output_dir else None
            if target is not None and not target.is_absolute():
                # A relative path would land in the server's working directory.
                st.warning(
                    f"'{output_dir}' is not a full folder path. "
                    "Enter one like P:\\Projects\\Job\\Figures."
                )
            elif target is not None:
                if st.button("Save to project folder", key="save_exports_folder", width="stretch"):
                    svg_bytes, png_bytes, pdf_bytes = _session_export_triple()
                    try:
                        with st.spinner("Saving files…"):
                            written = save_exports_to_directory(
                                str(target),
                                stem=base,
                                svg_bytes=svg_bytes,
                                png_bytes=png_bytes,
                                pdf_bytes=pdf_bytes,
                                metadata=metadata,
                                docx_bytes=docx_bytes or None,
                            )
                    except PermissionError:
                        st.error(f"No permission to write to {target}. Choose a folder you can write to.")
                    except ValueError as exc:
                        st.error(str(exc))
                    except OSError:
                        st.error(
                            f"Couldn't create or write to {target}. Check that the drive or network "
                            "share is connected and the path is spelled correctly."
                        )
                    else:
                        st.success(f"Saved {len(written)} file(s) to **{target.resolve()}**")
            else:
                st.caption(
                    "Set **Save exports to folder** under sidebar **Export** "
                    "to save files directly."
                )

    _render_batch_export(
        section_title=section_title,
        export_framing=export_framing,
        consulting_title_block=consulting_title_block,
        is_stale=is_stale,
    )
