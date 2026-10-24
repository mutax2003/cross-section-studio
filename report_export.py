"""Multi-format export helpers for cross-section figures."""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from app_identity import APP_NAME, AUTHOR, COPYRIGHT_NOTICE, ORGANIZATION
from export_framing import PAGE_FIGSIZE_IN
from renderer import CrossSectionRenderer
from stratigraphy import GeologicalPolygon


def _merge_pdf_pages(page_pdfs: Sequence[bytes]) -> bytes:
    """Concatenate PDF page streams (pypdf required for 2+ pages)."""
    import logging

    valid = [payload for payload in page_pdfs if payload]
    if not valid:
        return b""
    if len(valid) == 1:
        return valid[0]
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        logging.getLogger(__name__).warning(
            "pypdf is not installed; returning section page only (summary page omitted). "
            "Install pypdf>=4 for two-page section PDFs."
        )
        return valid[0]
    writer = PdfWriter()
    for payload in valid:
        writer.append(PdfReader(BytesIO(payload)))
    # Merging replaces the document info written by matplotlib; restore it.
    writer.add_metadata(
        {
            "/Author": f"{AUTHOR}, {ORGANIZATION}",
            "/Creator": APP_NAME,
            "/Subject": COPYRIGHT_NOTICE,
        }
    )
    out = BytesIO()
    writer.write(out)
    return out.getvalue()


# Text pages (summary, QA notes) are a fixed letter landscape sheet, the size
# a batch binder normalises every page to. ``bbox_inches="tight"`` used to crop
# them to their text, so a long disclaimer made a 19.9 x 8.9 in page.
LETTER_LANDSCAPE_IN = (11.0, 8.5)
_PAGE_MARGIN_IN = 0.75
# Rough average glyph width as a share of the font size (DejaVu Sans).
_CHAR_WIDTH_EM = 0.56


def _pdf_first_page_size_in(pdf_bytes: bytes | None) -> tuple[float, float] | None:
    """Width x height (inches) of the first page of ``pdf_bytes``; None if unreadable."""
    if not pdf_bytes:
        return None
    try:
        from pypdf import PdfReader

        box = PdfReader(BytesIO(pdf_bytes)).pages[0].mediabox
        return float(box.width) / 72.0, float(box.height) / 72.0
    except Exception:  # pypdf missing or an unreadable page: keep the default
        return None


# A text page needs at least this much room each way (margins plus a few
# lines); a tighter figure page falls back to letter in its orientation.
TEXT_PAGE_MIN_SIDE_IN = 4.0


def text_page_size_for(figure_page_in: tuple[float, float] | None) -> tuple[float, float]:
    """Sheet for the QA notes / summary pages that follow a figure page.

    One rule: the text pages take the figure page's own size, so a PDF never
    mixes page sizes. A paper sheet (letter / tabloid, either orientation)
    is matched exactly; a tight / auto-sized figure page is matched to its
    actual canvas size rounded to 0.01 in (QA round 8: 10.99 x 8.43 in
    figure followed by an 11 x 8.5 in QA page). Only a figure page too small
    to hold text (under ``TEXT_PAGE_MIN_SIDE_IN`` either way) gets letter in
    its own orientation.
    """
    if figure_page_in is None:
        return LETTER_LANDSCAPE_IN
    width, height = figure_page_in
    for paper in PAGE_FIGSIZE_IN.values():
        if abs(width - paper[0]) < 0.05 and abs(height - paper[1]) < 0.05:
            return float(paper[0]), float(paper[1])
    if width >= TEXT_PAGE_MIN_SIDE_IN and height >= TEXT_PAGE_MIN_SIDE_IN:
        return round(float(width), 2), round(float(height), 2)
    return LETTER_LANDSCAPE_IN if width >= height else (LETTER_LANDSCAPE_IN[1], LETTER_LANDSCAPE_IN[0])


def _wrap(text: str, *, fontsize: float, width_in: float) -> list[str]:
    chars = max(20, int(width_in * 72.0 / (fontsize * _CHAR_WIDTH_EM)))
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        lines.extend(textwrap.wrap(paragraph, chars) or [""])
    return lines


def _text_pages_pdf(
    blocks: Sequence[tuple[str, float, dict[str, object], float]],
    *,
    page_size_in: tuple[float, float] = LETTER_LANDSCAPE_IN,
) -> bytes:
    """Lay out ``(text, fontsize, text kwargs, indent_in)`` blocks top-down on
    fixed-size pages, wrapping long lines and starting new pages as needed."""
    width_in, height_in = page_size_in
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        fig: Figure | None = None
        y_in = 0.0

        def new_page() -> Figure:
            nonlocal y_in
            page = Figure(figsize=page_size_in)
            y_in = height_in - _PAGE_MARGIN_IN
            return page

        fig = new_page()
        for text, fontsize, kwargs, indent_in in blocks:
            line_in = fontsize * 1.35 / 72.0
            usable = width_in - 2 * _PAGE_MARGIN_IN - indent_in
            for line in _wrap(text, fontsize=fontsize, width_in=usable):
                if y_in - line_in < _PAGE_MARGIN_IN:
                    pdf.savefig(fig)
                    fig = new_page()
                fig.text(
                    (_PAGE_MARGIN_IN + indent_in) / width_in,
                    y_in / height_in,
                    line,
                    fontsize=fontsize,
                    va="top",
                    ha="left",
                    **kwargs,
                )
                y_in -= line_in
            y_in -= line_in * 0.35
        pdf.savefig(fig)
    return buffer.getvalue()


def _bullets(lines: Sequence[str], *, fontsize: float = 9.0) -> list[tuple[str, float, dict[str, object], float]]:
    return [(f"\u2022 {line}", fontsize, {}, 0.2) for line in lines if str(line).strip()]


def build_report_summary_pdf(
    renderer: CrossSectionRenderer,
    *,
    lithology_codes: Sequence[str] | None = None,
    qa_lines: Sequence[str] = (),
    page_size_in: tuple[float, float] = LETTER_LANDSCAPE_IN,
) -> bytes:
    """Page(s) with title, metadata, lithology list, and QA notes (letter landscape
    unless ``page_size_in`` matches the figure page)."""
    bold: dict[str, object] = {"fontweight": "bold"}
    blocks: list[tuple[str, float, dict[str, object], float]] = [
        (f"{APP_NAME} \u2014 Report Summary", 14.0, bold, 0.0),
        (renderer.title, 12.0, bold, 0.0),
    ]
    if renderer.disclaimer:
        blocks.append((renderer.disclaimer, 9.0, {"style": "italic"}, 0.0))
    blocks.extend((line, 9.0, {}, 0.0) for line in renderer._metadata_footer_lines())
    if lithology_codes:
        blocks.append(("Lithology codes:", 10.0, bold, 0.0))
        blocks.extend(_bullets(list(lithology_codes)))
    if qa_lines:
        blocks.append(("QA notes:", 10.0, bold, 0.0))
        blocks.extend(_bullets(list(qa_lines), fontsize=8.5))
    return _text_pages_pdf(blocks, page_size_in=page_size_in)


def build_qa_notes_pdf(
    renderer: CrossSectionRenderer,
    qa_lines: Sequence[str],
    *,
    page_size_in: tuple[float, float] = LETTER_LANDSCAPE_IN,
) -> bytes:
    """QA notes page for a single-sheet (consulting) figure, on ``page_size_in``."""
    bold: dict[str, object] = {"fontweight": "bold"}
    blocks: list[tuple[str, float, dict[str, object], float]] = [
        ("QA notes", 14.0, bold, 0.0),
        (renderer.title, 11.0, bold, 0.0),
        (
            "Notes found while drawing this section (also listed in the app after Generate). "
            "Turn off \u201cInclude QA notes (PDF)\u201d to leave this page out.",
            9.0,
            {"style": "italic"},
            0.0,
        ),
    ]
    blocks.extend(_bullets(list(qa_lines)))
    return _text_pages_pdf(blocks, page_size_in=page_size_in)


def assemble_section_pdf(
    renderer: CrossSectionRenderer,
    section_page_pdf: bytes,
    *,
    lithology_codes: Sequence[str] | None = None,
    qa_lines: Sequence[str] = (),
) -> bytes:
    """The section PDF the export hands out.

    Section sheet / Quick preview: the figure page plus the summary page.
    Consulting styles (one client sheet): the figure page, plus a QA notes
    page only when there are notes (``qa_lines`` is empty when "Include QA
    notes (PDF)" is off). Text pages use the figure page's sheet and
    orientation (``text_page_size_for``).
    """
    if renderer.profile.layout == "consulting_section":
        notes = [line for line in qa_lines if str(line).strip()]
        if not notes:
            return section_page_pdf
        page_size = text_page_size_for(_pdf_first_page_size_in(section_page_pdf))
        return _merge_pdf_pages(
            [section_page_pdf, build_qa_notes_pdf(renderer, notes, page_size_in=page_size)]
        )
    return export_section_pdf(
        renderer,
        [],
        pd.DataFrame(),
        lithology_codes=lithology_codes,
        qa_lines=qa_lines,
        section_page_pdf=section_page_pdf,
    )


def normalize_pdf_pages(
    pdf_bytes: bytes, *, page_size_in: tuple[float, float] = LETTER_LANDSCAPE_IN
) -> bytes:
    """Scale (and centre) every page onto one sheet size, keeping its aspect.

    A batch binder mixed tight section pages (e.g. 17.7 x 5.3 in) with
    letter text pages; each page is fitted onto letter landscape instead.
    Returns the input unchanged without pypdf.
    """
    if not pdf_bytes:
        return pdf_bytes
    try:
        from pypdf import PageObject, PdfReader, PdfWriter, Transformation
    except ImportError:
        return pdf_bytes
    target_w, target_h = page_size_in[0] * 72.0, page_size_in[1] * 72.0
    margin = 0.25 * 72.0
    writer = PdfWriter()
    for page in PdfReader(BytesIO(pdf_bytes)).pages:
        box = page.mediabox
        width, height = float(box.width), float(box.height)
        if abs(width - target_w) < 1.0 and abs(height - target_h) < 1.0:
            writer.add_page(page)
            continue
        scale = min((target_w - 2 * margin) / width, (target_h - 2 * margin) / height)
        sheet = PageObject.create_blank_page(width=target_w, height=target_h)
        tx = (target_w - width * scale) / 2.0 - float(box.left) * scale
        ty = (target_h - height * scale) / 2.0 - float(box.bottom) * scale
        sheet.merge_transformed_page(page, Transformation().scale(scale).translate(tx, ty))
        writer.add_page(sheet)
    writer.add_metadata(
        {
            "/Author": f"{AUTHOR}, {ORGANIZATION}",
            "/Creator": APP_NAME,
            "/Subject": COPYRIGHT_NOTICE,
        }
    )
    out = BytesIO()
    writer.write(out)
    return out.getvalue()


def export_section_pdf(
    renderer: CrossSectionRenderer,
    polygons: list[GeologicalPolygon],
    projected_df: pd.DataFrame,
    *,
    collar_depths: dict[str, float] | None = None,
    water_levels: Sequence | None = None,
    lithology_codes: Sequence[str] | None = None,
    qa_lines: Sequence[str] = (),
    section_figure: Figure | None = None,
    section_page_pdf: bytes | None = None,
) -> bytes:
    """Build a two-page PDF: section figure + legend/metadata summary.

    Prefer ``section_page_pdf`` (already encoded with export framing) so the live
    matplotlib figure is not ``savefig``'d a second time for page 1.
    """
    if section_page_pdf:
        page1 = section_page_pdf
        owns_figure = False
        section_fig = None
    else:
        owns_figure = section_figure is None
        section_fig = section_figure or renderer.render(
            polygons,
            projected_df,
            collar_depths=collar_depths,
            water_levels=water_levels,
            lithology_codes=lithology_codes,
        )
        # Legacy path: encode page 1 without a second framing pass when figure was pre-rendered.
        buffer = BytesIO()
        with PdfPages(buffer) as pdf:
            pdf.savefig(section_fig, bbox_inches="tight")
        if owns_figure and section_fig is not None:
            plt.close(section_fig)
        page1 = buffer.getvalue()

    summary = build_report_summary_pdf(
        renderer,
        lithology_codes=lithology_codes,
        qa_lines=qa_lines,
        page_size_in=text_page_size_for(_pdf_first_page_size_in(page1)),
    )
    return _merge_pdf_pages([page1, summary])
