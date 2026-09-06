"""Multi-format export helpers for cross-section figures."""

from __future__ import annotations

from io import BytesIO
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from renderer import CrossSectionRenderer
from stratigraphy import GeologicalPolygon


def _merge_pdf_pages(page_pdfs: Sequence[bytes]) -> bytes:
    """Concatenate PDF page streams (pypdf when available; else return first page)."""
    valid = [payload for payload in page_pdfs if payload]
    if not valid:
        return b""
    if len(valid) == 1:
        return valid[0]
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        return valid[0]
    writer = PdfWriter()
    for payload in valid:
        writer.append(PdfReader(BytesIO(payload)))
    out = BytesIO()
    writer.write(out)
    return out.getvalue()


def build_report_summary_pdf(
    renderer: CrossSectionRenderer,
    *,
    lithology_codes: Sequence[str] | None = None,
    qa_lines: Sequence[str] = (),
) -> bytes:
    """Single-page letter PDF with title, metadata, lithology list, and QA notes."""
    summary_fig, summary_ax = plt.subplots(figsize=(8.5, 11.0))
    summary_ax.axis("off")
    summary_ax.set_title(
        "Cross Section Studio — Report Summary",
        fontsize=14,
        fontweight="bold",
        loc="left",
    )
    y = 0.92
    summary_ax.text(
        0.05, y, renderer.title, fontsize=12, fontweight="bold", transform=summary_ax.transAxes
    )
    y -= 0.05
    if renderer.disclaimer:
        summary_ax.text(
            0.05,
            y,
            renderer.disclaimer,
            fontsize=9,
            style="italic",
            transform=summary_ax.transAxes,
        )
        y -= 0.05
    for line in renderer._metadata_footer_lines():
        summary_ax.text(0.05, y, line, fontsize=9, transform=summary_ax.transAxes)
        y -= 0.04
    if lithology_codes:
        summary_ax.text(
            0.05,
            y,
            "Lithology codes:",
            fontsize=10,
            fontweight="bold",
            transform=summary_ax.transAxes,
        )
        y -= 0.04
        for code in lithology_codes:
            summary_ax.text(0.07, y, f"• {code}", fontsize=9, transform=summary_ax.transAxes)
            y -= 0.035
    if qa_lines:
        summary_ax.text(
            0.05, y, "QA notes:", fontsize=10, fontweight="bold", transform=summary_ax.transAxes
        )
        y -= 0.04
        for line in qa_lines[:12]:
            summary_ax.text(0.07, y, f"• {line}", fontsize=8, transform=summary_ax.transAxes)
            y -= 0.03
    buffer = BytesIO()
    summary_fig.savefig(buffer, format="pdf", bbox_inches="tight")
    plt.close(summary_fig)
    buffer.seek(0)
    return buffer.getvalue()


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
    )
    return _merge_pdf_pages([page1, summary])
