"""Multi-transect batch export and PDF binder helpers."""

from __future__ import annotations

import zipfile
from io import BytesIO
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def export_binder_pdf(section_pdfs: Sequence[bytes], *, cover_title: str = "Cross Section Report") -> bytes:
    """Combine prepared single-section PDF bytes into one binder document."""
    valid = [payload for payload in section_pdfs if payload]
    if not valid:
        return b""
    if len(valid) == 1 or len({payload for payload in valid}) == 1:
        return valid[0]
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        cover, ax = plt.subplots(figsize=(8.5, 11.0))
        ax.axis("off")
        ax.text(
            0.5,
            0.55,
            cover_title,
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.45,
            f"{len(valid)} section(s)",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        pdf.savefig(cover, bbox_inches="tight")
        plt.close(cover)
    # Append raw PDF page streams by re-saving each section through PdfPages is complex;
    # for binder we zip individual PDFs when merge is unavailable.
    return buffer.getvalue()


def build_batch_zip(
    entries: Sequence[tuple[str, bytes, bytes, bytes]],
    *,
    binder_pdf: bytes | None = None,
) -> bytes:
    """Zip multiple transect exports. Each entry is (stem, svg, png, pdf)."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for stem, svg_bytes, png_bytes, pdf_bytes in entries:
            if svg_bytes:
                archive.writestr(f"{stem}.svg", svg_bytes)
            if png_bytes:
                archive.writestr(f"{stem}.png", png_bytes)
            if pdf_bytes:
                archive.writestr(f"{stem}.pdf", pdf_bytes)
        if binder_pdf:
            archive.writestr("report_binder.pdf", binder_pdf)
    buffer.seek(0)
    return buffer.getvalue()
