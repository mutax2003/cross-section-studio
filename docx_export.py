"""Word figure pack export for report drafters."""

from __future__ import annotations

import re
from collections.abc import Mapping
from io import BytesIO

from app_identity import AUTHOR, COPYRIGHT_NOTICE, ORGANIZATION

_XML_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _xml_safe(value: object) -> str:
    """Text Word can store: XML-illegal control characters removed."""
    return _XML_ILLEGAL.sub("", "" if value is None else str(value))


def build_figure_docx_bytes(
    *,
    png_bytes: bytes,
    caption: str,
    title: str,
    metadata: Mapping[str, object] | None = None,
) -> bytes:
    """Embed prepared PNG with caption and optional metadata table."""
    try:
        from docx import Document
        from docx.shared import Inches
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "python-docx is required for Word export. Install with: pip install python-docx"
        ) from exc

    # Word XML rejects control characters (e.g. the soft line break \x0b that
    # comes with text pasted from Word); drop them from everything we write.
    title = _xml_safe(title)
    caption = _xml_safe(caption)
    document = Document()
    document.core_properties.author = f"{AUTHOR}, {ORGANIZATION}"
    document.core_properties.last_modified_by = f"{AUTHOR}, {ORGANIZATION}"
    document.core_properties.comments = COPYRIGHT_NOTICE
    document.core_properties.title = title or "Cross Section"
    document.add_heading(title or "Cross Section", level=1)
    if caption:
        document.add_paragraph(caption)
    if png_bytes:
        stream = BytesIO(png_bytes)
        document.add_picture(stream, width=Inches(6.5))
    if metadata:
        document.add_heading("Figure metadata", level=2)
        table = document.add_table(rows=1, cols=2)
        header = table.rows[0].cells
        header[0].text = "Field"
        header[1].text = "Value"
        for key, value in metadata.items():
            row = table.add_row().cells
            row[0].text = _xml_safe(key)
            row[1].text = _xml_safe(value)
    buffer = BytesIO()
    document.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()
