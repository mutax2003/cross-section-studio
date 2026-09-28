"""Tests for Word figure pack export."""

from __future__ import annotations

import pytest

docx = pytest.importorskip("docx")

from docx_export import build_figure_docx_bytes


def test_build_figure_docx_bytes_minimal() -> None:
    payload = build_figure_docx_bytes(
        png_bytes=b"",
        caption="Figure 1 — Test section",
        title="Test Section",
        metadata={"project": "Demo"},
    )
    assert payload.startswith(b"PK")
    assert len(payload) > 500
