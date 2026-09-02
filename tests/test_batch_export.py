"""Tests for batch export helpers."""

from __future__ import annotations

from batch_export import export_binder_pdf


def test_binder_skips_cover_when_all_pdfs_identical() -> None:
    pdf = b"%PDF-1.4 fake"
    assert export_binder_pdf([pdf, pdf, pdf]) == pdf
