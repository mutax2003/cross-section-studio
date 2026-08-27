"""Tests for drafter export framing and packaging."""

from __future__ import annotations

import zipfile
from io import BytesIO

from export_framing import (
    ExportFramingConfig,
    annotate_svg_layers,
    build_export_filename,
    build_report_package_bytes,
    savefig_kwargs,
)
from section_build_request import SectionBuildRequest


def test_geometry_cache_ignores_export_framing() -> None:
    holes = ("BH-01", "BH-02")
    base = SectionBuildRequest(transect_points=((0.0, 0.0), (10.0, 0.0)))
    framed = base.model_copy(
        update={
            "export_framing": ExportFramingConfig(
                export_dpi=600,
                fence_only=True,
                show_draft_watermark=True,
            )
        }
    )
    assert base.geometry_cache_key(holes) == framed.geometry_cache_key(holes)
    assert base.cache_key(holes) != framed.cache_key(holes)


def test_build_export_filename_project_pattern() -> None:
    stem = build_export_filename(
        pattern="project_figure_transect_rev",
        section_title="Cross Section A-A prime",
        figure_number="3.1",
        project_number="P-100",
        transect_label="AA",
        revision="RevA",
    )
    assert stem
    assert "RevA" in stem
    assert "AA" in stem


def test_savefig_kwargs_tight_fence() -> None:
    kwargs = savefig_kwargs(
        ExportFramingConfig(page_preset="tight_fence"),
        layout="section_sheet",
    )
    assert kwargs["bbox_inches"] == "tight"


def test_report_package_zip_contains_formats() -> None:
    payload = build_report_package_bytes(
        stem="fig_01",
        svg_bytes=b"<svg></svg>",
        png_bytes=b"png",
        pdf_bytes=b"pdf",
        metadata={"title": "Test"},
    )
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        names = set(archive.namelist())
    assert "fig_01.svg" in names
    assert "fig_01.png" in names
    assert "fig_01.pdf" in names
    assert "fig_01_metadata.json" in names
    assert "README_deliverable.txt" in names


def test_annotate_svg_layers() -> None:
    raw = b'<svg metadata={"Creator": "Cross Section Studio"}></svg>'
    updated = annotate_svg_layers(raw)
    assert b"Cross Section Studio CAD" in updated
