"""Tests for drafter export framing and packaging."""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

from export_framing import (
    ExportFramingConfig,
    annotate_svg_layers,
    apply_export_page_size,
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


def test_save_exports_to_directory_writes_docx(tmp_path) -> None:
    from export_framing import save_exports_to_directory

    written = save_exports_to_directory(
        str(tmp_path),
        stem="fig",
        svg_bytes=b"<svg/>",
        png_bytes=b"png",
        pdf_bytes=b"pdf",
        metadata={"a": 1},
        docx_bytes=b"docx",
    )
    names = {Path(path).name for path in written}
    assert names == {"fig.svg", "fig.png", "fig.pdf", "fig.docx", "fig_metadata.json"}


def test_annotate_svg_layers() -> None:
    raw = b'<svg metadata={"Creator": "Cross Section Studio"}></svg>'
    updated = annotate_svg_layers(raw)
    assert b"Cross Section Studio CAD" in updated


def test_apply_export_page_size_letter_landscape() -> None:
    import matplotlib.pyplot as plt

    fig, _ax = plt.subplots(figsize=(4.0, 3.0))
    try:
        apply_export_page_size(
            fig,
            ExportFramingConfig(page_preset="letter_landscape"),
            layout="section_sheet",
        )
        width, height = fig.get_size_inches()
        assert width == 11.0
        assert height == 8.5
    finally:
        plt.close(fig)


def test_merge_framing_hides_water_profile_flags() -> None:
    from export_framing import merge_framing_into_profile_updates

    updates = merge_framing_into_profile_updates(
        ExportFramingConfig(include_water_table=False),
        {},
    )
    assert updates["show_water_elevation_labels"] is False
    assert updates["show_water_legend"] is False
    assert updates["interpolate_water_table_default"] is False


def test_fixed_page_margins_adjust_subplots() -> None:
    import matplotlib.pyplot as plt

    from export_framing import apply_fixed_page_margins

    fig, _ax = plt.subplots(figsize=(11.0, 8.5))
    try:
        apply_fixed_page_margins(
            fig,
            ExportFramingConfig(
                page_preset="letter_landscape",
                margin_left_in=1.0,
                margin_right_in=0.5,
                margin_top_in=0.25,
                margin_bottom_in=0.75,
            ),
            layout="section_sheet",
        )
        assert fig.subplotpars.left > 0.05
        assert fig.subplotpars.right < 0.98
    finally:
        plt.close(fig)
