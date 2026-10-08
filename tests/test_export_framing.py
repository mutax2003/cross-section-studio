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


def test_save_exports_to_directory_writes_docx(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CROSS_SECTION_EXPORT_ROOTS", str(tmp_path))
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
    raw = (
        b'<svg xmlns="http://www.w3.org/2000/svg" '
        b'metadata={"Creator": "Cross Section Studio"}>'
        b'<g id="fence"><path d="M0 0"/></g>'
        b'<g id="tracks"/>'
        b'<g id="other"/>'
        b"</svg>"
    )
    updated = annotate_svg_layers(raw)
    text = updated.decode("utf-8")
    assert "Cross Section Studio CAD" in text
    assert 'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"' in text
    assert 'id="fence"' in text
    assert 'inkscape:groupmode="layer"' in text
    assert 'inkscape:label="Fence"' in text
    assert 'inkscape:label="Tracks"' in text
    # Unmatched groups stay plain.
    assert 'id="other"' in text
    other_idx = text.index('id="other"')
    other_tag = text[text.rindex("<g", 0, other_idx) : text.index(">", other_idx) + 1]
    assert "inkscape:groupmode" not in other_tag


def test_annotate_svg_layers_idempotent_on_existing_inkscape() -> None:
    raw = (
        b'<svg xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
        b'metadata={"Creator": "Cross Section Studio CAD"}>'
        b'<g id="legend" inkscape:groupmode="layer" inkscape:label="Legend"/>'
        b"</svg>"
    )
    updated = annotate_svg_layers(raw)
    text = updated.decode("utf-8")
    assert text.count('xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"') == 1
    assert text.count('inkscape:groupmode="layer"') == 1
    assert text.count('inkscape:label="Legend"') == 1


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


def test_build_export_filename_transect_label_opt_in() -> None:
    kwargs = dict(pattern="section_title", section_title="Site", transect_label="T1")
    assert build_export_filename(**kwargs) == "Site"
    assert build_export_filename(**kwargs, include_transect_label=True) == "Site_T1"
    # label == title stays un-doubled even when opted in
    same = dict(pattern="section_title", section_title="Site", transect_label="Site")
    assert build_export_filename(**same, include_transect_label=True) == "Site"


def test_viewport_crop_rejects_non_finite_and_clamps_to_the_data() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pytest

    from export_framing import apply_viewport_crop

    for bad in (float("nan"), float("inf"), 1e308):
        with pytest.raises(ValueError, match="finite"):
            ExportFramingConfig(viewport_xmin=bad)

    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(110, 90)  # depth-style inverted axis
    # A box partly outside the data is clamped, keeping the axis direction.
    apply_viewport_crop(
        fig, ExportFramingConfig(viewport_xmin=50, viewport_xmax=500, viewport_ymin=80, viewport_ymax=100)
    )
    assert ax.get_xlim() == (50.0, 100.0)
    assert ax.get_ylim() == (100.0, 90.0)
    # A box entirely outside the data would export a blank sheet: ignored.
    apply_viewport_crop(
        fig, ExportFramingConfig(viewport_xmin=5000, viewport_xmax=6000, viewport_ymin=-900, viewport_ymax=-800)
    )
    assert ax.get_xlim() == (50.0, 100.0)
    plt.close(fig)


def test_save_exports_confined_to_allowed_roots(tmp_path, monkeypatch) -> None:
    import pytest

    from export_framing import save_exports_to_directory

    monkeypatch.setenv("CROSS_SECTION_EXPORT_ROOTS", str(tmp_path / "allowed"))
    inside = tmp_path / "allowed" / "job" / "figures"
    written = save_exports_to_directory(
        str(inside), stem="s", svg_bytes=b"<svg/>", png_bytes=b"", pdf_bytes=b"", metadata={}
    )
    assert written and Path(written[0]).parent == inside.resolve()

    with pytest.raises(ValueError, match="must be under"):
        save_exports_to_directory(
            str(tmp_path / "elsewhere"), stem="s", svg_bytes=b"<svg/>", png_bytes=b"", pdf_bytes=b"", metadata={}
        )
    # Traversal back out of the allowed root is caught after resolution.
    with pytest.raises(ValueError, match="must be under"):
        save_exports_to_directory(
            str(tmp_path / "allowed" / ".." / "elsewhere"),
            stem="s", svg_bytes=b"<svg/>", png_bytes=b"", pdf_bytes=b"", metadata={},
        )


def test_portrait_export_keeps_the_right_axis_label_on_the_page() -> None:
    from models import Collar, Lithology
    from render_profiles import CONSULTING_SECTION_PROFILE
    from renderer import CrossSectionRenderer
    from tests.conftest import run_pipeline

    ids = ["BH-01", "BH-02", "BH-03"]
    collars = [Collar(hole_id=h, easting=i * 40.0, northing=0.0, elevation=100.0, total_depth=8.0) for i, h in enumerate(ids)]
    lith = [Lithology(hole_id=h, from_depth=0.0, to_depth=8.0, lithology_code="Clay") for h in ids]
    projected, polygons, _ = run_pipeline(collars, lith, [(0.0, 0.0), (80.0, 0.0)])
    renderer = CrossSectionRenderer(
        show_legend=True,
        render_profile=CONSULTING_SECTION_PROFILE,
        export_framing=ExportFramingConfig(page_preset="letter_portrait", export_dpi=72),
    )
    figure = renderer.render(polygons, projected, collar_depths={h: 8.0 for h in ids})
    renderer.to_png_bytes(figure)
    figure.draw_without_rendering()
    mpl_renderer = figure.canvas.get_renderer()
    right = [ax.yaxis.label for ax in figure.axes if ax.yaxis.get_label_position() == "right" and ax.yaxis.label.get_text()]
    assert right and all(lbl.get_window_extent(mpl_renderer).x1 <= figure.bbox.x1 for lbl in right)


def test_export_filename_does_not_repeat_the_section_title() -> None:
    from export_framing import build_export_filename

    stem = build_export_filename(
        pattern="section_title",
        section_title="Test Section A-A'",
        transect_label="Test Section A-A' MW-01→MW-04",
        include_transect_label=True,
    )
    assert stem.count("Test_Section") == 1, stem
    assert "MW-01" in stem and "MW-04" in stem


def test_report_zip_readme_lists_the_word_file_when_included() -> None:
    import zipfile
    from io import BytesIO

    from export_framing import build_report_package_bytes

    payload = build_report_package_bytes(
        stem="A-A", svg_bytes=b"<svg/>", png_bytes=b"png", pdf_bytes=b"%PDF",
        metadata={}, docx_bytes=b"PK",
    )
    readme = zipfile.ZipFile(BytesIO(payload)).read("README_deliverable.txt").decode()
    assert "Word figure (.docx)" in readme


def test_export_filename_keeps_primes_and_names_the_section_once() -> None:
    """A consulting PDF was named A_-_A_WITH_CHLORIDE_A-A_BH-01_BH-06.pdf:
    the prime dropped and the section label printed twice."""
    from export_framing import build_export_filename

    stem = build_export_filename(
        pattern="section_title",
        section_title="A - A' WITH CHLORIDE",
        transect_label="A-A' BH-01→BH-06",
        include_transect_label=True,
    )
    assert stem == "A-A'_WITH_CHLORIDE_BH-01_BH-06"
    # A label the title does not carry is kept.
    stem = build_export_filename(
        pattern="section_title",
        section_title="Bay Area Site",
        transect_label="A BH-01→BH-06",
        include_transect_label=True,
    )
    assert stem == "Bay_Area_Site_A_BH-01_BH-06"


def test_file_names_keep_the_section_prime_without_stray_fragments() -> None:
    from export_framing import build_export_filename

    title = "Section A-A'"
    assert build_export_filename(pattern="section_title", section_title=title) == "Section_A-A'"
    with_label = build_export_filename(
        pattern="section_title",
        section_title=title,
        transect_label="Section A-A' BH 1",
        include_transect_label=True,
    )
    assert with_label == "Section_A-A'_BH_1"
    assert "_'_" not in with_label
