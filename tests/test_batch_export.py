"""Tests for batch export helpers."""

from __future__ import annotations

import zipfile
from io import BytesIO

import pytest

from batch_export import (
    BatchTransectSpec,
    build_batch_zip,
    build_multi_transect_exports,
    export_binder_pdf,
    parse_batch_transect_lines,
    prepare_batch_section_request,
    transect_points_from_collars,
)
from models import Collar, Lithology, ParseResult
from section_build_request import SectionBuildRequest


def test_binder_merges_distinct_pdfs_when_pypdf_available() -> None:
    pytest.importorskip("pypdf")
    # Minimal valid one-page PDFs from matplotlib
    from matplotlib.backends.backend_pdf import PdfPages

    def _one_page(label: str) -> bytes:
        buf = BytesIO()
        with PdfPages(buf) as pdf:
            fig, ax = __import__("matplotlib.pyplot", fromlist=["pyplot"]).subplots()
            ax.set_title(label)
            ax.plot([0, 1], [0, 1])
            pdf.savefig(fig)
            __import__("matplotlib.pyplot", fromlist=["pyplot"]).close(fig)
        return buf.getvalue()

    pdf_a = _one_page("A")
    pdf_b = _one_page("B")
    assert pdf_a != pdf_b
    binder = export_binder_pdf([pdf_a, pdf_b], cover_title="Binder")
    assert binder.startswith(b"%PDF")
    assert len(binder) > max(len(pdf_a), len(pdf_b))


def test_parse_batch_transect_lines() -> None:
    specs = parse_batch_transect_lines(
        "A-A' | BH-01, BH-02, BH-03\n\nB-B' | X1; X2\n"
    )
    assert specs == [
        BatchTransectSpec(label="A-A'", hole_ids=("BH-01", "BH-02", "BH-03")),
        BatchTransectSpec(label="B-B'", hole_ids=("X1", "X2")),
    ]


def test_parse_batch_transect_lines_rejects_filename_only() -> None:
    with pytest.raises(ValueError, match="pipe"):
        parse_batch_transect_lines("A-A prime\nB-B prime")


def test_transect_points_from_collars() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=10.0, northing=5.0, elevation=100.0, total_depth=20.0),
        Collar(hole_id="BH-02", easting=60.0, northing=-3.0, elevation=102.0, total_depth=25.0),
    ]
    assert transect_points_from_collars(collars, ("BH-01", "BH-02")) == (
        (10.0, 5.0),
        (60.0, -3.0),
    )


def _four_hole_parse() -> ParseResult:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=101.0, total_depth=22.0),
        Collar(hole_id="BH-03", easting=100.0, northing=0.0, elevation=102.0, total_depth=24.0),
        Collar(hole_id="BH-04", easting=50.0, northing=40.0, elevation=103.0, total_depth=18.0),
    ]
    lithologies = [
        Lithology(hole_id=h, from_depth=0.0, to_depth=5.0, lithology_code="Sandstone")
        for h in ("BH-01", "BH-02", "BH-03", "BH-04")
    ] + [
        Lithology(hole_id=h, from_depth=5.0, to_depth=15.0, lithology_code="Clay")
        for h in ("BH-01", "BH-02", "BH-03", "BH-04")
    ]
    return ParseResult(collars=collars, lithologies=lithologies, errors=())


def test_build_multi_transect_exports_distinct_figures() -> None:
    parse_result = _four_hole_parse()
    base = SectionBuildRequest(
        transect_points=((0.0, 0.0), (100.0, 0.0)),
        vertical_exaggeration=2.0,
        show_hatches=False,
        section_title="Site",
        allow_pinch_outs=True,
    )
    specs = [
        BatchTransectSpec(label="A-A", hole_ids=("BH-01", "BH-02", "BH-03")),
        BatchTransectSpec(label="B-B", hole_ids=("BH-01", "BH-04", "BH-03")),
    ]
    entries = build_multi_transect_exports(parse_result, base, specs)
    assert len(entries) == 2
    stems = [stem for stem, *_ in entries]
    assert stems == ["A-A", "B-B"]
    svg_a, svg_b = entries[0][1], entries[1][1]
    png_a, png_b = entries[0][2], entries[1][2]
    pdf_a, pdf_b = entries[0][3], entries[1][3]
    assert svg_a and svg_b and svg_a != svg_b
    assert png_a and png_b and png_a != png_b
    assert pdf_a and pdf_b and pdf_a != pdf_b

    zip_bytes = build_batch_zip(
        entries,
        binder_pdf=export_binder_pdf([pdf_a, pdf_b], cover_title="Site binder") or None,
    )
    with zipfile.ZipFile(BytesIO(zip_bytes)) as archive:
        names = set(archive.namelist())
    assert "A-A.svg" in names and "B-B.svg" in names
    assert "A-A.png" in names and "B-B.pdf" in names
    assert "report_binder.pdf" in names


def test_prepare_batch_section_request_overrides_geometry() -> None:
    parse_result = _four_hole_parse()
    base = SectionBuildRequest(
        transect_points=((0.0, 0.0), (100.0, 0.0)),
        section_title="Site",
    )
    subset, request = prepare_batch_section_request(
        parse_result,
        base,
        BatchTransectSpec(label="C-C", hole_ids=("BH-02", "BH-04")),
    )
    assert {c.hole_id for c in subset.collars} == {"BH-02", "BH-04"}
    assert request.transect_points == ((50.0, 0.0), (50.0, 40.0))
    assert "C-C" in request.section_title
    assert request.consulting_title_block is not None
    assert request.consulting_title_block.section_label == "C-C"
