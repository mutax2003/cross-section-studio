"""Tests for batch export helpers."""

from __future__ import annotations

import zipfile
from io import BytesIO

import pytest

from batch_export import (
    BatchTransectSpec,
    build_batch_zip,
    build_multi_transect_exports,
    build_one_transect_exports,
    clear_batch_geometry_memo,
    export_binder_pdf,
    parse_batch_transect_lines,
    prepare_batch_section_request,
    split_batch_transect_lines,
    transect_points_from_collars,
    validate_batch_transect_lines,
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
    # Default batch formats are PNG+PDF (skip SVG encode).
    assert svg_a == b"" and svg_b == b""
    assert png_a and png_b and png_a != png_b
    assert pdf_a and pdf_b and pdf_a != pdf_b

    zip_bytes = build_batch_zip(
        entries,
        binder_pdf=export_binder_pdf([pdf_a, pdf_b], cover_title="Site binder") or None,
    )
    with zipfile.ZipFile(BytesIO(zip_bytes)) as archive:
        names = set(archive.namelist())
    assert "A-A.svg" not in names and "B-B.svg" not in names
    assert "A-A.png" in names and "B-B.pdf" in names
    assert "report_binder.pdf" in names

    with_svg = build_multi_transect_exports(
        parse_result,
        base,
        specs[:1],
        export_formats=frozenset({"svg", "png", "pdf"}),
    )
    assert with_svg[0][1].startswith(b"<") or b"<svg" in with_svg[0][1][:200].lower()


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


def test_batch_geometry_memo_reuses_same_payload() -> None:
    clear_batch_geometry_memo()
    parse_result = _four_hole_parse()
    base = SectionBuildRequest(
        transect_points=((0.0, 0.0), (100.0, 0.0)),
        section_title="Site",
        export_formats=frozenset({"svg"}),
    )
    spec = BatchTransectSpec(label="A-A", hole_ids=("BH-01", "BH-02", "BH-03"))
    first = build_one_transect_exports(
        parse_result, base, spec, export_formats=frozenset({"svg"})
    )
    second = build_one_transect_exports(
        parse_result,
        base.model_copy(update={"section_title": "Site renamed"}),
        spec,
        export_formats=frozenset({"svg"}),
    )
    assert first[0] == second[0] == "A-A"
    assert first[1] and second[1]
    clear_batch_geometry_memo()


def test_build_batch_zip_uniquifies_colliding_and_derived_stems() -> None:
    import io
    import zipfile as _zipfile

    entries = [(stem, b"svg", b"png", b"pdf") for stem in ("A", "A", "A_2", "a")]
    names = _zipfile.ZipFile(io.BytesIO(build_batch_zip(entries))).namelist()
    assert len(names) == len(set(names))
    stems = {name.rsplit(".", 1)[0] for name in names if name != "README.txt"}
    assert len(stems) == 4  # every entry kept under a distinct stem
    lowered = [stem.lower() for stem in stems]
    assert len(lowered) == len(set(lowered))  # distinct even case-insensitively


def test_batch_transect_spec_requires_distinct_holes() -> None:
    with pytest.raises(ValueError, match="distinct"):
        parse_batch_transect_lines("A | BH-01, BH-01")


def test_batch_end_labels_use_section_letters_not_hole_ids() -> None:
    """End labels sit beside the edge holes' own names; using the hole IDs
    printed each edge hole twice on every batch sheet."""
    from batch_export import _consulting_for_spec
    from models import ConsultingTitleBlock

    base = ConsultingTitleBlock(
        section_label="A-A'",
        transect_start_primary="A",
        transect_start_secondary="WEST",
        transect_end_primary="A'",
        transect_end_secondary="EAST",
    )
    block = _consulting_for_spec(base, label="B-B'", hole_ids=("BH26-04", "BH26-05", "BH26-06"))
    assert (block.transect_start_primary, block.transect_end_primary) == ("B", "B'")
    # The base section's compass words do not describe this transect.
    assert block.transect_start_secondary == block.transect_end_secondary == ""

    free = _consulting_for_spec(None, label="North transect", hole_ids=("BH-1", "BH-2"))
    assert free.transect_start_primary == free.transect_end_primary == ""
    assert "BH-1" not in (free.transect_start_label, free.transect_end_label)


def test_batch_title_swaps_the_base_section_label() -> None:
    """The base title named A-A' and every batch figure was titled
    "<title> A-A' — B-B'"."""
    from batch_export import _batch_section_title
    from models import ConsultingTitleBlock

    class _Req:
        def __init__(self, title, block=None):
            self.section_title = title
            self.consulting_title_block = block

    assert _batch_section_title(_Req("Test Section A-A'"), "B-B'") == "Test Section B-B'"
    assert _batch_section_title(_Req("Site 4 – A–A′"), "C-C'") == "Site 4 – C-C'"
    block = ConsultingTitleBlock(section_label="North")
    assert _batch_section_title(_Req("North line", block), "South") == "South line"
    assert _batch_section_title(_Req("Borehole Cross-Section"), "B-B'") == "Borehole Cross-Section — B-B'"
    assert _batch_section_title(_Req("Plan"), "North line") == "Plan — North line"


def test_validate_batch_lines_reports_each_line_without_raising() -> None:
    text = (
        "A-A' | MW-01, MW-02\n"
        "\n"
        "C-C' | MW-01, MW-99\n"
        "D-D' | MW-02\n"
        "just some words\n"
        " | MW-01, MW-02\n"
        "A-A' | MW-02, MW-03\n"
        "E-E' | MW-01, MW-01\n"
    )
    statuses = validate_batch_transect_lines(text, ["MW-01", "MW-02", "MW-03"])
    assert [s.line_number for s in statuses] == [1, 3, 4, 5, 6, 7, 8]
    assert statuses[0].ok and statuses[0].spec.hole_ids == ("MW-01", "MW-02")
    assert statuses[1].message == "C-C': MW-99 not in Collars"
    assert statuses[2].message == "D-D': needs at least 2 holes"
    assert "'|'" in statuses[3].message and statuses[3].message.startswith("Line 5")
    assert "name" in statuses[4].problem
    assert statuses[5].problem == "name already used on line 1"
    assert "listed more than once" in statuses[6].problem
    assert not any(s.ok for s in statuses[1:])


def test_validate_batch_lines_skips_collar_check_without_known_ids() -> None:
    (status,) = validate_batch_transect_lines("X | Q-1, Q-2", None)
    assert status.ok and status.message == "X: 2 holes, ready"


def test_split_batch_lines_keeps_valid_and_lists_skipped() -> None:
    valid, skipped = split_batch_transect_lines(
        "A-A' | MW-01, MW-02\nC-C' | MW-01, MW-99", ["MW-01", "MW-02"]
    )
    assert [spec.label for spec in valid] == ["A-A'"]
    assert [status.label for status in skipped] == ["C-C'"]


def test_batch_cover_title_names_the_whole_batch() -> None:
    from types import SimpleNamespace

    from batch_export import batch_cover_title

    block = SimpleNamespace(section_label="A-A'")
    request = SimpleNamespace(section_title="Site X A-A'", consulting_title_block=block)
    labels = ["A-A'", "B-B'", "C-C'"]
    assert batch_cover_title(request, labels) == "Site X — Cross Sections A-A', B-B', C-C'"
    bare = SimpleNamespace(section_title="B-B'", consulting_title_block=None)
    assert batch_cover_title(bare, labels) == "Cross Sections A-A', B-B', C-C'"
    assert batch_cover_title(bare, ["A-A'"]) == "Cross Section A-A'"
