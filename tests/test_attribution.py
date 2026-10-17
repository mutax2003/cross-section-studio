"""Attribution appears everywhere a user or a downstream tool would look."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from app_identity import APP_NAME, AUTHOR, COPYRIGHT_NOTICE, CREATED_BY, ORGANIZATION

ROOT = Path(__file__).resolve().parent.parent


def test_identity_strings() -> None:
    assert CREATED_BY == "Created by Andrew Liu, Ecoventure, 2026"
    assert COPYRIGHT_NOTICE.startswith("Copyright © 2026 Andrew Liu, Ecoventure")
    assert (AUTHOR, ORGANIZATION, APP_NAME) == ("Andrew Liu", "Ecoventure", "Cross Section Studio")


def test_repository_level_notices() -> None:
    assert "Copyright (c) 2026 Andrew Liu, Ecoventure" in (ROOT / "COPYRIGHT").read_text(encoding="utf-8")
    assert CREATED_BY in (ROOT / "README.md").read_text(encoding="utf-8")
    assert CREATED_BY in (ROOT / "docs" / "help" / "about.md").read_text(encoding="utf-8")
    assert 'authors = [{ name = "Andrew Liu" }]' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'org.opencontainers.image.authors="Andrew Liu, Ecoventure"' in (ROOT / "Dockerfile").read_text(
        encoding="utf-8"
    )


def test_about_block_carries_attribution() -> None:
    from app_version import about_version_markdown

    text = about_version_markdown()
    assert CREATED_BY in text and COPYRIGHT_NOTICE in text


def test_windows_version_resource_is_well_formed() -> None:
    """PyInstaller evaluates this file with its own constructors; a syntax slip
    would only surface on the Windows build machine."""
    text = (ROOT / "windows_version_info.txt").read_text(encoding="utf-8")
    captured: dict[str, object] = {}

    class _Node:
        def __init__(self, *args, **kwargs):
            self.args, self.kwargs = args, kwargs
            if args and isinstance(args[0], str) and len(args) == 2:
                captured[args[0]] = args[1]

    names = {n: _Node for n in ("VSVersionInfo", "FixedFileInfo", "StringFileInfo", "StringTable", "StringStruct", "VarFileInfo", "VarStruct")}
    eval(text, {"__builtins__": {}}, names)  # noqa: S307 - trusted repo file, no builtins
    assert captured["CompanyName"] == "Ecoventure"
    assert "Andrew Liu, Ecoventure" in str(captured["LegalCopyright"])
    assert captured["FileVersion"] == (ROOT / "VERSION").read_text().strip().lstrip("v")
    assert 'version="windows_version_info.txt"' in (ROOT / "cross_section_studio.spec").read_text(encoding="utf-8")


def test_exports_embed_attribution_metadata() -> None:
    from models import Collar, Lithology
    from pipeline import build_cross_section
    from ui_helpers import export_metadata_payload

    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Clay"),
    ]
    result = build_cross_section(
        collars, lithologies, [(0.0, 0.0), (50.0, 0.0)], export_formats=frozenset({"svg", "png", "pdf"})
    )
    svg = result.svg_bytes.decode("utf-8")
    assert "Andrew Liu, Ecoventure" in svg  # dc:rights
    from pypdf import PdfReader

    info = PdfReader(io.BytesIO(result.pdf_bytes)).metadata
    assert info.author == "Andrew Liu, Ecoventure" and info.creator == APP_NAME
    from PIL import Image

    png_info = Image.open(io.BytesIO(result.png_bytes)).info
    assert png_info.get("Author") == "Andrew Liu, Ecoventure"
    assert "2026" in str(png_info.get("Copyright"))

    payload = export_metadata_payload(
        section_title="S", preset_label=None, vertical_exaggeration=1.0, hole_count=2,
        transect_label=None, overlap_warnings=(),
    )
    assert payload["created_by"] == CREATED_BY and payload["copyright"] == COPYRIGHT_NOTICE
    json.dumps(payload)


def test_word_and_zip_deliverables_carry_attribution() -> None:
    pytest.importorskip("docx")
    from docx import Document

    from docx_export import build_figure_docx_bytes
    from export_framing import _default_readme

    docx_bytes = build_figure_docx_bytes(
        title="Section A-A'", caption="", png_bytes=b"", metadata={}
    )
    props = Document(io.BytesIO(docx_bytes)).core_properties
    assert props.author == "Andrew Liu, Ecoventure"
    assert "2026" in props.comments
    assert zipfile.is_zipfile(io.BytesIO(docx_bytes))
    assert CREATED_BY in _default_readme("stem")


def test_batch_zip_carries_the_attribution_readme() -> None:
    import zipfile
    from io import BytesIO

    from app_identity import CREATED_BY
    from batch_export import build_batch_zip

    payload = build_batch_zip([("A-A", b"", b"png", b"%PDF")])
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        readme = archive.read("README.txt").decode("utf-8")
    assert CREATED_BY in readme
