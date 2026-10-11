"""Tests for ui_helpers display utilities."""

from __future__ import annotations

from ui_helpers import SvgDisplayMeta, svg_display_height, svg_display_meta, svg_is_valid


def test_svg_display_meta_valid_svg() -> None:
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" height="400" width="600"></svg>'
    meta = svg_display_meta(svg)
    assert isinstance(meta, SvgDisplayMeta)
    assert meta.valid is True
    assert meta.encoded
    assert 420 <= meta.height <= 760
    assert svg_is_valid(svg)
    assert svg_display_height(svg) == meta.height


def test_svg_display_meta_invalid_payload() -> None:
    meta = svg_display_meta(b"not svg")
    assert meta.valid is False
    assert meta.encoded == ""


def test_svg_natural_width_converts_matplotlib_points_to_css_pixels() -> None:
    svg = b'<svg width="720pt" height="360pt" viewBox="0 0 720 360"></svg>'
    assert svg_display_meta(svg).natural_width_px == 960
    assert svg_display_meta(b'<svg width="500px" height="1"></svg>').natural_width_px == 500
    assert svg_display_meta(b'<svg viewBox="0 0 640 480"></svg>').natural_width_px == 640
    assert svg_display_meta(b"not svg").natural_width_px == 0


def test_preview_img_style_scales_from_natural_width_and_falls_back_to_fit() -> None:
    from ui_helpers import preview_img_style

    style, zoomed = preview_img_style("150%", 800)
    assert zoomed and "width:1200px" in style and "max-width:none" in style
    # Streamlit markdown images default to scale-down, which would not enlarge.
    assert "object-fit:contain" in style
    assert preview_img_style("100%", 800) == (
        "width:800px;max-width:none;height:auto;display:block;object-fit:contain;",
        True,
    )
    for zoom, width in (("Fit width", 800), (None, 800), ("bogus", 800), ("150%", 0)):
        style, zoomed = preview_img_style(zoom, width)
        assert not zoomed and style.startswith("width:100%")


def test_svg_natural_width_tolerates_malformed_and_hostile_widths() -> None:
    def width(svg: bytes) -> int:
        return svg_display_meta(svg).natural_width_px

    assert width(b'<svg width="." height="10"></svg>') == 0  # used to raise ValueError
    assert width(b'<svg width="1.2.3" viewBox="0 0 300 1"></svg>') == 300
    assert width(b"<svg width='750pt'></svg>") == 1000
    assert width(b'<svg viewBox="0,0,640,480"></svg>') == 640
    assert width(b'<!-- <svg width="5"> --><svg width="720pt"></svg>') == 960
    assert width(b'<svg width="99999999pt"></svg>') == 12000


def test_legend_hatch_background_tolerates_whitespace_hatches() -> None:
    from ui_helpers import legend_hatch_background

    assert legend_hatch_background(" ") == "none"
    assert legend_hatch_background("") == "none"
    assert "gradient" in legend_hatch_background("OO")


def _same_length_pair(head: bytes, tail: bytes) -> tuple[bytes, bytes]:
    """Two >1 MiB payloads that differ only where Streamlit does not sample
    (a quarter of the way in: not head, middle or tail)."""
    filler = 3 * 1024 * 1024
    body_a = bytearray(b"a" * filler)
    body_b = bytearray(body_a)
    body_b[filler // 4] = ord("b")
    return head + bytes(body_a) + tail, head + bytes(body_b) + tail


def _file_id(data: bytes, mime: str, name: str) -> str:
    from streamlit.runtime.memory_media_file_storage import _calculate_file_id

    return _calculate_file_id(data, mime, name)


def test_same_length_downloads_get_different_file_ids() -> None:
    """Black vs red-label SVGs (same 1,478,243 B) shared one Streamlit file
    id, so the SVG button served the previous figure."""
    import io
    import zipfile

    from PIL import Image
    from pypdf import PdfReader
    from streamlit.runtime import memory_media_file_storage as storage

    from ui_helpers import DOWNLOAD_FULL_HASH_LIMIT, unique_download_bytes

    assert DOWNLOAD_FULL_HASH_LIMIT == storage._PARTIAL_HASH_THRESHOLD

    svg_a, svg_b = _same_length_pair(b'<svg xmlns="http://www.w3.org/2000/svg"><!--', b"--></svg>\n")
    assert _file_id(svg_a, "image/svg+xml", "f.svg") == _file_id(svg_b, "image/svg+xml", "f.svg")
    out_a, out_b = unique_download_bytes(svg_a), unique_download_bytes(svg_b)
    assert _file_id(out_a, "image/svg+xml", "f.svg") != _file_id(out_b, "image/svg+xml", "f.svg")
    assert out_a.startswith(svg_a) and out_a.rstrip().endswith(b"-->")

    # ZIP (also Word / Excel): digest goes in the archive comment.
    def big_zip(marker: bytes) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
            zf.writestr("a.bin", b"x" * (2 * 1024 * 1024) + marker + b"x" * (1024 * 1024))
            zf.writestr("b.txt", "end")
        return buf.getvalue()

    zip_a, zip_b = big_zip(b"1"), big_zip(b"2")
    assert len(zip_a) == len(zip_b)
    out_a, out_b = unique_download_bytes(zip_a), unique_download_bytes(zip_b)
    assert _file_id(out_a, "application/zip", "z.zip") != _file_id(out_b, "application/zip", "z.zip")
    with zipfile.ZipFile(io.BytesIO(out_a)) as zf:
        assert zf.testzip() is None and zf.comment.startswith(b"sha256:")

    # PNG: digest in a tEXt chunk before IEND; the image still opens.
    def big_png(seed: int) -> bytes:
        import numpy as np

        rng = np.random.default_rng(0)
        pixels = rng.integers(0, 255, (700, 700, 3), dtype=np.uint8)
        pixels[100, 100, 0] = seed
        buf = io.BytesIO()
        Image.fromarray(pixels).save(buf, format="PNG", compress_level=0)
        return buf.getvalue()

    png = big_png(7)
    assert len(png) > DOWNLOAD_FULL_HASH_LIMIT
    out = unique_download_bytes(png)
    assert out != png
    with Image.open(io.BytesIO(out)) as image:
        image.load()
        assert image.text.get("Digest", "").startswith("sha256:")

    # PDF: a comment line after %%EOF; the PDF still reads.
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 10), dpi=100)
    noise = __import__("numpy").random.default_rng(1).integers(0, 255, (1000, 1000, 3), dtype="uint8")
    fig.figimage(noise)
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    pdf = buf.getvalue()
    assert len(pdf) > DOWNLOAD_FULL_HASH_LIMIT
    out = unique_download_bytes(pdf)
    assert out.startswith(pdf) and b"%sha256:" in out[-100:]
    assert len(PdfReader(io.BytesIO(out)).pages) == 1

    # Small files are hashed in full by Streamlit: left untouched.
    assert unique_download_bytes(b"<svg></svg>") == b"<svg></svg>"
