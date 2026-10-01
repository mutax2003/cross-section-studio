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
