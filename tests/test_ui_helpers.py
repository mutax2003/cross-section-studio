"""Tests for ui_helpers display utilities."""

from __future__ import annotations

from ui_helpers import SvgDisplayMeta, svg_display_meta, svg_display_height, svg_is_valid


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
