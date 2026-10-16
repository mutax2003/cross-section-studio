"""SVG text is outlines by default and editable <text> on request."""

from __future__ import annotations

import re

from export_framing import ExportFramingConfig
from models import Collar, Lithology
from pipeline import build_cross_section


def _svg(**framing) -> str:
    collars = [
        Collar(hole_id="BH-01", easting=0, northing=0, elevation=100, total_depth=10),
        Collar(hole_id="BH-02", easting=40, northing=0, elevation=100, total_depth=10),
    ]
    lithology = [Lithology(hole_id=h, from_depth=0, to_depth=10, lithology_code="Clay") for h in ("BH-01", "BH-02")]
    result = build_cross_section(
        collars,
        lithology,
        [(0, 0), (40, 0)],
        export_framing=ExportFramingConfig(**framing),
        export_formats=frozenset({"svg"}),
    )
    return result.svg_bytes.decode("utf-8")


def test_svg_text_is_outlines_by_default() -> None:
    assert ExportFramingConfig().svg_editable_text is False
    svg = _svg()
    assert "<text" not in svg  # every label is drawn as outlines


def test_svg_text_is_editable_on_request() -> None:
    svg = _svg(svg_editable_text=True)
    assert re.search(r"<text[^>]*>", svg)
    assert ">BH-01<" in svg  # the hole id is real, editable text
