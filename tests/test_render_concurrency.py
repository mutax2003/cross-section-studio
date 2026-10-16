"""Renders from several threads at once must not corrupt each other (pyplot
keeps process-wide figure state; the pipeline serialises render → export)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from export_framing import ExportFramingConfig
from models import Collar, Lithology, ScreenInterval
from pipeline import build_cross_section


def _inputs(n_holes: int = 4):
    collars = [
        Collar(hole_id=f"BH-{i:02d}", easting=30.0 * i, northing=0, elevation=100, total_depth=12)
        for i in range(n_holes)
    ]
    lithology = []
    for c in collars:
        lithology.append(Lithology(hole_id=c.hole_id, from_depth=0, to_depth=5, lithology_code="Clay"))
        lithology.append(Lithology(hole_id=c.hole_id, from_depth=5, to_depth=12, lithology_code="Sand"))
    return collars, lithology, [(0.0, 0.0), (30.0 * (n_holes - 1), 0.0)]


def test_parallel_renders_all_succeed() -> None:
    collars, lithology, line = _inputs()

    def one(layout: str) -> int:
        result = build_cross_section(
            collars, lithology, line, render_layout=layout, export_formats=frozenset({"png", "pdf"})
        )
        assert result.png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
        assert result.pdf_bytes.startswith(b"%PDF")
        return len(result.png_bytes)

    layouts = ["consulting_section", "section_sheet", "chart"] * 2
    with ThreadPoolExecutor(max_workers=6) as pool:
        sizes = list(pool.map(one, layouts))
    assert all(size > 10_000 for size in sizes)


def test_screens_can_be_left_out_of_the_export() -> None:
    collars, lithology, line = _inputs(2)
    screens = [ScreenInterval(hole_id="BH-00", from_depth=6, to_depth=9)]
    with_screens = build_cross_section(
        collars, lithology, line, screen_intervals=screens,
        render_layout="consulting_section", export_formats=frozenset({"svg"}),
    ).svg_bytes.decode()
    without = build_cross_section(
        collars, lithology, line, screen_intervals=screens, render_layout="consulting_section",
        export_formats=frozenset({"svg"}), export_framing=ExportFramingConfig(include_screens=False),
    ).svg_bytes.decode()
    assert ExportFramingConfig().include_screens is True
    assert len(without) < len(with_screens)
