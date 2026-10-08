"""Vertical exaggeration: exact VE is drawn exactly; Auto prints the measured VE.

The VE caption used to be decorative: every layout let the axes stretch to fill
the frame, so "5× VERTICAL EXAGGERATION" could measure ~2× on the page. These
tests measure the VE actually drawn after export page framing.
"""

from __future__ import annotations

import io
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from PIL import Image

from export_framing import PAGE_FIGSIZE_IN, ExportFramingConfig
from models import Collar, Lithology
from pipeline import build_cross_section, compute_section_geometry, render_cross_section_from_geometry
from renderer_common import (
    VECaptionText,
    format_ve_caption,
    measured_vertical_exaggeration,
    round_vertical_exaggeration,
)
from section_build_request import SectionBuildRequest

LAYOUTS = ("consulting_section", "section_sheet", "chart")
PAGES = ("letter_portrait", "letter_landscape", "tabloid_landscape")


def _dataset():
    collars = [
        Collar(hole_id="BH-1", easting=0.0, northing=0.0, elevation=100.0, total_depth=18.0),
        Collar(hole_id="BH-2", easting=45.0, northing=0.0, elevation=101.5, total_depth=22.0),
        Collar(hole_id="BH-3", easting=110.0, northing=0.0, elevation=99.0, total_depth=16.0),
    ]
    lithologies = [
        Lithology(hole_id=h, from_depth=f, to_depth=t, lithology_code=c)
        for h, rows in {
            "BH-1": ((0.0, 4.0, "Sand"), (4.0, 18.0, "Clay")),
            "BH-2": ((0.0, 5.0, "Sand"), (5.0, 22.0, "Clay")),
            "BH-3": ((0.0, 3.0, "Sand"), (3.0, 16.0, "Clay")),
        }.items()
        for f, t, c in rows
    ]
    points = [(0.0, 0.0), (110.0, 0.0)]
    return collars, lithologies, points


@pytest.fixture(scope="module")
def geometry():
    collars, lithologies, points = _dataset()
    return compute_section_geometry(collars, lithologies, points), points


def _export(geometry, layout: str, page: str, ve: float | None):
    geo, points = geometry
    result = render_cross_section_from_geometry(
        geo,
        points,
        vertical_exaggeration=ve,
        render_layout=layout,
        export_formats=frozenset({"png"}),
        export_framing=ExportFramingConfig(page_preset=page, export_dpi=72),
        close_figure=False,
    )
    bundle = result.retained_export_bundle
    assert bundle is not None
    return result, bundle["figure"], bundle["renderer"]


def _captions(fig) -> list[str]:
    return [t.get_text() for t in fig.findobj(VECaptionText) if t.get_text().strip()]


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("ve", (1.0, 5.0))
def test_exact_ve_is_drawn_after_export(geometry, layout: str, page: str, ve: float) -> None:
    result, fig, renderer = _export(geometry, layout, page, ve)
    try:
        # The PNG really is the requested page.
        width, height = Image.open(io.BytesIO(result.png_bytes)).size
        page_w, page_h = PAGE_FIGSIZE_IN[page]
        assert (width, height) == (round(page_w * 72), round(page_h * 72))
        measured = measured_vertical_exaggeration(fig._css_ve_axes, fig._css_ve_storage)
        assert measured == pytest.approx(ve, rel=0.02)
        assert renderer.current_vertical_exaggeration() == ve
        captions = _captions(fig)
        assert captions, "VE caption missing"
        expected = format_ve_caption(ve, auto=False, style={
            "consulting_section": "consulting",
            "chart": "short",
            "section_sheet": "footer",
        }[layout])
        assert any(expected in text for text in captions), (expected, captions)
        assert not any("≈" in text for text in captions)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("page", PAGES)
def test_auto_caption_matches_measured_ve(geometry, layout: str, page: str) -> None:
    _result, fig, renderer = _export(geometry, layout, page, None)
    try:
        measured = measured_vertical_exaggeration(fig._css_ve_axes, fig._css_ve_storage)
        assert measured is not None and measured > 0
        assert renderer.current_vertical_exaggeration() == pytest.approx(measured)
        captions = _captions(fig)
        assert captions, "VE caption missing"
        if abs(measured - 1.0) <= 0.05:
            assert any("NO VERTICAL" in t or "none (1×)" in t or "1×" in t for t in captions)
        else:
            shown = [float(m) for t in captions for m in re.findall(r"≈([\d.]+)×", t)]
            assert shown, captions
            assert all(value == round_vertical_exaggeration(measured) for value in shown)
    finally:
        plt.close(fig)


def test_auto_caption_follows_export_page() -> None:
    """Re-framing to a different page re-measures the auto VE caption."""
    collars, lithologies, points = _dataset()
    portrait = build_cross_section(
        collars, lithologies, points, render_layout="consulting_section",
        export_formats=frozenset({"svg"}),
        export_framing=ExportFramingConfig(page_preset="letter_portrait"),
    )
    landscape = build_cross_section(
        collars, lithologies, points, render_layout="consulting_section",
        export_formats=frozenset({"svg"}),
        export_framing=ExportFramingConfig(page_preset="tabloid_landscape"),
    )
    pattern = re.compile(r"VERTICAL EXAGGERATION ≈([\d.]+)×")
    portrait_ve = pattern.search(portrait.svg_bytes.decode("utf-8", "ignore"))
    landscape_ve = pattern.search(landscape.svg_bytes.decode("utf-8", "ignore"))
    assert portrait_ve and landscape_ve
    # A tall page stretches the section more than a wide tabloid sheet.
    assert float(portrait_ve.group(1)) > float(landscape_ve.group(1))


def test_exact_ve_changes_geometry() -> None:
    """VE 1 / 5 / 10 must not draw identical plots (the original defect)."""
    collars, lithologies, points = _dataset()
    geo = compute_section_geometry(collars, lithologies, points)
    boxes = []
    for ve in (1.0, 5.0, 10.0):
        result = render_cross_section_from_geometry(
            geo, points, vertical_exaggeration=ve, render_layout="consulting_section",
            export_formats=frozenset({"svg"}), close_figure=False,
        )
        fig = result.retained_export_bundle["figure"]
        ax = fig._css_ve_axes
        ax.apply_aspect()
        pos = ax.get_position()
        boxes.append(pos.height / pos.width)
        plt.close(fig)
    assert boxes[0] < boxes[1] < boxes[2]


def test_consulting_twin_axis_follows_exact_box(geometry) -> None:
    _result, fig, _renderer = _export(geometry, "consulting_section", "letter_landscape", 1.0)
    try:
        main_ax, twin = fig._css_main_axes
        assert twin is not None
        fig.canvas.draw()
        assert twin.get_position().bounds == pytest.approx(main_ax.get_position().bounds)
        assert twin.get_ylim() == pytest.approx(main_ax.get_ylim())
    finally:
        plt.close(fig)


def test_scale_bar_ratio_uses_exact_box(geometry) -> None:
    """The consulting scale text is measured through the shrunk (exact VE) box."""
    _result, fig, _renderer = _export(geometry, "consulting_section", "letter_landscape", 10.0)
    try:
        main_ax = fig._css_main_axes[0]
        main_ax.apply_aspect()
        metres_per_px = abs(main_ax.get_xlim()[1] - main_ax.get_xlim()[0]) / main_ax.bbox.width
        ratio = metres_per_px * fig.dpi * 39.37
        texts = [t.get_text() for ax in fig.axes for t in ax.texts if "SCALE 1:" in t.get_text()]
        assert texts
        printed = float(re.search(r"1:([\d.]+)", texts[0]).group(1))
        assert printed == pytest.approx(ratio, rel=0.06)
    finally:
        plt.close(fig)


def test_section_build_request_ve_default_auto_and_not_geometry() -> None:
    request = SectionBuildRequest(transect_points=((0.0, 0.0), (1.0, 0.0)))
    assert request.vertical_exaggeration is None
    exact = request.model_copy(update={"vertical_exaggeration": 5.0})
    assert request.geometry_cache_key(("A",)) == exact.geometry_cache_key(("A",))
    assert request.cache_key(("A",)) != exact.cache_key(("A",))


def test_format_ve_caption_wording() -> None:
    assert format_ve_caption(1.0, auto=False) == "NO VERTICAL EXAGGERATION"
    assert format_ve_caption(5.0, auto=False) == "5× VERTICAL EXAGGERATION"
    assert format_ve_caption(2.5, auto=False) == "2.5× VERTICAL EXAGGERATION"
    assert format_ve_caption(2.2349, auto=True) == "VERTICAL EXAGGERATION ≈2.2×"
    assert format_ve_caption(1.03, auto=True) == "NO VERTICAL EXAGGERATION"
    assert format_ve_caption(12.6, auto=True, style="short") == "V.E. ≈13×"
    assert format_ve_caption(5.0, auto=False, style="footer") == "VE: 5×"
