"""Water level labels, gradient labels and series styling under stress:
three GW series 0.05 m apart at every hole (user report: numbers collide,
series all look the same blue)."""

from __future__ import annotations

import itertools
from pathlib import Path

from matplotlib.collections import PathCollection
from matplotlib.legend import Legend
from matplotlib.text import Text

from models import ConsultingTitleBlock, DataParser, WaterLevel
from render_profiles import CONSULTING_SECTION_PROFILE, SECTION_SHEET_PROFILE
from render_theme import (
    CONSULTING_GW_BLUE_SHADES,
    CONSULTING_GW_SERIES_STYLES,
    consulting_gw_series_style,
)
from renderer import CrossSectionRenderer
from renderer_water import _figure_renderer
from tests.conftest import run_pipeline

ROOT = Path(__file__).resolve().parents[1]
SERIES = (("2024-05", "May 2024"), ("2025-06", "June 2025"), ("2026-01", "Jan 2026"))


def _contrast_on_white(hex_colour: str) -> float:
    channels = [int(hex_colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    return 1.05 / (luminance + 0.05)


def _stress_render(profile):
    parsed = DataParser().parse_file(ROOT / "data" / "sample_boreholes.xlsx")
    water = [
        WaterLevel(
            hole_id=collar.hole_id,
            depth=3.0 + 0.05 * j + 0.02 * i,
            elevation_masl=collar.elevation - (3.0 + 0.05 * j + 0.02 * i),
            series_id=series_id,
            series_label=label,
        )
        for i, collar in enumerate(parsed.collars)
        for j, (series_id, label) in enumerate(SERIES)
    ]
    transect = [(collar.easting, collar.northing) for collar in parsed.collars]
    projected, polygons, _ = run_pipeline(parsed.collars, parsed.lithologies, transect)
    renderer = CrossSectionRenderer(
        vertical_exaggeration=5.0,
        show_legend=False,
        render_profile=profile,
        consulting_title_block=ConsultingTitleBlock(section_label="A-A'"),
    )
    figure = renderer.render(
        polygons,
        projected,
        collar_depths={collar.hole_id: collar.total_depth for collar in parsed.collars},
        water_levels=water,
    )
    return renderer, figure


def test_stress_water_and_gradient_labels_never_overlap() -> None:
    renderer, figure = _stress_render(CONSULTING_SECTION_PROFILE)
    mpl_renderer = _figure_renderer(figure)
    visible = [
        (kind, annotation)
        for kind, annotation, _c in renderer._water_labels
        if kind in {"rl", "gradient", "nm"} and annotation.get_visible()
    ]
    assert sum(1 for kind, _a in visible if kind == "rl") >= 9
    boxes = []
    for _kind, annotation in visible:
        annotation.update_positions(mpl_renderer)
        boxes.append((annotation.get_text(), Text.get_window_extent(annotation, mpl_renderer)))
    overlaps = [(a[0], b[0]) for a, b in itertools.combinations(boxes, 2) if a[1].overlaps(b[1])]
    assert overlaps == []


def test_gradient_label_drawn_once_per_segment() -> None:
    renderer, _figure = _stress_render(CONSULTING_SECTION_PROFILE)
    gradients = [
        (tuple(annotation.xy), annotation.get_text())
        for kind, annotation, _c in renderer._water_labels
        if kind == "gradient"
    ]
    assert gradients
    assert len(gradients) == len(set(gradients))


def test_every_series_label_colour_meets_contrast_and_series_differ() -> None:
    for colour in CONSULTING_GW_BLUE_SHADES:
        assert _contrast_on_white(colour) >= 4.5, colour
    for colour, _marker, _label in CONSULTING_GW_SERIES_STYLES.values():
        assert _contrast_on_white(colour) >= 4.5, colour
    styles = [consulting_gw_series_style(f"s{i}", series_index=i) for i in range(4)]
    assert len({(c, m) for c, m, _l in styles}) == 4
    assert len({m for _c, m, _l in styles}) == 4  # shape, not only shade
    renderer, _figure = _stress_render(CONSULTING_SECTION_PROFILE)
    entries = renderer.water_series_legend
    assert len(entries) == 3
    assert len({entry["color"] for entry in entries}) == 3
    assert len({entry["marker"] for entry in entries}) == 3
    for _kind, _annotation, colour in renderer._water_labels:
        if _kind in {"rl", "gradient"}:
            assert _contrast_on_white(colour) >= 4.5, colour


def test_section_sheet_shows_series_key_and_fans_out_markers() -> None:
    renderer, figure = _stress_render(SECTION_SHEET_PROFILE)
    texts = [text.get_text() for legend in figure.findobj(Legend) for text in legend.get_texts()]
    for _series_id, label in SERIES:
        assert f"GROUNDWATER LEVEL ({label.upper()})" in texts
    ax = renderer._water_series_key_ax
    figure.draw_without_rendering()
    scatters = [
        c
        for c in ax.collections
        if isinstance(c, PathCollection) and c.get_zorder() == 7 and len(c.get_offsets())
    ]
    assert len(scatters) == 3
    first_hole_px = [c.get_offset_transform().transform(c.get_offsets()[0])[0] for c in scatters]
    # Same hole, different series: markers sit apart, not on top of each other.
    assert min(abs(a - b) for a, b in itertools.combinations(first_hole_px, 2)) > 5.0


def test_single_series_section_sheet_has_no_series_key() -> None:
    parsed = DataParser().parse_file(ROOT / "data" / "sample_boreholes.xlsx")
    water = [
        WaterLevel(hole_id=collar.hole_id, depth=3.0, series_id="2024-05")
        for collar in parsed.collars
    ]
    transect = [(collar.easting, collar.northing) for collar in parsed.collars]
    projected, polygons, _ = run_pipeline(parsed.collars, parsed.lithologies, transect)
    renderer = CrossSectionRenderer(show_legend=False, render_profile=SECTION_SHEET_PROFILE)
    figure = renderer.render(
        polygons,
        projected,
        collar_depths={collar.hole_id: collar.total_depth for collar in parsed.collars},
        water_levels=water,
    )
    texts = [text.get_text() for legend in figure.findobj(Legend) for text in legend.get_texts()]
    assert not any("GROUNDWATER" in text for text in texts)


def test_groundwater_legend_switch_off_hides_the_series_key() -> None:
    """The section-sheet series key must follow the sidebar switch when it
    is turned off explicitly."""
    import pipeline
    from models import Collar, Lithology, WaterLevel

    collars = [
        Collar(hole_id=h, easting=10.0 * i, northing=0.0, elevation=100.0, total_depth=10.0)
        for i, h in enumerate(("BH-1", "BH-2"))
    ]
    liths = [Lithology(hole_id=c.hole_id, from_depth=0.0, to_depth=10.0, lithology_code="Clay") for c in collars]
    water = [
        WaterLevel(hole_id=c.hole_id, depth=2.0 + k * 0.5, series_id=f"S{k}")
        for c in collars
        for k in range(2)
    ]
    captured = {}
    original = pipeline.CrossSectionRenderer.__init__

    def spy(self, *args, **kwargs):
        original(self, *args, **kwargs)
        captured["profile"] = self.profile

    pipeline.CrossSectionRenderer.__init__ = spy
    try:
        pipeline.build_cross_section(
            collars, liths, [(0.0, 0.0), (10.0, 0.0)], water_levels=water, show_water_legend=False
        )
        assert captured["profile"].water_series_key_min_series == 0
        pipeline.build_cross_section(collars, liths, [(0.0, 0.0), (10.0, 0.0)], water_levels=water)
        assert captured["profile"].water_series_key_min_series > 0
    finally:
        pipeline.CrossSectionRenderer.__init__ = original
