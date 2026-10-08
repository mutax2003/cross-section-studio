"""Structural parity tests for EcoVenture GWM figures 3–6."""

from __future__ import annotations

import pytest

from gwm_reference import GWM_TRANSECTS, build_subset
from pipeline import build_cross_section
from ui_helpers import svg_is_valid


@pytest.mark.parametrize("transect_id", tuple(GWM_TRANSECTS.keys()))
def test_ecoventure_gwm_figure_structure(transect_id: str) -> None:
    spec, subset = build_subset(transect_id)
    transect_points = [(collar.easting, collar.northing) for collar in subset.collars]
    result = build_cross_section(
        subset.collars,
        subset.lithologies,
        transect_points,
        render_layout="consulting_section",
        vertical_exaggeration=spec.vertical_exaggeration,
        show_legend=False,
        show_hatches=True,
        water_levels=subset.water_levels,
        consulting_title_block=spec.title_block,
        screen_intervals=subset.screen_intervals,
        export_formats=frozenset({"svg"}),
    )
    assert svg_is_valid(result.svg_bytes)
    text = result.svg_bytes.decode("utf-8", errors="ignore")
    assert "DISTANCE (m)" in text
    assert "ELEVATION ABOVE SEA LEVEL (MASL)" in text
    assert "SCREENED INTERVAL" in text
    assert spec.title_block.prepared_for in text
    assert "PROJECT" in text
    assert "100/09-29" in text
    for hole_id in spec.hole_ids:
        assert hole_id in text
    assert "hatch" in text.lower() or "pattern" in text.lower()
    # Fence polygons present (path fills) — not borehole-only sticks.
    assert text.count("<path") >= 4
    if transect_id == "A_A":
        assert "WITH GROUNDWATER LEVELS" in text
        assert "May 2024" in text or "2024-05" in text or "GROUNDWATER LEVEL (MAY 2024)" in text.upper()
    # The client's 1:1 000 / 1:1 500 sheets are tabloid; on the default letter
    # page the section can't be drawn at that scale, so the title block defers
    # to the scale bar and the build reports why.
    assert "AS SHOWN" in text
    assert result.qa_notes
    assert spec.title_block.map_scale in result.qa_notes[0]
    assert "letter landscape" in result.qa_notes[0]


def test_ecoventure_gwm_d_d_prints_title_block_scale_on_tabloid() -> None:
    """Fig 6 at 1:1 500 fits the client's tabloid page: drawn at exactly that scale."""
    from export_framing import ExportFramingConfig

    spec, subset = build_subset("D_D")
    result = build_cross_section(
        subset.collars,
        subset.lithologies,
        [(collar.easting, collar.northing) for collar in subset.collars],
        render_layout="consulting_section",
        vertical_exaggeration=spec.vertical_exaggeration,
        show_legend=False,
        water_levels=subset.water_levels,
        consulting_title_block=spec.title_block,
        screen_intervals=subset.screen_intervals,
        export_formats=frozenset({"svg"}),
        export_framing=ExportFramingConfig(page_preset="tabloid_landscape"),
    )
    text = result.svg_bytes.decode("utf-8", errors="ignore")
    assert result.qa_notes == ()
    assert "1:1 500" in text
    assert "SCALE 1:1500" in text
    assert "AS SHOWN" not in text


def test_ecoventure_gwm_hole_order_matches_transect_spec() -> None:
    for transect_id, spec in GWM_TRANSECTS.items():
        _spec, subset = build_subset(transect_id)
        rendered_order = tuple(collar.hole_id for collar in subset.collars)
        assert rendered_order == spec.hole_ids
        assert tuple(subset.collars[i].easting for i in range(len(subset.collars))) == spec.profile_eastings


def test_ecoventure_dual_gw_series_a_a() -> None:
    spec, subset = build_subset("A_A")
    series_ids = {level.series_id for level in subset.water_levels}
    assert "2024-05" in series_ids
    assert "2025-06" in series_ids
    series_labels = {level.series_label for level in subset.water_levels}
    assert "May 2024" in series_labels
    assert "June 2025" in series_labels
    transect_points = [(collar.easting, collar.northing) for collar in subset.collars]
    result = build_cross_section(
        subset.collars,
        subset.lithologies,
        transect_points,
        render_layout="consulting_section",
        vertical_exaggeration=spec.vertical_exaggeration,
        show_legend=False,
        water_levels=subset.water_levels,
        consulting_title_block=spec.title_block,
        screen_intervals=subset.screen_intervals,
        export_formats=frozenset({"svg"}),
    )
    text = result.svg_bytes.decode("utf-8", errors="ignore").upper()
    assert "GROUNDWATER LEVEL (MAY 2024)" in text or "MAY 2024" in text
    assert "JUNE 2025" in text


def test_ecoventure_screens_are_per_hole_not_uniform() -> None:
    _spec, subset = build_subset("A_A")
    screens = {item.hole_id: (item.from_depth, item.to_depth) for item in subset.screen_intervals}
    assert "MW18-18" in screens
    assert screens["MW18-18"] != (12.0, 18.0)
    assert screens["MW18-06B"][0] < screens["MW18-06B"][1]


def test_transect_spec_validates_hole_profile_lengths() -> None:
    from gwm_reference.transects import TransectSpec
    from models import ConsultingTitleBlock

    with pytest.raises(ValueError, match="length mismatch"):
        TransectSpec(
            transect_id="bad",
            figure_number="0",
            hole_ids=("A", "B"),
            profile_eastings=(0.0,),
            title_block=ConsultingTitleBlock(),
        )


def test_screen_interval_warnings_helper() -> None:
    from models import ScreenInterval
    from ui_helpers import screen_interval_warnings

    warnings = screen_interval_warnings(
        ("MW-01", "MW-02"),
        (ScreenInterval(hole_id="MW-01", from_depth=5.0, to_depth=10.0),),
    )
    assert len(warnings) == 1
    assert "MW-02" in warnings[0]


@pytest.mark.parametrize("transect_id", ["A_A", "D_D"])
def test_end_labels_sit_above_the_edge_hole_headers(transect_id: str) -> None:
    """"A / NORTHWEST" shared the header row and the edge headers printed into
    it ("MW18-18NORTHWEST"); the client figures stack the end label above."""
    import matplotlib.pyplot as plt

    import renderer as renderer_mod

    spec, subset = build_subset(transect_id)
    captured = {}
    original = renderer_mod.CrossSectionRenderer.render

    def spy(self, *args, **kwargs):
        captured["fig"] = original(self, *args, **kwargs)
        return captured["fig"]

    renderer_mod.CrossSectionRenderer.render = spy
    try:
        build_cross_section(
            subset.collars,
            subset.lithologies,
            [(collar.easting, collar.northing) for collar in subset.collars],
            render_layout="consulting_section",
            vertical_exaggeration=spec.vertical_exaggeration,
            water_levels=subset.water_levels,
            consulting_title_block=spec.title_block,
            screen_intervals=subset.screen_intervals,
            export_formats=frozenset({"png"}),
        )
    finally:
        renderer_mod.CrossSectionRenderer.render = original
    fig = captured["fig"]
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        ax = fig.axes[0]
        ends = [t for t in ax.texts if t.get_text().split("\n")[0] in {spec.title_block.transect_start_primary, spec.title_block.transect_end_primary} and "\n" in t.get_text()]
        assert ends, "end labels drawn"
        hole_ids = {c.hole_id for c in subset.collars}
        headers = [t for t in list(ax.texts) + list(fig.texts) if t.get_text() in hole_ids]
        assert headers
        for end in ends:
            end_box = end.get_window_extent(renderer)
            for header in headers:
                assert not end_box.overlaps(header.get_window_extent(renderer)), header.get_text()
    finally:
        plt.close(fig)
