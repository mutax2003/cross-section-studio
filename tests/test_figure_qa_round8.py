"""Figure QA round 8 regressions (one block per confirmed defect).

1. An inclined hole is ONE continuous slanted column (with its track,
   screens, ticks and end-of-log bar on the trace); vertical holes keep
   their exact rectangles.
2. Consulting section-end labels ("A" / "A'") clear the rotated hole headers.
3. The wider-spacing zone is legible (sparse hatch + dashed outline) in the
   section and its legend key.
4. A tight / auto PDF never mixes page sizes.
5. A locked VE that leaves a thin strip gets a QA note and a caption.
6. Tight crops keep the framed headers inside the 0.25 in margin.
7. Section sheet / chart legends use one case convention.
8. A chemistry leader crosses a water line only when no other spot exists.
Extras: LNAPL bands are label obstacles; pinch-out wedges are no darker
than the sheet's fills; the plot keeps >= 60% beside an outside legend.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from export_framing import ExportFramingConfig  # noqa: E402
from models import (  # noqa: E402
    Collar,
    ConsultingTitleBlock,
    DeviationReading,
    Lithology,
    ScreenInterval,
    WaterLevel,
)
from pipeline import (  # noqa: E402
    build_cross_section,
    compute_section_geometry,
    render_cross_section_from_geometry,
)


def _collars(rls, *, spacing=50.0, td=20.0):
    return [
        Collar(hole_id=hole_id, easting=spacing * index, northing=0.0, elevation=rl, total_depth=td)
        for index, (hole_id, rl) in enumerate(rls)
    ]


def _liths(collars):
    return [
        Lithology(hole_id=c.hole_id, from_depth=0.0, to_depth=c.total_depth / 2, lithology_code="SAND")
        for c in collars
    ] + [
        Lithology(hole_id=c.hole_id, from_depth=c.total_depth / 2, to_depth=c.total_depth, lithology_code="CLAY")
        for c in collars
    ]


_DEVIATIONS = [
    DeviationReading(hole_id="B", depth=0.0, inclination_deg=0.0, azimuth_deg=90.0),
    DeviationReading(hole_id="B", depth=3.0, inclination_deg=30.0, azimuth_deg=90.0),
    DeviationReading(hole_id="B", depth=20.0, inclination_deg=30.0, azimuth_deg=90.0),
]


def _render(collars, *, deviations=(), water=None, formats=frozenset({"svg"}), **kwargs):
    points = [(c.easting, c.northing) for c in collars]
    geometry = compute_section_geometry(collars, _liths(collars), points, deviation_readings=list(deviations))
    result = render_cross_section_from_geometry(
        geometry,
        points,
        water_levels=water,
        export_formats=formats,
        close_figure=False,
        **kwargs,
    )
    bundle = result.retained_export_bundle
    fig = bundle["figure"]
    fig.canvas.draw()
    return result, bundle["renderer"], fig


def _ink_margins_in(png: bytes) -> tuple[float, float, float, float]:
    image = Image.open(io.BytesIO(png))
    dpi = float((image.info.get("dpi") or (300, 300))[0])
    ink = np.asarray(image.convert("L")) < 245
    ys, xs = np.where(ink)
    height, width = ink.shape
    return (xs.min() / dpi, (width - 1 - xs.max()) / dpi, ys.min() / dpi, (height - 1 - ys.max()) / dpi)


# --- 1. Inclined hole: one continuous slanted column --------------------------


def _paths(fig, zorder):
    return [
        path.vertices
        for ax in fig.axes
        for coll in ax.collections
        if coll.get_zorder() == zorder and hasattr(coll, "get_paths")
        for path in coll.get_paths()
    ]


def test_inclined_column_is_one_continuous_polygon_in_every_mode() -> None:
    collars = _collars([("A", 100.0), ("B", 100.0), ("C", 100.0)])
    screens = [ScreenInterval(hole_id="B", from_depth=8.0, to_depth=14.0)]
    for elevation_mode in ("absolute", "relative"):
        _result, renderer, fig = _render(
            collars,
            deviations=_DEVIATIONS,
            render_layout="consulting_section",
            interpretation_mode="borehole_only",
            screen_intervals=screens,
            elevation_mode=elevation_mode,
        )
        try:
            trace_y, trace_x = renderer._deviated_trace_plot("B", 100.0)
            columns = _paths(fig, 8)
            # A and C: plain 4-corner rectangles centred on the collar x.
            rects = [v for v in columns if len(v) <= 5]
            assert sorted(round(float(np.mean(v[:4, 0])), 6) for v in rects) == [0.0, 100.0]
            slanted = [v for v in columns if len(v) > 5]
            assert len(slanted) == 1, "inclined column must be one polygon"
            poly = slanted[0]
            # Its left edge follows the trace, collar to TD, without a gap.
            half = (float(poly[:, 0].max()) - float(trace_x.max())) if trace_x.max() > 0 else 0.0
            for y in np.linspace(float(trace_y.min()) + 0.1, float(trace_y.max()) - 0.1, 7):
                x_mid = renderer._deviated_x_at("B", 100.0, float(y), 0.0)
                left = poly[: len(poly) // 2]
                order = np.argsort(left[:, 1])
                x_left = float(np.interp(y, left[order, 1], left[order, 0]))
                assert abs((x_mid - x_left) - half) < 0.05
            # The screen follows the trace too (a quad, not a vertical box).
            screen_quads = [v for v in _paths(fig, 9) if len(v) >= 4]
            assert any(abs(float(v[0, 0]) - float(v[3, 0])) > 0.5 for v in screen_quads)
        finally:
            plt.close(fig)


def test_section_sheet_track_ticks_and_eol_bar_follow_the_trace() -> None:
    collars = _collars([("A", 100.0), ("B", 100.0), ("C", 100.0)])
    _result, renderer, fig = _render(collars, deviations=_DEVIATIONS, render_layout="section_sheet")
    try:
        trace_y, trace_x = renderer._deviated_trace_plot("B", 100.0)
        td_y = float(trace_y.min())
        bars = [
            seg
            for ax in fig.axes
            for coll in ax.collections
            if coll.get_zorder() == 8 and hasattr(coll, "get_segments")
            for seg in coll.get_segments()
            if len(seg) == 2 and abs(seg[0][1] - seg[1][1]) < 1e-9 and abs(seg[0][1] - td_y) < 1e-6
        ]
        assert bars, "EOL bar at B's TD"
        centre = 0.5 * (bars[0][0][0] + bars[0][1][0])
        assert abs(centre - renderer._deviated_x_at("B", 100.0, td_y, 0.0)) < 1e-6
        assert centre > 50.0 + 1.0  # off the collar x
        # Track border of B: one slanted polygon.
        borders = [v for v in _paths(fig, 6) if len(v) > 5]
        assert len(borders) == 1
    finally:
        plt.close(fig)


def test_vertical_holes_keep_exact_rectangles() -> None:
    collars = _collars([("A", 100.0), ("B", 100.0), ("C", 100.0)])
    _result, renderer, fig = _render(collars, render_layout="consulting_section", interpretation_mode="borehole_only")
    try:
        assert renderer._deviated_x == {}
        columns = _paths(fig, 8)
        assert all(len(v) <= 5 for v in columns)
    finally:
        plt.close(fig)


# --- 2. Section-end labels clear rotated headers ----------------------------


def test_section_end_labels_clear_rotated_headers() -> None:
    from renderer_common import _header_extent

    # The advantage_phase2 layout: two dense clusters with a wide gap.
    xs = [50.0 * i for i in range(12)] + [680.0, 915.0, 1145.0] + [1480.0 + 50.0 * i for i in range(8)]
    collars = [
        Collar(hole_id=f"BH26-{i:02d}", easting=x, northing=0.0, elevation=100.0, total_depth=12.0)
        for i, x in enumerate(xs, start=1)
    ]
    block = ConsultingTitleBlock(
        section_label="A-A'", transect_start_primary="A", transect_end_primary="A'"
    )
    _result, renderer, fig = _render(
        collars,
        render_layout="consulting_section",
        consulting_title_block=block,
        interpretation_mode="borehole_only",
        formats=frozenset({"png"}),
    )
    try:
        renderer_ = fig.canvas.get_renderer()
        headers = [h for h in renderer._header_labels if h.figure is fig and h.get_visible()]
        assert len(headers) == len(xs)
        ends = [t for ax in fig.axes for t in ax.texts if getattr(t, "_css_section_end", None)]
        assert len(ends) == 2
        for label in ends:
            box = label.get_window_extent(renderer_)
            masked = label.get_bbox_patch() is not None
            for header in headers:
                other = _header_extent(header, renderer_)
                assert masked or not box.overlaps(other), (label.get_text(), header.get_text())
    finally:
        plt.close(fig)


# --- 3. Wider-spacing zone legible -------------------------------------------


def test_uncertainty_zone_has_hatch_outline_and_matching_key() -> None:
    from renderer import UNCERTAINTY_HATCH
    from renderer_common import UncertaintyZoneKey

    collars = _collars([("A", 100.0), ("B", 100.0), ("C", 100.0)], spacing=400.0)
    _result, _renderer, fig = _render(collars, render_layout="chart", show_legend=True)
    try:
        ax = fig.axes[0]
        overlay = [c for c in ax.collections if c.get_gid() == "uncertainty-zone-hatch"]
        assert overlay and overlay[0].get_hatch() == UNCERTAINTY_HATCH
        assert overlay[0].get_zorder() > 2  # over the fence fills
        legend = ax.get_legend()
        sources = [h for h, _l in getattr(legend, "_css_source_keys", [])]
        assert any(isinstance(h, UncertaintyZoneKey) for h in sources)
    finally:
        plt.close(fig)


# --- 4. PDF page sizes ------------------------------------------------------


def test_tight_pdf_pages_share_one_size() -> None:
    from pypdf import PdfReader

    collars = _collars([("A", 100.0), ("B", 101.0), ("C", 99.0)])
    for layout in ("section_sheet", "chart"):
        result = build_cross_section(
            collars,
            _liths(collars),
            [(c.easting, c.northing) for c in collars],
            render_layout=layout,
            export_formats=frozenset({"pdf"}),
            export_framing=ExportFramingConfig(page_preset="tight_fence"),
        )
        pages = PdfReader(io.BytesIO(result.pdf_bytes)).pages
        assert len(pages) >= 2
        sizes = {(round(float(p.mediabox.width) / 72.0, 2), round(float(p.mediabox.height) / 72.0, 2)) for p in pages}
        assert len(sizes) == 1, sizes


# --- 5. Locked VE thin strip -------------------------------------------------


def test_locked_ve_strip_gets_note_and_caption() -> None:
    xs = np.linspace(0.0, 1800.0, 10)
    collars = [
        Collar(hole_id=f"H{i}", easting=float(x), northing=0.0, elevation=100.0, total_depth=12.0)
        for i, x in enumerate(xs)
    ]
    result, _renderer, fig = _render(
        collars,
        render_layout="consulting_section",
        vertical_exaggeration=1.0,
        interpretation_mode="borehole_only",
        formats=frozenset({"png"}),
    )
    try:
        assert any(note.startswith("Section too long for 1×") for note in result.qa_notes), result.qa_notes
        captions = [t for t in fig.texts if t.get_gid() == "locked-ve-strip-caption"]
        assert captions and "choose a VE or split the section" in captions[0].get_text()
    finally:
        plt.close(fig)
    # A short section at 1x keeps a usable height: no note.
    short = _collars([("A", 100.0), ("B", 100.0)], spacing=20.0)
    result, _renderer, fig = _render(
        short, render_layout="consulting_section", vertical_exaggeration=1.0, formats=frozenset({"png"})
    )
    plt.close(fig)
    assert not any("Section too long" in note for note in result.qa_notes)


# --- 6. Tight crop keeps framed headers inside the margin -------------------


def test_tight_crop_keeps_framed_headers_in_margin() -> None:
    xs = np.arange(0.0, 23 * 80.0, 80.0)
    collars = [
        Collar(hole_id=f"BH26-{i:02d}", easting=float(x), northing=0.0, elevation=100.0, total_depth=12.0)
        for i, x in enumerate(xs, start=1)
    ]
    result = build_cross_section(
        collars,
        _liths(collars),
        [(c.easting, c.northing) for c in collars],
        render_layout="chart",
        export_formats=frozenset({"png"}),
        export_framing=ExportFramingConfig(page_preset="tight_fence", export_dpi=150),
    )
    assert min(_ink_margins_in(result.png_bytes)) >= 0.25


# --- 7. One case convention per legend ----------------------------------------


def test_section_sheet_legend_uses_one_case(monkeypatch) -> None:
    import renderer as renderer_mod

    captured: dict = {}
    original = renderer_mod.CrossSectionRenderer.render

    def spy(self, *args, **kwargs):
        captured["fig"] = original(self, *args, **kwargs)
        return captured["fig"]

    monkeypatch.setattr(renderer_mod.CrossSectionRenderer, "render", spy)
    collars = _collars([("A", 100.0), ("B", 100.0), ("C", 100.0)])
    liths = [
        Lithology(hole_id=c.hole_id, from_depth=0.0, to_depth=20.0, lithology_code=code)
        for c, code in zip(collars, ("Sand", "Clay", "Sand"), strict=True)
    ]
    water = [WaterLevel(hole_id=h, depth=3.0, series_id="s", series_label="June 2025") for h in "ABC"] + [
        WaterLevel(hole_id=h, depth=3.6, series_id="t", series_label="Fall 2025") for h in "ABC"
    ]
    result = build_cross_section(
        collars,
        liths,
        [(c.easting, c.northing) for c in collars],
        render_layout="section_sheet",
        water_levels=water,
        show_legend=True,
        export_formats=frozenset({"png"}),
    )
    assert result.png_bytes
    fig = captured["fig"]
    try:
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert "Groundwater level (June 2025)" in labels, labels
        assert not [label for label in labels if label.isupper() and len(label) > 3], labels
    finally:
        plt.close(fig)


def test_sentence_case_keeps_parameter_and_unit() -> None:
    from renderer_common import sentence_case_legend_label

    assert sentence_case_legend_label("BENZENE CONCENTRATION (mg/L)", "Benzene") == "Benzene concentration (mg/L)"
    assert sentence_case_legend_label("PH CONCENTRATION", "pH") == "pH concentration"
    assert sentence_case_legend_label("Chloride (mg/L)", "Chloride") == "Chloride (mg/L)"


# --- 8. Chemistry leaders and water lines ------------------------------------


def test_chemistry_spot_without_crossing_leader_wins() -> None:
    from renderer_water import _prefer_clear_spots

    # Anchor at (0, 0); a horizontal water line just below it (y = -5 px).
    line = np.array([[-100.0, -5.0, 400.0, -5.0]])
    boxes = np.array(
        [
            [10.0, -40.0, 40.0, -30.0],  # far below the line: leader crosses it
            [10.0, -8.0, 40.0, 2.0],  # beside the sample, box touches the line
        ]
    )
    ddx = np.array([10.0, 10.0])
    ddy = np.array([-35.0, -3.0])
    idx = np.arange(2)
    kept = _prefer_clear_spots(
        idx,
        boxes,
        ddx,
        ddy,
        line_segs=line,
        line_clear=1.0,
        label_arrs=(np.zeros((0, 4)), np.zeros((0, 4))),
        anchor=(0.0, 0.0),
        label_height_pt=8.0,
        px_per_pt=1.0,
    )
    assert kept.tolist() == [1]


# --- Extras ------------------------------------------------------------------


def test_lnapl_bands_are_label_obstacles() -> None:
    collars = _collars([("A", 100.0), ("B", 100.0), ("C", 100.0)])
    water = [
        WaterLevel(hole_id="B", depth=4.0, product_thickness_m=0.3),
        WaterLevel(hole_id="A", depth=4.0),
        WaterLevel(hole_id="C", depth=4.0),
    ]
    _result, renderer, fig = _render(collars, water=water, render_layout="section_sheet")
    try:
        boxes = renderer._lnapl_band_boxes(fig)
        assert len(boxes) == 1
        assert boxes[0] in renderer._column_extra_obstacle_boxes(fig) or any(
            abs(b.x0 - boxes[0].x0) < 1e-6 for b in renderer._column_extra_obstacle_boxes(fig)
        )
    finally:
        plt.close(fig)


def test_pinch_out_wedges_no_darker_than_sheet_fills() -> None:
    from render_profiles import SECTION_SHEET_PROFILE
    from render_theme import PINCH_OUT_ALPHA

    assert SECTION_SHEET_PROFILE.fence_alpha < PINCH_OUT_ALPHA
    collars = _collars([("A", 100.0), ("B", 100.0), ("C", 100.0)])
    liths = [
        Lithology(hole_id="A", from_depth=0.0, to_depth=10.0, lithology_code="SAND"),
        Lithology(hole_id="A", from_depth=10.0, to_depth=12.0, lithology_code="SILT"),
        Lithology(hole_id="A", from_depth=12.0, to_depth=20.0, lithology_code="CLAY"),
        Lithology(hole_id="B", from_depth=0.0, to_depth=10.0, lithology_code="SAND"),
        Lithology(hole_id="B", from_depth=10.0, to_depth=20.0, lithology_code="CLAY"),
        Lithology(hole_id="C", from_depth=0.0, to_depth=10.0, lithology_code="SAND"),
        Lithology(hole_id="C", from_depth=10.0, to_depth=20.0, lithology_code="CLAY"),
    ]
    points = [(c.easting, c.northing) for c in collars]
    geometry = compute_section_geometry(collars, liths, points, allow_pinch_outs=True)
    result = render_cross_section_from_geometry(
        geometry, points, render_layout="section_sheet", export_formats=frozenset({"svg"}), close_figure=False
    )
    fig = result.retained_export_bundle["figure"]
    try:
        alphas = [
            c.get_alpha()
            for ax in fig.axes
            for c in ax.collections
            if "--" in str(c.get_linestyle()) or any(ls[1] for ls in c.get_linestyle() if ls[1] is not None)
        ]
        alphas = [a for a in alphas if a is not None]
        assert alphas, "pinch-out wedge expected"
        assert max(alphas) <= SECTION_SHEET_PROFILE.fence_alpha + 1e-9
    finally:
        plt.close(fig)


def test_plot_keeps_sixty_percent_beside_outside_legend() -> None:
    from renderer import LEGEND_PLOT_MIN_SHARE

    collars = _collars([("A", 100.0), ("B", 101.0), ("C", 99.0), ("D", 100.0)])
    water = [
        WaterLevel(
            hole_id=h, depth=3.0, series_id="s", series_label="June 2025, 2025-06-11 to 06-12 monitoring event"
        )
        for h in "ABCD"
    ] + [WaterLevel(hole_id=h, depth=3.5, series_id="t", series_label="Fall 2025") for h in "ABCD"]
    result, renderer, fig = _render(
        collars,
        water=water,
        render_layout="section_sheet",
        show_legend=True,
        vertical_exaggeration=5.0,
        formats=frozenset({"png"}),
        export_framing=ExportFramingConfig(page_preset="tight_fence"),
    )
    try:
        r = fig.canvas.get_renderer()
        ax = fig.axes[0]
        ax.apply_aspect()
        plot_w = ax.get_window_extent(r).width
        legend_w = ax.get_legend().get_window_extent(r).width
        assert plot_w / (plot_w + legend_w) >= LEGEND_PLOT_MIN_SHARE - 0.01
    finally:
        plt.close(fig)
