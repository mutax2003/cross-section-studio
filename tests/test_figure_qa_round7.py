"""Figure QA round 7 regressions (one block per confirmed defect).

1. GWM fixture water levels / collars / spacing match the client sheets.
2. Long title-block TITLE wraps (>= 7 pt), never "…" or a ~5 pt one-liner.
3. Section-end labels stay clear of rotated / fanned hole-header leaders.
4. Nearly collinear wells give no flow direction and no figure note.
5. Template sample data is self-consistent.
6. Section sheet groundwater key sits with the lithology legend, swatches
   carry the fence alpha.
7. Every fixed page keeps the 0.25 in printable margin; tight crops too;
   section sheet portrait is balanced vertically.
8. The QA notes PDF page uses the figure page's sheet.
9. Wording: masl axis, one VE family, no "1. NOTE:", "pH".
10. No generic threshold band for a parameter without limits.
"""

from __future__ import annotations

import importlib.util
import io
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from export_framing import ExportFramingConfig  # noqa: E402
from gwm_reference import GWM_TRANSECTS, build_subset  # noqa: E402
from gwm_reference.fixtures import COLLAR_META, GW_MASL  # noqa: E402
from models import Collar, ConsultingTitleBlock, Lithology, WaterLevel  # noqa: E402
from pipeline import build_cross_section  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _capture_render(monkeypatch):
    import renderer as renderer_mod

    captured: dict = {}
    original = renderer_mod.CrossSectionRenderer.render

    def spy(self, *args, **kwargs):
        captured["renderer"] = self
        captured["fig"] = original(self, *args, **kwargs)
        return captured["fig"]

    monkeypatch.setattr(renderer_mod.CrossSectionRenderer, "render", spy)
    return captured


def _ink_margins_in(png: bytes) -> tuple[float, float, float, float]:
    image = Image.open(io.BytesIO(png))
    dpi = float((image.info.get("dpi") or (300, 300))[0])
    ink = np.asarray(image.convert("L")) < 245
    ys, xs = np.where(ink)
    height, width = ink.shape
    return (
        xs.min() / dpi,
        (width - 1 - xs.max()) / dpi,
        ys.min() / dpi,
        (height - 1 - ys.max()) / dpi,
    )


# --- 1. GWM fixture ----------------------------------------------------------

# Read off data/pdf_extract Figs 3-6: (May 2024 cyan, June 2025 blue), masl.
_CLIENT_WATER = {
    "MW18-18": (629.903, 629.870),
    "MW18-06B": (630.142, 630.451),
    "MW18-16": (631.378, 630.491),
    "MW18-08D": (630.971, 629.900),
    "MW18-24": (631.618, 631.188),
    "MW18-17": (631.293, 630.868),
    "MW18-20": (629.311, 628.817),
    "MW18-21": (630.400, 630.060),
    "MW18-19": (629.847, 629.911),
    "MW18-22": (630.874, 630.308),
    "MW18-23": (631.674, 631.193),
}


def test_gwm_water_levels_sit_on_the_client_wells() -> None:
    assert {hid: (v["2024-05"], v["2025-06"]) for hid, v in GW_MASL.items()} == _CLIENT_WATER
    # Boreholes carry no reading on any client sheet.
    assert not any(hid.startswith("BH") for hid in GW_MASL)
    for transect_id in GWM_TRANSECTS:
        _spec, subset = build_subset(transect_id)
        collars = {c.hole_id: c for c in subset.collars}
        for level in subset.water_levels:
            # Head = collar - depth reproduces the printed value (never clipped
            # at the ground: MW18-17 used to sit 0.3 m below its own level).
            head = collars[level.hole_id].elevation - level.depth
            expected = _CLIENT_WATER[level.hole_id][0 if level.series_id == "2024-05" else 1]
            assert head == pytest.approx(expected, abs=1e-6)
            assert level.depth > 0.0


def test_gwm_fig3_spacing_and_depths_follow_the_client_sheet() -> None:
    spec = GWM_TRANSECTS["A_A"]
    # MW18-16 / BH18-05 are 7.5 m apart on Fig 3 (not an even 32 m grid).
    assert spec.profile_eastings == (0.0, 46.5, 93.0, 100.5, 116.0, 139.0)
    # Casings: 6-18 m deep on the client sheets (the fixture had 28-32 m).
    assert all(depth <= 18.0 for _elev, depth in COLLAR_META.values())


# --- 2. Title cell wraps -----------------------------------------------------


def _two_holes() -> tuple[list[Collar], list[Lithology]]:
    collars = [
        Collar(hole_id="MW-01", easting=0.0, northing=0.0, elevation=640.0, total_depth=15.0),
        Collar(hole_id="MW-02", easting=120.0, northing=0.0, elevation=639.0, total_depth=15.0),
    ]
    liths = [
        Lithology(hole_id=c.hole_id, from_depth=0.0, to_depth=15.0, lithology_code="Clay")
        for c in collars
    ]
    return collars, liths


@pytest.mark.parametrize("page", ["letter_landscape", "letter_portrait"])
def test_long_title_wraps_at_seven_points_without_ellipsis(monkeypatch, page: str) -> None:
    captured = _capture_render(monkeypatch)
    collars, liths = _two_holes()
    title = "CROSS SECTION A - A' WITH CHLORIDE AVERAGES"
    block = ConsultingTitleBlock(
        section_label=title,
        project_number="100/09-36-055-02 W4M",
        source="ECOVENTURE 2026",
        date="06/24/26",
        drawn_by="SL/JG",
        revised="A",
        figure_number="1",
        prepared_for="C-GROUP ENERGY INC.",
        prepared_by="ECOVENTURE",
    )
    build_cross_section(
        collars,
        liths,
        [(c.easting, c.northing) for c in collars],
        render_layout="consulting_section",
        consulting_title_block=block,
        export_formats=frozenset({"png"}),
        export_framing=ExportFramingConfig(page_preset=page),
    )
    fig = captured["fig"]
    try:
        texts = [t for ax in fig.axes for t in ax.texts]
        assert not any("…" in t.get_text() for t in texts)
        block_ax = next(ax for ax in fig.axes if any(t.get_text() == "TITLE" for t in ax.texts))
        cell = [
            t for t in block_ax.texts if t.get_text() and t.get_text() in title and len(t.get_text()) > 6
        ]
        # The TITLE value wraps onto lines >= 7 pt that together spell it.
        assert len(cell) >= 2, "TITLE wrapped into lines"
        assert min(t.get_fontsize() for t in cell) >= 7.0 - 1e-9
        assert " ".join(t.get_text() for t in cell) == title
    finally:
        plt.close(fig)


# --- 3. Section-end labels clear header leaders -------------------------------


def test_section_end_labels_clear_header_leaders(monkeypatch) -> None:
    from renderer_common import _leader_segments_px, _segment_x_span

    captured = _capture_render(monkeypatch)
    xs = [0.0, 41.0, 84.0, 126.0, 166.0, 207.0, 209.0, 247.0, 289.0]
    collars = [
        Collar(hole_id=f"MW-{i:02d}", easting=x, northing=0.0, elevation=640.0 - (i % 3), total_depth=20.0)
        for i, x in enumerate(xs, start=1)
    ]
    liths = [
        Lithology(hole_id=c.hole_id, from_depth=0.0, to_depth=20.0, lithology_code="Sand")
        for c in collars
    ]
    block = ConsultingTitleBlock(
        section_label="A - A'",
        transect_start_primary="A",
        transect_start_secondary="NORTHWEST",
        transect_end_primary="A'",
        transect_end_secondary="SOUTHEAST",
    )
    build_cross_section(
        collars,
        liths,
        [(c.easting, c.northing) for c in collars],
        render_layout="consulting_section",
        vertical_exaggeration=5.0,
        consulting_title_block=block,
        export_formats=frozenset({"png"}),
    )
    fig = captured["fig"]
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        segments = _leader_segments_px(fig)
        ends = [t for ax in fig.axes for t in ax.texts if getattr(t, "_css_section_end", None)]
        assert len(ends) == 2
        for label in ends:
            box = label.get_window_extent(renderer)
            masked = label.get_bbox_patch() is not None
            for segment in segments:
                span = _segment_x_span(segment, box.y0, box.y1)
                crossing = span is not None and span[1] > box.x0 and span[0] < box.x1
                assert masked or not crossing, (label.get_text(), segment)
    finally:
        plt.close(fig)


# --- 4. Collinear wells ------------------------------------------------------


def test_nearly_collinear_wells_give_no_flow_note() -> None:
    from hydro_metrics import (
        HeadPoint,
        fit_head_plane,
        format_flow_direction,
        groundwater_flow_notes,
    )

    # 16 m of northing spread over 280 m (QA: printed "toward 181°, i = 0.47").
    points = [
        HeadPoint("MW-01", 0.0, 0.0, 636.70),
        HeadPoint("MW-02", 70.0, 16.0, 636.84),
        HeadPoint("MW-03", 140.0, 4.0, 628.00),
        HeadPoint("MW-04", 210.0, 12.0, 638.78),
        HeadPoint("MW-05", 280.0, 2.0, 637.43),
    ]
    fit = fit_head_plane(points)
    assert fit is not None and fit.collinear and not fit.usable
    assert format_flow_direction(fit) == "wells nearly collinear — flow direction not determined"
    collars = [
        Collar(hole_id=p.hole_id, easting=p.easting, northing=p.northing, elevation=645.0, total_depth=30.0)
        for p in points
    ]
    levels = [WaterLevel(hole_id=p.hole_id, depth=645.0 - p.head_masl) for p in points]
    assert not groundwater_flow_notes(collars, levels, 90.0)


def test_well_spread_field_still_fits() -> None:
    from hydro_metrics import HeadPoint, fit_head_plane

    points = [
        HeadPoint("A", 0.0, 0.0, 60.0),
        HeadPoint("B", 200.0, 0.0, 58.0),
        HeadPoint("C", 0.0, 200.0, 60.0),
        HeadPoint("D", 200.0, 200.0, 58.0),
    ]
    fit = fit_head_plane(points)
    assert fit is not None and fit.usable
    assert fit.flow_azimuth_deg == pytest.approx(90.0)


# --- 5. Template sample ------------------------------------------------------


def test_template_sample_water_and_chemistry_are_consistent(tmp_path: Path) -> None:
    from models import DataParser
    from workbook_template import build_input_template

    result = DataParser().parse_file(build_input_template(tmp_path / "t.xlsx"))
    by_series: dict[tuple[str, str], set[str]] = {}
    for level in result.water_levels:
        by_series.setdefault((level.series_id, level.hole_id), set()).add(level.status or "measured")
    # No series holds both a level and DRY for one well.
    assert all(len(statuses) == 1 for statuses in by_series.values()), by_series
    for reading in result.environmental_readings:
        label = reading.value_label or ""
        assert not label or (reading.unit or "") not in label
    assert any(not reading.detected for reading in result.environmental_readings)


# --- 6. Section sheet legend -------------------------------------------------


def _water_section(render_layout: str = "section_sheet", **kwargs):
    collars = [
        Collar(hole_id=h, easting=x, northing=0.0, elevation=100.0 - i, total_depth=20.0)
        for i, (h, x) in enumerate((("MW-01", 0.0), ("MW-02", 45.0), ("MW-03", 90.0)))
    ]
    liths = []
    for c in collars:
        liths += [
            Lithology(hole_id=c.hole_id, from_depth=0.0, to_depth=6.0, lithology_code="Sand"),
            Lithology(hole_id=c.hole_id, from_depth=6.0, to_depth=20.0, lithology_code="Clay"),
        ]
    water = [
        WaterLevel(hole_id=c.hole_id, depth=3.0 + i, series_id="2025-06", series_label="June 2025")
        for i, c in enumerate(collars)
    ] + [
        WaterLevel(hole_id=c.hole_id, depth=12.0, series_id="2025-06-deep", series_label="June 2025 deep")
        for c in collars[::2]
    ]
    return collars, liths, water


def test_section_sheet_water_key_joins_outside_legend(monkeypatch) -> None:
    from render_profiles import SECTION_SHEET_PROFILE

    captured = _capture_render(monkeypatch)
    collars, liths, water = _water_section()
    build_cross_section(
        collars,
        liths,
        [(c.easting, c.northing) for c in collars],
        render_layout="section_sheet",
        water_levels=water,
        show_legend=True,
        export_formats=frozenset({"png"}),
    )
    fig = captured["fig"]
    try:
        ax = captured["renderer"]._ve_main_ax
        assert not any(a.get_gid() == "water_series_key" for a in ax.artists)
        legend = ax.get_legend()
        labels = [t.get_text() for t in legend.get_texts()]
        assert any("GROUNDWATER LEVEL" in label.upper() for label in labels)
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        # Outside the plot frame (right of it), not over the geology.
        assert legend.get_window_extent(renderer).x0 >= ax.get_window_extent(renderer).x1
        # The drawn swatches carry the fence alpha (face RGBA alpha).
        swatch_alphas = {
            round(float(patch.get_facecolor()[3]), 2)
            for patch in legend.get_patches()
            if patch.get_facecolor()[3] > 0
        }
        assert swatch_alphas == {SECTION_SHEET_PROFILE.fence_alpha}, swatch_alphas
    finally:
        plt.close(fig)


# --- 7. Margins --------------------------------------------------------------


def _test_workbook(tmp_path: Path) -> Path:
    spec = importlib.util.spec_from_file_location("btw_round7", ROOT / "scripts" / "build_test_workbook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUTPUT = tmp_path / "test_workbook.xlsx"
    import sys

    argv = sys.argv
    sys.argv = ["build_test_workbook"]
    try:
        return module.main()
    finally:
        sys.argv = argv


def _workbook_section(path: Path):
    from batch_export import transect_points_from_collars
    from ingestion import ingest_workbook

    parsed, _report = ingest_workbook(path)
    holes = list(parsed.section_specs[0].hole_ids) if parsed.section_specs else [c.hole_id for c in parsed.collars]
    keep = set(holes)
    return (
        [c for c in parsed.collars if c.hole_id in keep],
        [lith for lith in parsed.lithologies if lith.hole_id in keep],
        transect_points_from_collars(parsed.collars, holes),
        [w for w in parsed.water_levels if w.hole_id in keep],
        [s for s in parsed.screen_intervals if s.hole_id in keep],
    )


def test_consulting_letter_portrait_keeps_print_margin(tmp_path: Path) -> None:
    collars, liths, points, water, screens = _workbook_section(_test_workbook(tmp_path))
    result = build_cross_section(
        collars,
        liths,
        points,
        render_layout="consulting_section",
        vertical_exaggeration=5.0,
        water_levels=water,
        screen_intervals=screens,
        consulting_title_block=ConsultingTitleBlock(section_label="Test Section A-A'", figure_number="1"),
        export_formats=frozenset({"png"}),
        export_framing=ExportFramingConfig(page_preset="letter_portrait"),
    )
    # Was left 0.21 in (the y-axis label past the margin).
    assert min(_ink_margins_in(result.png_bytes)) >= 0.25 - 1.0 / 300.0


def test_tight_export_pads_past_the_print_margin() -> None:
    collars, liths, water = _water_section()
    result = build_cross_section(
        collars,
        liths,
        [(c.easting, c.northing) for c in collars],
        render_layout="section_sheet",
        water_levels=water,
        show_legend=True,
        export_formats=frozenset({"png"}),
        export_framing=ExportFramingConfig(page_preset="tight_fence"),
    )
    assert min(_ink_margins_in(result.png_bytes)) >= 0.25


def test_section_sheet_portrait_is_balanced_vertically(tmp_path: Path) -> None:
    collars, liths, points, water, screens = _workbook_section(_test_workbook(tmp_path))
    result = build_cross_section(
        collars,
        liths,
        points,
        render_layout="section_sheet",
        vertical_exaggeration=5.0,
        water_levels=water,
        screen_intervals=screens,
        show_legend=True,
        export_formats=frozenset({"png"}),
        export_framing=ExportFramingConfig(page_preset="letter_portrait"),
    )
    image = Image.open(io.BytesIO(result.png_bytes))
    dpi = float(image.info["dpi"][0])
    rows = np.where((np.asarray(image.convert("L")) < 200).any(axis=1))[0]
    gaps = np.diff(rows)
    top_blank = rows.min() / dpi
    # The widest blank band inside the sheet (stack -> footers).
    inner_blank = gaps.max() / dpi
    # Was 3.45 in above the title and ~2 in below the stack.
    assert abs(top_blank - inner_blank) < 0.6, (top_blank, inner_blank)


# --- 8. QA page sheet --------------------------------------------------------


def test_qa_notes_page_matches_tabloid_figure_page() -> None:
    from pypdf import PdfReader

    spec, subset = build_subset("D_D")
    result = build_cross_section(
        subset.collars,
        subset.lithologies,
        [(c.easting, c.northing) for c in subset.collars],
        render_layout="consulting_section",
        vertical_exaggeration=spec.vertical_exaggeration,
        water_levels=subset.water_levels,
        consulting_title_block=spec.title_block.model_copy(update={"map_scale": "1:500"}),
        export_formats=frozenset({"pdf"}),
        export_framing=ExportFramingConfig(page_preset="tabloid_landscape"),
    )
    assert result.qa_notes, "map scale note expected"
    pages = PdfReader(io.BytesIO(result.pdf_bytes)).pages
    assert len(pages) >= 2
    sizes = {(round(float(p.mediabox.width)), round(float(p.mediabox.height))) for p in pages}
    assert sizes == {(17 * 72, 11 * 72)}


def test_text_page_size_rules() -> None:
    from report_export import LETTER_LANDSCAPE_IN, text_page_size_for

    assert text_page_size_for((17.0, 11.0)) == (17.0, 11.0)
    assert text_page_size_for((8.5, 11.0)) == (8.5, 11.0)
    assert text_page_size_for((12.1, 7.1)) == LETTER_LANDSCAPE_IN
    assert text_page_size_for((5.0, 9.0)) == (8.5, 11.0)
    assert text_page_size_for(None) == LETTER_LANDSCAPE_IN


# --- 9. Wording --------------------------------------------------------------


def test_wording_is_consistent() -> None:
    from renderer_common import format_ve_caption, parameter_legend_text
    from renderer_consulting import _strip_note_prefix

    assert format_ve_caption(5.0, auto=False) == "5× VERTICAL EXAGGERATION"
    assert format_ve_caption(5.23, auto=True) == "≈5.2× VERTICAL EXAGGERATION"
    assert format_ve_caption(5.0, auto=False, style="short") == format_ve_caption(
        5.0, auto=False, style="footer"
    )
    assert _strip_note_prefix("NOTE: masl DENOTES METRES ABOVE SEA LEVEL.") == (
        "masl DENOTES METRES ABOVE SEA LEVEL."
    )
    assert _strip_note_prefix("Notes - check datum") == "check datum"
    assert _strip_note_prefix("Notebook values are field readings") == "Notebook values are field readings"
    assert parameter_legend_text("PH CONCENTRATION (-)") == "pH"
    assert parameter_legend_text("PH") == "pH"
    assert parameter_legend_text("CHLORIDE CONCENTRATION (mg/L)") == "CHLORIDE CONCENTRATION (mg/L)"
    assert parameter_legend_text("PHOSPHORUS") == "PHOSPHORUS"


def test_section_sheet_axis_reads_masl(monkeypatch) -> None:
    captured = _capture_render(monkeypatch)
    collars, liths, water = _water_section()
    build_cross_section(
        collars,
        liths,
        [(c.easting, c.northing) for c in collars],
        render_layout="section_sheet",
        export_formats=frozenset({"png"}),
    )
    try:
        assert captured["renderer"]._ve_main_ax.get_ylabel() == "Elevation (masl)"
    finally:
        plt.close(captured["fig"])


def test_gwm_notes_do_not_repeat_note_prefix() -> None:
    spec, subset = build_subset("A_A")
    result = build_cross_section(
        subset.collars,
        subset.lithologies,
        [(c.easting, c.northing) for c in subset.collars],
        render_layout="consulting_section",
        vertical_exaggeration=spec.vertical_exaggeration,
        consulting_title_block=spec.title_block.model_copy(
            update={"notes": ("NOTE: masl DENOTES METRES ABOVE SEA LEVEL.",)}
        ),
        export_formats=frozenset({"svg"}),
        export_framing=ExportFramingConfig(svg_editable_text=True),
    )
    text = result.svg_bytes.decode("utf-8", errors="ignore")
    assert "1. masl DENOTES" in text
    assert "NOTE: masl" not in text


# --- 10. No generic chemistry band -------------------------------------------


def test_parameter_without_limits_prints_black() -> None:
    from render_profiles import chemistry_limits_set, resolved_chemistry_thresholds_for
    from render_theme import CHEMISTRY_LABEL_BLACK, CHEMISTRY_LABEL_RED, chemistry_label_color

    green, orange = resolved_chemistry_thresholds_for("Benzene", {"Chloride": (100.0, 250.0)})
    assert not chemistry_limits_set(green, orange)
    assert math.isnan(green) and math.isnan(orange)
    assert chemistry_label_color(900.0, "threshold", green_max=green, yellow_max=orange) == CHEMISTRY_LABEL_BLACK
    # A parameter with its own limits still bands.
    green, orange = resolved_chemistry_thresholds_for("Chloride", {"Chloride": (100.0, 250.0)})
    assert (green, orange) == (100.0, 250.0)
    assert chemistry_label_color(900.0, "threshold", green_max=green, yellow_max=orange) == CHEMISTRY_LABEL_RED
    # A sheet-wide pair (what Configure seeds) still applies to every parameter.
    assert resolved_chemistry_thresholds_for("Benzene", None, 0.5, 5.0) == (0.5, 5.0)
