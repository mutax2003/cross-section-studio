"""Vector rendering for borehole cross-section profiles."""

from __future__ import annotations

import io
import logging
import os
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.text import Text

from app_identity import APP_NAME, AUTHOR, COPYRIGHT_NOTICE, ORGANIZATION
from constants import HATCH_LINE_COLOR, POLYGON_EDGE_COLOR
from export_framing import (
    ExportFramingConfig,
    annotate_svg_layers,
    apply_draft_watermark,
    prepare_export_figure,
    savefig_kwargs,
)
from models import (
    ConsultingTitleBlock,
    EnvironmentalReading,
    Fault,
    InterpretationMode,
    RasterLogStrip,
    ScreenInterval,
    SectionFigureMetadata,
    Unconformity,
    VerticalGradient,
    WaterLevel,
)
from render_profiles import (
    SECTION_SHEET_PROFILE,
    CrossSectionRenderProfile,
)
from render_theme import (
    CONTACT_TICK_COLOR,
    CONTACT_TICK_WIDTH,
    EOL_BAR_COLOR,
    LABEL_COLOR,
    OVERLAP_MARKER_COLOR,
    PINCH_OUT_ALPHA,
    SKY_FILL_COLOR,
    STICK_COLOR,
    SURFACE_COLOR,
    TRACK_BORDER_COLOR,
    UNCERTAINTY_COLOR,
    export_font_rc,
)
from renderer_chart import ChartLayoutMixin
from renderer_chemistry import (
    ParameterLegendEntry,
    RendererChemistryMixin,
    _resolve_parameter_label_offsets,  # noqa: F401
)
from renderer_common import (
    LEGEND_KEY_HANDLER_MAP,
    LEGEND_LABEL_MAX_CHARS,
    LEGEND_SWATCH_HANDLEHEIGHT,
    MIN_PRINT_MARGIN_IN,
    UNLOGGED_FILL_COLOR,
    UNLOGGED_LEGEND_LABEL,
    HatchBatchedPolyCollection,
    PinchOutKey,
    RendererGeometryMixin,
    SampleTextKey,
    UncertaintyZoneKey,
    VECaptionAnnotation,
    _measure_header_extent,
    add_ve_caption,
    apply_ve_aspect,
    below_axes_anchor,
    deviated_x_steps,
    fit_label,
    format_ve_caption,
    header_overlaps,
    legend_source_keys,
    measured_vertical_exaggeration,
    page_margins_in,
    parameter_legend_keys,
    remember_legend_keys,
    restore_stacked_headers,
    settle_figure_layout,
    stack_hole_headers,
    text_layout_signature,
    water_status_legend_label,
    water_status_texts,
)
from renderer_consulting import ConsultingLayoutMixin
from renderer_section_sheet import SectionSheetLayoutMixin
from renderer_water import RendererWaterMixin, WaterSeriesLegendEntry, resolve_header_collisions
from stratigraphy import GeologicalPolygon, PolygonOverlap

if TYPE_CHECKING:
    from projection import AlongHoleTVD

# Salted element ids (with no <dc:date>) make identical inputs give identical SVG.
mpl.rcParams["svg.hashsalt"] = "cross-section-studio"

# rcParams the export reads at draw / save time (size / family of text added
# while framing, the font list an editable SVG names, hatch strokes, SVG/PDF
# font embedding, path simplification, savefig defaults). ``rc_context``
# restores rcParams on exit, but another thread writing ``plt.rcParams[...]``
# meanwhile leaked into the bytes, so the whole export runs with these keys
# pinned to the process values at import — the values every export used
# before, so normal output is byte-identical.
_EXPORT_RC_PREFIXES = ("savefig.", "pdf.", "svg.", "path.", "mathtext.", "font.", "hatch.", "text.")
_EXPORT_RC_KEYS = frozenset({"agg.path.chunksize"})
_EXPORT_RC_BASELINE: dict[str, object] = {
    key: dict.__getitem__(mpl.rcParams, key)  # no deprecation lookups
    for key in dict.keys(mpl.rcParams)
    if key.startswith(_EXPORT_RC_PREFIXES) or key in _EXPORT_RC_KEYS
}

logger = logging.getLogger(__name__)

# Wider-spacing / offset zones: pale amber wash over the fence between a
# hole pair past the thresholds, keyed in the outside legend only when drawn.
UNCERTAINTY_ZONE_ALPHA = 0.28
UNCERTAINTY_LEGEND_LABEL = "Wider hole spacing (interpretation less certain)"
# The amber tint under the fence fills was near-invisible over sand / clay
# (QA round 8): a sparse grey diagonal hatch with a dashed outline is laid
# over the fence as well (low alpha, so the lithology stays readable), and
# the legend swatch carries the same hatch and outline.
# Back-slanted: "/" is the Silty Clay / Silt hatch and read as lithology.
UNCERTAINTY_HATCH = "\\"
UNCERTAINTY_HATCH_COLOR = "#6B7280"
UNCERTAINTY_HATCH_ALPHA = 0.55
UNCERTAINTY_OUTLINE_STYLE = (0, (4.0, 2.5))
UNCERTAINTY_OVERLAY_ZORDER = 2.6
# Outside legend (section sheet / chart): the plot keeps at least this share
# of the page width; a wider legend is rebuilt narrower first.
LEGEND_PLOT_MIN_SHARE = 0.6


def uncertainty_legend_patch() -> UncertaintyZoneKey:
    """Legend key of the wider-spacing zone: the tint as it prints over the
    page (pre-blended, so the hatch and dashed outline stay at full strength)
    with back-slanted grey lines and the dashed outline of the zone overlay."""
    from matplotlib.colors import to_rgb

    tint = tuple(
        UNCERTAINTY_ZONE_ALPHA * channel + (1.0 - UNCERTAINTY_ZONE_ALPHA)
        for channel in to_rgb(UNCERTAINTY_COLOR)
    )
    return UncertaintyZoneKey(
        facecolor=tint,
        edgecolor=UNCERTAINTY_HATCH_COLOR,
        linestyle=UNCERTAINTY_OUTLINE_STYLE,
        label=fit_label(UNCERTAINTY_LEGEND_LABEL, max_chars=LEGEND_LABEL_MAX_CHARS),
    )

# Fixed-page margin fit aims this far inside the printable margin (inches),
# and a tight crop pads this much past it, so strokes / antialiasing never
# print inside the 0.25 in margin (measured 0.24 in before).
_FIT_MARGIN_SLACK_IN = 0.02


# Auto-fit bounds for borehole columns, as fractions of the hole span. All
# columns together cover at most this share of the section, so fills between
# holes stay readable on short sections (7 holes over 32 m drew 44% grey bars).
TRACK_MAX_SPAN_COVERAGE = 0.15
# Floor per column so auto-fit never shrinks a column to a hairline on long
# sections (never wider than the requested width or the spacing cap).
TRACK_MIN_SPAN_FRACTION = 0.004


def resolve_track_half_width(
    track_width_m: float,
    *,
    auto_fit: bool = True,
    x_profiles: np.ndarray | None = None,
    x_sorted: bool = False,
    spacing_fraction: float = 0.4,
    max_span_coverage: float = TRACK_MAX_SPAN_COVERAGE,
    min_span_fraction: float = TRACK_MIN_SPAN_FRACTION,
) -> float:
    """Resolve borehole column half-width in profile metres.

    ``track_width_m`` is schematic fence width on the X axis (not well diameter).
    With ``auto_fit`` on, the requested width is kept when it fits and
    otherwise narrowed so that:

    * each column is at most ``spacing_fraction`` of the closest hole spacing
      (tracks never overlap and the fill between them stays visible);
    * all columns together cover at most ``max_span_coverage`` of the hole span;
    * each column stays at least ``min_span_fraction`` of the span wide (capped
      by the request and the spacing limit) so it never vanishes.

    ``auto_fit=False`` draws exactly the requested width.
    Pass ``x_sorted=True`` when ``x_profiles`` is already ascending.
    """
    half = max(float(track_width_m) * 0.5, 1e-6)
    if not auto_fit or x_profiles is None:
        return half
    xs = np.asarray(x_profiles, dtype=float).ravel()
    xs = xs[np.isfinite(xs)]
    if xs.size < 2:
        return half
    if not x_sorted:
        xs = np.sort(xs)
    span = float(xs[-1] - xs[0])
    gaps = np.diff(xs)
    positive = gaps[gaps > 1e-6]
    if span <= 1e-6 or positive.size == 0:
        return half
    spacing_half = float(positive.min()) * float(spacing_fraction) * 0.5
    coverage_half = span * float(max_span_coverage) * 0.5 / xs.size
    floor_half = min(half, span * float(min_span_fraction) * 0.5)
    fitted = max(min(half, coverage_half), floor_half)
    return max(min(fitted, spacing_half), 1e-6)


@dataclass(frozen=True)
class _HoleContext:
    summary: pd.DataFrame
    collar_lookup: dict[str, float]
    x_by_hole: dict[str, float]
    profile_lookup: dict[str, tuple[float, float]]
    x_span: float
    track_half: float


def _export_file_metadata(fmt: str) -> dict[str, str]:
    """Attribution written into each export's own metadata block.

    matplotlib accepts different keys per backend: Dublin Core for SVG, the
    PDF document-info dictionary, and free tEXt chunks for PNG.
    """
    author = f"{AUTHOR}, {ORGANIZATION}"
    if fmt == "svg":
        # Date omitted and ids salted (rcParam svg.hashsalt) so identical
        # inputs give byte-identical SVG.
        return {"Creator": APP_NAME, "Publisher": ORGANIZATION, "Rights": COPYRIGHT_NOTICE, "Date": None}
    if fmt == "pdf":
        # CreationDate omitted so identical inputs give byte-identical PDF.
        return {"Creator": APP_NAME, "Author": author, "CreationDate": None}
    if fmt == "png":
        return {"Software": APP_NAME, "Author": author, "Copyright": COPYRIGHT_NOTICE}
    return {}


def _title_engine(figure: Figure) -> bool:
    """The figure's layout engine places the title over the hole headers (sets the top)."""
    engine = figure.get_layout_engine()
    return bool(getattr(engine, "_title", None)) and hasattr(engine, "_headers")


class CrossSectionRenderer(
    ConsultingLayoutMixin,
    SectionSheetLayoutMixin,
    ChartLayoutMixin,
    RendererWaterMixin,
    RendererChemistryMixin,
    RendererGeometryMixin,
):
    """Render geological polygons and borehole tracks to a matplotlib figure."""

    def __init__(
        self,
        vertical_exaggeration: float | None = None,
        scale_bar_length_m: float = 10.0,
        *,
        show_hatches: bool = True,
        show_legend: bool = True,
        title: str | None = None,
        disclaimer: str | None = None,
        interpretation_mode: InterpretationMode = "interpolated",
        uncertainty_spacing_m: float = 80.0,
        uncertainty_offset_m: float = 50.0,
        show_stick_logs: bool = True,
        overlap_pairs: Sequence[PolygonOverlap] | None = None,
        show_ground_surface: bool | None = None,
        interpolate_water_table: bool = False,
        figure_metadata: SectionFigureMetadata | None = None,
        faults: Sequence[Fault] = (),
        unconformities: Sequence[Unconformity] = (),
        environmental_readings: Sequence[EnvironmentalReading] = (),
        environmental_parameters: Sequence[str] = (),
        render_profile: CrossSectionRenderProfile | None = None,
        raster_log_strips: Sequence[RasterLogStrip] = (),
        consulting_title_block: ConsultingTitleBlock | None = None,
        screen_intervals: Sequence[ScreenInterval] = (),
        vertical_gradients: Sequence[VerticalGradient] = (),
        stick_up_by_hole: dict[str, float] | None = None,
        along_hole_tvd: Mapping[str, AlongHoleTVD] | None = None,
        export_framing: ExportFramingConfig | None = None,
    ) -> None:
        # ``None`` = auto (fit page): data are stored at true scale (factor 1)
        # and the axes fill their frame; the caption prints the measured VE.
        # A number draws exactly that VE: data stored at x VE, aspect locked.
        self.ve_auto = vertical_exaggeration is None
        self.chosen_vertical_exaggeration: float | None = (
            None if vertical_exaggeration is None else float(vertical_exaggeration)
        )
        # Storage factor applied to every plotted y value (see ``_plot_y``).
        self.vertical_exaggeration = (
            1.0 if vertical_exaggeration is None else float(vertical_exaggeration)
        )
        self._ve_main_ax = None
        self.scale_bar_length_m = scale_bar_length_m
        self.show_hatches = show_hatches
        self.show_legend = show_legend
        # Very long titles otherwise stretch tight-bbox exports to the title width.
        self.title = textwrap.shorten(title, width=120, placeholder="…") if title else "Borehole Cross-Section"
        self.disclaimer = disclaimer
        self.interpretation_mode = interpretation_mode
        self.uncertainty_spacing_m = uncertainty_spacing_m
        self.uncertainty_offset_m = uncertainty_offset_m
        self.show_stick_logs = show_stick_logs
        self.overlap_pairs = tuple(overlap_pairs or ())
        self.interpolate_water_table = interpolate_water_table
        self.figure_metadata = figure_metadata
        self.faults = tuple(faults)
        self.unconformities = tuple(unconformities)
        self.environmental_readings = tuple(environmental_readings)
        self.environmental_parameters = tuple(environmental_parameters)
        self.profile = render_profile or SECTION_SHEET_PROFILE
        self.raster_log_strips = tuple(raster_log_strips)
        self.consulting_title_block = consulting_title_block
        self.screen_intervals = tuple(screen_intervals)
        self.vertical_gradients = tuple(vertical_gradients)
        self.stick_up_by_hole = dict(stick_up_by_hole or {})
        # Surveyed holes: measured depth → TVD (from pipeline geometry).
        self.along_hole_tvd = dict(along_hole_tvd or {})
        self.export_framing = export_framing
        self.water_series_legend: list[WaterSeriesLegendEntry] = []
        self.lnapl_drawn = False
        self.parameter_series_legend: list[ParameterLegendEntry] = []
        self._has_pinch_out = False
        self._has_unlogged_intervals = False
        self._has_uncertainty_zone = False
        if show_ground_surface is not None:
            self.profile = self.profile.model_copy(update={"show_ground_surface": show_ground_surface})

    def render(
        self,
        polygons: list[GeologicalPolygon],
        projected_df: pd.DataFrame,
        collar_depths: dict[str, float] | None = None,
        water_levels: Sequence[WaterLevel] | None = None,
        *,
        lithology_codes: Sequence[str] | None = None,
    ) -> Figure:
        hatch_context = {
            "hatch.color": HATCH_LINE_COLOR,
            "hatch.linewidth": 0.65,
            **export_font_rc(
                self.profile.export_font_family,
                self.profile.export_font_size,
            ),
        }
        with mpl.rc_context(hatch_context):
            return self._render_figure(
                polygons,
                projected_df,
                collar_depths,
                water_levels=water_levels,
                lithology_codes=lithology_codes,
            )

    def render_to_svg(
        self,
        polygons: list[GeologicalPolygon],
        projected_df: pd.DataFrame,
        collar_depths: dict[str, float] | None = None,
        water_levels: Sequence[WaterLevel] | None = None,
        *,
        lithology_codes: Sequence[str] | None = None,
    ) -> bytes:
        figure = self.render(
            polygons,
            projected_df,
            collar_depths=collar_depths,
            water_levels=water_levels,
            lithology_codes=lithology_codes,
        )
        try:
            return self.to_svg_bytes(figure)
        finally:
            plt.close(figure)

    def render_to_png(
        self,
        polygons: list[GeologicalPolygon],
        projected_df: pd.DataFrame,
        collar_depths: dict[str, float] | None = None,
        water_levels: Sequence[WaterLevel] | None = None,
        *,
        lithology_codes: Sequence[str] | None = None,
        dpi: int = 300,
    ) -> bytes:
        figure = self.render(
            polygons,
            projected_df,
            collar_depths=collar_depths,
            water_levels=water_levels,
            lithology_codes=lithology_codes,
        )
        try:
            return self.to_png_bytes(figure, dpi=dpi)
        finally:
            plt.close(figure)

    def _render_figure(
        self,
        polygons: list[GeologicalPolygon],
        projected_df: pd.DataFrame,
        collar_depths: dict[str, float] | None,
        *,
        water_levels: Sequence[WaterLevel] | None = None,
        lithology_codes: Sequence[str] | None = None,
    ) -> Figure:
        self.water_series_legend = []
        self.lnapl_drawn = False
        self._ve_main_ax = None
        self._water_labels = []
        self._water_compact_level = 0
        self._header_labels = []
        self._has_unlogged_intervals = False
        self._has_uncertainty_zone = False
        if self.profile.layout == "section_sheet":
            render_layout = self._render_section_sheet
        elif self.profile.layout == "consulting_section":
            render_layout = self._render_consulting_section
        else:
            render_layout = self._render_chart_layout
        fig = render_layout(
            polygons,
            projected_df,
            collar_depths,
            water_levels=water_levels,
            lithology_codes=lithology_codes,
        )
        # Positions in points are only final once limits and margins are set.
        self._settle_labels(fig)
        return fig

    @staticmethod
    def _plot_layout_key(figure: Figure) -> tuple:
        """Drawn axes boxes and limits (what header and label positions follow)."""
        key = []
        for axis_ax in figure.axes:
            if not axis_ax.get_visible():
                continue
            bounds = tuple(round(float(v) * 2.0) / 2.0 for v in axis_ax.bbox.bounds)
            limits = tuple(round(float(v), 9) for v in (*axis_ax.get_xlim(), *axis_ax.get_ylim()))
            key.append((bounds, limits))
        return tuple(key)

    def _layout_headers(self, figure: Figure) -> None:
        """Header passes: nudge / one tier (shared pass), then stack what still overlaps.

        Repeated until the plot box stops moving: the stacked header height
        feeds the title layout (and the section sheet's bottom fit), which on
        an aspect-locked (exact VE / map scale) box changes the box width and
        so every hole's position on the page.
        """
        headers = [t for t in getattr(self, "_header_labels", None) or [] if t.figure is figure]
        # Called again (chart render, label settle, page fit) with nothing
        # changed since a pass that settled: the result would be the same.
        state = self._header_pass_state(figure, headers)
        if figure.__dict__.get("_css_headers_settled") == state:
            return
        figure.__dict__.pop("_css_headers_settled", None)
        for _attempt in range(4):
            # Skipped when nothing that lays the figure out changed since the
            # last draw (the previous attempt's closing settle).
            settle_figure_layout(figure)
            before = self._plot_layout_key(figure)
            margins_before = self._side_margins(figure)
            restore_stacked_headers(headers)
            resolve_header_collisions(figure, headers)
            stack_hole_headers(figure, headers, robust=_attempt >= 2)
            if str(getattr(self.profile, "layout", "")) == "section_sheet":
                # Room under the (possibly taller) header row for the distance label.
                self._fit_section_sheet_bottom(figure)
            settle_figure_layout(figure)
            self._move_water_key_outside(figure)
            self._clear_outside_legend(figure)
            # A title layout engine (chart; section sheet in depth mode) sets
            # only the plot top, from the header extents alone. Each pass
            # starts from the restored headers, so with the side / bottom
            # margins unchanged a repeat would see the same boxes and place
            # the headers the same way: the box it moved to is final.
            if self._plot_layout_key(figure) == before or (
                _title_engine(figure) and self._side_margins(figure) == margins_before
            ):
                figure._css_headers_settled = self._header_pass_state(figure, headers)
                break

    @staticmethod
    def _side_margins(figure: Figure) -> tuple:
        params = figure.subplotpars
        return (tuple(figure.get_size_inches()), params.left, params.right, params.bottom)

    def header_overlap_notes(self, figure: Figure) -> tuple[str, ...]:
        """QA note when hole-ID headers still overprint after every header fallback.

        Read after export framing (the pipeline adds it to the section notes).
        """
        headers = [
            t for t in getattr(self, "_header_labels", None) or []
            if t.figure is figure and t.get_visible() and t.get_text().strip()
        ]
        if len(headers) < 2:
            return ()
        pairs = header_overlaps(figure, headers, figure.canvas.get_renderer(), pad_pt=0.0)
        if not pairs:
            return ()
        names: list[str] = []
        for i, j in pairs:
            for index in (i, j):
                name = headers[index]._header_stack_orig[0] if hasattr(headers[index], "_header_stack_orig") else headers[index].get_text()
                name = name.split("\n")[0]
                if name not in names:
                    names.append(name)
        shown = ", ".join(names[:6]) + (f" +{len(names) - 6} more" if len(names) > 6 else "")
        return (
            f"Hole labels overlap at this scale \u2014 {len(names)} hole ID(s) print over each other "
            f"({shown}). A larger page, a lower vertical exaggeration or fewer boreholes per section "
            "separates them.",
        )

    def _header_pass_state(self, figure: Figure, headers: list) -> tuple:
        """Inputs and outcome of a settled header pass (see ``_layout_headers``)."""
        params = figure.subplotpars
        legend_ax = getattr(self, "_ve_main_ax", None)
        legend = legend_ax.get_legend() if legend_ax is not None and legend_ax.figure is figure else None
        return (
            tuple(figure.get_size_inches()),
            float(figure.dpi),
            (params.left, params.right, params.bottom, params.top),
            self._plot_layout_key(figure),
            tuple(text_layout_signature(t) for t in headers),
            tuple((id(ax), len(ax.texts), len(ax.artists)) for ax in figure.axes),
            id(legend),
            # Read by the header fallbacks (fan bounds, room above the frame).
            tuple(getattr(figure, "_css_page_margins_in", None) or ()),
            bool(getattr(figure, "_css_headers_lift_top", False)),
        )

    def _move_water_key_outside(self, figure: Figure) -> None:
        """A water series key that does not fit a short plot box joins the outside legend.

        On a long, thin exact-VE box the in-plot key (lower left) stood taller
        than the box and printed over the title. Its entries are appended to
        the outside lithology legend instead (once).
        """
        if str(getattr(self.profile, "layout", "")) == "consulting_section" or not figure.axes:
            return
        ax = getattr(self, "_ve_main_ax", None) or figure.axes[0]
        if ax.figure is not figure:
            return
        key = next((a for a in ax.artists if a.get_gid() == "water_series_key"), None)
        legend = ax.get_legend()
        if key is None or legend is None:
            return
        renderer = figure.canvas.get_renderer()
        box = key.get_window_extent(renderer)
        frame = ax.get_window_extent(renderer)
        if box.height <= 0.45 * frame.height and box.y1 <= frame.y1 + 0.5 and box.x1 <= frame.x1 + 0.5:
            return
        extras = legend_source_keys(key)
        key.remove()
        self._rebuild_outside_legend(ax, extras)

    def _clear_outside_legend(self, figure: Figure) -> None:
        """Slide the outside legend right of any text it would print over.

        The legend hangs from the plot's top-right corner; beside a short
        plot box (long section at exact VE) it reaches down past the box,
        over a distance label, threshold key or VE note wider than the box.
        """
        if str(getattr(self.profile, "layout", "")) == "consulting_section" or not figure.axes:
            return
        ax = getattr(self, "_ve_main_ax", None) or figure.axes[0]
        legend = ax.get_legend() if ax.figure is figure else None
        if legend is None:
            return
        legend.set_loc("upper left")
        legend.set_bbox_to_anchor((1.01, 1.0), transform=ax.transAxes)
        renderer = figure.canvas.get_renderer()
        box = legend.get_window_extent(renderer)
        frame = ax.get_window_extent(renderer)
        if box.y0 >= frame.y0 - 0.5:
            return  # beside the plot box only
        candidates = [ax.xaxis.label, ax.title, getattr(figure, "_supxlabel", None)]
        candidates += [t for t in ax.texts if not t.get_gid() or "chem" not in str(t.get_gid())]
        candidates += [a for a in ax.artists if a is not legend]
        candidates += list(ax.xaxis.get_ticklabels())
        candidates += [t for t in getattr(self, "_header_labels", None) or [] if t.figure is figure]
        candidates += [t for t in figure.texts if t.get_position()[1] >= 0.2]
        reach = None
        pad = 6.0 * figure.dpi / 72.0
        extents = []
        for artist in candidates:
            if artist is None or not artist.get_visible():
                continue
            if isinstance(artist, Text) and not artist.get_text().strip():
                continue
            try:
                other = artist.get_window_extent(renderer)
            except Exception:  # pragma: no cover - artist without an extent
                continue
            if other.width <= 0 or other.height <= 0:
                continue
            # Only text outside the plot frame (in-plot labels stop at the frame).
            if other.y0 >= frame.y0 - 0.5 and other.x1 <= frame.x1 + 0.5:
                continue
            extents.append(other)

        def reach_of(target) -> float | None:
            hit = None
            for other in extents:
                if (
                    other.x0 < target.x1
                    and target.x0 < other.x1 + pad
                    and other.y0 < target.y1
                    and target.y0 < other.y1
                ):
                    hit = max(hit or other.x1, other.x1)
            return hit

        reach = reach_of(box)
        if reach is None or reach + pad <= box.x0:
            return
        margins = getattr(figure, "_css_page_margins_in", None) or (MIN_PRINT_MARGIN_IN,) * 4
        page_right = figure.bbox.width - float(margins[1]) * figure.dpi
        if reach + pad + box.width <= page_right or not self._fixed_page_export():
            x_axes = (reach + pad - frame.x0) / max(frame.width, 1.0)
            legend.set_bbox_to_anchor((x_axes, 1.0), transform=ax.transAxes)
            return
        # No room to the right on a fixed page. A legend taller than a short
        # plot box (a long section at exact VE on a portrait page) stands on
        # the box's bottom edge and grows upward, where the page fit makes
        # room by lowering the plot, when nothing sits beside the box there.
        legend.set_loc("lower left")
        legend.set_bbox_to_anchor((1.01, 0.0), transform=ax.transAxes)
        if reach_of(legend.get_window_extent(renderer)) is None:
            return
        legend.set_loc("upper left")
        # Else hang it under the texts, never past the page foot: a legend
        # off the bottom edge is cut from the print.
        lowest = min(
            (other.y0 for other in extents if other.x1 > box.x0 and other.y0 < frame.y0),
            default=frame.y0,
        )
        floor = float(margins[3]) * figure.dpi
        supx = getattr(figure, "_supxlabel", None)
        for text in figure.texts:
            if text is not supx and text.get_visible() and text.get_text().strip() and text.get_position()[1] < 0.2:
                floor = max(floor, text.get_window_extent(renderer).y1 + pad)
        top = max(lowest - pad, floor + box.height)
        y_axes = (top - frame.y0) / max(frame.height, 1.0)
        legend.set_bbox_to_anchor((1.01, y_axes), transform=ax.transAxes)

    def _settle_labels(self, figure: Figure, *, fit_page: bool = False) -> None:
        """Headers, page fit, then in-plot labels, against the final plot box.

        The chemistry / water pass runs last so its hard constraints (each
        value beside its own column, clear of the neighbours) hold in the
        drawn file; if it widens the x range (non-consulting sheets) the
        headers and page fit follow and the labels are placed again.
        """
        water = getattr(self, "_water_labels", None) or []
        water_here = bool(water) and all(annotation.figure is figure for _kind, annotation, _c in water)
        sheet = str(getattr(self.profile, "layout", "")) == "section_sheet"
        for _attempt in range(3):
            self._layout_headers(figure)
            if fit_page:
                self.fit_page_content(figure)
                self._lift_headers_if_overprinting(figure)
            if not water_here:
                return
            before = self._plot_layout_key(figure)
            key_before = self._threshold_key_state(figure)
            self._resolve_water_label_collisions(figure)
            if self._threshold_key_state(figure) != key_before:
                # The label pass added or changed the "N values not printed"
                # footnote under the distance label: the sheet's bottom fit
                # (part of the page fit) stacks it clear of the footer lines.
                if fit_page:
                    self.fit_page_content(figure)
                elif sheet:
                    self._fit_section_sheet_bottom(figure)
            if self._plot_layout_key(figure) == before:
                return

    def _threshold_key_state(self, figure: Figure) -> tuple:
        """Identity and rows of the threshold key / omitted-values footnote on ``figure``."""
        key = getattr(self, "_chemistry_threshold_key", None)
        if key is None or key.figure is not figure:
            return ()
        stack = getattr(key, "_css_stack", None)
        rows = len(stack.get_children()) if stack is not None else 1
        return (id(key), rows)

    def _lift_headers_if_overprinting(self, figure: Figure) -> None:
        """Last header fallback on a fixed page: a fan taller than the room above the frame.

        When the headers still overprint after the page fit, the header pass
        may fan them out above the current room (``stack_hole_headers``) and
        the page fit then lowers the plot top to keep them on the page.
        """
        if figure.__dict__.get("_css_headers_lift_top") or not self._fixed_page_export():
            return
        headers = [
            t for t in getattr(self, "_header_labels", None) or []
            if t.figure is figure and t.get_visible() and t.get_text().strip()
        ]
        if len(headers) < 2:
            return
        settle_figure_layout(figure)
        if not header_overlaps(figure, headers, figure.canvas.get_renderer(), pad_pt=0.0):
            return
        figure._css_headers_lift_top = True
        self._layout_headers(figure)
        self.fit_page_content(figure)

    def _hole_context(self, projected_df: pd.DataFrame) -> _HoleContext:
        # Surveyed holes: the trace the column / screens / water markers follow.
        self._deviated_x = deviated_x_steps(projected_df)
        self.__dict__["_deviated_trace_cache"] = {}
        summary = projected_df.groupby("hole_id", as_index=False, sort=False).agg(
            x_profile=("x_profile", "first"),
            collar_elevation=("collar_elevation", "first"),
            bottom_elevation=("bottom_elevation", "min"),
            offset_distance=("offset_distance", "first"),
        )
        if self._deviated_x:
            # An inclined hole's headers, stick-up and labels sit at its
            # collar, not at its first interval's mid-depth x.
            xs = summary["x_profile"].to_numpy(dtype=float).copy()
            for index, (hole_id, collar) in enumerate(
                zip(summary["hole_id"].astype(str), summary["collar_elevation"], strict=True)
            ):
                if hole_id in self._deviated_x:
                    xs[index] = self._deviated_collar_x(hole_id, float(collar), float(xs[index]))
            summary["x_profile"] = xs
        summary = summary.sort_values("x_profile", kind="mergesort")
        hole_ids = summary["hole_id"].astype(str).to_numpy()
        x_values = summary["x_profile"].to_numpy(dtype=float)
        collar_values = summary["collar_elevation"].to_numpy(dtype=float)
        profile_lookup: dict[str, tuple[float, float]] = {}
        collar_lookup: dict[str, float] = {}
        x_by_hole: dict[str, float] = {}
        for hole_id, x_profile, collar in zip(hole_ids, x_values, collar_values, strict=True):
            xf = float(x_profile)
            cf = float(collar)
            profile_lookup[hole_id] = (xf, cf)
            collar_lookup[hole_id] = cf
            x_by_hole[hole_id] = xf
        x_span = float(x_values[-1] - x_values[0]) if x_values.size >= 2 else 10.0
        return _HoleContext(
            summary=summary,
            collar_lookup=collar_lookup,
            x_by_hole=x_by_hole,
            profile_lookup=profile_lookup,
            x_span=x_span,
            track_half=self._track_half_width(x_values, x_sorted=True),
        )

    def _uncertainty_y_bounds(self, hole_summary: pd.DataFrame) -> tuple[float, float]:
        ve = self.vertical_exaggeration
        collars = hole_summary["collar_elevation"].to_numpy(dtype=float)
        bottoms = hole_summary["bottom_elevation"].to_numpy(dtype=float)
        if self.profile.y_axis_mode == "depth_below_collar":
            depths_top = (collars - collars) * ve
            depths_bottom = (collars - bottoms) * ve
            return float(depths_bottom.min()), float(depths_top.max())
        return float((bottoms * ve).min()), float((collars * ve).max())

    def _plot_y(self, elevation: float, collar_rl: float) -> float:
        ve = self.vertical_exaggeration
        if self.profile.y_axis_mode == "depth_below_collar":
            return (collar_rl - elevation) * ve
        return elevation * ve

    def _plot_depths_below_collar(
        self,
        hole_id: str,
        depths: Sequence[float],
        profile_lookup: dict[str, tuple[float, float]],
    ) -> np.ndarray:
        collar_rl = float(profile_lookup[hole_id][1])
        depth_arr = np.asarray(depths, dtype=float)
        elevations = self._hole_elevations_at_depths(hole_id, depth_arr, collar_rl)
        collars = np.full(depth_arr.shape, collar_rl, dtype=float)
        return self._plot_y_values(elevations, collars)

    def _plot_y_values(self, elevations: np.ndarray, collar_rls: np.ndarray) -> np.ndarray:
        ve = self.vertical_exaggeration
        if self.profile.y_axis_mode == "depth_below_collar":
            return (collar_rls - elevations) * ve
        return elevations * ve

    def _track_half_width(
        self,
        x_profiles: np.ndarray | None = None,
        *,
        x_sorted: bool = False,
    ) -> float:
        """Half-width of each borehole column in profile metres."""
        return resolve_track_half_width(
            float(self.profile.track_width_m),
            auto_fit=bool(self.profile.auto_fit_track_width),
            x_profiles=x_profiles,
            x_sorted=x_sorted,
        )

    def _fence_plot_coords(
        self,
        coords: np.ndarray,
        ve: float,
        collar_lookup: dict[str, float],
        hole_pair: tuple[str, str] | None,
        hole_x_lookup: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Map fence polygon (x, elevation) verts into plot Y (RL or depth-below-collar).

        Depth mode measures depth below the ground line between the pair's holes:
        the collar RL is interpolated linearly in x (``hole_x_lookup``), an affine
        map, so polygons that tile in elevation still tile in depth. (A per-vertex
        left/right collar step tore pinch-out wedges off the fills they bend.)
        """
        out = np.asarray(coords, dtype=float)
        if out.size == 0:
            return out
        if self.profile.y_axis_mode != "depth_below_collar":
            scaled = out.copy()
            scaled[:, 1] *= ve
            return scaled
        left_rl = collar_lookup.get(hole_pair[0]) if hole_pair else None
        right_rl = collar_lookup.get(hole_pair[1]) if hole_pair else None
        if left_rl is None and right_rl is None:
            scaled = out.copy()
            scaled[:, 1] *= ve
            return scaled
        scaled = out.copy()
        hole_xs = hole_x_lookup or {}
        left_x = hole_xs.get(hole_pair[0]) if hole_pair else None
        right_x = hole_xs.get(hole_pair[1]) if hole_pair else None
        if (
            left_rl is not None
            and right_rl is not None
            and left_x is not None
            and right_x is not None
            and left_x != right_x
        ):
            (x0, rl0), (x1, rl1) = sorted(((left_x, left_rl), (right_x, right_rl)))
            collar_rl = np.interp(scaled[:, 0], [x0, x1], [rl0, rl1])
        elif left_rl is not None and right_rl is not None:
            mid_x = float(scaled[:, 0].mean())
            collar_rl = np.where(scaled[:, 0] <= mid_x + 1e-9, left_rl, right_rl)
        else:
            collar_rl = left_rl if left_rl is not None else right_rl
        scaled[:, 1] = (collar_rl - scaled[:, 1]) * ve
        return scaled

    def _draw_fence_polygons(
        self,
        ax,
        polygons: list[GeologicalPolygon],
        style_cache: dict,
        ve: float,
        *,
        alpha: float,
        collar_lookup: dict[str, float] | None = None,
        hole_x_lookup: dict[str, float] | None = None,
    ) -> None:
        if self.interpretation_mode not in {"interpolated", "correlation_lines"}:
            self._has_pinch_out = False
            return
        self._has_pinch_out = any(polygon.is_pinch_out for polygon in polygons)
        collars = collar_lookup or {}
        if self.interpretation_mode == "correlation_lines":
            line_groups: dict[tuple[str, str], list[np.ndarray]] = {}
            for geo_polygon in polygons:
                style = self._resolve_style(geo_polygon.lithology_code, style_cache)
                coords = np.asarray(geo_polygon.polygon.exterior.coords, dtype=float)
                line_style = "--" if geo_polygon.is_pinch_out else "-"
                style_key = (style.edge_color, line_style)
                if len(coords) >= 4:
                    top_coords = self._fence_plot_coords(
                        coords[[0, 1]], ve, collars, geo_polygon.hole_pair, hole_x_lookup
                    )
                    bottom_coords = self._fence_plot_coords(
                        coords[[3, 2]], ve, collars, geo_polygon.hole_pair, hole_x_lookup
                    )
                    line_groups.setdefault(style_key, []).extend([top_coords, bottom_coords])
                elif len(coords) == 3:
                    top_coords = self._fence_plot_coords(
                        coords[[0, 2]], ve, collars, geo_polygon.hole_pair, hole_x_lookup
                    )
                    bottom_coords = self._fence_plot_coords(
                        coords[[1, 2]], ve, collars, geo_polygon.hole_pair, hole_x_lookup
                    )
                    line_groups.setdefault(style_key, []).extend([top_coords, bottom_coords])
            for (edge_color, line_style), segments in line_groups.items():
                collection = LineCollection(
                    segments,
                    colors=edge_color,
                    linewidths=1.0,
                    linestyles=line_style,
                    zorder=3,
                )
                if self._cad_svg_layers_enabled():
                    self._set_cad_gid(collection, "fence")
                ax.add_collection(collection)
            return
        polygon_groups: dict[tuple[str, str, str, str, float], list[np.ndarray]] = {}
        for geo_polygon in polygons:
            style = self._resolve_style(geo_polygon.lithology_code, style_cache)
            coords = np.asarray(geo_polygon.polygon.exterior.coords, dtype=float)
            linestyle = "--" if geo_polygon.is_pinch_out else "-"
            # A faded wedge, never darker than the sheet's fills (the section
            # sheet draws fills and legend swatches at a lower alpha).
            patch_alpha = min(PINCH_OUT_ALPHA, alpha) if geo_polygon.is_pinch_out else alpha
            verts = self._fence_plot_coords(
                coords, ve, collars, geo_polygon.hole_pair, hole_x_lookup
            )
            style_key = (style.color, style.hatch or "", style.edge_color, linestyle, patch_alpha)
            polygon_groups.setdefault(style_key, []).append(verts)
        pinch_verts = [
            verts
            for (_c, _h, _e, linestyle, _a), verts_list in polygon_groups.items()
            if linestyle == "--"
            for verts in verts_list
        ]
        if pinch_verts:
            # The faded wedges sit on an opaque page-colour underlay: the grid
            # no longer shows through them (the legend wedge has none either).
            underlay = PolyCollection(
                pinch_verts,
                facecolors=ax.get_facecolor(),
                edgecolors="none",
                linewidths=0.0,
                zorder=1.95,
            )
            underlay.set_gid("pinch-out-underlay")
            ax.add_collection(underlay)
        for style_key, verts_list in polygon_groups.items():
            color, hatch, edge_color, linestyle, patch_alpha = style_key
            batch_kwargs = {"merge_shared_edges": True} if hatch else {}
            collection = (HatchBatchedPolyCollection if hatch else PolyCollection)(
                verts_list,
                facecolors=color,
                edgecolors=edge_color,
                linewidths=0.75,
                linestyles=linestyle,
                hatch=hatch or None,
                alpha=patch_alpha,
                **batch_kwargs,
            )
            collection.set_zorder(2)
            if self._cad_svg_layers_enabled():
                self._set_cad_gid(collection, "fence")
            ax.add_collection(collection)

    def _draw_sky_and_surface(
        self,
        ax,
        hole_summary: pd.DataFrame,
        collar_lookup: dict[str, float],
        *,
        draw_sky: bool | None = None,
    ) -> None:
        if len(hole_summary) < 2:
            return
        surface_x = hole_summary["x_profile"].to_numpy(dtype=float)
        ve = self.vertical_exaggeration
        if draw_sky is None:
            draw_sky = self.profile.show_sky_fill
        if self.profile.y_axis_mode == "depth_below_collar":
            surface_y = np.zeros(len(hole_summary), dtype=float)
        else:
            surface_y = hole_summary["collar_elevation"].to_numpy(dtype=float) * ve
        sample_count = max(20, min(100, len(surface_x) * 20))
        x_dense = np.linspace(surface_x.min(), surface_x.max(), sample_count)
        y_dense = np.interp(x_dense, surface_x, surface_y)
        # Depth mode inverts later: positive Y is into geology, so skip sky fill there.
        if draw_sky and self.profile.y_axis_mode != "depth_below_collar":
            y_top = y_dense.max() + 0.12 * max(y_dense.max() - y_dense.min(), 1.0)
            ax.fill_between(x_dense, y_dense, y_top, facecolor=SKY_FILL_COLOR, edgecolor="none", alpha=0.85, zorder=1)
        if self.profile.show_ground_surface:
            (surface_line,) = ax.plot(
                x_dense,
                y_dense,
                color=SURFACE_COLOR,
                linewidth=3.0,
                solid_capstyle="round",
                zorder=6,
            )
            if self._cad_svg_layers_enabled():
                self._set_cad_gid(surface_line, "surface")

    def _draw_track_lithology(
        self,
        ax,
        projected_df: pd.DataFrame,
        style_cache: dict,
        track_half_width: float,
        collar_lookup: dict[str, float],
        *,
        collar_arr: np.ndarray | None = None,
    ) -> None:
        # Track fills are built in renderer_common; tag new collections here for CAD SVG.
        before = len(ax.collections)
        self._draw_lithology_interval_rects(
            ax,
            projected_df,
            style_cache,
            track_half_width,
            collar_lookup,
            collar_arr=collar_arr,
        )
        if self._cad_svg_layers_enabled():
            for collection in ax.collections[before:]:
                self._set_cad_gid(collection, "tracks")

    def _draw_track_borders(
        self,
        ax,
        hole_summary: pd.DataFrame,
        collar_depths: dict[str, float],
        collar_lookup: dict[str, float],
        track_half_width: float,
    ) -> None:
        geometry = self._well_rect_geometry(
            hole_summary, collar_depths, collar_lookup, track_half_width
        )
        if geometry is None:
            return
        self._add_rect_collection(
            ax,
            geometry,
            facecolors="none",
            edgecolors=TRACK_BORDER_COLOR,
            linewidths=1.4,
            zorder=6,
        )

    def _draw_contact_ticks(
        self,
        ax,
        projected_df: pd.DataFrame,
        track_half_width: float,
        collar_lookup: dict[str, float],
        *,
        collar_arr: np.ndarray | None = None,
    ) -> None:
        if projected_df.empty:
            return
        collars = (
            collar_arr
            if collar_arr is not None
            else self._collar_values(
                projected_df["hole_id"],
                projected_df["collar_elevation"],
                collar_lookup,
            )
        )
        x_profiles = projected_df["x_profile"].to_numpy(dtype=float)
        tops = projected_df["top_elevation"].to_numpy(dtype=float)
        bottoms = projected_df["bottom_elevation"].to_numpy(dtype=float)
        hole_ids = projected_df["hole_id"].astype(str).to_numpy()
        contact_x = np.concatenate([x_profiles, x_profiles])
        contact_elev = np.concatenate([tops, bottoms])
        contact_collar = np.concatenate([collars, collars])
        contact_hole = np.concatenate([hole_ids, hole_ids])
        order = np.lexsort((contact_x, contact_elev, contact_hole))
        contact_x = contact_x[order]
        contact_elev = contact_elev[order]
        contact_collar = contact_collar[order]
        contact_hole = contact_hole[order]
        keep = np.ones(len(contact_x), dtype=bool)
        keep[1:] = (contact_hole[1:] != contact_hole[:-1]) | (contact_elev[1:] != contact_elev[:-1])
        contact_x = contact_x[keep]
        contact_elev = contact_elev[keep]
        contact_collar = contact_collar[keep]
        contact_hole = contact_hole[keep]
        ys = self._plot_y_values(contact_elev, contact_collar)
        if getattr(self, "_deviated_x", None):
            # Surveyed holes: a tick sits on the trace at its contact.
            contact_x = contact_x.copy()
            for hole_id in self._deviated_x:
                rows = np.flatnonzero(contact_hole == hole_id)
                if rows.size:
                    contact_x[rows] = self._deviated_x_of(
                        hole_id, float(contact_collar[rows[0]]), ys[rows], float(contact_x[rows[0]])
                    )
        x_left = contact_x - track_half_width
        x_right = contact_x + track_half_width
        segments = np.stack(
            [
                np.column_stack([x_left, ys]),
                np.column_stack([x_right, ys]),
            ],
            axis=1,
        )
        if len(segments):
            collection = LineCollection(
                segments,
                colors=CONTACT_TICK_COLOR,
                linewidths=CONTACT_TICK_WIDTH,
                zorder=7,
            )
            ax.add_collection(collection)

    def _draw_eol_bars(
        self,
        ax,
        hole_summary: pd.DataFrame,
        collar_depths: dict[str, float],
        collar_lookup: dict[str, float],
        track_half_width: float,
    ) -> None:
        if hole_summary.empty or not collar_depths:
            return
        hole_ids = hole_summary["hole_id"].astype(str)
        td_values = hole_ids.map(collar_depths)
        mask = td_values.notna().to_numpy()
        if not mask.any():
            return
        x_values = hole_summary.loc[mask, "x_profile"].to_numpy(dtype=float)
        collars = self._collar_values(
            hole_summary.loc[mask, "hole_id"],
            hole_summary.loc[mask, "collar_elevation"],
            collar_lookup,
        )
        td_numeric = td_values[mask].to_numpy(dtype=float)
        td_elev = collars - td_numeric
        tvd_tables = getattr(self, "along_hole_tvd", None) or {}
        if tvd_tables:
            # End-of-log bar at the true vertical depth of a surveyed hole's TD.
            for index, hole_id in enumerate(hole_ids[mask]):
                if hole_id in tvd_tables:
                    td_elev[index] = self._hole_elevation_at_depth(
                        hole_id, float(td_numeric[index]), float(collars[index])
                    )
        ys = self._plot_y_values(td_elev, collars)
        if getattr(self, "_deviated_x", None):
            # Surveyed holes: the bar sits on the trace at TD.
            x_values = np.asarray(
                [
                    self._deviated_x_at(str(hole_id), float(collar), float(y), float(x))
                    for hole_id, collar, y, x in zip(hole_ids[mask], collars, ys, x_values, strict=True)
                ],
                dtype=float,
            )
        width = track_half_width * 1.15
        segments = np.stack(
            [
                np.column_stack([x_values - width, ys]),
                np.column_stack([x_values + width, ys]),
            ],
            axis=1,
        )
        if len(segments):
            collection = LineCollection(
                segments,
                colors=EOL_BAR_COLOR,
                linewidths=2.5,
                zorder=8,
            )
            ax.add_collection(collection)

    def _draw_column_headers(
        self,
        ax,
        hole_summary: pd.DataFrame,
        collar_depths: dict[str, float],
        collar_lookup: dict[str, float],
    ) -> None:
        header_transform = ax.get_xaxis_transform()
        header_shift_axes = 0.05
        if self.profile.y_axis_mode != "elevation_rl":
            # Headers above the plot (depth mode): no tick labels between
            # them and the frame, so a small fixed gap (an axes fraction left
            # ~0.4 in of white space on a tall page).
            from matplotlib.transforms import offset_copy

            header_transform = offset_copy(header_transform, fig=ax.figure, x=0.0, y=6.0, units="points")
            header_shift_axes = 0.0
        elif not self.ve_auto:
            # Exact VE may shrink the plot box: a fixed axes fraction would put
            # the headers on the tick labels of a short box. Keep the offset
            # the frame-filling layout gives, in points.
            from matplotlib.transforms import offset_copy

            slot_h_pt = ax.get_position(original=True).height * ax.figure.get_size_inches()[1] * 72.0
            # At least clear of the x tick labels (tick + pad + 8 pt label).
            shift_pt = max(header_shift_axes * slot_h_pt, 22.0)
            down = self.profile.y_axis_mode == "elevation_rl"
            header_transform = offset_copy(
                header_transform, fig=ax.figure, x=0.0, y=-shift_pt if down else shift_pt, units="points"
            )
            header_shift_axes = 0.0
        detail = self.profile.column_header_detail
        for row in hole_summary.itertuples(index=False):
            hole_id = fit_label(str(row.hole_id))
            if detail == "id_only":
                header_text = hole_id
            else:
                collar = float(collar_lookup.get(hole_id, row.collar_elevation))
                td = collar_depths.get(hole_id)
                td_text = f"TD {td:.1f} m" if td is not None else ""
                rl_text = (
                    f"RL {collar:.1f}"
                    if self.profile.y_axis_mode == "elevation_rl"
                    else "Depth section"
                )
                header_text = f"{hole_id}\n{rl_text}\n{td_text}".strip()
            if self.profile.y_axis_mode == "elevation_rl":
                y_pos = 0.0 - header_shift_axes
                va = "top"
            else:
                y_pos = 1.0 + header_shift_axes
                va = "bottom"
            text_artist = ax.text(
                float(row.x_profile),
                y_pos,
                header_text,
                transform=header_transform,
                ha="center",
                va=va,
                fontsize=8,
                fontweight="bold",
                color=LABEL_COLOR,
                linespacing=1.15,
                clip_on=False,
                zorder=10,
            )
            # The (wrapped) ID alone: the header fallback's short label.
            text_artist._header_hole_id = hole_id
            self._header_labels.append(text_artist)
            if self._cad_svg_layers_enabled():
                self._set_cad_gid(text_artist, "headers")

    def _draw_deviated_centerlines(
        self,
        ax,
        projected_df: pd.DataFrame,
        collar_lookup: dict[str, float],
    ) -> None:
        lines: list[np.ndarray] = []
        for hole_id, group in projected_df.groupby("hole_id", sort=False):
            if str(hole_id) not in (getattr(self, "_deviated_x", None) or {}):
                continue
            collar = float(collar_lookup.get(str(hole_id), group["collar_elevation"].iloc[0]))
            trace = self._deviated_trace_plot(str(hole_id), collar)
            if trace is None or trace[0].size < 2:
                continue
            ys, xs = trace
            lines.append(np.column_stack((xs, ys)))
        if lines:
            collection = LineCollection(
                lines,
                colors=STICK_COLOR,
                linestyles=":",
                linewidths=1.0,
                alpha=0.7,
                zorder=4,
            )
            ax.add_collection(collection)

    def _draw_raster_strips(
        self,
        ax,
        x_by_hole: dict[str, float],
        collar_lookup: dict[str, float],
        track_half_width: float,
    ) -> None:
        if not self.raster_log_strips:
            return
        x_lefts: list[float] = []
        y0s: list[float] = []
        heights: list[float] = []
        labels: list[tuple[float, float, str]] = []
        width = 2.0 * track_half_width
        for strip in self.raster_log_strips:
            x = x_by_hole.get(strip.hole_id)
            if x is None:
                continue
            collar = float(collar_lookup.get(strip.hole_id, 0.0))
            top = self._plot_y(collar - strip.depth_top, collar)
            bottom = self._plot_y(collar - strip.depth_bottom, collar)
            y0, height = (top, bottom - top) if top <= bottom else (bottom, top - bottom)
            x_f = float(x)
            if strip.image_bytes:
                try:
                    import matplotlib.image as mpimg

                    arr = mpimg.imread(io.BytesIO(strip.image_bytes), format="png")
                    ax.imshow(
                        arr,
                        extent=(x_f - track_half_width, x_f + track_half_width, y0, y0 + height),
                        aspect="auto",
                        zorder=2,
                    )
                    continue
                except Exception:
                    logger.debug("Raster log image render failed for %s", strip.hole_id, exc_info=True)
            x_lefts.append(x_f - track_half_width)
            y0s.append(y0)
            heights.append(height)
            labels.append((x_f, y0 + height / 2.0, strip.label))
        if x_lefts:
            count = len(x_lefts)
            self._add_rect_collection(
                ax,
                (
                    np.asarray(x_lefts, dtype=float),
                    np.asarray(y0s, dtype=float),
                    np.full(count, width),
                    np.asarray(heights, dtype=float),
                ),
                facecolors="#E2E8F0",
                edgecolors="#64748B",
                linewidths=1.0,
                zorder=1,
                hatch="///",
                alpha=0.5,
            )
        for x_f, y_mid, label in labels:
            ax.text(
                x_f,
                y_mid,
                label,
                ha="center",
                va="center",
                fontsize=6,
                rotation=90,
                color="#475569",
            )

    def _collar_rl_for_overlay(
        self,
        x_profile: float,
        collar_lookup: dict[str, float],
        *,
        hole_pair: tuple[str, str] | None = None,
        hole_summary: pd.DataFrame | None = None,
    ) -> float:
        """Collar RL for depth-mode overlays at a section X (pair average or surface interp)."""
        if hole_pair is not None:
            left = collar_lookup.get(hole_pair[0])
            right = collar_lookup.get(hole_pair[1])
            if left is not None and right is not None:
                return (float(left) + float(right)) / 2.0
            if left is not None:
                return float(left)
            if right is not None:
                return float(right)
        if hole_summary is not None and not hole_summary.empty:
            xs = hole_summary["x_profile"].to_numpy(dtype=float)
            rls = hole_summary["collar_elevation"].to_numpy(dtype=float)
            if len(xs) == 1:
                return float(rls[0])
            order = np.argsort(xs)
            return float(np.interp(x_profile, xs[order], rls[order]))
        if collar_lookup:
            return float(np.mean(list(collar_lookup.values())))
        return 0.0

    def _draw_overlap_markers(
        self,
        ax,
        collar_lookup: dict[str, float],
        *,
        hole_summary: pd.DataFrame | None = None,
    ) -> None:
        if not self.overlap_pairs:
            return
        overlaps = self.overlap_pairs[:12]
        xs = np.fromiter((overlap.centroid_x for overlap in overlaps), dtype=float, count=len(overlaps))
        centroid_ys = np.fromiter((overlap.centroid_y for overlap in overlaps), dtype=float, count=len(overlaps))
        collar_rls = np.fromiter(
            (
                self._collar_rl_for_overlay(
                    float(overlap.centroid_x),
                    collar_lookup,
                    hole_pair=overlap.hole_pair,
                    hole_summary=hole_summary,
                )
                for overlap in overlaps
            ),
            dtype=float,
            count=len(overlaps),
        )
        ys = self._plot_y_values(centroid_ys, collar_rls)
        ax.scatter(
            xs,
            ys,
            marker="x",
            c=OVERLAP_MARKER_COLOR,
            s=64,
            linewidths=1.5,
            zorder=9,
        )

    def _draw_uncertainty_zones(
        self,
        ax,
        hole_summary: pd.DataFrame,
        collar_lookup: dict[str, float],
        *,
        polygons: Sequence[GeologicalPolygon] = (),
        hole_x_lookup: dict[str, float] | None = None,
    ) -> None:
        """Shade the fence between hole pairs spaced (or offset) past the thresholds.

        Each zone is clipped to the union of the fence polygons drawn between
        that pair, so it never shows outside the geology: as a rectangle down
        to the section's deepest TD it hung below the fence under short holes
        as a mystery band. A pair with no fence polygons (nothing drawn to
        clip to) gets the quadrilateral between its collars and its own TDs.
        """
        self._has_uncertainty_zone = False
        if len(hole_summary) < 2:
            return
        x_values = hole_summary["x_profile"].to_numpy(dtype=float)
        offset_values = hole_summary["offset_distance"].to_numpy(dtype=float)
        spacing = np.abs(np.diff(x_values))
        max_offset = np.maximum(offset_values[:-1], offset_values[1:])
        uncertain = (spacing > self.uncertainty_spacing_m) | (max_offset > self.uncertainty_offset_m)
        idxs = np.flatnonzero(uncertain)
        if idxs.size == 0:
            return
        from shapely.geometry import MultiPolygon, Polygon
        from shapely.ops import unary_union

        hole_ids = hole_summary["hole_id"].astype(str).tolist()
        collars = hole_summary["collar_elevation"].to_numpy(dtype=float)
        bottoms = hole_summary["bottom_elevation"].to_numpy(dtype=float)
        by_pair: dict[frozenset[str], list[Polygon]] = {}
        for geo_polygon in polygons:
            if geo_polygon.polygon is None or geo_polygon.polygon.is_empty:
                continue
            by_pair.setdefault(frozenset(map(str, geo_polygon.hole_pair)), []).append(
                geo_polygon.polygon
            )
        ve = self.vertical_exaggeration
        verts_list: list[np.ndarray] = []
        for idx in idxs:
            left_id, right_id = hole_ids[idx], hole_ids[idx + 1]
            pair = (left_id, right_id)
            parts = by_pair.get(frozenset(pair))
            if parts:
                union = unary_union(parts)
                if union.is_empty:
                    continue
                geoms = list(union.geoms) if isinstance(union, MultiPolygon) else [union]
                for geom in geoms:
                    if not isinstance(geom, Polygon) or geom.is_empty:
                        continue
                    coords = np.asarray(geom.exterior.coords, dtype=float)
                    verts_list.append(
                        self._fence_plot_coords(coords, ve, collar_lookup, pair, hole_x_lookup)
                    )
                continue
            # No fence drawn for this pair: the strip between its ground line
            # and its own hole bases (never deeper than the pair's logs).
            x0, x1 = float(x_values[idx]), float(x_values[idx + 1])
            coords = np.array(
                [
                    (x0, collars[idx]),
                    (x1, collars[idx + 1]),
                    (x1, bottoms[idx + 1]),
                    (x0, bottoms[idx]),
                ],
                dtype=float,
            )
            verts_list.append(self._fence_plot_coords(coords, ve, collar_lookup, pair, hole_x_lookup))
        if not verts_list:
            return
        collection = PolyCollection(
            verts_list,
            facecolors=UNCERTAINTY_COLOR,
            edgecolors="none",
            linewidths=0.0,
            alpha=UNCERTAINTY_ZONE_ALPHA,
        )
        collection.set_zorder(0)
        collection.set_gid("uncertainty-zone")
        ax.add_collection(collection)
        # Over the fence fills (zorder 2), under the tracks: the hatch and
        # dashed outline stay legible whatever lithology lies beneath.
        overlay = PolyCollection(
            verts_list,
            facecolors="none",
            edgecolors=UNCERTAINTY_HATCH_COLOR,
            linewidths=0.9,
            linestyles=[UNCERTAINTY_OUTLINE_STYLE],
            hatch=UNCERTAINTY_HATCH,
            alpha=UNCERTAINTY_HATCH_ALPHA,
        )
        overlay.set_zorder(UNCERTAINTY_OVERLAY_ZORDER)
        overlay.set_gid("uncertainty-zone-hatch")
        ax.add_collection(overlay)
        self._has_uncertainty_zone = True

    def _draw_faults(
        self,
        ax,
        collar_lookup: dict[str, float],
        *,
        hole_summary: pd.DataFrame | None = None,
    ) -> None:
        if not self.faults:
            return
        segments: list[np.ndarray] = []
        for fault in self.faults:
            if len(fault.trace_points) < 2:
                continue
            xs, ys = zip(*fault.trace_points)
            collar_rls = np.fromiter(
                (
                    self._collar_rl_for_overlay(float(x), collar_lookup, hole_summary=hole_summary)
                    for x in xs
                ),
                dtype=float,
                count=len(xs),
            )
            plot_ys = self._plot_y_values(np.asarray(ys, dtype=float), collar_rls)
            segments.append(np.column_stack([xs, plot_ys]))
        if segments:
            collection = LineCollection(
                segments,
                colors="#B91C1C",
                linewidths=2.0,
                linestyles="-.",
                zorder=8,
            )
            ax.add_collection(collection)

    def _draw_unconformities(
        self,
        ax,
        collar_lookup: dict[str, float],
        *,
        hole_summary: pd.DataFrame | None = None,
    ) -> None:
        if not self.unconformities:
            return
        segments: list[np.ndarray] = []
        for surface in self.unconformities:
            if len(surface.elevation_profile) < 2:
                continue
            xs, ys = zip(*surface.elevation_profile)
            collar_rls = np.fromiter(
                (
                    self._collar_rl_for_overlay(float(x), collar_lookup, hole_summary=hole_summary)
                    for x in xs
                ),
                dtype=float,
                count=len(xs),
            )
            plot_ys = self._plot_y_values(np.asarray(ys, dtype=float), collar_rls)
            segments.append(np.column_stack([xs, plot_ys]))
        if segments:
            collection = LineCollection(
                segments,
                colors="#7C3AED",
                linewidths=1.8,
                linestyles=":",
                zorder=7,
            )
            ax.add_collection(collection)

    def _draw_legend(
        self,
        ax,
        style_cache: dict,
        lithology_codes: list[str],
        polygons: list[GeologicalPolygon],
        *,
        swatch_alpha: float | None = None,
    ) -> None:
        """Outside lithology legend.

        ``swatch_alpha`` is the fence fill's alpha: the swatches are drawn
        with it so they match the (paler) translucent fence, not an opaque
        colour the section never shows.
        """
        legend_handles = []
        for code in lithology_codes:
            style = self._resolve_style(code, style_cache)
            legend_handles.append(
                Patch(
                    facecolor=style.color,
                    edgecolor=POLYGON_EDGE_COLOR,
                    hatch=style.hatch or None,
                    linewidth=0.75,
                    alpha=swatch_alpha,
                    # Long codes wrap to two lines, then ellipsize (a 120-
                    # character code pushed the legend past the page edge).
                    label=fit_label(code, max_chars=LEGEND_LABEL_MAX_CHARS),
                )
            )
        if getattr(self, "_has_unlogged_intervals", False):
            legend_handles.append(
                Patch(
                    facecolor=UNLOGGED_FILL_COLOR,
                    edgecolor=POLYGON_EDGE_COLOR,
                    linewidth=0.75,
                    label=UNLOGGED_LEGEND_LABEL,
                )
            )
        labels = [handle.get_label() for handle in legend_handles]
        if self._has_pinch_out:
            # Keyed by a faded, dashed wedge like the ones drawn.
            legend_handles.append(PinchOutKey())
            labels.append("Inferred pinch-out")
        if getattr(self, "_has_uncertainty_zone", False):
            # Only when a wider-spacing zone is shaded: the pale amber wash
            # was drawn unexplained.
            legend_handles.append(uncertainty_legend_patch())
            labels.append(legend_handles[-1].get_label())
        legend = ax.legend(
            handles=legend_handles,
            labels=labels,
            title="Lithology",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            framealpha=0.95,
            edgecolor="#CBD5E1",
            fontsize=8,
            title_fontsize=9,
            ncol=max(1, self.profile.legend_ncol) if len(legend_handles) > 1 else 1,
            handler_map=LEGEND_KEY_HANDLER_MAP,
            handleheight=LEGEND_SWATCH_HANDLEHEIGHT,
        )
        remember_legend_keys(legend, legend_handles, labels)
        if self._cad_svg_layers_enabled():
            self._set_cad_gid(legend, "legend")

    def _water_series_measured(
        self,
        water_levels: Sequence[WaterLevel] | None,
        on_section: set[str],
    ) -> dict[str, int]:
        """Measured (not dry / NM) readings per water series on this section."""
        from hydro_metrics import water_status

        measured: dict[str, int] = {}
        for level in water_levels or ():
            if str(level.hole_id) not in on_section or water_status(level) in {"dry", "nm"}:
                continue
            series_id = level.series_id or "default"
            measured[series_id] = measured.get(series_id, 0) + 1
        return measured

    def _water_line_drawn(self, measured_count: int) -> bool:
        """True when the water pass joins a series' markers with a line."""
        interpolate = bool(self.interpolate_water_table or self.profile.interpolate_water_table_default)
        return interpolate and measured_count >= 2

    def _legend_extra_keys(self, ax, *, ground_surface: bool = False) -> list[tuple[object, str]]:
        """Chemistry, ground-surface and NM / DRY keys for a matplotlib legend.

        Only what the figure draws: a chemistry key per labelled parameter
        (three G / O / R dots in threshold mode, else the drawn marker or a
        sample value), the ground-surface line when drawn, and the grey
        NM / DRY note text when any hole carries one.
        """
        from matplotlib.lines import Line2D

        from render_theme import CONSULTING_NM_COLOR

        keys: list[tuple[object, str]] = []
        if self.profile.show_parameter_markers or self.profile.show_parameter_labels:
            keys += parameter_legend_keys(
                self.parameter_series_legend,
                draw_markers=self.profile.parameter_draw_markers,
                sentence_case=True,
            )
        if ground_surface:
            keys.append((Line2D([], [], color=SURFACE_COLOR, linewidth=3.0), "Ground surface"))
        status = water_status_texts(self, ax.figure)
        if status:
            keys.append(
                (
                    SampleTextKey(" / ".join(status), CONSULTING_NM_COLOR, fontweight="bold"),
                    water_status_legend_label(status),
                )
            )
        return keys

    def _rebuild_outside_legend(
        self,
        ax,
        extras: list[tuple[object, str]],
        *,
        title: str | None = None,
        ncol: int | None = None,
        fontsize: float = 8.0,
        force: bool = False,
    ):
        """Append ``extras`` to the outside lithology legend (rebuilt once).

        ``ncol`` / ``fontsize`` (with ``force``) rebuild it narrower when the
        page cannot hold it. Returns the rebuilt legend (or None).
        """
        legend = ax.get_legend()
        if legend is None or (not extras and not force):
            return None
        # Rebuild from the source proxies, never ``legend.legend_handles``
        # (drawn artists): a second rebuild lost the three-dot threshold key
        # and the NM sample text.
        keys = legend_source_keys(legend) + list(extras)
        handles = [handle for handle, _label in keys]
        labels = [label for _handle, label in keys]
        gid = legend.get_gid()
        rebuilt = ax.legend(
            handles=handles,
            labels=labels,
            title=title or legend.get_title().get_text(),
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            framealpha=0.95,
            edgecolor="#CBD5E1",
            fontsize=fontsize,
            title_fontsize=max(fontsize + 1.0, 7.0),
            ncol=ncol if ncol is not None else legend._ncols,
            handler_map=LEGEND_KEY_HANDLER_MAP,
            handleheight=LEGEND_SWATCH_HANDLEHEIGHT,
        )
        remember_legend_keys(rebuilt, handles, labels)
        if gid:
            rebuilt.set_gid(gid)
        return rebuilt

    def fit_right_margin_for_legend(self, figure: Figure) -> None:
        """Widen the right margin so an outside lithology legend fits the page.

        Section sheets and the chart anchor the legend just right of the plot
        with a fixed margin; long names in two columns ran off the page
        ("Sandy Clay Lo…"). Measure the legend and move the plot's right edge
        in until it fits (no-op without an outside legend, or for consulting
        sheets, whose legend lives in the title block).
        """
        if str(getattr(self.profile, "layout", "")) == "consulting_section" or not figure.axes:
            return
        ax = figure.axes[0]
        legend = ax.get_legend()
        if legend is None:
            return
        # The layout's own right edge (before any legend fit), for giving
        # room back when a narrower legend frees it.
        if figure.__dict__.get("_css_legend_stock_right") is None:
            figure._css_legend_stock_right = float(figure.subplotpars.right)
        settle_figure_layout(figure)
        renderer = figure.canvas.get_renderer()
        page_w = figure.bbox.width
        # Legend starts 1% of the axes width right of the plot; keep a 1.5%
        # page gap, or the printable margin on a fixed page.
        gap = 0.015
        if self._fixed_page_export():
            gap = max(gap, float(page_margins_in(self.export_framing)[1]) * figure.dpi / page_w)
        # The plot keeps at least this share of the page width: a wide
        # two-column legend left it 30-45% of a tabloid sheet (QA round 8).
        floor = LEGEND_PLOT_MIN_SHARE

        def right_for(current) -> float:
            legend_w = current.get_window_extent(renderer).width / page_w
            return 1.0 - gap - legend_w - 0.01 * ax.get_position().width

        right = right_for(legend)
        if right < floor:
            # Wider than the page can give it beside the narrowest plot
            # (two columns of long names on a portrait page): one column,
            # then smaller text, before it is let past the margin.
            for ncol, size in ((2, 7.0), (1, 8.0), (1, 7.0), (1, 6.5)):
                if ncol >= legend._ncols and size >= legend.get_texts()[0].get_fontsize() - 1e-6:
                    continue
                rebuilt = self._rebuild_outside_legend(ax, [], ncol=ncol, fontsize=size, force=True)
                if rebuilt is None:
                    break
                legend = rebuilt
                right = right_for(legend)
                if right >= floor:
                    break
        right = min(figure.subplotpars.right, max(floor, right))
        if right < figure.subplotpars.right - 1e-3:
            figure.subplots_adjust(right=right)
        self._narrow_legend_beside_drawn_plot(figure, ax, right_for)

    def _narrow_legend_beside_drawn_plot(self, figure: Figure, ax, right_for) -> None:
        """Keep the drawn plot at least ``LEGEND_PLOT_MIN_SHARE`` of plot + legend.

        An exact-VE box is height-bound: it never grows into the room the
        page gives it, so a wide two-column legend beside it (long series
        names) left the plot 30-50% of a tight sheet (QA round 8). The legend
        is rebuilt in one column (then smaller text) until it fits that
        share, and the plot's right edge moves back out into the freed room.
        """
        legend = ax.get_legend()
        if legend is None:
            return
        stock_right = figure.__dict__.get("_css_legend_stock_right")
        renderer = figure.canvas.get_renderer()

        def fits(current) -> bool:
            ax.apply_aspect()
            plot_w = ax.get_window_extent(renderer).width
            legend_w = current.get_window_extent(renderer).width
            return legend_w <= plot_w * (1.0 - LEGEND_PLOT_MIN_SHARE) / LEGEND_PLOT_MIN_SHARE

        settle_figure_layout(figure)
        if fits(legend):
            return
        base_size = float(legend.get_texts()[0].get_fontsize()) if legend.get_texts() else 8.0
        for ncol, size in ((1, base_size), (1, 7.0), (1, 6.5)):
            if ncol >= legend._ncols and size >= float(legend.get_texts()[0].get_fontsize()) - 1e-6:
                continue
            rebuilt = self._rebuild_outside_legend(ax, [], ncol=ncol, fontsize=size, force=True)
            if rebuilt is None:
                break
            legend = rebuilt
            # The narrower legend frees room on the right: give it back to
            # the plot (never past the layout's own right edge).
            limit = stock_right if stock_right is not None else figure.subplotpars.right
            target = min(limit, right_for(legend))
            if target > figure.subplotpars.right + 1e-3:
                figure.subplots_adjust(right=target)
            settle_figure_layout(figure)
            if fits(legend):
                break

    def _apply_ve_aspect(self, ax, twins: Sequence = (), *, anchor: str = "C") -> None:
        """Exact VE: lock the axes box to the chosen VE; auto: fill the frame."""
        self._ve_main_ax = ax
        ax.figure._css_ve_axes = ax
        ax.figure._css_ve_storage = float(self.vertical_exaggeration)
        apply_ve_aspect(ax, exact=not self.ve_auto, twins=twins, anchor=anchor)

    def current_vertical_exaggeration(self) -> float | None:
        """VE the figure prints: the chosen value, or the measured one in auto mode."""
        if not self.ve_auto:
            return self.chosen_vertical_exaggeration
        ax = self._ve_main_ax
        if ax is None or ax.figure is None:
            return None
        return measured_vertical_exaggeration(ax, self.vertical_exaggeration)

    def _ve_caption_kwargs(self, style: str, template: str = "{ve}") -> dict[str, object]:
        return {
            "main_ax": self._ve_main_ax,
            "storage_ve": self.vertical_exaggeration,
            "auto": self.ve_auto,
            "chosen_ve": self.chosen_vertical_exaggeration,
            "style": style,
            "template": template,
        }

    def _draw_ve_annotation(self, ax) -> None:
        """VE note under the plot (opposite the scale bar), not over the data.

        Inside the plot it sat on the geology and a water marker on a long
        thin 1x section. Measured VE is filled in at draw time.
        """
        if self._ve_main_ax is None:
            self._ve_main_ax = ax
        right = not self._scale_bar_on_right()
        note = VECaptionAnnotation(
            (1.0 if right else 0.0, 0.0),
            xycoords=below_axes_anchor(ax),
            xytext=(0.0, -self._BELOW_AXES_ROW_GAP_PT),
            textcoords="offset points",
            ha="right" if right else "left",
            va="top",
            fontsize=9,
            fontweight="bold",
            color=LABEL_COLOR,
            zorder=10,
            annotation_clip=False,
            **self._ve_caption_kwargs("short"),
        )
        note.set_gid("ve-note")
        ax._add_text(note)

    def _draw_footers(self, fig: Figure) -> None:
        if self.disclaimer:
            fig.text(0.5, 0.008, self.disclaimer, ha="center", va="bottom", fontsize=8, color="#64748B", style="italic")
        footer_y = 0.03 if self.disclaimer else 0.008
        metadata_lines = self._metadata_footer_lines(ve_text="{ve}")
        if metadata_lines and self.profile.title_block:
            line = " | ".join(metadata_lines)
            if "{ve}" in line and self._ve_main_ax is not None:
                # The VE item is measured from the final axes box at draw time.
                add_ve_caption(
                    fig,
                    0.5,
                    footer_y,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#475569",
                    **self._ve_caption_kwargs("footer", template=line),
                )
            else:
                line = line.replace("{ve}", self._ve_footer_text())
                fig.text(0.5, footer_y, line, ha="center", va="bottom", fontsize=7, color="#475569")
            footer_y += 0.022
        if self.overlap_pairs and self.profile.show_overlap_footer:
            fig.text(
                0.5,
                footer_y,
                f"Polygon overlap markers ({len(self.overlap_pairs)}): review layer correlation.",
                ha="center",
                va="bottom",
                fontsize=7,
                color=OVERLAP_MARKER_COLOR,
            )

    def _ve_footer_text(self) -> str:
        return format_ve_caption(
            self.current_vertical_exaggeration(), auto=self.ve_auto, style="footer"
        )

    def _metadata_footer_lines(self, ve_text: str | None = None) -> list[str]:
        """Footer items; the VE item is the drawn VE (``ve_text`` overrides it)."""
        if self.figure_metadata is None:
            return []
        meta = self.figure_metadata
        lines: list[str] = []
        if meta.coordinate_reference:
            lines.append(f"CRS: {meta.coordinate_reference}")
        if meta.elevation_datum:
            lines.append(f"Datum: {meta.elevation_datum}")
        lines.append(ve_text if ve_text is not None else self._ve_footer_text())
        if meta.transect_azimuth_deg is not None:
            lines.append(f"Azimuth: {meta.transect_azimuth_deg:.0f}°")
        lines.append(f"Holes: {meta.hole_count}")
        if meta.uses_placeholder_elevation:
            lines.append("WARNING: placeholder collar elevation")
        return lines

    def _cad_svg_layers_enabled(self) -> bool:
        return (
            self.export_framing is not None
            and bool(self.export_framing.cad_svg_layers)
        )

    @staticmethod
    def _set_cad_gid(artist, layer_id: str) -> None:
        """Tag a matplotlib artist so SVG export emits id= for Inkscape layer promotion."""
        if artist is None:
            return
        setter = getattr(artist, "set_gid", None)
        if callable(setter):
            setter(layer_id)

    def _savefig_kwargs(self, fig: Figure | None = None) -> dict[str, object]:
        """Export framing controls page crop; consulting default is fixed letter page."""
        kwargs = savefig_kwargs(
            self.export_framing,
            layout=str(getattr(self.profile, "layout", "")),
        )
        if kwargs.get("bbox_inches") == "tight":
            # A tight crop keeps the same least printable margin as a page.
            kwargs["pad_inches"] = max(
                float(kwargs.get("pad_inches") or 0.0), MIN_PRINT_MARGIN_IN + _FIT_MARGIN_SLACK_IN
            )
            if fig is not None:
                crop = self._tight_crop_bbox_in(fig)
                if crop is not None:
                    # An explicit crop box is not padded by savefig: pad it here.
                    kwargs["bbox_inches"] = crop.padded(float(kwargs["pad_inches"]))
        return kwargs

    def _tight_crop_bbox_in(self, fig: Figure):
        """The tight crop box (inches) including the framed hole headers.

        ``bbox_inches="tight"`` measures text without its frame, so a framed
        header at the page edge printed 0.02-0.04 in into the margin (QA
        round 8: quick preview left margin 0.23 in). None when nothing framed
        reaches past the stock tight box.
        """
        from matplotlib.transforms import Bbox

        headers = [
            t for t in getattr(self, "_header_labels", None) or []
            if t.figure is fig and t.get_visible() and t.get_text().strip() and t.get_bbox_patch() is not None
        ]
        if not headers:
            return None
        settle_figure_layout(fig)
        renderer = fig.canvas.get_renderer()
        tight = fig.get_tightbbox(renderer)
        if tight is None:
            return None
        # ``Figure.get_tightbbox`` is in inches; the header frames are display px.
        dpi = float(fig.dpi)
        boxes = [
            Bbox.from_extents(b.x0 / dpi, b.y0 / dpi, b.x1 / dpi, b.y1 / dpi)
            for b in (_measure_header_extent(t, renderer) for t in headers)
        ]
        union = Bbox.union([tight, *boxes])
        if all(abs(getattr(union, side) - getattr(tight, side)) < 1e-3 for side in ("x0", "x1", "y0", "y1")):
            return None
        return union

    def _prepare_export_figure(self, figure: Figure) -> Figure:
        prepare_export_figure(
            figure,
            self.export_framing,
            layout=str(getattr(self.profile, "layout", "")),
        )
        if (
            str(getattr(self.profile, "layout", "")) == "consulting_section"
            and figure.get_size_inches()[0] < 10.0
        ):
            # Portrait pages are narrower: give the twin RL axis label room.
            figure.subplots_adjust(right=0.925)
        # Consulting sheets: keep RL axis labels on the page (no-op otherwise).
        self.fit_consulting_page_margins(figure)
        # Section sheet / chart: keep the outside lithology legend on the page.
        self.fit_right_margin_for_legend(figure)
        if tuple(figure.get_size_inches()) == tuple(getattr(figure, "_css_prepared_size", ())):
            self.refit_consulting_fitted_text(figure)
            return figure  # already prepared at this size: label passes are current
        figure._css_prepared_size = tuple(figure.get_size_inches())
        # Page sizing moves every artist; redo label placement for the new
        # page: headers, printable margins, then the in-plot labels last.
        self._settle_labels(figure, fit_page=True)
        # Fitted title-block / band / notes text was measured at the render size.
        self.refit_consulting_fitted_text(figure)
        # A locked VE that leaves the section a thin strip gets its caption.
        self.annotate_locked_ve_strip(figure)
        return figure

    def _fixed_page_export(self) -> bool:
        """True when the export prints on a fixed page (not a tight crop)."""
        layout = str(getattr(self.profile, "layout", ""))
        if layout == "consulting_section":
            return True
        if self.export_framing is None:
            return False
        return self.export_framing.effective_page_preset(layout) != "tight_fence"

    def _place_page_footers(self, figure: Figure, margins: tuple[float, float, float, float]) -> float:
        """Stack the figure-level footer lines up from the bottom margin.

        Returns the display y (px) the footers reach (the bottom margin when
        there are none). Lines wider than the printable width wrap (plain
        text) or shrink (measured VE captions).
        """
        renderer = figure.canvas.get_renderer()
        dpi = figure.dpi
        left_px, right_px = margins[0] * dpi, figure.bbox.width - margins[1] * dpi
        bottom_px = margins[3] * dpi
        supx = getattr(figure, "_supxlabel", None)
        order = figure.__dict__.get("_css_footer_order")
        if order is None:
            footers = [
                t for t in figure.texts
                if t is not supx and t.get_visible() and t.get_text().strip()
                and t.get_position()[1] < 0.2
            ]
            order = figure._css_footer_order = sorted(footers, key=lambda t: t.get_position()[1])
            for text in order:
                text._css_footer_orig = (text.get_text(), float(text.get_fontsize()))
        cursor = bottom_px
        gap = 2.0 * dpi / 72.0
        for text in order:
            if text.figure is not figure or not text.get_visible():
                continue
            label, size = text._css_footer_orig
            avail = right_px - left_px
            if type(text).__name__ == "Text":
                text.set_text(label)
                text.set_fontsize(size)
                box = text.get_window_extent(renderer)
                if box.width > avail and len(label) > 20:
                    chars = max(20, int(len(label) * avail / box.width) - 2)
                    text.set_text("\n".join(textwrap.wrap(label, chars)))
            else:
                text.set_fontsize(size)
                for _ in range(6):
                    box = text.get_window_extent(renderer)
                    if box.width <= avail or text.get_fontsize() <= 5.0:
                        break
                    text.set_fontsize(max(5.0, text.get_fontsize() * avail / box.width))
            text.set_verticalalignment("bottom")
            box = text.get_window_extent(renderer)
            text.set_y(text.get_position()[1] + (cursor - box.y0) / figure.bbox.height)
            cursor = text.get_window_extent(renderer).y1 + gap
        return cursor - gap if cursor > bottom_px else bottom_px

    def fit_page_content(self, figure: Figure, *, max_passes: int = 8) -> bool:
        """Pull the plot frame in until all content keeps the printable margins.

        Fixed export pages only (consulting sheets and letter / tabloid
        presets): the hole headers, axis labels, legend, title and footers
        stay at least ``MIN_PRINT_MARGIN_IN`` (or the export framing margin)
        from every page edge. Returns True when a margin was moved.
        """
        if not self._fixed_page_export():
            return False
        margins = page_margins_in(self.export_framing)
        figure._css_page_margins_in = margins
        moved = False
        size = tuple(figure.get_size_inches())
        stored = figure.__dict__.get("_css_fit_base")
        if stored is None or stored[0] != size:
            params = figure.subplotpars
            stored = figure._css_fit_base = (size, (params.left, params.right, params.top, params.bottom))
        base = stored[1]
        sheet = str(getattr(self.profile, "layout", "")) == "section_sheet"
        engine_owns_top = _title_engine(figure)
        for _pass in range(max_passes):
            footer_top = self._place_page_footers(figure, margins)
            if sheet:
                self._fit_section_sheet_bottom(figure)
            engine = figure.get_layout_engine()
            if engine is not None and hasattr(engine, "execute"):
                engine.execute(figure)
            settle_figure_layout(figure)
            renderer = figure.canvas.get_renderer()
            boxes = []
            supx = getattr(figure, "_supxlabel", None)
            for axis_ax in figure.axes:
                if axis_ax.get_visible():
                    box = axis_ax.get_tightbbox(renderer)
                    if box is not None and box.width > 0:
                        boxes.append(box)
            for text in [supx, getattr(figure, "_suptitle", None)]:
                if text is not None and text.get_visible() and text.get_text().strip():
                    boxes.append(text.get_window_extent(renderer))
            if not boxes:
                return moved
            from matplotlib.transforms import Bbox

            content = Bbox.union(boxes)
            dpi = figure.dpi
            width, height = figure.bbox.width, figure.bbox.height
            # Aim a hair inside the margin: an aspect-locked box re-centres
            # after every move, so the overflow only decays pass by pass and
            # the last pass left content ~0.01-0.04 in past it (QA round 7:
            # letter portrait left margin 0.21 in).
            slack = _FIT_MARGIN_SLACK_IN * dpi
            floor = max(margins[3] * dpi, footer_top + 3.0 * dpi / 72.0)
            over_l = margins[0] * dpi + slack - content.x0
            over_r = content.x1 - (width - margins[1] * dpi - slack)
            over_t = content.y1 - (height - margins[2] * dpi - slack)
            over_b = floor - content.y0
            if os.environ.get("CSS_DEBUG_FIT"):
                print("fit pass", _pass, [round(v / dpi, 3) for v in (over_l, over_r, over_t, over_b)])
            params = figure.subplotpars
            b_left, b_right, b_top, b_bottom = base
            # Move in by any overflow; give back room (never past the
            # stock frame) where an earlier pass over-tightened, e.g. after
            # vertical headers turned into a lower tier layout.
            left = max(params.left + over_l / width, b_left)
            right = min(params.right - over_r / width, b_right)
            if engine_owns_top:
                # The title layout sets the top on every draw (from the
                # title and header heights); resetting it here only made the
                # next draw move it back, and every pass re-ran the headers.
                top = params.top
            else:
                top = min(params.top - over_t / height, b_top)
            if sheet:
                bottom = params.bottom + max(over_b, 0.0) / height
            else:
                bottom = max(params.bottom + over_b / height, b_bottom)
            right = max(right, left + 0.2)
            top = max(top, bottom + 0.2)
            change = max(
                abs(left - params.left) * width,
                abs(right - params.right) * width,
                abs(top - params.top) * height,
                abs(bottom - params.bottom) * height,
            )
            if change <= 0.5:
                break
            figure.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
            moved = True
            self._after_margin_change(figure)
            self._layout_headers(figure)
        return moved

    def _after_margin_change(self, figure: Figure) -> None:
        """Layout-specific refits after ``subplots_adjust`` (no-op by default)."""
        layout = str(getattr(self.profile, "layout", ""))
        if layout == "consulting_section":
            self._keep_band_heights(figure)
            self._refresh_last_hole_label_room(figure)
            self._reserve_threshold_key_band(figure)
            self._suppress_header_tick_collisions(figure)

    def _export_dpi(self, default: int = 300) -> int:
        if self.export_framing is not None:
            return int(self.export_framing.export_dpi)
        return default

    def _export_rc(self) -> dict[str, object]:
        """rcParams pinned for the export path (see ``_EXPORT_RC_BASELINE``)."""
        return dict(_EXPORT_RC_BASELINE)

    def _savefig_format(
        self,
        fig: Figure,
        *,
        fmt: str,
        dpi: int | None = None,
    ) -> bytes:
        buffer = io.BytesIO()
        kwargs: dict[str, object] = {
            "format": fmt,
            "facecolor": fig.get_facecolor(),
            **self._savefig_kwargs(fig),
        }
        kwargs["metadata"] = _export_file_metadata(fmt)
        if fmt == "png":
            kwargs["dpi"] = self._capped_png_dpi(fig, dpi if dpi is not None else self._export_dpi())
        # SVG text: outlines by default (identical on every machine); editable
        # <text> on request so labels can be nudged in Illustrator / Inkscape.
        editable = fmt == "svg" and bool(getattr(self.export_framing, "svg_editable_text", False))
        with mpl.rc_context({**self._export_rc(), "svg.fonttype": "none" if editable else "path"}):
            fig.savefig(buffer, **kwargs)
        buffer.seek(0)
        payload = buffer.getvalue()
        # TODO(cad-svg-layers): tag water artists with gid="water" in renderer_water.py
        # when ExportFramingConfig.cad_svg_layers is on (V1 promotes matching groups only).
        if fmt == "svg" and self._cad_svg_layers_enabled():
            return annotate_svg_layers(payload)
        return payload

    # Largest PNG written (pixels); a tall tight-cropped figure at 600 dpi
    # reached 275-524 Mpx and 1.2-1.7 GB of memory.
    MAX_PNG_PIXELS = 100_000_000

    def _capped_png_dpi(self, fig: Figure, dpi: float) -> float:
        """``dpi``, lowered so the PNG stays within ``MAX_PNG_PIXELS`` (QA note when lowered)."""
        kwargs = self._savefig_kwargs()
        width_in, height_in = (float(v) for v in fig.get_size_inches())
        if kwargs.get("bbox_inches") == "tight":
            try:
                box = fig.get_tightbbox(fig.canvas.get_renderer())
                pad = float(kwargs.get("pad_inches") or 0.0)
                width_in, height_in = box.width + 2.0 * pad, box.height + 2.0 * pad
            except Exception:  # pragma: no cover - keep the figure size estimate
                pass
        area = max(width_in * height_in, 1e-6)
        if area * float(dpi) ** 2 <= self.MAX_PNG_PIXELS:
            return dpi
        capped = max(1, int((self.MAX_PNG_PIXELS / area) ** 0.5))
        note = (
            f"PNG resolution lowered from {int(dpi)} to {capped} dpi: the "
            f"{width_in:.1f} x {height_in:.1f} in figure would exceed "
            f"{self.MAX_PNG_PIXELS // 1_000_000} megapixels."
        )
        notes = self.__dict__.setdefault("raster_notes", [])
        if note not in notes:
            notes.append(note)
        return capped

    def to_svg_bytes(self, fig: Figure) -> bytes:
        with mpl.rc_context(self._export_rc()):
            self._prepare_export_figure(fig)
            return self._savefig_format(fig, fmt="svg")

    def to_png_bytes(self, fig: Figure, *, dpi: int | None = None) -> bytes:
        with mpl.rc_context(self._export_rc()):
            self._prepare_export_figure(fig)
            apply_draft_watermark(fig, self.export_framing)
            return self._savefig_format(
                fig,
                fmt="png",
                dpi=dpi if dpi is not None else self._export_dpi(),
            )

    def to_pdf_bytes(self, fig: Figure) -> bytes:
        """Single-page vector PDF of the rendered figure."""
        with mpl.rc_context(self._export_rc()):
            self._prepare_export_figure(fig)
            apply_draft_watermark(fig, self.export_framing)
            return self._savefig_format(fig, fmt="pdf")

    def export_figure_bytes(
        self,
        figure: Figure,
        export_formats: frozenset[str],
        *,
        polygons: list[GeologicalPolygon],
        projected_df: pd.DataFrame,
        collar_depths: dict[str, float] | None = None,
        water_levels: Sequence[WaterLevel] | None = None,
        lithology_codes: Sequence[str] | None = None,
        qa_lines: Sequence[str] = (),
        append_render_notes: bool = False,
    ) -> tuple[bytes, bytes, bytes]:
        """Export the figure; ``append_render_notes`` adds the final (post page
        framing) chemistry-drop and map-scale notes, and the render-time water
        line notes (ground clipping, LNAPL correction), to the PDF QA lines."""
        with mpl.rc_context(self._export_rc()):
            return self._export_figure_bytes(
                figure,
                export_formats,
                lithology_codes=lithology_codes,
                qa_lines=qa_lines,
                append_render_notes=append_render_notes,
            )

    def _export_figure_bytes(
        self,
        figure: Figure,
        export_formats: frozenset[str],
        *,
        lithology_codes: Sequence[str] | None,
        qa_lines: Sequence[str],
        append_render_notes: bool,
    ) -> tuple[bytes, bytes, bytes]:
        # savefig() draws on demand; an explicit canvas.draw() here doubled render cost.
        # Prefer cheaper formats first so a later failure still leaves cheaper bytes usable.
        ordered = [fmt for fmt in ("svg", "png", "pdf") if fmt in export_formats]
        svg_bytes = b""
        png_bytes = b""
        pdf_bytes = b""
        self._prepare_export_figure(figure)
        self.raster_notes = []
        if "png" in export_formats:
            # Decided now so the PDF QA lines can carry the note.
            self._capped_png_dpi(figure, self._export_dpi())
        if append_render_notes:
            # Labels are re-placed and the map-scale fit decided for the page
            # only now, so the notes are read after framing.
            qa_lines = (
                tuple(qa_lines)
                + tuple(getattr(self, "chemistry_label_notes", None) or ())
                + tuple(getattr(self, "chemistry_rdl_notes", None) or ())
                + tuple(self.map_scale_notes(figure))
                + tuple(self.header_overlap_notes(figure))
                # Water lines clipped at the ground and the LNAPL head
                # correction: deliverables must say the heads were corrected.
                + tuple(getattr(self, "water_line_notes", None) or ())
                + tuple(self.raster_notes)
            )
        export_dpi = self._export_dpi()
        watermark_applied = False
        for fmt in ordered:
            if fmt in ("png", "pdf") and not watermark_applied:
                apply_draft_watermark(figure, self.export_framing)
                watermark_applied = True
            if fmt == "svg":
                svg_bytes = self._savefig_format(figure, fmt="svg")
            elif fmt == "png":
                png_bytes = self._savefig_format(figure, fmt="png", dpi=export_dpi)
            elif fmt == "pdf":
                # Lazy import: PDF backend/font tools are heavy and slow startup for SVG/PNG-only runs.
                from report_export import assemble_section_pdf

                # Encode page 1 once with framing/DPI; report_export adds the
                # summary page (or, for consulting sheets, a QA notes page).
                pdf_bytes = assemble_section_pdf(
                    self,
                    self._savefig_format(figure, fmt="pdf"),
                    lithology_codes=lithology_codes,
                    qa_lines=qa_lines,
                )
        return svg_bytes, png_bytes, pdf_bytes

    # Gap (points) between the under-axis row (VE note / scale bar) and the
    # distance label or threshold key above it.
    _BELOW_AXES_ROW_GAP_PT = 5.0

    def _scale_bar_on_right(self) -> bool:
        return self.profile.scale_bar_position == "bottom_right"

    def _draw_scale_bar(self, ax) -> None:
        """Scale bar below the plot, clear of the geology and markers.

        Drawn inside the plot it sat on the fence polygons (and its label on
        a water marker) on long thin sections. It now hangs in the row under
        the distance label / threshold key, at the plot's left edge (right
        edge with ``scale_bar_position="bottom_right"``); its length is in
        data metres, so it stays true on every page and VE box.
        """
        if not self.profile.show_scale_bar:
            return
        bar_length = self.scale_bar_length_m
        if bar_length <= 0:
            return
        from matplotlib.lines import Line2D
        from matplotlib.offsetbox import AnnotationBbox, AuxTransformBox, TextArea, VPacker
        from matplotlib.transforms import Affine2D, blended_transform_factory

        points = Affine2D().scale(1.0 / 72.0) + ax.figure.dpi_scale_trans
        bar_box = AuxTransformBox(blended_transform_factory(ax.transData, points))
        x0 = float(ax.get_xlim()[0])
        bar_box.add_artist(
            Line2D([x0, x0 + bar_length], [0.0, 0.0], color=STICK_COLOR, linewidth=4, solid_capstyle="butt")
        )
        for x in (x0, x0 + bar_length):
            bar_box.add_artist(Line2D([x, x], [-3.5, 3.5], color=STICK_COLOR, linewidth=1.5))
        label = TextArea(
            f"{bar_length:g} m",
            textprops={"fontsize": 8, "fontweight": "bold", "color": LABEL_COLOR},
        )
        right = self._scale_bar_on_right()
        pack = VPacker(children=[label, bar_box], align="center", pad=0, sep=2)
        bar = AnnotationBbox(
            pack,
            (1.0 if right else 0.0, 0.0),
            xycoords=below_axes_anchor(ax),
            xybox=(0.0, -self._BELOW_AXES_ROW_GAP_PT),
            boxcoords="offset points",
            box_alignment=(1.0 if right else 0.0, 1.0),
            frameon=False,
            pad=0.0,
            annotation_clip=False,
        )
        bar.set_gid("scale-bar")
        bar.set_zorder(9)
        ax.add_artist(bar)
