"""Consulting-section layout drawing (mixin for CrossSectionRenderer)."""

from __future__ import annotations

import io
import logging
import re
import textwrap
from collections.abc import Sequence

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyArrow, Rectangle
from matplotlib.ticker import Formatter, FuncFormatter, Locator, MultipleLocator
from matplotlib.transforms import blended_transform_factory, offset_copy
from shapely.geometry import LineString
from shapely.geometry import Polygon as ShapelyPolygon

from lithology_codes import collect_lithology_codes
from models import ConsultingTitleBlock, VerticalGradient, WaterLevel
from render_theme import (
    CONSULTING_COLUMN_FILL,
    CONSULTING_FIGURE_BG,
    CONSULTING_SURFACE_COLOR,
    CONSULTING_WATER_COLOR,
    DEFAULT_CONSULTING_NOTES,
    LABEL_COLOR,
    OVERLAP_MARKER_COLOR,
    PARAMETER_READING_COLOR,
    REPORT_GRID_ALPHA,
    REPORT_GRID_COLOR,
    SCREEN_INTERVAL_HATCH,
    STICK_COLOR,
    TRACK_BORDER_COLOR,
    TRACK_FILL_COLOR,
    consulting_section_title,
    export_font_rc,
    primary_water_depth_by_hole,
    water_has_multiple_series,
)
from renderer_common import UNLOGGED_FILL_COLOR, UNLOGGED_LEGEND_LABEL, legend_swatch_hatch
from stratigraphy import GeologicalPolygon

logger = logging.getLogger(__name__)

_MBGS_NOTE = "mbgs DENOTES METRES BELOW GROUND SURFACE."


def _is_masl_note(note: str) -> bool:
    """True for the stock "masl denotes metres above sea level" note."""
    return "MASL DENOTES METRES ABOVE SEA LEVEL" in note.upper()

# Minimum legend column width (axes fraction) before wrapping to two columns.
_LEGEND_MIN_COL_WIDTH = 0.14
_LEGEND_CHAR_WIDTH = 0.0065  # approx. axes fraction per character at 7.5 pt


_HEADER_FONT_PT = 8.0
_HEADER_CHAR_WIDTH_IN = _HEADER_FONT_PT * 0.62 / 72.0



def _end_label_reserve_fraction(
    endpoint_lines: tuple[tuple[str, str], tuple[str, str]],
    *,
    font_pt: float,
    page_height_in: float,
) -> float:
    """Page-height fraction for the stacked end labels above the header row."""
    line_counts = [sum(1 for part in end if part) for end in endpoint_lines]
    lines = max(line_counts, default=0)
    if not lines:
        return 0.0
    # Bold text at linespacing 0.9 (~1.1 x font per line) plus a small gap.
    height_in = (lines * font_pt * 1.1 + 2.0) / 72.0
    return min(height_in / page_height_in, 0.08)

def _vertical_header_reserve_fraction(
    hole_summary: pd.DataFrame,
    *,
    page_height_in: float,
    axes_width_in: float = 11.0 * 0.89,
    font_pt: float = _HEADER_FONT_PT,
) -> float:
    """Page-height fraction to keep free above the plot for hole-ID headers.

    The plot starts at the page top, so the header pass cannot stagger
    outward. When two horizontal rows cannot hold every ID (dense well
    fields), the headers are drawn vertically and need the longest ID's
    length above the axes. The stock top margin holds two horizontal tiers
    at the design size (``_HEADER_FONT_PT``); when ``export_font_size``
    enlarges the headers, the extra tier height is reserved so the second
    tier does not run off the page. At the design size nothing is reserved.
    """
    if hole_summary is None or hole_summary.empty:
        return 0.0
    ids = [str(h) for h in hole_summary["hole_id"]]
    char_width_in = font_pt * 0.62 / 72.0
    widths_in = [len(h) * char_width_in + 0.08 for h in ids]
    if sum(widths_in) > 2.0 * axes_width_in:
        longest_in = max(widths_in) + 0.15
        return min(longest_in / page_height_in, 0.16)
    tiers = 1 if sum(widths_in) <= axes_width_in else 2
    # Header pass tier step is the text height (~1.3 x font) plus 1 pt.
    growth_in = tiers * max(font_pt - _HEADER_FONT_PT, 0.0) * 1.3 / 72.0
    return min(growth_in / page_height_in, 0.16)


# Consulting sheets are designed at this base size; ``export_font_size`` scales
# every explicit point size from here (the sidebar default of 8 clamps up to 9,
# so the stock output is unchanged).
_CONSULTING_BASE_FONT_PT = 9.0
# Threshold key below DISTANCE (m): clearance above the subtitle band, and the
# least gap the band may leave above the title block when it moves down for it.
_THRESHOLD_KEY_BAND_PAD_PT = 3.0
_BAND_BLOCK_MIN_GAP = 0.012

# Title-block metadata values: start size, smallest one-line size, and the
# smallest size tried when wrapping is forced.
_TITLE_CELL_BASE_PT = 7.0
_TITLE_CELL_FLOOR_PT = 6.0
_TITLE_CELL_HARD_MIN_PT = 5.0
_TITLE_CELL_LINE_SPACING = 1.1

# Subtitle band title: start size and the smallest sizes tried before wrapping.
_BAND_TITLE_BASE_PT = 9.0
_BAND_TITLE_FLOOR_PT = 7.0
_BAND_TITLE_HARD_MIN_PT = 6.0
# Vertical room (axes fraction) for the band title between the rule at 0.38
# and the top of the band, centred on the design position y=0.58.
_BAND_TITLE_ROW_HEIGHT = 0.38

# Notes panel: smallest notes size (pt) tried before the last note is cut
# with "…", and the design width of the CAD block axes (letter landscape) that
# the legend character budget was tuned on.
_NOTES_MIN_PT = 4.5
_DESIGN_BLOCK_AXES_WIDTH_IN = 11.0 * (0.95 - 0.06)

# A label that already names a (cross) section as a word is printed as-is.
_SECTION_WORD_RE = re.compile(r"\bsection\b", re.IGNORECASE)

# RL / depth axis labelling: at most this many labelled ticks per side, and
# labels at least this many font heights apart. The client figures label every
# 1 m on short sections (GWM) and every 2 m on the 16 m P2 depth axis; taller
# ranges thin to the next round step (1, 2, 5, 10, 20, 50 ... true metres).
_Y_MAX_LABELS = 15
_Y_LABEL_SPACING_EM = 1.8
_Y_MIN_STEP_M = 1.0


def _round_step_at_least(value: float) -> float:
    """Smallest 1/2/5 x 10^k step that is >= ``value``."""
    exponent = 10.0 ** np.floor(np.log10(max(value, 1e-9)))
    for mantissa in (1.0, 2.0, 5.0, 10.0):
        if mantissa * exponent >= value * (1.0 - 1e-9):
            return mantissa * exponent
    return 10.0 * exponent  # pragma: no cover - loop always returns


def true_metre_major_step(span_m: float, axis_height_pt: float, label_pt: float) -> float:
    """Round labelled-tick step (true metres) for a ``span_m`` tall RL/depth axis.

    Keeps the client's 1 m step while it fits, otherwise the next 1/2/5 step so
    at most ``_Y_MAX_LABELS`` labels show and labels never crowd each other.
    """
    span_m = abs(float(span_m))
    room = int(axis_height_pt // max(label_pt * _Y_LABEL_SPACING_EM, 1e-6)) if axis_height_pt > 0 else 0
    max_labels = max(2, min(_Y_MAX_LABELS, room)) if room else _Y_MAX_LABELS
    step = _Y_MIN_STEP_M
    while np.floor(span_m / step + 1e-9) + 1 > max_labels:
        step = _round_step_at_least(step * 1.0001)
    return step


def true_metre_minor_step(major_step: float) -> float:
    """Unlabelled minor step: 0.5 m under a 1 m major, else 1 m up to 10 m majors."""
    if major_step <= _Y_MIN_STEP_M:
        return major_step / 2.0
    return max(_Y_MIN_STEP_M, major_step / 10.0)


# --- Subtitle-band scale bar -------------------------------------------------

_INCHES_PER_METRE = 39.37
# Standard engineering drawing scales: (1, 1.25, 2, 2.5, 5) x 10^n.
_STANDARD_SCALE_MANTISSAS = (1.0, 1.25, 2.0, 2.5, 5.0)
# A computed ratio within this of a standard scale prints as that scale.
_STANDARD_SCALE_TOLERANCE = 0.03
# User map_scale disagreeing with the printed scale by more than this is logged.
_MAP_SCALE_MISMATCH_TOLERANCE = 0.10
_SCALE_BAR_MAX_FRACTION = 0.80
_SCALE_BAR_X = 0.02
_SCALE_BAR_Y = 0.55
_SCALE_BAR_UNITS = "Metres"


def nice_scale_bar_length(max_m: float) -> float:
    """Largest 1/2/5 x 10^n metres that is <= ``max_m``."""
    max_m = float(max_m)
    if not np.isfinite(max_m) or max_m <= 0.0:
        return 0.0
    exponent = 10.0 ** np.floor(np.log10(max_m))
    for mantissa in (5.0, 2.0, 1.0):
        if mantissa * exponent <= max_m * (1.0 + 1e-9):
            return float(mantissa * exponent)
    return float(exponent)  # pragma: no cover - mantissa 1 always fits


def scale_bar_tick_step(length_m: float) -> float:
    """Round tick step giving 3-5 equal segments of ``length_m`` (else 2, else 1)."""
    length_m = float(length_m)
    if length_m <= 0.0:
        return 0.0
    for segments in (5, 4, 3, 2):
        step = length_m / segments
        exponent = 10.0 ** np.floor(np.log10(step))
        mantissa = step / exponent
        if any(abs(mantissa - nice) < 1e-6 for nice in (1.0, 2.0, 2.5, 5.0, 10.0)):
            return step
    return length_m


def nearest_standard_scale(ratio: float) -> int:
    """Standard engineering scale denominator nearest ``ratio`` (log distance)."""
    ratio = max(float(ratio), 1.0)
    exponent = int(np.floor(np.log10(ratio)))
    candidates = [
        mantissa * 10.0**power
        for power in (exponent - 1, exponent, exponent + 1)
        for mantissa in _STANDARD_SCALE_MANTISSAS
    ]
    return int(round(min(candidates, key=lambda c: abs(np.log(c / ratio)))))


def scale_ratio_text(ratio: float) -> str:
    """Honest band text for a printed scale of 1:``ratio``.

    Prints a standard scale only when the drawing really is at it (within
    3 %); otherwise the ratio to two significant figures, marked approximate.
    """
    ratio = float(ratio)
    standard = nearest_standard_scale(ratio)
    if abs(standard / ratio - 1.0) <= _STANDARD_SCALE_TOLERANCE:
        return f"SCALE 1:{standard}"
    if ratio < 10:
        # Very short sections: keep a decimal (1:3.3, not a 10 %-off 1:3).
        return f"APPROX. SCALE 1:{ratio:.1f}"
    digits = int(np.floor(np.log10(ratio)))
    rounded = int(round(ratio, -max(digits - 1, 0)))
    return f"APPROX. SCALE 1:{rounded}"


def parse_map_scale(text: str | None) -> float | None:
    """Denominator of a "1:1 000"-style scale string, or None."""
    if not text:
        return None
    match = re.fullmatch(r"\s*1\s*:\s*([\d\s,]+(?:\.\d+)?)\s*", str(text))
    if not match:
        return None
    try:
        value = float(re.sub(r"[\s,]", "", match.group(1)))
    except ValueError:
        return None
    return value if value > 0 else None


def _scale_bar_label(value_m: float) -> str:
    return f"{value_m:g}"


class _TrueMetreLocator(Locator):
    """Major/minor y ticks at round true-metre steps on a VE-exaggerated axis.

    Decided at draw time from the view range and the axes height, so an export
    page resize re-thins the labels instead of keeping the render-size choice.
    """

    def __init__(self, ve: float, label_pt: float, *, minor: bool = False) -> None:
        self._ve = float(ve) if ve and ve > 0 else 1.0
        self._label_pt = float(label_pt)
        self._minor = minor

    def major_step(self) -> float:
        vmin, vmax = self.axis.get_view_interval()
        axes = self.axis.axes
        height_pt = axes.bbox.height * 72.0 / axes.figure.dpi
        return true_metre_major_step((vmax - vmin) / self._ve, height_pt, self._label_pt)

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        low, high = sorted((vmin / self._ve, vmax / self._ve))
        step = self.major_step()
        if self._minor:
            step = true_metre_minor_step(step)
        first = np.ceil(low / step - 1e-9) * step
        values = np.arange(first, high + step * 1e-6, step)
        return self.raise_if_exceeds(np.round(values, 6) * self._ve)


class _TrueMetreFormatter(Formatter):
    """True-metre labels (plotted value / VE); ``suppressed`` values print blank."""

    def __init__(self, ve: float) -> None:
        self._ve = float(ve) if ve and ve > 0 else 1.0
        self.suppressed: set[float] = set()

    def __call__(self, value, pos=None):
        true_value = round(value / self._ve, 6)
        if true_value in self.suppressed:
            return ""
        text = f"{true_value:.0f}" if abs(true_value - round(true_value)) < 1e-6 else f"{true_value:g}"
        return "0" if text == "-0" else text


class ConsultingLayoutMixin:
    """Consulting report-sheet layout methods. Expects CrossSectionRenderer attributes."""

    def _consulting_font_scale(self) -> float:
        """Multiplier applied to every explicit consulting point size."""
        size = max(float(self.profile.export_font_size), _CONSULTING_BASE_FONT_PT)
        return size / _CONSULTING_BASE_FONT_PT

    def _fs(self, base: float) -> float:
        """Scale a design-time point size by the profile's ``export_font_size``."""
        return float(base) * self._consulting_font_scale()

    @staticmethod
    def _consulting_display_title(label: str | None) -> str:
        """Section title for the subtitle band and TITLE cell.

        A label that already says "section" as a word (the app default
        "Borehole Cross-Section", or "CROSS SECTION B-B'") is printed as-is
        instead of becoming "CROSS SECTION BOREHOLE CROSS-SECTION". The test
        is word-bounded: "Intersection of Main St" still gets the prefix.
        """
        text = (label or "").strip()
        if _SECTION_WORD_RE.search(text):
            return text
        return consulting_section_title(text)

    def _render_consulting_section(
        self,
        polygons: list[GeologicalPolygon],
        projected_df: pd.DataFrame,
        collar_depths: dict[str, float] | None,
        *,
        water_levels: Sequence[WaterLevel] | None = None,
        lithology_codes: Sequence[str] | None = None,
    ) -> Figure:
        title_block = self.consulting_title_block or ConsultingTitleBlock(section_label=self.title)
        if not title_block.notes:
            title_block = title_block.model_copy(update={"notes": DEFAULT_CONSULTING_NOTES})
        if self.profile.y_axis_mode == "depth_below_collar":
            # A depth axis is not in masl: swap the stock masl note (built-in
            # default or the workbook template's) for the mbgs one.
            title_block = title_block.model_copy(
                update={
                    "notes": tuple(
                        _MBGS_NOTE if _is_masl_note(n) else n for n in title_block.notes
                    )
                }
            )
        if not water_levels:
            # The stock groundwater note is false on a section with no water data.
            kept = tuple(n for n in title_block.notes if n != DEFAULT_CONSULTING_NOTES[0])
            title_block = title_block.model_copy(update={"notes": kept or DEFAULT_CONSULTING_NOTES[1:]})
        if (
            self.disclaimer
            and self.interpretation_mode in {"interpolated", "correlation_lines"}
            and self.disclaimer not in title_block.notes
        ):
            title_block = title_block.model_copy(
                update={"notes": (*title_block.notes, self.disclaimer)}
            )

        ctx = self._hole_context(projected_df)
        # Lock letter landscape (11×8.5 in) so PNG/PDF match client page extracts.
        fig = plt.figure(figsize=(11.0, 8.5))
        fig.patch.set_facecolor(CONSULTING_FIGURE_BG)
        # right=0.95 leaves room for the twin RL axis label; at 0.97 it fell
        # off the fixed letter page and was silently dropped from PNG/PDF.
        top = 0.97 - _vertical_header_reserve_fraction(
            ctx.summary, page_height_in=8.5, font_pt=self._fs(_HEADER_FONT_PT)
        )
        # End labels ("A" / "NORTHWEST") get their own band above the hole-ID
        # headers, as on the client figures; sharing a row, the edge headers
        # printed into them ("MW18-18NORTHWEST").
        top -= _end_label_reserve_fraction(
            self._transect_endpoint_lines(title_block), font_pt=self._fs(8), page_height_in=8.5
        )
        fig.subplots_adjust(left=0.06, right=0.95, top=top, bottom=0.04)
        grid = GridSpec(3, 1, figure=fig, height_ratios=[58, 12, 22], hspace=0.12)
        ax = fig.add_subplot(grid[0, 0])
        sub_gs = grid[1, 0].subgridspec(1, 3, width_ratios=[32, 36, 32], wspace=0.14)
        ax_scale = fig.add_subplot(sub_gs[0, 0])
        ax_center = fig.add_subplot(sub_gs[0, 1])
        ax_notes = fig.add_subplot(sub_gs[0, 2])
        ax_block = fig.add_subplot(grid[2, 0])
        ax.set_facecolor(CONSULTING_FIGURE_BG)

        with mpl.rc_context(
            export_font_rc(
                self.profile.export_font_family,
                max(self.profile.export_font_size, _CONSULTING_BASE_FONT_PT),
            )
        ):
            if lithology_codes is None:
                lithology_codes = collect_lithology_codes(projected_df, polygons)
            elif not isinstance(lithology_codes, list):
                lithology_codes = list(lithology_codes)
            style_cache = self._style_cache_for(lithology_codes)

            hole_summary = ctx.summary
            collar_lookup = ctx.collar_lookup
            track_half = ctx.track_half
            ve = self.vertical_exaggeration
            collar_depths = collar_depths or {}
            water_levels_list = water_levels or ()
            profile_lookup = ctx.profile_lookup
            show_nm = self.profile.show_dry_well_nm

            # Units logged in a hole but drawn by no fence polygon (one-hole
            # units when pinch-outs are off) would otherwise vanish behind the
            # plain grey column while still listed in the legend.
            orphan_rows, orphan_lenses = self._orphan_unit_geometry(
                projected_df, polygons, hole_summary, track_half
            )
            fence_polygons = list(polygons) + orphan_lenses
            if fence_polygons:
                self._draw_fence_polygons(
                    ax,
                    fence_polygons,
                    style_cache,
                    ve,
                    alpha=self.profile.fence_alpha,
                    collar_lookup=collar_lookup,
                )
            else:
                self._has_pinch_out = False
            self._draw_consulting_surface(ax, hole_summary, collar_lookup)
            self._draw_well_columns(ax, hole_summary, collar_depths, collar_lookup, track_half)
            column_rows = projected_df if self.profile.show_track_lithology else orphan_rows
            if not column_rows.empty:
                self._draw_lithology_interval_rects(
                    ax,
                    column_rows,
                    style_cache,
                    track_half * 0.92,
                    collar_lookup,
                    zorder=9,
                    alpha=1.0,
                )
            if self.profile.show_track_lithology:
                # The grey column shows through wherever the log has no row.
                self._draw_unlogged_intervals(
                    ax, projected_df, collar_depths, collar_lookup, track_half, draw=False
                )
            if self.screen_intervals:
                self._draw_screen_intervals(
                    ax,
                    hole_summary,
                    self.screen_intervals,
                    collar_lookup,
                    track_half,
                    profile_lookup=profile_lookup,
                )
            if water_levels_list or show_nm:
                multi_series = water_has_multiple_series(water_levels_list)
                self._draw_water_table(
                    ax,
                    hole_summary,
                    water_levels_list,
                    collar_lookup,
                    label_elevations=self.profile.show_water_elevation_labels,
                    label_dry_wells=show_nm and not multi_series,
                    label_series_gaps=show_nm,
                    # One blue for a single series; several series keep their
                    # own shades (client reference: cyan vs dark blue).
                    water_color=None if multi_series else CONSULTING_WATER_COLOR,
                    profile_lookup=profile_lookup,
                )
            if self.vertical_gradients:
                self._draw_vertical_gradients(
                    ax,
                    hole_summary,
                    self.vertical_gradients,
                    water_levels_list,
                    collar_lookup,
                    profile_lookup=profile_lookup,
                )
            self._draw_well_id_labels(ax, hole_summary)
            self._draw_transect_end_labels(ax, title_block)
            if self.profile.show_overlap_markers and self.overlap_pairs:
                self._draw_overlap_markers(ax, collar_lookup, hole_summary=hole_summary)
            if self.faults:
                self._draw_faults(ax, collar_lookup, hole_summary=hole_summary)
            if self.unconformities:
                self._draw_unconformities(ax, collar_lookup, hole_summary=hole_summary)

            depth_axis = self.profile.y_axis_mode == "depth_below_collar"
            y_label = title_block.y_axis_label or self.profile.y_axis_label or (
                "DEPTH (mbgs)" if depth_axis else "ELEVATION (m)"
            )
            if depth_axis and "MASL" in y_label.upper():
                # The stock elevation label on a depth axis (P2 sticks preset).
                y_label = "DEPTH (mbgs)"
            ax.set_xlabel("DISTANCE (m)", fontsize=self._fs(10), labelpad=2, color=LABEL_COLOR)
            ax.set_ylabel(y_label, fontsize=self._fs(10), labelpad=6, color=LABEL_COLOR)
            ax.set_aspect("auto")
            for spine in ax.spines.values():
                spine.set_color("#374151")
                spine.set_linewidth(1.0)

            if self.profile.y_axis_mode == "depth_below_collar":
                ax.invert_yaxis()

            if (
                self.profile.consulting_axis_from_zero
                and self.profile.y_axis_mode != "depth_below_collar"
            ):
                self._apply_consulting_axis_limits(ax, hole_summary, track_half, water_levels_list)

            # Parameter labels after axis limits so collision spacing uses final ylim.
            self._draw_parameter_readings(
                ax,
                hole_summary,
                collar_lookup,
                profile_lookup=profile_lookup,
                column_half_m=track_half,
            )

            ax_right: plt.Axes | None = None
            if self.profile.show_dual_y_axes:
                ax_right = ax.twinx()
                ax_right.set_ylim(ax.get_ylim())
                ax_right.set_ylabel(y_label, fontsize=self._fs(10), labelpad=6, color=LABEL_COLOR)
                for spine in ax_right.spines.values():
                    spine.set_color("#374151")
                    spine.set_linewidth(1.0)

            if self.profile.show_report_grid:
                x_grid = 20.0 if ctx.x_span > 200.0 else self.profile.x_major_grid_m
                self._apply_report_grid(ax, ax_right, consulting=True, x_major_step=x_grid)

            # RL axis labels must stay on the page before the band and title
            # block are fitted to their (margin-dependent) panels.
            fig._css_main_axes = (ax, ax_right)
            fig._css_band_axes = ((ax_scale, ax_center, ax_notes), ax_block)
            self.fit_consulting_page_margins(fig)

            self._draw_subtitle_band(ax_scale, ax_center, ax_notes, title_block)
            self._draw_cad_title_block(ax_block, style_cache, lithology_codes, title_block)
            self._draw_consulting_footers(fig)
        return fig

    def _reserve_threshold_key_band(self, fig: Figure) -> None:
        """Lower the subtitle band so the threshold key under DISTANCE (m) clears it.

        The key hangs below the x-axis label (threshold colour mode only). The
        band first moves down into its gap above the title block, keeping
        ``_BAND_BLOCK_MIN_GAP`` of it; only what is still missing comes off
        the band's top. Measured from the grid positions each time, so it is
        idempotent and survives ``subplots_adjust`` (which resets them) on an
        export page resize; a sheet without the key is left untouched.
        """
        key = getattr(self, "_chemistry_threshold_key", None)
        band = getattr(fig, "_css_band_axes", None)
        if key is None or key.figure is not fig or not band:
            return
        band_axes, ax_block = band
        bases = [axis.get_subplotspec().get_position(fig) for axis in band_axes]
        for axis, base in zip(band_axes, bases, strict=True):
            axis.set_position(base)
        renderer = fig.canvas.get_renderer()
        pad_px = _THRESHOLD_KEY_BAND_PAD_PT * fig.dpi / 72.0
        key_bottom = (key.get_window_extent(renderer).y0 - pad_px) / fig.bbox.height
        need = max(base.y1 for base in bases) - key_bottom
        if need <= 0.0:
            return
        band_bottom = min(base.y0 for base in bases)
        slack = max(0.0, band_bottom - ax_block.get_position().y1 - _BAND_BLOCK_MIN_GAP)
        shift = min(need, slack)
        trim = need - shift
        for axis, base in zip(band_axes, bases, strict=True):
            axis.set_position([base.x0, base.y0 - shift, base.width, base.height - trim])

    def _apply_consulting_axis_limits(
        self,
        ax,
        hole_summary: pd.DataFrame,
        track_half: float,
        water_levels: Sequence[WaterLevel],
    ) -> None:
        if hole_summary.empty:
            return
        ve = self.vertical_exaggeration
        x_max = float(hole_summary["x_profile"].max())
        x_min = float(hole_summary["x_profile"].min())
        # Right margin: room for the last column and its labels, scaled to the
        # section (a fixed 5 m left ~15% blank on a 32 m section); long
        # sections keep the 5 m they always had.
        span = max(x_max - x_min, 1.0)
        x_pad = max(track_half + max(0.25 * track_half, 0.3), min(5.0, 0.06 * span))
        last_hole = str(hole_summary.loc[hole_summary["x_profile"].idxmax(), "hole_id"])
        if self.profile.show_parameter_labels and any(
            str(getattr(reading, "hole_id", "")) == last_hole
            for reading in getattr(self, "environmental_readings", ()) or ()
        ):
            # The last hole's values print to its right: leave room for a
            # ~40 pt label inside the frame (they were shrunk against the
            # right frame line on portrait pages). Axes width ~ 0.85 x page.
            axes_width_pt = 0.85 * float(ax.figure.get_size_inches()[0]) * 72.0
            label_room = 44.0 * span / max(axes_width_pt - 44.0, 1.0)
            x_pad = max(x_pad, track_half + label_room)
        # The first hole sits at x = 0; an axis starting at exactly 0 cut its
        # left half off. Start slightly negative (column half width + margin)
        # and keep the tick labels non-negative.
        left = min(0.0, x_min - (track_half + max(0.25 * track_half, 0.3)))
        ax.set_xlim(left, x_max + x_pad)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: "" if value < 0 else f"{value:g}"))
        y_min, y_max = self._uncertainty_y_bounds(hole_summary)
        y_pad = max(ve * 0.5, 1.0)
        collar_lookup = {
            str(row.hole_id): float(row.collar_elevation)
            for row in hole_summary.itertuples(index=False)
        }
        if water_levels and self.profile.show_water_elevation_labels:
            for level in water_levels:
                collar_rl = collar_lookup.get(level.hole_id)
                if collar_rl is None:
                    continue
                water_y = self._plot_y(collar_rl - level.depth, collar_rl)
                y_min = min(y_min, water_y)
                y_max = max(y_max, water_y)
        # Include active environmental sample depths so chloride markers are not clipped.
        active = {name.strip() for name in (self.environmental_parameters or ()) if name.strip()}
        if active and self.environmental_readings:
            for reading in self.environmental_readings:
                if reading.parameter not in active:
                    continue
                collar_rl = collar_lookup.get(reading.hole_id)
                if collar_rl is None:
                    continue
                for depth in (
                    reading.sample_depth,
                    reading.from_depth,
                    reading.to_depth,
                ):
                    if depth is None:
                        continue
                    sample_y = self._plot_y(collar_rl - float(depth), collar_rl)
                    y_min = min(y_min, sample_y)
                    y_max = max(y_max, sample_y)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    def _draw_consulting_footers(self, fig: Figure) -> None:
        footer_y = 0.01
        if self.overlap_pairs and self.profile.show_overlap_footer:
            fig.text(
                0.5,
                footer_y,
                (
                    f"Polygon overlap markers ({len(self.overlap_pairs)}): "
                    "review layer correlation between adjacent holes."
                ),
                ha="center",
                va="bottom",
                fontsize=self._fs(7),
                color=OVERLAP_MARKER_COLOR,
            )
            footer_y += 0.02
        return fig

    def _apply_report_grid(self, ax, ax_right=None, *, consulting: bool = False, x_major_step: float | None = None) -> None:
        ve = self.vertical_exaggeration
        ax.xaxis.set_major_locator(MultipleLocator(x_major_step if x_major_step is not None else self.profile.x_major_grid_m))
        if not consulting:
            y_locator = MultipleLocator(max(ve, 0.5))
            ax.yaxis.set_major_locator(y_locator)
            if ax_right is not None:
                ax_right.yaxis.set_major_locator(y_locator)
            ax.grid(True, which="major", color=REPORT_GRID_COLOR, alpha=REPORT_GRID_ALPHA, linewidth=0.6, zorder=0)
            return
        # Labelled ticks at a round true-metre step (thinned on tall sections),
        # the finer 1 m / 0.5 m step kept as unlabelled minor ticks and grid.
        # Same rule on both sides and in depth (mbgs) mode.
        label_pt = self._fs(8)
        for axis_ax in (ax, ax_right):
            if axis_ax is None:
                continue
            axis_ax.yaxis.set_major_locator(_TrueMetreLocator(ve, label_pt))
            axis_ax.yaxis.set_minor_locator(_TrueMetreLocator(ve, label_pt, minor=True))
            axis_ax.yaxis.set_major_formatter(_TrueMetreFormatter(ve))
            axis_ax.tick_params(axis="y", which="major", labelsize=label_pt)
        ax.grid(True, which="major", color=REPORT_GRID_COLOR, alpha=REPORT_GRID_ALPHA, linewidth=0.6, zorder=0)
        ax.xaxis.set_minor_locator(MultipleLocator(5.0))
        ax.grid(True, which="minor", color=REPORT_GRID_COLOR, alpha=0.45, linewidth=0.35, zorder=0)
        ax.tick_params(axis="both", which="major", labelsize=label_pt)

    def _suppress_header_tick_collisions(self, figure: Figure) -> None:
        """Blank RL tick labels a hole-ID header would cover at its drawn spot.

        Headers sit just above the frame, so only the topmost labels can meet
        them; dropping that label beats stepping the header off the page.
        Measured with each header at its base (pre-collision-pass) position.
        """
        axes = getattr(figure, "_css_main_axes", None)
        headers = [
            text
            for text in getattr(self, "_header_labels", None) or []
            if text.figure is figure and text.get_visible() and text.get_text().strip()
        ]
        if not axes:
            return
        formatters = []
        for axis_ax in axes:
            if axis_ax is None:
                continue
            formatter = axis_ax.yaxis.get_major_formatter()
            if isinstance(formatter, _TrueMetreFormatter):
                formatter.suppressed.clear()
                formatters.append((axis_ax, formatter))
        if not headers or not formatters:
            return
        renderer = figure.canvas.get_renderer()
        figure.draw_without_rendering()
        pad = renderer.points_to_pixels(1.0)
        header_boxes = []
        for text in headers:
            current = (text.get_transform(), text.get_horizontalalignment(), text.get_rotation())
            base = getattr(text, "_header_base", None)
            if base is not None:
                text.set_transform(base[0])
                text.set_horizontalalignment(base[1])
                text.set_rotation(base[2])
            header_boxes.append(text.get_window_extent(renderer).padded(pad))
            if base is not None:
                text.set_transform(current[0])
                text.set_horizontalalignment(current[1])
                text.set_rotation(current[2])
        for axis_ax, formatter in formatters:
            low, high = sorted(axis_ax.get_ylim())
            locs = axis_ax.yaxis.get_majorticklocs()
            for tick, loc in zip(axis_ax.yaxis.get_major_ticks(len(locs)), locs):
                if not low <= loc <= high:
                    continue
                for label in (tick.label1, tick.label2):
                    if not label.get_visible() or not label.get_text().strip():
                        continue
                    box = label.get_window_extent(renderer)
                    if any(box.overlaps(other) for other in header_boxes):
                        formatter.suppressed.add(round(loc / formatter._ve, 6))
        if any(formatter.suppressed for _ax, formatter in formatters):
            figure.draw_without_rendering()

    def _orphan_unit_geometry(
        self,
        projected_df: pd.DataFrame,
        polygons: Sequence[GeologicalPolygon],
        hole_summary: pd.DataFrame,
        track_half: float,
    ) -> tuple[pd.DataFrame, list[GeologicalPolygon]]:
        """Logged intervals no fence polygon draws, plus short lenses for them.

        Consulting columns are plain grey (no track lithology), so a unit is
        only visible through the fence polygons. With pinch-outs off (the
        generic consulting default) stratigraphy builds no polygon for a unit
        logged in a single hole, which then appeared in the legend but nowhere
        on the section. Return those intervals (drawn in the column) and a
        short inferred pinch-out lens either side of the hole, toward each
        neighbouring hole.
        """
        empty = projected_df.iloc[0:0]
        if (
            projected_df.empty
            or self.profile.show_track_lithology
            or self.interpretation_mode not in {"interpolated", "correlation_lines"}
        ):
            return empty, []
        by_code_hole: dict[tuple[str, str], list] = {}
        for geo_polygon in polygons:
            for hole_id in set(geo_polygon.hole_pair):
                by_code_hole.setdefault(
                    (str(geo_polygon.lithology_code), str(hole_id)), []
                ).append(geo_polygon.polygon)

        orphan_index: list = []
        for index, row in projected_df.iterrows():
            top = float(row["top_elevation"])
            bottom = float(row["bottom_elevation"])
            thickness = abs(top - bottom)
            if not np.isfinite(thickness) or thickness <= 1e-9:
                continue
            candidates = by_code_hole.get((str(row["lithology_code"]), str(row["hole_id"])), [])
            if candidates:
                x = float(row["x_profile"])
                column = LineString([(x, top), (x, bottom)])
                covered = sum(
                    shape.buffer(1e-3).intersection(column).length for shape in candidates
                )
                if covered >= 0.5 * thickness:
                    continue
            orphan_index.append(index)
        if not orphan_index:
            return empty, []
        orphans = projected_df.loc[orphan_index]

        hole_x = (
            hole_summary.set_index(hole_summary["hole_id"].astype(str))["x_profile"].astype(float)
            if not hole_summary.empty
            else pd.Series(dtype=float)
        )
        xs_sorted = np.sort(hole_x.to_numpy(dtype=float))
        lenses: list[GeologicalPolygon] = []
        for _, row in orphans.iterrows():
            hole_id = str(row["hole_id"])
            x = float(hole_x.get(hole_id, row["x_profile"]))
            top = float(row["top_elevation"])
            bottom = float(row["bottom_elevation"])
            mid = (top + bottom) / 2.0
            position = int(np.searchsorted(xs_sorted, x))
            neighbours = []
            if position > 0:
                neighbours.append(float(xs_sorted[position - 1]))
            upper = position + 1 if position < len(xs_sorted) and abs(xs_sorted[position] - x) < 1e-9 else position
            if upper < len(xs_sorted):
                neighbours.append(float(xs_sorted[upper]))
            for neighbour_x in neighbours:
                spacing = neighbour_x - x
                if abs(spacing) <= 2.0 * track_half:
                    continue
                # A quarter of the way to the neighbour (half way to where a
                # pinch-out would close), but always clear of the column.
                reach = min(0.5 * abs(spacing), max(0.25 * abs(spacing), 3.0 * track_half))
                apex_x = x + np.sign(spacing) * reach
                lenses.append(
                    GeologicalPolygon(
                        lithology_code=str(row["lithology_code"]),
                        polygon=ShapelyPolygon([(x, top), (x, bottom), (apex_x, mid)]),
                        # Both ends on this hole: depth-axis plots keep its collar RL.
                        hole_pair=(hole_id, hole_id),
                        is_pinch_out=True,
                    )
                )
        return orphans, lenses

    def _draw_well_columns(
        self,
        ax,
        hole_summary: pd.DataFrame,
        collar_depths: dict[str, float],
        collar_lookup: dict[str, float],
        track_half: float,
    ) -> None:
        geometry = self._well_rect_geometry(
            hole_summary, collar_depths, collar_lookup, track_half
        )
        if geometry is None:
            return
        self._add_rect_collection(
            ax,
            geometry,
            facecolors=CONSULTING_COLUMN_FILL,
            edgecolors=TRACK_BORDER_COLOR,
            linewidths=0.8,
            zorder=8,
        )

    def _draw_consulting_surface(
        self,
        ax,
        hole_summary: pd.DataFrame,
        collar_lookup: dict[str, float],
    ) -> None:
        if len(hole_summary) < 2:
            return
        surface_x = hole_summary["x_profile"].to_numpy(dtype=float)
        if self.profile.y_axis_mode == "depth_below_collar":
            surface_y = np.zeros(len(hole_summary), dtype=float)
        else:
            collars = self._collar_values(
                hole_summary["hole_id"],
                hole_summary["collar_elevation"],
                collar_lookup,
            )
            surface_y = self._plot_y_values(collars, collars)
        ax.plot(
            surface_x,
            surface_y,
            color=CONSULTING_SURFACE_COLOR,
            linewidth=1.0,
            solid_capstyle="round",
            zorder=6,
        )

    def _draw_vertical_gradients(
        self,
        ax,
        hole_summary: pd.DataFrame,
        vertical_gradients: Sequence[VerticalGradient],
        water_levels: Sequence[WaterLevel],
        collar_lookup: dict[str, float],
        *,
        profile_lookup: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if not vertical_gradients or hole_summary.empty:
            return
        water_depth_by_hole = primary_water_depth_by_hole(water_levels)
        if profile_lookup is None:
            profile_lookup = self._profile_lookup(hole_summary, collar_lookup)
        arrow_len = 0.12 * self.vertical_exaggeration
        for gradient in vertical_gradients:
            profile = profile_lookup.get(gradient.hole_id)
            if profile is None:
                continue
            x_profile, collar_rl = profile
            water_depth = water_depth_by_hole.get(gradient.hole_id, 1.0)
            anchor_rl = collar_rl - water_depth
            y = self._plot_y(anchor_rl, collar_rl)
            if gradient.direction == "up":
                arrow = FancyArrow(
                    float(x_profile),
                    float(y - arrow_len * 0.5),
                    0.0,
                    float(arrow_len),
                    width=0.35,
                    head_width=0.9,
                    head_length=0.25 * self.vertical_exaggeration,
                    length_includes_head=True,
                    facecolor=CONSULTING_WATER_COLOR,
                    edgecolor=CONSULTING_WATER_COLOR,
                    linewidth=0.0,
                    zorder=10,
                )
            else:
                arrow = FancyArrow(
                    float(x_profile),
                    float(y + arrow_len * 0.5),
                    0.0,
                    float(-arrow_len),
                    width=0.35,
                    head_width=0.9,
                    head_length=0.25 * self.vertical_exaggeration,
                    length_includes_head=True,
                    facecolor=CONSULTING_WATER_COLOR,
                    edgecolor=CONSULTING_WATER_COLOR,
                    linewidth=0.0,
                    zorder=10,
                )
            ax.add_patch(arrow)

    def _draw_well_id_labels(self, ax, hole_summary: pd.DataFrame) -> None:
        header_transform = ax.get_xaxis_transform()
        for row in hole_summary.itertuples(index=False):
            header = ax.text(
                float(row.x_profile),
                1.02,
                str(row.hole_id),
                transform=header_transform,
                ha="center",
                va="bottom",
                fontsize=self._fs(8),
                fontweight="bold",
                color=LABEL_COLOR,
                clip_on=False,
                zorder=10,
            )
            self._header_labels.append(header)

    def _transect_endpoint_lines(self, title_block: ConsultingTitleBlock) -> tuple[tuple[str, str], tuple[str, str]]:
        start_primary = title_block.transect_start_primary or title_block.transect_start_label
        start_secondary = title_block.transect_start_secondary
        end_primary = title_block.transect_end_primary or title_block.transect_end_label
        end_secondary = title_block.transect_end_secondary
        return (start_primary, start_secondary), (end_primary, end_secondary)

    def _draw_transect_end_labels(self, ax, title_block: ConsultingTitleBlock) -> None:
        (start_primary, start_secondary), (end_primary, end_secondary) = self._transect_endpoint_lines(
            title_block
        )
        if not start_primary and not start_secondary and not end_primary and not end_secondary:
            return
        # x in axes fraction, y in figure fraction: the labels sit just below
        # the page edge whatever the axes height. At transAxes y=1.06 they
        # landed above the page on every letter sheet and were never printed.
        # Inside the plot width at the two corners; the hole-ID header pass
        # treats these as obstacles and steps the end headers aside. (In the
        # page margins a long "NORTHWEST" ran off the sheet.)
        header_transform = blended_transform_factory(ax.transAxes, ax.figure.transFigure)
        start_transform = offset_copy(header_transform, fig=ax.figure, x=2.0, y=0.0, units="points")
        end_transform = offset_copy(header_transform, fig=ax.figure, x=-2.0, y=0.0, units="points")
        if start_primary or start_secondary:
            start_lines = [line for line in (start_primary, start_secondary) if line]
            ax.text(
                0.0,
                0.992,
                "\n".join(start_lines),
                transform=start_transform,
                ha="left",
                va="top",
                fontsize=self._fs(8),
                fontweight="bold",
                color=LABEL_COLOR,
                clip_on=False,
                zorder=10,
                linespacing=0.9,
            )
        if end_primary or end_secondary:
            end_lines = [line for line in (end_primary, end_secondary) if line]
            ax.text(
                1.0,
                0.992,
                "\n".join(end_lines),
                transform=end_transform,
                ha="right",
                va="top",
                fontsize=self._fs(8),
                fontweight="bold",
                color=LABEL_COLOR,
                clip_on=False,
                zorder=10,
                linespacing=0.9,
            )

    @staticmethod
    def _main_axes_metres_per_px(figure: Figure) -> float | None:
        axes = getattr(figure, "_css_main_axes", None)
        if not axes:
            return None
        main_ax = axes[0]
        lo, hi = main_ax.get_xlim()
        width_px = main_ax.bbox.width
        span = abs(float(hi) - float(lo))
        if width_px <= 0.0 or span <= 0.0 or not np.isfinite(span):
            return None
        return span / width_px

    @staticmethod
    def _scale_bar_key(figure: Figure) -> tuple:
        axes = getattr(figure, "_css_main_axes", None)
        if not axes:
            return ()
        main_ax = axes[0]
        return (tuple(main_ax.get_position().bounds), tuple(main_ax.get_xlim()))

    def _draw_consulting_scale_bar(self, ax_scale, title_block: ConsultingTitleBlock) -> None:
        """Scale bar drawn to the main section's true horizontal scale.

        Length is a round 1/2/5 x 10^n metres (or the title block's
        ``scale_bar_m`` when explicitly set and it fits) measured through the
        main axes' data->display transform; the SCALE text is the ratio at the
        current page size, never the title block's nominal ``map_scale``.
        """
        figure = ax_scale.figure
        metres_per_px = self._main_axes_metres_per_px(figure)
        panel_px = ax_scale.bbox.width
        if metres_per_px is None or panel_px <= 0.0:
            return
        font_pt = self._fs(7)
        bar_x, bar_y = _SCALE_BAR_X, _SCALE_BAR_Y
        renderer = figure.canvas.get_renderer()

        units = ax_scale.text(
            0.0,
            bar_y,
            _SCALE_BAR_UNITS,
            ha="left",
            va="center",
            fontsize=font_pt,
            color=LABEL_COLOR,
            transform=ax_scale.transAxes,
        )
        gap = 0.03
        units_frac = units.get_window_extent(renderer).width / panel_px
        max_frac = min(_SCALE_BAR_MAX_FRACTION, 1.0 - bar_x - gap - units_frac - 0.01)
        max_m = max(max_frac, 0.2) * panel_px * metres_per_px

        length_m = nice_scale_bar_length(max_m)
        requested = title_block.scale_bar_m if "scale_bar_m" in title_block.model_fields_set else None
        if requested and 0.4 * max_m <= float(requested) <= max_m:
            length_m = float(requested)
        if length_m <= 0.0:
            units.remove()
            return
        bar_w = length_m / metres_per_px / panel_px
        units.set_x(bar_x + bar_w + gap)

        step = scale_bar_tick_step(length_m)
        count = int(round(length_m / step))
        ticks = [step * i for i in range(count + 1)]
        tick_xs = [bar_x + (tick_m / length_m) * bar_w for tick_m in ticks]
        for tick_x in tick_xs:
            ax_scale.plot(
                [tick_x, tick_x],
                [bar_y - 0.05, bar_y + 0.05],
                color=STICK_COLOR,
                linewidth=1.0,
                transform=ax_scale.transAxes,
                clip_on=False,
            )
        labels = [
            ax_scale.text(
                tick_x,
                bar_y - 0.10,
                _scale_bar_label(tick_m),
                ha="center",
                va="top",
                fontsize=font_pt,
                color=LABEL_COLOR,
                transform=ax_scale.transAxes,
            )
            for tick_x, tick_m in zip(tick_xs, ticks)
        ]
        # Thin the labels (never the ends) if neighbours would touch.
        pad_px = 2.0 * figure.dpi / 72.0
        boxes = [label.get_window_extent(renderer) for label in labels]
        for stride in range(1, len(labels)):
            keep = {i for i in range(0, len(labels), stride)} | {len(labels) - 1}
            kept = sorted(keep)
            if all(
                boxes[a].x1 + pad_px <= boxes[b].x0 for a, b in zip(kept, kept[1:])
            ):
                break
        else:  # pragma: no cover - two end labels always fit an 80 % bar
            kept = [0, len(labels) - 1]
        for index, label in enumerate(labels):
            if index not in kept:
                label.remove()

        ax_scale.plot(
            [bar_x, bar_x + bar_w],
            [bar_y, bar_y],
            color=STICK_COLOR,
            linewidth=2.5,
            solid_capstyle="butt",
            transform=ax_scale.transAxes,
            clip_on=False,
        )
        ratio = metres_per_px * figure.dpi * _INCHES_PER_METRE
        ax_scale.text(
            # Left-aligned with the bar so the text can't spill out of the
            # panel's left edge (centred, it overflowed at large fonts).
            bar_x,
            0.12,
            scale_ratio_text(ratio),
            ha="left",
            va="center",
            fontsize=font_pt,
            fontweight="bold",
            color=LABEL_COLOR,
            transform=ax_scale.transAxes,
        )
        nominal = (
            parse_map_scale(title_block.map_scale)
            if "map_scale" in title_block.model_fields_set
            else None
        )
        if nominal and abs(nominal / ratio - 1.0) > _MAP_SCALE_MISMATCH_TOLERANCE:
            logger.warning(
                "Title block map scale %s differs from the printed section scale "
                "1:%.0f on this %.1f x %.1f in page; the scale bar shows the "
                "printed scale.",
                title_block.map_scale,
                ratio,
                *figure.get_size_inches(),
            )

    def _draw_subtitle_band(
        self,
        ax_scale,
        ax_center,
        ax_notes,
        title_block: ConsultingTitleBlock,
    ) -> None:
        for panel in (ax_scale, ax_center, ax_notes):
            panel.set_axis_off()
            panel.set_xlim(0, 1)
            panel.set_ylim(0, 1)

        if self.profile.show_scale_bar:
            # The bar's length depends on the main axes' metres per inch, which
            # an export page resize or margin fit changes: record it for re-fit.
            self._draw_refittable(
                ax_scale,
                lambda: self._draw_consulting_scale_bar(ax_scale, title_block),
                key_extra=lambda: self._scale_bar_key(ax_scale.figure),
            )

        section_title = self._consulting_display_title(title_block.section_label or self.title)
        # Fit the band title to the centre panel: a long label shrinks, then
        # wraps, instead of overprinting the scale bar and the notes panel.
        self._draw_refittable(ax_center, lambda: self._draw_fitted_cell_text(
            ax_center,
            section_title,
            x=0.5,
            y_centre=0.58,
            # The wspace gutters (~0.128 of this panel each side) are empty;
            # stop short of the scale-bar and notes panels themselves.
            cell_left=-0.12,
            cell_right=1.12,
            row_height=_BAND_TITLE_ROW_HEIGHT,
            max_chars=40,
            ha="center",
            fontweight="bold",
            base_pt=_BAND_TITLE_BASE_PT,
            floor_pt=_BAND_TITLE_FLOOR_PT,
            hard_min_pt=_BAND_TITLE_HARD_MIN_PT,
            clip_on=False,
        ))
        ax_center.plot(
            [0.10, 0.90],
            [0.38, 0.38],
            color=LABEL_COLOR,
            linewidth=0.8,
            transform=ax_center.transAxes,
            clip_on=False,
        )
        # Subtitle VE follows report chrome: on with the scale band, or when
        # show_ve_annotation is explicitly enabled. Both off = GIS paste mode.
        if self.profile.show_scale_bar or self.profile.show_ve_annotation:
            ve_text = (
                "NO VERTICAL EXAGGERATION"
                if abs(float(self.vertical_exaggeration) - 1.0) < 1e-9
                else f"{self.vertical_exaggeration:.0f}× VERTICAL EXAGGERATION"
            )
            ax_center.text(
                0.5,
                0.18,
                ve_text,
                ha="center",
                va="center",
                fontsize=self._fs(8),
                color=LABEL_COLOR,
                transform=ax_center.transAxes,
            )

        notes = title_block.notes or DEFAULT_CONSULTING_NOTES
        ax_notes.text(
            0.04,
            0.86,
            "NOTES:",
            ha="left",
            va="top",
            fontsize=self._fs(8),
            fontweight="bold",
            color=LABEL_COLOR,
            transform=ax_notes.transAxes,
        )
        self._draw_refittable(ax_notes, lambda: self._draw_notes_block(ax_notes, notes))

    def _draw_notes_block(self, ax_notes, notes: Sequence[str]) -> None:
        """Numbered notes, fitted to the notes panel by measurement.

        Geometry follows the font scale: wider text wraps sooner and each line
        advances by its (scaled) height. Each note is measured at the figure's
        current size; when the notes would cross the panel's right edge they
        re-wrap narrower, and when they would run below the panel the notes
        size steps down (below the design size if need be, to
        ``_NOTES_MIN_PT``). If even that does not fit, the last note that fits
        is cut with "…". At the design size notes that fit draw as before.
        """
        font_scale = self._consulting_font_scale()
        shown = [f"{index}. {note}" for index, note in enumerate(notes[:4], start=1)]
        if not shown:
            return
        note_top = 0.68 - 0.18 * (font_scale - 1.0)
        pad_px = 1.5 * ax_notes.figure.dpi / 72.0
        # The notes panel is the rightmost: notes may use the page's right
        # margin (stock sheets do) but must stay clear of the page edge.
        right_px = ax_notes.figure.bbox.width - 0.12 * ax_notes.figure.dpi
        bottom_px = ax_notes.transAxes.transform((0.0, 0.0))[1] + pad_px

        scales: list[float] = []
        candidate = font_scale
        while candidate > 1.0 + 1e-9:
            scales.append(candidate)
            candidate -= 0.05
        min_scale = _NOTES_MIN_PT / 6.5
        candidate = 1.0
        while candidate >= min_scale - 1e-9:
            scales.append(candidate)
            candidate -= 0.05
        for index, scale in enumerate(scales):
            last = index == len(scales) - 1
            artists = self._layout_notes(
                ax_notes,
                shown,
                scale=scale,
                font_pt=self._fs(6.5) if scale == font_scale else 6.5 * scale,
                note_top=note_top,
                right_px=right_px,
                bottom_px=bottom_px,
                truncate=last,
            )
            if artists is not None:
                return

    def _layout_notes(
        self,
        ax_notes,
        shown: list[str],
        *,
        scale: float,
        font_pt: float,
        note_top: float,
        right_px: float,
        bottom_px: float,
        truncate: bool,
    ) -> list | None:
        """Draw ``shown`` at ``font_pt``; ``None`` (nothing left drawn) if it does not fit."""
        ax_height_px = (
            ax_notes.transAxes.transform((0.0, 1.0))[1] - ax_notes.transAxes.transform((0.0, 0.0))[1]
        )
        line_step = 0.11 * scale**1.05
        note_y = note_top
        artists = []

        def _abandon() -> None:
            for artist in artists:
                artist.remove()

        for note in shown:
            wrap_width = max(12, int(58 / scale))
            lines = textwrap.wrap(note, width=wrap_width) or [note]
            artist = None
            for attempt in range(6):
                # Wrap ourselves and advance by the number of lines: matplotlib's
                # wrap=True kept a fixed step, so a two-line note printed its
                # second line over the next note.
                artist = ax_notes.text(
                    0.04,
                    note_y,
                    "\n".join(lines),
                    ha="left",
                    va="top",
                    fontsize=font_pt,
                    color=LABEL_COLOR,
                    transform=ax_notes.transAxes,
                    linespacing=1.15,
                )
                extent = artist.get_window_extent()
                if extent.x1 <= right_px or wrap_width <= 12 or attempt == 5:
                    break
                ratio = (right_px - extent.x0) / max(extent.width, 1.0)
                wrap_width = max(12, min(wrap_width - 1, int(wrap_width * ratio)))
                lines = textwrap.wrap(note, width=wrap_width) or [note]
                artist.remove()
            assert artist is not None
            extent = artist.get_window_extent()
            if extent.y0 < bottom_px:
                if not truncate:
                    artist.remove()
                    _abandon()
                    return None
                # Last resort: keep the lines that fit and mark the cut.
                line_px = extent.height / max(len(lines), 1)
                keep = int((extent.y1 - bottom_px) / max(line_px, 1.0))
                artist.remove()
                if keep >= 1:
                    kept = lines[:keep]
                    kept[-1] = kept[-1].rstrip(" .;,") + "…"
                    artists.append(
                        ax_notes.text(
                            0.04,
                            note_y,
                            "\n".join(kept),
                            ha="left",
                            va="top",
                            fontsize=font_pt,
                            color=LABEL_COLOR,
                            transform=ax_notes.transAxes,
                            linespacing=1.15,
                        )
                    )
                return artists
            if extent.x1 > right_px and not truncate:
                artist.remove()
                _abandon()
                return None
            artists.append(artist)
            note_y -= max(line_step * len(lines), extent.height / max(ax_height_px, 1.0))
        return artists

    def _draw_cad_title_block(
        self,
        ax,
        style_cache: dict,
        lithology_codes: list[str],
        title_block: ConsultingTitleBlock,
    ) -> None:
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Three independent panels so legend / title / prepared content stay boxed.
        legend_box = (0.01, 0.05, 0.32, 0.90)
        has_right = bool(title_block.prepared_for or title_block.prepared_by)
        if has_right:
            title_box = (0.34, 0.05, 0.31, 0.90)
            right_box = (0.66, 0.05, 0.33, 0.90)
            panels = (legend_box, title_box, right_box)
        else:
            # Give the title/meta table the remaining width when no prepared logos.
            title_box = (0.34, 0.05, 0.65, 0.90)
            right_box = None
            panels = (legend_box, title_box)
        for box in panels:
            ax.add_patch(
                Rectangle(
                    (box[0], box[1]),
                    box[2],
                    box[3],
                    fill=False,
                    edgecolor="#94A3B8",
                    linewidth=1.0,
                    transform=ax.transAxes,
                    clip_on=False,
                )
            )

        self._draw_refittable(ax, lambda: self._draw_legend_panel(
            ax,
            style_cache,
            lithology_codes,
            title_block,
            panel=legend_box,
        ))

        meta_rows: list[tuple[str, str]] = []
        if title_block.project_number:
            meta_rows.append(("PROJECT", title_block.project_number))
        section_label = title_block.section_label or self.title
        if section_label:
            meta_rows.append(("TITLE", self._consulting_display_title(section_label)))
        if title_block.source:
            meta_rows.append(("SOURCE", title_block.source))
        if title_block.map_scale:
            # A scale the user never set (the model default "1:1000") would
            # contradict the true-scale bar; "AS SHOWN" defers to the bar.
            user_scale = "map_scale" in title_block.model_fields_set
            meta_rows.append(("SCALE", title_block.map_scale if user_scale else "AS SHOWN"))
        if title_block.date:
            meta_rows.append(("DATE", title_block.date))
        if title_block.drawn_by:
            meta_rows.append(("DRAWN BY", title_block.drawn_by))
        if title_block.revised:
            meta_rows.append(("REVISED", title_block.revised))
        if title_block.figure_number:
            meta_rows.append(("FIGURE NO.", title_block.figure_number))
        self._draw_refittable(
            ax, lambda: self._draw_title_block_metadata_table(ax, meta_rows, panel=title_box)
        )

        if right_box is None:
            return
        self._draw_refittable(ax, lambda: self._draw_prepared_panel(ax, title_block, right_box))

    def _draw_prepared_panel(
        self,
        ax,
        title_block: ConsultingTitleBlock,
        right_box: tuple[float, float, float, float],
    ) -> None:
        """PREPARED FOR / PREPARED BY labels and values, fitted to the right box."""
        right_x = right_box[0] + 0.03
        # Label-to-value gap grows with the font scale (identical at 1.0).
        value_extra = 0.10 * (self._consulting_font_scale() ** 1.1 - 1.0)
        if title_block.prepared_for:
            ax.text(
                right_x,
                0.88,
                "PREPARED FOR",
                fontsize=self._fs(8),
                fontweight="bold",
                color=LABEL_COLOR,
                transform=ax.transAxes,
            )
            self._draw_prepared_value(ax, title_block.prepared_for, right_x, 0.78 - value_extra, right_box)
            self._draw_logo_image(ax, title_block.logo_prepared_for_bytes, (0.78, 0.62))
        if title_block.prepared_by:
            ax.text(
                right_x,
                0.48,
                "PREPARED BY",
                fontsize=self._fs(8),
                fontweight="bold",
                color=LABEL_COLOR,
                transform=ax.transAxes,
            )
            self._draw_prepared_value(ax, title_block.prepared_by, right_x, 0.38 - value_extra, right_box)
            self._draw_logo_image(ax, title_block.logo_prepared_by_bytes, (0.78, 0.22))

    def _draw_prepared_value(
        self,
        ax,
        value: str,
        x: float,
        y: float,
        right_box: tuple[float, float, float, float],
    ) -> None:
        """A PREPARED FOR/BY value: shrinks, then wraps downward, inside the box.

        A value that fits on one line at the design size draws exactly as
        before (baseline at ``y``).
        """
        self._draw_fitted_cell_text(
            ax,
            value,
            x=x,
            y_centre=y,
            cell_right=right_box[0] + right_box[2] - 0.005,
            row_height=0.24,
            max_chars=30,
            base_pt=8.0,
            floor_pt=6.5,
            hard_min_pt=5.0,
            va="baseline",
            clip_on=False,
        )

    @staticmethod
    def _legend_label_char_budget(
        col_width_axes: float,
        font_size: float,
        *,
        swatch_w: float,
        axes_width_in: float | None = None,
    ) -> int:
        """Estimate how many characters fit in a legend column without bleeding sideways.

        ``_LEGEND_CHAR_WIDTH`` was tuned on the letter-landscape block axes; a
        narrower axes (letter portrait) fits proportionally fewer characters.
        """
        usable = max(col_width_axes - swatch_w - 0.018, 0.02)
        char_w = _LEGEND_CHAR_WIDTH * (font_size / 7.5)
        if axes_width_in and abs(axes_width_in - _DESIGN_BLOCK_AXES_WIDTH_IN) > 1e-3:
            char_w *= _DESIGN_BLOCK_AXES_WIDTH_IN / axes_width_in
        return max(8, int(usable / char_w))

    @staticmethod
    def _legend_panel_clip(ax, panel: tuple[float, float, float, float]) -> Rectangle:
        left, bottom, width, height = panel
        clip_rect = Rectangle(
            (left, bottom),
            width,
            height,
            transform=ax.transAxes,
            visible=False,
        )
        ax.add_patch(clip_rect)
        return clip_rect

    def _draw_legend_panel(
        self,
        ax,
        style_cache: dict,
        lithology_codes: list[str],
        title_block: ConsultingTitleBlock,
        *,
        panel: tuple[float, float, float, float],
    ) -> None:
        left, bottom, width, height = panel
        pad_x = 0.025
        pad_y = 0.04
        content_left = left + pad_x
        content_top = bottom + height - pad_y
        content_bottom = bottom + pad_y
        swatch_w = 0.03

        entries: list[tuple[str, str, dict[str, object]]] = []
        for code in lithology_codes:
            style = self._resolve_style(code, style_cache)
            entries.append(
                (
                    "swatch",
                    code.upper(),
                    {
                        "facecolor": style.color,
                        "edgecolor": style.edge_color,
                        "hatch": style.hatch or None,
                    },
                )
            )
        n_code_entries = len(entries)
        if getattr(self, "_has_unlogged_intervals", False):
            entries.append(
                (
                    "swatch",
                    UNLOGGED_LEGEND_LABEL.upper(),
                    {
                        "facecolor": UNLOGGED_FILL_COLOR,
                        "edgecolor": TRACK_BORDER_COLOR,
                        "hatch": None,
                    },
                )
            )
        if self.screen_intervals:
            entries.append(
                (
                    "swatch",
                    title_block.screen_legend_label or "SCREENED INTERVAL",
                    {
                        "facecolor": TRACK_FILL_COLOR,
                        "edgecolor": TRACK_BORDER_COLOR,
                        "hatch": SCREEN_INTERVAL_HATCH,
                    },
                )
            )
        if title_block.show_gradient_legend and self.vertical_gradients:
            entries.append(("gradient", "VERTICAL GRADIENT DIRECTION", {}))

        gw_legend = self.water_series_legend or []
        compact_gw = self.profile.compact_water_legend
        gw_linestyle = "-" if self.profile.water_line_solid else "--"
        if self.profile.show_water_legend and gw_legend:
            relative = self.profile.y_axis_mode == "depth_below_collar"
            default_elev = (
                "GROUNDWATER DEPTH (mbgs)" if relative else "GROUNDWATER ELEVATION masl"
            )
            default_level = (
                "GROUNDWATER LEVEL (mbgs)" if relative else "GROUNDWATER LEVEL (masl)"
            )
            for entry in gw_legend:
                if compact_gw:
                    legend_label = entry.get("level_label") or entry.get(
                        "elevation_label", default_level
                    )
                    entries.append(
                        (
                            "line_marker",
                            str(legend_label),
                            {
                                "color": entry.get("color", CONSULTING_WATER_COLOR),
                                "marker": entry.get("marker", "v"),
                                "linestyle": gw_linestyle,
                            },
                        )
                    )
                else:
                    entries.append(
                        (
                            "marker",
                            str(entry.get("elevation_label", default_elev)),
                            {
                                "color": entry.get("color", CONSULTING_WATER_COLOR),
                                "marker": entry.get("marker", "v"),
                            },
                        )
                    )
                    entries.append(
                        (
                            "line",
                            str(entry.get("level_label", default_level)),
                            {
                                "color": entry.get("color", CONSULTING_WATER_COLOR),
                                "linestyle": gw_linestyle,
                            },
                        )
                    )

        draw_param_markers = self.profile.parameter_draw_markers
        for entry in self.parameter_series_legend or []:
            kind = "line_marker" if draw_param_markers else "text"
            entries.append(
                (
                    kind,
                    str(entry.get("label", entry.get("parameter", "PARAMETER"))),
                    {
                        "color": entry.get("color", PARAMETER_READING_COLOR),
                        "marker": entry.get("marker", "D"),
                        "linestyle": "--",
                        # Red P2-style sample value ("120") shown in the swatch column.
                        "parameter_entry": entry,
                    },
                )
            )
        if getattr(self, "_has_pinch_out", False) and self.profile.show_pinch_out_legend:
            entries.append(
                (
                    "line",
                    "INFERRED PINCH-OUT",
                    {"color": LABEL_COLOR, "linestyle": "--"},
                )
            )

        # Header + entries must stay inside the legend box (never bleed into title block).
        ncol = max(1, self.profile.legend_ncol)
        usable_width = width - 2 * pad_x
        col_width_single = usable_width
        col_width_two = (usable_width - 0.02) / 2
        # Two columns when asked for, or whenever a single stack would need
        # more than eight rows (entries past the panel were silently dropped).
        # Row geometry follows the font scale so enlarged labels keep their
        # line pitch (and the capacity / "+N MORE UNITS" trim stays honest).
        scale = self._consulting_font_scale()
        header_h = 0.14 * scale  # "LEGEND" at 8.5 pt plus a gap above the first row
        min_step = 0.055 * scale
        two_col_possible = col_width_two >= _LEGEND_MIN_COL_WIDTH
        rows_available = max(1, int((content_top - header_h - content_bottom) // min_step))
        capacity = rows_available * (2 if two_col_possible else 1)
        if len(entries) > capacity:
            # Trim lithology swatches (never the water/screen/parameter keys)
            # and say how many units are not listed.
            overflow = len(entries) - capacity + 1
            keep_codes = max(0, n_code_entries - overflow)
            hidden = n_code_entries - keep_codes
            entries = (
                entries[:keep_codes]
                + [("text", f"+{hidden} MORE UNITS (SEE LOG)", {"color": LABEL_COLOR})]
                + entries[n_code_entries:]
            )
        use_two_cols = two_col_possible and ((ncol >= 2 and len(entries) > 6) or len(entries) > 8)
        if use_two_cols:
            mid = (len(entries) + 1) // 2
            column_groups: list[list[tuple[str, str, dict[str, object]]]] = [
                entries[:mid],
                entries[mid:],
            ]
            col_gap = 0.02
            col_width = col_width_two
            column_layouts = [
                (content_left, content_left + swatch_w + 0.012),
                (
                    content_left + col_width + col_gap,
                    content_left + col_width + col_gap + swatch_w + 0.012,
                ),
            ]
        else:
            column_groups = [entries]
            col_width = col_width_single
            column_layouts = [(content_left, content_left + swatch_w + 0.012)]

        clip_rect = self._legend_panel_clip(ax, panel)

        n_rows = max(len(group) for group in column_groups)
        step = min(
            0.10 * scale,
            max(min_step, (content_top - header_h - content_bottom) / max(n_rows, 1)),
        )
        font_size = self._fs(7.5) if step >= 0.08 * scale else self._fs(6.5)
        swatch_h = 0.04 * scale
        y_header = content_top
        header = ax.text(
            content_left,
            y_header,
            "LEGEND",
            fontsize=self._fs(8.5),
            fontweight="bold",
            color=LABEL_COLOR,
            transform=ax.transAxes,
            va="top",
            clip_on=True,
        )
        header.set_clip_path(clip_rect)
        entry_top = y_header - max(step, header_h)

        for group, (col_left, col_text_x) in zip(column_groups, column_layouts, strict=True):
            y = entry_top
            max_label_chars = self._legend_label_char_budget(
                col_width,
                font_size,
                swatch_w=swatch_w,
                axes_width_in=ax.get_position().width * ax.figure.get_size_inches()[0],
            )
            for kind, label, style in group:
                if y < content_bottom + 0.02:
                    break
                display = label
                if kind == "text" and len(label) > max_label_chars:
                    # "+N MORE UNITS (SEE LOG)": drop a word, never cut mid-word.
                    label = label.replace(" UNITS", "")
                    display = label
                if len(label) > max_label_chars:
                    paren = label.rfind("(")
                    if paren > 0 and label.endswith(")") and len(label) - paren <= 14:
                        prefix_budget = max_label_chars - (len(label) - paren) - 1
                        if prefix_budget >= 8:
                            display = label[:prefix_budget].rstrip(" -:") + "…" + label[paren:]
                        else:
                            display = label[: max_label_chars - 1] + "…"
                    else:
                        display = label[: max_label_chars - 1] + "…"
                if kind == "swatch":
                    rect = Rectangle(
                        (col_left, y - swatch_h * 0.55),
                        swatch_w,
                        swatch_h,
                        facecolor=style["facecolor"],
                        edgecolor=style["edgecolor"],
                        linewidth=0.6,
                        hatch=legend_swatch_hatch(
                            style.get("hatch"),
                            swatch_h * ax.get_position().height * ax.figure.get_size_inches()[1],
                        ),
                        transform=ax.transAxes,
                        clip_on=True,
                    )
                    ax.add_patch(rect)
                    rect.set_clip_path(clip_rect)
                elif kind == "gradient":
                    arrow = FancyArrow(
                        col_left + 0.012,
                        y - 0.01,
                        0.0,
                        0.03,
                        width=0.006,
                        head_width=0.016,
                        head_length=0.01,
                        length_includes_head=True,
                        transform=ax.transAxes,
                        facecolor=CONSULTING_WATER_COLOR,
                        edgecolor=CONSULTING_WATER_COLOR,
                        clip_on=True,
                    )
                    ax.add_patch(arrow)
                    arrow.set_clip_path(clip_rect)
                elif kind == "marker":
                    (marker_line,) = ax.plot(
                        [col_left, col_left + 0.03],
                        [y, y - 0.02],
                        marker=style.get("marker", "v"),
                        color=style.get("color", CONSULTING_WATER_COLOR),
                        linewidth=0,
                        markersize=5,
                        transform=ax.transAxes,
                        clip_on=True,
                    )
                    marker_line.set_clip_path(clip_rect)
                elif kind == "line":
                    (line,) = ax.plot(
                        [col_left, col_left + 0.03],
                        [y, y],
                        color=style.get("color", LABEL_COLOR),
                        linewidth=1.4,
                        linestyle=style.get("linestyle", "--"),
                        transform=ax.transAxes,
                        clip_on=True,
                    )
                    line.set_clip_path(clip_rect)
                elif kind == "text":
                    sample = self._draw_parameter_legend_sample(
                        ax,
                        col_left,
                        y,
                        style.get("parameter_entry") or {},
                        font_size=font_size,
                        clip_path=clip_rect,
                    )
                    if sample is not None:
                        # Label moves to the text column, beside its sample.
                        kind = "sampled_text"
                else:  # line_marker
                    (lm_line,) = ax.plot(
                        [col_left, col_left + 0.03],
                        [y, y],
                        marker=style.get("marker", "D"),
                        color=style.get("color", PARAMETER_READING_COLOR),
                        linewidth=1.4,
                        linestyle=style.get("linestyle", "--"),
                        markersize=5,
                        transform=ax.transAxes,
                        clip_on=True,
                    )
                    lm_line.set_clip_path(clip_rect)
                label_artist = ax.text(
                    col_left if kind == "text" else col_text_x,  # sampled_text → text column
                    y,
                    display,
                    fontsize=font_size,
                    va="center",
                    color=style.get("color", LABEL_COLOR) if kind == "text" else LABEL_COLOR,
                    transform=ax.transAxes,
                    clip_on=True,
                )
                label_artist.set_clip_path(clip_rect)
                y -= step

    def _draw_logo_image(self, ax, logo_bytes: bytes | None, position: tuple[float, float]) -> None:
        if not logo_bytes:
            return
        try:
            image = mpimg.imread(io.BytesIO(logo_bytes), format="png")
        except Exception:
            logger.warning("Could not decode consulting logo image")
            return
        imagebox = OffsetImage(image, zoom=0.18)
        ab = AnnotationBbox(
            imagebox,
            position,
            xycoords=ax.transAxes,
            frameon=False,
            box_alignment=(0.0, 0.5),
        )
        ax.add_artist(ab)

    def _fit_title_cell_text(
        self,
        ax,
        text: str,
        *,
        x: float,
        y_centre: float,
        cell_right: float,
        row_height: float,
        max_chars: int,
        cell_left: float | None = None,
        ha: str = "left",
        fontweight: str | None = None,
        base_pt: float | None = None,
        floor_pt: float | None = None,
        hard_min_pt: float | None = None,
    ) -> tuple[list[str], float, float]:
        """Lines, point size and line step (axes fraction) that keep ``text`` in its cell.

        The value is measured with the figure's renderer at the figure's
        current size. It stays on one line at the base size when it fits;
        otherwise the size steps down to the floor and, only when that is
        still too wide, the text wraps at a size whose lines fit the row
        height, so a long TITLE never crosses the cell rule. A short value
        renders exactly as before. ``cell_left`` bounds centred / right-aligned
        text on the left as well; the design sizes default to the TITLE cell's.
        """
        base_pt = self._fs(_TITLE_CELL_BASE_PT if base_pt is None else base_pt)
        floor_pt = self._fs(_TITLE_CELL_FLOOR_PT if floor_pt is None else floor_pt)
        hard_min_pt = self._fs(_TITLE_CELL_HARD_MIN_PT if hard_min_pt is None else hard_min_pt)
        step_pt = self._fs(0.5)
        try:
            probe = ax.text(
                x,
                y_centre,
                text,
                fontsize=base_pt,
                va="center",
                ha=ha,
                fontweight=fontweight,
                transform=ax.transAxes,
            )
        except Exception:  # pragma: no cover - defensive
            return textwrap.wrap(text, width=max_chars) or [""], base_pt, 0.0
        try:
            pad_px = 1.5 * ax.figure.dpi / 72.0
            origin = ax.transAxes.transform((0.0, 0.0))
            right_px = ax.transAxes.transform((cell_right, 0.0))[0] - pad_px
            left_px = (
                None
                if cell_left is None
                else ax.transAxes.transform((cell_left, 0.0))[0] + pad_px
            )
            ax_height_px = ax.transAxes.transform((0.0, 1.0))[1] - origin[1]
            row_px = row_height * ax_height_px - 2.0 * pad_px

            def _fits(extent) -> bool:
                return extent.x1 <= right_px and (left_px is None or extent.x0 >= left_px)

            pt = base_pt
            while True:
                probe.set_fontsize(pt)
                extent = probe.get_window_extent()
                if _fits(extent):
                    return [text], pt, 0.0
                if pt - step_pt < floor_pt - 1e-9:
                    break
                pt -= step_pt

            # Too wide even at the floor: wrap, shrinking below the floor only
            # when the wrapped lines would not fit the row height.
            pt = floor_pt
            while True:
                probe.set_fontsize(pt)
                extent = probe.get_window_extent()
                avail_px = right_px - (extent.x0 if left_px is None else left_px)
                chars = max(8, int(len(text) * avail_px / max(extent.width, 1.0)))
                lines = textwrap.wrap(text, width=chars) or [text]
                line_px = extent.height * _TITLE_CELL_LINE_SPACING
                max_lines = max(1, int(row_px / line_px))
                if len(lines) <= max_lines or pt - step_pt < hard_min_pt - 1e-9:
                    if len(lines) > max_lines:
                        lines = lines[:max_lines]
                        if len(lines[-1]) > 3:
                            lines[-1] = lines[-1][: max(3, len(lines[-1]) - 1)] + "…"
                    return lines, pt, line_px / ax_height_px
                pt -= step_pt
        except Exception:
            logger.debug("Title cell measurement unavailable; using character wrap", exc_info=True)
            wrapped = textwrap.wrap(text, width=max_chars) or [""]
            return wrapped, (self._fs(6.5) if len(wrapped) > 1 else base_pt), min(
                0.10, (row_height * 0.7) / max(len(wrapped), 1)
            )
        finally:
            probe.remove()

    def _draw_fitted_cell_text(
        self,
        ax,
        text: str,
        *,
        x: float,
        y_centre: float,
        cell_right: float,
        row_height: float,
        max_chars: int,
        cell_left: float | None = None,
        ha: str = "left",
        va: str = "center",
        fontweight: str | None = None,
        base_pt: float | None = None,
        floor_pt: float | None = None,
        hard_min_pt: float | None = None,
        color: str = LABEL_COLOR,
        clip_on: bool = True,
    ) -> list:
        """Fit ``text`` to its cell (``_fit_title_cell_text``) and draw it.

        ``va="center"`` centres the wrapped block on ``y_centre``;
        ``va="baseline"`` puts the first line's baseline at ``y_centre`` and
        wraps downward. Fitting measures pixels at the figure's current size,
        so callers draw through ``_draw_refittable`` to re-fit after an export
        page resize.
        """
        wrapped, value_pt, line_gap = self._fit_title_cell_text(
            ax,
            text,
            x=x,
            y_centre=y_centre,
            cell_right=cell_right,
            row_height=row_height,
            max_chars=max_chars,
            cell_left=cell_left,
            ha=ha,
            fontweight=fontweight,
            base_pt=base_pt,
            floor_pt=floor_pt,
            hard_min_pt=hard_min_pt,
        )
        if va == "center":
            text_top = y_centre + (len(wrapped) - 1) * line_gap * 0.5
        else:
            text_top = y_centre
        artists = []
        for line_index, line in enumerate(wrapped):
            artists.append(
                ax.text(
                    x,
                    text_top - line_index * line_gap,
                    line,
                    fontsize=value_pt,
                    fontweight=fontweight,
                    va=va,
                    ha=ha,
                    color=color,
                    transform=ax.transAxes,
                    clip_on=clip_on,
                )
            )
        return artists

    @staticmethod
    def _refit_key(ax) -> tuple:
        return (tuple(ax.figure.get_size_inches()), tuple(ax.get_position().bounds))

    def _draw_refittable(self, ax, draw, *, key_extra=None) -> None:
        """Run ``draw`` (a measured, fitted drawing unit) and record it for re-fit.

        Fitting measures pixels at the figure's current size and margins. An
        export page resize (letter portrait is narrower than the 11×8.5 render
        size) or a margin change would leave the fitted text too wide, so the
        artists ``draw`` adds are recorded and ``refit_consulting_fitted_text``
        removes and redraws them once the export page geometry is applied.
        ``key_extra`` adds state outside ``ax`` the unit depends on (the scale
        bar follows the main axes' position and x limits).
        """
        before = {id(child) for child in ax.get_children()}
        draw()
        artists = [child for child in ax.get_children() if id(child) not in before]
        registry = getattr(ax.figure, "_css_fitted_cells", None)
        if registry is None:
            registry = ax.figure._css_fitted_cells = []
        key_fn = (lambda: (self._refit_key(ax), key_extra())) if key_extra else (lambda: self._refit_key(ax))
        registry.append({"ax": ax, "draw": draw, "artists": artists, "key": key_fn(), "key_fn": key_fn})

    def refit_consulting_fitted_text(self, figure: Figure) -> None:
        """Redo every recorded fitted unit whose axes moved or resized.

        Called from the export preparation pass; a unit whose axes is still
        where it was fitted is left untouched (same artists, same pixels).
        """
        registry = getattr(figure, "_css_fitted_cells", None)
        if not registry:
            return
        for unit in registry:
            ax = unit["ax"]
            if ax.figure is not figure:
                continue
            key = unit["key_fn"]() if "key_fn" in unit else self._refit_key(ax)
            if unit["key"] == key:
                continue
            for artist in unit["artists"]:
                try:
                    artist.remove()
                except (ValueError, NotImplementedError):  # pragma: no cover - defensive
                    pass
            before = {id(child) for child in ax.get_children()}
            unit["draw"]()
            unit["artists"] = [child for child in ax.get_children() if id(child) not in before]
            unit["key"] = key

    def fit_consulting_page_margins(self, figure: Figure) -> None:
        """Pull the side margins in when an RL axis label would leave the page.

        At larger ``export_font_size`` (or on the narrower portrait page) the
        y-axis label and tick labels of the main and twin axes need more room
        than the fixed 0.06 / 0.95 margins give. Only moves a margin when text
        is actually off the page, so a sheet that fits is untouched.
        """
        axes = getattr(figure, "_css_main_axes", None)
        if not axes:
            return
        main_ax, twin_ax = axes
        renderer = figure.canvas.get_renderer()
        width_px = figure.bbox.width
        pad_px = 2.0 * figure.dpi / 72.0
        params = figure.subplotpars
        left, right = params.left, params.right
        moved = False
        left_box = main_ax.yaxis.get_tightbbox(renderer)
        # Trigger only when text is actually past the page edge, then pad.
        if left_box is not None and left_box.x0 < 0.0:
            left = min(left + (pad_px - left_box.x0) / width_px, 0.3)
            moved = True
        if twin_ax is not None:
            right_box = twin_ax.yaxis.get_tightbbox(renderer)
            if right_box is not None and right_box.x1 > width_px:
                right = max(right - (right_box.x1 - (width_px - pad_px)) / width_px, 0.7)
                moved = True
        if moved:
            figure.subplots_adjust(left=left, right=right)
        # After any margin change: subplots_adjust resets the band to its grid slot.
        self._reserve_threshold_key_band(figure)
        self._suppress_header_tick_collisions(figure)

    def _title_label_column_need(self, ax, labels: list[str]) -> float | None:
        """Axes-fraction width the bold row labels need (offset + text + pad)."""
        try:
            probe = ax.text(0.0, 0.5, "", fontsize=self._fs(7), fontweight="bold", transform=ax.transAxes)
        except Exception:  # pragma: no cover - defensive
            return None
        try:
            ax_width_px = ax.transAxes.transform((1.0, 0.0))[0] - ax.transAxes.transform((0.0, 0.0))[0]
            widest = 0.0
            for label in labels:
                probe.set_text(label)
                widest = max(widest, probe.get_window_extent().width)
            # Same 1.5 pt cell pad as the fitted values (_fit_title_cell_text).
            pad_px = 1.5 * ax.figure.dpi / 72.0
            return 0.01 + (widest + pad_px) / max(ax_width_px, 1.0)
        except Exception:
            logger.debug("Title label measurement unavailable", exc_info=True)
            return None
        finally:
            probe.remove()

    def _draw_title_block_metadata_table(
        self,
        ax,
        rows: list[tuple[str, str]],
        *,
        panel: tuple[float, float, float, float] | None = None,
    ) -> None:
        if not rows:
            return
        if panel is None:
            table_left, table_bottom, table_width, table_height = 0.35, 0.08, 0.30, 0.84
        else:
            table_left, table_bottom, table_width, table_height = panel
            # Inset slightly so cell rules sit inside the panel border.
            inset = 0.01
            table_left += inset
            table_bottom += inset
            table_width -= 2 * inset
            table_height -= 2 * inset

        # The label column grows with the font scale (capped at the panel
        # share) so "FIGURE NO." at a larger size does not cross the rule.
        scale = self._consulting_font_scale()
        label_col_w = min(0.09 * scale, table_width * 0.28)
        # Measure the widest row label: on the narrower portrait page (or at a
        # large font) the design column is too narrow and the label ran into
        # its value ("DRAWN BY" over "AL"). A column that already fits is kept.
        label_need = self._title_label_column_need(ax, [label for label, _value in rows])
        if label_need is not None and label_need > label_col_w:
            label_col_w = min(label_need, table_width * 0.45)
        value_col_w = table_width - label_col_w
        row_height = table_height / max(len(rows), 1)
        # Character budget from panel fraction; consulting sheets are typically wide.
        max_chars = max(28, int(value_col_w * 170 / scale))

        for index, (label, value) in enumerate(rows):
            row_bottom = table_bottom + (len(rows) - index - 1) * row_height
            ax.plot(
                [table_left, table_left + table_width],
                [row_bottom, row_bottom],
                color="#94A3B8",
                linewidth=0.6,
                transform=ax.transAxes,
                clip_on=False,
            )
            ax.plot(
                [table_left + label_col_w, table_left + label_col_w],
                [row_bottom, row_bottom + row_height],
                color="#94A3B8",
                linewidth=0.6,
                transform=ax.transAxes,
                clip_on=False,
            )
            # Row labels are fitted to their column too: at the design size
            # every label fits at 7 pt and renders exactly as before.
            self._draw_fitted_cell_text(
                ax,
                label,
                x=table_left + 0.01,
                y_centre=row_bottom + row_height * 0.5,
                cell_right=table_left + label_col_w,
                row_height=row_height,
                max_chars=max(4, int(label_col_w * 170 / scale)),
                fontweight="bold",
                hard_min_pt=_TITLE_CELL_HARD_MIN_PT,
            )
            value_x = table_left + label_col_w + 0.012
            self._draw_fitted_cell_text(
                ax,
                str(value),
                x=value_x,
                y_centre=row_bottom + row_height * 0.5,
                cell_right=table_left + table_width,
                row_height=row_height,
                max_chars=max_chars,
            )

        top_y = table_bottom + len(rows) * row_height
        for x_pos in (table_left, table_left + table_width):
            ax.plot(
                [x_pos, x_pos],
                [table_bottom, top_y],
                color="#94A3B8",
                linewidth=0.6,
                transform=ax.transAxes,
                clip_on=False,
            )
        ax.plot(
            [table_left, table_left + table_width],
            [top_y, top_y],
            color="#94A3B8",
            linewidth=0.6,
            transform=ax.transAxes,
            clip_on=False,
        )

