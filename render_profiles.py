"""Cross-section render layout profiles (chart vs Strater-like section sheet)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, Field

LayoutMode = Literal["chart", "section_sheet", "consulting_section"]
YAxisMode = Literal["elevation_rl", "depth_below_collar"]
WaterSymbol = Literal["circle", "triangle", "diamond"]
ScaleBarPosition = Literal["bottom_left", "bottom_right"]
ColumnHeaderDetail = Literal["id_only", "id_rl_td"]
# "black" (default) and "red" print every value in one fixed colour ("red"
# matches the client P2 chloride figures); "threshold" bands green/orange/red.
# "black_red" and "green_red" are two-band exceedance modes: one limit per
# parameter, black (or green) at or below it and red above. A per-reading
# workbook label_color always wins. Blue stays for groundwater.
ChemistryColorMode = Literal["black", "red", "black_red", "green_red", "threshold"]
# Modes that band values by per-parameter limits (coloured dots, a key under
# the distance label, the per-parameter limits editor in Configure).
THRESHOLD_STYLE_MODES: frozenset[str] = frozenset({"black_red", "green_red", "threshold"})
# Two-band exceedance modes: a single limit per parameter (stored as
# ``green_max``; ``orange_max`` is set equal to it and never shown).
TWO_BAND_MODES: frozenset[str] = frozenset({"black_red", "green_red"})


def is_threshold_style_mode(mode: str | None) -> bool:
    """True for the modes that colour values by per-parameter limits."""
    return mode in THRESHOLD_STYLE_MODES


def is_two_band_mode(mode: str | None) -> bool:
    """True for the one-limit exceedance modes (``black_red`` / ``green_red``)."""
    return mode in TWO_BAND_MODES
# How a value label is kept legible over hatched fills. "strip" (default)
# knocks a clean background strip out of the fills beside each column, so
# values read on white with the hatching resuming beyond them.
ChemistryLabelStyle = Literal["strip", "plain", "box", "dot", "stroke"]


# The Configure step's starting bands (green <= 100, orange 100-250, red >
# 250). The app seeds them into its threshold editor; the engine no longer
# applies them to a parameter that has no limit at all (see below).
DEFAULT_CHEMISTRY_THRESHOLD_GREEN_MAX = 100.0
DEFAULT_CHEMISTRY_THRESHOLD_YELLOW_MAX = 250.0
# "No limit set": a parameter with neither limit (no own entry, no sheet-wide
# pair) resolves to NaN limits, which ``render_theme.chemistry_label_color``
# prints black and the threshold key states as "no limit set". A generic
# 100 / 250 band coloured Benzene (mg/L) by chloride-scale limits.
NO_CHEMISTRY_LIMIT = float("nan")


def chemistry_limit_value(limit: float | None) -> float | None:
    """A limit as a float, or ``None`` when it is blank / "no limit set" (NaN)."""
    if limit is None:
        return None
    value = float(limit)
    return None if math.isnan(value) else value


def chemistry_limits_set(green_max: float | None, yellow_max: float | None) -> bool:
    """True when a resolved pair carries real limits (not "no limit set")."""
    return chemistry_limit_value(green_max) is not None and chemistry_limit_value(yellow_max) is not None


def resolved_chemistry_thresholds(
    green_max: float | None, yellow_max: float | None
) -> tuple[float, float]:
    """Threshold-mode limits, one given limit completing the other.

    A given limit is kept; a missing orange limit never falls below the green
    one (green 300 alone gives orange <= 300, i.e. no orange band), and green
    never exceeds orange (orange 50 alone gives green <= 50); the Configure
    defaults fill the other side. With *neither* limit the pair is
    ``NO_CHEMISTRY_LIMIT`` (NaN, NaN): values print black, not by a generic band.
    """
    green_max = chemistry_limit_value(green_max)
    yellow_max = chemistry_limit_value(yellow_max)
    if green_max is None and yellow_max is None:
        return NO_CHEMISTRY_LIMIT, NO_CHEMISTRY_LIMIT
    if yellow_max is not None:
        yellow = float(yellow_max)
        # The bands must not invert: an orange limit alone below the default
        # green limit pulls green down to it, and green above orange is clamped.
        green = float(green_max) if green_max is not None else DEFAULT_CHEMISTRY_THRESHOLD_GREEN_MAX
        return min(green, yellow), yellow
    green = float(green_max) if green_max is not None else DEFAULT_CHEMISTRY_THRESHOLD_GREEN_MAX
    return green, max(green, DEFAULT_CHEMISTRY_THRESHOLD_YELLOW_MAX)


def resolved_chemistry_limits(
    green_max: float | None, yellow_max: float | None, mode: str | None = "threshold"
) -> tuple[float, float]:
    """Limits a colour mode uses, as a ``(green_max, orange_max)`` pair.

    Threshold mode: :func:`resolved_chemistry_thresholds`. The two-band
    exceedance modes keep one limit per parameter — the green (exceedance)
    limit — and return it twice (``orange_max == green_max``, so no orange
    band exists); ``yellow_max`` is ignored there. With no limit the pair is
    ``NO_CHEMISTRY_LIMIT`` (NaN, NaN): values print black, "no limit set".
    """
    if is_two_band_mode(mode):
        limit = chemistry_limit_value(green_max)
        if limit is None:
            return NO_CHEMISTRY_LIMIT, NO_CHEMISTRY_LIMIT
        return float(limit), float(limit)
    return resolved_chemistry_thresholds(green_max, yellow_max)


# Per-parameter threshold limits: parameter name -> (green_max, orange_max),
# each in the parameter's own units (whatever unit the workbook gives). A
# ``None`` limit falls back to the sheet-wide pair, then to the defaults.
ChemistryThresholdPair = tuple[float | None, float | None]
ChemistryThresholdMap = dict[str, ChemistryThresholdPair]


def chemistry_parameter_key(parameter: str) -> str:
    """Lookup key for a parameter name: trimmed, case-folded (``" Chloride "``
    and ``"chloride"`` share one limit set)."""
    return str(parameter or "").strip().casefold()


def _optional_limit(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        value = float(text)
    number = float(value)  # type: ignore[arg-type]
    if number != number:  # NaN from a blank spreadsheet cell
        return None
    if number < 0:
        raise ValueError(f"threshold limits must be >= 0 (got {number:g})")
    return number


def normalise_chemistry_thresholds(
    thresholds: Mapping[str, object] | None,
) -> ChemistryThresholdMap:
    """Clean a per-parameter threshold mapping into ``{name: (green, orange)}``.

    Keys keep their first spelling (trimmed) but collapse case-insensitively;
    values may be ``(green, orange)`` pairs/lists or ``{"green_max": .., "orange_max": ..}``
    (``yellow_max`` is accepted for ``orange_max``). Entries with no limit at
    all are dropped, so an empty editor row never shadows the global pair.
    """
    cleaned: ChemistryThresholdMap = {}
    seen: dict[str, str] = {}
    for raw_name, raw_pair in (thresholds or {}).items():
        name = str(raw_name or "").strip()
        key = chemistry_parameter_key(name)
        if not key:
            continue
        if isinstance(raw_pair, Mapping):
            green = raw_pair.get("green_max")
            orange = raw_pair.get("orange_max", raw_pair.get("yellow_max"))
        else:
            pair = tuple(raw_pair or ())  # type: ignore[arg-type]
            green = pair[0] if len(pair) > 0 else None
            orange = pair[1] if len(pair) > 1 else None
        green_limit, orange_limit = _optional_limit(green), _optional_limit(orange)
        if green_limit is None and orange_limit is None:
            continue
        canonical = seen.setdefault(key, name)
        cleaned[canonical] = (green_limit, orange_limit)
    return cleaned


def chemistry_threshold_pair_for(
    parameter: str,
    thresholds: Mapping[str, ChemistryThresholdPair] | None,
) -> ChemistryThresholdPair:
    """The raw ``(green, orange)`` entry for ``parameter`` (case-insensitive), or ``(None, None)``."""
    key = chemistry_parameter_key(parameter)
    if not key or not thresholds:
        return (None, None)
    for name, pair in thresholds.items():
        if chemistry_parameter_key(name) == key:
            green, orange = (tuple(pair) + (None, None))[:2]
            return (green, orange)
    return (None, None)


def resolved_chemistry_thresholds_for(
    parameter: str,
    thresholds: Mapping[str, ChemistryThresholdPair] | None,
    green_max: float | None = None,
    yellow_max: float | None = None,
) -> tuple[float, float]:
    """Threshold limits for one parameter.

    The parameter's own entry wins limit by limit; a limit it leaves blank
    comes from the sheet-wide pair (``green_max`` / ``yellow_max``), and
    anything still missing from the Configure defaults, with the same
    no-inversion rules as :func:`resolved_chemistry_thresholds`.
    """
    own_green, own_orange = chemistry_threshold_pair_for(parameter, thresholds)
    own_green, own_orange = chemistry_limit_value(own_green), chemistry_limit_value(own_orange)
    green = own_green if own_green is not None else chemistry_limit_value(green_max)
    orange = own_orange if own_orange is not None else chemistry_limit_value(yellow_max)
    return resolved_chemistry_thresholds(green, orange)


def resolved_chemistry_limits_for(
    parameter: str,
    thresholds: Mapping[str, ChemistryThresholdPair] | None,
    green_max: float | None = None,
    yellow_max: float | None = None,
    *,
    mode: str | None = "threshold",
) -> tuple[float, float]:
    """Per-parameter limits for a colour mode (see :func:`resolved_chemistry_limits`).

    Threshold mode is :func:`resolved_chemistry_thresholds_for`. In a two-band
    exceedance mode the parameter's own green limit, else the sheet-wide
    ``green_max``, is the single exceedance limit (returned as an equal
    pair); no Configure default is applied, so a parameter with no limit
    prints black with "no limit set".
    """
    if not is_two_band_mode(mode):
        return resolved_chemistry_thresholds_for(parameter, thresholds, green_max, yellow_max)
    own_green, _own_orange = chemistry_threshold_pair_for(parameter, thresholds)
    own_green = chemistry_limit_value(own_green)
    limit = own_green if own_green is not None else chemistry_limit_value(green_max)
    return resolved_chemistry_limits(limit, None, mode)


def cap_chemistry_parameters(
    parameters: Iterable[str] | None, limit: int | None = None
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(plotted, dropped)`` parameter names for one section.

    Names are trimmed and de-duplicated in the order given (Configure order)
    by :func:`chemistry_parameter_key` (``"Cl"`` and ``" cl"`` are one
    parameter; the first spelling is kept), the same key thresholds and
    readings match on. At most ``limit`` (default
    :data:`models.MAX_CHEMISTRY_PARAMETERS`) are plotted, the rest are named
    in a QA note by the caller.
    """
    if limit is None:
        from models import MAX_CHEMISTRY_PARAMETERS

        limit = MAX_CHEMISTRY_PARAMETERS
    by_key: dict[str, str] = {}
    for name in parameters or ():
        text = str(name).strip()
        if text:
            by_key.setdefault(chemistry_parameter_key(text), text)
    ordered = tuple(by_key.values())
    return ordered[:limit], ordered[limit:]


def chemistry_parameter_cap_note(plotted: Sequence[str], dropped: Sequence[str]) -> str:
    """QA note naming the parameters beyond the per-section cap."""
    from models import MAX_CHEMISTRY_PARAMETERS

    return (
        f"Up to {MAX_CHEMISTRY_PARAMETERS} chemistry parameters per section: plotted "
        f"{', '.join(plotted)}; not plotted: {', '.join(dropped)} "
        f"(the first {MAX_CHEMISTRY_PARAMETERS} in Configure order are drawn)."
    )


def chemistry_thresholds_from_readings(readings: Iterable[object]) -> ChemistryThresholdMap:
    """Per-parameter limits typed in the workbook (Environmental ``green_max`` /
    ``orange_max`` columns): the first row of each parameter that gives a limit
    seeds it, so one filled row per parameter is enough."""
    found: dict[str, list[float | None]] = {}
    names: dict[str, str] = {}
    for reading in readings:
        name = str(getattr(reading, "parameter", "") or "").strip()
        key = chemistry_parameter_key(name)
        if not key:
            continue
        green = getattr(reading, "green_max", None)
        orange = getattr(reading, "orange_max", None)
        if green is None and orange is None:
            continue
        names.setdefault(key, name)
        pair = found.setdefault(key, [None, None])
        if pair[0] is None and green is not None:
            pair[0] = float(green)
        if pair[1] is None and orange is not None:
            pair[1] = float(orange)
    return {names[key]: (pair[0], pair[1]) for key, pair in found.items()}


class CrossSectionRenderProfile(BaseModel, frozen=True):
    layout: LayoutMode = "section_sheet"
    track_width_m: float = Field(
        default=3.0,
        gt=0.0,
        description="Schematic borehole column width on the profile X axis (metres), not casing diameter.",
    )
    auto_fit_track_width: bool = Field(
        default=True,
        description="Narrow columns when adjacent holes are closer than track_width_m to avoid overlap.",
    )
    show_grid: bool = False
    show_ground_surface: bool = True
    show_sky_fill: bool = True
    show_contact_ticks: bool = True
    show_eol_bar: bool = True
    show_track_border: bool = True
    show_centerline: bool = False
    show_column_headers: bool = True
    column_header_detail: ColumnHeaderDetail = "id_only"
    show_track_lithology: bool = True
    show_dual_y_axes: bool = False
    show_report_grid: bool = False
    legend_in_title_block: bool = False
    interpolate_water_table_default: bool = False
    water_interpolate_segments: bool = True
    water_interpolate_across_gaps: bool = False
    show_water_elevation_labels: bool = False
    show_water_legend: bool = False
    # Draw a per-series GW key (marker + line + label) once a sheet has at
    # least this many water series, even with show_water_legend off: several
    # blue series are unreadable without one. 0 disables (consulting sheets
    # carry the series in the title-block legend instead).
    water_series_key_min_series: int = Field(default=0, ge=0)
    show_dry_well_nm: bool = False
    y_axis_mode: YAxisMode = "elevation_rl"
    water_symbol: WaterSymbol = "triangle"
    title_block: bool = True
    show_ve_annotation: bool = False
    show_scale_bar: bool = False
    show_parameter_legend_text: bool = False
    scale_bar_position: ScaleBarPosition = "bottom_left"
    fence_alpha: float = Field(default=0.58, ge=0.0, le=1.0)
    show_overlap_markers: bool = False
    show_overlap_footer: bool = False
    use_consulting_palette: bool = False
    show_pinch_out_legend: bool = True
    compact_water_legend: bool = False
    water_line_solid: bool = False
    consulting_axis_from_zero: bool = False
    show_parameter_markers: bool = False
    show_parameter_labels: bool = True
    parameter_interpolate_segments: bool = False
    parameter_interpolate_across_gaps: bool = False
    parameter_draw_markers: bool = True
    parameter_marker: str = "o"
    parameter_marker_size: float = Field(default=16.0, gt=0.0)
    parameter_draw_leaders: bool = False
    parameter_label_include_units: bool = False
    export_font_family: str = "Arial"
    export_font_size: float = Field(default=8.0, gt=0.0)
    legend_ncol: int = Field(default=1, ge=1, le=3)
    chemistry_color_mode: ChemistryColorMode = "black"
    chemistry_threshold_green_max: float | None = None
    chemistry_threshold_yellow_max: float | None = None
    # Per-parameter limits (parameter -> (green_max, orange_max)); the pair
    # above is the fallback for parameters without their own entry.
    chemistry_thresholds: ChemistryThresholdMap = Field(default_factory=dict)
    chemistry_label_style: ChemistryLabelStyle = "strip"
    x_major_grid_m: float = 10.0
    y_axis_label: str = ""


CHART_PROFILE = CrossSectionRenderProfile(
    layout="chart",
    track_width_m=1.6,
    show_grid=True,
    show_ground_surface=False,
    show_sky_fill=False,
    show_contact_ticks=False,
    show_eol_bar=False,
    show_track_border=False,
    show_centerline=True,
    show_column_headers=False,
    y_axis_mode="elevation_rl",
    water_symbol="circle",
    title_block=False,
    show_ve_annotation=False,
    show_scale_bar=True,
    fence_alpha=0.92,
    show_overlap_markers=True,
    show_overlap_footer=True,
)

SECTION_SHEET_PROFILE = CrossSectionRenderProfile(
    layout="section_sheet",
    track_width_m=3.0,
    show_grid=False,
    show_ground_surface=True,
    show_sky_fill=True,
    show_contact_ticks=True,
    show_eol_bar=True,
    show_track_border=True,
    show_centerline=False,
    show_column_headers=True,
    column_header_detail="id_only",
    y_axis_mode="elevation_rl",
    water_symbol="triangle",
    title_block=True,
    show_ve_annotation=False,
    show_scale_bar=False,
    show_parameter_legend_text=False,
    fence_alpha=0.58,
    show_parameter_markers=True,
    parameter_interpolate_segments=False,
    parameter_marker="o",
    parameter_draw_leaders=False,
    parameter_label_include_units=False,
    legend_ncol=2,
    chemistry_color_mode="black",
    water_series_key_min_series=2,
)

CONSULTING_SECTION_PROFILE = CrossSectionRenderProfile(
    layout="consulting_section",
    track_width_m=1.2,
    show_grid=False,
    show_ground_surface=True,
    show_sky_fill=False,
    show_contact_ticks=False,
    show_eol_bar=False,
    show_track_border=False,
    show_centerline=True,
    show_column_headers=False,
    show_track_lithology=False,
    show_dual_y_axes=True,
    show_report_grid=True,
    legend_in_title_block=True,
    interpolate_water_table_default=True,
    water_interpolate_segments=True,
    water_interpolate_across_gaps=False,
    show_water_elevation_labels=True,
    show_water_legend=True,
    show_dry_well_nm=True,
    y_axis_mode="elevation_rl",
    water_symbol="triangle",
    title_block=True,
    show_ve_annotation=False,
    # Consulting subtitle band includes scale / VE (report chrome).
    show_scale_bar=True,
    show_parameter_legend_text=False,
    fence_alpha=1.0,
    show_overlap_markers=True,
    show_overlap_footer=True,
    use_consulting_palette=True,
    show_pinch_out_legend=False,
    compact_water_legend=True,
    water_line_solid=True,
    consulting_axis_from_zero=True,
    x_major_grid_m=10.0,
    y_axis_label="ELEVATION ABOVE SEA LEVEL (MASL)",
    show_parameter_markers=True,
    show_parameter_labels=True,
    parameter_interpolate_segments=False,
    parameter_marker="o",
    parameter_draw_leaders=False,
    parameter_label_include_units=False,
    legend_ncol=2,
    chemistry_color_mode="black",
)


def profile_for_layout(layout: LayoutMode) -> CrossSectionRenderProfile:
    if layout == "chart":
        return CHART_PROFILE
    if layout == "consulting_section":
        return CONSULTING_SECTION_PROFILE
    return SECTION_SHEET_PROFILE


def profile_with_elevation_mode(
    profile: CrossSectionRenderProfile,
    elevation_mode: str,
) -> CrossSectionRenderProfile:
    y_mode: YAxisMode = (
        "depth_below_collar" if elevation_mode == "relative" else "elevation_rl"
    )
    return profile.model_copy(update={"y_axis_mode": y_mode})
