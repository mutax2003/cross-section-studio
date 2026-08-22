"""Visual theme tokens for cross-section rendering."""

from __future__ import annotations

SURFACE_COLOR = "#2F5D3A"
SKY_FILL_COLOR = "#E8F4FC"
STICK_COLOR = "#1F2937"
TRACK_BORDER_COLOR = "#111827"
TRACK_FILL_COLOR = "#FFFFFF"
CONSULTING_COLUMN_FILL = "#D0D5DD"
LABEL_COLOR = "#111827"
GRID_COLOR = "#D1D5DB"
WATER_COLOR = "#1D4ED8"
UNCERTAINTY_COLOR = "#FDE68A"
PINCH_OUT_ALPHA = 0.72
FENCE_ALPHA = 0.58
OVERLAP_MARKER_COLOR = "#DC2626"
CONTACT_TICK_COLOR = "#374151"
EOL_BAR_COLOR = "#111827"
FIGURE_BG = "#F8FAFC"
AXES_BG = "#FFFFFF"
CONTACT_TICK_WIDTH = 0.8
REPORT_GRID_COLOR = "#E5E7EB"
REPORT_GRID_ALPHA = 0.85
CONSULTING_FIGURE_BG = "#FFFFFF"
CONSULTING_WATER_COLOR = "#007FFF"
PARAMETER_READING_COLOR = "#EA580C"
PARAMETER_TEXT_COLOR = "#DC2626"
CHEMISTRY_LABEL_BLACK = "#111827"
CHEMISTRY_LABEL_GREEN = "#059669"
CHEMISTRY_LABEL_YELLOW = "#CA8A04"
CHEMISTRY_LABEL_RED = "#DC2626"
PARAMETER_PALETTE: tuple[str, ...] = (
    "#EA580C",
    "#7C3AED",
    "#059669",
    "#DC2626",
    "#2563EB",
)
CONSULTING_SURFACE_COLOR = "#8B6914"
CONSULTING_NM_COLOR = "#64748B"
DEFAULT_CONSULTING_NOTES: tuple[str, ...] = (
    "GROUNDWATER BASED ON GROUNDWATER MONITORING WELL OBSERVATIONS ONLY.",
    "masl DENOTES METRES ABOVE SEA LEVEL.",
)
CONSULTING_SCALE_BAR_M = 30.0

# Prefer Arial for GIS/CAD PDF edit parity; fall back when unavailable.
EXPORT_FONT_FAMILY_DEFAULT = "Arial"
EXPORT_FONT_FALLBACKS: tuple[str, ...] = ("Arial", "Calibri", "DejaVu Sans", "sans-serif")
EXPORT_FONT_SIZE_DEFAULT = 8.0
PARAMETER_MARKER_SIZE_DEFAULT = 16.0

# Up to four GW event / nest series — inverted triangles, blue shades (Wave B).
CONSULTING_GW_BLUE_SHADES: tuple[str, ...] = (
    "#5EB8FF",  # light
    "#007FFF",  # primary
    "#0055CC",  # mid
    "#003399",  # deep
)

CONSULTING_GW_SERIES_STYLES: dict[str, tuple[str, str, str]] = {
    "2024-05": (CONSULTING_GW_BLUE_SHADES[0], "v", "May 2024"),
    "2024-06": (CONSULTING_GW_BLUE_SHADES[2], "v", "June 2024"),
    "2025-06": (CONSULTING_GW_BLUE_SHADES[3], "v", "June 2025"),
    "default": (CONSULTING_WATER_COLOR, "v", ""),
}


def consulting_gw_series_style(
    series_id: str,
    level_label: str = "",
    *,
    series_index: int | None = None,
) -> tuple[str, str, str]:
    """Return (color, marker, label) for a groundwater series.

    Known EcoVenture series ids keep stable colours. Unknown ids cycle through
    ``CONSULTING_GW_BLUE_SHADES`` (all inverted triangles).
    """
    if series_id in CONSULTING_GW_SERIES_STYLES:
        color, marker, label = CONSULTING_GW_SERIES_STYLES[series_id]
        return color, marker, level_label or label
    if series_index is not None:
        shade = CONSULTING_GW_BLUE_SHADES[int(series_index) % len(CONSULTING_GW_BLUE_SHADES)]
        return shade, "v", level_label or series_id
    return CONSULTING_WATER_COLOR, "v", level_label or series_id


def chemistry_label_color(
    value: float,
    mode: str,
    *,
    green_max: float | None = None,
    yellow_max: float | None = None,
) -> str:
    """Return label colour for a chemistry value (black default or G/Y/R thresholds)."""
    if mode != "threshold" or green_max is None or yellow_max is None:
        return CHEMISTRY_LABEL_BLACK
    if value <= green_max:
        return CHEMISTRY_LABEL_GREEN
    if value <= yellow_max:
        return CHEMISTRY_LABEL_YELLOW
    return CHEMISTRY_LABEL_RED


def filter_water_levels_for_plot(
    water_levels,
    selected_series_ids: tuple[str, ...] | list[str] | None = None,
    *,
    max_series: int = 4,
) -> list:
    """Filter water levels to selected series (empty selection = all), capped at ``max_series``."""
    if not water_levels:
        return []
    levels = list(water_levels)
    if selected_series_ids:
        allowed = {str(sid).strip() or "default" for sid in selected_series_ids}
        levels = [
            level
            for level in levels
            if (getattr(level, "series_id", None) or "default") in allowed
        ]
    # Preserve workbook order of first appearance when capping.
    order: list[str] = []
    for level in levels:
        series_id = getattr(level, "series_id", None) or "default"
        if series_id not in order:
            order.append(series_id)
    if len(order) > max_series:
        keep = set(order[:max_series])
        levels = [
            level
            for level in levels
            if (getattr(level, "series_id", None) or "default") in keep
        ]
    return levels


def export_font_rc(
    family: str | None = None,
    size: float | None = None,
) -> dict[str, object]:
    """Matplotlib rcParams for export typography (Wave A drafting defaults)."""
    preferred = (family or EXPORT_FONT_FAMILY_DEFAULT).strip() or EXPORT_FONT_FAMILY_DEFAULT
    families = [preferred, *[f for f in EXPORT_FONT_FALLBACKS if f.lower() != preferred.lower()]]
    return {
        "font.family": "sans-serif",
        "font.sans-serif": families,
        "font.size": float(size if size is not None else EXPORT_FONT_SIZE_DEFAULT),
    }


def water_has_multiple_series(water_levels) -> bool:
    """True only when two or more distinct GW ``series_id`` values are present.

    A single named series (e.g. all rows ``2024-05``) is not multi-series.
    Missing / blank ``series_id`` values normalize to ``"default"``.
    """
    if not water_levels:
        return False
    series_ids = {(getattr(level, "series_id", None) or "default") for level in water_levels}
    return len(series_ids) > 1


def primary_water_depth_by_hole(water_levels) -> dict[str, float]:
    """Map hole_id → depth for the primary GW series (last series_id in workbook order).

    Workbooks typically list older snapshots first; the last distinct ``series_id``
    is treated as the current/primary series for consulting gradient anchors.
    """
    if not water_levels:
        return {}
    by_series: dict[str, list] = {}
    order: list[str] = []
    for level in water_levels:
        series_id = getattr(level, "series_id", None) or "default"
        if series_id not in by_series:
            order.append(series_id)
        by_series.setdefault(series_id, []).append(level)
    primary_series = order[-1]
    return {level.hole_id: level.depth for level in by_series[primary_series]}


def strip_cross_section_prefix(label: str) -> str:
    """Remove a leading CROSS SECTION prefix from a section label."""
    text = label.strip()
    if text.upper().startswith("CROSS SECTION"):
        return text[len("CROSS SECTION") :].strip(" :-")
    return text


def consulting_section_title(label: str | None) -> str:
    """Build display title without doubling a leading CROSS SECTION prefix."""
    text = (label or "").strip()
    if not text:
        return "CROSS SECTION"
    if text.upper().startswith("CROSS SECTION"):
        return text
    return f"CROSS SECTION {text}"
