"""Visual theme tokens for cross-section rendering."""

from __future__ import annotations

SURFACE_COLOR = "#2F5D3A"
SKY_FILL_COLOR = "#E8F4FC"
STICK_COLOR = "#1F2937"
TRACK_BORDER_COLOR = "#111827"
TRACK_FILL_COLOR = "#FFFFFF"
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
CONSULTING_SURFACE_COLOR = "#8B6914"
CONSULTING_NM_COLOR = "#64748B"
DEFAULT_CONSULTING_NOTES: tuple[str, ...] = (
    "GROUNDWATER BASED ON GROUNDWATER MONITORING WELL OBSERVATIONS ONLY.",
    "masl DENOTES METRES ABOVE SEA LEVEL.",
)
CONSULTING_SCALE_BAR_M = 30.0

CONSULTING_GW_SERIES_STYLES: dict[str, tuple[str, str, str]] = {
    "2024-05": ("#5EB8FF", "v", "May 2024"),
    "2025-06": ("#003399", "P", "June 2025"),
    "default": (CONSULTING_WATER_COLOR, "v", ""),
}


def consulting_gw_series_style(series_id: str, level_label: str = "") -> tuple[str, str, str]:
    """Return (color, marker, label) for a groundwater series."""
    if series_id in CONSULTING_GW_SERIES_STYLES:
        color, marker, label = CONSULTING_GW_SERIES_STYLES[series_id]
        return color, marker, level_label or label
    return CONSULTING_WATER_COLOR, "v", level_label or series_id


def water_has_multiple_series(water_levels) -> bool:
    """True when more than one GW snapshot series is present."""
    if not water_levels:
        return False
    series_ids = {(getattr(level, "series_id", None) or "default") for level in water_levels}
    return len(series_ids) > 1 or any(s not in {"", "default"} for s in series_ids)


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
