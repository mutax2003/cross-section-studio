"""Streamlit-free UI helpers for cross-section display and cache keys."""

from __future__ import annotations

import json
import re

from pipeline import DEFAULT_UNCERTAINTY_SPACING_M
from projection import DEFAULT_OFFSET_WARNING_M

_SVG_HEIGHT_RE = re.compile(r'height="([0-9.]+)', re.IGNORECASE)
_SVG_VIEWBOX_RE = re.compile(r'viewBox="[^"]*\s+[^"]*\s+[^"]*\s+([0-9.]+)"', re.IGNORECASE)


def svg_is_valid(svg_bytes: bytes) -> bool:
    if not svg_bytes:
        return False
    text = svg_bytes.decode("utf-8", errors="replace").strip().lower()
    return "<svg" in text


def svg_display_height(
    svg_bytes: bytes,
    *,
    min_height: int = 420,
    max_height: int = 760,
    default_height: int = 540,
) -> int:
    """Estimate iframe height from SVG attributes to reduce clipping."""
    if not svg_bytes:
        return default_height
    text = svg_bytes.decode("utf-8", errors="replace")
    for pattern in (_SVG_HEIGHT_RE, _SVG_VIEWBOX_RE):
        match = pattern.search(text)
        if match:
            try:
                raw = float(match.group(1))
            except ValueError:
                continue
            # Matplotlib SVG height is often in points; scale for screen display.
            scaled = int(raw * 1.15)
            return max(min_height, min(max_height, scaled))
    return default_height


def workflow_stage(
    *,
    has_upload: bool,
    has_parse_result: bool,
    has_profile: bool,
) -> int:
    """Return workflow step index: 0 upload, 1 validate, 2 configure, 3 profile."""
    if has_profile:
        return 3
    if has_parse_result:
        return 2
    if has_upload:
        return 1
    return 0


def legend_hatch_background(hatch: str) -> str:
    """Return a CSS background-image stack approximating matplotlib hatch chars."""
    if not hatch:
        return "none"
    token = hatch.strip()[0]
    line = "repeating-linear-gradient(0deg, #334155 0 1px, transparent 1px 5px)"
    slash = "repeating-linear-gradient(45deg, #334155 0 1px, transparent 1px 5px)"
    backslash = "repeating-linear-gradient(-45deg, #334155 0 1px, transparent 1px 5px)"
    dot = "radial-gradient(circle, #334155 0.6px, transparent 0.7px)"
    if token in {"/"}:
        return slash
    if token in {"\\", "|"}:
        return backslash
    if token in {"-", "_"}:
        return line
    if token in {".", "o", "O", "*", "+", "x"}:
        return (
            f"{dot}, "
            f"radial-gradient(circle, #334155 0.6px, transparent 0.7px)"
        )
    return slash


def transect_cache_key(
    hole_ids: tuple[str, ...],
    transect_points: tuple[tuple[float, float], ...],
    vertical_exaggeration: float,
    offset_warning_m: float,
    show_hatches: bool,
    show_legend: bool,
    section_title: str,
    interpretation_mode: str = "interpolated",
    allow_pinch_outs: bool = True,
    uncertainty_spacing_m: float = DEFAULT_UNCERTAINTY_SPACING_M,
    uncertainty_offset_m: float = DEFAULT_OFFSET_WARNING_M,
) -> str:
    return json.dumps(
        {
            "holes": hole_ids,
            "points": transect_points,
            "ve": vertical_exaggeration,
            "offset": offset_warning_m,
            "hatches": show_hatches,
            "legend": show_legend,
            "title": section_title,
            "mode": interpretation_mode,
            "pinch_outs": allow_pinch_outs,
            "unc_spacing": uncertainty_spacing_m,
            "unc_offset": uncertainty_offset_m,
        },
        sort_keys=True,
    )
