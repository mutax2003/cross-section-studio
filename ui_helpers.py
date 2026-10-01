"""Streamlit-free UI helpers for cross-section display and cache keys."""

from __future__ import annotations

import base64
import html
import math
import re
from dataclasses import dataclass
from typing import Sequence

from app_identity import APP_NAME, COPYRIGHT_NOTICE, CREATED_BY
from models import Collar, Lithology, ScreenInterval, Transect
from projection import (
    DEFAULT_OFFSET_WARNING_M,
    select_and_order_holes_near_transect,
)

_SVG_HEIGHT_RE = re.compile(r'height="([0-9.]+)', re.IGNORECASE)
_SVG_VIEWBOX_RE = re.compile(r'viewBox="[^"]*\s+[^"]*\s+[^"]*\s+([0-9.]+)"', re.IGNORECASE)
_SVG_WIDTH_RE = re.compile(r'<svg\b[^>]*?\swidth=["\']([0-9.]+)\s*(pt|px)?["\']', re.IGNORECASE)
_SVG_VIEWBOX_WIDTH_RE = re.compile(
    r'<svg\b[^>]*?\sviewBox=["\'][^"\'\s,]+[\s,]+[^"\'\s,]+[\s,]+([0-9.]+)', re.IGNORECASE
)
_SVG_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# Cap zoomed previews so a bogus SVG width cannot request a gigapixel <img>.
_MAX_PREVIEW_WIDTH_PX = 12000
_PT_TO_PX = 96.0 / 72.0

# Preview zoom choices -> scale of the SVG's natural CSS-pixel width (None = fit).
PREVIEW_ZOOM_OPTIONS: dict[str, float | None] = {"Fit width": None, "100%": 1.0, "150%": 1.5}


@dataclass(frozen=True)
class SvgDisplayMeta:
    valid: bool
    height: int
    encoded: str
    natural_width_px: int = 0


def escape_html(text: str | int | float) -> str:
    """Escape user-controlled strings before embedding in markdown HTML."""
    return html.escape(str(text), quote=True)


def svg_is_valid(svg_bytes: bytes) -> bool:
    return svg_display_meta(svg_bytes).valid


def svg_display_height(
    svg_bytes: bytes,
    *,
    min_height: int = 420,
    max_height: int = 760,
    default_height: int = 540,
) -> int:
    """Estimate iframe height from SVG attributes to reduce clipping."""
    return svg_display_meta(
        svg_bytes,
        min_height=min_height,
        max_height=max_height,
        default_height=default_height,
    ).height


def svg_display_meta(
    svg_bytes: bytes,
    *,
    min_height: int = 420,
    max_height: int = 760,
    default_height: int = 540,
) -> SvgDisplayMeta:
    """Validate SVG and compute display height + base64 in a single decode pass."""
    if not svg_bytes:
        return SvgDisplayMeta(valid=False, height=default_height, encoded="")
    text = svg_bytes.decode("utf-8", errors="replace")
    lowered = text.strip().lower()
    valid = lowered.startswith("<svg") or "<svg" in lowered[:200]
    height = default_height
    for pattern in (_SVG_HEIGHT_RE, _SVG_VIEWBOX_RE):
        match = pattern.search(text)
        if match:
            try:
                raw = float(match.group(1))
            except ValueError:
                continue
            scaled = int(raw * 1.15)
            height = max(min_height, min(max_height, scaled))
            break
    encoded = base64.b64encode(svg_bytes).decode("ascii") if valid else ""
    return SvgDisplayMeta(
        valid=valid,
        height=height,
        encoded=encoded,
        natural_width_px=_svg_natural_width_px(text) if valid else 0,
    )


def _svg_natural_width_px(text: str) -> int:
    """CSS-pixel width the browser gives the SVG at 100% (matplotlib writes pt).

    Returns 0 (fit-to-width) when no usable width can be read.
    """
    text = _SVG_COMMENT_RE.sub("", text)
    match = _SVG_WIDTH_RE.search(text)
    if match:
        scale = _PT_TO_PX if (match.group(2) or "").lower() == "pt" else 1.0
        width = _safe_float(match.group(1))
        if width:
            return min(int(round(width * scale)), _MAX_PREVIEW_WIDTH_PX)
    match = _SVG_VIEWBOX_WIDTH_RE.search(text)
    width = _safe_float(match.group(1)) if match else None
    return min(int(round(width)), _MAX_PREVIEW_WIDTH_PX) if width else 0


def _safe_float(raw: str) -> float | None:
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0 else None


def preview_img_style(zoom: str | None, natural_width_px: int) -> tuple[str, bool]:
    """Return (img inline style, zoomed?) for a preview zoom choice.

    Unknown choices or an unknown natural width fall back to fit-to-width.
    ``object-fit:contain`` overrides Streamlit's markdown ``scale-down`` default,
    which would otherwise draw the SVG at natural size centred inside the
    enlarged box instead of scaling it up.
    """
    scale = PREVIEW_ZOOM_OPTIONS.get(zoom or "")
    if scale is None or natural_width_px <= 0:
        return "width:100%;height:auto;display:block;object-fit:contain;", False
    width = int(round(natural_width_px * scale))
    return f"width:{width}px;max-width:none;height:auto;display:block;object-fit:contain;", True


def workflow_stage(
    *,
    has_upload: bool,
    has_parse_result: bool,
    has_profile: bool,
    has_blocking_errors: bool = False,
    has_transect: bool = False,
) -> int:
    """Return workflow step index: 0 upload, 1 validate, 2 configure, 3 generate.

    Stay on Validate until QA is clear and a transect is selected so the stepper
    matches the main-pane work. Generate (3) requires a live parse — leftover
    SVG alone must not advance the stepper after a failed re-upload.
    """
    if has_profile and has_parse_result and not has_blocking_errors:
        return 3
    if has_parse_result and not has_blocking_errors and has_transect:
        return 2
    if has_parse_result or has_upload:
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
    if token in {"o", "O"}:  # gravel / cobble
        return "radial-gradient(circle, transparent 1.6px, #334155 1.7px 2.4px, transparent 2.5px)"
    if token in {"+", "x"}:
        return f"{line}, repeating-linear-gradient(90deg, #334155 0 1px, transparent 1px 5px)"
    if token in {".", "*"}:
        return (
            f"{dot}, "
            f"radial-gradient(circle, #334155 0.6px, transparent 0.7px)"
        )
    return slash


def parse_coordinate_lines(text: str) -> list[tuple[float, float]]:
    """Parse easting/northing pairs from newline-separated coordinate text."""
    points: list[tuple[float, float]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        parts = re.split(r"[,\s]+", line)
        if len(parts) != 2:
            raise ValueError(f"Line {line_number}: expected 'easting northing' pair")
        try:
            easting = float(parts[0])
            northing = float(parts[1])
        except ValueError as exc:
            raise ValueError(f"Line {line_number}: invalid numeric coordinate") from exc
        if not math.isfinite(easting) or not math.isfinite(northing):
            raise ValueError(f"Line {line_number}: coordinates must be finite numbers")
        points.append((easting, northing))
    if len(points) < 2:
        raise ValueError("At least two coordinate pairs are required")
    return points


def active_transect_selection(
    collars: Sequence[Collar],
    transect_mode: str,
    selected_holes: Sequence[str],
    coordinate_text: str,
    offset_warning_m: float,
) -> tuple[tuple[str, ...], tuple[tuple[float, float], ...]] | None:
    """Resolve active hole IDs and transect polyline from sidebar mode."""
    collar_lookup = {collar.hole_id: collar for collar in collars}
    if transect_mode == "By coordinates":
        try:
            transect_points = tuple(parse_coordinate_lines(coordinate_text))
        except ValueError:
            return None
        transect = Transect(points=list(transect_points))
        ordered_holes = select_and_order_holes_near_transect(
            collars, transect, offset_warning_m
        )
        if len(ordered_holes) < 2:
            return None
        return ordered_holes, transect_points
    if len(selected_holes) < 2:
        return None
    missing = [hole_id for hole_id in selected_holes if hole_id not in collar_lookup]
    if missing:
        return None
    transect_points = tuple(
        (collar_lookup[hole_id].easting, collar_lookup[hole_id].northing)
        for hole_id in selected_holes
    )
    return tuple(selected_holes), transect_points


def dedupe_messages(messages: Sequence[str]) -> tuple[str, ...]:
    """Preserve order while removing duplicate warning strings."""
    seen: set[str] = set()
    unique: list[str] = []
    for message in messages:
        if message in seen:
            continue
        seen.add(message)
        unique.append(message)
    return tuple(unique)


def holes_missing_lithology(
    lithologies: Sequence[Lithology],
    hole_ids: Sequence[str],
) -> tuple[str, ...]:
    """Return hole IDs in the selection that have no lithology intervals."""
    lithology_holes = {lithology.hole_id for lithology in lithologies}
    return tuple(hole_id for hole_id in hole_ids if hole_id not in lithology_holes)


def sanitize_filename(text: str, *, fallback: str = "cross_section") -> str:
    """Return a safe filename stem for downloads."""
    cleaned = re.sub(r"[^\w\-]+", "_", text.strip())[:80].strip("_")
    return cleaned or fallback


def screen_interval_warnings(
    hole_ids: Sequence[str],
    screen_intervals: Sequence[ScreenInterval],
) -> list[str]:
    """Warn when transect holes lack screened-interval rows."""
    if not hole_ids:
        return []
    screened = {interval.hole_id for interval in screen_intervals}
    missing = [hole_id for hole_id in hole_ids if hole_id not in screened]
    if not missing:
        return []
    return [f"No screen interval for transect hole(s): {', '.join(missing)}"]


def build_export_framing_from_mapping(values: dict[str, object]) -> "ExportFramingConfig":
    """Construct export framing from sidebar/session values."""
    from export_framing import ExportFramingConfig

    return ExportFramingConfig(
        page_preset=values.get("export_page_preset", "auto"),  # type: ignore[arg-type]
        margin_top_in=float(values.get("export_margin_top_in", 0.0)),
        margin_bottom_in=float(values.get("export_margin_bottom_in", 0.0)),
        margin_left_in=float(values.get("export_margin_left_in", 0.0)),
        margin_right_in=float(values.get("export_margin_right_in", 0.0)),
        export_dpi=int(values.get("export_dpi", 300)),
        show_draft_watermark=bool(values.get("export_show_draft_watermark", False)),
        include_title_block=bool(values.get("export_include_title_block", True)),
        include_legend=bool(values.get("export_include_legend", True)),
        include_water_table=bool(values.get("export_include_water_table", True)),
        include_qa_footer=bool(values.get("export_include_qa_footer", True)),
        fence_only=bool(values.get("export_fence_only", False)),
        filename_pattern=values.get("export_filename_pattern", "section_title"),  # type: ignore[arg-type]
        export_revision=str(values.get("export_revision", "")),
        viewport_xmin=_optional_float(values.get("export_viewport_xmin")),
        viewport_xmax=_optional_float(values.get("export_viewport_xmax")),
        viewport_ymin=_optional_float(values.get("export_viewport_ymin")),
        viewport_ymax=_optional_float(values.get("export_viewport_ymax")),
        cad_svg_layers=bool(values.get("export_cad_svg_layers", False)),
    )


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def export_metadata_payload(
    *,
    section_title: str,
    preset_label: str | None,
    vertical_exaggeration: float,
    hole_count: int | None,
    transect_label: str | None,
    overlap_warnings: Sequence[str],
    consulting_fields: dict[str, str] | None = None,
) -> dict[str, object]:
    """JSON-serializable metadata for report packages."""
    payload: dict[str, object] = {
        "section_title": section_title,
        "preset": preset_label,
        "vertical_exaggeration": vertical_exaggeration,
        "hole_count": hole_count,
        "transect_label": transect_label,
        "qa_notes": list(overlap_warnings[:20]),
        "generated_by": APP_NAME,
        "created_by": CREATED_BY,
        "copyright": COPYRIGHT_NOTICE,
    }
    if consulting_fields:
        payload.update(consulting_fields)
    return payload
