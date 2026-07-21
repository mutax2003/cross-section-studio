"""Shared lithology palette, hatch patterns, and canonical code registry."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from paths import (
    bh_log_lithology_legend_path,
    bh_log_lithology_legend_xlsx_path,
    lithology_styles_path,
)

logger = logging.getLogger(__name__)

_HEX_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")

# Fallback colours for codes not listed in the BH Log legend workbook.
_BASE_LITHOLOGY_COLORS: dict[str, str] = {
    "Bedrock": "#4A4A4A",
    "Limestone": "#E0E4E8",
    "Shale": "#5C6B7A",
    "Flare Pit Material": "#6D4C41",
    "Topsoil": "#8B6914",
    "Sand and Clay": "#E8B84A",
}

DEFAULT_LITHOLOGY_COLOR = "#B8B8B8"
DEFAULT_LITHOLOGY_HATCH = ".."
HATCH_LINE_COLOR = "#3D3D3D"
POLYGON_EDGE_COLOR = "#2C2C2C"

# Matplotlib hatch strings (repeat chars for density): / \ | - + x o O . *
_BASE_LITHOLOGY_HATCHES: dict[str, str] = {
    "Sandstone": "...",
    "Sand": "...",
    "Clay": "---",
    "Silt": "///",
    "Gravel": "+++",
    "Bedrock": "xxx",
    "Limestone": "..",
    "Shale": "\\\\",
    "Silty Clay": "ooo",
    "Sandy Clay": "/.",
    "Sandy Clay Loam": "/.",
    "Clay Loam": "---",
    "Silty Clay Loam": "ooo",
    "Loamy Sand": "...",
    "Loam": "..",
    "Silty Loam": "///",
    "Sand and Gravel": "+++",
    "Organics": "|||",
    "Drilling Waste": "xx",
    "Flare Pit Material": "**",
    "Topsoil": "...",
    "Sand and Clay": "/.",
    "Siltstone": "\\\\",
    "Mudstone": "xx",
    "Fill": "..",
    "Bentonite": "--",
    "Coal": "xxx",
    "Refuse": "|||",
    "Other": "..",
    "No Recovery": "",
}


def _normalize_hex_colour(value: object) -> str | None:
    colour = str(value).strip()
    if not colour or colour.lower() in {"nan", "none"}:
        return None
    if not colour.startswith("#"):
        colour = f"#{colour}"
    colour = colour.upper()
    if not _HEX_PATTERN.match(colour):
        return None
    return colour


normalize_hex_colour = _normalize_hex_colour


def parse_bh_log_legend_xlsx(path: Path) -> dict[str, str]:
    """Parse BH Log Lithology Legend.xlsx into lithology_code → #RRGGBB."""
    try:
        import pandas as pd
    except ImportError:
        return {}
    if not path.exists():
        return {}
    try:
        frame = pd.read_excel(path, header=1)
    except (OSError, ValueError):
        return {}
    if frame.shape[1] < 3:
        return {}
    frame = frame.iloc[:, :4].copy()
    frame.columns = ["colour", "unused", "lithology", "rgb"][: frame.shape[1]]
    if "lithology" not in frame.columns or "colour" not in frame.columns:
        return {}
    frame = frame.dropna(subset=["lithology"])
    colors: dict[str, str] = {}
    for _, row in frame.iterrows():
        lithology = str(row["lithology"]).strip()
        if not lithology or lithology.casefold() == "lithology":
            continue
        colour = _normalize_hex_colour(row["colour"])
        if colour is None:
            continue
        colors[lithology] = colour
    return colors


def parse_bh_log_legend_json(path: Path) -> dict[str, str]:
    """Parse cached BH Log legend JSON into lithology_code → #RRGGBB."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(payload, dict):
        return {}
    colors: dict[str, str] = {}
    for code, colour in payload.items():
        normalized = _normalize_hex_colour(colour)
        if normalized is None:
            continue
        colors[str(code)] = normalized
    return colors


@lru_cache(maxsize=1)
def _load_bh_log_lithology_colors() -> dict[str, str]:
    """Prefer live Excel legend; fall back to bundled JSON cache."""
    xlsx_colors = parse_bh_log_legend_xlsx(bh_log_lithology_legend_xlsx_path())
    if xlsx_colors:
        return xlsx_colors
    return parse_bh_log_legend_json(bh_log_lithology_legend_path())


def _build_lithology_palette() -> dict[str, str]:
    palette = dict(_BASE_LITHOLOGY_COLORS)
    palette.update(_load_bh_log_lithology_colors())
    return palette


def _build_lithology_hatches(palette: dict[str, str]) -> dict[str, str]:
    hatches = dict(_BASE_LITHOLOGY_HATCHES)
    for code in palette:
        if code in hatches:
            continue
        hatches[code] = "" if code.casefold() == "no recovery" else DEFAULT_LITHOLOGY_HATCH
    return hatches


USGS_LITHOLOGY_COLORS: dict[str, str] = _build_lithology_palette()
CONSULTING_LITHOLOGY_COLORS: dict[str, str] = USGS_LITHOLOGY_COLORS
USGS_LITHOLOGY_HATCHES: dict[str, str] = _build_lithology_hatches(USGS_LITHOLOGY_COLORS)

CANONICAL_LITHOLOGY_CODES = frozenset(USGS_LITHOLOGY_COLORS)

BOREHOLE_ONLY_DISCLAIMER = "Observed borehole data only — no inter-borehole correlation."
INTERPOLATED_DISCLAIMER = (
    "Interpreted fence diagram — contacts are linear between adjacent boreholes."
)
CORRELATION_LINES_DISCLAIMER = (
    "Contact lines only — linear contacts between adjacent boreholes (no unit fill)."
)
DEFAULT_PROFILE_ELEVATION_M = 100.0


@dataclass(frozen=True)
class LithologyStyle:
    color: str
    hatch: str
    edge_color: str = POLYGON_EDGE_COLOR


@lru_cache(maxsize=256)
def get_lithology_style(
    lithology_code: str,
    use_hatch: bool = True,
    *,
    consulting_palette: bool = False,
) -> LithologyStyle:
    del consulting_palette  # consulting and USGS share BH Log palette
    overrides = _load_lithology_style_overrides()
    base = overrides.get(lithology_code)
    if base is None:
        override_lookup = {key.casefold(): value for key, value in overrides.items()}
        base = override_lookup.get(lithology_code.casefold())
    if base is not None:
        if use_hatch:
            return base
        return LithologyStyle(color=base.color, hatch="", edge_color=base.edge_color)
    palette = USGS_LITHOLOGY_COLORS
    color = palette.get(lithology_code)
    if color is None:
        lowered = {key.casefold(): value for key, value in palette.items()}
        color = lowered.get(lithology_code.casefold(), DEFAULT_LITHOLOGY_COLOR)
    hatch = ""
    if use_hatch:
        hatch = USGS_LITHOLOGY_HATCHES.get(lithology_code)
        if hatch is None:
            hatch_lookup = {key.casefold(): value for key, value in USGS_LITHOLOGY_HATCHES.items()}
            hatch = hatch_lookup.get(lithology_code.casefold(), DEFAULT_LITHOLOGY_HATCH)
    return LithologyStyle(color=color, hatch=hatch)


@lru_cache(maxsize=1)
def _load_lithology_style_overrides() -> dict[str, LithologyStyle]:
    path = lithology_styles_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    styles: dict[str, LithologyStyle] = {}
    for code, entry in payload.items():
        if not isinstance(entry, dict):
            continue
        color = _normalize_hex_colour(entry.get("color")) or DEFAULT_LITHOLOGY_COLOR
        hatch = str(entry.get("hatch", DEFAULT_LITHOLOGY_HATCH))
        styles[str(code)] = LithologyStyle(color=color, hatch=hatch)
    return styles


def save_lithology_style_override(lithology_code: str, color: str, hatch: str) -> None:
    path = lithology_styles_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, dict[str, str]] = {}
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                payload = {}
        normalized = _normalize_hex_colour(color)
        if normalized is None:
            raise ValueError(f"Invalid lithology color {color!r}; expected #RRGGBB")
        payload[lithology_code] = {"color": normalized, "hatch": hatch}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except ValueError:
        raise
    except (OSError, PermissionError) as exc:
        logger.warning("Failed to save lithology style override to %s: %s", path, exc)
        raise RuntimeError(
            f"Could not save lithology style override to {path}: {exc}"
        ) from exc
    _load_lithology_style_overrides.cache_clear()
    get_lithology_style.cache_clear()
