"""Shared lithology palette, hatch patterns, and canonical code registry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from paths import lithology_styles_path

USGS_LITHOLOGY_COLORS: dict[str, str] = {
    "Sandstone": "#E8C547",
    "Sand": "#F5D76E",
    "Clay": "#9B5E3C",
    "Silt": "#D4B896",
    "Gravel": "#A8A9AD",
    "Bedrock": "#4A4A4A",
    "Limestone": "#E0E4E8",
    "Shale": "#5C6B7A",
    "Silty Clay": "#C4956A",
    "Sandy Clay": "#A0714F",
    "Organics": "#3D5C3A",
    "Drilling Waste": "#7A8B94",
    "Flare Pit Material": "#6D4C41",
    "Topsoil": "#8B6914",
    "Sand and Clay": "#E8B84A",
}

CONSULTING_LITHOLOGY_COLORS: dict[str, str] = {
    "Clay": "#C8C8C8",
    "Sand and Clay": "#E8B84A",
    "Sand": "#F5D76E",
    "Silt": "#D4B896",
    "Topsoil": "#8B6914",
}

# Matplotlib hatch strings (repeat chars for density): / \ | - + x o O . *
USGS_LITHOLOGY_HATCHES: dict[str, str] = {
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
    "Organics": "|||",
    "Drilling Waste": "xx",
    "Flare Pit Material": "**",
    "Topsoil": "...",
    "Sand and Clay": "/.",
}

CANONICAL_LITHOLOGY_CODES = frozenset(USGS_LITHOLOGY_COLORS)
DEFAULT_LITHOLOGY_COLOR = "#B8B8B8"
DEFAULT_LITHOLOGY_HATCH = ".."
HATCH_LINE_COLOR = "#3D3D3D"
POLYGON_EDGE_COLOR = "#2C2C2C"

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
    overrides = _load_lithology_style_overrides()
    if lithology_code in overrides:
        base = overrides[lithology_code]
        if use_hatch:
            return base
        return LithologyStyle(color=base.color, hatch="", edge_color=base.edge_color)
    palette = CONSULTING_LITHOLOGY_COLORS if consulting_palette else USGS_LITHOLOGY_COLORS
    color = palette.get(lithology_code, USGS_LITHOLOGY_COLORS.get(lithology_code, DEFAULT_LITHOLOGY_COLOR))
    hatch = USGS_LITHOLOGY_HATCHES.get(lithology_code, DEFAULT_LITHOLOGY_HATCH) if use_hatch else ""
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
        color = str(entry.get("color", DEFAULT_LITHOLOGY_COLOR))
        hatch = str(entry.get("hatch", DEFAULT_LITHOLOGY_HATCH))
        styles[str(code)] = LithologyStyle(color=color, hatch=hatch)
    return styles


def save_lithology_style_override(lithology_code: str, color: str, hatch: str) -> None:
    path = lithology_styles_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, dict[str, str]] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
    payload[lithology_code] = {"color": color, "hatch": hatch}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _load_lithology_style_overrides.cache_clear()
    get_lithology_style.cache_clear()
