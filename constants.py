"""Shared lithology palette, hatch patterns, and canonical code registry."""

from functools import lru_cache
from dataclasses import dataclass

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


@dataclass(frozen=True)
class LithologyStyle:
    color: str
    hatch: str
    edge_color: str = POLYGON_EDGE_COLOR


@lru_cache(maxsize=64)
def get_lithology_style(lithology_code: str, use_hatch: bool = True) -> LithologyStyle:
    color = USGS_LITHOLOGY_COLORS.get(lithology_code, DEFAULT_LITHOLOGY_COLOR)
    hatch = USGS_LITHOLOGY_HATCHES.get(lithology_code, DEFAULT_LITHOLOGY_HATCH) if use_hatch else ""
    return LithologyStyle(color=color, hatch=hatch)
