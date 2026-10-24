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
    lithology_styles_path,
)

logger = logging.getLogger(__name__)

_HEX_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")

DEFAULT_LITHOLOGY_COLOR = "#B8B8B8"
# Codes outside the scheme cycle through these so two unknown units on one
# section do not look identical (first entry = the historic default).
FALLBACK_LITHOLOGY_STYLES: tuple[tuple[str, str], ...] = (
    ("#B8B8B8", "++"),
    ("#A3A3A3", "xx"),
    ("#CFCFCF", "//"),
    ("#8F8F8F", "\\\\"),
    ("#DEDEDE", "oo"),
)
DEFAULT_LITHOLOGY_HATCH = "++"  # "+" covers any remaining type
POLYGON_EDGE_COLOR = "#2C2C2C"

# Lithology scheme: client legend 261023. Source workbook
# data/BH Log Lithology Legend.xlsx, generated into
# data/bh_log_lithology_legend.json by scripts/build_lithology_legend.py.
# Every lithology colour and hatch comes from that JSON; no hex is repeated
# here. Each soil group shares one base colour and the hatch marks the
# secondary component: sandy = dots, silty = 45° dashes, loam = "+",
# gravel = stone outlines; Clay, Silt, Clay Loam, Topsoil, Organics, Coal,
# Mudstone and the singles are solid. Hatch values are matplotlib hatch
# strings; hatch_patterns.py draws ".", "/", "+" and "O" as the CAD legend's
# DOTS / DASH / CROSS / GRAVEL marks.


@dataclass(frozen=True)
class LithologyScheme:
    """The bundled lithology scheme (``data/bh_log_lithology_legend.json``)."""

    scheme_id: str
    colors: dict[str, str]
    hatches: dict[str, str]
    legend_order: tuple[str, ...]
    legacy_codes: tuple[str, ...]
    family_hatches: dict[str, str]
    hatch_line_color: str


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


def _read_json_object(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def parse_lithology_scheme(path: Path) -> LithologyScheme:
    """Read a scheme JSON (schema 2; an old flat ``{code: hex}`` map also works)."""
    payload = _read_json_object(path)
    entries = payload.get("lithologies")
    if not isinstance(entries, dict):
        # Schema 1: a flat code -> hex map without hatches.
        entries = {code: {"color": colour} for code, colour in payload.items()}
    colors: dict[str, str] = {}
    hatches: dict[str, str] = {}
    for code, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        colour = _normalize_hex_colour(entry.get("color"))
        if colour is None:
            continue
        colors[str(code)] = colour
        if isinstance(entry.get("hatch"), str):
            hatches[str(code)] = entry["hatch"]
    families = payload.get("hatch_families")
    family_hatches = {
        str(name): str(spec.get("hatch", ""))
        for name, spec in (families.items() if isinstance(families, dict) else ())
        if isinstance(spec, dict)
    }
    order = [str(code) for code in payload.get("legend_order") or () if str(code) in colors]
    legacy = [str(code) for code in payload.get("legacy_codes") or () if str(code) in colors]
    return LithologyScheme(
        scheme_id=str(payload.get("scheme_id") or "unversioned"),
        colors=colors,
        hatches=hatches,
        legend_order=tuple(order or colors),
        legacy_codes=tuple(legacy),
        family_hatches=family_hatches,
        hatch_line_color=_normalize_hex_colour(payload.get("hatch_line_color")) or "#323232",
    )


def parse_bh_log_legend_json(path: Path) -> dict[str, str]:
    """Scheme JSON -> lithology_code -> #RRGGBB."""
    return dict(parse_lithology_scheme(path).colors)


@lru_cache(maxsize=1)
def load_lithology_scheme() -> LithologyScheme:
    """The bundled scheme (client legend 261023)."""
    path = bh_log_lithology_legend_path()
    scheme = parse_lithology_scheme(path)
    if not scheme.colors:
        logger.error("Lithology scheme %s is missing or empty; every code draws grey", path)
    return scheme


LITHOLOGY_SCHEME: LithologyScheme = load_lithology_scheme()
LITHOLOGY_SCHEME_ID: str = LITHOLOGY_SCHEME.scheme_id
# Swatch order of the client legend sheet (legacy codes excluded).
LITHOLOGY_LEGEND_ORDER: tuple[str, ...] = tuple(
    code for code in LITHOLOGY_SCHEME.legend_order if code not in LITHOLOGY_SCHEME.legacy_codes
)
HATCH_LINE_COLOR = LITHOLOGY_SCHEME.hatch_line_color  # mark colour on the CAD legend
HATCH_SANDY = LITHOLOGY_SCHEME.family_hatches.get("sandy", ".")
HATCH_SILTY = LITHOLOGY_SCHEME.family_hatches.get("silty", "/")
HATCH_PLUS = LITHOLOGY_SCHEME.family_hatches.get("loam", "+")
HATCH_GRAVEL = LITHOLOGY_SCHEME.family_hatches.get("gravel", "O")


def _build_lithology_palette() -> dict[str, str]:
    return dict(LITHOLOGY_SCHEME.colors)


def _build_lithology_hatches(palette: dict[str, str]) -> dict[str, str]:
    hatches: dict[str, str] = {}
    for code in palette:
        hatch = LITHOLOGY_SCHEME.hatches.get(code)
        if hatch is None:
            hatch = "" if code.casefold() == "no recovery" else DEFAULT_LITHOLOGY_HATCH
        hatches[code] = hatch
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
WATER_TABLE_OVERLAY_DISCLAIMER = (
    "Groundwater markers/lines are schematic linear connectors between measured "
    "levels — not a potentiometric surface."
)
CHEMISTRY_OVERLAY_DISCLAIMER = (
    "Chemistry fences are schematic depth-matched connectors — not concentration "
    "contours or plume envelopes."
)
DEFAULT_PROFILE_ELEVATION_M = 100.0


@dataclass(frozen=True)
class LithologyStyle:
    color: str
    hatch: str
    edge_color: str = POLYGON_EDGE_COLOR


def _fallback_style_for(lithology_code: str) -> tuple[str, str]:
    key = lithology_code.strip().casefold()
    digest = sum((index + 1) * ord(char) for index, char in enumerate(key))
    return FALLBACK_LITHOLOGY_STYLES[digest % len(FALLBACK_LITHOLOGY_STYLES)]


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
        fallback = _fallback_style_for(lithology_code)
        color = lowered.get(lithology_code.casefold(), fallback[0])
    hatch = ""
    if use_hatch:
        hatch = USGS_LITHOLOGY_HATCHES.get(lithology_code)
        if hatch is None:
            hatch_lookup = {key.casefold(): value for key, value in USGS_LITHOLOGY_HATCHES.items()}
            hatch = hatch_lookup.get(lithology_code.casefold(), fallback[1])
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


def clear_lithology_style_override(lithology_code: str) -> bool:
    """Drop a saved override (any key case) so the code falls back to the scheme."""
    path = lithology_styles_path()
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    keys = [key for key in payload if key.casefold() == lithology_code.casefold()]
    if not keys:
        return False
    for key in keys:
        payload.pop(key, None)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _load_lithology_style_overrides.cache_clear()
    get_lithology_style.cache_clear()
    return True


def has_lithology_style_override(lithology_code: str) -> bool:
    overrides = _load_lithology_style_overrides()
    return any(key.casefold() == lithology_code.casefold() for key in overrides)


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
