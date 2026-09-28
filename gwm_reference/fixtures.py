"""Golden fixture builders for EcoVenture GWM reference figures.

Lithology / screen intervals are hand-digitized approximations from
``data/pdf_extract/`` (Figs 3–6). Fence contacts remain linear in
``stratigraphy.py`` — CAD sand lenses will not match pixel-for-pixel.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gwm_reference.transects import GWM_TRANSECTS, TransectSpec
from models import Collar, Lithology, ParseResult, ScreenInterval, WaterLevel

COLLAR_META: dict[str, tuple[float, float]] = {
    "MW18-18": (632.5, 30.0),
    "MW18-06B": (632.0, 30.0),
    "MW18-16": (631.5, 28.0),
    "BH18-05": (631.8, 28.0),
    "MW18-08D": (631.2, 32.0),
    "MW18-24": (633.0, 30.0),
    "MW18-17": (631.0, 30.0),
    "BH18-03": (630.5, 32.0),
    "MW18-20": (631.6, 30.0),
    "BH18-08": (631.4, 28.0),
    "BH18-02": (631.1, 28.0),
    "BH18-07": (631.3, 30.0),
    "BH18-04": (631.0, 28.0),
    "MW18-21": (630.8, 28.0),
    "MW18-19": (631.2, 30.0),
    "BH18-09": (631.0, 28.0),
    "MW18-22": (631.5, 30.0),
    "MW18-23": (632.0, 28.0),
}

# Per-hole lithology (from_depth, to_depth, code) digitized from PDF extracts.
_LITHOLOGY_STACKS: dict[str, tuple[tuple[float, float, str], ...]] = {
    "MW18-18": (
        (0.0, 0.4, "Topsoil"),
        (0.4, 2.5, "Sand"),
        (2.5, 8.0, "Sand and Clay"),
        (8.0, 30.0, "Clay"),
    ),
    "MW18-06B": (
        (0.0, 0.5, "Topsoil"),
        (0.5, 3.0, "Sand"),
        (3.0, 9.0, "Sand and Clay"),
        (9.0, 30.0, "Clay"),
    ),
    "MW18-16": (
        (0.0, 0.4, "Topsoil"),
        (0.4, 2.0, "Sand"),
        (2.0, 7.5, "Sand and Clay"),
        (7.5, 12.0, "Silt"),
        (12.0, 28.0, "Clay"),
    ),
    "BH18-05": (
        (0.0, 0.5, "Topsoil"),
        (0.5, 2.5, "Sand"),
        (2.5, 8.0, "Sand and Clay"),
        (8.0, 12.0, "Silt"),
        (12.0, 28.0, "Clay"),
    ),
    "MW18-08D": (
        (0.0, 0.4, "Topsoil"),
        (0.4, 4.0, "Sand"),
        (4.0, 10.0, "Sand and Clay"),
        (10.0, 14.0, "Silt"),
        (14.0, 32.0, "Clay"),
    ),
    "MW18-24": (
        (0.0, 0.5, "Topsoil"),
        (0.5, 3.5, "Sand"),
        (3.5, 10.0, "Sand and Clay"),
        (10.0, 30.0, "Clay"),
    ),
    "MW18-17": (
        (0.0, 0.4, "Topsoil"),
        (0.4, 2.0, "Sand"),
        (2.0, 8.0, "Sand and Clay"),
        (8.0, 13.0, "Silt"),
        (13.0, 30.0, "Clay"),
    ),
    "BH18-03": (
        (0.0, 0.5, "Topsoil"),
        (0.5, 5.0, "Sand"),
        (5.0, 10.0, "Sand and Clay"),
        (10.0, 14.0, "Silt"),
        (14.0, 32.0, "Clay"),
    ),
    "BH18-02": (
        (0.0, 0.4, "Topsoil"),
        (0.4, 3.0, "Sand"),
        (3.0, 9.0, "Sand and Clay"),
        (9.0, 13.0, "Silt"),
        (13.0, 28.0, "Clay"),
    ),
    "MW18-20": (
        (0.0, 0.5, "Topsoil"),
        (0.5, 2.5, "Sand"),
        (2.5, 8.0, "Sand and Clay"),
        (8.0, 12.5, "Silt"),
        (12.5, 30.0, "Clay"),
    ),
    "BH18-08": (
        (0.0, 0.4, "Topsoil"),
        (0.4, 2.0, "Sand"),
        (2.0, 7.0, "Sand and Clay"),
        (7.0, 12.0, "Silt"),
        (12.0, 28.0, "Clay"),
    ),
    "BH18-07": (
        (0.0, 0.5, "Topsoil"),
        (0.5, 3.0, "Sand"),
        (3.0, 9.0, "Sand and Clay"),
        (9.0, 30.0, "Clay"),
    ),
    "BH18-04": (
        (0.0, 0.4, "Topsoil"),
        (0.4, 4.0, "Sand"),
        (4.0, 10.0, "Sand and Clay"),
        (10.0, 28.0, "Clay"),
    ),
    "MW18-21": (
        (0.0, 0.5, "Topsoil"),
        (0.5, 2.0, "Sand"),
        (2.0, 8.0, "Sand and Clay"),
        (8.0, 28.0, "Clay"),
    ),
    "MW18-19": (
        (0.0, 0.4, "Topsoil"),
        (0.4, 2.5, "Sand"),
        (2.5, 6.0, "Sand and Clay"),
        (6.0, 9.0, "Sand"),
        (9.0, 14.0, "Silt"),
        (14.0, 30.0, "Clay"),
    ),
    "BH18-09": (
        (0.0, 0.5, "Topsoil"),
        (0.5, 3.0, "Sand and Clay"),
        (3.0, 7.0, "Sand"),
        (7.0, 28.0, "Clay"),
    ),
    "MW18-22": (
        (0.0, 0.4, "Topsoil"),
        (0.4, 3.5, "Sand and Clay"),
        (3.5, 30.0, "Clay"),
    ),
    "MW18-23": (
        (0.0, 0.5, "Topsoil"),
        (0.5, 4.0, "Sand and Clay"),
        (4.0, 28.0, "Clay"),
    ),
}

# Screen intervals (mbgs) — monitoring wells only; approximate from PDF MASL bands.
_SCREEN_INTERVALS: dict[str, tuple[float, float]] = {
    "MW18-18": (7.5, 9.5),
    "MW18-06B": (8.0, 10.0),
    "MW18-16": (9.0, 12.0),
    "MW18-08D": (10.0, 13.0),
    "MW18-24": (8.5, 11.0),
    "MW18-17": (9.5, 12.5),
    "MW18-20": (9.0, 12.0),
    "MW18-21": (10.0, 13.0),
    "MW18-19": (8.0, 11.0),
    "MW18-22": (9.0, 12.0),
    "MW18-23": (8.5, 11.5),
}

GW_MASL: dict[str, dict[str, float]] = {
    "MW18-18": {"2024-05": 631.618, "2025-06": 631.188},
    "MW18-06B": {"2024-05": 630.451, "2025-06": 630.142},
    "MW18-16": {"2024-05": 629.903, "2025-06": 629.870},
    "BH18-05": {"2024-05": 630.491, "2025-06": 630.971},
    "MW18-08D": {"2024-05": 629.870},
    "MW18-24": {"2024-05": 632.631},
    "MW18-17": {"2024-05": 629.311, "2025-06": 629.870},
    "BH18-03": {"2024-05": 628.817},
    "MW18-20": {"2024-05": 630.868, "2025-06": 630.971},
    "BH18-08": {"2024-05": 631.293},
    "BH18-02": {"2024-05": 630.200},
    "BH18-07": {"2024-05": 630.400},
    "BH18-04": {"2024-05": 630.491, "2025-06": 630.491},
    "MW18-21": {"2024-05": 629.311, "2025-06": 630.060},
    "MW18-19": {"2024-05": 629.847, "2025-06": 630.060},
    "BH18-09": {"2024-05": 629.911},
    "MW18-22": {"2024-05": 630.308, "2025-06": 631.193},
    "MW18-23": {"2024-05": 631.674},
}

SERIES_LABELS = {
    "2024-05": "May 2024",
    "2025-06": "June 2025",
}

# Holes with a Silt unit in the digitized stack (compat for older tests).
SILT_HOLES = frozenset(
    hole_id
    for hole_id, layers in _LITHOLOGY_STACKS.items()
    if any(code == "Silt" for _a, _b, code in layers)
)


def survey_eastings_by_hole() -> dict[str, float]:
    """Authoritative profile chainage from GWM transect registry (figures 3–6)."""
    eastings: dict[str, float] = {}
    for spec in GWM_TRANSECTS.values():
        for hole_id, easting in zip(spec.hole_ids, spec.profile_eastings, strict=True):
            eastings.setdefault(hole_id, easting)
    return eastings


SURVEY_EASTINGS = survey_eastings_by_hole()


def _lithology_for_hole(hole_id: str, total_depth: float) -> list[Lithology]:
    stack = _LITHOLOGY_STACKS.get(hole_id)
    if stack is None:
        return [
            Lithology(hole_id=hole_id, from_depth=0.0, to_depth=0.5, lithology_code="Topsoil", unit_order=1),
            Lithology(hole_id=hole_id, from_depth=0.5, to_depth=total_depth, lithology_code="Clay", unit_order=2),
        ]
    rows: list[Lithology] = []
    for order, (from_depth, to_depth, code) in enumerate(stack, start=1):
        clipped_to = min(to_depth, total_depth)
        if clipped_to <= from_depth + 1e-9:
            continue
        rows.append(
            Lithology(
                hole_id=hole_id,
                from_depth=from_depth,
                to_depth=clipped_to,
                lithology_code=code,
                unit_order=order,
            )
        )
    if rows and rows[-1].to_depth < total_depth - 1e-6:
        last = rows[-1]
        rows[-1] = Lithology(
            hole_id=hole_id,
            from_depth=last.from_depth,
            to_depth=total_depth,
            lithology_code=last.lithology_code,
            unit_order=last.unit_order,
        )
    return rows


def _collars_for_transect(spec: TransectSpec) -> tuple[Collar, ...]:
    collars: list[Collar] = []
    for hole_id, easting in zip(spec.hole_ids, spec.profile_eastings, strict=True):
        elevation, total_depth = COLLAR_META[hole_id]
        collars.append(
            Collar(
                hole_id=hole_id,
                easting=easting,
                northing=0.0,
                elevation=elevation,
                total_depth=total_depth,
            )
        )
    return tuple(collars)


def _lithologies_for_holes(hole_ids: tuple[str, ...]) -> tuple[Lithology, ...]:
    rows: list[Lithology] = []
    for hole_id in hole_ids:
        _, total_depth = COLLAR_META[hole_id]
        rows.extend(_lithology_for_hole(hole_id, total_depth))
    return tuple(rows)


def _screens_for_holes(hole_ids: tuple[str, ...]) -> tuple[ScreenInterval, ...]:
    screens: list[ScreenInterval] = []
    for hole_id in hole_ids:
        interval = _SCREEN_INTERVALS.get(hole_id)
        if interval is None:
            continue
        from_depth, to_depth = interval
        screens.append(ScreenInterval(hole_id=hole_id, from_depth=from_depth, to_depth=to_depth))
    return tuple(screens)


def _water_for_holes(hole_ids: tuple[str, ...]) -> tuple[WaterLevel, ...]:
    levels: list[WaterLevel] = []
    for hole_id in hole_ids:
        elevation, _ = COLLAR_META[hole_id]
        for series_id, gw_masl in GW_MASL.get(hole_id, {}).items():
            levels.append(
                WaterLevel(
                    hole_id=hole_id,
                    depth=max(0.0, elevation - gw_masl),
                    series_id=series_id,
                    series_label=SERIES_LABELS.get(series_id, series_id),
                )
            )
    return tuple(levels)


def build_parse_result(transect_id: str) -> tuple[TransectSpec, ParseResult]:
    spec = GWM_TRANSECTS[transect_id]
    parse_result = ParseResult(
        collars=_collars_for_transect(spec),
        lithologies=_lithologies_for_holes(spec.hole_ids),
        errors=(),
        water_levels=_water_for_holes(spec.hole_ids),
        screen_intervals=_screens_for_holes(spec.hole_ids),
    )
    return spec, parse_result


def build_subset(transect_id: str) -> tuple[TransectSpec, ParseResult]:
    return build_parse_result(transect_id)


def write_fixture_workbook(path: Path) -> Path:
    """Write master workbook (collar coordinates are synthetic UTM; transect chainage is in GWM_TRANSECTS)."""
    all_hole_ids = tuple(COLLAR_META.keys())
    collars = tuple(
        Collar(
            hole_id=hole_id,
            easting=float(index * 40),
            northing=0.0,
            elevation=COLLAR_META[hole_id][0],
            total_depth=COLLAR_META[hole_id][1],
        )
        for index, hole_id in enumerate(all_hole_ids)
    )
    lithologies = _lithologies_for_holes(all_hole_ids)
    screens = _screens_for_holes(all_hole_ids)
    water = _water_for_holes(all_hole_ids)

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                {
                    "hole_id": c.hole_id,
                    "easting": c.easting,
                    "northing": c.northing,
                    "elevation": c.elevation,
                    "total_depth": c.total_depth,
                }
                for c in collars
            ]
        ).to_excel(writer, sheet_name="Collars", index=False)
        pd.DataFrame(
            [
                {
                    "hole_id": item.hole_id,
                    "from_depth": item.from_depth,
                    "to_depth": item.to_depth,
                    "lithology_code": item.lithology_code,
                    "unit_order": item.unit_order,
                }
                for item in lithologies
            ]
        ).to_excel(writer, sheet_name="Lithology", index=False)
        pd.DataFrame(
            [
                {
                    "hole_id": item.hole_id,
                    "from_depth": item.from_depth,
                    "to_depth": item.to_depth,
                }
                for item in screens
            ]
        ).to_excel(writer, sheet_name="Screens", index=False)
        pd.DataFrame(
            [
                {
                    "hole_id": item.hole_id,
                    "depth": item.depth,
                    "series_id": item.series_id,
                    "series_label": item.series_label,
                }
                for item in water
            ]
        ).to_excel(writer, sheet_name="Water", index=False)
        pd.DataFrame(
            [
                {"field": "client_name", "value": "C-GROUP ENERGY INC."},
                {"field": "prepared_by", "value": "ECOVENTURE"},
                {"field": "figure_preset", "value": "gwm_fence"},
                {"field": "vertical_exaggeration", "value": "5"},
            ]
        ).to_excel(writer, sheet_name="Project", index=False)
    return path
