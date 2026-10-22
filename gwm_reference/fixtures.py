"""Golden fixture builders for EcoVenture GWM reference figures.

Collars, lithology, screens and water levels are digitized from the client
sheets in ``data/pdf_extract/`` (Figs 3–6); lithology is sampled at each
casing, so lens geometry between holes is approximate. Fence contacts remain linear in
``stratigraphy.py`` — CAD sand lenses will not match pixel-for-pixel.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gwm_reference.transects import GWM_TRANSECTS, TransectSpec
from models import Collar, Lithology, ParseResult, ScreenInterval, WaterLevel

# (ground elevation masl, total depth m) digitized from the client sheets in
# ``data/pdf_extract/`` (casing top / bottom against the 1 m elevation grid).
COLLAR_META: dict[str, tuple[float, float]] = {
    "MW18-18": (631.1, 9.0),
    "MW18-06B": (631.5, 10.5),
    "MW18-16": (631.6, 6.0),
    "BH18-05": (631.65, 6.0),
    "MW18-08D": (631.5, 18.0),
    "MW18-24": (632.1, 9.0),
    "MW18-17": (632.7, 9.0),
    "BH18-03": (631.6, 6.0),
    "MW18-20": (632.3, 18.0),
    "BH18-08": (632.3, 9.1),
    "BH18-02": (631.3, 6.0),
    "BH18-07": (633.9, 9.0),
    "BH18-04": (631.4, 6.0),
    "MW18-21": (631.4, 9.0),
    "MW18-19": (631.1, 9.0),
    "BH18-09": (631.35, 9.0),
    "MW18-22": (631.75, 9.0),
    "MW18-23": (632.0, 9.0),
}

# Per-hole lithology (from_depth, to_depth, code) sampled beside each casing on
# the client sheets: a clay matrix with thin topsoil and discontinuous sand /
# sand-and-clay lenses (silt only at the B' end, BH18-08).
_LITHOLOGY_STACKS: dict[str, tuple[tuple[float, float, str], ...]] = {
    "MW18-18": ((0.0, 0.3, "Topsoil"), (0.3, 3.1, "Clay"), (3.1, 4.5, "Sand"), (4.5, 9.0, "Clay")),
    "MW18-06B": ((0.0, 0.3, "Topsoil"), (0.3, 2.6, "Clay"), (2.6, 3.3, "Sand"), (3.3, 10.5, "Clay")),
    "MW18-16": ((0.0, 0.2, "Topsoil"), (0.2, 3.0, "Clay"), (3.0, 4.55, "Sand"), (4.55, 6.0, "Clay")),
    "BH18-05": ((0.0, 0.2, "Topsoil"), (0.2, 3.05, "Clay"), (3.05, 4.85, "Sand"), (4.85, 6.0, "Clay")),
    "MW18-08D": (
        (0.0, 0.2, "Topsoil"),
        (0.2, 2.8, "Clay"),
        (2.8, 3.4, "Sand"),
        (3.4, 4.7, "Sand and Clay"),
        (4.7, 6.5, "Sand"),
        (6.5, 18.0, "Clay"),
    ),
    "MW18-24": ((0.0, 0.3, "Topsoil"), (0.3, 4.5, "Clay"), (4.5, 4.9, "Sand"), (4.9, 9.0, "Clay")),
    "MW18-17": ((0.0, 0.3, "Topsoil"), (0.3, 3.4, "Clay"), (3.4, 4.0, "Sand and Clay"), (4.0, 9.0, "Clay")),
    "BH18-03": ((0.0, 0.2, "Topsoil"), (0.2, 3.3, "Clay"), (3.3, 5.0, "Sand"), (5.0, 6.0, "Clay")),
    "MW18-20": ((0.0, 18.0, "Clay"),),
    "BH18-08": ((0.0, 0.3, "Topsoil"), (0.3, 0.6, "Silt"), (0.6, 9.1, "Clay")),
    "BH18-02": ((0.0, 6.0, "Clay"),),
    "BH18-07": ((0.0, 0.2, "Topsoil"), (0.2, 4.2, "Clay"), (4.2, 7.9, "Sand"), (7.9, 9.0, "Clay")),
    "BH18-04": ((0.0, 1.7, "Clay"), (1.7, 2.8, "Sand"), (2.8, 6.0, "Clay")),
    "MW18-21": ((0.0, 0.5, "Topsoil"), (0.5, 2.25, "Clay"), (2.25, 2.8, "Sand"), (2.8, 9.0, "Clay")),
    "MW18-19": ((0.0, 0.3, "Topsoil"), (0.3, 5.0, "Clay"), (5.0, 6.35, "Sand"), (6.35, 9.0, "Clay")),
    "BH18-09": ((0.0, 0.2, "Topsoil"), (0.2, 4.5, "Clay"), (4.5, 6.05, "Sand"), (6.05, 9.0, "Clay")),
    "MW18-22": ((0.0, 0.2, "Topsoil"), (0.2, 9.0, "Clay")),
    "MW18-23": ((0.0, 0.5, "Topsoil"), (0.5, 1.0, "Sand"), (1.0, 9.0, "Clay")),
}

# Screen intervals (mbgs) — monitoring wells only; read from the screen ticks
# drawn inside each casing on the client sheets.
_SCREEN_INTERVALS: dict[str, tuple[float, float]] = {
    "MW18-18": (3.0, 6.1),
    "MW18-06B": (7.6, 10.3),
    "MW18-16": (3.0, 5.9),
    "MW18-08D": (15.0, 17.9),
    "MW18-24": (3.2, 5.1),
    "MW18-17": (3.0, 6.0),
    "MW18-20": (15.0, 17.6),
    "MW18-21": (2.0, 5.0),
    "MW18-19": (3.0, 6.0),
    "MW18-22": (2.5, 5.5),
    "MW18-23": (1.6, 4.6),
}

# Groundwater elevations (masl) printed on the client sheets. Cyan labels and
# lines are May 2024, blue are June 2025 (legend on every sheet). Boreholes
# (BH18-xx) carry no reading on any sheet. Wells shared between sheets
# (MW18-08D on Figs 3/4, MW18-16 on 3/5, MW18-20 on 4/5) print the same pair on
# each, which cross-checks the per-well assignment.
GW_MASL: dict[str, dict[str, float]] = {
    "MW18-18": {"2024-05": 629.903, "2025-06": 629.870},
    "MW18-06B": {"2024-05": 630.142, "2025-06": 630.451},
    "MW18-16": {"2024-05": 631.378, "2025-06": 630.491},
    "MW18-08D": {"2024-05": 630.971, "2025-06": 629.900},
    "MW18-24": {"2024-05": 631.618, "2025-06": 631.188},
    "MW18-17": {"2024-05": 631.293, "2025-06": 630.868},
    "MW18-20": {"2024-05": 629.311, "2025-06": 628.817},
    "MW18-21": {"2024-05": 630.400, "2025-06": 630.060},
    "MW18-19": {"2024-05": 629.847, "2025-06": 629.911},
    "MW18-22": {"2024-05": 630.874, "2025-06": 630.308},
    "MW18-23": {"2024-05": 631.674, "2025-06": 631.193},
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
