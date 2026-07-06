"""Golden fixture builders for EcoVenture GWM reference figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from models import Collar, Lithology, ParseResult, ScreenInterval, WaterLevel, subset_parse_result
from gwm_reference.transects import GWM_TRANSECTS, TransectSpec

SILT_HOLES = frozenset(
    {
        "MW18-17",
        "BH18-03",
        "MW18-20",
        "BH18-08",
        "BH18-02",
        "MW18-08D",
    }
)

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

GW_MASL: dict[str, dict[str, float]] = {
    "MW18-18": {"2024-05": 631.618, "2025-06": 631.188},
    "MW18-06B": {"2024-05": 630.451, "2025-06": 630.142},
    "MW18-16": {"2024-05": 629.903, "2025-06": 629.870},
    "BH18-05": {"2024-05": 630.491, "2025-06": 630.971},
    "MW18-08D": {"2024-05": 629.870},
    "MW18-24": {"2024-05": 632.631},
    "MW18-17": {"2024-05": 629.311},
    "BH18-03": {"2024-05": 628.817},
    "MW18-20": {"2024-05": 630.868, "2025-06": 630.971},
    "BH18-08": {"2024-05": 631.293},
    "BH18-02": {"2024-05": 630.200},
    "BH18-07": {"2024-05": 630.400},
    "BH18-04": {"2024-05": 630.491},
    "MW18-21": {"2024-05": 629.311},
    "MW18-19": {"2024-05": 629.847},
    "BH18-09": {"2024-05": 629.911},
    "MW18-22": {"2024-05": 630.308},
    "MW18-23": {"2024-05": 631.674},
}

SERIES_LABELS = {
    "2024-05": "May 2024",
    "2025-06": "June 2025",
}


def _lithology_for_hole(hole_id: str, total_depth: float) -> list[Lithology]:
    rows = [
        Lithology(hole_id=hole_id, from_depth=0.0, to_depth=0.5, lithology_code="Topsoil"),
        Lithology(hole_id=hole_id, from_depth=0.5, to_depth=3.0, lithology_code="Sand"),
        Lithology(hole_id=hole_id, from_depth=3.0, to_depth=8.0, lithology_code="Sand and Clay"),
    ]
    if hole_id in SILT_HOLES:
        rows.append(Lithology(hole_id=hole_id, from_depth=8.0, to_depth=12.0, lithology_code="Silt"))
        rows.append(Lithology(hole_id=hole_id, from_depth=12.0, to_depth=total_depth, lithology_code="Clay"))
    else:
        rows.append(Lithology(hole_id=hole_id, from_depth=8.0, to_depth=total_depth, lithology_code="Clay"))
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
    return tuple(
        ScreenInterval(hole_id=hole_id, from_depth=12.0, to_depth=18.0) for hole_id in hole_ids
    )


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
    """Write a master workbook (survey eastings are placeholders — use fixtures for transect geometry)."""
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
    return path
