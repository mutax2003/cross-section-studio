"""Synthetic fixture builders for Advantage Phase 2 chloride transects."""

from __future__ import annotations

from advantage_p2_reference.chlorides import load_chloride_readings
from advantage_p2_reference.transects import ADVANTAGE_P2_TRANSECTS, TransectSpec
from models import Collar, Lithology, ParseResult

# BH Log legend names (from data/bh_log_lithology_legend.json) — same algorithm as
# scripts/build_data2_test_workbook._lithology_stack without requiring client xlsx.
_LEGEND_NAMES: tuple[str, ...] = (
    "Organics",
    "Loam",
    "Sand",
    "Loamy Sand",
    "Sand and Gravel",
    "Clay Loam",
    "Silty Clay Loam",
    "Sandy Clay Loam",
    "Clay",
    "Silty Clay",
    "Sandy Clay",
    "Silt",
    "Silty Loam",
)


def _pick(legend: set[str], *candidates: str) -> str:
    for name in candidates:
        if name in legend:
            return name
    return candidates[0]


def _lithology_stack(hole_id: str, total_depth: float) -> tuple[Lithology, ...]:
    """Data2-style BH Log sticks covering 0 → total_depth (seeded by hole_id)."""
    legend = set(_LEGEND_NAMES)
    topsoil = _pick(legend, "Organics", "Loam")
    sand = _pick(legend, "Sand", "Loamy Sand", "Sand and Gravel")
    transition = _pick(legend, "Clay Loam", "Silty Clay Loam", "Sandy Clay Loam")
    clay = _pick(legend, "Clay", "Silty Clay", "Sandy Clay")
    silt = _pick(legend, "Silt", "Silty Loam")

    seed = sum(ord(ch) for ch in hole_id) % 3
    shallow_sand = 1.5 + 0.5 * seed
    mid = 4.0 + seed
    deep = min(total_depth, 9.0 + seed)

    layers = [
        (0.0, min(0.4, total_depth), topsoil),
        (min(0.4, total_depth), min(shallow_sand, total_depth), sand),
        (min(shallow_sand, total_depth), min(mid, total_depth), transition),
        (min(mid, total_depth), min(deep, total_depth), clay),
    ]
    if deep < total_depth:
        layers.append((deep, total_depth, silt if seed else clay))

    rows: list[Lithology] = []
    order = 1
    for from_depth, to_depth, code in layers:
        if to_depth <= from_depth + 1e-9:
            continue
        rows.append(
            Lithology(
                hole_id=hole_id,
                from_depth=round(from_depth, 2),
                to_depth=round(to_depth, 2),
                lithology_code=code,
                unit_order=order,
            )
        )
        order += 1
    if not rows:
        rows.append(
            Lithology(
                hole_id=hole_id,
                from_depth=0.0,
                to_depth=total_depth,
                lithology_code=clay,
                unit_order=1,
            )
        )
    return tuple(rows)


def build_parse_result(transect_id: str) -> tuple[TransectSpec, ParseResult]:
    spec = ADVANTAGE_P2_TRANSECTS[transect_id]
    chloride_readings = load_chloride_readings(transect_id=transect_id)
    max_depth = 15.0
    if chloride_readings:
        max_depth = max(float(reading.sample_depth) for reading in chloride_readings)
        max_depth = max(max_depth, 15.0)
    collars: list[Collar] = []
    lithologies: list[Lithology] = []
    for hole_id, easting in zip(spec.hole_ids, spec.profile_eastings, strict=True):
        collars.append(
            Collar(
                hole_id=hole_id,
                easting=easting,
                northing=0.0,
                elevation=635.0,
                total_depth=max_depth,
            )
        )
        lithologies.extend(_lithology_stack(hole_id, max_depth))
    parse_result = ParseResult(
        collars=tuple(collars),
        lithologies=tuple(lithologies),
        errors=(),
        environmental_readings=chloride_readings,
    )
    return spec, parse_result
