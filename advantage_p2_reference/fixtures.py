"""Fixture builders for Advantage Phase 2 chloride transects."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from advantage_p2_reference.chlorides import load_chloride_readings
from advantage_p2_reference.lithology_digitized import (
    digitized_lithology,
    digitized_total_depth,
)
from advantage_p2_reference.transects import ADVANTAGE_P2_TRANSECTS, TransectSpec
from models import Collar, Lithology, ParseResult

_ROOT = Path(__file__).resolve().parents[1]
_DATA2 = _ROOT / "data" / "Data2"
_WORKBOOK_BY_TRANSECT: dict[str, Path] = {
    "A_A": _DATA2 / "Cross_Section_Test_AA_Lithology_Chlorides.xlsx",
    "B_B": _DATA2 / "Cross_Section_Test_BB_Lithology_Chlorides.xlsx",
}


def load_lithology_csv(path: Path) -> dict[str, list[Lithology]]:
    """Load per-hole lithology from CSV (hole_id, from_depth, to_depth, lithology_code)."""
    frame = pd.read_csv(path)
    normalized = {str(col).strip().lower(): col for col in frame.columns}
    required = ("hole_id", "from_depth", "to_depth", "lithology_code")
    missing = [key for key in required if key not in normalized]
    if missing:
        raise ValueError(f"Lithology CSV missing columns: {missing}")
    rename = {normalized[key]: key for key in required}
    if "unit_order" in normalized:
        rename[normalized["unit_order"]] = "unit_order"
    frame = frame.rename(columns=rename)

    by_hole: dict[str, list[Lithology]] = {}
    for record in frame.to_dict(orient="records"):
        hole_id = str(record["hole_id"]).strip()
        order_raw = record.get("unit_order")
        unit_order = None
        if order_raw is not None and str(order_raw).strip() not in {"", "nan"}:
            unit_order = int(float(order_raw))
        by_hole.setdefault(hole_id, []).append(
            Lithology(
                hole_id=hole_id,
                from_depth=float(record["from_depth"]),
                to_depth=float(record["to_depth"]),
                lithology_code=str(record["lithology_code"]).strip(),
                unit_order=unit_order,
            )
        )
    for hole_id, rows in by_hole.items():
        rows.sort(key=lambda item: (item.from_depth, item.to_depth))
        if any(row.unit_order is None for row in rows):
            by_hole[hole_id] = [
                Lithology(
                    hole_id=row.hole_id,
                    from_depth=row.from_depth,
                    to_depth=row.to_depth,
                    lithology_code=row.lithology_code,
                    unit_order=index,
                )
                for index, row in enumerate(rows, start=1)
            ]
    return by_hole


def _lithology_from_workbook(path: Path) -> dict[str, list[Lithology]] | None:
    if not path.is_file():
        return None
    try:
        from ingestion import ingest_workbook

        parse_result, _report = ingest_workbook(path)
    except Exception:
        return None
    by_hole: dict[str, list[Lithology]] = {}
    for interval in parse_result.lithologies:
        by_hole.setdefault(interval.hole_id, []).append(interval)
    return by_hole or None


def _lithology_for_hole(
    transect_id: str,
    hole_id: str,
    total_depth: float,
    *,
    csv_by_hole: dict[str, list[Lithology]] | None,
    workbook_by_hole: dict[str, list[Lithology]] | None,
) -> tuple[Lithology, ...]:
    if csv_by_hole and hole_id in csv_by_hole:
        return tuple(csv_by_hole[hole_id])
    if workbook_by_hole and hole_id in workbook_by_hole:
        return tuple(workbook_by_hole[hole_id])
    digitized = digitized_lithology(transect_id, hole_id)
    if digitized:
        return digitized
    # Last-resort single clay stick so rendering never crashes.
    return (
        Lithology(
            hole_id=hole_id,
            from_depth=0.0,
            to_depth=total_depth,
            lithology_code="Clay",
            unit_order=1,
        ),
    )


def build_parse_result(
    transect_id: str,
    *,
    lithology_csv: Path | None = None,
    workbook: Path | None = None,
) -> tuple[TransectSpec, ParseResult]:
    spec = ADVANTAGE_P2_TRANSECTS[transect_id]
    chloride_readings = load_chloride_readings(transect_id=transect_id)
    assert isinstance(chloride_readings, tuple)

    csv_by_hole = load_lithology_csv(lithology_csv) if lithology_csv else None
    workbook_path = workbook or _WORKBOOK_BY_TRANSECT.get(transect_id)
    workbook_by_hole = (
        _lithology_from_workbook(workbook_path) if workbook_path is not None else None
    )

    max_depth = 15.0
    if chloride_readings:
        max_depth = max(float(reading.sample_depth) for reading in chloride_readings)
        max_depth = max(max_depth, 15.0)

    collars: list[Collar] = []
    lithologies: list[Lithology] = []
    for hole_id, easting in zip(spec.hole_ids, spec.profile_eastings, strict=True):
        total = digitized_total_depth(transect_id, hole_id, fallback=max_depth)
        if chloride_readings:
            hole_cl = [r.sample_depth for r in chloride_readings if r.hole_id == hole_id]
            if hole_cl:
                total = max(total, max(hole_cl) + 0.5)
        collars.append(
            Collar(
                hole_id=hole_id,
                easting=easting,
                northing=0.0,
                elevation=635.0,
                total_depth=total,
            )
        )
        lithologies.extend(
            _lithology_for_hole(
                transect_id,
                hole_id,
                total,
                csv_by_hole=csv_by_hole,
                workbook_by_hole=workbook_by_hole,
            )
        )
    parse_result = ParseResult(
        collars=tuple(collars),
        lithologies=tuple(lithologies),
        errors=(),
        environmental_readings=chloride_readings,
    )
    return spec, parse_result
