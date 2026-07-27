"""Synthetic Advantage Phase 2 field-export workbook for CI (no client xlsx)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyproj import Transformer

# 23 holes; BH26-13 and BH26-15 share WGS84 so override offset separates them.
# Total lithology rows must stay at 70 to match conversion/ingest asserts.
_HOLE_IDS = tuple(f"BH26-{index:02d}" for index in range(1, 24))
_SHARED_LAT_LONG = (58.5700, -119.1880)

# BH26-01: 4 intervals including Silty Clay; all others: 3 intervals → 4 + 22*3 = 70.
_LITHOLOGY_STACKS: dict[str, tuple[tuple[str, str], ...]] = {
    "BH26-01": (
        ("0.00-0.50m", "Topsoil"),
        ("0.50-2.00m", "Sand"),
        ("2.00-5.00m", "Silty Clay"),
        ("5.00-12.00m", "Clay"),
    ),
}


def _default_stack(hole_id: str) -> tuple[tuple[str, str], ...]:
    seed = sum(ord(ch) for ch in hole_id) % 3
    if seed == 0:
        return (
            ("0.00-0.40m", "Topsoil"),
            ("0.40-2.50m", "Sand"),
            ("2.50-11.00m", "Clay"),
        )
    if seed == 1:
        return (
            ("0.00-0.30m", "Organics"),
            ("0.30-1.80m", "Sand"),
            ("1.80-10.00m", "Clay"),
        )
    return (
        ("0.00-0.50m", "Topsoil"),
        ("0.50-3.00m", "Sand"),
        ("3.00-13.00m", "Clay Loam"),
    )


def _lat_long_for_hole(hole_id: str) -> tuple[float, float]:
    if hole_id in {"BH26-13", "BH26-15"}:
        return _SHARED_LAT_LONG
    index = int(hole_id.split("-")[1])
    # Spread holes ~50–100 m apart in projected space via small lon/lat steps.
    lat = 58.5670 + (index - 1) * 0.00035
    lon = -119.1925 + (index - 1) * 0.00055
    return lat, lon


def write_advantage_source_workbook(path: Path) -> Path:
    """Write field-export Lithology sheet consumed by ``field_export_v1``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for hole_id in _HOLE_IDS:
        lat, lon = _lat_long_for_hole(hole_id)
        stack = _LITHOLOGY_STACKS.get(hole_id, _default_stack(hole_id))
        for depth, code in stack:
            rows.append(
                {
                    "Label": hole_id,
                    "Depth": depth,
                    "Lithology": code,
                    "Lat": lat,
                    "Long": lon,
                }
            )
    frame = pd.DataFrame(rows)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Lithology", index=False)
    return path


def ensure_advantage_source_workbook(path: Path) -> Path:
    if path.exists():
        return path
    return write_advantage_source_workbook(path)


def expected_lithology_row_count() -> int:
    total = 0
    for hole_id in _HOLE_IDS:
        total += len(_LITHOLOGY_STACKS.get(hole_id, _default_stack(hole_id)))
    return total


def verify_projected_bbox() -> None:
    """Dev helper: confirm synthetic WGS84 lands in test easting/northing windows."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32611", always_xy=True)
    for hole_id in _HOLE_IDS:
        lat, lon = _lat_long_for_hole(hole_id)
        easting, northing = transformer.transform(lon, lat)
        assert 372000 <= easting <= 373500, (hole_id, easting)
        assert 6493000 <= northing <= 6495000, (hole_id, northing)
