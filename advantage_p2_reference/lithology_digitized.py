"""Digitized lithology sticks for Advantage P2 Figs 6–7 (PDF extract approximation).

Intervals are hand-traced from ``data/pdf_extract_p2/`` against the client legend
(FILL / CLAY / SAND / …). Not a substitute for lab logs — replace via Lithology CSV
or workbook when available.
"""

from __future__ import annotations

from models import Lithology

# hole_id -> ((from, to, code), ...)
_DIGITIZED: dict[str, dict[str, tuple[tuple[float, float, str], ...]]] = {
    "A_A": {
        "BH23-10": (
            (0.0, 0.4, "Fill"),
            (0.4, 2.0, "Clay"),
            (2.0, 3.5, "Sand"),
            (3.5, 5.5, "Clay Loam"),
            (5.5, 7.0, "Sandy Clay"),
        ),
        "2017-BH05": (
            (0.0, 0.3, "Fill"),
            (0.3, 1.5, "Clay"),
            (1.5, 3.0, "Sand"),
            (3.0, 5.0, "Clay Loam"),
            (5.0, 7.5, "Sandy Clay Loam"),
            (7.5, 9.0, "Silty Clay"),
        ),
        "2017-BH18": (
            (0.0, 0.5, "Fill"),
            (0.5, 2.5, "Clay"),
            (2.5, 5.0, "Sand"),
            (5.0, 8.0, "Clay Loam"),
            (8.0, 11.0, "Sandy Clay"),
            (11.0, 13.0, "Silty Clay"),
        ),
        "2017-BH11 / BH24-11": (
            (0.0, 0.4, "Fill"),
            (0.4, 1.5, "Clay"),
            (1.5, 3.5, "Sand"),
            (3.5, 6.0, "Clay Loam"),
            (6.0, 9.0, "Sandy Clay"),
            (9.0, 12.0, "Sandy Clay Loam"),
            (12.0, 14.5, "Silty Clay"),
        ),
        "2017-BH09": (
            (0.0, 0.3, "Fill"),
            (0.3, 2.0, "Clay"),
            (2.0, 4.0, "Sand"),
            (4.0, 7.0, "Clay Loam"),
            (7.0, 10.0, "Sandy Clay"),
        ),
        "2017-BH10": (
            (0.0, 0.4, "Fill"),
            (0.4, 1.8, "Clay"),
            (1.8, 3.5, "Sand"),
            (3.5, 6.0, "Clay Loam"),
            (6.0, 8.5, "Sandy Clay Loam"),
        ),
        "BH23-07": (
            (0.0, 0.3, "Fill"),
            (0.3, 1.5, "Clay"),
            (1.5, 3.0, "Sand"),
            (3.0, 5.5, "Clay Loam"),
            (5.5, 7.0, "Sandy Clay"),
        ),
    },
    "B_B": {
        "BH24-12 / HA25-02": (
            (0.0, 0.5, "Fill"),
            (0.5, 2.0, "Clay"),
            (2.0, 4.5, "Clay Loam"),
            (4.5, 7.0, "Sand"),
            (7.0, 10.0, "Sandy Clay"),
            (10.0, 12.0, "Silty Clay"),
        ),
        "BH23-04": (
            (0.0, 0.4, "Fill"),
            (0.4, 2.5, "Clay"),
            (2.5, 5.0, "Loamy Sand"),
            (5.0, 8.0, "Clay Loam"),
            (8.0, 11.5, "Sandy Clay"),
            (11.5, 15.0, "Silty Clay"),
        ),
        "2017-BH11 / BH24-11": (
            (0.0, 0.4, "Fill"),
            (0.4, 1.5, "Clay"),
            (1.5, 3.5, "Sand"),
            (3.5, 6.0, "Clay Loam"),
            (6.0, 9.0, "Sandy Clay"),
            (9.0, 12.0, "Sandy Clay Loam"),
            (12.0, 14.5, "Silty Clay"),
        ),
        "BH23-03": (
            (0.0, 0.3, "Fill"),
            (0.3, 2.0, "Clay"),
            (2.0, 4.0, "Sand"),
            (4.0, 7.0, "Clay Loam"),
            (7.0, 10.0, "Sandy Clay Loam"),
        ),
        "2017-BH12": (
            (0.0, 0.4, "Fill"),
            (0.4, 1.8, "Clay"),
            (1.8, 3.5, "Loamy Sand"),
            (3.5, 6.5, "Clay Loam"),
            (6.5, 9.0, "Sandy Clay"),
        ),
        "BH24-08": (
            (0.0, 0.3, "Fill"),
            (0.3, 1.5, "Clay"),
            (1.5, 3.5, "Sand"),
            (3.5, 6.0, "Clay Loam"),
            (6.0, 8.0, "Silty Clay"),
        ),
        "BH23-01": (
            (0.0, 0.3, "Fill"),
            (0.3, 1.2, "Clay"),
            (1.2, 2.5, "Sand"),
            (2.5, 4.5, "Clay Loam"),
            (4.5, 6.0, "Sandy Clay"),
        ),
    },
}


def digitized_lithology(transect_id: str, hole_id: str) -> tuple[Lithology, ...] | None:
    stacks = _DIGITIZED.get(transect_id)
    if not stacks:
        return None
    layers = stacks.get(hole_id)
    if not layers:
        return None
    rows: list[Lithology] = []
    for order, (from_depth, to_depth, code) in enumerate(layers, start=1):
        rows.append(
            Lithology(
                hole_id=hole_id,
                from_depth=from_depth,
                to_depth=to_depth,
                lithology_code=code,
                unit_order=order,
            )
        )
    return tuple(rows)


def digitized_total_depth(transect_id: str, hole_id: str, fallback: float = 15.0) -> float:
    layers = digitized_lithology(transect_id, hole_id)
    if not layers:
        return fallback
    return max(interval.to_depth for interval in layers)
