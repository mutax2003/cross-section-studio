"""Digitized lithology sticks for Advantage P2 Figs 6–7 (client sheet extracts).

Intervals are digitised pixel-by-pixel from ``data/pdf_extract_p2/`` against the
sheet legend swatches (FILL / CLAY / CLAY LOAM / LOAMY SAND / SAND / SANDY CLAY /
SANDY CLAY LOAM / SILTY CLAY). Both sheets plot at 59.05 px/m on each axis
(1:200, no vertical exaggeration), so a depth boundary is read to ~0.05 m.
Boundaries are rounded to the 0.05 m grid the client logs use.

Not a substitute for lab logs — replace via Lithology CSV or workbook when
available (``fixtures.build_parse_result`` prefers those sources).
"""

from __future__ import annotations

from models import Lithology

# hole_id -> ((from, to, code), ...)
_DIGITIZED: dict[str, dict[str, tuple[tuple[float, float, str], ...]]] = {
    "A_A": {
        # Fig 6, west → east. TDs 4.5 / 4.5 / 9.0 / 13.5 / 3.0 / 3.0 / 4.5 m.
        "BH23-10": (
            (0.0, 1.25, "Sandy Clay Loam"),
            (1.25, 4.5, "Clay Loam"),
        ),
        "2017-BH05": (
            (0.0, 2.3, "Sandy Clay"),
            (2.3, 4.5, "Clay"),
        ),
        "2017-BH18": (
            (0.0, 4.5, "Sandy Clay"),
            (4.5, 5.25, "Sand"),
            (5.25, 9.0, "Sandy Clay"),
        ),
        "2017-BH11 / BH24-11": (
            (0.0, 2.0, "Sandy Clay"),
            (2.0, 3.0, "Sand"),
            (3.0, 10.5, "Sandy Clay"),
            (10.5, 13.5, "Silty Clay"),
        ),
        "2017-BH09": ((0.0, 3.0, "Sandy Clay"),),
        "2017-BH10": (
            (0.0, 1.5, "Sandy Clay"),
            (1.5, 3.0, "Sand"),
        ),
        "BH23-07": (
            (0.0, 0.9, "Fill"),
            (0.9, 4.5, "Clay Loam"),
        ),
    },
    "B_B": {
        # Fig 7, south → north. TDs 12 / 15 / 13.5 / 12 / 7.5 / 6 / 6 m.
        "BH24-12 / HA25-02": (
            (0.0, 3.0, "Sand"),
            (3.0, 7.0, "Sandy Clay"),
            (7.0, 7.5, "Sand"),
            (7.5, 12.0, "Sandy Clay"),
        ),
        "BH23-04": (
            (0.0, 1.0, "Fill"),
            (1.0, 2.5, "Sandy Clay"),
            (2.5, 3.0, "Loamy Sand"),
            (3.0, 8.75, "Clay Loam"),
            (8.75, 10.5, "Sandy Clay Loam"),
            (10.5, 15.0, "Clay"),
        ),
        # Same hole as Fig 6 — identical log on both sheets.
        "2017-BH11 / BH24-11": (
            (0.0, 2.0, "Sandy Clay"),
            (2.0, 3.0, "Sand"),
            (3.0, 10.5, "Sandy Clay"),
            (10.5, 13.5, "Silty Clay"),
        ),
        "BH23-03": (
            (0.0, 0.75, "Fill"),
            (0.75, 9.0, "Clay Loam"),
            (9.0, 11.0, "Sand"),
            (11.0, 12.0, "Clay"),
        ),
        "2017-BH12": (
            (0.0, 4.5, "Clay"),
            (4.5, 7.5, "Silty Clay"),
        ),
        "BH24-08": (
            (0.0, 2.5, "Sandy Clay"),
            (2.5, 3.0, "Sand"),
            (3.0, 6.0, "Sandy Clay"),
        ),
        "BH23-01": (
            (0.0, 0.3, "Fill"),
            (0.3, 1.5, "Clay"),
            (1.5, 2.5, "Sand"),
            (2.5, 5.5, "Clay"),
            (5.5, 6.0, "Sand"),
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
