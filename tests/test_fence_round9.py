"""Round-9 fence QA: the fill beside every hole equals its log at close spacings.

The round-8 harness probed at a fixed 5 cm from each hole and reported 26
"violations", all beside holes with matched units on either side. Every one was a
*contact dip*: on the hole edge the fill was the logged unit, but contacts leaving
the hole at 2–10 m/m (a 2 m hole 3 m from a 35 m one; code-only matches logged
10 m apart in elevation) had moved 0.1–0.5 m vertically 5 cm out, so a probe at
the logged elevation read the next unit across the contact. The repo check
(``tests/test_fence_logs_honoured._log_mismatches``) had the same blind spot for
thin units (seeds 12/16/19 of the round-8 generator), so it now separates a dip
(one straight edge from the logged contact on the hole) from a fill that takes
over a logged interval at the hole edge, which stays a hard failure.

This fuzz runs every fence generator with holes 5 cm, 1 m and 10 m apart, plus
fences whose inner holes keep matched units on both sides around an extra lens or
a missing unit.
"""

from __future__ import annotations

import random

import pytest
from shapely.geometry import Polygon

from stratigraphy import GeologicalPolygon, build_stratigraphy, detect_polygon_overlaps
from tests.test_fence_logs_honoured import (
    AT_HOLE_FRACTION,
    EPS_FRACTION,
    _drawn_at,
    _log_mismatches,
)
from tests.test_fence_voids_round7 import _df, _layered_holes, _random_holes

_CODES = ["FILL", "CL", "SA", "GR", "TILL", "BR", "PT"]
_SEQUENCE = ("Fill", "Sand", "Silt", "Clay", "Till", "Bedrock")
SPACINGS = (0.05, 1.0, 10.0)


def _round8_fence(seed: int):
    """The round-8 QA generator: 2–4 holes 3 m / 5 m / 8–60 m apart, 2–35 m deep,
    thin units (from 0.3 m, clipped at TD to a few cm), logging gaps and repeated
    codes, so short holes sit beside deep ones with matched units at very
    different elevations."""
    rng = random.Random(seed)
    holes = []
    x = 0.0
    for index in range(rng.randint(2, 4)):
        if index:
            x += round(rng.choice([3.0, 5.0, rng.uniform(8.0, 60.0)]), 3)
        collar = round(100.0 + rng.uniform(-3.0, 3.0), 3)
        td = rng.choice([2.0, 6.0, 12.0, 20.0, 35.0])
        units = []
        depth = 0.0
        while depth < td - 1e-9 and len(units) < 8:
            top = depth
            if rng.random() < 0.15:
                top = min(td, depth + round(rng.uniform(0.1, 1.0), 2))
            bottom = min(td, top + round(rng.uniform(0.3, td / 2.0), 2))
            if bottom <= top:
                break
            units.append((rng.choice(_CODES[: rng.randint(2, len(_CODES))]), top, bottom))
            depth = bottom
        holes.append((f"H{index}", x, collar, units or [("CL", 0.0, td)]))
    return holes


def _matched_both_sides(seed: int):
    """3–4 holes logging one sequence; inner holes occasionally drop a unit or gain
    a lens, so they keep matched units on both sides around a local change."""
    rng = random.Random(seed)
    count = rng.randint(3, 4)
    holes = []
    for index in range(count):
        inner = 0 < index < count - 1
        collar = round(100.0 + rng.uniform(-3.0, 3.0), 3)
        depth = 0.0
        units = []
        for code in _SEQUENCE:
            if inner and code not in ("Clay", "Bedrock") and rng.random() < 0.25:
                continue
            thickness = round(rng.uniform(0.3, 4.0), 3)
            units.append((code, round(depth, 3), round(depth + thickness, 3)))
            depth += thickness
            if inner and rng.random() < 0.2:
                lens = round(rng.uniform(0.2, 1.5), 3)
                units.append(
                    (
                        rng.choice(["Peat", "Gravel", "Sand"]),
                        round(depth, 3),
                        round(depth + lens, 3),
                    )
                )
                depth += lens
        holes.append((f"H{index}", 0.0, collar, units))
    return holes


_GENERATORS = {
    "round8": _round8_fence,
    "random": lambda seed: _random_holes(seed)[0],
    "layered": _layered_holes,
    "matched_both_sides": _matched_both_sides,
}


def _respaced(holes, spacing: float):
    return [
        (hole, index * spacing, collar, units)
        for index, (hole, _x, collar, units) in enumerate(holes)
    ]


def _assert_fence_honours_logs(holes, allow_pinch_outs: bool) -> None:
    projected = _df(holes)
    polygons = build_stratigraphy(projected, allow_pinch_outs=allow_pinch_outs)
    assert all(p.polygon.is_valid and p.polygon.geom_type == "Polygon" for p in polygons)
    assert detect_polygon_overlaps(polygons) == []
    assert _log_mismatches(projected, polygons) == []


@pytest.mark.parametrize("allow_pinch_outs", [True, False])
@pytest.mark.parametrize("spacing", SPACINGS)
@pytest.mark.parametrize("generator", sorted(_GENERATORS))
def test_fuzz_close_spacings_honour_logs(
    generator: str, spacing: float, allow_pinch_outs: bool
) -> None:
    for seed in range(40):
        holes = _respaced(_GENERATORS[generator](seed), spacing)
        _assert_fence_honours_logs(holes, allow_pinch_outs)


@pytest.mark.parametrize("allow_pinch_outs", [True, False])
def test_fuzz_round8_generator_at_its_own_spacings(allow_pinch_outs: bool) -> None:
    for seed in range(60):
        _assert_fence_honours_logs(_round8_fence(seed), allow_pinch_outs)


# ---------------------------------------------------------------------------
# The reported cases: a dipping contact, not a fill over the logged interval
# ---------------------------------------------------------------------------

# (seed, hole, side, logged code, depth below collar) from the round-8 5 cm harness
# and the round-8 generator under the repo check (seeds 16, 19).
_REPORTED = [
    (2, "H1", -1, "CL", 1.585),
    (8, "H0", 1, "GR", 14.045),
    (10, "H2", 1, "FILL", 17.947),
    (10, "H3", -1, "CL", 5.385),
    (12, "H1", -1, "CL", 1.752),
    (16, "H1", -1, "SA", 1.937),
    (19, "H0", 1, "CL", 34.972),
    (23, "H0", 1, "FILL", 1.073),
    (29, "H0", 1, "FILL", 5.415),
    (35, "H3", -1, "SA", 19.845),
    (36, "H1", 1, "PT", 1.595),
    (38, "H3", -1, "CL", 0.115),
    (43, "H1", -1, "FILL", 19.862),
    (46, "H1", -1, "GR", 1.752),
    (54, "H1", -1, "FILL", 0.562),
]


@pytest.mark.parametrize("allow_pinch_outs", [True, False])
@pytest.mark.parametrize(("seed", "hole", "side", "code", "depth"), _REPORTED)
def test_reported_mismatches_are_logged_on_the_hole_edge(
    seed: int, hole: str, side: int, code: str, depth: float, allow_pinch_outs: bool
) -> None:
    holes = _round8_fence(seed)
    projected = _df(holes)
    polygons = build_stratigraphy(projected, allow_pinch_outs=allow_pinch_outs)
    x = next(hx for name, hx, _c, _u in holes if name == hole)
    collar = next(c for name, _x, c, _u in holes if name == hole)
    neighbour = min(abs(hx - x) for name, hx, _c, _u in holes if (hx - x) * side > 0)
    z = collar - depth
    beside = [p for p in polygons if hole in p.hole_pair]
    assert _drawn_at(beside, x + side * AT_HOLE_FRACTION * neighbour, z) == [code]
    assert _log_mismatches(projected, polygons) == []


# ---------------------------------------------------------------------------
# The dip allowance never hides a fill that takes over a logged interval
# ---------------------------------------------------------------------------

_PAIR = [
    ("A", 0.0, 100.0, [("Sand", 0, 4), ("Clay", 4, 10)]),
    ("B", 10.0, 100.0, [("Sand", 0, 4), ("Clay", 4, 10)]),
]


def _flat_pair_polygons() -> list[GeologicalPolygon]:
    return build_stratigraphy(_df(_PAIR), allow_pinch_outs=True)


def _with_sand_sliver(sliver: Polygon) -> list[GeologicalPolygon]:
    hacked = []
    for item in _flat_pair_polygons():
        if item.lithology_code == "Sand":
            shape = item.polygon.union(sliver)
        else:
            shape = item.polygon.difference(sliver)
        hacked.append(
            GeologicalPolygon(item.lithology_code, shape, item.hole_pair, item.is_pinch_out)
        )
    return hacked


@pytest.mark.parametrize(
    "sliver",
    [
        # On the hole edge: caught by the hole-edge probe even though it does not
        # reach the outer probe (1 cm out) at the sampled depth.
        Polygon([(0.0, 96.0), (0.02, 96.0), (0.0, 95.0)]),
        # Detached 2 mm from the hole: the hole edge is right, but the Clay has
        # vertices between the hole and the probe, so it is not a straight dip.
        Polygon([(0.002, 96.0), (0.03, 96.0), (0.002, 95.0)]),
    ],
)
def test_check_reports_a_foreign_sliver_beside_the_hole(sliver: Polygon) -> None:
    """A Sand sliver pushed into hole A's logged Clay (a clip artefact or a stray
    bend) is reported; the dip allowance does not cover it."""
    assert _log_mismatches(_df(_PAIR), _flat_pair_polygons()) == []
    found = _log_mismatches(_df(_PAIR), _with_sand_sliver(sliver))
    assert ("A", 1, 95.4, "Clay", ["Sand"]) in found


def test_check_reports_a_unit_stolen_on_the_hole_edge() -> None:
    """Hole B's top metre of Clay drawn as Sand along the hole edge itself fails,
    whatever the dip of the contact further out."""
    polygons = _flat_pair_polygons()
    hacked = []
    for item in polygons:
        coords = [
            (x, 95.0 if x == 10.0 and abs(z - 96.0) < 1e-9 else z)
            for x, z in item.polygon.exterior.coords
        ]
        hacked.append(
            GeologicalPolygon(
                item.lithology_code, Polygon(coords), item.hole_pair, item.is_pinch_out
            )
        )
    found = _log_mismatches(_df(_PAIR), hacked)
    assert ("B", -1, 95.4, "Clay", ["Sand"]) in found


def test_check_reports_overlap_at_the_probe() -> None:
    polygons = _flat_pair_polygons()
    extra = GeologicalPolygon(
        "Silt", Polygon([(0, 99), (10, 99), (10, 98), (0, 98)]), ("A", "B"), True
    )
    found = _log_mismatches(_df(_PAIR), [*polygons, extra])
    assert any(drawn == ["Sand", "Silt"] for *_rest, drawn in found)


def test_probe_offsets_are_small() -> None:
    assert AT_HOLE_FRACTION < EPS_FRACTION <= 1e-3
