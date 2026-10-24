"""Round-10 fence QA: a short hole on a hill is judged on the elevation of its TD.

``_extend_short_holes`` decided "short" on depth below collar but builds the
inferred column in elevation. A hole on a hill only a little shallower than its
neighbours (TD 8 m vs 9 m) yet ending 16 m above their bottoms was not extended,
and a white notch opened beneath it between its TD and the neighbours' bases.
Now a hole is short when both neighbours end at least ``max(1 m, 25 % of its TD)``
LOWER in elevation. A hole in a valley (shallower, but ending no higher than its
neighbours) is not short, as before; and when the hill hole stands above the
neighbours' fence altogether, the units it logged near surface are not inferred
again beneath its logged base.
"""

from __future__ import annotations

import random

import pytest
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from stratigraphy import build_stratigraphy, detect_polygon_overlaps
from tests.test_fence_logs_honoured import _log_mismatches
from tests.test_fence_voids_round7 import _df, _pairwise_overlap, _unit_unions

GAP_TOL_M2 = 1e-6


def _units(td: float):
    return [("Sand", 0, 3), ("Clay", 3, td)]


def _hill(td_mid: float, td_neighbours: float, rise: float = 15.0):
    return [
        ("A", 0.0, 100.0, _units(td_neighbours)),
        ("H", 40.0, 100.0 + rise, _units(td_mid)),
        ("B", 80.0, 100.0, _units(td_neighbours)),
    ]


def _cut(polygons, x: float):
    """``(code, top, bottom)`` along the vertical at ``x``, top-down."""
    probe = LineString([(x, -1000.0), (x, 1000.0)])
    runs = []
    for item in polygons:
        hit = item.polygon.intersection(probe)
        if hit.is_empty or hit.length <= 1e-9:
            continue
        runs.append((item.lithology_code, hit.bounds[3], hit.bounds[1]))
    return sorted(runs, key=lambda run: -run[1])


@pytest.mark.parametrize("allow_pinch_outs", [True, False])
@pytest.mark.parametrize(
    ("td_mid", "td_neighbours", "rise"),
    [(8.0, 9.0, 15.0), (8.0, 10.0, 15.0), (8.0, 9.0, 5.0), (8.0, 8.5, 3.0)],
)
def test_hill_hole_leaves_no_notch(
    td_mid: float, td_neighbours: float, rise: float, allow_pinch_outs: bool
) -> None:
    holes = _hill(td_mid, td_neighbours, rise)
    projected = _df(holes)
    before = projected.copy()
    polygons = build_stratigraphy(projected, allow_pinch_outs=allow_pinch_outs)
    assert projected.equals(before)  # hole data untouched
    for item in polygons:
        assert item.polygon.geom_type == "Polygon" and item.polygon.is_valid
    assert detect_polygon_overlaps(polygons) == []
    assert _pairwise_overlap(polygons) == pytest.approx(0.0, abs=1e-6)
    assert _log_mismatches(projected, polygons) == []

    # The union covers ground line down to the shallower neighbour base.
    floor = 100.0 - td_neighbours
    region = Polygon([(0, 100), (40, 100 + rise), (80, 100), (80, floor), (0, floor)])
    assert region.difference(unary_union([p.polygon for p in polygons])).area < GAP_TOL_M2

    # Beneath H the logged Clay continues to the neighbours' base: Sand (logged
    # at surface) is not inferred again under it, so no polygon is a pinch-out.
    assert _cut(polygons, 40.0 + 1e-3)[-1][0] == "Clay"
    assert [code for code, _t, _b in _cut(polygons, 40.0 + 1e-3)] == ["Sand", "Clay"]
    assert not any(p.is_pinch_out for p in polygons)


@pytest.mark.parametrize("allow_pinch_outs", [True, False])
def test_valley_hole_is_not_extended(allow_pinch_outs: bool) -> None:
    """Shallower by depth but ending lower than its neighbours: the fence base
    follows the hole bottoms, nothing is inferred."""
    holes = [
        ("A", 0.0, 100.0, _units(10.0)),
        ("V", 40.0, 85.0, _units(8.0)),  # bottom 77 m, neighbours' 90 m
        ("B", 80.0, 100.0, _units(10.0)),
    ]
    polygons = build_stratigraphy(_df(holes), allow_pinch_outs=allow_pinch_outs)
    assert not any(p.is_pinch_out for p in polygons)
    assert detect_polygon_overlaps(polygons) == []
    covered = unary_union([p.polygon for p in polygons])
    cut = covered.intersection(LineString([(40.0, 0.0), (40.0, 200.0)]))
    assert cut.bounds[1] == pytest.approx(77.0)
    assert cut.bounds[3] == pytest.approx(85.0)


@pytest.mark.parametrize(
    ("td_neighbours", "expected_base"),
    [
        (10.0, 90.0),  # neighbours end 2.5 m lower (> 2 m margin): extended
        (9.5, 90.5),  # exactly the margin (92.5 - max(1, 0.25 * 8)): extended
        (9.0, 92.5),  # only 1.5 m lower: ordinary, the base follows the hole bottom
    ],
)
def test_short_margin_is_judged_on_bottom_elevation(
    td_neighbours: float, expected_base: float
) -> None:
    """H (TD 8 m, collar 0.5 m up, bottom 92.5 m) is short only when both
    neighbours end at least max(1 m, 25 % of 8 m) = 2 m lower in elevation."""
    polygons = build_stratigraphy(_df(_hill(8.0, td_neighbours, rise=0.5)), allow_pinch_outs=True)
    base = unary_union([p.polygon for p in polygons]).intersection(
        LineString([(40.0, 0.0), (40.0, 200.0)])
    )
    assert base.bounds[1] == pytest.approx(expected_base)


@pytest.mark.parametrize("allow_pinch_outs", [True, False])
@pytest.mark.parametrize(("td_mid", "td_neighbours", "rise"), [(8.0, 9.0, 15.0), (8.0, 10.0, 15.0)])
def test_hill_hole_mirror_symmetric(
    td_mid: float, td_neighbours: float, rise: float, allow_pinch_outs: bool
) -> None:
    holes = _hill(td_mid, td_neighbours, rise)
    forward = build_stratigraphy(_df(holes), allow_pinch_outs=allow_pinch_outs)
    backward = build_stratigraphy(_df(holes, mirror_span=80.0), allow_pinch_outs=allow_pinch_outs)
    fwd, rev = _unit_unions(forward, None), _unit_unions(backward, 80.0)
    assert fwd.keys() == rev.keys()
    for key, shape in fwd.items():
        assert shape.symmetric_difference(rev[key]).area < 1e-6, key


# ---------------------------------------------------------------------------
# Fuzz: hilly sections (collars ±15 m), one sequence, some holes stopped short
# ---------------------------------------------------------------------------

_SEQUENCE = ("Fill", "Sand", "Silt", "Clay", "Till", "Bedrock")


def _hilly_holes(seed: int):
    rng = random.Random(seed)
    holes = []
    x = 0.0
    count = rng.randint(3, 5)
    for index in range(count):
        collar = round(100.0 + rng.uniform(-15.0, 15.0), 3)
        depth = 0.0
        units = []
        for code in _SEQUENCE:
            if code not in ("Clay", "Bedrock") and rng.random() < 0.25:
                continue
            thickness = round(rng.uniform(0.5, 5.0), 3)
            units.append((code, round(depth, 3), round(depth + thickness, 3)))
            depth += thickness
        if 0 < index < count - 1 and rng.random() < 0.6:
            # Stop short, but below the top of the Clay every hole logs: two
            # neighbours sharing no unit at all build a wedge-only fence that does
            # not tile whatever the short-hole rule does.
            clay = next(i for i, unit in enumerate(units) if unit[0] == "Clay")
            units = units[: rng.randint(clay + 1, len(units) - 1)]
        holes.append((f"H{index}", x, collar, units))
        x += round(rng.uniform(10.0, 60.0), 3)
    return holes, x


@pytest.mark.parametrize("allow_pinch_outs", [True, False])
@pytest.mark.parametrize("seed", range(40))
def test_fuzz_hilly_short_holes_tile_and_mirror(seed: int, allow_pinch_outs: bool) -> None:
    holes, span = _hilly_holes(seed)
    projected = _df(holes)
    forward = build_stratigraphy(projected, allow_pinch_outs=allow_pinch_outs)
    for item in forward:
        assert item.polygon.geom_type == "Polygon" and item.polygon.is_valid
    assert detect_polygon_overlaps(forward) == []
    assert _pairwise_overlap(forward) == pytest.approx(0.0, abs=1e-6)
    assert _log_mismatches(projected, forward) == []

    covered = unary_union([p.polygon for p in forward])
    for (_a, xa, ca, ua), (_s, xs, cs, us), (_b, xb, cb, ub) in zip(holes, holes[1:], holes[2:]):
        bottom_a, bottom_s, bottom_b = ca - ua[-1][2], cs - us[-1][2], cb - ub[-1][2]
        needed = bottom_s - max(1.0, 0.25 * us[-1][2])
        if bottom_a > needed or bottom_b > needed:
            continue
        floor = max(bottom_a, bottom_b)
        region = Polygon(
            [
                (xa, ca - ua[0][1]),
                (xs, cs - us[0][1]),
                (xb, cb - ub[0][1]),
                (xb, floor),
                (xa, floor),
            ]
        )
        if region.is_valid:
            assert region.difference(covered).area < 1e-4

    backward = build_stratigraphy(_df(holes, mirror_span=span), allow_pinch_outs=allow_pinch_outs)
    fwd, rev = _unit_unions(forward, None), _unit_unions(backward, span)
    assert fwd.keys() == rev.keys()
    for key, shape in fwd.items():
        assert shape.symmetric_difference(rev[key]).area < 1e-4, key
