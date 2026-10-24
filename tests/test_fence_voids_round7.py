"""Round-7 fence QA: voids under short holes, mid-span walls and bow-ties.

* qa_hydro A-A' (gwm_fence): MW-06S (6 m) sits 1.5 m from MW-06D (16 m) between
  MW-05 (15 m) and MW-06D. The fence base line notched up to its TD, leaving a
  white wedge under the nested pair and a sliver of Clay Loam. Units below a
  short hole's TD now continue beneath it, interpolated between the deeper
  neighbours and drawn as inferred (``is_pinch_out``).
* data/sample_boreholes.xlsx (quick_preview, pinch-outs on): opposing base / top
  units met along vertical walls at mid-span (~25 m, ~125 m), and Clay vs Silt
  between BH-02 and BH-03 tapered to one shared tip, pinching Sandstone and
  Bedrock to a point (a bow-tie). Opposing units now share their slot along a
  sloping facies boundary, each thinning to zero at the other hole.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest
from shapely import affinity
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from stratigraphy import GeologicalPolygon, build_stratigraphy, detect_polygon_overlaps

REPO = Path(__file__).resolve().parents[1]
GAP_TOL_M2 = 1e-6

# qa_hydro A-A' around the nested pair: (hole, x, collar, [(code, from, to)]).
_HYDRO = [
    ("MW-04", 126.01, 642.0, [("Topsoil", 0, 0.5), ("Sand", 0.5, 4), ("Clay Loam", 4, 9), ("Clay", 9, 15)]),
    ("MW-05", 166.8, 641.0, [("Topsoil", 0, 0.5), ("Sand", 0.5, 4), ("Clay Loam", 4, 9), ("Clay", 9, 15)]),
    ("MW-06S", 206.8, 640.0, [("Topsoil", 0, 0.5), ("Sand", 0.5, 4), ("Clay Loam", 4, 6)]),
    ("MW-06D", 208.3, 640.0, [("Topsoil", 0, 0.5), ("Sand", 0.5, 4), ("Clay Loam", 4, 9), ("Clay", 9, 16)]),
    ("MW-07", 247.13, 635.0, [("Topsoil", 0, 0.5), ("Sand", 0.5, 4), ("Clay Loam", 4, 9), ("Clay", 9, 12)]),
]  # fmt: skip


def _df(holes, *, mirror_span: float | None = None) -> pd.DataFrame:
    rows = []
    for hole, x, collar, units in holes:
        for code, top, bottom in units:
            rows.append(
                {
                    "hole_id": hole,
                    "x_profile": x if mirror_span is None else mirror_span - x,
                    "collar_elevation": collar,
                    "top_elevation": collar - top,
                    "bottom_elevation": collar - bottom,
                    "lithology_code": code,
                }
            )
    return pd.DataFrame(rows)


def _hole_extents(projected: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    grouped = projected.groupby("hole_id").agg(
        x=("x_profile", "first"), top=("top_elevation", "max"), bottom=("bottom_elevation", "min")
    )
    return {hole: (row.x, row.top, row.bottom) for hole, row in grouped.iterrows()}


def _pairwise_overlap(polygons: list[GeologicalPolygon]) -> float:
    total = 0.0
    for index, first in enumerate(polygons):
        for second in polygons[index + 1 :]:
            if first.hole_pair == second.hole_pair:
                total += first.polygon.intersection(second.polygon).area
    return total


def _mid_span_vertical_edges(
    polygons: list[GeologicalPolygon], hole_x: dict[str, float]
) -> list[tuple[str, float]]:
    """Vertical exterior edges that are not on one of the polygon's holes."""
    walls = []
    for item in polygons:
        holes = [hole_x[hole] for hole in item.hole_pair]
        coords = list(item.polygon.exterior.coords)
        for (x0, z0), (x1, z1) in zip(coords, coords[1:]):
            if abs(x1 - x0) <= 1e-9 and abs(z1 - z0) > 1e-6:
                if min(abs(x0 - x) for x in holes) > 1e-6:
                    walls.append((item.lithology_code, x0))
    return walls


def _mid_span_fill_contacts(
    polygons: list[GeologicalPolygon], hole_x: dict[str, float]
) -> list[tuple[str, str]]:
    """Continuous fills touching each other strictly between their holes (bow-ties).

    Matched fills only ever meet along a shared contact that runs hole to hole;
    two fills squeezed to a point mid-span is the bow-tie of two opposing tips.
    """
    found = []
    fills = [p for p in polygons if not p.is_pinch_out]
    for index, first in enumerate(fills):
        for second in fills[index + 1 :]:
            if first.hole_pair != second.hole_pair:
                continue
            x0, x1 = sorted(hole_x[hole] for hole in first.hole_pair)
            shared = first.polygon.intersection(second.polygon)
            if shared.is_empty:
                continue
            if shared.length == 0.0:  # point contact(s)
                points = getattr(shared, "geoms", [shared])
                if any(
                    x0 + 1e-6 < point.x < x1 - 1e-6
                    for point in points
                    if point.geom_type == "Point"
                ):
                    found.append((first.lithology_code, second.lithology_code))
    return found


def _assert_clean(polygons: list[GeologicalPolygon], hole_x: dict[str, float]) -> None:
    assert polygons
    for item in polygons:
        assert item.polygon.geom_type == "Polygon" and item.polygon.is_valid, item
        assert item.polygon.area > 0.0
    assert _pairwise_overlap(polygons) == pytest.approx(0.0, abs=1e-6)
    assert detect_polygon_overlaps(polygons) == []
    assert _mid_span_fill_contacts(polygons, hole_x) == []


# ---------------------------------------------------------------------------
# Defect 1: short hole between deeper holes (nested MW-06S / MW-06D)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("allow_pinch_outs", [False, True])
def test_units_continue_beneath_short_hole_without_void(
    mirror: bool, allow_pinch_outs: bool
) -> None:
    projected = _df(_HYDRO, mirror_span=300.0 if mirror else None)
    before = projected.copy()
    polygons = build_stratigraphy(projected, allow_pinch_outs=allow_pinch_outs)
    pd.testing.assert_frame_equal(projected, before)  # hole data untouched
    extents = _hole_extents(projected)
    hole_x = {hole: x for hole, (x, _top, _bottom) in extents.items()}
    _assert_clean(polygons, hole_x)

    # Between MW-05 and MW-06D the fence covers everything from the ground line
    # down to the shallower of the two deep holes' TDs (MW-05, 626 m), right
    # under MW-06S (TD 634 m).
    (xa, ta, ba), (xs, ts, _bs), (xb, tb, bb) = (extents[h] for h in ("MW-05", "MW-06S", "MW-06D"))
    (x0, t0), (x1, t1) = sorted([(xa, ta), (xb, tb)])
    floor = max(ba, bb)
    region = Polygon([(x0, t0), (xs, ts), (x1, t1), (x1, floor), (x0, floor)])
    near = [p.polygon for p in polygons if set(p.hole_pair) <= {"MW-05", "MW-06S", "MW-06D"}]
    assert region.difference(unary_union(near)).area < GAP_TOL_M2

    # Below MW-06S's TD the fill is inferred (dashed / faded), not logged.
    below = [
        p
        for p in polygons
        if "MW-06S" in p.hole_pair
        and p.polygon.bounds[1] < 634.0 - 1.0
        and p.lithology_code == "Clay"
    ]
    assert below and all(p.is_pinch_out for p in below)
    # The Clay continues as two hole-to-hole fills meeting under MW-06S, from
    # MW-05's TD to MW-06D's TD (the base line), not as a sliver.
    clay_bottom = unary_union([p.polygon for p in below]).intersection(
        LineString([(xs, 600.0), (xs, 640.0)])
    )
    t = (xs - xa) / (xb - xa)
    assert clay_bottom.bounds[1] == pytest.approx(ba + t * (bb - ba), abs=1e-6)
    # No sliver: every polygon in the nested pair spans the whole 1.5 m gap.
    for item in polygons:
        if set(item.hole_pair) == {"MW-06S", "MW-06D"}:
            x_min, _z0, x_max, _z1 = item.polygon.bounds
            assert x_max - x_min == pytest.approx(1.5)
            assert item.polygon.area > 0.5


def test_short_hole_mirror_symmetric() -> None:
    forward = build_stratigraphy(_df(_HYDRO), allow_pinch_outs=True)
    backward = build_stratigraphy(_df(_HYDRO, mirror_span=300.0), allow_pinch_outs=True)
    assert _unit_unions(forward, None).keys() == _unit_unions(backward, 300.0).keys()
    fwd, rev = _unit_unions(forward, None), _unit_unions(backward, 300.0)
    for key, shape in fwd.items():
        assert shape.symmetric_difference(rev[key]).area < 1e-6, key


def test_run_of_short_holes_chains_off_deeper_neighbours() -> None:
    units = [("Sand", 0, 4), ("Clay", 4, 10), ("Till", 10, 20)]
    holes = [
        ("A", 0.0, 100.0, units),
        ("S1", 20.0, 100.0, units[:1] + [("Clay", 4, 6)]),
        ("S2", 40.0, 100.0, units[:1] + [("Clay", 4, 5)]),
        ("B", 60.0, 100.0, units),
    ]
    polygons = build_stratigraphy(_df(holes), allow_pinch_outs=False)
    hole_x = {"A": 0.0, "S1": 20.0, "S2": 40.0, "B": 60.0}
    _assert_clean(polygons, hole_x)
    region = Polygon([(0.0, 100.0), (60.0, 100.0), (60.0, 80.0), (0.0, 80.0)])
    assert region.difference(unary_union([p.polygon for p in polygons])).area < GAP_TOL_M2
    till = [p for p in polygons if p.lithology_code == "Till"]
    assert len(till) == 3 and all(p.is_pinch_out for p in till)


def test_ordinary_depth_differences_keep_fence_on_hole_bottoms() -> None:
    """A hole a little shorter than its neighbours (or one in a valley, whose base
    is already deeper) is not extended: the fence base follows the hole bottoms."""
    units = [("Sand", 0, 4), ("Clay", 4, 15)]
    holes = [
        ("A", 0.0, 100.0, units),
        ("B", 40.0, 101.0, [("Sand", 0, 4), ("Clay", 4, 14.5)]),  # 0.5 m shorter
        ("C", 80.0, 92.0, [("Sand", 0, 4), ("Clay", 4, 12)]),  # valley: shorter, deeper RL
        ("D", 120.0, 100.0, units),
    ]
    polygons = build_stratigraphy(_df(holes), allow_pinch_outs=True)
    assert not any(p.is_pinch_out for p in polygons)
    for x, bottom in ((40.0, 86.5), (80.0, 80.0)):
        cut = unary_union([p.polygon for p in polygons]).intersection(
            LineString([(x, 0.0), (x, 200.0)])
        )
        assert cut.bounds[1] == pytest.approx(bottom)


@pytest.mark.parametrize("gap", [0.0, 1e-7, 0.01])
def test_coincident_or_near_coincident_short_hole_is_graceful(gap: float) -> None:
    units = [("Sand", 0, 4), ("Clay", 4, 16)]
    holes = [
        ("A", 0.0, 100.0, units),
        ("S", 40.0 - gap, 100.0, [("Sand", 0, 4), ("Clay", 4, 6)]),
        ("D", 40.0, 100.0, units),
        ("B", 80.0, 100.0, units),
    ]
    polygons = build_stratigraphy(_df(holes), allow_pinch_outs=True)
    for item in polygons:
        assert item.polygon.is_valid and item.polygon.area > 0.0
    assert detect_polygon_overlaps(polygons) == []
    region = Polygon([(0.0, 100.0), (80.0, 100.0), (80.0, 84.0), (0.0, 84.0)])
    assert region.difference(unary_union([p.polygon for p in polygons])).area < 1e-3


# ---------------------------------------------------------------------------
# Defect 2: quick_preview walls and bow-ties (pinch-outs on)
# ---------------------------------------------------------------------------


def _sample_geometry(**overrides):
    from batch_export import transect_points_from_collars
    from ingestion import ingest_workbook
    from pipeline import compute_section_geometry
    from ui_output_presets import OUTPUT_PRESETS

    preset = OUTPUT_PRESETS["quick_preview"]
    parse_result, _report = ingest_workbook(str(REPO / "data" / "sample_boreholes.xlsx"))
    holes = [collar.hole_id for collar in parse_result.collars]
    kwargs = dict(
        interpretation_mode=preset.interpretation_mode or "interpolated",
        allow_pinch_outs=preset.allow_pinch_outs,
    )
    kwargs.update(overrides)
    return compute_section_geometry(
        parse_result.collars,
        parse_result.lithologies,
        transect_points_from_collars(parse_result.collars, holes),
        **kwargs,
    )


def test_quick_preview_pinch_outs_are_wedges_without_walls_or_bow_ties() -> None:
    geometry = _sample_geometry()
    assert geometry.polygons and any(p.is_pinch_out for p in geometry.polygons)
    extents = _hole_extents(geometry.projected)
    hole_x = {hole: x for hole, (x, _top, _bottom) in extents.items()}
    _assert_clean(list(geometry.polygons), hole_x)
    assert _mid_span_vertical_edges(list(geometry.polygons), hole_x) == []
    assert not [w for w in geometry.overlap_warnings if " / " in w]
    # Every pinch-out tapers to zero thickness before (or at) the other hole.
    for item in geometry.polygons:
        if not item.is_pinch_out:
            continue
        xs = sorted(hole_x[hole] for hole in item.hole_pair)
        for x in xs:
            cut = item.polygon.intersection(LineString([(x, -1e4), (x, 1e4)]))
            if cut.length == 0.0:
                break
        else:  # pragma: no cover - failure path
            pytest.fail(f"{item.lithology_code} {item.hole_pair} has thickness at both holes")
    # The section is fully tiled between the hole tops and bottoms.
    ordered = sorted(extents.values())
    for (x0, t0, b0), (x1, t1, b1) in zip(ordered, ordered[1:]):
        region = Polygon([(x0, t0), (x1, t1), (x1, b1), (x0, b0)])
        assert (
            region.difference(unary_union([p.polygon for p in geometry.polygons])).area < GAP_TOL_M2
        )


def test_quick_preview_bh02_bh03_facies_change_keeps_bracketing_fills_straight() -> None:
    geometry = _sample_geometry()
    pair = [p for p in geometry.polygons if p.hole_pair == ("BH-02", "BH-03")]
    for code in ("Sandstone", "Bedrock"):
        fill = next(p for p in pair if p.lithology_code == code)
        assert not fill.is_pinch_out
        assert len(fill.polygon.exterior.coords) == 5  # hole-to-hole quadrilateral
    clay = next(p for p in pair if p.lithology_code == "Clay").polygon
    silt = next(p for p in pair if p.lithology_code == "Silt").polygon
    shared = clay.intersection(silt)
    assert shared.geom_type == "LineString" and shared.length > 40.0  # sloping boundary


# ---------------------------------------------------------------------------
# Fuzz: random sections with short holes, both pinch-out modes, both directions
# ---------------------------------------------------------------------------

_CODES = ("Fill", "Sand", "Silt", "Clay", "Till", "Gravel", "Bedrock")


def _random_holes(seed: int):
    rng = random.Random(seed)
    holes = []
    x = 0.0
    for index in range(rng.randint(3, 5)):
        collar = round(100.0 + rng.uniform(-3.0, 3.0), 3)
        depth = 0.0
        units = []
        codes = rng.sample(_CODES, rng.randint(2, 5))
        codes.sort(key=_CODES.index)
        for code in codes:
            thickness = round(rng.uniform(0.5, 6.0), 3)
            units.append((code, round(depth, 3), round(depth + thickness, 3)))
            depth += thickness
        if index not in (0,) and rng.random() < 0.4:
            units = units[: rng.randint(1, len(units))]  # short hole
        holes.append((f"H{index}", x, collar, units))
        x += round(rng.choice([1.5, rng.uniform(10.0, 60.0)]), 3)
    return holes, x


def _unit_unions(polygons: list[GeologicalPolygon], mirror_span: float | None):
    groups = defaultdict(list)
    for item in polygons:
        shape = item.polygon
        if mirror_span is not None:
            shape = affinity.scale(shape, xfact=-1.0, yfact=1.0, origin=(mirror_span / 2.0, 0.0))
        groups[(frozenset(item.hole_pair), item.lithology_code, item.is_pinch_out)].append(shape)
    return {key: unary_union(shapes) for key, shapes in groups.items()}


@pytest.mark.parametrize("allow_pinch_outs", [True, False])
@pytest.mark.parametrize("seed", range(40))
def test_fuzz_fence_is_valid_tiled_and_mirror_symmetric(seed: int, allow_pinch_outs: bool) -> None:
    holes, span = _random_holes(seed)
    forward_df = _df(holes)
    forward = build_stratigraphy(forward_df, allow_pinch_outs=allow_pinch_outs)
    backward = build_stratigraphy(_df(holes, mirror_span=span), allow_pinch_outs=allow_pinch_outs)
    hole_x = {hole: x for hole, x, _collar, _units in holes}
    for item in forward:
        assert item.polygon.is_valid and item.polygon.geom_type == "Polygon"
    assert detect_polygon_overlaps(forward) == []
    assert _pairwise_overlap(forward) == pytest.approx(0.0, abs=1e-6)
    if allow_pinch_outs:
        assert _mid_span_vertical_edges(forward, hole_x) == []
    fwd, rev = _unit_unions(forward, None), _unit_unions(backward, span)
    assert fwd.keys() == rev.keys()
    for key, shape in fwd.items():
        assert shape.symmetric_difference(rev[key]).area <= 1e-6 * max(1.0, shape.area), key


_SEQUENCE = ("Fill", "Sand", "Silt", "Clay", "Till", "Bedrock")


def _layered_holes(seed: int):
    """One stratigraphic sequence (units occasionally absent), some holes stopped short."""
    rng = random.Random(seed)
    holes = []
    x = 0.0
    count = rng.randint(3, 6)
    for index in range(count):
        collar = round(100.0 + rng.uniform(-4.0, 4.0), 3)
        depth = 0.0
        units = []
        for code in _SEQUENCE:
            if code not in ("Clay", "Bedrock") and rng.random() < 0.25:
                continue
            thickness = round(rng.uniform(0.5, 5.0), 3)
            units.append((code, round(depth, 3), round(depth + thickness, 3)))
            depth += thickness
        if 0 < index < count - 1 and rng.random() < 0.5:
            units = units[: rng.randint(1, len(units) - 1)]
        holes.append((f"H{index}", x, collar, units))
        x += round(rng.choice([1.5, rng.uniform(10.0, 60.0)]), 3)
    return holes


@pytest.mark.parametrize("allow_pinch_outs", [True, False])
@pytest.mark.parametrize("seed", range(60))
def test_fuzz_short_holes_leave_no_void_between_deeper_neighbours(
    seed: int, allow_pinch_outs: bool
) -> None:
    """Union covers each short hole's neighbour span down to the shallower of the
    deeper neighbours' TDs (no gap > tolerance); nothing overlaps."""
    holes = _layered_holes(seed)
    polygons = build_stratigraphy(_df(holes), allow_pinch_outs=allow_pinch_outs)
    assert detect_polygon_overlaps(polygons) == []
    assert _pairwise_overlap(polygons) == pytest.approx(0.0, abs=1e-6)
    covered = unary_union([p.polygon for p in polygons])
    for (_a, xa, ca, ua), (_s, xs, cs, us), (_b, xb, cb, ub) in zip(holes, holes[1:], holes[2:]):
        depth_a, depth_s, depth_b = ua[-1][2], us[-1][2], ub[-1][2]
        # "Short" is judged on bottom elevations (the frame the fence is built in).
        needed = cs - depth_s - max(1.0, 0.25 * depth_s)
        if ca - depth_a > needed or cb - depth_b > needed:
            continue
        floor = max(ca - depth_a, cb - depth_b)
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
