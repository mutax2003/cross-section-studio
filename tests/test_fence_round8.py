"""Round 8 fence defects: wedge tips past an overlapping neighbour unit (D3) and
coincident collars (D5).

D3 — fuzz seed 46: two holes 5 m apart, H0 logging Sand 0–15.57 m over Fill and
H1 a 2 m hole (Gravel / Till / Gravel). No unit correlates, so every H1 unit is
an unbracketed pinch-out whose tip came from ``_pinch_out_z_mid``. That helper
classified the neighbour's intervals only as wholly *above* or *below* the
wedge; the Sand, which overlaps the whole 2 m hole, was neither, so the tip was
taken from the next contact down — the Fill top, 16 m below — and each H1 unit
was drawn as a needle plunging to it. The fill 5 cm from H1 no longer matched
its log. A wedge now tapers at its own mid-elevation whenever a neighbour unit
overlaps its elevation span (it projects *into* that unit), and likewise when
it lies wholly below the neighbour's logged base (the neighbour never reached
it; its TD is not a contact).

D5 — two holes at one projected position (a nested MW-06S / MW-06D pair, or a
section line at right angles to their offset) stayed separate profiles. The
fence on one side paired with whichever came first and the other side with the
other hole, so the section depended on transect direction, and one log was
absent from each side with no note. Profiles are now sorted by (x, deeper
first, hole ID) and collapsed to one per position; the dropped hole is drawn as
a column only and a fence note names the pair and the log used.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from shapely.affinity import scale, translate
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from models import Collar, Lithology
from pipeline import compute_section_geometry
from stratigraphy import (
    COINCIDENT_HOLE_TOLERANCE_M,
    NEAR_FIELD_PINCH_FRACTION,
    PINCH_OUT_FRACTION,
    _LayerInterval,
    _pinch_out_z_mid,
    build_stratigraphy,
    detect_polygon_overlaps,
    preview_correlation_health,
)
from tests.test_fence_logs_honoured import _log_mismatches
from tests.test_fence_voids_round7 import _df

# ---------------------------------------------------------------------------
# D3: seed 46 exact case
# ---------------------------------------------------------------------------

SEED46_COLLARS = [
    Collar(hole_id="H0", easting=1000.0, northing=5000.0, elevation=99.40, total_depth=35.0),
    Collar(hole_id="H1", easting=1005.0, northing=5000.0, elevation=100.17, total_depth=2.0),
]
SEED46_LITHOLOGIES = [
    Lithology(hole_id="H0", from_depth=0.0, to_depth=15.57, lithology_code="SA"),
    Lithology(hole_id="H0", from_depth=15.57, to_depth=35.0, lithology_code="FILL"),
    Lithology(hole_id="H1", from_depth=0.0, to_depth=0.79, lithology_code="GR"),
    Lithology(hole_id="H1", from_depth=0.79, to_depth=1.67, lithology_code="TILL"),
    Lithology(hole_id="H1", from_depth=1.67, to_depth=2.0, lithology_code="GR"),
]
SEED46_TRANSECT = ((995.0, 5000.0), (1010.0, 5000.0))


def _seed46_geometry(allow_pinch_outs: bool, *, reverse: bool = False):
    transect = tuple(reversed(SEED46_TRANSECT)) if reverse else SEED46_TRANSECT
    return compute_section_geometry(
        SEED46_COLLARS,
        SEED46_LITHOLOGIES,
        transect,
        allow_pinch_outs=allow_pinch_outs,
        offset_warning_m=1000.0,
    )


def _hole_positions(projected) -> dict[str, float]:
    return projected.groupby("hole_id")["x_profile"].first().to_dict()


def _hole_collars(projected) -> dict[str, float]:
    return projected.groupby("hole_id")["collar_elevation"].first().to_dict()


def _fill_beside_hole(
    polygons, projected, hole: str, offset_m: float, *, above_elevation: float | None = None
) -> list[tuple]:
    """``(depth, logged, drawn)`` wherever the fill ``offset_m`` from ``hole`` (toward
    its neighbour) at 25 / 50 / 75 % of each logged unit is not that unit. Units
    wholly above ``above_elevation`` (the neighbour's logged top) are skipped: they
    taper to that top, not inside a neighbour unit (not part of D3)."""
    xs = _hole_positions(projected)
    x = xs[hole]
    others = [other for other in xs.values() if other != x]
    neighbour = min(others, key=lambda other: abs(other - x))
    px = x + offset_m * (1.0 if neighbour > x else -1.0)
    collar = _hole_collars(projected)[hole]
    found = []
    for row in projected[projected["hole_id"] == hole].itertuples():
        if above_elevation is not None and row.bottom_elevation >= above_elevation - 1e-9:
            continue
        for fraction in (0.25, 0.5, 0.75):
            z = row.top_elevation - fraction * (row.top_elevation - row.bottom_elevation)
            drawn = sorted({p.lithology_code for p in polygons if p.polygon.covers(Point(px, z))})
            if drawn != [row.lithology_code]:
                found.append((round(collar - z, 3), row.lithology_code, drawn or "VOID"))
    return found


def _wedge_tips(polygons, hole_x: dict[str, float]) -> list[tuple[str, str, float, float, float]]:
    """``(code, source_hole, x_tip, z_tip, own_mid)`` for every triangular wedge."""
    tips = []
    for item in polygons:
        coords = list(item.polygon.exterior.coords)[:-1]
        assert len(coords) == 3, item
        for hole in item.hole_pair:
            on_hole = [z for x, z in coords if abs(x - hole_x[hole]) < 1e-9]
            if len(on_hole) == 2:
                (x_tip, z_tip) = next((x, z) for x, z in coords if abs(x - hole_x[hole]) >= 1e-9)
                tips.append((item.lithology_code, hole, x_tip, z_tip, sum(on_hole) / 2.0))
                break
        else:  # pragma: no cover - failure path
            pytest.fail(f"{item.lithology_code} {item.hole_pair} has no edge on a hole")
    return tips


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("allow_pinch_outs", [True, False])
def test_seed46_units_taper_at_their_own_mid_elevation(
    allow_pinch_outs: bool, reverse: bool
) -> None:
    geometry = _seed46_geometry(allow_pinch_outs, reverse=reverse)
    polygons = list(geometry.polygons)
    hole_x = _hole_positions(geometry.projected)
    reach = PINCH_OUT_FRACTION if allow_pinch_outs else NEAR_FIELD_PINCH_FRACTION
    spacing = hole_x["H1"] - hole_x["H0"]
    assert len(polygons) == 5 and all(p.is_pinch_out for p in polygons)
    tips = _wedge_tips(polygons, hole_x)
    assert sorted(code for code, *_ in tips) == ["FILL", "GR", "GR", "SA", "TILL"]
    for code, source, x_tip, z_tip, own_mid in tips:
        other = "H0" if source == "H1" else "H1"
        # Every unit thins laterally: the tip sits at the unit's own mid-elevation
        # ``reach`` of the way to the neighbour, not 16 m down at the Fill top.
        assert z_tip == pytest.approx(own_mid, abs=1e-9), (code, source)
        assert x_tip == pytest.approx(hole_x[source] + reach * (hole_x[other] - hole_x[source]))
    # H1's units no longer plunge, so nothing is clipped back against the Sand.
    assert not [w for w in geometry.overlap_warnings if w.startswith("Fence clipped")]
    assert detect_polygon_overlaps(polygons) == []
    # Invariant: the fill 5 cm from each hole equals its log.
    for hole in ("H0", "H1"):
        assert _fill_beside_hole(polygons, geometry.projected, hole, 0.05) == [], hole
    assert _log_mismatches(geometry.projected, polygons) == []
    # The wedges stay between the ground line and the fence base.
    collars = _hole_collars(geometry.projected)
    xs = sorted(hole_x.values())
    ground = [collars[h] for h in sorted(hole_x, key=hole_x.get)]
    deepest = min(collars["H0"] - 35.0, collars["H1"] - 2.0)
    for item in polygons:
        for x, z in item.polygon.exterior.coords:
            assert z <= float(np.interp(x, xs, ground)) + 1e-9
            assert z >= deepest - 1e-9
    assert abs(spacing) == pytest.approx(5.0)


def test_seed46_before_and_after_tips_differ_by_the_needle() -> None:
    """Guard the exact defect: no H1 wedge reaches anywhere near the Fill top (83.83)."""
    geometry = _seed46_geometry(True)
    hole_x = _hole_positions(geometry.projected)
    for code, source, _x_tip, z_tip, _own_mid in _wedge_tips(list(geometry.polygons), hole_x):
        if source == "H1":
            assert z_tip > 98.0, (code, z_tip)


# ---------------------------------------------------------------------------
# D3: fuzz — the helper's contract and the fence invariant
# ---------------------------------------------------------------------------


def _interval(code: str, top: float, bottom: float, collar: float) -> _LayerInterval:
    return _LayerInterval(
        lithology_code=code,
        from_depth=collar - top,
        to_depth=collar - bottom,
        top_elevation=top,
        bottom_elevation=bottom,
    )


def test_pinch_out_tip_never_jumps_past_an_overlapping_neighbour_unit() -> None:
    """``_pinch_out_z_mid`` no longer reads the neighbour's log at all: wherever its
    units lie — overlapping the wedge, wholly above or below it — the tip is the
    wedge's own mid-elevation, held at or below the ground line."""
    rng = random.Random(8)
    for _case in range(500):
        collar = round(100.0 + rng.uniform(-3.0, 3.0), 3)
        wedge_top = round(rng.uniform(collar - 30.0, collar + 3.0), 3)
        wedge = _interval("W", wedge_top, round(wedge_top - rng.uniform(0.2, 6.0), 3), collar)
        own_mid = 0.5 * (wedge.top_elevation + wedge.bottom_elevation)
        assert _pinch_out_z_mid(wedge) == pytest.approx(own_mid), wedge
        ground = round(rng.uniform(collar - 3.0, collar + 3.0), 3)
        assert _pinch_out_z_mid(wedge, ground_at_tip=ground) == pytest.approx(
            min(own_mid, ground)
        ), (wedge, ground)
        assert _pinch_out_z_mid(wedge, ground_at_tip=float("nan")) == pytest.approx(own_mid)


_POOLS = (["FILL", "SA", "PT"], ["GR", "TILL", "CL"])


def _uncorrelated_pair(seed: int):
    """Two holes 3–60 m apart whose logs share no lithology code, so nothing
    correlates and every unit is an unbracketed pinch-out (the ``_pinch_out_z_mid``
    path): random logs with occasional logging gaps and repeated codes, one hole
    often much shorter than the other (seed 46 was a 2 m hole beside a 35 m one)."""
    rng = random.Random(seed)
    holes = []
    for index, pool in enumerate(_POOLS):
        x = 0.0 if index == 0 else round(rng.choice([3.0, 5.0, rng.uniform(8.0, 60.0)]), 3)
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
            units.append((rng.choice(pool), top, bottom))
            depth = bottom
        holes.append((f"H{index}", x, collar, units or [(pool[0], 0.0, td)]))
    return holes, holes[1][1]


@pytest.mark.parametrize("allow_pinch_outs", [True, False])
@pytest.mark.parametrize("seed", range(60))
def test_fuzz_uncorrelated_wedges_taper_inside_the_neighbour_log(
    seed: int, allow_pinch_outs: bool
) -> None:
    holes, span = _uncorrelated_pair(seed)
    reach = PINCH_OUT_FRACTION if allow_pinch_outs else NEAR_FIELD_PINCH_FRACTION
    for mirror_span in (None, span):
        projected = _df(holes, mirror_span=mirror_span)
        notes: list[str] = []
        polygons = build_stratigraphy(
            projected, allow_pinch_outs=allow_pinch_outs, clip_warnings=notes
        )
        assert polygons and all(p.is_pinch_out for p in polygons)
        for item in polygons:
            assert item.polygon.is_valid and item.polygon.area > 0.0, item
        assert detect_polygon_overlaps(polygons) == []
        # No wedge plunges past the neighbour's units, so none is clipped back.
        assert notes == [], (seed, notes)
        hole_x = _hole_positions(projected)
        collars = _hole_collars(projected)
        order = sorted(hole_x, key=hole_x.get)
        for code, source, x_tip, z_tip, own_mid in _wedge_tips(polygons, hole_x):
            other = "H0" if source == "H1" else "H1"
            assert x_tip == pytest.approx(hole_x[source] + reach * (hole_x[other] - hole_x[source]))
            # Own mid-elevation, held at or below the collar-to-collar ground line.
            ground = float(
                np.interp(x_tip, [hole_x[h] for h in order], [collars[h] for h in order])
            )
            assert z_tip == pytest.approx(min(own_mid, ground)), (
                seed,
                code,
                source,
                z_tip,
                own_mid,
            )
        # Invariant: the fill 5 cm from each hole equals its log (units wholly
        # above the neighbour's collar excepted: the ground line cuts them).
        for hole, other in (("H0", "H1"), ("H1", "H0")):
            assert (
                _fill_beside_hole(polygons, projected, hole, 0.05, above_elevation=collars[other])
                == []
            ), (seed, hole)
        assert _log_mismatches(projected, polygons) == []
        # Nothing drawn above the ground line or below the deepest hole.
        order = sorted(hole_x, key=hole_x.get)
        xs = [hole_x[h] for h in order]
        ground = [collars[h] for h in order]
        deepest = float(projected["bottom_elevation"].min())
        for item in polygons:
            for x, z in item.polygon.exterior.coords:
                assert z <= float(np.interp(x, xs, ground)) + 1e-6, (seed, item.lithology_code)
                assert z >= deepest - 1e-6, (seed, item.lithology_code)


# ---------------------------------------------------------------------------
# D5: coincident collars
# ---------------------------------------------------------------------------

_LAYERED = [("Sand", 0, 8), ("Clay", 8, 20)]
_NESTED = [
    ("MW-05", 0.0, 100.0, _LAYERED),
    ("MW-06S", 10.0, 100.0, [("Fill", 0, 5)]),
    ("MW-06D", 10.0, 100.0, _LAYERED),
    ("MW-07", 20.0, 100.0, _LAYERED),
]
_NESTED_NOTE = (
    "MW-06S and MW-06D share one position; the fence uses MW-06D's log (deeper) "
    "— MW-06S is drawn as a column only"
)


def _mirrored_union(polygons, span: float):
    return unary_union(
        [translate(scale(p.polygon, xfact=-1, yfact=1, origin=(0, 0)), xoff=span) for p in polygons]
    )


@pytest.mark.parametrize("row_order", ["S-first", "D-first"])
@pytest.mark.parametrize("allow_pinch_outs", [True, False])
def test_coincident_collars_use_the_deeper_log_on_both_sides(
    allow_pinch_outs: bool, row_order: str
) -> None:
    holes = (
        list(_NESTED)
        if row_order == "S-first"
        else [_NESTED[0], _NESTED[2], _NESTED[1], _NESTED[3]]
    )
    results = {}
    for mirror_span in (None, 20.0):
        notes: list[str] = []
        polygons = build_stratigraphy(
            _df(holes, mirror_span=mirror_span),
            allow_pinch_outs=allow_pinch_outs,
            clip_warnings=notes,
        )
        pairs = {p.hole_pair for p in polygons}
        assert not any("MW-06S" in pair for pair in pairs)
        expected = (
            {("MW-05", "MW-06D"), ("MW-06D", "MW-07")}
            if mirror_span is None
            else {("MW-07", "MW-06D"), ("MW-06D", "MW-05")}
        )
        assert pairs == expected
        assert notes == [_NESTED_NOTE]
        assert detect_polygon_overlaps(polygons) == []
        results[mirror_span] = polygons
    forward = unary_union([p.polygon for p in results[None]])
    mirrored = _mirrored_union(results[20.0], 20.0)
    assert forward.symmetric_difference(mirrored).area == pytest.approx(0.0, abs=1e-9)
    # The fence is Sand over Clay right through the nested position, both sides.
    for x in (9.9, 10.1):
        cut = unary_union([p.polygon for p in results[None] if p.lithology_code == "Sand"])
        probe = LineString([(x, 0.0), (x, 200.0)])
        assert cut.intersection(probe).length == pytest.approx(8.0)


def test_coincident_collars_equal_depth_tie_breaks_by_hole_id() -> None:
    holes = [
        ("MW-05", 0.0, 100.0, _LAYERED),
        ("MW-06S", 10.0, 100.0, [("Fill", 0, 20)]),
        ("MW-06D", 10.0, 100.0, _LAYERED),
        ("MW-07", 20.0, 100.0, _LAYERED),
    ]
    for mirror_span in (None, 20.0):
        notes: list[str] = []
        polygons = build_stratigraphy(_df(holes, mirror_span=mirror_span), clip_warnings=notes)
        assert {hole for p in polygons for hole in p.hole_pair} == {"MW-05", "MW-06D", "MW-07"}
        assert notes == [
            "MW-06S and MW-06D share one position; the fence uses MW-06D's log "
            "(same depth, first by hole ID) — MW-06S is drawn as a column only"
        ]


def test_coincident_collars_prefer_the_deeper_hole_whatever_its_id() -> None:
    holes = [
        ("MW-05", 0.0, 100.0, _LAYERED),
        ("MW-06A", 10.0, 100.0, [("Fill", 0, 5)]),
        ("MW-06B", 10.0, 100.0, _LAYERED),
        ("MW-07", 20.0, 100.0, _LAYERED),
    ]
    notes: list[str] = []
    polygons = build_stratigraphy(_df(holes), clip_warnings=notes)
    assert {hole for p in polygons for hole in p.hole_pair} == {"MW-05", "MW-06B", "MW-07"}
    assert notes == [
        "MW-06A and MW-06B share one position; the fence uses MW-06B's log (deeper) "
        "— MW-06A is drawn as a column only"
    ]


@pytest.mark.parametrize("gap", [0.0, COINCIDENT_HOLE_TOLERANCE_M / 2.0, 0.01])
def test_near_coincident_collars_collapse_only_within_tolerance(gap: float) -> None:
    holes = [
        ("MW-05", 0.0, 100.0, _LAYERED),
        ("MW-06S", 10.0 - gap, 100.0, [("Fill", 0, 5)]),
        ("MW-06D", 10.0, 100.0, _LAYERED),
        ("MW-07", 20.0, 100.0, _LAYERED),
    ]
    notes: list[str] = []
    polygons = build_stratigraphy(_df(holes), clip_warnings=notes)
    holes_in_fence = {hole for p in polygons for hole in p.hole_pair}
    if gap <= COINCIDENT_HOLE_TOLERANCE_M:
        assert "MW-06S" not in holes_in_fence and notes == [_NESTED_NOTE]
    else:
        assert "MW-06S" in holes_in_fence and notes == []


def test_preview_correlation_health_skips_the_column_only_hole() -> None:
    summaries = preview_correlation_health(_df(_NESTED))
    assert [(s.left_hole_id, s.right_hole_id) for s in summaries] == [
        ("MW-05", "MW-06D"),
        ("MW-06D", "MW-07"),
    ]


def test_coincident_collar_note_reaches_the_pipeline_warnings() -> None:
    """A real nested pair through ``compute_section_geometry``: the note is surfaced
    with the fence warnings, the shallow log stays in the projected (column) data
    and the fence mirrors on reversal."""
    collars = [
        Collar(hole_id="MW-05", easting=1000.0, northing=5000.0, elevation=100.0, total_depth=20.0),
        Collar(hole_id="MW-06S", easting=1010.0, northing=5000.0, elevation=100.0, total_depth=5.0),
        Collar(
            hole_id="MW-06D", easting=1010.0, northing=5000.0, elevation=100.0, total_depth=20.0
        ),
        Collar(hole_id="MW-07", easting=1020.0, northing=5000.0, elevation=100.0, total_depth=20.0),
    ]
    lithologies = [
        Lithology(hole_id=hole, from_depth=top, to_depth=bottom, lithology_code=code)
        for hole, _x, _collar, units in _NESTED
        for code, top, bottom in units
    ]
    transect = ((995.0, 5000.0), (1025.0, 5000.0))
    geometries = {}
    for name, points in (("forward", transect), ("reverse", tuple(reversed(transect)))):
        geometry = compute_section_geometry(
            collars, lithologies, points, allow_pinch_outs=True, offset_warning_m=1000.0
        )
        assert _NESTED_NOTE in geometry.overlap_warnings
        assert not any("MW-06S" in p.hole_pair for p in geometry.polygons)
        assert set(geometry.projected["hole_id"]) == {"MW-05", "MW-06S", "MW-06D", "MW-07"}
        geometries[name] = geometry
    forward = unary_union([p.polygon for p in geometries["forward"].polygons])
    span = float(
        geometries["reverse"].projected["x_profile"].max()
        + geometries["reverse"].projected["x_profile"].min()
    )
    mirrored = _mirrored_union(geometries["reverse"].polygons, span)
    mirrored = translate(mirrored, xoff=forward.bounds[0] - mirrored.bounds[0])
    assert forward.symmetric_difference(mirrored).area == pytest.approx(0.0, abs=1e-6)
