"""The fence never contradicts a borehole log, with pinch-outs on or off.

Browser QA (round 7): the default Section sheet style — and Consulting report /
Groundwater fence — turn "Show layers that end between holes" off. Stratigraphy
then absorbed every one-hole unit into the bracketing matched fill, so the fence
beside a hole showed a neighbour's unit: data/sample_boreholes.xlsx drew BH-01's
Silt (10-25 m) as Clay, dropped BH-04's Gravel (0-6 m) and showed Sandstone
beside BH-02's logged Clay; data/advantage_phase2_platform.xlsx drew a ~2 m Sand
~10 m thick. Now one-hole units stay drawn at their hole (a near-field taper when
pinch-outs are off), so at every borehole the fill equals the log.

Hard invariant: at x = hole ± a small epsilon the fill at each logged depth is
the logged lithology.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest
from shapely.geometry import Point

from batch_export import transect_points_from_collars
from ingestion import ingest_workbook
from pipeline import compute_section_geometry
from stratigraphy import NEAR_FIELD_PINCH_FRACTION, build_stratigraphy, detect_polygon_overlaps
from tests.test_fence_voids_round7 import _df, _layered_holes, _random_holes
from ui_output_presets import OUTPUT_PRESETS

REPO = Path(__file__).resolve().parents[1]
EPS_FRACTION = 1e-3  # of the spacing to the neighbour on that side
# "On the hole edge": no fence edge dips enough to cross a contact this close.
AT_HOLE_FRACTION = 1e-6
_HOLE_X_TOL_M = 1e-8  # vertices snapped to the 1e-9 m clip grid still sit "on" the hole


def _straight_beside_hole(shape, x: float, px: float) -> bool:
    """No vertex of ``shape`` lies strictly between the hole (``x``) and ``px``, so
    its edges run straight from the hole edge out to the probe."""
    lo, hi = sorted((x, px))
    rings = [shape.exterior, *shape.interiors]
    return not any(
        lo < vx < hi and abs(vx - x) > _HOLE_X_TOL_M for ring in rings for vx, _vz in ring.coords
    )


def _drawn_at(polygons, x: float, z: float) -> list[str]:
    point = Point(x, z)
    return sorted({p.lithology_code for p in polygons if p.polygon.covers(point)})


def _log_mismatches(projected: pd.DataFrame, polygons) -> list[tuple]:
    """``(hole, side, z, logged, drawn)`` wherever the fence beside a hole disagrees
    with its log (``drawn`` is ``"VOID"`` when nothing covers a logged depth).

    Each logged depth is probed at ``EPS_FRACTION`` of the spacing out from the
    hole. A probe at a fixed elevation reads *across* a dipping contact: a contact
    that leaves the hole at slope ``m`` has moved ``m * EPS_FRACTION * spacing``
    vertically at the probe, so a sample 5 mm inside a thin unit (10 % of 5 cm)
    lands in the next unit once ``m`` is a few m/m — a 2 m hole 3 m from a 35 m one,
    or matched units logged 10 m apart in elevation (round-9 QA: every one of the
    26 violations in the 5 cm-offset harness, and seeds 12/16/19 here, was exactly
    this; the fill on the hole edge matched the log in all of them). Such a probe
    is a *contact dip*, not a contradiction of the log, and is not reported when:

    * on the hole edge (``AT_HOLE_FRACTION`` of the spacing) the fill is exactly
      the logged unit — the hard invariant, checked for every sample and never
      relaxed; and
    * the logged unit's polygon there has no vertex between the hole and the
      probe, so the probe left it across one straight edge that starts on the
      hole at the logged contact (a clip notch, a bend vertex or a foreign sliver
      next to the hole puts a vertex in that range and is still reported); and
    * nothing else overlaps the logged unit at the probe (the logged unit is not
      among the drawn ones).
    """
    xs = projected.groupby("hole_id")["x_profile"].first().to_dict()
    order = sorted(xs.values())
    found = []
    for hole, group in projected.groupby("hole_id"):
        x = xs[hole]
        k = order.index(x)
        spacing = {
            -1: x - order[k - 1] if k > 0 else 0.0,
            1: order[k + 1] - x if k + 1 < len(order) else 0.0,
        }
        nearby = [p for p in polygons if hole in p.hole_pair]
        for side in (-1, 1):
            if spacing[side] <= 0.0:
                continue
            px = x + side * EPS_FRACTION * spacing[side]
            edge_x = x + side * AT_HOLE_FRACTION * spacing[side]
            if not any(p.polygon.bounds[0] <= px <= p.polygon.bounds[2] for p in nearby):
                continue
            for row in group.itertuples():
                if row.top_elevation - row.bottom_elevation < 0.05:
                    continue
                logged = row.lithology_code
                for fraction in (0.1, 0.5, 0.9):
                    z = row.top_elevation - fraction * (row.top_elevation - row.bottom_elevation)
                    on_edge = _drawn_at(nearby, edge_x, z)
                    if on_edge != [logged]:
                        found.append((hole, side, round(z, 3), logged, on_edge or "VOID"))
                        continue
                    drawn = _drawn_at(nearby, px, z)
                    if drawn == [logged]:
                        continue
                    if logged not in drawn:
                        own = [
                            p
                            for p in nearby
                            if p.lithology_code == logged and p.polygon.covers(Point(edge_x, z))
                        ]
                        if all(_straight_beside_hole(p.polygon, x, px) for p in own):
                            continue  # contact dip
                    found.append((hole, side, round(z, 3), logged, drawn or "VOID"))
    return found


def _workbook_geometry(name: str, preset_id: str):
    preset = OUTPUT_PRESETS[preset_id]
    parse_result, _report = ingest_workbook(str(REPO / "data" / name))
    holes = [collar.hole_id for collar in parse_result.collars]
    return compute_section_geometry(
        parse_result.collars,
        parse_result.lithologies,
        transect_points_from_collars(parse_result.collars, holes),
        interpretation_mode=preset.interpretation_mode or "interpolated",
        allow_pinch_outs=preset.allow_pinch_outs,
    )


@pytest.mark.parametrize("preset_id", sorted(OUTPUT_PRESETS))
@pytest.mark.parametrize("name", ["sample_boreholes.xlsx", "advantage_phase2_platform.xlsx"])
def test_every_preset_fence_matches_logs_at_each_hole(name: str, preset_id: str) -> None:
    geometry = _workbook_geometry(name, preset_id)
    if not geometry.polygons:  # borehole-only styles draw no fence
        return
    assert _log_mismatches(geometry.projected, list(geometry.polygons)) == []
    assert detect_polygon_overlaps(list(geometry.polygons)) == []
    assert all(p.polygon.is_valid for p in geometry.polygons)


def test_section_sheet_sample_boreholes_reported_holes() -> None:
    """The three cases from the QA report, checked directly."""
    geometry = _workbook_geometry("sample_boreholes.xlsx", "section_sheet")
    assert OUTPUT_PRESETS["section_sheet"].allow_pinch_outs is False
    polygons = list(geometry.polygons)

    def drawn(x: float, z: float) -> set[str]:
        return {p.lithology_code for p in polygons if p.polygon.covers(Point(x, z))}

    assert drawn(0.05, 102.5 - 17.5) == {"Silt"}  # BH-01 silt 10-25 m, not Clay
    assert drawn(149.95, 100.2 - 3.0) == {"Gravel"}  # BH-04 gravel 0-6 m is drawn
    for x in (49.95, 50.05):  # BH-02: Clay 97.8-89.8 over Bedrock, both sides
        assert drawn(x, 93.8) == {"Clay"}
        assert drawn(x, 85.0) == {"Bedrock"}


def test_advantage_sand_is_not_inflated_with_pinch_outs_off() -> None:
    """BH26-02-BH26-03: Sand is ~1.5-2.5 m thick in both logs and stays so (it was
    drawn down to the hole bottoms, ~10 m)."""
    geometry = _workbook_geometry("advantage_phase2_platform.xlsx", "section_sheet")
    sand = next(
        p
        for p in geometry.polygons
        if p.hole_pair == ("BH26-02", "BH26-03") and p.lithology_code == "Sand"
    )
    _x0, z_min, _x1, z_max = sand.polygon.bounds
    assert z_max - z_min <= 3.0 + 1e-6


@pytest.mark.parametrize("allow_pinch_outs", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_fuzz_fence_matches_logs(seed: int, allow_pinch_outs: bool) -> None:
    for holes in (_random_holes(seed)[0], _layered_holes(seed)):
        projected = _df(holes)
        polygons = build_stratigraphy(projected, allow_pinch_outs=allow_pinch_outs)
        assert _log_mismatches(projected, polygons) == []


def test_pinch_outs_off_lens_tapers_in_the_near_field() -> None:
    """A lens logged in one hole between matched units reaches only a quarter of the
    spacing with pinch-outs off (half with them on); the fills bend around it."""
    holes = [
        ("A", 0.0, 100.0, [("Sand", 0, 4), ("Silt", 4, 6), ("Clay", 6, 12)]),
        ("B", 40.0, 100.0, [("Sand", 0, 5), ("Clay", 5, 12)]),
    ]
    for allow, reach in ((False, NEAR_FIELD_PINCH_FRACTION), (True, 0.5)):
        polygons = build_stratigraphy(_df(holes), allow_pinch_outs=allow)
        silt = next(p for p in polygons if p.lithology_code == "Silt")
        assert silt.is_pinch_out and silt.polygon.bounds[2] == pytest.approx(40.0 * reach)
        assert _log_mismatches(_df(holes), polygons) == []


@pytest.mark.parametrize("gap", [0.0, 1e-12])
def test_coincident_holes_build_no_invalid_polygons(gap: float, caplog) -> None:
    """Template workbook QA: two holes projected to one station (a section line at
    right angles to their offset) logged "Invalid polygon for Sand/Topsoil/Clay Loam
    between MW-02 and MW-03" for every unit of the zero-width pair."""
    units = [("Topsoil", 0, 0.5), ("Sand", 0.5, 3), ("Clay Loam", 3, 8), ("Clay", 8, 12)]
    holes = [
        ("MW-01", 0.0, 635.0, units),
        ("MW-02", 45.0, 634.5, units),
        ("MW-03", 45.0 + gap, 634.0, units),
        ("MW-04", 90.0, 633.0, units),
    ]
    with caplog.at_level(logging.WARNING, logger="stratigraphy"):
        polygons = build_stratigraphy(_df(holes), allow_pinch_outs=False)
    assert not [r for r in caplog.records if "Invalid polygon" in r.getMessage()]
    assert all(p.polygon.is_valid and p.polygon.area > 0.0 for p in polygons)
    # One log represents the shared position on both sides (equal depths: the
    # first hole ID); the other hole is drawn as a column only (round 8, D5).
    assert {p.hole_pair for p in polygons} == {("MW-01", "MW-02"), ("MW-02", "MW-04")}
