"""Lithology stratigraphy and pinch-out polygon construction."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import shapely
from shapely.errors import GEOSException
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from models import CorrelationOverride

logger = logging.getLogger(__name__)

# Fraction of the hole spacing a pinch-out wedge spans from its source hole before
# tapering to its apex (0.5 = mid-way toward the neighbouring hole).
PINCH_OUT_FRACTION = 0.5

# With "Show layers that end between holes" off a unit logged in one hole only is
# not carried toward the neighbour: its wedge tapers out this fraction of the
# spacing from its own hole (the near field), so the hole's log is still honoured
# at the hole — never replaced by a neighbouring unit — and the bracketing fills
# close the rest of the gap.
NEAR_FIELD_PINCH_FRACTION = 0.25

# Intervals thinner than this (from_depth == to_depth within float noise) carry no
# fence geometry: they are left out of correlation / fence building (columns still
# show them) because a degenerate match collapses the neighbouring fills. Kept in
# sync with ``ai_quality.ZERO_THICKNESS_M``.
MIN_FENCE_THICKNESS_M = 1e-3


@dataclass(frozen=True)
class GeologicalPolygon:
    lithology_code: str
    polygon: Polygon
    hole_pair: tuple[str, str]
    is_pinch_out: bool = False


@dataclass(frozen=True)
class _LayerInterval:
    lithology_code: str
    from_depth: float
    to_depth: float
    top_elevation: float
    bottom_elevation: float
    unit_order: int | None = None
    # Fence-only unit continued below a short hole's total depth from its deeper
    # neighbours (``_extend_short_holes``); never logged in the hole itself.
    inferred: bool = False


def _intervals_for_hole(hole_df: pd.DataFrame) -> list[_LayerInterval]:
    if hole_df.empty:
        return []
    collar_elevation = float(hole_df["collar_elevation"].iloc[0])
    top_elevations = hole_df["top_elevation"].to_numpy(dtype=float)
    bottom_elevations = hole_df["bottom_elevation"].to_numpy(dtype=float)
    lithology_codes = hole_df["lithology_code"].astype(str).to_numpy()
    from_depths = collar_elevation - top_elevations
    to_depths = collar_elevation - bottom_elevations
    has_unit_order = "unit_order" in hole_df.columns
    if has_unit_order:
        unit_orders = hole_df["unit_order"].to_numpy(dtype=float)
        if np.any(np.isfinite(unit_orders)):
            order = np.argsort(np.where(np.isfinite(unit_orders), unit_orders, from_depths))
        else:
            order = from_depths.argsort()
    else:
        unit_orders = None
        order = from_depths.argsort()
    return [
        _LayerInterval(
            lithology_code=str(lithology_codes[index]),
            from_depth=float(from_depths[index]),
            to_depth=float(to_depths[index]),
            top_elevation=float(top_elevations[index]),
            bottom_elevation=float(bottom_elevations[index]),
            unit_order=(
                int(raw_order)
                if unit_orders is not None
                and np.isfinite(raw_order := float(unit_orders[index]))
                and raw_order.is_integer()
                else None
            ),
        )
        for index in order
    ]


def _close_logging_gaps(intervals: list[_LayerInterval]) -> list[_LayerInterval]:
    """Fence-only copy of a hole's intervals with internal logging gaps closed.

    A not-logged interval (no recovery) between two logged units is unknown only
    *at* the hole — the column renders it grey. For correlation between holes the
    gap is treated as absent: the unit above extends down and the unit below
    extends up to the gap midpoint, so the inter-hole fence tiles without a white
    wedge. Gaps above the first or below the last logged interval are left alone
    (the fence simply follows the logged extent there), as are overlapping
    intervals. Returned in the same order as ``intervals``; unchanged intervals
    keep their identity.
    """
    if len(intervals) < 2:
        return intervals
    by_elevation = sorted(
        range(len(intervals)),
        key=lambda index: (-intervals[index].top_elevation, -intervals[index].bottom_elevation),
    )
    tops = {index: intervals[index].top_elevation for index in by_elevation}
    bottoms = {index: intervals[index].bottom_elevation for index in by_elevation}
    changed = False
    # Walk down the hole tracking the deepest logged base seen so far, so an
    # interval nested inside a longer one never reads as a gap.
    upper: int | None = None
    for lower in by_elevation:
        lower_top = intervals[lower].top_elevation
        lower_bottom = intervals[lower].bottom_elevation
        if not (np.isfinite(lower_top) and np.isfinite(lower_bottom)):
            continue
        if upper is not None:
            upper_bottom = intervals[upper].bottom_elevation
            if upper_bottom - lower_top > 1e-9:
                middle = 0.5 * (upper_bottom + lower_top)
                bottoms[upper] = middle
                tops[lower] = middle
                changed = True
        if upper is None or lower_bottom < intervals[upper].bottom_elevation:
            upper = lower
    if not changed:
        return intervals
    closed: list[_LayerInterval] = []
    for index, interval in enumerate(intervals):
        top, bottom = tops[index], bottoms[index]
        if top == interval.top_elevation and bottom == interval.bottom_elevation:
            closed.append(interval)
            continue
        closed.append(
            replace(
                interval,
                from_depth=interval.from_depth - (top - interval.top_elevation),
                to_depth=interval.to_depth - (bottom - interval.bottom_elevation),
                top_elevation=top,
                bottom_elevation=bottom,
            )
        )
    return closed


def _correlation_keys(intervals: list[_LayerInterval]) -> dict[Hashable, _LayerInterval]:
    code_counts = Counter(interval.lithology_code for interval in intervals)
    keys: dict[Hashable, _LayerInterval] = {}
    for index, interval in enumerate(intervals):
        if interval.unit_order is not None:
            key: Hashable = ("order", interval.unit_order, interval.lithology_code)
        elif code_counts[interval.lithology_code] == 1:
            key = ("code", interval.lithology_code)
        else:
            key = ("pos", index, interval.lithology_code)
        keys[key] = interval
    return keys


def _pinch_out_z_mid(
    pinch_interval: _LayerInterval,
    *,
    ground_at_tip: float | None = None,
) -> float:
    """Tip elevation of an unbracketed pinch-out (no matched unit above or below it).

    The wedge thins laterally at its own mid-elevation, held at or below the
    ground line (``ground_at_tip``: the collar-to-collar line at the tip's x).

    The tip used to come from the neighbour's contacts, classifying its
    intervals only as wholly *above* or *below* the wedge. A neighbour unit that
    overlaps the wedge's span — the unit the wedge projects *into* — was
    neither, so the tip jumped past it to the next contact: every unit of a 2 m
    hole was drawn as a needle plunging 16 m to the base of the neighbour's 15 m
    Sand, the 19 m Fill under that Sand rose 14 m to the 2 m hole's TD (then
    clipped back against the Sand), and the fill a few centimetres from the hole
    no longer matched its log (fuzz seed 46). The neighbour's TD and its top
    are not contacts the wedge abuts either: a unit above the neighbour's log
    following the ground to it crossed the own-mid wedges of the units below
    and clipped them. The rule no longer looks at the neighbour's units at all.
    """
    own_mid = (pinch_interval.top_elevation + pinch_interval.bottom_elevation) / 2.0
    if ground_at_tip is not None and np.isfinite(ground_at_tip):
        return min(own_mid, ground_at_tip)
    return own_mid


def _collar_elevation(intervals: list[_LayerInterval]) -> float:
    """Collar RL the hole's intervals were logged from (``top_elevation + from_depth``)."""
    collars = [
        interval.top_elevation + interval.from_depth
        for interval in intervals
        if np.isfinite(interval.top_elevation) and np.isfinite(interval.from_depth)
    ]
    return max(collars) if collars else float("nan")


def _interpolate(x0: float, z0: float, x1: float, z1: float, x: float) -> float:
    """Linear interpolation of the line ``(x0, z0)``–``(x1, z1)`` at ``x``."""
    span = x1 - x0
    t = (x - x0) / span if span else 0.5
    return z0 + t * (z1 - z0)


def _dedupe_vertices(
    coords: list[tuple[float, float]], *, rel_tol: float = 1e-9
) -> list[tuple[float, float]]:
    """Drop consecutive (and closing) vertices that coincide to within float noise.

    Interpolated tips and bends can land ~1e-15 apart from an existing vertex;
    GEOS then builds a needle spike whose overlay is precision-sensitive (one
    intersection direction reports area where the other reports none), which
    surfaced as false overlap warnings.
    """
    if len(coords) < 2:
        return coords

    def _same(a: tuple[float, float], b: tuple[float, float]) -> bool:
        scale = max(1.0, abs(a[0]), abs(a[1]), abs(b[0]), abs(b[1]))
        return abs(a[0] - b[0]) <= rel_tol * scale and abs(a[1] - b[1]) <= rel_tol * scale

    kept: list[tuple[float, float]] = []
    for point in coords:
        if kept and _same(kept[-1], point):
            continue
        kept.append(point)
    while len(kept) > 1 and _same(kept[0], kept[-1]):
        kept.pop()
    return kept


def _clean_polygon(polygon: Polygon) -> Polygon:
    """Drop near-coincident vertices a clip left behind (keeps the input if repair fails)."""
    exterior = list(polygon.exterior.coords)[:-1]
    deduped = _dedupe_vertices(exterior)
    if len(deduped) == len(exterior) or len(deduped) < 3:
        return polygon
    try:
        cleaned = Polygon(deduped, [list(ring.coords) for ring in polygon.interiors])
    except (GEOSException, ValueError):
        return polygon
    return cleaned if cleaned.is_valid and not cleaned.is_empty else polygon


def _make_polygon(
    coords: list[tuple[float, float]],
    lithology_code: str,
    hole_pair: tuple[str, str],
    *,
    is_pinch_out: bool = False,
) -> GeologicalPolygon | None:
    coords = _dedupe_vertices(coords)
    try:
        polygon = Polygon(coords)
    except (GEOSException, ValueError) as exc:
        # Non-finite coordinates (e.g. NaN collar elevation) — warn-and-skip,
        # matching the platform's graceful-degradation rule.
        logger.warning(
            "Invalid polygon coordinates for %s between %s and %s: %s",
            lithology_code,
            hole_pair[0],
            hole_pair[1],
            exc,
        )
        return None
    if not polygon.is_valid:
        repaired = polygon.buffer(0)
        if repaired.is_empty or not repaired.is_valid:
            logger.warning(
                "Invalid polygon for %s between %s and %s",
                lithology_code,
                hole_pair[0],
                hole_pair[1],
            )
            return None
        polygon = repaired

    # Renderer expects a single Polygon (.exterior); buffer(0) may yield MultiPolygon.
    largest = _largest_polygon(polygon)
    if largest is None or largest.is_empty:
        logger.warning(
            "Empty polygon after repair for %s between %s and %s",
            lithology_code,
            hole_pair[0],
            hole_pair[1],
        )
        return None

    return GeologicalPolygon(
        lithology_code=lithology_code,
        polygon=largest,
        hole_pair=hole_pair,
        is_pinch_out=is_pinch_out,
    )


def _correlation_sort_key(
    key: Hashable,
    left_lookup: dict[Hashable, _LayerInterval],
    right_lookup: dict[Hashable, _LayerInterval],
) -> float:
    tops: list[float] = []
    if key in left_lookup:
        tops.append(left_lookup[key].top_elevation)
    if key in right_lookup:
        tops.append(right_lookup[key].top_elevation)
    return -max(tops) if tops else float("inf")


def _correlation_overrides_by_pair(
    overrides: Sequence[CorrelationOverride],
) -> dict[tuple[str, str], tuple[CorrelationOverride, ...]]:
    buckets: dict[tuple[str, str], list[CorrelationOverride]] = defaultdict(list)
    for override in overrides:
        buckets[(override.left_hole_id, override.right_hole_id)].append(override)
    return {pair: tuple(items) for pair, items in buckets.items()}


def _overrides_for_hole_pair(
    left_hole_id: str,
    right_hole_id: str,
    override_index: dict[tuple[str, str], tuple[CorrelationOverride, ...]],
) -> tuple[CorrelationOverride, ...]:
    """Return overrides for transect left→right, accepting reversed hole order."""
    exact = override_index.get((left_hole_id, right_hole_id))
    if exact:
        return exact
    reversed_items = override_index.get((right_hole_id, left_hole_id))
    if not reversed_items:
        return ()
    return tuple(
        CorrelationOverride(
            left_hole_id=left_hole_id,
            right_hole_id=right_hole_id,
            left_unit_order=item.right_unit_order,
            right_unit_order=item.left_unit_order,
        )
        for item in reversed_items
    )


def _apply_correlation_overrides(
    left_hole_id: str,
    right_hole_id: str,
    left_intervals: list[_LayerInterval],
    right_intervals: list[_LayerInterval],
    left_lookup: dict[Hashable, _LayerInterval],
    right_lookup: dict[Hashable, _LayerInterval],
    overrides: Sequence[CorrelationOverride],
) -> tuple[dict[Hashable, _LayerInterval], dict[Hashable, _LayerInterval]]:
    if not overrides:
        return left_lookup, right_lookup
    left_by_order = {
        interval.unit_order: interval
        for interval in left_intervals
        if interval.unit_order is not None
    }
    right_by_order = {
        interval.unit_order: interval
        for interval in right_intervals
        if interval.unit_order is not None
    }
    merged_left = dict(left_lookup)
    merged_right = dict(right_lookup)
    for override in overrides:
        if (override.left_hole_id, override.right_hole_id) != (left_hole_id, right_hole_id):
            continue
        left_layer = left_by_order.get(override.left_unit_order)
        right_layer = right_by_order.get(override.right_unit_order)
        if left_layer is None or right_layer is None:
            continue
        key: Hashable = (
            "override",
            override.left_unit_order,
            override.right_unit_order,
            left_layer.lithology_code,
        )
        # Drop original correlation keys for remapped intervals so they are not
        # drawn twice (matched override + leftover pinch-out).
        for lookup, layer in ((merged_left, left_layer), (merged_right, right_layer)):
            for existing_key, existing_layer in list(lookup.items()):
                if existing_layer is layer:
                    del lookup[existing_key]
            lookup[key] = layer
    return merged_left, merged_right


def _rematch_shifted_units(
    left_intervals: list[_LayerInterval],
    right_intervals: list[_LayerInterval],
    left_lookup: dict[Hashable, _LayerInterval],
    right_lookup: dict[Hashable, _LayerInterval],
) -> tuple[dict[Hashable, _LayerInterval], dict[Hashable, _LayerInterval]]:
    """Correlate same-code units whose position keys differ only by an inserted unit.

    ``('order', n, code)`` / ``('pos', i, code)`` keys stop matching as soon as one
    hole logs an extra unit above (e.g. a Silt lens shifts the underlying Clay from
    order 4 to 5). Left unmatched, the same Clay is drawn as two crossing pinch-out
    wedges that cut diagonally across the section and leave gaps beside the holes.

    The same happens when a hole repeats a unit (Clay above and below a Gravel
    lens): duplicate codes get ``('pos', index, code)`` keys that rarely line up.

    Unmatched intervals of the same lithology code are paired under shared
    ``('shifted', code, i, j)`` keys, best ``_pair_rank`` first across all codes,
    skipping any pairing that would cross an already-matched correlation, so
    hole-local stratigraphic order is preserved and the result does not depend on
    transect direction. Surplus repeats stay pinch-outs.
    """
    left_unmatched = [key for key in left_lookup if key not in right_lookup]
    right_unmatched = [key for key in right_lookup if key not in left_lookup]
    if not left_unmatched or not right_unmatched:
        return left_lookup, right_lookup

    def _by_code(
        keys: list[Hashable], lookup: dict[Hashable, _LayerInterval]
    ) -> dict[str, list[Hashable]]:
        grouped: dict[str, list[Hashable]] = defaultdict(list)
        for key in keys:
            grouped[lookup[key].lithology_code].append(key)
        return grouped

    left_by_code = _by_code(left_unmatched, left_lookup)
    right_by_code = _by_code(right_unmatched, right_lookup)
    left_index = _stratigraphic_positions(left_intervals)
    right_index = _stratigraphic_positions(right_intervals)
    matched_positions = [
        (left_index[id(left_lookup[key])], right_index[id(right_lookup[key])])
        for key in left_lookup
        if key in right_lookup
        and id(left_lookup[key]) in left_index
        and id(right_lookup[key]) in right_index
    ]

    # Rank every same-code candidate pair globally (not hole-by-hole greedy) so the
    # result does not depend on transect direction (see ``_pair_rank``).
    candidates: list[tuple[tuple, tuple[str, Hashable, Hashable], int, int]] = []
    for code, left_keys in left_by_code.items():
        right_keys = right_by_code.get(code)
        if not right_keys:
            continue
        for left_key in left_keys:
            left_layer = left_lookup[left_key]
            left_pos = left_index.get(id(left_layer))
            if left_pos is None:
                continue
            for right_key in right_keys:
                right_layer = right_lookup[right_key]
                right_pos = right_index.get(id(right_layer))
                if right_pos is None:
                    continue
                candidates.append(
                    (
                        _pair_rank(left_layer, right_layer, left_pos, right_pos),
                        (code, left_key, right_key),
                        left_pos,
                        right_pos,
                    )
                )
    accepted = _accept_non_crossing(candidates, matched_positions)
    crossed = _crossed_candidates(candidates, accepted, matched_positions)
    if not accepted and not crossed:
        return left_lookup, right_lookup
    merged_left = dict(left_lookup)
    merged_right = dict(right_lookup)
    for (code, left_key, right_key), left_pos, right_pos in accepted:
        new_key: Hashable = ("shifted", code, min(left_pos, right_pos), max(left_pos, right_pos))
        merged_left[new_key] = merged_left.pop(left_key)
        merged_right[new_key] = merged_right.pop(right_key)
    # Same-code units logged in opposite order relative to a kept match (e.g. Till
    # over Sand in one hole, Sand over Till in the next, numbered with unit_order so
    # their keys never matched) are the same geological conflict as a dropped
    # crossing match: re-key them so the pair summary reports a crossing.
    for side, key, pos in crossed:
        merged = merged_left if side == "left" else merged_right
        if key in merged:
            layer = merged.pop(key)
            merged[("crossed", side, pos, layer.lithology_code)] = layer
    return merged_left, merged_right


def _crossed_candidates(
    candidates: list[tuple[tuple, tuple[str, Hashable, Hashable], int, int]],
    accepted: list[tuple[object, int, int]],
    final_positions: Sequence[tuple[int, int]],
) -> list[tuple[str, Hashable, int]]:
    """Rejected same-code candidates whose only obstacle is a strict crossing.

    Both intervals must stay unmatched (no accepted pairing reuses either), and the
    pair must cross a final match without sharing an interval with it. Returned as
    ``(side, key, position)`` with ``side`` ``"left"`` / ``"right"``.
    """
    matched_left = {lp for lp, _rp in final_positions}
    matched_right = {rp for _lp, rp in final_positions}
    accepted_payloads = {id(payload) for payload, _lp, _rp in accepted}
    flagged: dict[tuple[str, int], tuple[str, Hashable, int]] = {}
    for _rank, payload, left_pos, right_pos in candidates:
        if id(payload) in accepted_payloads:
            continue
        if left_pos in matched_left or right_pos in matched_right:
            continue
        if not any(
            (left_pos < other_left) != (right_pos < other_right)
            for other_left, other_right in final_positions
        ):
            continue
        _code, left_key, right_key = payload
        flagged[("left", left_pos)] = ("left", left_key, left_pos)
        flagged[("right", right_pos)] = ("right", right_key, right_pos)
    return list(flagged.values())


def _pair_rank(
    left_layer: _LayerInterval,
    right_layer: _LayerInterval,
    left_pos: int,
    right_pos: int,
) -> tuple:
    """Symmetric (left/right-invariant) preference for correlating two intervals.

    Most shared elevation range first (thick units that clearly line up win), then
    nearest mid-elevations, then shallowest stratigraphic position. Floats are
    rounded so mirror-image transects tie exactly instead of on float noise.
    """
    overlap = min(left_layer.top_elevation, right_layer.top_elevation) - max(
        left_layer.bottom_elevation, right_layer.bottom_elevation
    )
    mid_offset = (
        abs(
            (left_layer.top_elevation + left_layer.bottom_elevation)
            - (right_layer.top_elevation + right_layer.bottom_elevation)
        )
        / 2.0
    )
    return (
        -round(max(overlap, 0.0), 6),
        round(mid_offset, 6),
        left_pos + right_pos,
        min(left_pos, right_pos),
        abs(left_pos - right_pos),
    )


def _stratigraphic_positions(intervals: list[_LayerInterval]) -> dict[int, int]:
    """Map ``id(interval)`` → rank from the top of the hole by elevation.

    ``_intervals_for_hole`` orders a hole by ``unit_order`` when it is set, and that
    numbering can contradict the logged depths (a typo, or a per-hole sequence that
    runs against depth). Crossing checks must compare the real stacking order, or
    two fills logged in opposite order would be accepted as non-crossing and the
    overlap clip would silently swallow one of them.
    """
    ranked = sorted(
        range(len(intervals)),
        key=lambda index: (-intervals[index].top_elevation, index),
    )
    return {id(intervals[index]): rank for rank, index in enumerate(ranked)}


def _crosses(left_pos: int, right_pos: int, positions: Sequence[tuple[int, int]]) -> bool:
    """Whether pairing ``left_pos``↔``right_pos`` crosses (or reuses) any accepted pair."""
    return any(
        left_pos == other_left
        or right_pos == other_right
        or (left_pos < other_left) != (right_pos < other_right)
        for other_left, other_right in positions
    )


def _accept_non_crossing(
    candidates: list[tuple[tuple, object, int, int]],
    fixed_positions: list[tuple[int, int]],
) -> list[tuple[object, int, int]]:
    """Accept candidate pairs best-rank first, keeping hole-local order (no crossings).

    Candidates of exactly equal rank are judged together: when they cross (or share
    an interval with) each other the choice is ambiguous, so none of them is
    accepted. That keeps the result symmetric under reversing the transect instead
    of depending on input order. ``fixed_positions`` is extended in place.
    """
    accepted: list[tuple[object, int, int]] = []
    ordered = sorted(candidates, key=lambda item: item[0])
    index = 0
    while index < len(ordered):
        group_end = index
        while group_end < len(ordered) and ordered[group_end][0] == ordered[index][0]:
            group_end += 1
        survivors = [
            (payload, left_pos, right_pos)
            for _rank, payload, left_pos, right_pos in ordered[index:group_end]
            if not _crosses(left_pos, right_pos, fixed_positions)
        ]
        group_accepted = [
            (payload, left_pos, right_pos)
            for position, (payload, left_pos, right_pos) in enumerate(survivors)
            if not _crosses(
                left_pos,
                right_pos,
                [(lp, rp) for i, (_p, lp, rp) in enumerate(survivors) if i != position],
            )
        ]
        accepted.extend(group_accepted)
        fixed_positions.extend((lp, rp) for _payload, lp, rp in group_accepted)
        index = group_end
    return accepted


def _drop_crossing_matches(
    left_intervals: list[_LayerInterval],
    right_intervals: list[_LayerInterval],
    left_lookup: dict[Hashable, _LayerInterval],
    right_lookup: dict[Hashable, _LayerInterval],
) -> tuple[dict[Hashable, _LayerInterval], dict[Hashable, _LayerInterval]]:
    """Un-match automatic correlations that cross each other between the two holes.

    Code-only (or order) keys can pair Sand-over-Clay in one hole with
    Clay-over-Sand in the next. Those fills cross, overlap clipping then keeps only
    the largest fragment, and a logged interval silently vanishes from the fence.
    Matches are kept best-rank first (``_pair_rank``) while they preserve both
    holes' order; the rest are re-keyed as ``('crossed', side, index, code)`` so they
    are drawn as pinch-outs anchored at their own holes and reported in the pair
    summary. Explicit ``('override', ...)`` correlations are always kept; matches
    keyed by user ``unit_order`` are hints and are un-matched like code-only ones.
    Positions are elevation ranks (``_stratigraphic_positions``), not the
    ``unit_order`` sort, so numbering that runs against depth cannot hide a cross.
    """
    left_index = _stratigraphic_positions(left_intervals)
    right_index = _stratigraphic_positions(right_intervals)
    fixed: list[tuple[int, int]] = []
    candidates: list[tuple[tuple, Hashable, int, int]] = []
    for key, left_layer in left_lookup.items():
        right_layer = right_lookup.get(key)
        if right_layer is None:
            continue
        left_pos = left_index.get(id(left_layer))
        right_pos = right_index.get(id(right_layer))
        if left_pos is None or right_pos is None:
            continue
        if isinstance(key, tuple) and key and key[0] == "override":
            fixed.append((left_pos, right_pos))
            continue
        candidates.append(
            (_pair_rank(left_layer, right_layer, left_pos, right_pos), key, left_pos, right_pos)
        )
    if len(candidates) + len(fixed) < 2:
        return left_lookup, right_lookup
    all_positions = fixed + [(lp, rp) for _rank, _key, lp, rp in candidates]
    if not any(
        _crosses(lp, rp, all_positions[:i] + all_positions[i + 1 :])
        for i, (lp, rp) in enumerate(all_positions)
    ):
        return left_lookup, right_lookup  # common case: nothing crosses
    kept = {key for key, _lp, _rp in _accept_non_crossing(candidates, fixed)}
    merged_left = dict(left_lookup)
    merged_right = dict(right_lookup)
    for _rank, key, left_pos, right_pos in candidates:
        if key in kept:
            continue
        left_layer = merged_left.pop(key)
        right_layer = merged_right.pop(key)
        merged_left[("crossed", "left", left_pos, left_layer.lithology_code)] = left_layer
        merged_right[("crossed", "right", right_pos, right_layer.lithology_code)] = right_layer
    return merged_left, merged_right


def _correlate_pair(
    left_hole_id: str,
    right_hole_id: str,
    left_intervals: list[_LayerInterval],
    right_intervals: list[_LayerInterval],
    left_lookup: dict[Hashable, _LayerInterval],
    right_lookup: dict[Hashable, _LayerInterval],
    overrides: Sequence[CorrelationOverride],
) -> tuple[dict[Hashable, _LayerInterval], dict[Hashable, _LayerInterval]]:
    """Apply explicit overrides, un-match crossing automatic correlations, then
    re-match same-code units shifted by an inserted unit."""
    left_lookup, right_lookup = _apply_correlation_overrides(
        left_hole_id,
        right_hole_id,
        left_intervals,
        right_intervals,
        left_lookup,
        right_lookup,
        overrides,
    )
    left_lookup, right_lookup = _drop_crossing_matches(
        left_intervals, right_intervals, left_lookup, right_lookup
    )
    return _rematch_shifted_units(left_intervals, right_intervals, left_lookup, right_lookup)


def _polygons_for_pair(
    left_hole_id: str,
    right_hole_id: str,
    x_left: float,
    x_right: float,
    left_intervals: list[_LayerInterval],
    right_intervals: list[_LayerInterval],
    *,
    allow_pinch_outs: bool = True,
    left_lookup: dict[Hashable, _LayerInterval] | None = None,
    right_lookup: dict[Hashable, _LayerInterval] | None = None,
    correlation_overrides: Sequence[CorrelationOverride] = (),
    pair_summaries: list[CorrelationPairSummary] | None = None,
) -> list[GeologicalPolygon]:
    polygons: list[GeologicalPolygon] = []
    if left_lookup is None:
        left_lookup = _correlation_keys(left_intervals)
    if right_lookup is None:
        right_lookup = _correlation_keys(right_intervals)
    left_lookup, right_lookup = _correlate_pair(
        left_hole_id,
        right_hole_id,
        left_intervals,
        right_intervals,
        left_lookup,
        right_lookup,
        correlation_overrides,
    )
    if pair_summaries is not None:
        pair_summaries.append(
            _correlation_pair_summary(
                left_hole_id,
                right_hole_id,
                left_lookup,
                right_lookup,
                allow_pinch_outs=allow_pinch_outs,
            )
        )
    if not x_right - x_left > 0.0:
        # Coincident holes (a nested pair logged at one projected position): no
        # fence area between them, so no zero-width polygons to warn about.
        return polygons
    all_keys = sorted(
        set(left_lookup) | set(right_lookup),
        key=lambda key: _correlation_sort_key(key, left_lookup, right_lookup),
    )
    hole_pair = (left_hole_id, right_hole_id)
    ground_left = _collar_elevation(left_intervals)
    ground_right = _collar_elevation(right_intervals)
    matched = [
        (key, left_lookup[key], right_lookup[key])
        for key in all_keys
        if key in left_lookup and key in right_lookup
    ]
    matched_right_first = [(key, right, left) for key, left, right in matched]

    # Pass 1: plan pinch-out wedges. Each wedge tapers to a tip on the "mean contact
    # line" of the matched units bracketing it, and those bracketing fills are bent
    # through the same tip so upper unit, wedge(s) and lower unit tile the fence.
    top_bends: dict[Hashable, list[tuple[float, float]]] = defaultdict(list)
    bottom_bends: dict[Hashable, list[tuple[float, float]]] = defaultdict(list)
    wedges: list[tuple[_LayerInterval, float, float, float]] = []
    # Bracketed wedges grouped by their ``(above, below)`` matched brackets. Groups
    # logged in both holes (a lateral facies change) and groups with one bracket
    # only (base / top of the fence) are tiled in pass 1b instead of tapering to
    # lone tips (see ``_tile_unmatched_region``).
    bracketed: dict[
        tuple[Hashable | None, Hashable | None],
        list[tuple[bool, tuple[_LayerInterval, float, float, float]]],
    ] = defaultdict(list)
    # Pinch-outs on or off, every unit stays drawn at the hole that logged it
    # (absorbing it into a bracketing fill drew the neighbour's unit over the
    # log). Off only shortens the reach of one-sided lenses to the near field.
    pinch_fraction = PINCH_OUT_FRACTION if allow_pinch_outs else NEAR_FIELD_PINCH_FRACTION
    for key in all_keys:
        left_layer = left_lookup.get(key)
        right_layer = right_lookup.get(key)
        if (left_layer is None) == (right_layer is None):
            continue
        if left_layer is not None:
            source_layer, x_source, x_other = left_layer, x_left, x_right
            source_matches = matched
        else:
            source_layer, x_source, x_other = right_layer, x_right, x_left  # type: ignore[assignment]
            source_matches = matched_right_first
        above, below = _bracketing_matches(source_layer, source_matches)
        x_tip = x_source + pinch_fraction * (x_other - x_source)
        if above is None and below is None:
            z_tip = _pinch_out_z_mid(
                source_layer,
                ground_at_tip=_interpolate(x_left, ground_left, x_right, ground_right, x_tip),
            )
            wedges.append((source_layer, x_source, x_other, z_tip))
            continue
        z_tip = _bracket_tip_z(
            above, below, left_lookup, right_lookup, x_left, x_right, x_tip
        )
        bracketed[(above, below)].append(
            (left_layer is not None, (source_layer, x_source, x_other, z_tip))
        )

    def _tile(
        above: Hashable | None,
        below: Hashable | None,
        members: list[tuple[bool, tuple[_LayerInterval, float, float, float]]],
    ) -> list[GeologicalPolygon] | None:
        return _tile_unmatched_region(
            above,
            below,
            [wedge[0] for is_left, wedge in members if is_left],
            [wedge[0] for is_left, wedge in members if not is_left],
            left_intervals,
            right_intervals,
            left_lookup,
            right_lookup,
            bottom_bends.get(above, ()) if above is not None else (),
            top_bends.get(below, ()) if below is not None else (),
            x_left,
            x_right,
            hole_pair,
        )

    # Pass 1b: two-sided groups first. Units logged only in the left hole facing
    # units logged only in the right hole between the same matched contacts are a
    # lateral facies change: they share the slot along a sloping boundary, each
    # thinning to zero at the opposite hole (two tips meeting mid-span drew a
    # bow-tie with the bracketing fills pinched to a point). A lens logged on one
    # side only keeps its tip, with the bracketing fills bent through it.
    open_polygons: list[GeologicalPolygon] = []
    for (above, below), members in bracketed.items():
        if above is None or below is None:
            continue
        sides = {is_left for is_left, _wedge in members}
        tiles = _tile(above, below, members) if len(sides) == 2 else None
        if tiles is not None:
            open_polygons.extend(tiles)
            continue
        for _is_left, wedge in members:
            _source, x_source, x_other, z_tip = wedge
            x_tip = x_source + pinch_fraction * (x_other - x_source)
            if _bend_keeps_fills_valid(
                above, below, left_lookup, right_lookup, x_left, x_right, x_tip, z_tip
            ):
                bottom_bends[above].append((x_tip, z_tip))
                top_bends[below].append((x_tip, z_tip))
            wedges.append(wedge)

    # Pass 1c: wedges open toward the fence base (or top). Opposing groups meet
    # along a sloping facies-change boundary; a lone group runs out to the
    # neighbour's hole end. Groups that cannot tile cleanly keep their tip wedges.
    for (above, below), members in bracketed.items():
        if above is not None and below is not None:
            continue
        tiles = _tile(above, below, members)
        if tiles is None:
            wedges.extend(wedge for _is_left, wedge in members)
        else:
            open_polygons.extend(tiles)

    # Pass 2: continuous fills (with bent contacts), then wedges.
    for key, left_layer, right_layer in matched:
        top_path = _bend_path(top_bends.get(key, ()), x_left, x_right)
        bottom_path = _bend_path(bottom_bends.get(key, ()), x_left, x_right)
        polygon = _make_polygon(
            [
                (x_left, left_layer.top_elevation),
                *top_path,
                (x_right, right_layer.top_elevation),
                (x_right, right_layer.bottom_elevation),
                *reversed(bottom_path),
                (x_left, left_layer.bottom_elevation),
            ],
            left_layer.lithology_code,
            hole_pair,
            # Below a short hole's TD the fill is interpolated, not logged.
            is_pinch_out=left_layer.inferred or right_layer.inferred,
        )
        if polygon:
            polygons.append(polygon)

    for source_layer, x_source, x_other, z_tip in wedges:
        polygon = _make_polygon(
            _pinch_out_wedge_coords(
                source_layer, x_source, x_other, z_tip, pinch_fraction=pinch_fraction
            ),
            source_layer.lithology_code,
            hole_pair,
            is_pinch_out=True,
        )
        if polygon:
            polygons.append(polygon)
    polygons.extend(open_polygons)

    return polygons


def _left_group_on_top(
    left_units: list[_LayerInterval],
    right_units: list[_LayerInterval],
    left_slot: tuple[float, float],
    right_slot: tuple[float, float],
) -> bool:
    """Which side of a lateral facies change overlies the other (mirror-invariant).

    The group whose slot sits higher (mid-elevation at its own hole) is drawn over
    the other, so the facies boundary runs from that hole's base contact to the
    other hole's upper contact. Ties fall back to the lithology codes, so
    reversing the transect keeps the same unit on top.
    """
    left_mid = round(0.5 * (left_slot[0] + left_slot[1]), 6)
    right_mid = round(0.5 * (right_slot[0] + right_slot[1]), 6)
    if left_mid != right_mid:
        return left_mid > right_mid

    def _codes(units: list[_LayerInterval]) -> tuple[str, ...]:
        ordered = sorted(units, key=lambda item: (-item.top_elevation, -item.bottom_elevation))
        return tuple(unit.lithology_code for unit in ordered)

    return _codes(left_units) <= _codes(right_units)


def _tile_unmatched_region(
    above: Hashable | None,
    below: Hashable | None,
    left_units: list[_LayerInterval],
    right_units: list[_LayerInterval],
    left_intervals: list[_LayerInterval],
    right_intervals: list[_LayerInterval],
    left_lookup: dict[Hashable, _LayerInterval],
    right_lookup: dict[Hashable, _LayerInterval],
    above_bends: Sequence[tuple[float, float]],
    below_bends: Sequence[tuple[float, float]],
    x_left: float,
    x_right: float,
    hole_pair: tuple[str, str],
) -> list[GeologicalPolygon] | None:
    """Tile unmatched units that share one bracketing slot without tip wedges.

    The slot runs between the bracketing contact(s) hole to hole: the base of the
    matched unit ``above`` and/or the top of the matched unit ``below``; with one
    bracket only, the fence base line (hole bottom to hole bottom) or top line
    (hole top to hole top) closes it. A tip wedge there would leave a white
    triangle against that line, and two opposing tips would pinch the bracketing
    fills to a point (a bow-tie). Instead:

    * **Opposing groups** (unit(s) X only in the left hole, Y only in the right,
      each filling its hole's slot) are a lateral facies change. They meet along
      a sloping boundary from one hole's base of the slot to the other hole's top
      of it, so each tapers to zero thickness at the hole that did not log it (no
      vertical wall mid-span). Which group overlies is decided by
      ``_left_group_on_top`` (mirror-invariant).
    * **A lone group** (open slot only: nothing logged opposite because the
      neighbour ends at the contact) extends to the neighbour, thinning to zero
      at its hole end — the neighbour simply did not penetrate the unit.

    Stacked units in a group keep their thickness proportions across the region.
    At least one of ``above`` / ``below`` must be set (unbracketed wedges keep
    their tips); a two-sided slot needs units on both sides (a one-sided lens
    keeps its tip and bends). Returns ``None`` (caller keeps the tip wedges) when
    a group does not fill its slot or the region is not well ordered.
    """
    if above is None and below is None:
        return None
    two_sided = above is not None and below is not None
    if two_sided and not (left_units and right_units):
        return None
    if not left_units and not right_units:
        return None
    span = x_right - x_left
    if not (np.isfinite(span) and span > 0.0):
        return None

    def _extent(intervals: list[_LayerInterval]) -> tuple[float, float]:
        return (
            max(interval.top_elevation for interval in intervals),
            min(interval.bottom_elevation for interval in intervals),
        )

    sides = {
        "left": (left_units, left_intervals, left_lookup, x_left),
        "right": (right_units, right_intervals, right_lookup, x_right),
    }
    upper_ends: dict[str, float] = {}
    lower_ends: dict[str, float] = {}
    for side, (units, intervals, lookup, _x) in sides.items():
        hole_top, hole_bottom = _extent(intervals)
        unit_top, unit_bottom = _extent(units) if units else (hole_top, hole_bottom)
        upper = lookup[above].bottom_elevation if above is not None else unit_top
        lower = lookup[below].top_elevation if below is not None else unit_bottom
        scale = max(1.0, abs(upper), abs(lower))
        tol = 1e-6 * scale
        if units:
            # The group must fill its hole's slot (to the hole end when open).
            if abs(unit_top - upper) > tol or abs(unit_bottom - lower) > tol:
                return None
            if above is None and abs(unit_top - hole_top) > tol:
                return None
            if below is None and abs(unit_bottom - hole_bottom) > tol:
                return None
        else:
            # Lone group: the neighbour must end exactly at the contact.
            if below is None and abs(hole_bottom - upper) > tol:
                return None
            if above is None and abs(hole_top - lower) > tol:
                return None
            lower = upper if below is None else lower
            upper = lower if above is None else upper
        if not (np.isfinite(upper) and np.isfinite(lower)) or upper < lower - tol:
            return None
        upper_ends[side], lower_ends[side] = upper, lower

    upper_path = [
        (x_left, upper_ends["left"]),
        *_bend_path(above_bends, x_left, x_right),
        (x_right, upper_ends["right"]),
    ]
    lower_path = [
        (x_left, lower_ends["left"]),
        *_bend_path(below_bends, x_left, x_right),
        (x_right, lower_ends["right"]),
    ]
    upper_xs = np.array([x for x, _z in upper_path])
    upper_zs = np.array([z for _x, z in upper_path])
    lower_xs = np.array([x for x, _z in lower_path])
    lower_zs = np.array([z for _x, z in lower_path])

    def _upper(x: float) -> float:
        return float(np.interp(x, upper_xs, upper_zs))

    def _lower(x: float) -> float:
        return float(np.interp(x, lower_xs, lower_zs))

    breaks = sorted({x for x, _z in upper_path} | {x for x, _z in lower_path})
    for x in breaks:
        if _upper(x) < _lower(x) - 1e-9 * max(1.0, abs(_upper(x))):
            return None

    # Each side's share of the slot as column fractions (0 = upper, 1 = lower) at x.
    if left_units and right_units:
        left_on_top = _left_group_on_top(
            left_units,
            right_units,
            (upper_ends["left"], lower_ends["left"]),
            (upper_ends["right"], lower_ends["right"]),
        )

        def _share(side: str, x: float) -> tuple[float, float]:
            if x <= x_left:
                t = 0.0
            elif x >= x_right:
                t = 1.0
            else:
                t = (x - x_left) / span
            if left_on_top:
                boundary = 1.0 - t
                return (0.0, boundary) if side == "left" else (boundary, 1.0)
            return (t, 1.0) if side == "left" else (0.0, t)
    else:

        def _share(side: str, x: float) -> tuple[float, float]:
            return 0.0, 1.0

    def _curve(side: str, fraction: float, xs: list[float]) -> list[tuple[float, float]]:
        points = []
        for x in xs:
            upper, lower = _upper(x), _lower(x)
            start, stop = _share(side, x)
            # Exact ends keep shared edges bit-identical between tiles.
            column = start if fraction <= 0.0 else stop if fraction >= 1.0 else (
                start + fraction * (stop - start)
            )
            if column <= 0.0:
                z = upper
            elif column >= 1.0:
                z = lower
            else:
                z = upper - column * (upper - lower)
            points.append((x, z))
        return points

    xs = [x_left, *(x for x in breaks if x_left < x < x_right), x_right]
    tiles: list[GeologicalPolygon] = []
    for side, (units, _intervals, _lookup, x_source) in sides.items():
        if not units:
            continue
        source_upper, source_lower = _upper(x_source), _lower(x_source)
        thickness = source_upper - source_lower
        if thickness <= 0.0:
            return None
        for unit in sorted(units, key=lambda item: (-item.top_elevation, -item.bottom_elevation)):
            f_top = (source_upper - unit.top_elevation) / thickness
            f_bottom = (source_upper - unit.bottom_elevation) / thickness
            top_curve = _curve(side, f_top, xs)
            bottom_curve = _curve(side, f_bottom, xs)
            index = 0 if x_source == xs[0] else -1
            top_curve[index] = (x_source, unit.top_elevation)
            bottom_curve[index] = (x_source, unit.bottom_elevation)
            polygon = _make_polygon(
                [*top_curve, *reversed(bottom_curve)],
                unit.lithology_code,
                hole_pair,
                is_pinch_out=True,
            )
            if polygon:
                tiles.append(polygon)
    return tiles


def _bracketing_matches(
    pinch_interval: _LayerInterval,
    matched: list[tuple[Hashable, _LayerInterval, _LayerInterval]],
) -> tuple[Hashable | None, Hashable | None]:
    """Correlation keys of the nearest matched units above and below a pinch-out.

    ``matched`` holds ``(key, source_interval, neighbour_interval)`` for every unit
    correlated across the pair; "above"/"below" are judged in the source hole.
    """
    above: tuple[float, Hashable] | None = None
    below: tuple[float, Hashable] | None = None
    for key, source, _neighbour in matched:
        if source.bottom_elevation >= pinch_interval.top_elevation - 1e-9:
            if above is None or source.bottom_elevation < above[0]:
                above = (source.bottom_elevation, key)
        elif source.top_elevation <= pinch_interval.bottom_elevation + 1e-9:
            if below is None or source.top_elevation > below[0]:
                below = (source.top_elevation, key)
    return (
        above[1] if above is not None else None,
        below[1] if below is not None else None,
    )


def _bracket_tip_z(
    above: Hashable | None,
    below: Hashable | None,
    left_lookup: dict[Hashable, _LayerInterval],
    right_lookup: dict[Hashable, _LayerInterval],
    x_left: float,
    x_right: float,
    x_tip: float,
) -> float:
    """Wedge-tip elevation on the mean line of the bracketing matched contacts.

    The contact line of each bracketing unit (bottom of the unit above, top of the
    unit below) is interpolated hole-to-hole; the tip sits on their average at
    ``x_tip``. With a single bracketing unit the tip lies exactly on its straight
    contact, so that fill needs no bending; with two, both fills are bent through
    the tip and stay ordered (the tip lies between their interpolated contacts).
    """
    left_contacts: list[float] = []
    right_contacts: list[float] = []
    if above is not None:
        left_contacts.append(left_lookup[above].bottom_elevation)
        right_contacts.append(right_lookup[above].bottom_elevation)
    if below is not None:
        left_contacts.append(left_lookup[below].top_elevation)
        right_contacts.append(right_lookup[below].top_elevation)
    z_left = sum(left_contacts) / len(left_contacts)
    z_right = sum(right_contacts) / len(right_contacts)
    span = x_right - x_left
    t = (x_tip - x_left) / span if span else 0.5
    return z_left + t * (z_right - z_left)


def _bend_keeps_fills_valid(
    above: Hashable,
    below: Hashable,
    left_lookup: dict[Hashable, _LayerInterval],
    right_lookup: dict[Hashable, _LayerInterval],
    x_left: float,
    x_right: float,
    x_tip: float,
    z_tip: float,
) -> bool:
    """Whether bending both bracketing fills through ``(x_tip, z_tip)`` stays sane.

    Crossing correlations (explicit overrides, or code keys matched in crossing
    order) can put the "above" unit below the "below" unit in one hole. Bending
    then drags a fill's bottom edge above its own top, and ``buffer(0)`` repair
    silently drops area. Only bend when the bracketing contacts are ordered in
    both holes and the tip lies inside both fills' vertical extent at ``x_tip``.
    """
    tol = 1e-9
    upper_left, upper_right = left_lookup[above], right_lookup[above]
    lower_left, lower_right = left_lookup[below], right_lookup[below]
    if upper_left.bottom_elevation < lower_left.top_elevation - tol:
        return False
    if upper_right.bottom_elevation < lower_right.top_elevation - tol:
        return False
    span = x_right - x_left
    t = (x_tip - x_left) / span if span else 0.5

    def _at_tip(left_z: float, right_z: float) -> float:
        return left_z + t * (right_z - left_z)

    upper_top = _at_tip(upper_left.top_elevation, upper_right.top_elevation)
    lower_bottom = _at_tip(lower_left.bottom_elevation, lower_right.bottom_elevation)
    return lower_bottom - tol <= z_tip <= upper_top + tol


def _bend_path(
    bends: Sequence[tuple[float, float]], x_left: float, x_right: float
) -> list[tuple[float, float]]:
    """Interior bend vertices of a contact edge, ordered left→right, deduplicated by x."""
    by_x: dict[float, float] = {}
    for x, z in bends:
        if min(x_left, x_right) < x < max(x_left, x_right):
            by_x.setdefault(x, z)
    return sorted(by_x.items())


def _pinch_out_wedge_coords(
    source_layer: _LayerInterval,
    x_source: float,
    x_other: float,
    z_apex: float,
    *,
    pinch_fraction: float = PINCH_OUT_FRACTION,
) -> list[tuple[float, float]]:
    """Triangle anchored on the logged interval at the source hole, tapering toward the neighbour.

    The wedge's vertical edge sits exactly on ``x_source`` from the logged top to the
    logged bottom elevation (so the fill attaches to the borehole column); the apex
    lies ``pinch_fraction`` of the way toward ``x_other`` at ``z_apex``.
    """
    x_apex = x_source + pinch_fraction * (x_other - x_source)
    return [
        (x_source, source_layer.top_elevation),
        (x_source, source_layer.bottom_elevation),
        (x_apex, z_apex),
    ]


def _largest_polygon(geom) -> Polygon | None:
    if geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda part: part.area)
    if geom.geom_type == "GeometryCollection":
        parts = [part for part in geom.geoms if part.geom_type == "Polygon" and not part.is_empty]
        if not parts:
            return None
        return max(parts, key=lambda part: part.area)
    return None


def _pinch_anchor_edge(polygon: Polygon, hole_xs: Sequence[float] = ()) -> LineString | None:
    """Longest vertical exterior edge of a pinch-out wedge (its logged interval at the hole).

    With ``hole_xs``, edges on a hole position win over interior vertical edges
    (a facies-change tile also has a vertical edge at mid-span).
    """
    coords = list(polygon.exterior.coords)
    best: tuple[bool, float, LineString] | None = None
    for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
        span = abs(y1 - y0)
        tol = 1e-9 * max(1.0, abs(x0))
        if span <= 0.0 or abs(x1 - x0) > tol:
            continue
        on_hole = any(abs(x0 - hole_x) <= tol for hole_x in hole_xs)
        if best is None or (on_hole, span) > best[:2]:
            best = (on_hole, span, LineString([(x0, y0), (x1, y1)]))
    return best[2] if best is not None else None


def _anchored_fragment(geom, anchor: LineString | None) -> Polygon | None:
    """Pick the clip fragment that still touches ``anchor``; fall back to the largest."""
    if anchor is None or geom.is_empty or geom.geom_type == "Polygon":
        return _largest_polygon(geom)
    parts = [
        part
        for part in getattr(geom, "geoms", ())
        if part.geom_type == "Polygon" and not part.is_empty
    ]
    attached = [(_overlay("intersection", part.boundary, anchor).length, part.area) for part in parts]
    candidates = [
        (contact, area, part) for (contact, area), part in zip(attached, parts) if contact > 0.0
    ]
    if not candidates:
        return _largest_polygon(geom)
    return max(candidates, key=lambda item: (item[1], item[0]))[2]


# Fraction of a fence polygon's area that overlap clipping may discard before the
# loss is reported to the user (``clip_warnings``) rather than only logged.
CLIP_LOSS_WARN_FRACTION = 0.15

# Coordinate grid (m) the overlap clipping works on. Fence polygons of one pair
# share edges exactly (a wedge's top edge *is* the fill's base; wedges from one
# hole taper to one tip), and GEOS's floating overlay is not robust on such
# input: it reported a 4 mm sliver wedge as lying wholly inside the fill it
# touched (``difference`` → empty) and split a ring joined by a ~1e-13 m "neck"
# into one polygon in one direction and two in the other — so a reversed
# transect kept different wedges (fuzz seeds 7308, 7440, 7428). With a fixed
# precision model the overlay uses snap rounding, which is exact: both
# directions give the same fragments. A nanometre is the order of the float
# noise the floating overlay already leaves on intersection points, and far
# below anything drawn.
CLIP_PRECISION_GRID_M = 1e-9

# Overlap detection ignores an intersection thinner than this across the two
# polygons' perimeters. Snap rounding leaves a clipped edge's nodes up to half
# a grid step off the neighbour it was cut against, so polygons that only share
# an edge can "overlap" by grid × edge length (2e-7 m² on a 7 km perimeter);
# the floating overlay left ~1e-9 m² slivers of the same kind.
OVERLAP_SLIVER_WIDTH_M = 10.0 * CLIP_PRECISION_GRID_M


def _snap_for_overlay(geom):
    """``geom`` rounded to ``CLIP_PRECISION_GRID_M`` and carrying that precision
    model, so overlays on it snap-round (robust) instead of floating.

    Pointwise: vertices are rounded in place (ring order and start unchanged;
    every polygon of a pair is rounded the same way, so shared edges stay
    shared). Falls back to the input when rounding would leave it invalid.
    """
    try:
        snapped = shapely.set_precision(geom, CLIP_PRECISION_GRID_M, mode="pointwise")
    except (GEOSException, ValueError):  # pragma: no cover - defensive
        return geom
    if snapped is None or snapped.is_empty or not snapped.is_valid:
        return geom
    return snapped


def _unsnapped(geom):
    """``geom`` without a precision model (floating overlay fallback)."""
    try:
        plain = shapely.set_precision(geom, 0.0)
    except (GEOSException, ValueError):  # pragma: no cover - defensive
        plain = geom
    if plain is None or plain.is_empty:
        return plain
    return plain if plain.is_valid else plain.buffer(0)


def _overlay(op: str, left, right):
    """``left.<op>(right)`` on the precision grid, falling back to a floating
    overlay when GEOS's snap-rounding rejects the pair (seen with coordinates
    near 5e8 m: "Ring edge missing"). The result keeps whichever model worked."""
    try:
        return getattr(left, op)(right)
    except GEOSException:
        logger.debug("Grid overlay failed (%s); retrying without the precision model", op)
        return getattr(_unsnapped(left), op)(_unsnapped(right))


def _union_all(parts):
    try:
        return unary_union(parts) if len(parts) > 1 else parts[0]
    except GEOSException:
        logger.debug("Grid union failed; retrying without the precision model")
        plain = [_unsnapped(part) for part in parts]
        return unary_union(plain) if len(plain) > 1 else plain[0]


def _resolve_overlaps_in_pair(
    polygons: list[GeologicalPolygon],
    *,
    clip_warnings: list[str] | None = None,
    hole_xs: Sequence[float] = (),
    log_discards: bool = True,
) -> list[GeologicalPolygon]:
    """Clip deeper fence polygons so inter-hole fills do not stack on top of shallower units.

    When ``clip_warnings`` is given, a user-facing message is appended for every
    polygon that loses more than ``CLIP_LOSS_WARN_FRACTION`` of its area (or is
    dropped entirely) — a logged interval partly missing from the fence.
    """
    if len(polygons) <= 1:
        return polygons

    from shapely.prepared import prep

    ordered = sorted(
        polygons,
        key=lambda item: (
            item.is_pinch_out,
            -item.polygon.bounds[3],
            item.polygon.bounds[1],
        ),
    )
    resolved: list[GeologicalPolygon] = []
    occupied_geom: Polygon | None = None
    occupied_prep = None
    batch: list[Polygon] = []
    batch_limit = 4
    # With no matched unit in the pair every fill is a wedge; wedges trimming
    # each other is expected geometry, not a correlation conflict to report.
    warn_on_clip = clip_warnings is not None and not all(p.is_pinch_out for p in polygons)
    last_index = len(ordered) - 1
    for index, geo_polygon in enumerate(ordered):
        original_area = float(geo_polygon.polygon.area)
        # Every overlay below runs on the fixed grid (see CLIP_PRECISION_GRID_M);
        # the polygon is returned on that grid too, clipped or not.
        snapped = _snap_for_overlay(geo_polygon.polygon)
        geom = snapped
        if occupied_geom is not None and not occupied_geom.is_empty:
            # Prepared predicates for skip checks; difference still uses raw geom.
            probe = occupied_prep if occupied_prep is not None else occupied_geom
            if probe.intersects(geom) and not probe.touches(geom):
                geom = _overlay("difference", geom, occupied_geom)
        # Polygons kept earlier in the current (not yet unioned) batch also occupy
        # area; without this, up to ``batch_limit`` fills could stack unclipped.
        for pending in batch:
            if geom.is_empty:
                break
            if pending.intersects(geom) and not pending.touches(geom):
                geom = _overlay("difference", geom, pending)
        if geo_polygon.is_pinch_out and geom is not snapped:
            # A wedge must stay attached to the hole where the unit was logged; the
            # largest clip fragment can be a sliver floating mid-way between holes.
            # The anchor comes from the snapped wedge, so it meets the fragment's
            # hole edge exactly.
            largest = _anchored_fragment(geom, _pinch_anchor_edge(snapped, hole_xs))
        else:
            largest = _largest_polygon(geom)
        if largest is not None and not largest.is_empty and largest is not snapped:
            # Back on the grid after the vertex clean-up (a rebuilt Polygon has
            # no precision model), so later unions stay snap-rounded.
            largest = _snap_for_overlay(_clean_polygon(largest))
        kept_area = 0.0 if largest is None or largest.is_empty else float(largest.area)
        if (
            warn_on_clip
            and original_area > 0
            and kept_area < (1.0 - CLIP_LOSS_WARN_FRACTION) * original_area
        ):
            clip_warnings.append(
                f"Fence clipped: {geo_polygon.lithology_code} between "
                f"{geo_polygon.hole_pair[0]}–{geo_polygon.hole_pair[1]} kept "
                f"{100.0 * kept_area / original_area:.0f}% of its area "
                "(conflicting correlation; check unit order or overrides)"
            )
        if largest is None or largest.is_empty:
            continue
        # Difference + MultiPolygon keep-largest can discard secondary fragments.
        if log_discards and original_area > 0 and largest.area < 0.85 * original_area:
            logger.warning(
                "Overlap clip discarded fragments for %s between %s–%s "
                "(kept %.0f%% of original area)",
                geo_polygon.lithology_code,
                geo_polygon.hole_pair[0],
                geo_polygon.hole_pair[1],
                100.0 * largest.area / original_area,
            )
        resolved.append(
            GeologicalPolygon(
                lithology_code=geo_polygon.lithology_code,
                polygon=largest,
                hole_pair=geo_polygon.hole_pair,
                is_pinch_out=geo_polygon.is_pinch_out,
            )
        )
        batch.append(largest)
        if len(batch) >= batch_limit or index == last_index:
            chunk = _union_all(batch)
            occupied_geom = chunk if occupied_geom is None else _overlay("union", occupied_geom, chunk)
            occupied_prep = (
                prep(occupied_geom)
                if occupied_geom is not None and not occupied_geom.is_empty
                else None
            )
            batch.clear()
    return resolved


@dataclass(frozen=True)
class PolygonOverlap:
    left_lithology_code: str
    right_lithology_code: str
    hole_pair: tuple[str, str]
    centroid_x: float
    centroid_y: float

    def message(self) -> str:
        return (
            f"{self.left_lithology_code} / {self.right_lithology_code} "
            f"between {self.hole_pair[0]}–{self.hole_pair[1]}"
        )


def detect_polygon_overlaps(polygons: list[GeologicalPolygon]) -> list[PolygonOverlap]:
    """Return overlapping polygon pairs with profile-plane centroids for annotation."""
    overlaps: list[PolygonOverlap] = []
    if len(polygons) < 2:
        return overlaps

    by_pair: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, polygon in enumerate(polygons):
        by_pair[polygon.hole_pair].append(index)

    for indices in by_pair.values():
        if len(indices) < 2:
            continue
        geoms = [polygons[index].polygon for index in indices]
        tree = STRtree(geoms)
        for left_pos, left_index in enumerate(indices):
            left = polygons[left_index]
            left_geom = geoms[left_pos]
            # Use intersects (not overlaps alone) so nested/contained pairs are found.
            for right_pos in tree.query(left_geom, predicate="intersects"):
                if right_pos <= left_pos:
                    continue
                right_index = indices[right_pos]
                right = polygons[right_index]
                intersection = _overlay("intersection", left.polygon, right.polygon)
                # Skip empty/tiny area and pure line touches: anything thinner
                # than ``OVERLAP_SLIVER_WIDTH_M`` across the pair's perimeter is
                # a precision artefact of the clip grid, not a correlation conflict.
                sliver_area = max(
                    1e-9, OVERLAP_SLIVER_WIDTH_M * (left.polygon.length + right.polygon.length)
                )
                if intersection.is_empty or intersection.area < sliver_area:
                    continue
                # GEOS overlay on near-degenerate rings is direction-sensitive; a
                # real overlap has area both ways round, a precision artefact not.
                reverse = _overlay("intersection", right.polygon, left.polygon)
                if reverse.is_empty or reverse.area < sliver_area:
                    continue
                centroid = intersection.centroid
                overlaps.append(
                    PolygonOverlap(
                        left_lithology_code=left.lithology_code,
                        right_lithology_code=right.lithology_code,
                        hole_pair=left.hole_pair,
                        centroid_x=float(centroid.x),
                        centroid_y=float(centroid.y),
                    )
                )
    return overlaps


def log_polygon_overlaps(overlaps: Sequence[PolygonOverlap]) -> None:
    for overlap in overlaps:
        logger.warning(
            "Overlapping polygons detected for %s and %s between %s",
            overlap.left_lithology_code,
            overlap.right_lithology_code,
            overlap.hole_pair,
        )


@dataclass(frozen=True)
class CorrelationPairSummary:
    left_hole_id: str
    right_hole_id: str
    matched_count: int
    left_only_codes: tuple[str, ...]
    right_only_codes: tuple[str, ...]
    pinch_out_candidates: int
    # Lithology codes whose automatic correlation crossed another one between the
    # two holes and was un-matched (drawn as pinch-outs instead); see
    # ``_drop_crossing_matches``.
    crossing_codes: tuple[str, ...] = ()

    @property
    def unmatched_keys_count(self) -> int:
        return len(self.left_only_codes) + len(self.right_only_codes)

    @property
    def match_rate(self) -> float:
        total = self.matched_count + self.unmatched_keys_count
        if total == 0:
            return 1.0
        return self.matched_count / total


def _is_zero_thickness(interval: _LayerInterval) -> bool:
    """True for a logged interval with no thickness (``from_depth == to_depth``)."""
    thickness = interval.top_elevation - interval.bottom_elevation
    return bool(np.isfinite(thickness)) and abs(thickness) < MIN_FENCE_THICKNESS_M


def _sorted_hole_profiles(
    projected_df: pd.DataFrame,
    *,
    fence_notes: list[str] | None = None,
) -> list[tuple[float, str, list[_LayerInterval], dict[Hashable, _LayerInterval]]]:
    """Build x-sorted hole interval profiles from a projected DataFrame.

    Holes at one projected position (a nested pair, or a section line at right
    angles to their offset) are collapsed to one profile — the deeper hole's log
    (equal depths: the first hole ID) — so the fence either side of the position
    is built from the same log whichever way the transect runs. The others are
    drawn as columns only; a note per dropped hole is appended to ``fence_notes``
    (see ``_collapse_coincident_holes``).
    """
    if projected_df.empty:
        return []
    x_profile = projected_df["x_profile"]
    sorted_df = (
        projected_df
        if x_profile.is_monotonic_increasing
        else projected_df.sort_values("x_profile", kind="mergesort")
    )
    hole_profiles: list[
        tuple[float, str, list[_LayerInterval], dict[Hashable, _LayerInterval]]
    ] = []
    for hole_id, group in sorted_df.groupby("hole_id", sort=False):
        intervals = _close_logging_gaps(
            [
                interval
                for interval in _intervals_for_hole(group)
                if not _is_zero_thickness(interval)
            ]
        )
        hole_profiles.append(
            (
                float(group["x_profile"].iloc[0]),
                str(hole_id),
                intervals,
                _correlation_keys(intervals),
            )
        )
    # Deeper hole first at equal x, then hole ID: a tie-break that does not
    # depend on transect direction or DataFrame row order.
    hole_profiles.sort(key=lambda item: (item[0], -_hole_depth_or_zero(item[2]), item[1]))
    return _collapse_coincident_holes(hole_profiles, fence_notes=fence_notes)


# A hole is "short" (continued beneath its total depth by ``_extend_short_holes``)
# when the nearest holes on both sides end LOWER in elevation (bottom RL) by at
# least ``max(SHORT_HOLE_MIN_DEPTH_GAIN_M, SHORT_HOLE_MIN_DEPTH_GAIN_FRACTION * TD)``
# (TD = its depth below collar). Elevation, not depth, because the fence is built
# in elevation: a hole on a hill can be nearly as deep as its neighbours yet end
# far above them. Ordinary differences keep the convention that the fence base
# follows the hole bottoms; a hole ending well above both neighbours (a shallow
# well of a nested pair, a hole on a hill) would otherwise notch a white wedge
# into the fence.
SHORT_HOLE_MIN_DEPTH_GAIN_M = 1.0
SHORT_HOLE_MIN_DEPTH_GAIN_FRACTION = 0.25

_HoleProfile = tuple[float, str, list[_LayerInterval], dict[Hashable, _LayerInterval]]


# Holes whose projected positions differ by no more than this are "coincident":
# the fence between them would be a zero-width strip.
COINCIDENT_HOLE_TOLERANCE_M = 1e-6


def _hole_depth(intervals: list[_LayerInterval]) -> float:
    """Deepest logged depth below collar (the hole's TD as far as the fence knows)."""
    depths = [i.to_depth for i in intervals if np.isfinite(i.to_depth)]
    return max(depths) if depths else float("nan")


def _hole_bottom(intervals: list[_LayerInterval]) -> float:
    """Lowest logged elevation (RL of the hole's TD as far as the fence knows)."""
    bottoms = [i.bottom_elevation for i in intervals if np.isfinite(i.bottom_elevation)]
    return min(bottoms) if bottoms else float("nan")


def _hole_depth_or_zero(intervals: list[_LayerInterval]) -> float:
    """``_hole_depth`` with an unlogged hole counted as 0 m (sortable)."""
    depth = _hole_depth(intervals)
    return depth if np.isfinite(depth) else 0.0


def _collapse_coincident_holes(
    hole_profiles: list[_HoleProfile],
    *,
    fence_notes: list[str] | None = None,
) -> list[_HoleProfile]:
    """One fence profile per projected position (``hole_profiles`` sorted by x).

    Two holes at one position (MW-06S / MW-06D) used to stay separate: the fence
    on the left paired with whichever came first, the fence on the right with
    the other, and reversing the transect swapped them — the section depended on
    its direction, and one log was silently absent from each side. Now the
    deeper hole's log (equal depths: the first hole ID, the sort order of
    ``_sorted_hole_profiles``) represents the position on both sides and every
    other hole there is left to its column; a note naming the pair and the log
    used is appended to ``fence_notes``.
    """

    def _rank(profile: _HoleProfile) -> tuple[float, str]:
        return (-_hole_depth_or_zero(profile[2]), profile[1])

    groups: list[list[_HoleProfile]] = []
    for profile in hole_profiles:
        # Compare with the group's first (x-lowest) member so a run of nearly
        # coincident holes cannot drift past the tolerance one step at a time.
        if groups and abs(profile[0] - groups[-1][0][0]) <= COINCIDENT_HOLE_TOLERANCE_M:
            groups[-1].append(profile)
        else:
            groups.append([profile])
    collapsed: list[_HoleProfile] = []
    for group in groups:
        kept = min(group, key=_rank)
        collapsed.append(kept)
        if fence_notes is None:
            continue
        kept_depth = _hole_depth_or_zero(kept[2])
        for dropped in group:
            if dropped is kept:
                continue
            reason = (
                "deeper"
                if kept_depth > _hole_depth_or_zero(dropped[2])
                else "same depth, first by hole ID"
            )
            fence_notes.append(
                f"{dropped[1]} and {kept[1]} share one position; the fence uses "
                f"{kept[1]}'s log ({reason}) — {dropped[1]} is drawn as a column only"
            )
    return collapsed


def _fence_column_at(polygons: list[GeologicalPolygon], x: float) -> list[tuple[str, float, float]]:
    """``(code, top, bottom)`` runs where the vertical line at ``x`` crosses the fence,
    top-down; touching runs of one code are merged."""
    runs: list[tuple[str, float, float]] = []
    for item in polygons:
        min_x, min_z, max_x, max_z = item.polygon.bounds
        if not (min_x < x < max_x):
            continue
        probe = LineString([(x, min_z - 1.0), (x, max_z + 1.0)])
        try:
            hit = item.polygon.intersection(probe)
        except GEOSException:  # pragma: no cover - defensive
            hit = _unsnapped(item.polygon).intersection(probe)
        for part in getattr(hit, "geoms", [hit]):
            if part.is_empty or part.geom_type != "LineString":
                continue
            zs = [z for _x, z in part.coords]
            if max(zs) - min(zs) >= MIN_FENCE_THICKNESS_M:
                runs.append((item.lithology_code, max(zs), min(zs)))
    runs.sort(key=lambda run: (-run[1], -run[2]))
    merged: list[tuple[str, float, float]] = []
    for code, top, bottom in runs:
        if merged and merged[-1][0] == code and abs(merged[-1][2] - top) <= 1e-6:
            merged[-1] = (code, merged[-1][1], min(bottom, merged[-1][2]))
        else:
            merged.append((code, top, bottom))
    return merged


def _below_td_intervals(
    intervals: list[_LayerInterval], column: list[tuple[str, float, float]]
) -> list[_LayerInterval] | None:
    """``intervals`` continued below the hole's TD with the fence ``column`` beneath it.

    The deepest logged unit is extended when the column continues the same unit
    across the TD; every other run below the TD becomes an ``inferred`` interval.
    Returns ``None`` when nothing is added.

    A hole on a hill can stand above the neighbours' fence altogether: the column
    then starts below the TD (between the neighbours' collars), and its upper runs
    are units the hole already logged higher up. Closing that gap at its midpoint
    put an inferred copy of e.g. the Sand the hole logged at surface beneath its
    logged Clay. Instead the runs above the column's first run of the deepest
    logged unit (or, without one, the leading runs of units the hole logged) are
    dropped, and the deepest logged unit continues down across the gap.
    """
    finite = [
        interval
        for interval in intervals
        if np.isfinite(interval.top_elevation) and np.isfinite(interval.bottom_elevation)
    ]
    if not finite:
        return None
    deepest = min(finite, key=lambda interval: (interval.bottom_elevation, interval.top_elevation))
    td = deepest.bottom_elevation
    collar = deepest.top_elevation + deepest.from_depth
    below = [
        (code, min(top, td), bottom)
        for code, top, bottom in column
        if bottom < td - MIN_FENCE_THICKNESS_M
    ]
    if not below:
        return None
    extended = list(intervals)
    index = next(i for i, interval in enumerate(extended) if interval is deepest)
    code, top, bottom = below[0]
    if top < td - 1e-6:
        # The neighbours' fence passes beneath the TD (a hole on a hill).
        column_bottom = below[-1][2]
        match = next((i for i, run in enumerate(below) if run[0] == deepest.lithology_code), None)
        if match is not None:
            below = below[match:]
        else:
            logged = {interval.lithology_code for interval in intervals}
            while below and below[0][0] in logged:
                below = below[1:]
        if below and below[0][0] == deepest.lithology_code:
            continued_to = below[0][2]
            below = below[1:]
        else:
            continued_to = below[0][1] if below else column_bottom
        extended[index] = replace(
            deepest, bottom_elevation=continued_to, to_depth=collar - continued_to
        )
    elif code == deepest.lithology_code and top >= td - 1e-6:
        extended[index] = replace(deepest, bottom_elevation=bottom, to_depth=collar - bottom)
        below = below[1:]
    for code, top, bottom in below:
        if top - bottom < MIN_FENCE_THICKNESS_M:
            continue
        extended.append(
            _LayerInterval(
                lithology_code=code,
                from_depth=collar - top,
                to_depth=collar - bottom,
                top_elevation=top,
                bottom_elevation=bottom,
                inferred=True,
            )
        )
    return _close_logging_gaps(extended)


def _extend_short_holes(
    hole_profiles: list[_HoleProfile],
    *,
    allow_pinch_outs: bool,
    override_index: dict[tuple[str, str], tuple[CorrelationOverride, ...]],
) -> list[_HoleProfile]:
    """Continue units beneath holes that stop short of both neighbours (fence copy only).

    A short hole between two deeper ones (MW-06S between MW-05 and MW-06D) pinched
    the fence base line up to its TD: a white wedge under the hole, with the
    deeper units squeezed into slivers. The strata do not end because a hole
    did: below its TD the hole gets the column the fence between its nearest
    lower-ending neighbours shows at its position (interpolated, matched and tiled
    exactly like any other pair), flagged ``inferred`` so those fills are drawn
    as pinch-outs (dashed, faded). "Short" compares the *elevations* of the hole
    bottoms (see ``SHORT_HOLE_MIN_DEPTH_GAIN_M``), the frame the fence is built
    in: judged on depth below collar, a hole on a hill only a little shallower
    than its neighbours (but ending far above their bottoms) was not extended and
    a white notch opened under it, while a hole in a valley (shallower, but its
    bottom no higher than theirs) is not short at all. Holes are extended lowest
    bottom first (equal bottoms together, so the result does not depend on
    transect direction), so a run of short holes chains off the already-extended
    ones. Column / projected data are untouched.
    """
    if len(hole_profiles) < 3:
        return hole_profiles
    profiles = list(hole_profiles)
    depths = [_hole_depth(intervals) for _x, _id, intervals, _lookup in profiles]
    bottoms = [_hole_bottom(intervals) for _x, _id, intervals, _lookup in profiles]
    by_bottom: dict[float, list[int]] = defaultdict(list)
    for index in range(1, len(profiles) - 1):
        if np.isfinite(depths[index]) and np.isfinite(bottoms[index]):
            by_bottom[round(bottoms[index], 6)].append(index)
    for level in sorted(by_bottom):
        updates: dict[int, _HoleProfile] = {}
        for index in by_bottom[level]:
            bottom = bottoms[index]
            needed = bottom - max(
                SHORT_HOLE_MIN_DEPTH_GAIN_M, SHORT_HOLE_MIN_DEPTH_GAIN_FRACTION * depths[index]
            )
            # Nearest hole each side whose bottom is at least as low as this one
            # (only holes ending higher are skipped, so a run of short holes
            # chains); it must reach down to ``needed``, or the hole is not short
            # next to that neighbour.
            left = next((i for i in range(index - 1, -1, -1) if bottoms[i] <= bottom), None)
            right = next((i for i in range(index + 1, len(profiles)) if bottoms[i] <= bottom), None)
            if left is None or right is None or max(bottoms[left], bottoms[right]) > needed:
                continue
            x_hole, hole_id, intervals, _lookup = profiles[index]
            x_left, left_id, left_intervals, left_lookup = profiles[left]
            x_right, right_id, right_intervals, right_lookup = profiles[right]
            # Profiles hold one hole per position (``_collapse_coincident_holes``),
            # so the pair spans a strictly positive width.
            if not x_left < x_hole < x_right:  # pragma: no cover - sorted by x
                continue
            reference = _resolve_overlaps_in_pair(
                _polygons_for_pair(
                    left_hole_id=left_id,
                    right_hole_id=right_id,
                    x_left=x_left,
                    x_right=x_right,
                    left_intervals=left_intervals,
                    right_intervals=right_intervals,
                    allow_pinch_outs=allow_pinch_outs,
                    left_lookup=left_lookup,
                    right_lookup=right_lookup,
                    correlation_overrides=_overrides_for_hole_pair(
                        left_id, right_id, override_index
                    ),
                ),
                hole_xs=(x_left, x_right),
                # Internal reference between non-adjacent holes: not a drawn fence.
                log_discards=False,
            )
            extended = _below_td_intervals(intervals, _fence_column_at(reference, x_hole))
            if extended is not None:
                updates[index] = (x_hole, hole_id, extended, _correlation_keys(extended))
        for index, profile in updates.items():
            profiles[index] = profile
            depths[index] = _hole_depth(profile[2])
            bottoms[index] = _hole_bottom(profile[2])
    return profiles


def _fence_hole_profiles(
    projected_df: pd.DataFrame,
    *,
    allow_pinch_outs: bool,
    override_index: dict[tuple[str, str], tuple[CorrelationOverride, ...]],
    fence_notes: list[str] | None = None,
) -> list[_HoleProfile]:
    """Hole profiles the fence is built from: sorted, gap-closed, one per position,
    short holes extended."""
    return _extend_short_holes(
        _sorted_hole_profiles(projected_df, fence_notes=fence_notes),
        allow_pinch_outs=allow_pinch_outs,
        override_index=override_index,
    )


def _correlation_pair_summary(
    left_id: str,
    right_id: str,
    left_lookup: dict[Hashable, _LayerInterval],
    right_lookup: dict[Hashable, _LayerInterval],
    *,
    allow_pinch_outs: bool,
) -> CorrelationPairSummary:
    matched = 0
    left_only: set[str] = set()
    right_only: set[str] = set()
    pinch_outs = 0
    crossing: set[str] = set()
    for key, left_layer in left_lookup.items():
        if isinstance(key, tuple) and key and key[0] == "crossed":
            crossing.add(left_layer.lithology_code)
        right_layer = right_lookup.get(key)
        if right_layer is not None:
            matched += 1
            continue
        left_only.add(left_layer.lithology_code)
        if allow_pinch_outs:
            pinch_outs += 1
    for key, right_layer in right_lookup.items():
        if key in left_lookup:
            continue
        right_only.add(right_layer.lithology_code)
        if allow_pinch_outs:
            pinch_outs += 1
    return CorrelationPairSummary(
        left_hole_id=left_id,
        right_hole_id=right_id,
        matched_count=matched,
        left_only_codes=tuple(sorted(left_only)),
        right_only_codes=tuple(sorted(right_only)),
        pinch_out_candidates=pinch_outs,
        crossing_codes=tuple(sorted(crossing)),
    )


def preview_correlation_health(
    projected_df: pd.DataFrame,
    *,
    allow_pinch_outs: bool = True,
    correlation_overrides: Sequence[CorrelationOverride] = (),
) -> list[CorrelationPairSummary]:
    """Summarize unit matching between adjacent holes without building polygons."""
    override_index = _correlation_overrides_by_pair(correlation_overrides)
    hole_profiles = _fence_hole_profiles(
        projected_df, allow_pinch_outs=allow_pinch_outs, override_index=override_index
    )
    if len(hole_profiles) < 2:
        return []
    summaries: list[CorrelationPairSummary] = []
    for (_x_left, left_id, left_intervals, left_lookup), (
        _x_right,
        right_id,
        right_intervals,
        right_lookup,
    ) in zip(hole_profiles, hole_profiles[1:]):
        left_lookup, right_lookup = _correlate_pair(
            left_id,
            right_id,
            left_intervals,
            right_intervals,
            left_lookup,
            right_lookup,
            _overrides_for_hole_pair(left_id, right_id, override_index),
        )
        summaries.append(
            _correlation_pair_summary(
                left_id,
                right_id,
                left_lookup,
                right_lookup,
                allow_pinch_outs=allow_pinch_outs,
            )
        )
    return summaries


def build_stratigraphy(
    projected_df: pd.DataFrame,
    *,
    allow_pinch_outs: bool = True,
    correlation_overrides: Sequence[CorrelationOverride] = (),
    pair_summaries: list[CorrelationPairSummary] | None = None,
    clip_warnings: list[str] | None = None,
) -> list[GeologicalPolygon]:
    """Construct geological polygons between adjacent projected boreholes.

    ``pair_summaries`` / ``clip_warnings`` are optional out-parameters: per-pair
    correlation summaries, and user-facing fence notes — polygons that overlap
    clipping cut down substantially (see ``_resolve_overlaps_in_pair``) and holes
    sharing one projected position, where the fence uses one log and the other
    hole is drawn as a column only (see ``_collapse_coincident_holes``).
    """
    override_index = _correlation_overrides_by_pair(correlation_overrides)
    hole_profiles = _fence_hole_profiles(
        projected_df,
        allow_pinch_outs=allow_pinch_outs,
        override_index=override_index,
        fence_notes=clip_warnings,
    )
    if len(hole_profiles) < 2:
        return []

    polygons: list[GeologicalPolygon] = []
    for (
        (x_left, left_id, left_intervals, left_lookup),
        (x_right, right_id, right_intervals, right_lookup),
    ) in zip(hole_profiles, hole_profiles[1:]):
        pair_polygons = _resolve_overlaps_in_pair(
            _polygons_for_pair(
                left_hole_id=left_id,
                right_hole_id=right_id,
                x_left=x_left,
                x_right=x_right,
                left_intervals=left_intervals,
                right_intervals=right_intervals,
                allow_pinch_outs=allow_pinch_outs,
                left_lookup=left_lookup,
                right_lookup=right_lookup,
                correlation_overrides=_overrides_for_hole_pair(left_id, right_id, override_index),
                pair_summaries=pair_summaries,
            ),
            clip_warnings=clip_warnings,
            hole_xs=(x_left, x_right),
        )
        polygons.extend(pair_polygons)

    polygons.sort(
        key=lambda item: (
            item.polygon.bounds[0],
            -item.polygon.bounds[3],
            item.lithology_code,
        )
    )
    return polygons
