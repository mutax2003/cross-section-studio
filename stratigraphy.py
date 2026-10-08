"""Lithology stratigraphy and pinch-out polygon construction."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from shapely.errors import GEOSException
from shapely.geometry import LineString, Polygon
from shapely.strtree import STRtree

from models import CorrelationOverride

logger = logging.getLogger(__name__)

# Fraction of the hole spacing a pinch-out wedge spans from its source hole before
# tapering to its apex (0.5 = mid-way toward the neighbouring hole).
PINCH_OUT_FRACTION = 0.5

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
    neighbor_intervals: list[_LayerInterval],
    *,
    by_order: dict[int, _LayerInterval] | None = None,
) -> float:
    """Average contact elevation from units above/below the pinch-out at the neighbor hole.

    When ``unit_order`` is set on the pinch interval, prefer neighbor intervals with
    adjacent ``unit_order`` values before falling back to elevation-position neighbors.
    This stabilizes pinch-outs when duplicate lithology codes appear in one hole.

    Pass ``by_order`` when calling repeatedly for the same neighbor hole to avoid
    rebuilding the order map on every unmatched key.
    """
    if pinch_interval.unit_order is not None:
        order = pinch_interval.unit_order
        if by_order is None:
            by_order = {
                interval.unit_order: interval
                for interval in neighbor_intervals
                if interval.unit_order is not None
            }
        contacts: list[float] = []
        above = by_order.get(order - 1)
        below = by_order.get(order + 1)
        if above is not None:
            contacts.append(above.bottom_elevation)
        if below is not None:
            contacts.append(below.top_elevation)
        if contacts:
            return sum(contacts) / len(contacts)

    # Elevation-position neighbors: shallower (above) and deeper (below) the pinch interval.
    # Compare elevations (not hole-local depths) so unequal collar RLs stay correct.
    above = None
    below = None
    for interval in neighbor_intervals:
        if interval.bottom_elevation >= pinch_interval.top_elevation - 1e-9:
            if above is None or interval.bottom_elevation < above.bottom_elevation:
                above = interval
        if interval.top_elevation <= pinch_interval.bottom_elevation + 1e-9:
            if below is None or interval.top_elevation > below.top_elevation:
                below = interval

    contacts = []
    if above is not None:
        contacts.append(above.bottom_elevation)
    if below is not None:
        contacts.append(below.top_elevation)

    if not contacts:
        return (pinch_interval.top_elevation + pinch_interval.bottom_elevation) / 2.0
    return sum(contacts) / len(contacts)


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
    all_keys = sorted(
        set(left_lookup) | set(right_lookup),
        key=lambda key: _correlation_sort_key(key, left_lookup, right_lookup),
    )
    hole_pair = (left_hole_id, right_hole_id)
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
    # Wedges with exactly one bracketing match (base / top of the fence), grouped by
    # their ``(above, below)`` brackets; tiled against the fence base / top line in
    # pass 1b instead of tapering to a lone tip (see ``_tile_open_wedges``).
    open_wedges: dict[
        tuple[Hashable | None, Hashable | None],
        list[tuple[bool, tuple[_LayerInterval, float, float, float]]],
    ] = defaultdict(list)
    if allow_pinch_outs:
        for key in all_keys:
            left_layer = left_lookup.get(key)
            right_layer = right_lookup.get(key)
            if (left_layer is None) == (right_layer is None):
                continue
            if left_layer is not None:
                source_layer, x_source, x_other = left_layer, x_left, x_right
                neighbor_intervals, neighbor_by_order = right_intervals, right_by_order
                source_matches = matched
            else:
                source_layer, x_source, x_other = right_layer, x_right, x_left  # type: ignore[assignment]
                neighbor_intervals, neighbor_by_order = left_intervals, left_by_order
                source_matches = matched_right_first
            x_tip = x_source + PINCH_OUT_FRACTION * (x_other - x_source)
            above, below = _bracketing_matches(source_layer, source_matches)
            if above is not None or below is not None:
                z_tip = _bracket_tip_z(
                    above,
                    below,
                    left_lookup,
                    right_lookup,
                    x_left,
                    x_right,
                    x_tip,
                )
                if (
                    above is not None
                    and below is not None
                    and _bend_keeps_fills_valid(
                        above, below, left_lookup, right_lookup, x_left, x_right, x_tip, z_tip
                    )
                ):
                    bottom_bends[above].append((x_tip, z_tip))
                    top_bends[below].append((x_tip, z_tip))
            else:
                z_tip = _pinch_out_z_mid(
                    source_layer, neighbor_intervals, by_order=neighbor_by_order
                )
            wedge = (source_layer, x_source, x_other, z_tip)
            if (above is None) != (below is None):
                open_wedges[(above, below)].append((left_layer is not None, wedge))
            else:
                wedges.append(wedge)

    # Pass 1b: wedges open toward the fence base (or top). Opposing groups meet
    # along a facies-change boundary at mid-span; a lone group runs out to the
    # neighbour's hole end. Groups that cannot tile cleanly keep their tip wedges.
    open_polygons: list[GeologicalPolygon] = []
    for (above, below), members in open_wedges.items():
        tiles = _tile_open_wedges(
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
        if tiles is None:
            wedges.extend(wedge for _is_left, wedge in members)
        else:
            open_polygons.extend(tiles)

    # Pinch-outs off: unmatched units are not drawn between the holes, so the
    # bracketing fills close the space they leave at the source hole instead.
    top_ends: dict[tuple[Hashable, str], float] = {}
    bottom_ends: dict[tuple[Hashable, str], float] = {}
    if not allow_pinch_outs:
        for side, lookup, other, source_matches in (
            ("left", left_lookup, right_lookup, matched),
            ("right", right_lookup, left_lookup, matched_right_first),
        ):
            sides_top, sides_bottom = _closed_unmatched_ends(
                [layer for key, layer in lookup.items() if key not in other],
                source_matches,
            )
            top_ends.update({(key, side): z for key, z in sides_top.items()})
            bottom_ends.update({(key, side): z for key, z in sides_bottom.items()})

    # Pass 2: continuous fills (with bent contacts), then wedges.
    for key, left_layer, right_layer in matched:
        top_path = _bend_path(top_bends.get(key, ()), x_left, x_right)
        bottom_path = _bend_path(bottom_bends.get(key, ()), x_left, x_right)
        polygon = _make_polygon(
            [
                (x_left, top_ends.get((key, "left"), left_layer.top_elevation)),
                *top_path,
                (x_right, top_ends.get((key, "right"), right_layer.top_elevation)),
                (x_right, bottom_ends.get((key, "right"), right_layer.bottom_elevation)),
                *reversed(bottom_path),
                (x_left, bottom_ends.get((key, "left"), left_layer.bottom_elevation)),
            ],
            left_layer.lithology_code,
            hole_pair,
        )
        if polygon:
            polygons.append(polygon)

    for source_layer, x_source, x_other, z_tip in wedges:
        polygon = _make_polygon(
            _pinch_out_wedge_coords(source_layer, x_source, x_other, z_tip),
            source_layer.lithology_code,
            hole_pair,
            is_pinch_out=True,
        )
        if polygon:
            polygons.append(polygon)
    polygons.extend(open_polygons)

    return polygons


def _closed_unmatched_ends(
    unmatched: list[_LayerInterval],
    matched: list[tuple[Hashable, _LayerInterval, _LayerInterval]],
) -> tuple[dict[Hashable, float], dict[Hashable, float]]:
    """Source-hole contact elevations that close the space of undrawn unmatched units.

    Used when pinch-outs are off: the unmatched units of one hole are not drawn
    between the holes, so the matched units bracketing them take over the space
    (the hole column still shows the logged units). Returns ``(top_ends,
    bottom_ends)`` keyed by matched correlation key, the new elevation of that
    fill's top / bottom edge at the source hole:

    * **Two-sided** (matched unit above and below): the upper unit's base and the
      lower unit's top both run to the mid-elevation of the space between them,
      so the two contacts converge at the hole — the same tiling a pinch-out bend
      gives, with the tip on the hole instead of mid-span.
    * **Above only** (units below the lowest match): the upper unit's base runs
      down to the deepest unmatched base, i.e. along the fence base line.
    * **Below only** (units above the highest match): the lower unit's top runs
      up to the highest unmatched top, i.e. along the fence top line.

    Unbracketed units and crossing (mis-ordered) brackets are left alone.
    """
    groups: dict[tuple[Hashable | None, Hashable | None], list[_LayerInterval]] = defaultdict(list)
    for interval in unmatched:
        above, below = _bracketing_matches(interval, matched)
        if above is not None or below is not None:
            groups[(above, below)].append(interval)
    source = {key: layer for key, layer, _neighbour in matched}
    top_ends: dict[Hashable, float] = {}
    bottom_ends: dict[Hashable, float] = {}
    for (above, below), members in groups.items():
        finite = [
            interval
            for interval in members
            if np.isfinite(interval.top_elevation) and np.isfinite(interval.bottom_elevation)
        ]
        if not finite:
            continue
        if above is not None and below is not None:
            upper = source[above].bottom_elevation
            lower = source[below].top_elevation
            if not (np.isfinite(upper) and np.isfinite(lower)) or upper < lower - 1e-9:
                continue
            middle = 0.5 * (upper + lower)
            bottom_ends[above] = middle
            top_ends[below] = middle
        elif above is not None:
            base = min(interval.bottom_elevation for interval in finite)
            if base < source[above].bottom_elevation:
                bottom_ends[above] = base
        else:
            top = max(interval.top_elevation for interval in finite)
            if top > source[below].top_elevation:  # type: ignore[index]
                top_ends[below] = top
    return top_ends, bottom_ends


def _tile_open_wedges(
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
    """Tile unmatched units that sit below the lowest (or above the highest) match.

    Such a unit has a bracketing contact on one side only, so a tip wedge leaves
    a white triangle against the fence base line (hole bottom to hole bottom) or
    the fence top line (hole top to hole top). The open region between that
    contact and the base/top line is instead shared out:

    * **Opposing groups** (unit(s) X only in the left hole, Y only in the right,
      same bracketing contact, both reaching their hole ends) meet along a
      vertical facies-change boundary at ``PINCH_OUT_FRACTION`` of the span:
      X fills the left part, Y the right part.
    * **A lone group** (nothing logged opposite because the neighbour ends at the
      contact) extends to the neighbour, thinning to zero at its hole end — the
      neighbour simply did not penetrate the unit.

    Stacked units in a group keep their thickness proportions across the region.
    Exactly one of ``above`` / ``below`` must be set (unbracketed wedges, e.g.
    crossing un-matches in a pair with no correlation, keep their tips). Returns
    ``None`` (caller keeps the tip wedges) when the group does not reach its hole
    ends or the region is not well ordered.
    """
    if (above is None) == (below is None) or (not left_units and not right_units):
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
            # The group must fill its hole from the contact to the hole end.
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

    x_mid = x_left + PINCH_OUT_FRACTION * span
    if left_units and right_units:
        ranges = {"left": (x_left, x_mid), "right": (x_mid, x_right)}
    else:
        ranges = {"left": (x_left, x_right), "right": (x_left, x_right)}

    def _curve(fraction: float, xs: list[float]) -> list[tuple[float, float]]:
        points = []
        for x in xs:
            upper, lower = _upper(x), _lower(x)
            # Exact ends keep shared edges bit-identical between tiles.
            if fraction <= 0.0:
                z = upper
            elif fraction >= 1.0:
                z = lower
            else:
                z = upper - fraction * (upper - lower)
            points.append((x, z))
        return points

    tiles: list[GeologicalPolygon] = []
    for side, (units, _intervals, _lookup, x_source) in sides.items():
        if not units:
            continue
        x0, x1 = ranges[side]
        xs = [x0, *(x for x in breaks if x0 < x < x1), x1]
        source_upper, source_lower = _upper(x_source), _lower(x_source)
        thickness = source_upper - source_lower
        if thickness <= 0.0:
            return None
        for unit in sorted(units, key=lambda item: (-item.top_elevation, -item.bottom_elevation)):
            f_top = (source_upper - unit.top_elevation) / thickness
            f_bottom = (source_upper - unit.bottom_elevation) / thickness

            top_curve = _curve(f_top, xs)
            bottom_curve = _curve(f_bottom, xs)
            if x_source == x0:
                top_curve[0] = (x_source, unit.top_elevation)
                bottom_curve[0] = (x_source, unit.bottom_elevation)
            else:
                top_curve[-1] = (x_source, unit.top_elevation)
                bottom_curve[-1] = (x_source, unit.bottom_elevation)
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
    attached = [(part.boundary.intersection(anchor).length, part.area) for part in parts]
    candidates = [
        (contact, area, part) for (contact, area), part in zip(attached, parts) if contact > 0.0
    ]
    if not candidates:
        return _largest_polygon(geom)
    return max(candidates, key=lambda item: (item[1], item[0]))[2]


# Fraction of a fence polygon's area that overlap clipping may discard before the
# loss is reported to the user (``clip_warnings``) rather than only logged.
CLIP_LOSS_WARN_FRACTION = 0.15


def _resolve_overlaps_in_pair(
    polygons: list[GeologicalPolygon],
    *,
    clip_warnings: list[str] | None = None,
    hole_xs: Sequence[float] = (),
) -> list[GeologicalPolygon]:
    """Clip deeper fence polygons so inter-hole fills do not stack on top of shallower units.

    When ``clip_warnings`` is given, a user-facing message is appended for every
    polygon that loses more than ``CLIP_LOSS_WARN_FRACTION`` of its area (or is
    dropped entirely) — a logged interval partly missing from the fence.
    """
    if len(polygons) <= 1:
        return polygons

    from shapely.ops import unary_union
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
        geom = geo_polygon.polygon
        if occupied_geom is not None and not occupied_geom.is_empty:
            # Prepared predicates for skip checks; difference still uses raw geom.
            probe = occupied_prep if occupied_prep is not None else occupied_geom
            if probe.intersects(geom) and not probe.touches(geom):
                geom = geom.difference(occupied_geom)
        # Polygons kept earlier in the current (not yet unioned) batch also occupy
        # area; without this, up to ``batch_limit`` fills could stack unclipped.
        for pending in batch:
            if geom.is_empty:
                break
            if pending.intersects(geom) and not pending.touches(geom):
                geom = geom.difference(pending)
        if geo_polygon.is_pinch_out and geom is not geo_polygon.polygon:
            # A wedge must stay attached to the hole where the unit was logged; the
            # largest clip fragment can be a sliver floating mid-way between holes.
            largest = _anchored_fragment(geom, _pinch_anchor_edge(geo_polygon.polygon, hole_xs))
        else:
            largest = _largest_polygon(geom)
        if largest is not None and not largest.is_empty and largest is not geo_polygon.polygon:
            largest = _clean_polygon(largest)
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
        if original_area > 0 and largest.area < 0.85 * original_area:
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
            chunk = unary_union(batch) if len(batch) > 1 else batch[0]
            occupied_geom = chunk if occupied_geom is None else occupied_geom.union(chunk)
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
                intersection = left.polygon.intersection(right.polygon)
                # Skip empty/tiny area and pure line touches (area < 1e-9).
                if intersection.is_empty or intersection.area < 1e-9:
                    continue
                # GEOS overlay on near-degenerate rings is direction-sensitive; a
                # real overlap has area both ways round, a precision artefact not.
                reverse = right.polygon.intersection(left.polygon)
                if reverse.is_empty or reverse.area < 1e-9:
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
) -> list[tuple[float, str, list[_LayerInterval], dict[Hashable, _LayerInterval]]]:
    """Build x-sorted hole interval profiles from a projected DataFrame."""
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
    hole_profiles.sort(key=lambda item: item[0])
    return hole_profiles


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
    hole_profiles = _sorted_hole_profiles(projected_df)
    if len(hole_profiles) < 2:
        return []
    summaries: list[CorrelationPairSummary] = []
    override_index = _correlation_overrides_by_pair(correlation_overrides)
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
    correlation summaries, and user-facing messages for fence polygons that
    overlap clipping cut down substantially (see ``_resolve_overlaps_in_pair``).
    """
    hole_profiles = _sorted_hole_profiles(projected_df)
    if len(hole_profiles) < 2:
        return []

    polygons: list[GeologicalPolygon] = []
    override_index = _correlation_overrides_by_pair(correlation_overrides)
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
