"""Lithology stratigraphy and pinch-out polygon construction."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Hashable, Sequence

from models import CorrelationOverride

import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.strtree import STRtree

logger = logging.getLogger(__name__)


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
) -> float:
    """Average contact elevation from units above/below the pinch-out at the neighbor hole.

    When ``unit_order`` is set on the pinch interval, prefer neighbor intervals with
    adjacent ``unit_order`` values before falling back to elevation-position neighbors.
    This stabilizes pinch-outs when duplicate lithology codes appear in one hole.
    """
    if pinch_interval.unit_order is not None:
        order = pinch_interval.unit_order
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


def _make_polygon(
    coords: list[tuple[float, float]],
    lithology_code: str,
    hole_pair: tuple[str, str],
    *,
    is_pinch_out: bool = False,
) -> GeologicalPolygon | None:
    polygon = Polygon(coords)
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
) -> list[GeologicalPolygon]:
    polygons: list[GeologicalPolygon] = []
    if left_lookup is None:
        left_lookup = _correlation_keys(left_intervals)
    if right_lookup is None:
        right_lookup = _correlation_keys(right_intervals)
    left_lookup, right_lookup = _apply_correlation_overrides(
        left_hole_id,
        right_hole_id,
        left_intervals,
        right_intervals,
        left_lookup,
        right_lookup,
        correlation_overrides,
    )
    all_keys = sorted(
        set(left_lookup) | set(right_lookup),
        key=lambda key: _correlation_sort_key(key, left_lookup, right_lookup),
    )
    x_mid = (x_left + x_right) / 2.0
    hole_pair = (left_hole_id, right_hole_id)

    for key in all_keys:
        left_layer = left_lookup.get(key)
        right_layer = right_lookup.get(key)
        if left_layer is None and right_layer is None:
            continue
        lithology_code = (left_layer or right_layer).lithology_code  # type: ignore[union-attr]

        if left_layer and right_layer:
            polygon = _make_polygon(
                [
                    (x_left, left_layer.top_elevation),
                    (x_right, right_layer.top_elevation),
                    (x_right, right_layer.bottom_elevation),
                    (x_left, left_layer.bottom_elevation),
                ],
                lithology_code,
                hole_pair,
            )
            if polygon:
                polygons.append(polygon)
            continue

        if not allow_pinch_outs:
            continue

        if left_layer and not right_layer:
            z_mid = _pinch_out_z_mid(left_layer, right_intervals)
            polygon = _make_polygon(
                [
                    (x_left, left_layer.top_elevation),
                    (x_left, left_layer.bottom_elevation),
                    (x_mid, z_mid),
                ],
                lithology_code,
                hole_pair,
                is_pinch_out=True,
            )
            if polygon:
                polygons.append(polygon)
            continue

        if right_layer and not left_layer:
            z_mid = _pinch_out_z_mid(right_layer, left_intervals)
            polygon = _make_polygon(
                [
                    (x_right, right_layer.top_elevation),
                    (x_right, right_layer.bottom_elevation),
                    (x_mid, z_mid),
                ],
                lithology_code,
                hole_pair,
                is_pinch_out=True,
            )
            if polygon:
                polygons.append(polygon)

    return polygons


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


def _resolve_overlaps_in_pair(polygons: list[GeologicalPolygon]) -> list[GeologicalPolygon]:
    """Clip deeper fence polygons so inter-hole fills do not stack on top of shallower units."""
    if len(polygons) <= 1:
        return polygons

    from shapely.ops import unary_union

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
    batch: list[Polygon] = []
    batch_limit = 4
    last_index = len(ordered) - 1
    for index, geo_polygon in enumerate(ordered):
        geom = geo_polygon.polygon
        if occupied_geom is not None and not occupied_geom.is_empty:
            geom = geom.difference(occupied_geom)
        largest = _largest_polygon(geom)
        if largest is None or largest.is_empty:
            continue
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

    @property
    def unmatched_keys_count(self) -> int:
        return len(self.left_only_codes) + len(self.right_only_codes)

    @property
    def match_rate(self) -> float:
        total = self.matched_count + self.unmatched_keys_count
        if total == 0:
            return 1.0
        return self.matched_count / total


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
        intervals = _intervals_for_hole(group)
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
        left_lookup, right_lookup = _apply_correlation_overrides(
            left_id,
            right_id,
            left_intervals,
            right_intervals,
            left_lookup,
            right_lookup,
            _overrides_for_hole_pair(left_id, right_id, override_index),
        )
        all_keys = set(left_lookup) | set(right_lookup)
        matched = 0
        left_only: list[str] = []
        right_only: list[str] = []
        pinch_outs = 0
        for key in all_keys:
            left_layer = left_lookup.get(key)
            right_layer = right_lookup.get(key)
            if left_layer and right_layer:
                matched += 1
            elif left_layer:
                left_only.append(left_layer.lithology_code)
                if allow_pinch_outs:
                    pinch_outs += 1
            elif right_layer:
                right_only.append(right_layer.lithology_code)
                if allow_pinch_outs:
                    pinch_outs += 1
        summaries.append(
            CorrelationPairSummary(
                left_hole_id=left_id,
                right_hole_id=right_id,
                matched_count=matched,
                left_only_codes=tuple(sorted(set(left_only))),
                right_only_codes=tuple(sorted(set(right_only))),
                pinch_out_candidates=pinch_outs,
            )
        )
    return summaries


def build_stratigraphy(
    projected_df: pd.DataFrame,
    *,
    allow_pinch_outs: bool = True,
    correlation_overrides: Sequence[CorrelationOverride] = (),
) -> list[GeologicalPolygon]:
    """Construct geological polygons between adjacent projected boreholes."""
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
            )
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
