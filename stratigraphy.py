"""Lithology stratigraphy and pinch-out polygon construction."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

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
    """Average contact elevation from units above/below pinch-out depth at neighbor hole."""
    upper = None
    lower = None

    for interval in neighbor_intervals:
        if interval.from_depth >= pinch_interval.to_depth - 1e-9:
            if upper is None or interval.from_depth < upper.from_depth:
                upper = interval
        if interval.to_depth <= pinch_interval.from_depth + 1e-9:
            if lower is None or interval.to_depth > lower.to_depth:
                lower = interval

    contacts: list[float] = []
    if upper is not None:
        contacts.append(upper.bottom_elevation)
    if lower is not None:
        contacts.append(lower.top_elevation)

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

    return GeologicalPolygon(
        lithology_code=lithology_code,
        polygon=polygon,
        hole_pair=hole_pair,
        is_pinch_out=is_pinch_out,
    )


def _polygons_for_pair(
    left_hole_id: str,
    right_hole_id: str,
    x_left: float,
    x_right: float,
    left_intervals: list[_LayerInterval],
    right_intervals: list[_LayerInterval],
    *,
    allow_pinch_outs: bool = True,
) -> list[GeologicalPolygon]:
    polygons: list[GeologicalPolygon] = []
    left_lookup = _correlation_keys(left_intervals)
    right_lookup = _correlation_keys(right_intervals)
    all_keys = sorted(set(left_lookup) | set(right_lookup), key=str)
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
        for left_pos, left_index in enumerate(indices):
            left = polygons[left_index]
            left_bounds = left.polygon.bounds
            for right_index in indices[left_pos + 1 :]:
                right = polygons[right_index]
                right_bounds = right.polygon.bounds
                if (
                    left_bounds[2] < right_bounds[0]
                    or right_bounds[2] < left_bounds[0]
                    or left_bounds[3] < right_bounds[1]
                    or right_bounds[3] < left_bounds[1]
                ):
                    continue
                if not left.polygon.intersects(right.polygon) or left.polygon.touches(right.polygon):
                    continue
                intersection = left.polygon.intersection(right.polygon)
                if intersection.is_empty:
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


def build_stratigraphy(
    projected_df: pd.DataFrame,
    *,
    allow_pinch_outs: bool = True,
) -> list[GeologicalPolygon]:
    """Construct geological polygons between adjacent projected boreholes."""
    if projected_df.empty:
        return []

    hole_profiles: list[tuple[float, str, list[_LayerInterval]]] = []
    for hole_id, group in projected_df.groupby("hole_id", sort=False):
        hole_profiles.append(
            (
                float(group["x_profile"].iloc[0]),
                str(hole_id),
                _intervals_for_hole(group),
            )
        )
    hole_profiles.sort(key=lambda item: item[0])
    if len(hole_profiles) < 2:
        return []

    polygons: list[GeologicalPolygon] = []
    for (x_left, left_id, left_intervals), (x_right, right_id, right_intervals) in zip(
        hole_profiles,
        hole_profiles[1:],
    ):
        pair_polygons = _polygons_for_pair(
            left_hole_id=left_id,
            right_hole_id=right_id,
            x_left=x_left,
            x_right=x_right,
            left_intervals=left_intervals,
            right_intervals=right_intervals,
            allow_pinch_outs=allow_pinch_outs,
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
