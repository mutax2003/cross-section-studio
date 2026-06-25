"""Spatial projection engine: 3D borehole coordinates to 2D profile plane."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
import pandas as pd

from models import Collar, Lithology, Transect

logger = logging.getLogger(__name__)

DEFAULT_OFFSET_WARNING_M = 50.0

_PROJECTED_COLUMNS = [
    "hole_id",
    "x_profile",
    "collar_elevation",
    "top_elevation",
    "bottom_elevation",
    "lithology_code",
    "offset_distance",
    "unit_order",
]


def _points_cache_key(transect: Transect) -> tuple[tuple[float, float], ...]:
    return tuple((float(point[0]), float(point[1])) for point in transect.points)


@lru_cache(maxsize=64)
def _geometry_from_points(points: tuple[tuple[float, float], ...]) -> "_TransectGeometry":
    array = np.asarray(points, dtype=float)
    deltas = np.diff(array, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    prefix_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    return _TransectGeometry(
        points=array,
        segment_lengths=segment_lengths,
        prefix_lengths=prefix_lengths,
    )


@dataclass(frozen=True)
class _TransectGeometry:
    """Precomputed transect segments for repeated collar projections."""

    points: np.ndarray
    segment_lengths: np.ndarray
    prefix_lengths: np.ndarray

    @classmethod
    def from_transect(cls, transect: Transect) -> _TransectGeometry:
        return _geometry_from_points(_points_cache_key(transect))

    def project(self, easting: float, northing: float) -> tuple[float, float]:
        x_profiles, distances = self.project_many(
            np.asarray([easting], dtype=float),
            np.asarray([northing], dtype=float),
        )
        return float(x_profiles[0]), float(distances[0])

    def project_many(
        self,
        eastings: np.ndarray | Sequence[float],
        northings: np.ndarray | Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        b = np.column_stack(
            [
                np.asarray(eastings, dtype=float),
                np.asarray(northings, dtype=float),
            ]
        )
        n_points = len(b)
        best_distance = np.full(n_points, np.inf)
        best_x_profile = np.zeros(n_points)

        for index in range(len(self.segment_lengths)):
            p1 = self.points[index]
            p2 = self.points[index + 1]
            segment = p2 - p1
            length_sq = float(np.dot(segment, segment))
            if length_sq == 0.0:
                t = np.zeros(n_points)
                projected = np.broadcast_to(p1, (n_points, 2))
                distance = np.linalg.norm(b - projected, axis=1)
            else:
                t_raw = np.dot(b - p1, segment) / length_sq
                t = np.clip(t_raw, 0.0, 1.0)
                projected = p1 + t[:, np.newaxis] * segment
                distance = np.linalg.norm(b - projected, axis=1)

            closer = distance < best_distance
            best_distance = np.where(closer, distance, best_distance)
            x_candidate = self.prefix_lengths[index] + t * self.segment_lengths[index]
            best_x_profile = np.where(closer, x_candidate, best_x_profile)

        return best_x_profile, best_distance


def _as_point(point: Sequence[float]) -> np.ndarray:
    return np.array([float(point[0]), float(point[1])], dtype=float)


def _project_point_to_segment(
    b: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    """Project point B onto segment P1-P2. Returns (t_clamped, projected_point, distance)."""
    segment = p2 - p1
    length_sq = float(np.dot(segment, segment))
    if length_sq == 0.0:
        projected = p1.copy()
        distance = float(np.linalg.norm(b - p1))
        return 0.0, projected, distance

    t = float(np.dot(b - p1, segment) / length_sq)
    t_clamped = max(0.0, min(1.0, t))
    projected = p1 + t_clamped * segment
    distance = float(np.linalg.norm(b - projected))
    return t_clamped, projected, distance


def _find_closest_segment(
    b: np.ndarray,
    transect_points: Sequence[tuple[float, float]],
) -> tuple[int, float, np.ndarray, float]:
    """Return closest segment index, clamped t, projected point, and perpendicular distance."""
    best_index = 0
    best_distance = float("inf")
    best_t = 0.0
    best_projected = _as_point(transect_points[0])

    for index in range(len(transect_points) - 1):
        p1 = _as_point(transect_points[index])
        p2 = _as_point(transect_points[index + 1])
        t, projected, distance = _project_point_to_segment(b, p1, p2)
        if distance < best_distance:
            best_index = index
            best_distance = distance
            best_t = t
            best_projected = projected

    return best_index, best_t, best_projected, best_distance


def _cumulative_distance(
    transect_points: Sequence[tuple[float, float]],
    segment_index: int,
    t: float,
) -> float:
    """Cumulative distance from transect start to projection point on segment_index."""
    distance = 0.0
    for index in range(segment_index):
        p1 = _as_point(transect_points[index])
        p2 = _as_point(transect_points[index + 1])
        distance += float(np.linalg.norm(p2 - p1))

    p1 = _as_point(transect_points[segment_index])
    p2 = _as_point(transect_points[segment_index + 1])
    distance += t * float(np.linalg.norm(p2 - p1))
    return distance


def project_collar_to_transect(
    easting: float,
    northing: float,
    transect: Transect,
    geometry: _TransectGeometry | None = None,
) -> tuple[float, float]:
    """Project a collar XY location onto the transect. Returns (x_profile, offset_distance)."""
    if geometry is None:
        geometry = _TransectGeometry.from_transect(transect)
    return geometry.project(easting, northing)


def off_transect_warnings(
    collars: Sequence[Collar],
    transect: Transect,
    offset_threshold_m: float,
) -> list[str]:
    """Return warning messages for collars far from the transect line."""
    if not collars:
        return []
    geometry = _TransectGeometry.from_transect(transect)
    hole_ids = [collar.hole_id for collar in collars]
    _, offsets = geometry.project_many(
        [collar.easting for collar in collars],
        [collar.northing for collar in collars],
    )
    messages: list[str] = []
    for hole_id, offset in zip(hole_ids, offsets):
        if float(offset) > offset_threshold_m:
            messages.append(
                f"{hole_id} is {float(offset):.1f} m from transect "
                f"(threshold {offset_threshold_m:.1f} m)"
            )
    return messages


def suggest_offset_threshold_m(collars: Sequence[Collar], *, floor_m: float = 50.0) -> float:
    """Heuristic default transect offset warning based on collar spread."""
    if len(collars) < 2:
        return floor_m
    eastings = np.fromiter((collar.easting for collar in collars), dtype=float)
    northings = np.fromiter((collar.northing for collar in collars), dtype=float)
    span = max(float(eastings.max() - eastings.min()), float(northings.max() - northings.min()))
    return max(floor_m, min(150.0, round(span * 0.15, 1)))


def project_boreholes(
    collars: Sequence[Collar],
    lithologies: Sequence[Lithology],
    transect: Transect,
    offset_warning_m: float = DEFAULT_OFFSET_WARNING_M,
) -> pd.DataFrame:
    """Project borehole intervals onto the profile coordinate system."""
    if not lithologies:
        return pd.DataFrame(columns=_PROJECTED_COLUMNS)

    collar_by_id = {collar.hole_id: collar for collar in collars}
    geometry = _TransectGeometry.from_transect(transect)

    unique_hole_ids = tuple(collar_by_id)
    x_profiles, offsets = geometry.project_many(
        [collar_by_id[hole_id].easting for hole_id in unique_hole_ids],
        [collar_by_id[hole_id].northing for hole_id in unique_hole_ids],
    )
    projection_cache = dict(zip(unique_hole_ids, zip(x_profiles.tolist(), offsets.tolist()), strict=True))

    hole_ids: list[str] = []
    x_profile_values: list[float] = []
    collar_elevations: list[float] = []
    top_elevations: list[float] = []
    bottom_elevations: list[float] = []
    lithology_codes: list[str] = []
    offset_values: list[float] = []
    unit_orders: list[object] = []

    for lithology in lithologies:
        collar = collar_by_id.get(lithology.hole_id)
        if collar is None:
            logger.warning("Skipping lithology for unknown hole_id '%s'", lithology.hole_id)
            continue
        x_profile, offset = projection_cache[lithology.hole_id]
        if offset > offset_warning_m:
            logger.warning(
                "Hole '%s' is %.1f m from transect (threshold %.1f m)",
                collar.hole_id,
                offset,
                offset_warning_m,
            )

        hole_ids.append(lithology.hole_id)
        x_profile_values.append(float(x_profile))
        collar_elevations.append(collar.elevation)
        top_elevations.append(collar.elevation - lithology.from_depth)
        bottom_elevations.append(collar.elevation - lithology.to_depth)
        lithology_codes.append(lithology.lithology_code)
        offset_values.append(float(offset))
        unit_orders.append(lithology.unit_order)

    if not hole_ids:
        return pd.DataFrame(columns=_PROJECTED_COLUMNS)

    projected = pd.DataFrame(
        {
            "hole_id": hole_ids,
            "x_profile": x_profile_values,
            "collar_elevation": collar_elevations,
            "top_elevation": top_elevations,
            "bottom_elevation": bottom_elevations,
            "lithology_code": lithology_codes,
            "offset_distance": offset_values,
            "unit_order": unit_orders,
        }
    )
    return projected.sort_values(["x_profile", "hole_id", "top_elevation"], ascending=[True, True, False])
