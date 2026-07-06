"""Spatial projection engine: 3D borehole coordinates to 2D profile plane."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
import pandas as pd

from models import Collar, DeviationReading, Lithology, Transect

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


def _warn_off_transect_holes(
    hole_ids: Sequence[str],
    offsets: np.ndarray,
    offset_warning_m: float,
    warned: set[str],
) -> None:
    if offset_warning_m <= 0:
        return
    for hole_id, offset in zip(hole_ids, offsets, strict=True):
        if offset > offset_warning_m and hole_id not in warned:
            logger.warning(
                "Hole '%s' is %.1f m from transect (threshold %.1f m)",
                hole_id,
                float(offset),
                offset_warning_m,
            )
            warned.add(hole_id)


def _vertical_projection_frame(
    vertical_lithologies: Sequence[Lithology],
    collar_by_id: dict[str, Collar],
    hole_projection: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    count = len(vertical_lithologies)
    if count == 0:
        return pd.DataFrame(columns=_PROJECTED_COLUMNS)

    hole_ids = [lithology.hole_id for lithology in vertical_lithologies]
    x_profiles = np.fromiter(
        (hole_projection[hole_id][0] for hole_id in hole_ids),
        dtype=float,
        count=count,
    )
    offsets = np.fromiter(
        (hole_projection[hole_id][1] for hole_id in hole_ids),
        dtype=float,
        count=count,
    )
    collar_elevs = np.fromiter(
        (collar_by_id[hole_id].elevation for hole_id in hole_ids),
        dtype=float,
        count=count,
    )
    from_depths = np.fromiter(
        (lithology.from_depth for lithology in vertical_lithologies),
        dtype=float,
        count=count,
    )
    to_depths = np.fromiter(
        (lithology.to_depth for lithology in vertical_lithologies),
        dtype=float,
        count=count,
    )
    return pd.DataFrame(
        {
            "hole_id": hole_ids,
            "x_profile": x_profiles,
            "collar_elevation": collar_elevs,
            "top_elevation": collar_elevs - from_depths,
            "bottom_elevation": collar_elevs - to_depths,
            "lithology_code": [lithology.lithology_code for lithology in vertical_lithologies],
            "offset_distance": offsets,
            "unit_order": [lithology.unit_order for lithology in vertical_lithologies],
        }
    )


def _deviated_projection_frame(
    hole_id: str,
    hole_lithologies: Sequence[Lithology],
    collar: Collar,
    survey: Sequence[_SurveyStation],
    geometry: "_TransectGeometry",
) -> pd.DataFrame:
    count = len(hole_lithologies)
    if count == 0:
        return pd.DataFrame(columns=_PROJECTED_COLUMNS)

    from_depths = np.fromiter(
        (lithology.from_depth for lithology in hole_lithologies),
        dtype=float,
        count=count,
    )
    to_depths = np.fromiter(
        (lithology.to_depth for lithology in hole_lithologies),
        dtype=float,
        count=count,
    )
    mid_depths = (from_depths + to_depths) / 2.0
    all_depths = np.concatenate([from_depths, to_depths, mid_depths])
    all_eastings, all_northings, all_tvds = interpolate_survey_many(survey, all_depths)
    from_tvds = all_tvds[:count]
    to_tvds = all_tvds[count : 2 * count]
    mid_eastings = all_eastings[2 * count :]
    mid_northings = all_northings[2 * count :]
    x_profiles, offsets = geometry.project_many(mid_eastings, mid_northings)
    collar_elev = collar.elevation
    return pd.DataFrame(
        {
            "hole_id": hole_id,
            "x_profile": x_profiles,
            "collar_elevation": collar_elev,
            "top_elevation": collar_elev - from_tvds,
            "bottom_elevation": collar_elev - to_tvds,
            "lithology_code": [lithology.lithology_code for lithology in hole_lithologies],
            "offset_distance": offsets,
            "unit_order": [lithology.unit_order for lithology in hole_lithologies],
        }
    )


def _points_cache_key(transect: Transect) -> tuple[tuple[float, float], ...]:
    return tuple((float(point[0]), float(point[1])) for point in transect.points)


@lru_cache(maxsize=64)
def _geometry_from_points(points: tuple[tuple[float, float], ...]) -> "_TransectGeometry":
    array = np.asarray(points, dtype=float)
    deltas = np.diff(array, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    prefix_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    segment_starts = array[:-1]
    length_sq = np.sum(deltas * deltas, axis=1)
    return _TransectGeometry(
        points=array,
        segment_lengths=segment_lengths,
        prefix_lengths=prefix_lengths,
        segment_starts=segment_starts,
        segment_vectors=deltas,
        length_sq=length_sq,
    )


@dataclass(frozen=True)
class _TransectGeometry:
    """Precomputed transect segments for repeated collar projections."""

    points: np.ndarray
    segment_lengths: np.ndarray
    prefix_lengths: np.ndarray
    segment_starts: np.ndarray
    segment_vectors: np.ndarray
    length_sq: np.ndarray

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
        if n_points == 0:
            return np.array([]), np.array([])

        n_segments = len(self.segment_lengths)
        if n_segments == 0:
            return np.zeros(n_points), np.linalg.norm(b - self.points[0], axis=1)

        p1 = self.segment_starts[:, np.newaxis, :]
        seg = self.segment_vectors[:, np.newaxis, :]
        b_exp = b[np.newaxis, :, :]
        length_sq = self.length_sq[:, np.newaxis]

        diff = b_exp - p1
        safe_length_sq = np.where(length_sq > 0.0, length_sq, 1.0)
        t_raw = np.sum(diff * seg, axis=2) / safe_length_sq
        t = np.clip(t_raw, 0.0, 1.0)
        projected = p1 + t[:, :, np.newaxis] * seg

        dist = np.linalg.norm(b_exp - projected, axis=2)
        dist_to_start = np.linalg.norm(b_exp - p1, axis=2)
        dist = np.where(length_sq > 0.0, dist, dist_to_start)

        seg_index = np.argmin(dist, axis=0)
        point_index = np.arange(n_points)
        best_distance = dist[seg_index, point_index]
        t_best = t[seg_index, point_index]
        best_x_profile = self.prefix_lengths[seg_index] + t_best * self.segment_lengths[seg_index]
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


def transect_length_m(transect: Transect) -> float:
    """Total polyline length along the transect (m)."""
    geometry = _TransectGeometry.from_transect(transect)
    return float(np.sum(geometry.segment_lengths))


def select_and_order_holes_near_transect(
    collars: Sequence[Collar],
    transect: Transect,
    offset_threshold_m: float,
) -> tuple[str, ...]:
    """Select holes within offset threshold, ordered along transect (single geometry pass)."""
    if not collars:
        return ()
    geometry = _TransectGeometry.from_transect(transect)
    hole_ids = [collar.hole_id for collar in collars]
    x_profiles, offsets = geometry.project_many(
        [collar.easting for collar in collars],
        [collar.northing for collar in collars],
    )
    mask = offsets <= offset_threshold_m
    if np.count_nonzero(mask) < 2:
        return ()
    selected_ids = [hole_ids[index] for index in np.flatnonzero(mask)]
    selected_x = x_profiles[mask]
    return tuple(
        hole_id
        for _, hole_id in sorted(zip(selected_x.tolist(), selected_ids, strict=True))
    )


def select_holes_near_transect(
    collars: Sequence[Collar],
    transect: Transect,
    offset_threshold_m: float,
) -> tuple[str, ...]:
    """Return hole IDs whose perpendicular offset from the transect is within threshold."""
    if not collars:
        return ()
    geometry = _TransectGeometry.from_transect(transect)
    hole_ids = [collar.hole_id for collar in collars]
    _, offsets = geometry.project_many(
        [collar.easting for collar in collars],
        [collar.northing for collar in collars],
    )
    return tuple(
        hole_id
        for hole_id, offset in zip(hole_ids, offsets, strict=True)
        if offset <= offset_threshold_m
    )


def order_holes_along_transect(
    collars: Sequence[Collar],
    transect: Transect,
    hole_ids: Sequence[str],
) -> tuple[str, ...]:
    """Sort hole IDs by projected distance along the transect (left to right on section)."""
    if not hole_ids:
        return ()
    collar_lookup = {collar.hole_id: collar for collar in collars}
    ordered_ids = [hole_id for hole_id in hole_ids if hole_id in collar_lookup]
    if not ordered_ids:
        return ()
    geometry = _TransectGeometry.from_transect(transect)
    x_profiles, _ = geometry.project_many(
        [collar_lookup[hole_id].easting for hole_id in ordered_ids],
        [collar_lookup[hole_id].northing for hole_id in ordered_ids],
    )
    return tuple(
        hole_id for _, hole_id in sorted(zip(x_profiles.tolist(), ordered_ids, strict=True))
    )


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
    threshold_text = f"{offset_threshold_m:.1f}"
    for hole_id, offset in zip(hole_ids, offsets, strict=True):
        if offset > offset_threshold_m:
            messages.append(
                f"{hole_id} is {offset:.1f} m from transect (threshold {threshold_text} m)"
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


@dataclass(frozen=True)
class _SurveyStation:
    measured_depth: float
    easting: float
    northing: float
    true_vertical_depth: float


def _deg_to_rad(value: float) -> float:
    return math.radians(value)


def _minimum_curvature_delta(
    md_start: float,
    inc_start_rad: float,
    azi_start_rad: float,
    md_end: float,
    inc_end_rad: float,
    azi_end_rad: float,
) -> tuple[float, float, float]:
    """Return (dN, dE, dTVD) between two survey stations using minimum curvature."""
    delta_md = md_end - md_start
    if delta_md <= 0.0:
        return 0.0, 0.0, 0.0

    cos_dogleg = (
        math.cos(inc_end_rad - inc_start_rad)
        - math.sin(inc_start_rad) * math.sin(inc_end_rad) * (1.0 - math.cos(azi_end_rad - azi_start_rad))
    )
    cos_dogleg = max(-1.0, min(1.0, cos_dogleg))
    dogleg = math.acos(cos_dogleg)
    if dogleg < 1e-12:
        ratio_factor = 1.0
    else:
        ratio_factor = 2.0 / dogleg * math.tan(dogleg / 2.0)

    half = delta_md / 2.0 * ratio_factor
    delta_northing = half * (
        math.sin(inc_start_rad) * math.cos(azi_start_rad) + math.sin(inc_end_rad) * math.cos(azi_end_rad)
    )
    delta_easting = half * (
        math.sin(inc_start_rad) * math.sin(azi_start_rad) + math.sin(inc_end_rad) * math.sin(azi_end_rad)
    )
    delta_tvd = half * (math.cos(inc_start_rad) + math.cos(inc_end_rad))
    return delta_northing, delta_easting, delta_tvd


def build_minimum_curvature_survey(
    collar: Collar,
    readings: Sequence[DeviationReading],
) -> list[_SurveyStation]:
    """Build a minimum-curvature survey path from collar and deviation readings."""
    sorted_readings = sorted(readings, key=lambda reading: reading.depth)
    stations: list[_SurveyStation] = [
        _SurveyStation(
            measured_depth=0.0,
            easting=collar.easting,
            northing=collar.northing,
            true_vertical_depth=0.0,
        )
    ]
    prev_md = 0.0
    prev_inc = _deg_to_rad(collar.inclination_deg or 0.0)
    prev_azi = _deg_to_rad(collar.azimuth_deg or 0.0)
    easting = collar.easting
    northing = collar.northing
    tvd = 0.0

    for reading in sorted_readings:
        if reading.depth <= prev_md:
            continue
        inc = _deg_to_rad(reading.inclination_deg)
        azi = _deg_to_rad(reading.azimuth_deg)
        delta_n, delta_e, delta_tvd = _minimum_curvature_delta(
            prev_md, prev_inc, prev_azi, reading.depth, inc, azi
        )
        northing += delta_n
        easting += delta_e
        tvd += delta_tvd
        stations.append(
            _SurveyStation(
                measured_depth=reading.depth,
                easting=easting,
                northing=northing,
                true_vertical_depth=tvd,
            )
        )
        prev_md = reading.depth
        prev_inc = inc
        prev_azi = azi

    if collar.total_depth > prev_md:
        inc = prev_inc
        azi = prev_azi
        delta_n, delta_e, delta_tvd = _minimum_curvature_delta(
            prev_md, inc, azi, collar.total_depth, inc, azi
        )
        northing += delta_n
        easting += delta_e
        tvd += delta_tvd
        stations.append(
            _SurveyStation(
                measured_depth=collar.total_depth,
                easting=easting,
                northing=northing,
                true_vertical_depth=tvd,
            )
        )
    return stations


def _survey_station_arrays(
    stations: Sequence[_SurveyStation],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    measured_depths = np.fromiter(
        (station.measured_depth for station in stations),
        dtype=float,
        count=len(stations),
    )
    eastings = np.fromiter(
        (station.easting for station in stations),
        dtype=float,
        count=len(stations),
    )
    northings = np.fromiter(
        (station.northing for station in stations),
        dtype=float,
        count=len(stations),
    )
    tvds = np.fromiter(
        (station.true_vertical_depth for station in stations),
        dtype=float,
        count=len(stations),
    )
    return measured_depths, eastings, northings, tvds


def interpolate_survey_many(
    stations: Sequence[_SurveyStation],
    depths: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized (easting, northing, TVD) interpolation at measured depths."""
    if not stations:
        raise ValueError("survey stations are required")
    depth_array = np.asarray(depths, dtype=float)
    if depth_array.size == 0:
        return np.array([]), np.array([]), np.array([])

    measured_depths, eastings, northings, tvds = _survey_station_arrays(stations)
    result_e = np.empty_like(depth_array)
    result_n = np.empty_like(depth_array)
    result_tvd = np.empty_like(depth_array)

    below = depth_array <= measured_depths[0]
    above = depth_array >= measured_depths[-1]
    interior = ~(below | above)

    result_e[below] = eastings[0]
    result_n[below] = northings[0]
    result_tvd[below] = tvds[0]
    result_e[above] = eastings[-1]
    result_n[above] = northings[-1]
    result_tvd[above] = tvds[-1]

    if np.any(interior):
        segment_index = np.searchsorted(measured_depths, depth_array[interior], side="right") - 1
        segment_index = np.clip(segment_index, 0, len(measured_depths) - 2)
        left_md = measured_depths[segment_index]
        right_md = measured_depths[segment_index + 1]
        span = right_md - left_md
        fraction = np.where(
            span > 0.0,
            (depth_array[interior] - left_md) / span,
            0.0,
        )
        result_e[interior] = eastings[segment_index] + fraction * (
            eastings[segment_index + 1] - eastings[segment_index]
        )
        result_n[interior] = northings[segment_index] + fraction * (
            northings[segment_index + 1] - northings[segment_index]
        )
        result_tvd[interior] = tvds[segment_index] + fraction * (
            tvds[segment_index + 1] - tvds[segment_index]
        )

    return result_e, result_n, result_tvd


def interpolate_survey_at_depth(
    stations: Sequence[_SurveyStation],
    depth: float,
) -> tuple[float, float, float]:
    """Interpolate (easting, northing, TVD) at a measured depth along the survey."""
    easting, northing, tvd = interpolate_survey_many(stations, [depth])
    return float(easting[0]), float(northing[0]), float(tvd[0])


def project_boreholes(
    collars: Sequence[Collar],
    lithologies: Sequence[Lithology],
    transect: Transect,
    offset_warning_m: float = DEFAULT_OFFSET_WARNING_M,
    deviation_readings: Sequence[DeviationReading] | None = None,
) -> pd.DataFrame:
    """Project borehole intervals onto the profile coordinate system."""
    if not lithologies:
        return pd.DataFrame(columns=_PROJECTED_COLUMNS)

    collar_by_id = {collar.hole_id: collar for collar in collars}
    geometry = _TransectGeometry.from_transect(transect)

    readings_by_hole: dict[str, list[DeviationReading]] = {}
    if deviation_readings:
        for reading in deviation_readings:
            readings_by_hole.setdefault(reading.hole_id, []).append(reading)

    surveys_by_hole = {
        hole_id: build_minimum_curvature_survey(collar_by_id[hole_id], readings)
        for hole_id, readings in readings_by_hole.items()
        if hole_id in collar_by_id and readings
    }

    vertical_lithologies: list[Lithology] = []
    deviated_lithologies: dict[str, list[Lithology]] = {}
    for lithology in lithologies:
        if lithology.hole_id not in collar_by_id:
            logger.warning("Skipping lithology for unknown hole_id '%s'", lithology.hole_id)
            continue
        if lithology.hole_id in surveys_by_hole:
            deviated_lithologies.setdefault(lithology.hole_id, []).append(lithology)
        else:
            vertical_lithologies.append(lithology)

    warned: set[str] = set()
    frames: list[pd.DataFrame] = []

    if vertical_lithologies:
        vertical_hole_ids = list(dict.fromkeys(lith.hole_id for lith in vertical_lithologies))
        eastings = np.fromiter(
            (collar_by_id[hole_id].easting for hole_id in vertical_hole_ids),
            dtype=float,
            count=len(vertical_hole_ids),
        )
        northings = np.fromiter(
            (collar_by_id[hole_id].northing for hole_id in vertical_hole_ids),
            dtype=float,
            count=len(vertical_hole_ids),
        )
        x_profiles, offsets = geometry.project_many(eastings, northings)
        hole_projection = {
            hole_id: (float(x_profile), float(offset))
            for hole_id, x_profile, offset in zip(
                vertical_hole_ids, x_profiles, offsets, strict=True
            )
        }
        _warn_off_transect_holes(vertical_hole_ids, offsets, offset_warning_m, warned)
        frames.append(
            _vertical_projection_frame(vertical_lithologies, collar_by_id, hole_projection)
        )

    for hole_id, hole_lithologies in deviated_lithologies.items():
        collar = collar_by_id[hole_id]
        survey = surveys_by_hole[hole_id]
        deviated_frame = _deviated_projection_frame(
            hole_id,
            hole_lithologies,
            collar,
            survey,
            geometry,
        )
        _warn_off_transect_holes(
            [hole_id],
            deviated_frame["offset_distance"].to_numpy(dtype=float),
            offset_warning_m,
            warned,
        )
        frames.append(deviated_frame)

    if not frames:
        return pd.DataFrame(columns=_PROJECTED_COLUMNS)

    projected = pd.concat(frames, ignore_index=True)
    return projected.sort_values(["x_profile", "hole_id", "top_elevation"], ascending=[True, True, False])


def transect_azimuth_deg(transect_points: Sequence[tuple[float, float]]) -> float | None:
    """Return azimuth in degrees from first to last transect point (0–360, clockwise from north)."""
    if len(transect_points) < 2:
        return None
    easting_0, northing_0 = transect_points[0]
    easting_1, northing_1 = transect_points[-1]
    dx = easting_1 - easting_0
    dy = northing_1 - northing_0
    if dx == 0.0 and dy == 0.0:
        return None
    azimuth = math.degrees(math.atan2(dx, dy))
    return azimuth % 360.0


# Public alias for transect geometry (used by transect_planner and tests).
TransectGeometry = _TransectGeometry
