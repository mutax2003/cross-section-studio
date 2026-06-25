"""Non-LLM transect recommender for borehole cross-sections."""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass

from models import Collar, Lithology, Transect
from projection import _TransectGeometry


@dataclass(frozen=True)
class TransectCandidate:
    hole_ids: tuple[str, ...]
    score: float
    mean_offset_m: float
    lithology_diversity: int
    length_m: float
    pinch_out_count: int


def _path_length(collars_by_id: dict[str, Collar], hole_ids: tuple[str, ...]) -> float:
    total = 0.0
    for left, right in zip(hole_ids, hole_ids[1:]):
        left_collar = collars_by_id[left]
        right_collar = collars_by_id[right]
        dx = right_collar.easting - left_collar.easting
        dy = right_collar.northing - left_collar.northing
        total += (dx**2 + dy**2) ** 0.5
    return total


def _mean_perpendicular_offset(
    collars_by_id: dict[str, Collar],
    hole_ids: tuple[str, ...],
) -> float:
    points = [(collars_by_id[hole_id].easting, collars_by_id[hole_id].northing) for hole_id in hole_ids]
    geometry = _TransectGeometry.from_transect(Transect(points=points))
    _, offsets = geometry.project_many(
        [collars_by_id[hole_id].easting for hole_id in hole_ids],
        [collars_by_id[hole_id].northing for hole_id in hole_ids],
    )
    return float(offsets.mean())


def _lithology_diversity(codes_by_hole: dict[str, set[str]], hole_ids: tuple[str, ...]) -> int:
    codes: set[str] = set()
    for hole_id in hole_ids:
        codes.update(codes_by_hole.get(hole_id, set()))
    return len(codes)


def _pinch_out_count(codes_by_hole: dict[str, set[str]], hole_ids: tuple[str, ...]) -> int:
    count = 0
    for left, right in zip(hole_ids, hole_ids[1:]):
        count += len(codes_by_hole.get(left, set()).symmetric_difference(codes_by_hole.get(right, set())))
    return count


def score_transect(
    collars: list[Collar],
    lithologies: list[Lithology],
    hole_ids: tuple[str, ...],
) -> TransectCandidate:
    collars_by_id = {collar.hole_id: collar for collar in collars}
    codes_by_hole: dict[str, set[str]] = defaultdict(set)
    for lithology in lithologies:
        codes_by_hole[lithology.hole_id].add(lithology.lithology_code)
    return _score_transect(collars_by_id, codes_by_hole, hole_ids)


def _score_transect(
    collars_by_id: dict[str, Collar],
    codes_by_hole: dict[str, set[str]],
    hole_ids: tuple[str, ...],
) -> TransectCandidate:
    mean_offset = _mean_perpendicular_offset(collars_by_id, hole_ids)
    diversity = _lithology_diversity(codes_by_hole, hole_ids)
    length = _path_length(collars_by_id, hole_ids)
    pinch_outs = _pinch_out_count(codes_by_hole, hole_ids)

    score = (
        diversity * 10.0
        + pinch_outs * 5.0
        + min(length, 500.0) / 50.0
        - mean_offset * 2.0
    )
    return TransectCandidate(
        hole_ids=hole_ids,
        score=score,
        mean_offset_m=mean_offset,
        lithology_diversity=diversity,
        length_m=length,
        pinch_out_count=pinch_outs,
    )


def _order_holes_along_dominant_axis(
    collars_by_id: dict[str, Collar],
    hole_ids: tuple[str, ...],
) -> tuple[str, ...]:
    eastings = [collars_by_id[hole_id].easting for hole_id in hole_ids]
    northings = [collars_by_id[hole_id].northing for hole_id in hole_ids]
    if max(eastings) - min(eastings) >= max(northings) - min(northings):
        return tuple(sorted(hole_ids, key=lambda hole_id: collars_by_id[hole_id].easting))
    return tuple(sorted(hole_ids, key=lambda hole_id: collars_by_id[hole_id].northing))


def recommend_transects(
    collars: list[Collar],
    lithologies: list[Lithology],
    *,
    min_holes: int = 2,
    max_holes: int = 6,
    top_n: int = 3,
) -> list[TransectCandidate]:
    """Return top-N ordered hole sequences scored for cross-section quality."""
    if len(collars) < min_holes:
        return []

    collars_by_id = {collar.hole_id: collar for collar in collars}
    codes_by_hole: dict[str, set[str]] = defaultdict(set)
    for lithology in lithologies:
        codes_by_hole[lithology.hole_id].add(lithology.lithology_code)

    hole_ids = _order_holes_along_dominant_axis(
        collars_by_id,
        tuple(collars_by_id),
    )
    candidates: list[TransectCandidate] = []
    max_size = min(max_holes, len(hole_ids))

    if len(hole_ids) > 12:
        for size in range(min_holes, max_size + 1):
            for start in range(len(hole_ids) - size + 1):
                ordered = tuple(hole_ids[start : start + size])
                candidates.append(_score_transect(collars_by_id, codes_by_hole, ordered))
    else:
        for size in range(min_holes, max_size + 1):
            for combo in itertools.combinations(hole_ids, size):
                ordered = _order_holes_along_dominant_axis(collars_by_id, combo)
                candidates.append(_score_transect(collars_by_id, codes_by_hole, ordered))

    candidates.sort(key=lambda item: item.score, reverse=True)
    unique: list[TransectCandidate] = []
    seen: set[tuple[str, ...]] = set()
    for candidate in candidates:
        if candidate.hole_ids in seen:
            continue
        seen.add(candidate.hole_ids)
        unique.append(candidate)
        if len(unique) >= top_n:
            break
    return unique
