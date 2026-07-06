"""ParseResult transforms: subset, unit_order, serialization helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Sequence

from models import Lithology, ParseResult

def lithologies_by_hole(
    lithologies: Sequence[Lithology],
) -> dict[str, tuple[Lithology, ...]]:
    """Index lithology intervals by hole_id for O(1) subset lookups."""
    buckets: dict[str, list[Lithology]] = defaultdict(list)
    for lithology in lithologies:
        buckets[lithology.hole_id].append(lithology)
    return {hole_id: tuple(items) for hole_id, items in buckets.items()}


def holes_with_duplicate_lithology_codes(lithologies: Sequence[Lithology]) -> frozenset[str]:
    """Return hole IDs where the same lithology_code appears more than once."""
    code_counts: dict[str, dict[str, int]] = {}
    duplicate_holes: set[str] = set()
    for lithology in lithologies:
        hole_counts = code_counts.setdefault(lithology.hole_id, {})
        hole_counts[lithology.lithology_code] = hole_counts.get(lithology.lithology_code, 0) + 1
        if hole_counts[lithology.lithology_code] > 1:
            duplicate_holes.add(lithology.hole_id)
    return frozenset(duplicate_holes)


def assign_missing_unit_orders(
    lithologies: Sequence[Lithology],
    *,
    only_duplicate_holes: bool = True,
    force_all_holes: bool = False,
) -> tuple[tuple[Lithology, ...], tuple[str, ...]]:
    """Assign unit_order 1..n by from_depth per hole; preserve explicit Excel values."""
    by_hole_lists: dict[str, list[Lithology]] = defaultdict(list)
    duplicate_holes: set[str] = set()
    codes_per_hole: dict[str, set[str]] = defaultdict(set)
    for lithology in lithologies:
        by_hole_lists[lithology.hole_id].append(lithology)
        hole_codes = codes_per_hole[lithology.hole_id]
        if lithology.lithology_code in hole_codes:
            duplicate_holes.add(lithology.hole_id)
        else:
            hole_codes.add(lithology.lithology_code)
    duplicate_hole_set = frozenset(duplicate_holes)
    updated: list[Lithology] = []
    messages: list[str] = []

    for hole_id in sorted(by_hole_lists):
        intervals = by_hole_lists[hole_id]
        sorted_intervals = sorted(intervals, key=lambda item: (item.from_depth, item.to_depth))
        if only_duplicate_holes and not force_all_holes and hole_id not in duplicate_hole_set:
            updated.extend(sorted_intervals)
            continue

        used_orders = {
            interval.unit_order
            for interval in sorted_intervals
            if interval.unit_order is not None
        }
        next_order = 1
        hole_changed = False
        reassigned: list[Lithology] = []
        for interval in sorted_intervals:
            if interval.unit_order is not None:
                reassigned.append(interval)
                continue
            while next_order in used_orders:
                next_order += 1
            reassigned.append(
                Lithology(
                    hole_id=interval.hole_id,
                    from_depth=interval.from_depth,
                    to_depth=interval.to_depth,
                    lithology_code=interval.lithology_code,
                    hatch_pattern=interval.hatch_pattern,
                    unit_order=next_order,
                )
            )
            used_orders.add(next_order)
            next_order += 1
            hole_changed = True
        if hole_changed:
            messages.append(
                f"{hole_id}: assigned unit_order 1–{len(reassigned)} from depth sequence"
            )
        updated.extend(reassigned)

    return tuple(updated), tuple(messages)


def geology_sheet_counts(parse_result: ParseResult) -> dict[str, int]:
    """Count optional geology records loaded from workbook sheets."""
    return {
        "water_levels": len(parse_result.water_levels),
        "screen_intervals": len(parse_result.screen_intervals),
        "vertical_gradients": len(parse_result.vertical_gradients),
        "deviation_readings": len(parse_result.deviation_readings),
        "correlation_overrides": len(parse_result.correlation_overrides),
        "environmental_readings": len(parse_result.environmental_readings),
        "faults": len(parse_result.faults),
        "unconformities": len(parse_result.unconformities),
    }


def lithology_has_unit_order_column(lithologies: Sequence[Lithology]) -> bool:
    return any(lithology.unit_order is not None for lithology in lithologies)


def apply_unit_order_fix(parse_result: ParseResult) -> ParseResult:
    """Return ParseResult with missing unit_order filled from depth per hole."""
    lithologies, _ = assign_missing_unit_orders(
        parse_result.lithologies,
        only_duplicate_holes=True,
    )
    return parse_result.model_copy(update={"lithologies": lithologies})


def subset_parse_result(
    parse_result: ParseResult,
    hole_ids: Sequence[str],
    *,
    lithology_index: dict[str, tuple[Lithology, ...]] | None = None,
) -> ParseResult:
    """Return collars and lithologies limited to the given hole IDs (order preserved)."""
    if not hole_ids:
        return parse_result.model_copy(
            update={
                "collars": (),
                "lithologies": (),
                "water_levels": (),
                "screen_intervals": (),
                "vertical_gradients": (),
                "deviation_readings": (),
                "correlation_overrides": (),
                "environmental_readings": (),
            }
        )
    hole_set = frozenset(hole_ids)
    collar_lookup = {collar.hole_id: collar for collar in parse_result.collars}
    ordered_collars = tuple(
        collar_lookup[hole_id] for hole_id in hole_ids if hole_id in collar_lookup
    )
    hole_pairs = {
        (hole_ids[index], hole_ids[index + 1]) for index in range(len(hole_ids) - 1)
    }
    if lithology_index is None:
        lithology_index = lithologies_by_hole(parse_result.lithologies)
    selected_lithologies = tuple(
        lithology
        for hole_id in hole_ids
        for lithology in lithology_index.get(hole_id, ())
    )
    def _for_holes(items):
        return tuple(item for item in items if item.hole_id in hole_set)

    return ParseResult(
        collars=ordered_collars,
        lithologies=selected_lithologies,
        errors=parse_result.errors,
        water_levels=_for_holes(parse_result.water_levels),
        screen_intervals=_for_holes(parse_result.screen_intervals),
        vertical_gradients=_for_holes(parse_result.vertical_gradients),
        deviation_readings=_for_holes(parse_result.deviation_readings),
        correlation_overrides=tuple(
            override
            for override in parse_result.correlation_overrides
            if (override.left_hole_id, override.right_hole_id) in hole_pairs
            or (override.right_hole_id, override.left_hole_id) in hole_pairs
        ),
        faults=parse_result.faults,
        unconformities=parse_result.unconformities,
        environmental_readings=_for_holes(parse_result.environmental_readings),
    )


def parse_result_to_json_bundle(parse_result: ParseResult) -> tuple[str, str, str]:
    """Serialize collars, lithologies, and water levels for session/cache storage."""
    return (
        json.dumps([collar.model_dump() for collar in parse_result.collars]),
        json.dumps([lit.model_dump() for lit in parse_result.lithologies]),
        json.dumps([level.model_dump() for level in parse_result.water_levels]),
    )


def subset_json_bundle(
    collars_json: str,
    lithologies_json: str,
    water_levels_json: str,
    hole_ids: Sequence[str],
) -> tuple[str, str, str]:
    """Filter JSON bundles to selected holes without Pydantic validation."""
    if not hole_ids:
        return "[]", "[]", "[]"
    hole_set = frozenset(hole_ids)
    collars_data = json.loads(collars_json)
    lithologies_data = json.loads(lithologies_json)
    water_data = json.loads(water_levels_json)
    collar_lookup = {item["hole_id"]: item for item in collars_data}
    ordered_collars = [collar_lookup[hole_id] for hole_id in hole_ids if hole_id in collar_lookup]
    lithology_index: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in lithologies_data:
        hole_id = item.get("hole_id")
        if hole_id in hole_set:
            lithology_index[str(hole_id)].append(item)
    lithologies = [
        item for hole_id in hole_ids for item in lithology_index.get(hole_id, ())
    ]
    water_levels = [item for item in water_data if item.get("hole_id") in hole_set]
    return json.dumps(ordered_collars), json.dumps(lithologies), json.dumps(water_levels)


def parse_bundle_from_json(
    collars_json: str,
    lithologies_json: str,
    water_levels_json: str,
) -> tuple[tuple[Collar, ...], tuple[Lithology, ...], tuple[WaterLevel, ...]]:
    """Deserialize cached JSON bundles into validated models."""
    return (
        tuple(Collar.model_validate(item) for item in json.loads(collars_json)),
        tuple(Lithology.model_validate(item) for item in json.loads(lithologies_json)),
        tuple(WaterLevel.model_validate(item) for item in json.loads(water_levels_json)),
    )


