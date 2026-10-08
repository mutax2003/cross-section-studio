"""ParseResult transforms: subset, unit_order, serialization helpers."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from itertools import chain

from models import (
    Collar,
    DeviationReading,
    EnvironmentalReading,
    Lithology,
    ParseResult,
    ScreenInterval,
    VerticalGradient,
    WaterLevel,
    WorkbookSectionSpec,
)


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


# A row's place in the site-wide stratigraphic column: (unit_id, piece). ``unit_id``
# identifies one aligned unit (e.g. the upper Clay); ``piece`` counts extra rows of
# one unit that a hole logs as consecutive same-code rows (a logging gap or a
# re-described sample), so a split never shifts the deeper units.
UnitToken = tuple[int, int]


def _code_runs(intervals: Sequence[Lithology]) -> list[tuple[str, float, int]]:
    """Collapse consecutive same-code rows: ``[(code, top_depth, n_rows), …]``."""
    runs: list[tuple[str, float, int]] = []
    for interval in intervals:
        if runs and runs[-1][0] == interval.lithology_code:
            code, top, count = runs[-1]
            runs[-1] = (code, top, count + 1)
        else:
            runs.append((interval.lithology_code, interval.from_depth, 1))
    return runs


def _align_runs(
    runs: Sequence[tuple[str, float]],
    reference: Sequence[tuple[str, float]],
) -> list[int | None]:
    """Order-preserving match of a hole's runs to reference units (same code only).

    Maximises the number of matches, then minimises the summed depth difference
    (so a merged Clay matches the nearer of two reference Clays). Returns, per
    run, the matched reference index or None.
    """
    rows, cols = len(runs), len(reference)
    # best[i][j] = (matches, -depth_cost) for runs[i:] vs reference[j:]
    best = [[(0, 0.0)] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows - 1, -1, -1):
        for j in range(cols - 1, -1, -1):
            options = [best[i + 1][j], best[i][j + 1]]
            if runs[i][0] == reference[j][0]:
                matches, cost = best[i + 1][j + 1]
                options.append((matches + 1, cost - abs(runs[i][1] - reference[j][1])))
            best[i][j] = max(options)
    matched: list[int | None] = []
    i = j = 0
    while i < rows:
        if j < cols and runs[i][0] == reference[j][0]:
            matches, cost = best[i + 1][j + 1]
            if best[i][j] == (matches + 1, cost - abs(runs[i][1] - reference[j][1])):
                matched.append(j)
                i += 1
                j += 1
                continue
        if j < cols and best[i][j] == best[i][j + 1]:
            j += 1
            continue
        matched.append(None)
        i += 1
    return matched


def site_unit_sequence(
    hole_runs: Mapping[str, Sequence[tuple[str, float]]],
) -> tuple[tuple[str, ...], dict[str, list[int]], dict[str, list[str]]]:
    """Align per-hole code sequences into one site-wide top-down unit column.

    ``hole_runs`` maps hole_id → ``[(code, top_depth), …]`` (consecutive same-code
    rows already collapsed). Holes are aligned progressively, longest first, to a
    growing reference column (:func:`_align_runs`); a run that matches nothing —
    a unit present in only some holes, such as a Silt in the north — is inserted
    into the column between its matched neighbours, ordered by median depth
    against other units already in that gap.

    Returns ``(unit_codes, units_by_hole, unaligned)``: the column top-down as codes
    (index = unit id), each hole's unit id per run, and per hole the codes that
    exist elsewhere in the column but could not be lined up (holes disagree on
    their order), for callers to warn about.
    """
    unit_codes: list[str] = []
    unit_depths: list[list[float]] = []
    column: list[int] = []  # unit ids, top-down
    units_by_hole: dict[str, list[int]] = {}
    unaligned: dict[str, list[str]] = {}

    def median_depth(unit: int) -> float:
        return statistics.median(unit_depths[unit])

    order = sorted(hole_runs, key=lambda hole: (-len(hole_runs[hole]), hole))
    for hole_id in order:
        runs = hole_runs[hole_id]
        reference = [(unit_codes[unit], median_depth(unit)) for unit in column]
        matched = _align_runs(runs, reference)
        new_column: list[int] = []
        hole_units: list[int] = []
        ref_cursor = 0
        pending: list[tuple[int, str, float]] = []  # unmatched runs awaiting a gap merge

        def flush_gap(until: int) -> None:
            nonlocal ref_cursor
            gap = column[ref_cursor:until]
            k = 0
            for run_index, code, depth in pending:
                while k < len(gap) and median_depth(gap[k]) <= depth:
                    new_column.append(gap[k])
                    k += 1
                unit = len(unit_codes)
                unit_codes.append(code)
                unit_depths.append([depth])
                new_column.append(unit)
                hole_units[run_index] = unit
            new_column.extend(gap[k:])
            pending.clear()
            ref_cursor = until

        for run_index, ((code, depth), ref_index) in enumerate(zip(runs, matched, strict=True)):
            hole_units.append(-1)
            if ref_index is None:
                pending.append((run_index, code, depth))
                continue
            flush_gap(ref_index)
            unit = column[ref_index]
            new_column.append(unit)
            unit_depths[unit].append(depth)
            hole_units[run_index] = unit
            ref_cursor = ref_index + 1
        flush_gap(len(column))

        used = set(hole_units)
        skipped = [
            runs[run_index][0]
            for run_index, ref_index in enumerate(matched)
            if ref_index is None
            and any(unit_codes[unit] == runs[run_index][0] and unit not in used for unit in column)
        ]
        if skipped:
            unaligned[hole_id] = skipped
        column = new_column
        units_by_hole[hole_id] = hole_units

    # Renumber unit ids top-down so id == position in the column.
    position = {unit: index for index, unit in enumerate(column)}
    return (
        tuple(unit_codes[unit] for unit in column),
        {hole: [position[unit] for unit in units] for hole, units in units_by_hole.items()},
        unaligned,
    )


def _unit_labels(unit_codes: Sequence[str]) -> list[str]:
    seen: dict[str, int] = defaultdict(int)
    labels: list[str] = []
    for code in unit_codes:
        seen[code] += 1
        labels.append(code if seen[code] == 1 else f"{code} #{seen[code]}")
    return labels


def assign_missing_unit_orders(
    lithologies: Sequence[Lithology],
    *,
    only_duplicate_holes: bool = True,
    force_all_holes: bool = False,
) -> tuple[tuple[Lithology, ...], tuple[str, ...]]:
    """Fill missing ``unit_order`` with one site-wide stratigraphic numbering.

    ``stratigraphy._correlation_keys`` joins layers between holes on
    ``(unit_order, lithology_code)`` (or on the code alone when a row has no
    unit_order and the code is unique in its hole). Numbering each hole 1..n by
    depth would shift every deeper number whenever a unit is missing from a hole,
    so instead the hole sequences are aligned into one site-wide column
    (:func:`site_unit_sequence`) and every interval gets its unit's rank there.
    Consecutive same-code rows in a hole are one unit; the extra rows get the
    following numbers (they never share a number — ``ai_quality`` rejects that).

    Trigger: with ``only_duplicate_holes`` (the default) assignment runs only when
    some hole repeats a lithology code and has blank unit_order rows — otherwise
    code-only keys already correlate. ``only_duplicate_holes=False`` or
    ``force_all_holes=True`` assign whenever any row is blank. Once triggered,
    *every* hole with blank rows is numbered, so all holes key the same way.

    Explicit Excel values are always kept. When some rows carry explicit values,
    a blank row takes the explicit number most often given to the same aligned
    unit elsewhere; units never numbered explicitly get numbers above the largest
    explicit value, in site-wide order. A number already used in the same hole is
    bumped to the next free value so no hole repeats a unit_order.
    """
    by_hole: dict[str, list[Lithology]] = defaultdict(list)
    for lithology in lithologies:
        by_hole[lithology.hole_id].append(lithology)
    sorted_by_hole = {
        # Code / unit_order break depth ties so shuffled rows number the same.
        hole_id: sorted(
            items,
            key=lambda item: (
                item.from_depth,
                item.to_depth,
                str(item.lithology_code),
                -1 if item.unit_order is None else item.unit_order,
            ),
        )
        for hole_id, items in by_hole.items()
    }
    holes_missing = {
        hole_id
        for hole_id, items in sorted_by_hole.items()
        if any(item.unit_order is None for item in items)
    }
    if only_duplicate_holes and not force_all_holes:
        triggered = bool(holes_missing & holes_with_duplicate_lithology_codes(lithologies))
    else:
        triggered = bool(holes_missing)
    hole_ids = sorted(sorted_by_hole)
    if not triggered:
        return tuple(chain.from_iterable(sorted_by_hole[h] for h in hole_ids)), ()

    runs_by_hole = {hole_id: _code_runs(items) for hole_id, items in sorted_by_hole.items()}
    unit_codes, units_by_hole, unaligned = site_unit_sequence(
        {hole: [(code, top) for code, top, _n in runs] for hole, runs in runs_by_hole.items()}
    )
    tokens_by_hole: dict[str, list[UnitToken]] = {}
    max_pieces: dict[int, int] = defaultdict(int)
    for hole_id, runs in runs_by_hole.items():
        tokens: list[UnitToken] = []
        for unit, (_code, _top, count) in zip(units_by_hole[hole_id], runs, strict=True):
            tokens.extend((unit, piece) for piece in range(count))
            max_pieces[unit] = max(max_pieces[unit], count)
        tokens_by_hole[hole_id] = tokens
    sequence = [
        (unit, piece) for unit in range(len(unit_codes)) for piece in range(max_pieces[unit])
    ]

    explicit: dict[UnitToken, Counter[int]] = defaultdict(Counter)
    for hole_id, items in sorted_by_hole.items():
        for token, item in zip(tokens_by_hole[hole_id], items, strict=True):
            if item.unit_order is not None:
                explicit[token][item.unit_order] += 1
    order_for_token: dict[UnitToken, int] = {}
    if explicit:
        next_new = max(max(counts) for counts in explicit.values()) + 1
        for token in sequence:
            if token in explicit:
                # Most common given value; ties go to the smallest (not to
                # whichever hole came first in the sheet).
                counts = explicit[token]
                order_for_token[token] = min(counts, key=lambda value: (-counts[value], value))
            else:
                order_for_token[token] = next_new
                next_new += 1
    else:
        order_for_token = {token: rank for rank, token in enumerate(sequence, start=1)}
    ceiling = max(order_for_token.values(), default=0)

    updated: list[Lithology] = []
    bumped_holes: list[str] = []
    for hole_id in hole_ids:
        items = sorted_by_hole[hole_id]
        if hole_id not in holes_missing:
            updated.extend(items)
            continue
        used = {item.unit_order for item in items if item.unit_order is not None}
        bumped = False
        for token, item in zip(tokens_by_hole[hole_id], items, strict=True):
            if item.unit_order is not None:
                updated.append(item)
                continue
            order = order_for_token[token]
            if order in used:
                ceiling += 1
                order = ceiling
                bumped = True
            used.add(order)
            updated.append(item.model_copy(update={"unit_order": order}))
        if bumped:
            bumped_holes.append(hole_id)

    messages = [
        f"Auto-assigned unit_order to {len(holes_missing)} hole(s) from one site-wide "
        f"sequence of {len(unit_codes)} units (top-down: {', '.join(_unit_labels(unit_codes))})"
        " — add a unit_order column to control correlation."
    ]
    for hole_id in sorted(unaligned):
        messages.append(
            f"{hole_id}: {', '.join(unaligned[hole_id])} could not be lined up with other "
            "holes (layer order differs) and will pinch out — set unit_order to correlate it."
        )
    if bumped_holes:
        messages.append(
            f"{', '.join(bumped_holes)}: auto unit_order clashed with an explicit value "
            "in the same hole and was renumbered — check unit_order there."
        )
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


def format_section_specs_as_batch_text(specs: Sequence[WorkbookSectionSpec]) -> str:
    """Format Sections rows as ``Label | hole1, hole2, …`` lines for Configure batch ZIP."""
    return "\n".join(f"{spec.label} | {', '.join(spec.hole_ids)}" for spec in specs)


def lithology_has_unit_order_column(lithologies: Sequence[Lithology]) -> bool:
    return any(lithology.unit_order is not None for lithology in lithologies)


def apply_unit_order_fix(parse_result: ParseResult) -> ParseResult:
    """Return ParseResult with missing unit_order filled from the site-wide unit sequence."""
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
        if isinstance(lithology, Lithology)
        else Lithology.model_validate(lithology)
        for hole_id in hole_ids
        for lithology in (lithology_index.get(hole_id) or ())
    )
    water_levels: list[WaterLevel] = []
    screen_intervals: list[ScreenInterval] = []
    vertical_gradients: list[VerticalGradient] = []
    deviation_readings: list[DeviationReading] = []
    environmental_readings: list[EnvironmentalReading] = []
    for item in chain(
        parse_result.water_levels,
        parse_result.screen_intervals,
        parse_result.vertical_gradients,
        parse_result.deviation_readings,
        parse_result.environmental_readings,
    ):
        if item.hole_id not in hole_set:
            continue
        if isinstance(item, WaterLevel):
            water_levels.append(item)
        elif isinstance(item, ScreenInterval):
            screen_intervals.append(item)
        elif isinstance(item, VerticalGradient):
            vertical_gradients.append(item)
        elif isinstance(item, DeviationReading):
            deviation_readings.append(item)
        elif isinstance(item, EnvironmentalReading):
            environmental_readings.append(item)

    # model_copy avoids full-graph revalidation quirks when filtering already-valid rows
    return parse_result.model_copy(
        update={
            "collars": ordered_collars,
            "lithologies": selected_lithologies,
            "water_levels": tuple(water_levels),
            "screen_intervals": tuple(screen_intervals),
            "vertical_gradients": tuple(vertical_gradients),
            "deviation_readings": tuple(deviation_readings),
            "correlation_overrides": tuple(
                override
                for override in parse_result.correlation_overrides
                if (override.left_hole_id, override.right_hole_id) in hole_pairs
                or (override.right_hole_id, override.left_hole_id) in hole_pairs
            ),
            "environmental_readings": tuple(environmental_readings),
        }
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


