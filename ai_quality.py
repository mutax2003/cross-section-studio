"""Local data quality checks, column mapping, and lithology normalization."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Literal

import pandas as pd
from rapidfuzz import fuzz

from chemistry_qc import lab_qualifier_warnings, mixed_unit_warnings, rdl_above_guideline_notes
from constants import CANONICAL_LITHOLOGY_CODES
from hydro_metrics import (
    APPARENT_LNAPL_THICKNESS_WARN_M,
    DEFAULT_NEST_DISTANCE_M,
    DEFAULT_SYNOPTIC_WINDOW_DAYS,
    NestedVerticalGradient,
    absurd_horizontal_gradient_warnings,
    assign_screens_to_connect_groups,
    format_vertical_gradient_note,
    has_lnapl,
    horizontal_gradients_along_profile,
    is_measured_water_level,
    latest_environmental_readings,
    latest_water_by_hole,
    latest_water_readings,
    lnapl_correction_note,
    lnapl_thickness_m,
    nested_vertical_gradients,
    reading_date,
    screen_elevation_range,
    synoptic_date_warnings,
    vertical_gradient_arrow_conflicts,
    water_depth_along_hole_bgs,
    water_head_masl,
)
from models import (
    COLLAR_COLUMNS,
    LITHOLOGY_COLUMNS,
    Collar,
    DeviationReading,
    EnvironmentalReading,
    Lithology,
    ScreenInterval,
    Transect,
    VerticalGradient,
    WaterLevel,
)
from paths import lithology_aliases_path
from projection import TransectGeometry, build_along_hole_tvd, project_collar_to_transect
from render_profiles import chemistry_thresholds_from_readings

logger = logging.getLogger(__name__)
MAPPING_CONFIDENCE_THRESHOLD = 0.8
# Intervals thinner than this are left out of the inter-hole fence by
# ``stratigraphy.MIN_FENCE_THICKNESS_M`` (kept in sync; no engine import here).
ZERO_THICKNESS_M = 1e-3

COLLAR_ALIASES: dict[str, set[str]] = {
    "hole_id": {"hole_id", "hole", "bh_id", "borehole", "borehole_id", "id", "bh"},
    "easting": {"easting", "east", "e", "x", "utm_e", "utm_easting"},
    "northing": {"northing", "north", "n", "y", "utm_n", "utm_northing"},
    "elevation": {"elevation", "rl", "z", "collar_rl", "collar_elevation", "reduced_level"},
    "total_depth": {"total_depth", "td", "depth", "max_depth", "hole_depth", "final_depth"},
}
LITHOLOGY_ALIASES: dict[str, set[str]] = {
    "hole_id": {"hole_id", "hole", "bh_id", "borehole", "borehole_id", "id", "bh"},
    "from_depth": {"from_depth", "from", "top", "depth_from", "start_depth"},
    "to_depth": {"to_depth", "to", "base", "depth_to", "end_depth", "bottom_depth"},
    "lithology_code": {"lithology_code", "lithology", "lith", "code", "unit", "geo_code", "strat"},
    "hatch_pattern": {"hatch_pattern", "hatch", "pattern", "fill"},
}
SHEET_ALIASES: dict[str, set[str]] = {
    "collars": {"collars", "collar", "boreholes", "holes", "bh"},
    "lithology": {"lithology", "lith", "intervals", "geology", "stratigraphy"},
}


class Severity(str, Enum):  # noqa: UP042 — keep (str, Enum): str() formatting of members differs under StrEnum
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class ColumnMapping:
    source_column: str
    canonical_column: str
    confidence: float


@dataclass(frozen=True)
class MappingProposal:
    collars_sheet: str
    lithology_sheet: str
    collar_column_mappings: tuple[ColumnMapping, ...]
    lithology_column_mappings: tuple[ColumnMapping, ...]

    @property
    def low_confidence_mappings(self) -> tuple[ColumnMapping, ...]:
        all_mappings = self.collar_column_mappings + self.lithology_column_mappings
        return tuple(
            mapping
            for mapping in all_mappings
            if mapping.confidence < MAPPING_CONFIDENCE_THRESHOLD
        )


@dataclass(frozen=True)
class QualityIssue:
    code: str
    message: str
    severity: Literal["error", "warning", "info"]
    hole_id: str | None = None
    row: int | None = None


@dataclass(frozen=True)
class QualityReport:
    issues: tuple[QualityIssue, ...]
    mapping_proposal: MappingProposal | None
    unmapped_lithologies: tuple[str, ...]
    normalized_lithology_count: int
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0

    def __post_init__(self) -> None:
        if self.error_count or self.warning_count or self.info_count:
            return
        errors = warnings = infos = 0
        for issue in self.issues:
            if issue.severity == Severity.ERROR.value:
                errors += 1
            elif issue.severity == Severity.WARNING.value:
                warnings += 1
            else:
                infos += 1
        object.__setattr__(self, "error_count", errors)
        object.__setattr__(self, "warning_count", warnings)
        object.__setattr__(self, "info_count", infos)

    @property
    def has_blocking_errors(self) -> bool:
        return self.error_count > 0


def _normalize_header(value: object) -> str:
    return re.sub(r"\s+", "_", str(value).strip().lower())


@lru_cache(maxsize=1)
def load_lithology_aliases(path: str | None = None) -> dict[str, str]:
    alias_path = Path(path) if path else lithology_aliases_path()
    if not alias_path.exists():
        return {}
    with alias_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return {_normalize_header(key): str(value).strip() for key, value in raw.items()}


def save_lithology_alias(source_code: str, canonical_code: str, path: Path | None = None) -> None:
    alias_path = path or lithology_aliases_path()
    aliases = dict(load_lithology_aliases(str(alias_path)))
    aliases[_normalize_header(source_code)] = canonical_code
    try:
        alias_path.parent.mkdir(parents=True, exist_ok=True)
        with alias_path.open("w", encoding="utf-8") as handle:
            json.dump(aliases, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except (OSError, PermissionError) as exc:
        logger.warning("Failed to save lithology alias to %s: %s", alias_path, exc)
        raise RuntimeError(
            f"Could not save lithology alias to {alias_path}: {exc}"
        ) from exc
    load_lithology_aliases.cache_clear()


@lru_cache(maxsize=1)
def _canonical_by_lower() -> dict[str, str]:
    """Lowercase → canonical code map for O(1) case-insensitive matching."""
    return {canonical.lower(): canonical for canonical in CANONICAL_LITHOLOGY_CODES}


def normalize_lithology_code(code: str, aliases: dict[str, str] | None = None) -> str:
    stripped = code.strip()
    if not stripped:
        return ""
    lookup = aliases if aliases is not None else load_lithology_aliases()
    normalized_key = _normalize_header(stripped)
    canonical = _canonical_by_lower().get(normalized_key)
    if canonical is not None:
        return canonical
    title = stripped.title()
    if title in CANONICAL_LITHOLOGY_CODES:
        return title
    if normalized_key in lookup:
        return lookup[normalized_key]
    return stripped


def normalize_lithologies(
    lithologies: Sequence[Lithology],
    aliases: dict[str, str] | None = None,
) -> tuple[tuple[Lithology, ...], int, tuple[str, ...]]:
    lookup = aliases if aliases is not None else load_lithology_aliases()
    normalized: list[Lithology] = []
    changed = 0
    unmapped: set[str] = set()

    for lithology in lithologies:
        canonical = normalize_lithology_code(lithology.lithology_code, lookup)
        if canonical != lithology.lithology_code:
            changed += 1
        if canonical not in CANONICAL_LITHOLOGY_CODES and canonical == lithology.lithology_code.strip():
            unmapped.add(lithology.lithology_code)
        normalized.append(
            Lithology(
                hole_id=lithology.hole_id,
                from_depth=lithology.from_depth,
                to_depth=lithology.to_depth,
                lithology_code=canonical,
                hatch_pattern=lithology.hatch_pattern,
            )
        )

    return tuple(normalized), changed, tuple(sorted(unmapped))


def _match_sheet_name(sheet_names: list[str], target: str) -> tuple[str, float]:
    normalized = {_normalize_header(name): name for name in sheet_names}
    aliases = SHEET_ALIASES[target]
    for alias in aliases:
        if alias in normalized:
            return normalized[alias], 1.0
    best_name = sheet_names[0]
    best_score = 0.0
    for name in sheet_names:
        score = fuzz.ratio(_normalize_header(name), target) / 100.0
        if score > best_score:
            best_score = score
            best_name = name
    return best_name, best_score


def _match_column(
    column_name: str,
    canonical: str,
    alias_map: dict[str, set[str]],
) -> ColumnMapping:
    normalized = _normalize_header(column_name)
    aliases = alias_map.get(canonical, set())
    if normalized == canonical or normalized in aliases:
        return ColumnMapping(column_name, canonical, 1.0)

    scores = [fuzz.ratio(normalized, alias) / 100.0 for alias in aliases | {canonical}]
    best_score = max(scores) if scores else 0.0
    return ColumnMapping(column_name, canonical, best_score)


def propose_column_mappings(
    columns: Sequence[str],
    required: set[str],
    alias_map: dict[str, set[str]],
) -> tuple[ColumnMapping, ...]:
    candidates: list[ColumnMapping] = []
    for canonical in sorted(required):
        for column in columns:
            candidates.append(_match_column(column, canonical, alias_map))

    candidates.sort(key=lambda item: item.confidence, reverse=True)
    assigned_sources: set[str] = set()
    assigned_canonical: set[str] = set()
    chosen: dict[str, ColumnMapping] = {}

    for candidate in candidates:
        if candidate.canonical_column in assigned_canonical:
            continue
        if candidate.source_column in assigned_sources:
            continue
        chosen[candidate.canonical_column] = candidate
        assigned_sources.add(candidate.source_column)
        assigned_canonical.add(candidate.canonical_column)

    mappings: list[ColumnMapping] = []
    for canonical in sorted(required):
        if canonical in chosen:
            mappings.append(chosen[canonical])
        else:
            mappings.append(ColumnMapping(f"<missing:{canonical}>", canonical, 0.0))

    return tuple(mappings)


def propose_workbook_mapping(workbook: pd.ExcelFile) -> MappingProposal:
    collars_sheet, _ = _match_sheet_name(workbook.sheet_names, "collars")
    lithology_sheet, _ = _match_sheet_name(workbook.sheet_names, "lithology")

    collar_columns = list(pd.read_excel(workbook, sheet_name=collars_sheet, nrows=0).columns)
    lithology_columns = list(pd.read_excel(workbook, sheet_name=lithology_sheet, nrows=0).columns)

    return MappingProposal(
        collars_sheet=collars_sheet,
        lithology_sheet=lithology_sheet,
        collar_column_mappings=propose_column_mappings(
            [str(column) for column in collar_columns],
            COLLAR_COLUMNS,
            COLLAR_ALIASES,
        ),
        lithology_column_mappings=propose_column_mappings(
            [str(column) for column in lithology_columns],
            LITHOLOGY_COLUMNS,
            LITHOLOGY_ALIASES,
        ),
    )


def apply_column_mapping(df: pd.DataFrame, mappings: Sequence[ColumnMapping]) -> pd.DataFrame:
    rename_map = {
        mapping.source_column: mapping.canonical_column
        for mapping in mappings
        if not mapping.source_column.startswith("<missing:")
    }
    renamed = df.rename(columns=rename_map)
    normalized = {col: _normalize_header(col) for col in renamed.columns}
    return renamed.rename(columns=normalized)


def read_mapped_sheets(
    source: str | Path | BinaryIO | BytesIO,
    mapping: MappingProposal,
    workbook: pd.ExcelFile | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_workbook = workbook or pd.ExcelFile(source)
    collars_df = apply_column_mapping(
        pd.read_excel(active_workbook, sheet_name=mapping.collars_sheet),
        mapping.collar_column_mappings,
    )
    lithology_df = apply_column_mapping(
        pd.read_excel(active_workbook, sheet_name=mapping.lithology_sheet),
        mapping.lithology_column_mappings,
    )
    return collars_df, lithology_df


def _m(value: float) -> str:
    """Depth for messages: 3.0 -> "3", 8.25 -> "8.25"."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _hole_quality_issues(collar: Collar, intervals: list[Lithology]) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    if not intervals:
        issues.append(
            QualityIssue(
                code="no_lithology",
                message=f"{collar.hole_id} has no lithology intervals — add them in the Lithology sheet",
                severity=Severity.WARNING.value,
                hole_id=collar.hole_id,
            )
        )
        return issues

    if len(intervals) == 1:
        issues.append(
            QualityIssue(
                code="single_interval",
                message=f"{collar.hole_id} has only one lithology interval",
                severity=Severity.INFO.value,
                hole_id=collar.hole_id,
            )
        )

    # Full key so equal tops (e.g. a zero-thickness row) sort the same way
    # whatever the sheet's row order.
    sorted_intervals = sorted(
        intervals, key=lambda item: (item.from_depth, item.to_depth, str(item.lithology_code))
    )
    if sorted_intervals[0].from_depth > 0.01:
        issues.append(
            QualityIssue(
                code="depth_gap",
                message=(
                    f"{collar.hole_id}: nothing logged from 0 to "
                    f"{_m(sorted_intervals[0].from_depth)} m (missing coverage)"
                ),
                severity=Severity.WARNING.value,
                hole_id=collar.hole_id,
            )
        )

    # Zero-thickness rows get their own warning (zero_thickness_interval) and
    # are dropped from the fence; pairing them here read "1.2–3.29 m overlaps
    # 1.2–1.2 m".
    thick = [i for i in sorted_intervals if i.to_depth - i.from_depth >= ZERO_THICKNESS_M]
    for left, right in zip(thick, thick[1:]):
        if right.from_depth > left.to_depth + 0.01:
            issues.append(
                QualityIssue(
                    code="depth_gap",
                    message=(
                        f"{collar.hole_id}: gap between {_m(left.to_depth)} and "
                        f"{_m(right.from_depth)} m — nothing logged there"
                    ),
                    severity=Severity.WARNING.value,
                    hole_id=collar.hole_id,
                )
            )
        if right.from_depth < left.to_depth - 0.01:
            issues.append(
                QualityIssue(
                    code="depth_overlap",
                    message=(
                        f"{left.hole_id}: {_m(left.from_depth)}–{_m(left.to_depth)} m overlaps "
                        f"{_m(right.from_depth)}–{_m(right.to_depth)} m — fix from_depth/to_depth "
                        "in Lithology"
                    ),
                    severity=Severity.ERROR.value,
                    hole_id=left.hole_id,
                )
            )

    last = max(thick or sorted_intervals, key=lambda interval: interval.to_depth)
    if last.to_depth < collar.total_depth - 0.01:
        issues.append(
            QualityIssue(
                code="depth_gap",
                message=(
                    f"{collar.hole_id}: nothing logged from {_m(last.to_depth)} m to "
                    f"total depth {_m(collar.total_depth)} m (missing coverage)"
                ),
                severity=Severity.WARNING.value,
                hole_id=collar.hole_id,
            )
        )

    code_counts: dict[str, int] = {}
    for interval in sorted_intervals:
        code_counts[interval.lithology_code] = code_counts.get(interval.lithology_code, 0) + 1
    duplicate_codes = {code for code, count in code_counts.items() if count > 1}
    if duplicate_codes:
        missing_order = [
            interval.lithology_code
            for interval in sorted_intervals
            if interval.lithology_code in duplicate_codes and interval.unit_order is None
        ]
        if missing_order:
            issues.append(
                QualityIssue(
                    code="duplicate_lithology_no_unit_order",
                    message=(
                        f"{collar.hole_id} repeats lithology code(s) "
                        f"{', '.join(sorted(set(missing_order)))} without unit_order — "
                        "add unit_order so layers correlate between holes"
                    ),
                    severity=Severity.ERROR.value,
                    hole_id=collar.hole_id,
                )
            )
        orders = [interval.unit_order for interval in sorted_intervals if interval.unit_order is not None]
        if len(orders) != len(set(orders)):
            issues.append(
                QualityIssue(
                    code="duplicate_unit_order",
                    message=f"{collar.hole_id} uses the same unit_order twice — give each layer its own number",
                    severity=Severity.ERROR.value,
                    hole_id=collar.hole_id,
                )
            )

    issues.extend(_zero_thickness_issues(collar.hole_id, sorted_intervals))
    against_depth = _unit_order_against_depth_issue(collar.hole_id, sorted_intervals)
    if against_depth is not None:
        issues.append(against_depth)
    return issues


def _zero_thickness_issues(hole_id: str, intervals: Sequence[Lithology]) -> list[QualityIssue]:
    """Warn for intervals with from_depth == to_depth (dropped from the fence)."""
    return [
        QualityIssue(
            code="zero_thickness_interval",
            message=(
                f"{hole_id}: zero-thickness interval {interval.lithology_code} at "
                f"{_m(interval.from_depth)} m (from_depth = to_depth) ignored in the fence — "
                "fix to_depth or delete the row"
            ),
            severity=Severity.WARNING.value,
            hole_id=hole_id,
        )
        for interval in intervals
        if interval.to_depth - interval.from_depth < ZERO_THICKNESS_M
    ]


def _unit_order_against_depth_issue(
    hole_id: str, intervals: Sequence[Lithology]
) -> QualityIssue | None:
    """Warn once per hole when unit_order decreases going down the hole.

    The fence judges correlation crossings on depth, so contradictory numbering
    makes numbered matches drop out as crossing pinch-outs.
    """
    # Zero-thickness rows are left out of the fence (and warned about on
    # their own), so they cannot make the numbering run against depth.
    numbered = [
        interval
        for interval in intervals
        if interval.unit_order is not None and interval.to_depth - interval.from_depth >= ZERO_THICKNESS_M
    ]
    for upper, lower in zip(numbered, numbered[1:]):
        if lower.from_depth < upper.to_depth - 0.01 or lower.from_depth <= upper.from_depth:
            continue  # overlapping / same-top rows are reported elsewhere
        if lower.unit_order < upper.unit_order:
            return QualityIssue(
                code="unit_order_against_depth",
                message=(
                    f"unit_order runs against depth in {hole_id}: {upper.lithology_code} "
                    f"(unit_order {upper.unit_order}) at {_m(upper.from_depth)} m sits above "
                    f"{lower.lithology_code} (unit_order {lower.unit_order}) at "
                    f"{_m(lower.from_depth)} m — check numbering"
                ),
                severity=Severity.WARNING.value,
                hole_id=hole_id,
            )
    return None


def collars_use_placeholder_elevation(
    collars: Sequence[Collar],
    placeholder_m: float,
    *,
    tolerance: float = 0.01,
) -> bool:
    if not collars:
        return False
    return all(abs(collar.elevation - placeholder_m) <= tolerance for collar in collars)


def _coordinates_look_like_degrees(collars: Sequence[Collar]) -> QualityIssue | None:
    """Warn when easting/northing hold latitude/longitude, not metres.

    Degrees typed into easting/northing (e.g. -79.38 / 43.65) project as a
    section a few centimetres long. Local site grids can also be small
    numbers (0, 5, 10 m), so only flag a sub-metre spread of non-integer
    values that all fit the degree range.
    """
    if len(collars) < 2:
        return None
    eastings = [float(c.easting) for c in collars]
    northings = [float(c.northing) for c in collars]
    in_range = all(abs(e) <= 180 for e in eastings) and all(abs(n) <= 90 for n in northings)
    spread = max(max(eastings) - min(eastings), max(northings) - min(northings))
    decimal_degrees = any(
        abs(v) >= 1 and abs(v - round(v)) > 1e-6 for v in (*eastings, *northings)
    )
    if not (in_range and 0 < spread < 1.0 and decimal_degrees):
        return None
    return QualityIssue(
        code="coordinates_in_degrees",
        message=(
            "Collar easting/northing look like latitude/longitude (all within ±180/±90 and "
            f"only {spread:.4f} apart), so the section would be a few centimetres long — "
            "enter projected coordinates in metres (e.g. UTM), or put degrees in latitude/"
            "longitude columns"
        ),
        severity=Severity.WARNING.value,
    )


def hole_id_case_variant_issues(collars: Sequence[Collar]) -> list[QualityIssue]:
    """Collars whose IDs differ only by case / spacing (``MW-01`` vs ``mw-01``).

    They are separate holes to the section (IDs match exactly), which is
    almost never what the logger meant.
    """
    groups: dict[str, list[str]] = {}
    for collar in collars:
        key = re.sub(r"\s+", "", collar.hole_id).casefold()
        groups.setdefault(key, []).append(collar.hole_id)
    issues: list[QualityIssue] = []
    for ids in groups.values():
        distinct = list(dict.fromkeys(ids))
        if len(distinct) < 2:
            continue
        issues.append(
            QualityIssue(
                code="hole_id_case_variants",
                message=(
                    f"Hole IDs {' and '.join(distinct)} differ only by case or spaces — they are "
                    "treated as separate holes; use one spelling in every sheet"
                ),
                severity=Severity.WARNING.value,
                hole_id=distinct[0],
            )
        )
    return issues


def _horizontal_hole_issue(hole_id: str, where: str) -> QualityIssue:
    return QualityIssue(
        code="horizontal_hole",
        message=(
            f"{hole_id}: inclination 90° (horizontal) {where} — intervals logged along a "
            "horizontal hole have no vertical extent and won't show on the section; check "
            "the angle (0 = straight down, or −90 for a mining dip)"
        ),
        severity=Severity.WARNING.value,
        hole_id=hole_id,
    )


def survey_quality_issues(
    collars: Sequence[Collar],
    deviation_readings: Sequence[DeviationReading] | None,
) -> list[QualityIssue]:
    """Horizontal (90°) holes / survey stations and Deviations rows with no collar."""
    issues: list[QualityIssue] = []
    collar_ids = {collar.hole_id for collar in collars}
    for collar in collars:
        # +90 only: −90 is a mining dip, i.e. straight down.
        if collar.inclination_deg == 90.0:
            issues.append(_horizontal_hole_issue(collar.hole_id, "on Collars"))
    horizontal: dict[str, list[float]] = {}
    orphans: dict[str, int] = {}
    for reading in deviation_readings or ():
        if reading.hole_id not in collar_ids:
            orphans[reading.hole_id] = orphans.get(reading.hole_id, 0) + 1
            continue
        if reading.inclination_deg == 90.0:
            horizontal.setdefault(reading.hole_id, []).append(reading.depth)
    for hole_id, depths in horizontal.items():
        shown = ", ".join(f"{_m(depth)} m" for depth in sorted(depths)[:5])
        issues.append(_horizontal_hole_issue(hole_id, f"on Deviations at {shown}"))
    if orphans:
        listed = ", ".join(f"{hole} ({count})" for hole, count in sorted(orphans.items()))
        issues.append(
            QualityIssue(
                code="orphan_deviations",
                message=(
                    f"Deviations rows for holes not in Collars are not used: {listed} — add "
                    "the holes to Collars or fix the spelling"
                ),
                severity=Severity.WARNING.value,
            )
        )
    return issues


def analyze_parsed_data(
    collars: Sequence[Collar],
    lithologies: Sequence[Lithology],
    *,
    transect: Transect | None = None,
    offset_threshold_m: float = 50.0,
    mapping_proposal: MappingProposal | None = None,
    aliases: dict[str, str] | None = None,
    placeholder_elevation_m: float | None = None,
    deviation_readings: Sequence[DeviationReading] | None = None,
) -> QualityReport:
    issues: list[QualityIssue] = []
    collar_by_id = {collar.hole_id: collar for collar in collars}
    collar_ids = set(collar_by_id)
    lithology_by_hole: dict[str, list[Lithology]] = {}
    degrees_issue = _coordinates_look_like_degrees(collars)
    if degrees_issue is not None:
        issues.append(degrees_issue)
    issues.extend(hole_id_case_variant_issues(collars))
    issues.extend(survey_quality_issues(collars, deviation_readings))

    for lithology in lithologies:
        lithology_by_hole.setdefault(lithology.hole_id, []).append(lithology)
        if lithology.hole_id not in collar_ids:
            issues.append(
                QualityIssue(
                    code="orphan_lithology",
                    message=(
                        f"{lithology.hole_id} is in Lithology but not in Collars — "
                        "add it to Collars or fix the spelling"
                    ),
                    severity=Severity.ERROR.value,
                    hole_id=lithology.hole_id,
                )
            )
            continue

        collar = collar_by_id[lithology.hole_id]
        if lithology.to_depth > collar.total_depth + 0.01:
            issues.append(
                QualityIssue(
                    code="below_td",
                    message=(
                        f"{lithology.hole_id}: interval to {_m(lithology.to_depth)} m exceeds total "
                        f"depth {_m(collar.total_depth)} m — fix to_depth or total_depth in Collars"
                    ),
                    severity=Severity.ERROR.value,
                    hole_id=lithology.hole_id,
                )
            )

    if len(collars) >= 2:
        unique_xy = {(collar.easting, collar.northing) for collar in collars}
        if len(unique_xy) == 1:
            issues.append(
                QualityIssue(
                    code="flat_collar_grid",
                    message="All collars have the same easting/northing — check the Collars coordinates",
                    severity=Severity.WARNING.value,
                )
            )

    for collar in collars:
        issues.extend(_hole_quality_issues(collar, lithology_by_hole.get(collar.hole_id, [])))

    if placeholder_elevation_m is not None and collars_use_placeholder_elevation(
        collars,
        placeholder_elevation_m,
    ):
        issues.append(
            QualityIssue(
                code="placeholder_elevation",
                message=(
                    f"All collar elevations use the profile placeholder ({placeholder_elevation_m:.1f} m). "
                    "Set a site elevation or ingest survey RL before interpolated sections."
                ),
                severity=Severity.WARNING.value,
            )
        )

    if transect is not None and collars:
        geometry = TransectGeometry.from_transect(transect)
        hole_ids = [collar.hole_id for collar in collars]
        _, offsets = geometry.project_many(
            [collar.easting for collar in collars],
            [collar.northing for collar in collars],
        )
        for hole_id, offset in zip(hole_ids, offsets, strict=True):
            if offset > offset_threshold_m:
                issues.append(
                    QualityIssue(
                        code="off_transect",
                        message=(
                            f"{hole_id} is {offset:.1f} m from the section line "
                            f"(limit {offset_threshold_m:.1f} m)"
                        ),
                        severity=Severity.WARNING.value,
                        hole_id=hole_id,
                    )
                )

    _, normalized_count, unmapped = normalize_lithologies(lithologies, aliases)
    return QualityReport(
        issues=tuple(issues),
        mapping_proposal=mapping_proposal,
        unmapped_lithologies=unmapped,
        normalized_lithology_count=normalized_count,
    )


def analyze_workbook(
    source: str | Path | BinaryIO | BytesIO,
) -> MappingProposal:
    workbook = pd.ExcelFile(source)
    return propose_workbook_mapping(workbook)


@dataclass(frozen=True)
class WaterSeriesSummary:
    series_id: str
    series_label: str
    hole_count: int
    missing_hole_ids: tuple[str, ...]


@dataclass(frozen=True)
class WaterQualitySummary:
    series: tuple[WaterSeriesSummary, ...]
    holes_without_any_water: tuple[str, ...]
    warnings: tuple[str, ...]
    total_levels: int
    # Nested / co-located well vertical gradients (needs ``screen_intervals``).
    vertical_gradients: tuple[NestedVerticalGradient, ...] = ()
    # Informational lines (artesian heads, dry screens, computed iv). Each one
    # is also in ``warnings`` (prefixed ``Info:``) so the existing Validate
    # list shows it; a UI may render ``notes`` with a softer style instead.
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScreenQualitySummary:
    warnings: tuple[str, ...]
    total_intervals: int


# --- Overlay readings far outside their hole --------------------------------
# A water / screen / chemistry depth this far beyond the hole is a typo (mm
# typed as m, a pasted serial) — plotted, it stretches the section's depth axis
# until the geology is a sliver. The figure leaves such readings out with a QA
# note (pipeline.exclude_implausible_overlays); Validate names them the same way.
OVERLAY_BEYOND_TD_MIN_M = 5.0
OVERLAY_BEYOND_TD_FRACTION = 0.25
# Flowing-artesian heads above ground are real, but not by tens of metres.
OVERLAY_ABOVE_GROUND_M = 30.0
_NOT_PLOTTED = "not plotted; check units"


def _num(value: float) -> str:
    """Compact number for messages: 3200 -> "3200", 1e12 -> "1e+12"."""
    return _m(value) if abs(value) < 1e5 else f"{value:g}"


def overlay_depth_limit_m(total_depth: float) -> float:
    """Deepest plausible overlay depth: TD + max(5 m, 25 % of TD)."""
    td = max(float(total_depth), 0.0)
    return td + max(OVERLAY_BEYOND_TD_MIN_M, OVERLAY_BEYOND_TD_FRACTION * td)


def implausible_water_reading(
    level: WaterLevel,
    *,
    collar_rl: float,
    total_depth: float,
    stick_up_m: float | None = None,
) -> str | None:
    """Why a measured water reading lies far outside its hole (``None`` = plausible).

    Depth below ground past :func:`overlay_depth_limit_m`, or a head more than
    ``OVERLAY_ABOVE_GROUND_M`` above ground. Dry / NM readings carry no level.
    """
    if not is_measured_water_level(level):
        return None
    hole, td = level.hole_id, float(total_depth)
    limit = overlay_depth_limit_m(td)
    if level.elevation_masl is not None:
        head = float(level.elevation_masl)
        above = head - float(collar_rl)
        if above > OVERLAY_ABOVE_GROUND_M:
            return (
                f"{hole} water elevation {_num(head)} m is {_num(above)} m above ground "
                f"(collar {_num(collar_rl)} m) — {_NOT_PLOTTED}"
            )
        if -above > limit:
            return (
                f"{hole} water elevation {_num(head)} m is beyond the hole (collar "
                f"{_num(collar_rl)} m, TD {_num(td)} m) — {_NOT_PLOTTED}"
            )
        return None
    depth_bgs = water_depth_along_hole_bgs(level, stick_up_m)
    if depth_bgs > limit:
        return f"{hole} water depth {_num(level.depth)} m is beyond the hole (TD {_num(td)} m) — {_NOT_PLOTTED}"
    if -depth_bgs > OVERLAY_ABOVE_GROUND_M:
        return (
            f"{hole} water depth {_num(level.depth)} m below top of casing puts the water "
            f"{_num(-depth_bgs)} m above ground — {_NOT_PLOTTED}"
        )
    return None


def implausible_screen_interval(screen: ScreenInterval, *, total_depth: float) -> str | None:
    """Why a screen lies far below its hole (``None`` = plausible)."""
    if screen.to_depth <= overlay_depth_limit_m(total_depth):
        return None
    return (
        f"{screen.hole_id} screen {_num(screen.from_depth)}–{_num(screen.to_depth)} m is beyond "
        f"the hole (TD {_num(total_depth)} m) — {_NOT_PLOTTED}"
    )


def implausible_environmental_reading(
    reading: EnvironmentalReading, *, total_depth: float
) -> str | None:
    """Why a sample lies far below its hole (``None`` = plausible)."""
    deepest = reading.depth if reading.depth is not None else reading.to_depth
    if deepest is None or deepest <= overlay_depth_limit_m(total_depth):
        return None
    if reading.depth is not None:
        where = f"sample depth {_num(reading.depth)} m"
    else:
        where = f"sample interval {_num(reading.from_depth or 0.0)}–{_num(deepest)} m"
    return (
        f"{reading.hole_id} {reading.parameter} {where} is beyond the hole "
        f"(TD {_num(total_depth)} m) — {_NOT_PLOTTED}"
    )


def chemistry_threshold_inversion_warnings(
    thresholds: Mapping[str, tuple[float | None, float | None]] | None,
    *,
    green_max: float | None = None,
    orange_max: float | None = None,
) -> list[str]:
    """Colour bands entered upside down (``green_max`` > ``orange_max``).

    The figure caps green at the orange limit (render_profiles), so there is
    no orange band — almost always the two limits were swapped. Covers the
    per-parameter map (workbook or Configure) and the sheet-wide pair.
    """
    warnings: list[str] = []

    def _check(name: str, green: float | None, orange: float | None) -> None:
        if green is None or orange is None or float(green) <= float(orange):
            return
        warnings.append(
            f"{name}: colour bands inverted (green_max {float(green):g} > orange_max "
            f"{float(orange):g}) — green is capped at {float(orange):g}, so there is no "
            "orange band; swap the limits"
        )

    for name, pair in (thresholds or {}).items():
        green, orange = (tuple(pair) + (None, None))[:2]
        _check(str(name), green, orange)
    _check("All parameters", green_max, orange_max)
    return warnings


# Parameters / units that are legitimately negative (redox potential,
# temperature); everything else on the Environmental sheet is a concentration
# or a non-negative field reading.
_SIGNED_PARAMETER_RE = re.compile(r"\b(orp|eh|redox|temp|temperature)\b", re.IGNORECASE)
_SIGNED_UNIT_RE = re.compile(r"^\s*(m?v|°\s*[cf]?|deg\s*[cf]?|[cf])\s*$", re.IGNORECASE)


def negative_concentration_warnings(readings: Sequence[EnvironmentalReading]) -> list[str]:
    """Negative values of parameters that cannot be negative (one line per parameter)."""
    by_parameter: dict[str, list[EnvironmentalReading]] = {}
    for reading in readings:
        if reading.value >= 0:
            continue
        if _SIGNED_PARAMETER_RE.search(reading.parameter) or _SIGNED_UNIT_RE.match(reading.unit or ""):
            continue
        by_parameter.setdefault(reading.parameter, []).append(reading)
    warnings: list[str] = []
    for parameter in sorted(by_parameter):
        items = by_parameter[parameter]
        shown = ", ".join(
            f"{item.hole_id} {item.value:g}{f' {item.unit}' if item.unit else ''}" for item in items[:5]
        )
        more = f" and {len(items) - 5} more" if len(items) > 5 else ""
        warnings.append(
            f"{parameter}: negative value(s) ({shown}{more}) — a concentration can't be "
            "negative; check the sign (a non-detect is written <RDL, e.g. <0.5)"
        )
    return warnings


def summarize_screen_intervals(
    collars: Sequence[Collar],
    screen_intervals: Sequence[ScreenInterval],
    hole_ids_on_transect: Sequence[str] | None = None,
) -> ScreenQualitySummary:
    """Warn when screens extend past total depth or overlap on the same hole."""
    collar_by_id = {collar.hole_id: collar for collar in collars}
    hole_set = set(hole_ids_on_transect) if hole_ids_on_transect is not None else None
    warnings: list[str] = []
    by_hole: dict[str, list[ScreenInterval]] = {}
    for interval in screen_intervals:
        if hole_set is not None and interval.hole_id not in hole_set:
            continue
        collar = collar_by_id.get(interval.hole_id)
        implausible = (
            implausible_screen_interval(interval, total_depth=collar.total_depth)
            if collar is not None
            else None
        )
        if implausible is not None:
            # Left off the figure: not part of the overlap check either.
            warnings.append(implausible)
            continue
        by_hole.setdefault(interval.hole_id, []).append(interval)
        if collar is not None and interval.to_depth > collar.total_depth + 1e-9:
            warnings.append(
                f"{interval.hole_id}: screen to_depth {interval.to_depth:.2f} m exceeds total depth "
                f"{collar.total_depth:.2f} m"
            )
    for hole_id, intervals in by_hole.items():
        ordered = sorted(intervals, key=lambda item: (item.from_depth, item.to_depth))
        for left, right in zip(ordered, ordered[1:]):
            if right.from_depth < left.to_depth - 1e-9:
                warnings.append(
                    f"{hole_id}: overlapping screens "
                    f"{left.from_depth:.2f}-{left.to_depth:.2f} m and "
                    f"{right.from_depth:.2f}-{right.to_depth:.2f} m"
                )
    return ScreenQualitySummary(
        warnings=tuple(warnings),
        total_intervals=sum(len(items) for items in by_hole.values()),
    )


def summarize_water_levels(
    collars: Sequence[Collar],
    water_levels: Sequence[WaterLevel],
    hole_ids_on_transect: Sequence[str],
    *,
    placeholder_elevation_m: float | None = None,
    deviation_readings: Sequence[DeviationReading] | None = None,
    screen_intervals: Sequence[ScreenInterval] | None = None,
    lithologies: Sequence[Lithology] | None = None,
    vertical_gradients: Sequence[VerticalGradient] | None = None,
    nest_distance_m: float = DEFAULT_NEST_DISTANCE_M,
    synoptic_window_days: int = DEFAULT_SYNOPTIC_WINDOW_DAYS,
    transect_points: Sequence[tuple[float, float]] | None = None,
) -> WaterQualitySummary:
    """Summarize groundwater coverage for Validate UI and QA heuristics.

    Horizontal-gradient QA uses the along-section distance of the collars
    projected onto ``transect_points`` (as the figure does); without a section
    line it falls back to the plan distance between consecutive collars and
    says so.

    Depth checks use the along-hole depth below ground (``btoc`` readings
    minus the collar stick-up). Gradient QA heads come from
    ``hydro_metrics.water_head_masl`` — with ``deviation_readings`` surveyed
    holes use true vertical depth, as on the section.

    Heads above ground are flowing-artesian *information*, not errors. With
    ``screen_intervals`` it also reports nested-well vertical gradients
    (``hydro_metrics.nested_vertical_gradients``), readings below the screens,
    dry screens and — with ``lithologies`` — connected water lines joining
    wells screened in different units; ``vertical_gradients`` (user arrows)
    are cross-checked against the computed heads.

    Monitoring dates: a series / nest whose dated readings span more than
    ``synoptic_window_days`` is warned (not one potentiometric surface); a
    dated + undated mix is information. Duplicate readings of one hole ×
    series × nest use the latest date (``hydro_metrics.prefer_water_reading``).
    """
    transect_holes = tuple(hole_ids_on_transect)
    hole_set = set(transect_holes)
    collar_by_id = {collar.hole_id: collar for collar in collars}
    warnings: list[str] = []
    by_series: dict[str, list[WaterLevel]] = {}
    # Deduplicate per hole × series × connect_group nest (blank nest = shared polyline).
    readings_by_key: dict[tuple[str, str, str], list[WaterLevel]] = {}
    holes_with_any: set[str] = set()
    orphan_ids: set[str] = set()
    masl_on_placeholder = False
    btoc_without_stick_up: set[str] = set()
    for level in water_levels:
        series_id = level.series_id or "default"
        connect_group = (level.connect_group or "").strip()
        key = (level.hole_id, series_id, connect_group)
        readings_by_key.setdefault(key, []).append(level)
        if level.hole_id not in hole_set:
            orphan_ids.add(level.hole_id)
            continue
        collar = collar_by_id.get(level.hole_id)
        implausible = (
            implausible_water_reading(
                level,
                collar_rl=collar.elevation,
                total_depth=collar.total_depth,
                stick_up_m=collar.stick_up_m,
            )
            if collar is not None
            else None
        )
        if implausible is not None:
            # The figure leaves it out; so do the gradient / artesian checks.
            warnings.append(implausible)
            continue
        holes_with_any.add(level.hole_id)
        by_series.setdefault(series_id, []).append(level)
        if collar is None:
            continue
        is_btoc = getattr(level, "depth_reference", "bgs") == "btoc"
        if is_btoc and level.elevation_masl is None and not collar.stick_up_m:
            btoc_without_stick_up.add(level.hole_id)
        depth_bgs = water_depth_along_hole_bgs(level, collar.stick_up_m)
        if is_measured_water_level(level) and depth_bgs > collar.total_depth:
            datum = f" ({level.depth:.2f} m below top of casing)" if is_btoc else ""
            warnings.append(
                f"{level.hole_id} ({series_id}): water depth {depth_bgs:.2f} m below ground{datum} "
                f"exceeds total depth {collar.total_depth:.2f} m"
            )
        if (
            not masl_on_placeholder
            and placeholder_elevation_m is not None
            and level.elevation_masl is not None
            and abs(collar.elevation - placeholder_elevation_m) <= 0.01
        ):
            masl_on_placeholder = True

    warnings[0:0] = _duplicate_water_reading_warnings(readings_by_key)
    # Dates of the readings actually used (one per hole × series × nest):
    # superseded duplicates are reported by the duplicate warning above.
    synoptic_warnings, synoptic_info = synoptic_date_warnings(
        latest_water_readings(water_levels), window_days=synoptic_window_days
    )
    warnings.extend(synoptic_warnings)
    if btoc_without_stick_up:
        warnings.append(
            "Water depths below top of casing (depth_reference = btoc) on "
            + ", ".join(sorted(btoc_without_stick_up))
            + " but Collars.stick_up_m is blank — stick-up taken as 0 m (heads may be "
            "too low); enter the casing stick-up."
        )
    if masl_on_placeholder:
        warnings.append(
            "Water elevation_masl was converted using placeholder collar RL — "
            "switch to relative (depth) mode or set true collar elevations."
        )

    holes_without_any = tuple(hole_id for hole_id in transect_holes if hole_id not in holes_with_any)
    series_summaries: list[WaterSeriesSummary] = []
    for series_id in sorted(by_series):
        levels = by_series[series_id]
        measured_holes = {level.hole_id for level in levels}
        label = next(
            (level.series_label for level in levels if level.series_label),
            series_id,
        )
        missing = tuple(hole_id for hole_id in transect_holes if hole_id not in measured_holes)
        series_summaries.append(
            WaterSeriesSummary(
                series_id=series_id,
                series_label=label,
                hole_count=len(measured_holes),
                missing_hole_ids=missing,
            )
        )
    if transect_holes and not by_series:
        warnings.append("No groundwater readings on the selected section line.")
    if orphan_ids:
        warnings.append(
            "Groundwater readings for holes not on the selected section line: "
            + ", ".join(sorted(orphan_ids))
        )

    # Schematic |Δh/Δx| QA on the same Δx as the figure: collars projected onto
    # the section line (along-section x_profile). Without a section line the
    # plan distance between consecutive collars is used, and a note says so.
    collar_xy = {c.hole_id: (float(c.easting), float(c.northing)) for c in collars}
    x_along: dict[str, float] = {}
    gradient_order: tuple[str, ...] = transect_holes
    along_section = transect_points is not None and len(transect_points) >= 2
    if along_section:
        transect = Transect(points=[(float(x), float(y)) for x, y in transect_points])  # type: ignore[union-attr]
        for hole_id in transect_holes:
            xy = collar_xy.get(hole_id)
            if xy is None:
                continue
            x_along[hole_id], _offset = project_collar_to_transect(xy[0], xy[1], transect)
        # Left-to-right as drawn (stable for equal x).
        gradient_order = tuple(sorted(x_along, key=lambda hid: x_along[hid]))
    else:
        cumulative = 0.0
        prev_xy: tuple[float, float] | None = None
        for hole_id in transect_holes:
            xy = collar_xy.get(hole_id)
            if xy is None:
                continue
            if prev_xy is not None:
                cumulative += (
                    (xy[0] - prev_xy[0]) ** 2 + (xy[1] - prev_xy[1]) ** 2
                ) ** 0.5
            x_along[hole_id] = cumulative
            prev_xy = xy
    plan_distance_gradients = False
    collar_rl = {c.hole_id: float(c.elevation) for c in collars}
    stick_up_by_hole = {c.hole_id: float(c.stick_up_m) for c in collars if c.stick_up_m}
    tvd_tables = build_along_hole_tvd(collars, deviation_readings, hole_set)
    md_to_elevation_by_hole = {
        hole_id: (lambda md, _t=table, _rl=collar_rl[hole_id]: _t.elevation_at(_rl, md))
        for hole_id, table in tvd_tables.items()
        if hole_id in collar_rl
    }
    for series_id, levels in by_series.items():
        label = next((level.series_label for level in levels if level.series_label), series_id)
        by_nest: dict[str, list[WaterLevel]] = {}
        for level in levels:
            by_nest.setdefault((level.connect_group or "").strip(), []).append(level)
        for nest_id, nest_levels in by_nest.items():
            series_label = label or series_id
            if nest_id:
                series_label = f"{series_label} [{nest_id}]"
            segments = horizontal_gradients_along_profile(
                nest_levels,
                hole_order=gradient_order,
                x_by_hole=x_along,
                collar_rl_by_hole=collar_rl,
                series_id=series_id,
                connect_group=nest_id,
                stick_up_by_hole=stick_up_by_hole,
                md_to_elevation_by_hole=md_to_elevation_by_hole,
            )
            absurd = absurd_horizontal_gradient_warnings(segments, series_label=series_label)
            warnings.extend(absurd)
            if absurd and not along_section:
                plan_distance_gradients = True
    if plan_distance_gradients:
        warnings.append(
            "Horizontal gradients above use the plan distance between consecutive collars "
            "(no section line drawn yet) — the figure uses the along-section distance, "
            "which is shorter for holes off the line."
        )

    on_section = [level for levels in by_series.values() for level in levels]
    # Readings the section actually uses (latest per hole × series × nest):
    # superseded readings must not raise artesian / screen / LNAPL findings.
    used_on_section = list(latest_water_readings(on_section))
    notes: list[str] = list(synoptic_info)
    notes.extend(
        _artesian_notes(used_on_section, collar_by_id, stick_up_by_hole, md_to_elevation_by_hole)
    )
    lnapl_warnings, lnapl_notes = _lnapl_checks(used_on_section)
    warnings.extend(lnapl_warnings)
    notes.extend(lnapl_notes)
    nested: tuple[NestedVerticalGradient, ...] = ()
    screens = [s for s in (screen_intervals or ()) if s.hole_id in hole_set]
    if screens:
        screen_warnings, screen_notes = _water_vs_screen_checks(
            used_on_section,
            screens,
            collar_by_id,
            stick_up_by_hole,
            md_to_elevation_by_hole,
        )
        warnings.extend(screen_warnings)
        notes.extend(screen_notes)
        if lithologies:
            warnings.extend(
                _mixed_unit_water_line_warnings(
                    used_on_section, screens, lithologies, transect_holes
                )
            )
        analysis = nested_vertical_gradients(
            on_section,
            screens,
            collar_rl_by_hole=collar_rl,
            plan_xy_by_hole=collar_xy,
            stick_up_by_hole=stick_up_by_hole,
            md_to_elevation_by_hole=md_to_elevation_by_hole,
            nest_distance_m=nest_distance_m,
        )
        nested = analysis.gradients
        warnings.extend(analysis.notes)
        notes.extend(format_vertical_gradient_note(gradient) for gradient in nested)
        if vertical_gradients:
            warnings.extend(
                vertical_gradient_arrow_conflicts(
                    [arrow for arrow in vertical_gradients if arrow.hole_id in hole_set],
                    nested,
                )
            )
    warnings.extend(f"Info: {note}" for note in notes)

    return WaterQualitySummary(
        series=tuple(series_summaries),
        holes_without_any_water=holes_without_any,
        warnings=tuple(warnings),
        total_levels=sum(len(levels) for levels in by_series.values()),
        vertical_gradients=nested,
        notes=tuple(notes),
    )


def _duplicate_water_reading_warnings(
    readings_by_key: Mapping[tuple[str, str, str], Sequence[WaterLevel]],
) -> list[str]:
    """Duplicate readings per hole × series × nest and which one the section uses."""
    warnings: list[str] = []
    for (hole_id, series_id, connect_group), group in readings_by_key.items():
        if len(group) < 2:
            continue
        nest_note = f", nest {connect_group}" if connect_group else ""
        where = f"{hole_id} ({series_id}{nest_note})"
        dates = sorted({d for d in (reading_date(level) for level in group) if d is not None})
        if not dates:
            warnings.extend(
                f"{where}: duplicate groundwater reading — last value is used on the section"
                for _ in group[1:]
            )
            continue
        chosen_date = reading_date(latest_water_by_hole(group)[hole_id])
        assert chosen_date is not None
        if len(dates) == 1 and all(reading_date(level) is not None for level in group):
            warnings.append(
                f"{where}: duplicate groundwater reading — {len(group)} readings on "
                f"{chosen_date.isoformat()}; last value is used on the section"
            )
            continue
        listed = ", ".join(d.isoformat() for d in dates)
        undated = sum(1 for level in group if reading_date(level) is None)
        undated_note = f" (+{undated} undated)" if undated else ""
        warnings.append(
            f"{where}: duplicate groundwater reading — {len(group)} readings dated "
            f"{listed}{undated_note}; the latest ({chosen_date.isoformat()}) is used on the "
            "section, gradients and flow fit (by date, not row order)"
        )
    return warnings


def _head_and_md_fn(
    level: WaterLevel,
    collar: Collar,
    stick_up_by_hole: Mapping[str, float],
    md_to_elevation_by_hole: Mapping[str, object],
):
    md_fn = md_to_elevation_by_hole.get(level.hole_id)
    head = water_head_masl(
        level,
        float(collar.elevation),
        stick_up_m=stick_up_by_hole.get(level.hole_id),
        md_to_elevation=md_fn,
    )
    return head, md_fn


def _lnapl_checks(levels: Sequence[WaterLevel]) -> tuple[list[str], list[str]]:
    """LNAPL (free product) Validate checks: ``(warnings, notes)``.

    * info: the wells whose heads are corrected (thickness × SG);
    * warning: apparent thickness > 1 m — the thickness in a well overstates
      the mobile product thickness in the formation;
    * warning: product recorded on a dry / NM reading (no head is plotted, so
      the product is not shown or corrected).
    """
    warnings: list[str] = []
    notes: list[str] = []
    note = lnapl_correction_note(levels)
    if note:
        notes.append(note + " (head = water level + apparent thickness × SG).")
    thick: list[str] = []
    unmeasured: list[str] = []
    for level in levels:
        if not has_lnapl(level):
            continue
        if not is_measured_water_level(level):
            entry = f"{level.hole_id} ({level.status})"
            if entry not in unmeasured:
                unmeasured.append(entry)
            continue
        thickness = lnapl_thickness_m(level)
        if thickness > APPARENT_LNAPL_THICKNESS_WARN_M:
            entry = f"{level.hole_id} ({thickness:.2f} m)"
            if entry not in thick:
                thick.append(entry)
    if thick:
        warnings.append(
            "LNAPL apparent thickness over "
            f"{APPARENT_LNAPL_THICKNESS_WARN_M:g} m at "
            + ", ".join(thick)
            + " — apparent thickness in wells overstates formation thickness; the head "
            "correction uses it as measured (check the reading and the product density)."
        )
    if unmeasured:
        warnings.append(
            "LNAPL thickness recorded on dry / not-measured readings at "
            + ", ".join(unmeasured)
            + " — no head is plotted there, so the product is not shown or corrected; "
            "check the status or the product columns."
        )
    return warnings, notes


def _artesian_notes(
    levels: Sequence[WaterLevel],
    collar_by_id: Mapping[str, Collar],
    stick_up_by_hole: Mapping[str, float],
    md_to_elevation_by_hole: Mapping[str, object],
) -> list[str]:
    """Heads above ground: flowing artesian (above top of casing) or artesian in the casing."""
    notes: list[str] = []
    seen: set[str] = set()
    for level in levels:
        collar = collar_by_id.get(level.hole_id)
        if collar is None or not is_measured_water_level(level):
            continue
        head, _md_fn = _head_and_md_fn(level, collar, stick_up_by_hole, md_to_elevation_by_hole)
        above = head - float(collar.elevation)
        if above <= 1e-6:
            continue
        stick_up = float(stick_up_by_hole.get(level.hole_id) or 0.0)
        series = "" if (level.series_id or "default") == "default" else f" ({level.series_id})"
        if stick_up > 0.0 and above <= stick_up + 1e-6:
            text = (
                f"Artesian at {level.hole_id}{series} (head +{above:.2f} m above ground, "
                "within the casing stick-up) — plotted above the ground line"
            )
        else:
            text = (
                f"Flowing artesian at {level.hole_id}{series} (head +{above:.2f} m above "
                "ground) — plotted above the ground line"
            )
        if text not in seen:
            seen.add(text)
            notes.append(text)
    return notes


def _water_vs_screen_checks(
    levels: Sequence[WaterLevel],
    screens: Sequence[ScreenInterval],
    collar_by_id: Mapping[str, Collar],
    stick_up_by_hole: Mapping[str, float],
    md_to_elevation_by_hole: Mapping[str, object],
) -> tuple[list[str], list[str]]:
    """Reading below every screen (warning) and individual dry screens (info)."""
    warnings: list[str] = []
    notes: list[str] = []
    screens_by_hole: dict[str, list[ScreenInterval]] = {}
    for screen in screens:
        screens_by_hole.setdefault(screen.hole_id, []).append(screen)
    top_head_by_hole: dict[str, float] = {}
    for level in levels:
        collar = collar_by_id.get(level.hole_id)
        hole_screens = screens_by_hole.get(level.hole_id)
        if collar is None or not hole_screens or not is_measured_water_level(level):
            continue
        head, md_fn = _head_and_md_fn(level, collar, stick_up_by_hole, md_to_elevation_by_hole)
        ranges = [screen_elevation_range(s, float(collar.elevation), md_fn) for s in hole_screens]
        lowest_bottom = min(bottom for _top, bottom in ranges)
        if head < lowest_bottom - 1e-6:
            series = "" if (level.series_id or "default") == "default" else f" ({level.series_id})"
            warnings.append(
                f"{level.hole_id}{series}: water level {head:.2f} masl is below the bottom of the "
                f"screen(s) ({lowest_bottom:.2f} masl) — reading below screen — well may be dry "
                "or reading unreliable"
            )
            continue
        top_head_by_hole[level.hole_id] = max(head, top_head_by_hole.get(level.hole_id, head))
    for hole_id, head in top_head_by_hole.items():
        collar = collar_by_id[hole_id]
        md_fn = md_to_elevation_by_hole.get(hole_id)
        for screen in sorted(screens_by_hole[hole_id], key=lambda s: s.from_depth):
            _top, bottom = screen_elevation_range(screen, float(collar.elevation), md_fn)
            if bottom > head + 1e-6:
                notes.append(
                    f"{hole_id}: screen {screen.from_depth:.2f}–{screen.to_depth:.2f} m is "
                    f"entirely above the water level ({head:.2f} masl) — dry screen"
                )
    return warnings, notes


def _logged_unit_at_depth(
    intervals: Sequence[Lithology], depth: float
) -> Lithology | None:
    for interval in intervals:
        if interval.from_depth - 1e-9 <= depth < interval.to_depth + 1e-9:
            return interval
    return None


def _unit_label(interval: Lithology) -> str:
    code = str(interval.lithology_code).strip()
    return f"{code} (unit {interval.unit_order})" if interval.unit_order is not None else code


def _same_unit(left: Lithology, right: Lithology) -> bool:
    if left.unit_order is not None and right.unit_order is not None:
        return left.unit_order == right.unit_order
    return str(left.lithology_code).strip().upper() == str(right.lithology_code).strip().upper()


def _mixed_unit_water_line_warnings(
    levels: Sequence[WaterLevel],
    screens: Sequence[ScreenInterval],
    lithologies: Sequence[Lithology],
    transect_holes: Sequence[str],
) -> list[str]:
    """Connected water lines (same series + connect_group) joining wells screened in different units.

    Conservative: both wells need a screen whose mid-point (along-hole depth)
    falls in a logged interval; a hole whose screens sit in several units is
    resolved by its connect_group (shallow/deep naming) or skipped. Units
    compare by ``unit_order`` when both rows carry one, else by lithology code.
    """
    screens_by_hole: dict[str, list[ScreenInterval]] = {}
    for screen in screens:
        screens_by_hole.setdefault(screen.hole_id, []).append(screen)
    lith_by_hole: dict[str, list[Lithology]] = {}
    for interval in lithologies:
        lith_by_hole.setdefault(interval.hole_id, []).append(interval)
    groups_by_hole: dict[str, set[str]] = {}
    for level in levels:
        groups_by_hole.setdefault(level.hole_id, set()).add((level.connect_group or "").strip())

    def unit_for(hole_id: str, group: str) -> Lithology | None:
        hole_screens = screens_by_hole.get(hole_id)
        intervals = lith_by_hole.get(hole_id)
        if not hole_screens or not intervals:
            return None
        mapping, _note = assign_screens_to_connect_groups(
            hole_id, hole_screens, sorted(groups_by_hole.get(hole_id, {group}))
        )
        candidates = [mapping[group]] if group in mapping else list(hole_screens)
        units = [
            _logged_unit_at_depth(intervals, 0.5 * (s.from_depth + s.to_depth)) for s in candidates
        ]
        if not units or any(unit is None for unit in units):
            return None
        first = units[0]
        if any(not _same_unit(first, unit) for unit in units[1:]):
            return None  # screens in several units, nothing tells which one reads
        return first

    order = {hole_id: index for index, hole_id in enumerate(transect_holes)}
    lines: dict[tuple[str, str], dict[str, WaterLevel]] = {}
    for level in levels:
        if not is_measured_water_level(level) or level.hole_id not in order:
            continue
        key = (level.series_id or "default", (level.connect_group or "").strip())
        lines.setdefault(key, {})[level.hole_id] = level
    warnings: list[str] = []
    for (series_id, group), by_hole in sorted(lines.items()):
        ordered = sorted(by_hole, key=lambda hole_id: order[hole_id])
        for left_id, right_id in zip(ordered, ordered[1:], strict=False):
            left_unit = unit_for(left_id, group)
            right_unit = unit_for(right_id, group)
            if left_unit is None or right_unit is None or _same_unit(left_unit, right_unit):
                continue
            nest = f" [{group}]" if group else ""
            series = "" if series_id == "default" else f" ({series_id}{nest})"
            if not series and nest:
                series = f" ({nest.strip()})"
            warnings.append(
                f"{left_id}–{right_id}{series}: water line joins wells screened in different "
                f"units ({_unit_label(left_unit)} vs {_unit_label(right_unit)}) — heads may "
                "belong to different aquifers; set connect_group to separate them"
            )
    return warnings


@dataclass(frozen=True)
class ParameterSummary:
    parameter: str
    hole_count: int
    missing_hole_ids: tuple[str, ...]
    min_depth: float
    max_depth: float


@dataclass(frozen=True)
class EnvironmentalQualitySummary:
    parameters: tuple[ParameterSummary, ...]
    holes_without_any_readings: tuple[str, ...]
    orphan_hole_ids: tuple[str, ...]
    warnings: tuple[str, ...]
    total_readings: int


def summarize_environmental_readings(
    collars: Sequence[Collar],
    readings: Sequence[EnvironmentalReading],
    hole_ids_on_transect: Sequence[str],
    *,
    thresholds: Mapping[str, tuple[float | None, float | None]] | None = None,
) -> EnvironmentalQualitySummary:
    """Summarize lab/parameter coverage for Validate UI.

    Also warns about non-detects whose detection limit is above the
    parameter's green limit (``thresholds``; default: the workbook
    ``green_max`` values) and parameters reported in more than one unit,
    negative concentrations, inverted colour bands (green above orange) and
    samples far below their hole (left off the figure).
    """
    transect_holes = tuple(hole_ids_on_transect)
    hole_set = set(transect_holes)
    collar_by_id = {collar.hole_id: collar for collar in collars}
    warnings: list[str] = []
    orphan_hole_ids: set[str] = set()
    by_parameter: dict[str, list[EnvironmentalReading]] = {}

    for reading in readings:
        if reading.hole_id not in hole_set:
            orphan_hole_ids.add(reading.hole_id)
            continue
        collar = collar_by_id.get(reading.hole_id)
        implausible = (
            implausible_environmental_reading(reading, total_depth=collar.total_depth)
            if collar is not None
            else None
        )
        if implausible is not None:
            warnings.append(implausible)
            continue
        by_parameter.setdefault(reading.parameter, []).append(reading)
        if collar is not None and reading.sample_depth > collar.total_depth:
            warnings.append(
                f"{reading.hole_id} ({reading.parameter}): sample depth {reading.sample_depth:.2f} m "
                f"exceeds total depth {collar.total_depth:.2f} m"
            )

    parameter_summaries: list[ParameterSummary] = []
    for parameter in sorted(by_parameter):
        items = by_parameter[parameter]
        measured_holes = {item.hole_id for item in items}
        depths = [item.sample_depth for item in items]
        missing = tuple(hole_id for hole_id in transect_holes if hole_id not in measured_holes)
        parameter_summaries.append(
            ParameterSummary(
                parameter=parameter,
                hole_count=len(measured_holes),
                missing_hole_ids=missing,
                min_depth=min(depths),
                max_depth=max(depths),
            )
        )

    on_line = [item for items in by_parameter.values() for item in items]
    # Several dated samples at one hole/parameter/depth: the figure draws the latest.
    latest_on_line, date_notes = latest_environmental_readings(on_line)
    warnings.extend(f"Info: {note}" for note in date_notes)
    warnings.extend(mixed_unit_warnings(on_line))
    warnings.extend(negative_concentration_warnings(on_line))
    if thresholds is None:
        thresholds = chemistry_thresholds_from_readings(readings)
    warnings.extend(chemistry_threshold_inversion_warnings(thresholds))
    # Only the samples the figure draws (superseded dates are not plotted).
    warnings.extend(rdl_above_guideline_notes(latest_on_line, thresholds))
    warnings.extend(lab_qualifier_warnings(latest_on_line))

    if not readings and transect_holes:
        warnings.append("No environmental/lab readings on the selected section line.")
    holes_with_any = {reading.hole_id for reading in readings if reading.hole_id in hole_set}
    holes_without_any = tuple(hole_id for hole_id in transect_holes if hole_id not in holes_with_any)
    return EnvironmentalQualitySummary(
        parameters=tuple(parameter_summaries),
        holes_without_any_readings=holes_without_any,
        orphan_hole_ids=tuple(sorted(orphan_hole_ids)),
        warnings=tuple(warnings),
        total_readings=sum(len(items) for items in by_parameter.values()),
    )
