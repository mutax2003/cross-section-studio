"""Local data quality checks, column mapping, and lithology normalization."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Literal, Sequence

import pandas as pd
from rapidfuzz import fuzz

from constants import CANONICAL_LITHOLOGY_CODES
from models import (
    COLLAR_COLUMNS,
    LITHOLOGY_COLUMNS,
    Collar,
    EnvironmentalReading,
    Lithology,
    ScreenInterval,
    Transect,
    WaterLevel,
)
from projection import TransectGeometry

logger = logging.getLogger(__name__)

from paths import lithology_aliases_path
MAPPING_CONFIDENCE_THRESHOLD = 0.8

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


class Severity(str, Enum):
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


def _hole_quality_issues(collar: Collar, intervals: list[Lithology]) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    if not intervals:
        issues.append(
            QualityIssue(
                code="no_lithology",
                message=f"{collar.hole_id} has no lithology intervals",
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

    sorted_intervals = sorted(intervals, key=lambda item: item.from_depth)
    if sorted_intervals[0].from_depth > 0.01:
        issues.append(
            QualityIssue(
                code="depth_gap",
                message=(
                    f"{collar.hole_id} missing coverage from 0 to "
                    f"{sorted_intervals[0].from_depth:.2f} m"
                ),
                severity=Severity.WARNING.value,
                hole_id=collar.hole_id,
            )
        )

    for left, right in zip(sorted_intervals, sorted_intervals[1:]):
        if right.from_depth > left.to_depth + 0.01:
            issues.append(
                QualityIssue(
                    code="depth_gap",
                    message=(
                        f"{collar.hole_id} gap between {left.to_depth:.2f} and "
                        f"{right.from_depth:.2f} m"
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
                        f"{left.hole_id} intervals overlap between "
                        f"{left.from_depth:.2f}-{left.to_depth:.2f} m and "
                        f"{right.from_depth:.2f}-{right.to_depth:.2f} m"
                    ),
                    severity=Severity.ERROR.value,
                    hole_id=left.hole_id,
                )
            )

    last = sorted_intervals[-1]
    if last.to_depth < collar.total_depth - 0.01:
        issues.append(
            QualityIssue(
                code="depth_gap",
                message=(
                    f"{collar.hole_id} missing coverage from {last.to_depth:.2f} to "
                    f"{collar.total_depth:.2f} m (TD)"
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
                        f"{collar.hole_id} has duplicate lithology code(s) "
                        f"({', '.join(sorted(set(missing_order)))}) without unit_order — "
                        "correlation across holes will fail"
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
                    message=f"{collar.hole_id} has duplicate unit_order values",
                    severity=Severity.ERROR.value,
                    hole_id=collar.hole_id,
                )
            )

    return issues


def collars_use_placeholder_elevation(
    collars: Sequence[Collar],
    placeholder_m: float,
    *,
    tolerance: float = 0.01,
) -> bool:
    if not collars:
        return False
    return all(abs(collar.elevation - placeholder_m) <= tolerance for collar in collars)


def analyze_parsed_data(
    collars: Sequence[Collar],
    lithologies: Sequence[Lithology],
    *,
    transect: Transect | None = None,
    offset_threshold_m: float = 50.0,
    mapping_proposal: MappingProposal | None = None,
    aliases: dict[str, str] | None = None,
    placeholder_elevation_m: float | None = None,
) -> QualityReport:
    issues: list[QualityIssue] = []
    collar_by_id = {collar.hole_id: collar for collar in collars}
    collar_ids = set(collar_by_id)
    lithology_by_hole: dict[str, list[Lithology]] = {}

    for lithology in lithologies:
        lithology_by_hole.setdefault(lithology.hole_id, []).append(lithology)
        if lithology.hole_id not in collar_ids:
            issues.append(
                QualityIssue(
                    code="orphan_lithology",
                    message=f"Lithology references unknown hole_id '{lithology.hole_id}'",
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
                        f"{lithology.hole_id} interval exceeds total depth "
                        f"({lithology.to_depth:.2f} m > {collar.total_depth:.2f} m TD)"
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
                    message="All collars share identical easting/northing coordinates",
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
                            f"{hole_id} is {offset:.1f} m from transect "
                            f"(threshold {offset_threshold_m:.1f} m)"
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


@dataclass(frozen=True)
class ScreenQualitySummary:
    warnings: tuple[str, ...]
    total_intervals: int


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
        by_hole.setdefault(interval.hole_id, []).append(interval)
        collar = collar_by_id.get(interval.hole_id)
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
) -> WaterQualitySummary:
    """Summarize groundwater coverage for Validate UI and QA heuristics."""
    transect_holes = tuple(hole_ids_on_transect)
    hole_set = set(transect_holes)
    collar_by_id = {collar.hole_id: collar for collar in collars}
    warnings: list[str] = []
    by_series: dict[str, list[WaterLevel]] = {}
    for level in water_levels:
        if level.hole_id not in hole_set:
            continue
        series_id = level.series_id or "default"
        by_series.setdefault(series_id, []).append(level)
        collar = collar_by_id.get(level.hole_id)
        if collar is not None and level.depth > collar.total_depth:
            warnings.append(
                f"{level.hole_id} ({series_id}): water depth {level.depth:.2f} m exceeds total depth "
                f"{collar.total_depth:.2f} m"
            )

    holes_with_any = {level.hole_id for level in water_levels if level.hole_id in hole_set}
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
        warnings.append("No groundwater readings on the selected transect.")
    orphan_ids = sorted({level.hole_id for level in water_levels if level.hole_id not in hole_set})
    if orphan_ids:
        warnings.append(
            "Groundwater readings for holes not on the selected transect: " + ", ".join(orphan_ids)
        )
    return WaterQualitySummary(
        series=tuple(series_summaries),
        holes_without_any_water=holes_without_any,
        warnings=tuple(warnings),
        total_levels=sum(len(levels) for levels in by_series.values()),
    )


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
) -> EnvironmentalQualitySummary:
    """Summarize lab/parameter coverage for Validate UI."""
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
        by_parameter.setdefault(reading.parameter, []).append(reading)
        collar = collar_by_id.get(reading.hole_id)
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

    if not readings and transect_holes:
        warnings.append("No environmental/lab readings on the selected transect.")
    holes_with_any = {reading.hole_id for reading in readings if reading.hole_id in hole_set}
    holes_without_any = tuple(hole_id for hole_id in transect_holes if hole_id not in holes_with_any)
    return EnvironmentalQualitySummary(
        parameters=tuple(parameter_summaries),
        holes_without_any_readings=holes_without_any,
        orphan_hole_ids=tuple(sorted(orphan_hole_ids)),
        warnings=tuple(warnings),
        total_readings=sum(len(items) for items in by_parameter.values()),
    )
