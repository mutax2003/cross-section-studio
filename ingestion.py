"""Config-driven multi-format workbook ingestion."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Literal, Sequence

import pandas as pd
from pydantic import BaseModel, Field

from ai_quality import (
    MappingProposal,
    QualityReport,
    analyze_parsed_data,
    load_lithology_aliases,
    normalize_lithology_code,
    propose_workbook_mapping,
    read_mapped_sheets,
)
from constants import DEFAULT_PROFILE_ELEVATION_M
from models import (
    COLLAR_COLUMNS,
    LITHOLOGY_COLUMNS,
    DataParser,
    ParseResult,
    assign_missing_unit_orders,
    geology_sheet_counts,
    lithology_has_unit_order_column,
)

logger = logging.getLogger(__name__)

from paths import import_profiles_dir

PROFILES_DIR = import_profiles_dir()
OVERRIDES_DIR = PROFILES_DIR / "overrides"
NATIVE_PROFILE_ID = "native_platform"

DEPTH_INTERVAL_PATTERN = re.compile(r"([\d.]+)\s*-\s*([\d.]+)")


class DetectRules(BaseModel):
    required_sheets: list[str] = Field(default_factory=list)
    optional_sheets: list[str] = Field(default_factory=list)
    required_columns: dict[str, list[str]] = Field(default_factory=dict)


class CoordinateRules(BaseModel):
    mode: Literal["wgs84_to_utm", "already_projected"] = "wgs84_to_utm"
    source_crs: str = "EPSG:4326"
    target_crs: str = "EPSG:32611"


class ProfileDefaults(BaseModel):
    elevation_m: float = 100.0


class ImportProfile(BaseModel):
    id: str
    label: str
    detect: DetectRules = Field(default_factory=DetectRules)
    lithology_sheet: str = "Lithology"
    columns: dict[str, str] = Field(default_factory=dict)
    depth_format: Literal["interval_string", "from_to_columns", "numeric_pair"] = "interval_string"
    coordinates: CoordinateRules = Field(default_factory=CoordinateRules)
    defaults: ProfileDefaults = Field(default_factory=ProfileDefaults)
    extends: str | None = None
    coordinate_offsets_m: dict[str, list[float]] = Field(default_factory=dict)


@dataclass(frozen=True)
class DetectionResult:
    profile_id: str
    label: str
    confidence: float
    is_native: bool = False


@dataclass
class ImportReport:
    profile_id: str
    profile_label: str
    detection_confidence: float
    hole_count: int = 0
    lithology_interval_count: int = 0
    normalized_lithology_count: int = 0
    coordinate_offsets_applied: dict[str, tuple[float, float]] = field(default_factory=dict)
    optional_sheets_detected: list[str] = field(default_factory=list)
    geology_sheet_counts: dict[str, int] = field(default_factory=dict)
    lithology_has_unit_order_column: bool = False
    unit_order_auto_assigned: bool = False
    warnings: list[str] = field(default_factory=list)
    mapping_proposal: MappingProposal | None = None
    quality_report: QualityReport | None = None
    uses_placeholder_elevation: bool = False
    suggested_utm_crs: str | None = None
    profile_default_elevation_m: float | None = None


def _as_workbook(source: str | Path | BinaryIO | BytesIO | pd.ExcelFile) -> pd.ExcelFile:
    if isinstance(source, pd.ExcelFile):
        return source
    if hasattr(source, "seek"):
        source.seek(0)
    return pd.ExcelFile(source)


def _normalize_header(value: object) -> str:
    return re.sub(r"\s+", "_", str(value).strip().lower())


def _sheet_columns(workbook: pd.ExcelFile, sheet_name: str) -> list[str]:
    return [str(column) for column in pd.read_excel(workbook, sheet_name=sheet_name, nrows=0).columns]


def _resolve_column_name(columns: list[str], expected: str) -> str | None:
    normalized = {_normalize_header(column): column for column in columns}
    expected_key = _normalize_header(expected)
    if expected_key in normalized:
        return normalized[expected_key]
    for column in columns:
        if _normalize_header(column) == expected_key:
            return column
    return None


def load_profile(profile_id: str) -> ImportProfile:
    if profile_id == NATIVE_PROFILE_ID:
        return _native_profile()
    return _load_profile_from_disk(profile_id)


@lru_cache(maxsize=16)
def _load_profile_from_disk(profile_id: str) -> ImportProfile:
    path = PROFILES_DIR / f"{profile_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Import profile not found: {profile_id}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("extends"):
        base = load_profile(data["extends"]).model_dump()
        base.update({key: value for key, value in data.items() if key != "extends"})
        return ImportProfile.model_validate(base)
    return ImportProfile.model_validate(data)


def _native_profile() -> ImportProfile:
    return ImportProfile(
        id=NATIVE_PROFILE_ID,
        label="Native platform (Collars + Lithology)",
        detect=DetectRules(
            required_sheets=["Collars", "Lithology"],
            required_columns={
                "Collars": sorted(COLLAR_COLUMNS),
                "Lithology": sorted(LITHOLOGY_COLUMNS),
            },
        ),
    )


def load_override(override_id: str | None) -> ImportProfile | None:
    if not override_id:
        return None
    path = OVERRIDES_DIR / f"{override_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Import override not found: {override_id}")
    data = json.loads(path.read_text(encoding="utf-8"))
    base_id = data.get("extends")
    if not base_id:
        raise ValueError(f"Override {override_id} must specify 'extends'")
    base = load_profile(base_id).model_dump()
    base.update({key: value for key, value in data.items() if key != "extends"})
    base["id"] = override_id
    return ImportProfile.model_validate(base)


@lru_cache(maxsize=1)
def list_profiles() -> tuple[ImportProfile, ...]:
    profiles = [_load_profile_from_disk(path.stem) for path in PROFILES_DIR.glob("*.json")]
    return tuple(sorted(profiles, key=lambda item: item.label))


class DepthParser:
    def __init__(self, depth_format: str, columns: dict[str, str]) -> None:
        self.depth_format = depth_format
        self.columns = columns

    def parse_row(self, row: pd.Series) -> tuple[float, float]:
        if self.depth_format == "interval_string":
            column = self.columns.get("depth_interval", "Depth")
            return parse_depth_interval(row[column])
        if self.depth_format == "from_to_columns":
            from_col = self.columns.get("from_depth", "from_depth")
            to_col = self.columns.get("to_depth", "to_depth")
            return float(row[from_col]), float(row[to_col])
        from_col = self.columns.get("from_depth", "from_depth")
        to_col = self.columns.get("to_depth", "to_depth")
        return float(row[from_col]), float(row[to_col])


def parse_depth_interval(depth_value: object) -> tuple[float, float]:
    if depth_value is None or (isinstance(depth_value, float) and pd.isna(depth_value)):
        raise ValueError("missing depth interval")
    text = str(depth_value).strip().lower().replace("m", "")
    match = DEPTH_INTERVAL_PATTERN.match(text)
    if not match:
        raise ValueError(f"cannot parse depth interval: {depth_value!r}")
    return float(match.group(1)), float(match.group(2))


class CoordinateTransformer:
    _TRANSFORMERS: dict[tuple[str, str], object] = {}

    def __init__(self, rules: CoordinateRules) -> None:
        self.rules = rules

    def transform(self, latitude: float, longitude: float) -> tuple[float, float]:
        if self.rules.mode == "already_projected":
            return float(longitude), float(latitude)
        try:
            from pyproj import Transformer

            key = (self.rules.source_crs, self.rules.target_crs)
            transformer = self._TRANSFORMERS.get(key)
            if transformer is None:
                transformer = Transformer.from_crs(
                    self.rules.source_crs,
                    self.rules.target_crs,
                    always_xy=True,
                )
                self._TRANSFORMERS[key] = transformer
            easting, northing = transformer.transform(longitude, latitude)
            return float(easting), float(northing)
        except ImportError as exc:
            raise RuntimeError("pyproj is required: pip install pyproj") from exc


class FormatDetector:
    def __init__(self) -> None:
        self._column_cache: dict[tuple[int, str], list[str]] = {}

    def _cached_sheet_columns(self, workbook: pd.ExcelFile, sheet_name: str) -> list[str]:
        cache_key = (id(workbook), sheet_name)
        if cache_key not in self._column_cache:
            self._column_cache[cache_key] = _sheet_columns(workbook, sheet_name)
        return self._column_cache[cache_key]

    def detect(self, source: str | Path | BinaryIO | BytesIO | pd.ExcelFile) -> DetectionResult:
        workbook = _as_workbook(source)
        sheet_names = workbook.sheet_names
        sheet_names_lower = {_normalize_header(name): name for name in sheet_names}

        if "collars" in sheet_names_lower and "lithology" in sheet_names_lower:
            collars_sheet = sheet_names_lower["collars"]
            lithology_sheet = sheet_names_lower["lithology"]
            collar_cols = self._cached_sheet_columns(workbook, collars_sheet)
            lith_cols = self._cached_sheet_columns(workbook, lithology_sheet)
            collar_hits = sum(1 for col in COLLAR_COLUMNS if _resolve_column_name(collar_cols, col))
            lith_hits = sum(1 for col in LITHOLOGY_COLUMNS if _resolve_column_name(lith_cols, col))
            confidence = (collar_hits / len(COLLAR_COLUMNS) + lith_hits / len(LITHOLOGY_COLUMNS)) / 2
            if confidence >= 0.8:
                return DetectionResult(
                    profile_id=NATIVE_PROFILE_ID,
                    label="Native platform (Collars + Lithology)",
                    confidence=confidence,
                    is_native=True,
                )

        best: DetectionResult | None = None
        for profile in list_profiles():
            score = self._score_profile(workbook, profile)
            if best is None or score > best.confidence:
                best = DetectionResult(
                    profile_id=profile.id,
                    label=profile.label,
                    confidence=score,
                    is_native=False,
                )

        if best is None or best.confidence < 0.5:
            sheets_info = ", ".join(sheet_names)
            raise ValueError(
                f"Could not detect a supported workbook format. Sheets found: {sheets_info}"
            )
        return best

    def _score_profile(self, workbook: pd.ExcelFile, profile: ImportProfile) -> float:
        sheet_names = {_normalize_header(name): name for name in workbook.sheet_names}
        scores: list[float] = []

        for required_sheet in profile.detect.required_sheets:
            key = _normalize_header(required_sheet)
            if key not in sheet_names:
                return 0.0
            actual_sheet = sheet_names[key]
            columns = self._cached_sheet_columns(workbook, actual_sheet)
            required_cols = profile.detect.required_columns.get(required_sheet, [])
            if not required_cols:
                required_cols = list(profile.columns.values())
            if not required_cols:
                scores.append(1.0)
                continue
            hits = sum(1 for col in required_cols if _resolve_column_name(columns, col))
            scores.append(hits / len(required_cols))

        return sum(scores) / len(scores) if scores else 0.0


def suggest_utm_crs(latitudes: Sequence[float], longitudes: Sequence[float]) -> str | None:
    """Suggest a WGS84 UTM EPSG code from mean lat/long."""
    if not latitudes or not longitudes:
        return None
    mean_lat = float(sum(latitudes) / len(latitudes))
    mean_lon = float(sum(longitudes) / len(longitudes))
    zone = int((mean_lon + 180.0) / 6.0) + 1
    zone = max(1, min(60, zone))
    if mean_lat >= 0.0:
        return f"EPSG:{32600 + zone}"
    return f"EPSG:{32700 + zone}"


def _read_field_data_collar_elevations(workbook: pd.ExcelFile) -> dict[str, float]:
    """Read collar RL per hole from Field Data sheet when available."""
    sheet_lookup = {_normalize_header(name): name for name in workbook.sheet_names}
    field_key = sheet_lookup.get("field_data")
    if field_key is None:
        return {}
    frame = pd.read_excel(workbook, sheet_name=field_key)
    if frame.empty:
        return {}
    columns = {_normalize_header(col): col for col in frame.columns}
    hole_col = None
    for candidate in ("label", "hole_id", "hole", "bh_id"):
        if candidate in columns:
            hole_col = columns[candidate]
            break
    elev_col = None
    for candidate in ("elevation", "rl", "collar_rl", "reduced_level", "collar_elevation"):
        if candidate in columns:
            elev_col = columns[candidate]
            break
    if hole_col is None or elev_col is None:
        return {}
    frame = frame[[hole_col, elev_col]].dropna(subset=[hole_col, elev_col])
    if frame.empty:
        return {}
    frame = frame.copy()
    frame["_hole_id"] = frame[hole_col].astype(str).str.strip()
    frame = frame[frame["_hole_id"].ne("") & frame["_hole_id"].str.lower().ne("nan")]
    frame["_rl"] = pd.to_numeric(frame[elev_col], errors="coerce")
    frame = frame.dropna(subset=["_rl"])
    if frame.empty:
        return {}
    return frame.groupby("_hole_id", as_index=True)["_rl"].first().astype(float).to_dict()


def _read_field_data_total_depths(workbook: pd.ExcelFile) -> dict[str, float]:
    """Read measured total depth per hole from Field Data sheet when available."""
    sheet_lookup = {_normalize_header(name): name for name in workbook.sheet_names}
    field_key = sheet_lookup.get("field_data")
    if field_key is None:
        return {}
    frame = pd.read_excel(workbook, sheet_name=field_key)
    if frame.empty:
        return {}
    columns = {_normalize_header(col): col for col in frame.columns}
    hole_col = None
    for candidate in ("label", "hole_id", "hole", "bh_id"):
        if candidate in columns:
            hole_col = columns[candidate]
            break
    td_col = None
    for candidate in ("total_depth", "td", "max_depth", "depth"):
        if candidate in columns:
            td_col = columns[candidate]
            break
    if hole_col is None or td_col is None:
        return {}
    frame = frame[[hole_col, td_col]].dropna(subset=[hole_col, td_col])
    if frame.empty:
        return {}
    frame = frame.copy()
    frame["_hole_id"] = frame[hole_col].astype(str).str.strip()
    frame = frame[frame["_hole_id"].ne("") & frame["_hole_id"].str.lower().ne("nan")]
    frame["_td"] = pd.to_numeric(frame[td_col], errors="coerce")
    frame = frame.dropna(subset=["_td"])
    if frame.empty:
        return {}
    return frame.groupby("_hole_id", as_index=True)["_td"].max().astype(float).to_dict()


class FieldExportAdapter:
    def adapt(
        self,
        source: str | Path | BinaryIO | BytesIO | pd.ExcelFile,
        profile: ImportProfile,
        *,
        elevation_m: float | None = None,
        target_crs: str | None = None,
        workbook: pd.ExcelFile | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        active_workbook = workbook or _as_workbook(source)
        sheet_key = _normalize_header(profile.lithology_sheet)
        sheet_lookup = {_normalize_header(name): name for name in active_workbook.sheet_names}
        if sheet_key not in sheet_lookup:
            raise ValueError(f"Lithology sheet '{profile.lithology_sheet}' not found")
        lithology_raw = pd.read_excel(active_workbook, sheet_name=sheet_lookup[sheet_key])

        column_map = profile.columns.copy()
        resolved_map: dict[str, str] = {}
        available = list(lithology_raw.columns)
        for key, expected in column_map.items():
            resolved = _resolve_column_name(available, expected)
            if resolved is None:
                raise ValueError(f"Required column '{expected}' not found on lithology sheet")
            resolved_map[key] = resolved

        coord_rules = profile.coordinates.model_copy()
        if target_crs:
            coord_rules.target_crs = target_crs
        transformer = CoordinateTransformer(coord_rules)
        aliases = load_lithology_aliases()
        elevation = elevation_m if elevation_m is not None else profile.defaults.elevation_m

        frame = lithology_raw.rename(
            columns={resolved_map[key]: key for key in resolved_map},
        )
        frame = frame.copy()
        frame["hole_id"] = frame["hole_id"].astype(str).str.strip()
        if profile.depth_format == "interval_string":
            depth_col = resolved_map.get("depth_interval", "depth_interval")
            if "depth_interval" not in frame.columns and depth_col in lithology_raw.columns:
                frame["depth_interval"] = lithology_raw[depth_col]
            parsed_depths = [parse_depth_interval(value) for value in frame["depth_interval"]]
            frame["from_depth"] = [item[0] for item in parsed_depths]
            frame["to_depth"] = [item[1] for item in parsed_depths]
        else:
            frame["from_depth"] = pd.to_numeric(frame["from_depth"], errors="raise")
            frame["to_depth"] = pd.to_numeric(frame["to_depth"], errors="raise")

        frame["lithology_code"] = frame["lithology_code"].astype(str).str.strip().map(
            lambda raw: normalize_lithology_code(raw, aliases),
        )

        coord_cache: dict[str, tuple[float, float]] = {}
        for hole_id, group in frame.groupby("hole_id", sort=False):
            row = group.iloc[0]
            if profile.coordinates.mode == "already_projected":
                easting = float(row["easting"])
                northing = float(row["northing"])
            else:
                latitude = float(row["latitude"])
                longitude = float(row["longitude"])
                easting, northing = transformer.transform(latitude, longitude)
            coord_cache[str(hole_id)] = (easting, northing)

        max_depth_by_hole = frame.groupby("hole_id", sort=False)["to_depth"].max()
        collar_accum = {
            str(hole_id): {
                "hole_id": str(hole_id),
                "easting": coord_cache[str(hole_id)][0],
                "northing": coord_cache[str(hole_id)][1],
                "elevation": elevation,
                "total_depth": float(max_depth_by_hole[hole_id]),
            }
            for hole_id in max_depth_by_hole.index
        }

        measured_td = _read_field_data_total_depths(active_workbook)
        measured_rl = _read_field_data_collar_elevations(active_workbook)
        for hole_id, rl in measured_rl.items():
            if hole_id in collar_accum:
                collar_accum[hole_id]["elevation"] = float(rl)
        for hole_id, td in measured_td.items():
            if hole_id in collar_accum:
                collar_accum[hole_id]["total_depth"] = max(
                    float(collar_accum[hole_id]["total_depth"]),
                    td,
                )

        collars_df = pd.DataFrame(collar_accum.values()).sort_values("hole_id")
        for hole_id, offset in profile.coordinate_offsets_m.items():
            if len(offset) != 2:
                continue
            de, dn = float(offset[0]), float(offset[1])
            mask = collars_df["hole_id"] == hole_id
            if mask.any():
                collars_df.loc[mask, "easting"] = collars_df.loc[mask, "easting"] + de
                collars_df.loc[mask, "northing"] = collars_df.loc[mask, "northing"] + dn

        lithology_df = frame[["hole_id", "from_depth", "to_depth", "lithology_code"]].sort_values(
            ["hole_id", "from_depth"],
        )
        collars_out = collars_df[["hole_id", "easting", "northing", "elevation", "total_depth"]].copy()
        return collars_out, lithology_df


def _native_adapt(
    source: str | Path | BinaryIO | BytesIO | pd.ExcelFile,
    mapping_proposal: MappingProposal | None = None,
    *,
    workbook: pd.ExcelFile | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, MappingProposal]:
    active_workbook = workbook or _as_workbook(source)
    proposal = mapping_proposal or propose_workbook_mapping(active_workbook)
    collars_df, lithology_df = read_mapped_sheets(
        source,
        proposal,
        workbook=active_workbook,
    )
    return collars_df, lithology_df, proposal


def _detect_optional_workbook_sheets(workbook: pd.ExcelFile) -> list[str]:
    known = {
        "water": "Water",
        "deviations": "Deviations",
        "correlations": "Correlations",
        "environmental": "Environmental",
        "faults": "Faults",
        "unconformities": "Unconformities",
        "field_data": "Field Data",
    }
    detected: list[str] = []
    for sheet_name in workbook.sheet_names:
        key = _normalize_header(sheet_name)
        if key in known:
            detected.append(known[key])
    return detected


def ingest_workbook(
    source: str | Path | BinaryIO | BytesIO,
    *,
    profile_id: str | None = None,
    override_id: str | None = None,
    elevation_m: float | None = None,
    target_crs: str | None = None,
    lithology_aliases: dict[str, str] | None = None,
    auto_assign_unit_order: bool = True,
) -> tuple[ParseResult, ImportReport]:
    """Single entry point: detect format, adapt, validate, return ParseResult."""
    workbook = _as_workbook(source)
    detection = FormatDetector().detect(workbook) if profile_id is None else None
    resolved_profile_id = profile_id or (detection.profile_id if detection else NATIVE_PROFILE_ID)
    confidence = detection.confidence if detection else 1.0

    aliases = lithology_aliases or load_lithology_aliases()
    warnings: list[str] = []
    optional_sheets = _detect_optional_workbook_sheets(workbook)
    mapping_proposal: MappingProposal | None = None
    offsets_applied: dict[str, tuple[float, float]] = {}
    suggested_utm_crs: str | None = None
    profile_default_elevation_m: float | None = None

    sheet_names_lower = {_normalize_header(name) for name in workbook.sheet_names}
    if "field_data" in sheet_names_lower or "field data" in workbook.sheet_names:
        if "Field Data" not in optional_sheets:
            optional_sheets.append("Field Data")
        if resolved_profile_id != NATIVE_PROFILE_ID:
            measured = _read_field_data_total_depths(workbook)
            measured_rl = _read_field_data_collar_elevations(workbook)
            if measured:
                warnings.append(
                    f"Field Data sheet: applied measured total depth for {len(measured)} hole(s)."
                )
            if measured_rl:
                warnings.append(
                    f"Field Data sheet: applied collar RL for {len(measured_rl)} hole(s)."
                )
            if not measured:
                warnings.append(
                    "Field Data sheet detected — no TD column mapped; total depth inferred from lithology."
                )
        else:
            warnings.append(
                "Field Data sheet detected — not used for stratigraphy (OVA overlay is future work)."
            )

    if resolved_profile_id == NATIVE_PROFILE_ID:
        collars_df, lithology_df, mapping_proposal = _native_adapt(
            source,
            workbook=workbook,
        )
        profile_label = "Native platform (Collars + Lithology)"
    else:
        profile = load_override(override_id) if override_id else load_profile(resolved_profile_id)
        resolved_profile_id = profile.id
        profile_label = profile.label
        profile_default_elevation_m = profile.defaults.elevation_m

        collars_df, lithology_df = FieldExportAdapter().adapt(
            source,
            profile,
            elevation_m=elevation_m,
            target_crs=target_crs,
            workbook=workbook,
        )
        for hole_id, offset in profile.coordinate_offsets_m.items():
            if len(offset) == 2:
                offsets_applied[hole_id] = (float(offset[0]), float(offset[1]))
        if elevation_m is None:
            warnings.append(
                f"Collar elevation uses profile default ({profile.defaults.elevation_m:.1f} m) — "
                "set sidebar elevation for absolute RL sections."
            )
        if profile.coordinates.mode == "wgs84_to_utm" and "latitude" in profile.columns:
            sheet_key = _normalize_header(profile.lithology_sheet)
            sheet_lookup = {_normalize_header(name): name for name in workbook.sheet_names}
            if sheet_key in sheet_lookup:
                lith_raw = pd.read_excel(workbook, sheet_name=sheet_lookup[sheet_key])
                lat_col = _resolve_column_name(list(lith_raw.columns), profile.columns.get("latitude", "Lat"))
                lon_col = _resolve_column_name(list(lith_raw.columns), profile.columns.get("longitude", "Long"))
                if lat_col and lon_col:
                    lats = pd.to_numeric(lith_raw[lat_col], errors="coerce").dropna()
                    lons = pd.to_numeric(lith_raw[lon_col], errors="coerce").dropna()
                    if not lats.empty and not lons.empty:
                        suggested_utm_crs = suggest_utm_crs(lats.tolist(), lons.tolist())
                        if suggested_utm_crs and target_crs is None:
                            warnings.append(f"Suggested target CRS from coordinates: {suggested_utm_crs}")

    placeholder_elevation = (
        profile_default_elevation_m
        if resolved_profile_id != NATIVE_PROFILE_ID
        else None
    )
    if elevation_m is not None:
        placeholder_elevation = None

    parse_result = DataParser().parse_file(
        source,
        collars_df=collars_df,
        lithology_df=lithology_df,
        lithology_aliases=aliases,
    )

    had_unit_order_column = lithology_has_unit_order_column(parse_result.lithologies)
    unit_order_auto_assigned = False
    if auto_assign_unit_order:
        new_lithologies, assign_messages = assign_missing_unit_orders(
            parse_result.lithologies,
            only_duplicate_holes=True,
        )
        if assign_messages:
            unit_order_auto_assigned = True
            warnings.extend(assign_messages)
            parse_result = ParseResult(
                collars=parse_result.collars,
                lithologies=new_lithologies,
                errors=parse_result.errors,
                water_levels=parse_result.water_levels,
                deviation_readings=parse_result.deviation_readings,
                correlation_overrides=parse_result.correlation_overrides,
                faults=parse_result.faults,
                unconformities=parse_result.unconformities,
                environmental_readings=parse_result.environmental_readings,
            )

    qa = analyze_parsed_data(
        parse_result.collars,
        parse_result.lithologies,
        mapping_proposal=mapping_proposal,
        aliases=aliases,
        placeholder_elevation_m=placeholder_elevation,
    )

    uses_placeholder = (
        placeholder_elevation is not None
        and bool(parse_result.collars)
        and all(abs(collar.elevation - placeholder_elevation) < 0.01 for collar in parse_result.collars)
    )

    report = ImportReport(
        profile_id=resolved_profile_id,
        profile_label=profile_label,
        detection_confidence=confidence,
        hole_count=len(parse_result.collars),
        lithology_interval_count=len(parse_result.lithologies),
        normalized_lithology_count=qa.normalized_lithology_count,
        coordinate_offsets_applied=offsets_applied,
        optional_sheets_detected=optional_sheets,
        geology_sheet_counts=geology_sheet_counts(parse_result),
        lithology_has_unit_order_column=had_unit_order_column or unit_order_auto_assigned,
        unit_order_auto_assigned=unit_order_auto_assigned,
        warnings=warnings,
        mapping_proposal=mapping_proposal,
        quality_report=qa,
        uses_placeholder_elevation=uses_placeholder,
        suggested_utm_crs=suggested_utm_crs,
        profile_default_elevation_m=profile_default_elevation_m,
    )
    return parse_result, report


def export_platform_workbook(
    source: str | Path | BinaryIO | BytesIO,
    output: Path,
    *,
    profile_id: str | None = None,
    override_id: str | None = None,
    elevation_m: float | None = None,
    target_crs: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert any supported workbook to canonical Collars/Lithology xlsx."""
    workbook = _as_workbook(source)
    detection = FormatDetector().detect(workbook) if profile_id is None else None
    resolved_profile_id = profile_id or detection.profile_id

    if resolved_profile_id == NATIVE_PROFILE_ID:
        collars_df, lithology_df, _ = _native_adapt(source, workbook=workbook)
    else:
        profile = load_override(override_id) if override_id else load_profile(resolved_profile_id)
        collars_df, lithology_df = FieldExportAdapter().adapt(
            source,
            profile,
            elevation_m=elevation_m,
            target_crs=target_crs,
            workbook=workbook,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        collars_df.to_excel(writer, sheet_name="Collars", index=False)
        lithology_df.to_excel(writer, sheet_name="Lithology", index=False)
    return collars_df, lithology_df
