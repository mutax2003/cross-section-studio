"""Immutable data schemas and Excel ingestion for borehole datasets."""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Sequence

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

COLLAR_COLUMNS = {"hole_id", "easting", "northing", "elevation", "total_depth"}
LITHOLOGY_COLUMNS = {"hole_id", "from_depth", "to_depth", "lithology_code"}
LITHOLOGY_OPTIONAL_COLUMNS = {"hatch_pattern", "unit_order"}
WATER_COLUMNS = {"hole_id", "depth"}


class Collar(BaseModel, frozen=True):
    hole_id: str
    easting: float
    northing: float
    elevation: float
    total_depth: float

    @field_validator("hole_id", mode="before")
    @classmethod
    def strip_hole_id(cls, value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            raise ValueError("hole_id is required")
        return str(value).strip()

    @field_validator("total_depth")
    @classmethod
    def validate_total_depth(cls, value: float) -> float:
        if value < 0:
            raise ValueError("total_depth must be non-negative")
        return value


class Lithology(BaseModel, frozen=True):
    hole_id: str
    from_depth: float
    to_depth: float
    lithology_code: str
    hatch_pattern: str | None = None
    unit_order: int | None = None

    @field_validator("hole_id", "lithology_code", mode="before")
    @classmethod
    def strip_strings(cls, value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            raise ValueError("required string field is missing")
        return str(value).strip()

    @field_validator("hatch_pattern", mode="before")
    @classmethod
    def optional_string(cls, value: object) -> str | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = str(value).strip()
        return text or None

    @field_validator("unit_order", mode="before")
    @classmethod
    def optional_unit_order(cls, value: object) -> int | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return int(value)

    @model_validator(mode="after")
    def validate_depths(self) -> Lithology:
        if self.from_depth < 0 or self.to_depth < 0:
            raise ValueError("depths must be non-negative")
        if self.to_depth < self.from_depth:
            raise ValueError("to_depth must be >= from_depth")
        return self


class WaterLevel(BaseModel, frozen=True):
    hole_id: str
    depth: float

    @field_validator("hole_id", mode="before")
    @classmethod
    def strip_hole_id(cls, value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            raise ValueError("hole_id is required")
        return str(value).strip()

    @field_validator("depth")
    @classmethod
    def validate_depth(cls, value: float) -> float:
        if value < 0:
            raise ValueError("depth must be non-negative")
        return value


class Transect(BaseModel, frozen=True):
    points: list[tuple[float, float]] = Field(min_length=2)

    @field_validator("points")
    @classmethod
    def validate_points(cls, value: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(value) < 2:
            raise ValueError("transect requires at least two points")
        return value


class ParseResult(BaseModel, frozen=True):
    collars: tuple[Collar, ...]
    lithologies: tuple[Lithology, ...]
    errors: tuple[str, ...]
    water_levels: tuple[WaterLevel, ...] = ()


def subset_parse_result(parse_result: ParseResult, hole_ids: Sequence[str]) -> ParseResult:
    """Return collars and lithologies limited to the given hole IDs (order preserved)."""
    if not hole_ids:
        return ParseResult(
            collars=(),
            lithologies=(),
            errors=parse_result.errors,
            water_levels=(),
        )
    hole_set = frozenset(hole_ids)
    return ParseResult(
        collars=tuple(collar for collar in parse_result.collars if collar.hole_id in hole_set),
        lithologies=tuple(
            lithology for lithology in parse_result.lithologies if lithology.hole_id in hole_set
        ),
        errors=parse_result.errors,
        water_levels=tuple(
            level for level in parse_result.water_levels if level.hole_id in hole_set
        ),
    )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {col: str(col).strip().lower().replace(" ", "_") for col in df.columns}
    return df.rename(columns=renamed)


class DataParser:
    """Reads Excel workbooks and validates rows against borehole schemas."""

    COLLARS_SHEET = "Collars"
    LITHOLOGY_SHEET = "Lithology"
    WATER_SHEET = "Water"

    def parse_file(
        self,
        source: str | Path | BinaryIO | BytesIO,
        *,
        collars_df: pd.DataFrame | None = None,
        lithology_df: pd.DataFrame | None = None,
        collars_sheet: str | None = None,
        lithology_sheet: str | None = None,
        lithology_aliases: dict[str, str] | None = None,
        water_df: pd.DataFrame | None = None,
    ) -> ParseResult:
        workbook: pd.ExcelFile | None = None
        if collars_df is not None and lithology_df is not None:
            collars_frame = _normalize_columns(collars_df)
            lithology_frame = _normalize_columns(lithology_df)
        else:
            workbook = pd.ExcelFile(source)
            collars_name = collars_sheet or self.COLLARS_SHEET
            lithology_name = lithology_sheet or self.LITHOLOGY_SHEET
            missing = [
                sheet
                for sheet in (collars_name, lithology_name)
                if sheet not in workbook.sheet_names
            ]
            if missing:
                raise ValueError(f"Missing required sheet(s): {', '.join(missing)}")

            collars_frame = _normalize_columns(pd.read_excel(workbook, sheet_name=collars_name))
            lithology_frame = _normalize_columns(pd.read_excel(workbook, sheet_name=lithology_name))

        collars, collar_errors = self._parse_collars(collars_frame)
        lithologies, lithology_errors = self._parse_lithologies(lithology_frame, collars)
        water_levels: list[WaterLevel] = []
        water_errors: list[str] = []
        if water_df is not None:
            water_levels, water_errors = self._parse_water_levels(
                _normalize_columns(water_df), collars
            )
        elif workbook is not None:
            water_sheet = self._find_sheet(workbook.sheet_names, self.WATER_SHEET)
            if water_sheet:
                water_frame = _normalize_columns(pd.read_excel(workbook, sheet_name=water_sheet))
                water_levels, water_errors = self._parse_water_levels(water_frame, collars)

        if lithology_aliases:
            lithologies = self._apply_lithology_aliases(lithologies, lithology_aliases)

        errors = tuple(collar_errors + lithology_errors + water_errors)
        if errors:
            for message in errors:
                logger.warning("Parse issue: %s", message)

        return ParseResult(
            collars=tuple(collars),
            lithologies=tuple(lithologies),
            errors=errors,
            water_levels=tuple(water_levels),
        )

    def _apply_lithology_aliases(
        self,
        lithologies: list[Lithology],
        aliases: dict[str, str],
    ) -> list[Lithology]:
        return [
            Lithology(
                hole_id=lithology.hole_id,
                from_depth=lithology.from_depth,
                to_depth=lithology.to_depth,
                lithology_code=aliases.get(
                    str(lithology.lithology_code).strip().lower().replace(" ", "_"),
                    lithology.lithology_code,
                ),
                hatch_pattern=lithology.hatch_pattern,
                unit_order=lithology.unit_order,
            )
            for lithology in lithologies
        ]

    @staticmethod
    def _find_sheet(sheet_names: list[str], target: str) -> str | None:
        target_key = target.strip().lower()
        for name in sheet_names:
            if name.strip().lower() == target_key:
                return name
        return None

    def _parse_collars(self, df: pd.DataFrame) -> tuple[list[Collar], list[str]]:
        errors: list[str] = []
        collars: list[Collar] = []
        seen_ids: set[str] = set()

        missing_cols = COLLAR_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(f"Collars sheet missing columns: {', '.join(sorted(missing_cols))}")

        for index, row in enumerate(df.itertuples(index=True)):
            row_num = int(row.Index) + 2
            try:
                collar = Collar.model_validate(
                    {
                        "hole_id": row.hole_id,
                        "easting": row.easting,
                        "northing": row.northing,
                        "elevation": row.elevation,
                        "total_depth": row.total_depth,
                    }
                )
            except Exception as exc:
                errors.append(f"Collars row {row_num}: {exc}")
                continue

            if collar.hole_id in seen_ids:
                errors.append(f"Collars row {row_num}: duplicate hole_id '{collar.hole_id}'")
                continue

            seen_ids.add(collar.hole_id)
            collars.append(collar)

        return collars, errors

    def _parse_lithologies(
        self,
        df: pd.DataFrame,
        collars: list[Collar],
    ) -> tuple[list[Lithology], list[str]]:
        errors: list[str] = []
        lithologies: list[Lithology] = []
        valid_hole_ids = {collar.hole_id for collar in collars}

        missing_cols = LITHOLOGY_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(f"Lithology sheet missing columns: {', '.join(sorted(missing_cols))}")

        optional_cols = LITHOLOGY_OPTIONAL_COLUMNS & set(df.columns)
        for index, row in enumerate(df.itertuples(index=True)):
            row_num = int(row.Index) + 2
            try:
                payload: dict[str, object] = {
                    "hole_id": row.hole_id,
                    "from_depth": row.from_depth,
                    "to_depth": row.to_depth,
                    "lithology_code": row.lithology_code,
                }
                if "hatch_pattern" in optional_cols:
                    payload["hatch_pattern"] = row.hatch_pattern
                if "unit_order" in optional_cols:
                    payload["unit_order"] = row.unit_order
                lithology = Lithology.model_validate(payload)
            except Exception as exc:
                errors.append(f"Lithology row {row_num}: {exc}")
                continue

            if valid_hole_ids and lithology.hole_id not in valid_hole_ids:
                errors.append(
                    f"Lithology row {row_num}: unknown hole_id '{lithology.hole_id}'"
                )
                continue

            lithologies.append(lithology)

        return lithologies, errors

    def _parse_water_levels(
        self,
        df: pd.DataFrame,
        collars: list[Collar],
    ) -> tuple[list[WaterLevel], list[str]]:
        errors: list[str] = []
        levels: list[WaterLevel] = []
        valid_hole_ids = {collar.hole_id for collar in collars}
        missing_cols = WATER_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(f"Water sheet missing columns: {', '.join(sorted(missing_cols))}")

        for index, row in enumerate(df.itertuples(index=True)):
            row_num = int(row.Index) + 2
            try:
                level = WaterLevel.model_validate(
                    {"hole_id": row.hole_id, "depth": row.depth}
                )
            except Exception as exc:
                errors.append(f"Water row {row_num}: {exc}")
                continue
            if valid_hole_ids and level.hole_id not in valid_hole_ids:
                errors.append(f"Water row {row_num}: unknown hole_id '{level.hole_id}'")
                continue
            levels.append(level)

        return levels, errors
