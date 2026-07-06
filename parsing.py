"""Native Excel workbook parsing (DataParser)."""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from models import (
    COLLAR_COLUMNS,
    GRADIENT_COLUMNS,
    LITHOLOGY_COLUMNS,
    LITHOLOGY_OPTIONAL_COLUMNS,
    SCREEN_COLUMNS,
    WATER_COLUMNS,
    WATER_OPTIONAL_COLUMNS,
    Collar,
    CorrelationOverride,
    DeviationReading,
    EnvironmentalReading,
    Fault,
    Lithology,
    ParseResult,
    ScreenInterval,
    Unconformity,
    VerticalGradient,
    WaterLevel,
)

logger = logging.getLogger(__name__)

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {col: str(col).strip().lower().replace(" ", "_") for col in df.columns}
    return df.rename(columns=renamed)


class DataParser:
    """Reads Excel workbooks and validates rows against borehole schemas."""

    COLLARS_SHEET = "Collars"
    LITHOLOGY_SHEET = "Lithology"
    WATER_SHEET = "Water"
    SCREENS_SHEET = "Screens"
    GRADIENTS_SHEET = "Gradients"

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
        deviation_readings: list[DeviationReading] = []
        correlation_overrides: list[CorrelationOverride] = []
        environmental_readings: list[EnvironmentalReading] = []
        faults: list[Fault] = []
        unconformities: list[Unconformity] = []
        screen_intervals: list[ScreenInterval] = []
        vertical_gradients: list[VerticalGradient] = []
        if water_df is not None:
            water_levels, water_errors = self._parse_water_levels(
                _normalize_columns(water_df), collars
            )
        elif workbook is not None:
            water_sheet = self._find_sheet(workbook.sheet_names, self.WATER_SHEET)
            if water_sheet:
                water_frame = _normalize_columns(pd.read_excel(workbook, sheet_name=water_sheet))
                water_levels, water_errors = self._parse_water_levels(water_frame, collars)
            deviation_readings = self._parse_deviation_sheet(workbook)
            correlation_overrides = self._parse_correlation_sheet(workbook)
            environmental_readings = self._parse_environmental_sheet(workbook)
            faults = self._parse_fault_sheet(workbook)
            unconformities = self._parse_unconformity_sheet(workbook)
            screen_intervals = self._parse_screens_sheet(workbook, collars)
            vertical_gradients = self._parse_gradients_sheet(workbook, collars)

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
            screen_intervals=tuple(screen_intervals),
            vertical_gradients=tuple(vertical_gradients),
            deviation_readings=tuple(deviation_readings),
            correlation_overrides=tuple(correlation_overrides),
            faults=tuple(faults),
            unconformities=tuple(unconformities),
            environmental_readings=tuple(environmental_readings),
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
        optional_cols = {"elevation_datum", "inclination_deg", "azimuth_deg"} & set(df.columns)

        for index, row in enumerate(df.itertuples(index=True)):
            row_num = int(row.Index) + 2
            try:
                payload: dict[str, object] = {
                    "hole_id": row.hole_id,
                    "easting": row.easting,
                    "northing": row.northing,
                    "elevation": row.elevation,
                    "total_depth": row.total_depth,
                }
                if "elevation_datum" in optional_cols:
                    payload["elevation_datum"] = row.elevation_datum
                if "inclination_deg" in optional_cols:
                    payload["inclination_deg"] = row.inclination_deg
                if "azimuth_deg" in optional_cols:
                    payload["azimuth_deg"] = row.azimuth_deg
                collar = Collar.model_validate(payload)
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
                payload: dict[str, object] = {"hole_id": row.hole_id, "depth": row.depth}
                for col in WATER_OPTIONAL_COLUMNS:
                    if hasattr(row, col):
                        payload[col] = getattr(row, col)
                level = WaterLevel.model_validate(payload)
            except Exception as exc:
                errors.append(f"Water row {row_num}: {exc}")
                continue
            if valid_hole_ids and level.hole_id not in valid_hole_ids:
                errors.append(f"Water row {row_num}: unknown hole_id '{level.hole_id}'")
                continue
            levels.append(level)

        return levels, errors

    def _parse_screens_sheet(
        self,
        workbook: pd.ExcelFile,
        collars: list[Collar],
    ) -> list[ScreenInterval]:
        sheet = self._find_sheet(workbook.sheet_names, self.SCREENS_SHEET)
        if not sheet:
            return []
        frame = _normalize_columns(pd.read_excel(workbook, sheet_name=sheet))
        if not SCREEN_COLUMNS.issubset(frame.columns):
            return []
        valid_hole_ids = {collar.hole_id for collar in collars}
        intervals: list[ScreenInterval] = []
        for row in frame.itertuples(index=False):
            try:
                interval = ScreenInterval.model_validate(
                    {
                        "hole_id": row.hole_id,
                        "from_depth": row.from_depth,
                        "to_depth": row.to_depth,
                    }
                )
            except Exception as exc:
                logger.warning("Skipping screen interval row: %s", exc)
                continue
            if valid_hole_ids and interval.hole_id not in valid_hole_ids:
                logger.warning("Skipping screen interval for unknown hole_id '%s'", interval.hole_id)
                continue
            intervals.append(interval)
        return intervals

    def _parse_gradients_sheet(
        self,
        workbook: pd.ExcelFile,
        collars: list[Collar],
    ) -> list[VerticalGradient]:
        sheet = self._find_sheet(workbook.sheet_names, self.GRADIENTS_SHEET)
        if not sheet:
            return []
        frame = _normalize_columns(pd.read_excel(workbook, sheet_name=sheet))
        if "hole_id" not in frame.columns:
            return []
        valid_hole_ids = {collar.hole_id for collar in collars}
        gradients: list[VerticalGradient] = []
        for row in frame.itertuples(index=False):
            payload: dict[str, object] = {"hole_id": row.hole_id}
            if hasattr(row, "direction"):
                payload["direction"] = row.direction
            try:
                gradient = VerticalGradient.model_validate(payload)
            except Exception as exc:
                logger.warning("Skipping gradient row: %s", exc)
                continue
            if valid_hole_ids and gradient.hole_id not in valid_hole_ids:
                logger.warning("Skipping gradient for unknown hole_id '%s'", gradient.hole_id)
                continue
            gradients.append(gradient)
        return gradients

    def _parse_deviation_sheet(self, workbook: pd.ExcelFile) -> list[DeviationReading]:
        sheet = self._find_sheet(workbook.sheet_names, "Deviations")
        if not sheet:
            return []
        frame = _normalize_columns(pd.read_excel(workbook, sheet_name=sheet))
        required = {"hole_id", "depth", "inclination_deg", "azimuth_deg"}
        if not required.issubset(frame.columns):
            return []
        readings: list[DeviationReading] = []
        for row in frame.itertuples(index=False):
            try:
                readings.append(
                    DeviationReading.model_validate(
                        {
                            "hole_id": row.hole_id,
                            "depth": row.depth,
                            "inclination_deg": row.inclination_deg,
                            "azimuth_deg": row.azimuth_deg,
                        }
                    )
                )
            except Exception as exc:
                logger.warning("Skipping deviation row: %s", exc)
        return readings

    def _parse_correlation_sheet(self, workbook: pd.ExcelFile) -> list[CorrelationOverride]:
        sheet = self._find_sheet(workbook.sheet_names, "Correlations")
        if not sheet:
            return []
        frame = _normalize_columns(pd.read_excel(workbook, sheet_name=sheet))
        required = {"left_hole_id", "right_hole_id", "left_unit_order", "right_unit_order"}
        if not required.issubset(frame.columns):
            return []
        overrides: list[CorrelationOverride] = []
        for row in frame.itertuples(index=False):
            try:
                overrides.append(CorrelationOverride.model_validate(row._asdict()))
            except Exception as exc:
                logger.warning("Skipping correlation override row: %s", exc)
        return overrides

    def _parse_environmental_sheet(self, workbook: pd.ExcelFile) -> list[EnvironmentalReading]:
        sheet = self._find_sheet(workbook.sheet_names, "Environmental")
        if not sheet:
            return []
        frame = _normalize_columns(pd.read_excel(workbook, sheet_name=sheet))
        required = {"hole_id", "from_depth", "to_depth", "parameter", "value"}
        if not required.issubset(frame.columns):
            return []
        readings: list[EnvironmentalReading] = []
        for row in frame.itertuples(index=False):
            payload = row._asdict()
            if "unit" not in payload:
                payload["unit"] = ""
            try:
                readings.append(EnvironmentalReading.model_validate(payload))
            except Exception as exc:
                logger.warning("Skipping environmental row: %s", exc)
        return readings

    def _parse_fault_sheet(self, workbook: pd.ExcelFile) -> list[Fault]:
        sheet = self._find_sheet(workbook.sheet_names, "Faults")
        if not sheet:
            return []
        frame = _normalize_columns(pd.read_excel(workbook, sheet_name=sheet))
        if not {"name", "x_profile", "elevation"}.issubset(frame.columns):
            return []
        faults: dict[str, list[tuple[float, float]]] = {}
        for row in frame.itertuples(index=False):
            faults.setdefault(str(row.name), []).append((float(row.x_profile), float(row.elevation)))
        return [
            Fault(name=name, trace_points=points)
            for name, points in faults.items()
            if len(points) >= 2
        ]

    def _parse_unconformity_sheet(self, workbook: pd.ExcelFile) -> list[Unconformity]:
        sheet = self._find_sheet(workbook.sheet_names, "Unconformities")
        if not sheet:
            return []
        frame = _normalize_columns(pd.read_excel(workbook, sheet_name=sheet))
        if not {"name", "x_profile", "elevation"}.issubset(frame.columns):
            return []
        surfaces: dict[str, list[tuple[float, float]]] = {}
        for row in frame.itertuples(index=False):
            surfaces.setdefault(str(row.name), []).append(
                (float(row.x_profile), float(row.elevation))
            )
        return [
            Unconformity(name=name, elevation_profile=points)
            for name, points in surfaces.items()
            if len(points) >= 2
        ]
