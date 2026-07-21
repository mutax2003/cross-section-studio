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
    WATER_VALUE_COLUMNS,
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
        water_frame = None
        environmental_frame = None
        screens_frame = None
        gradients_frame = None
        if collars_df is not None and lithology_df is not None:
            collars_frame = _normalize_columns(collars_df)
            lithology_frame = _normalize_columns(lithology_df)
            # Still open the workbook so optional Water / Environmental / Screens load.
            try:
                workbook = pd.ExcelFile(source)
            except Exception:
                workbook = None
        else:
            workbook = pd.ExcelFile(source)
            data_entry_sheet = self._find_sheet(workbook.sheet_names, "Data Entry")
            collars_name = collars_sheet or self.COLLARS_SHEET
            lithology_name = lithology_sheet or self.LITHOLOGY_SHEET
            has_native = (
                self._find_sheet(workbook.sheet_names, collars_name) is not None
                and self._find_sheet(workbook.sheet_names, lithology_name) is not None
            )
            data_entry = None
            if data_entry_sheet:
                from workbook_template import parse_data_entry_sheet

                entry_frame = pd.read_excel(workbook, sheet_name=data_entry_sheet, header=None)
                data_entry = parse_data_entry_sheet(entry_frame)
                if not data_entry.water.empty:
                    water_frame = _normalize_columns(data_entry.water)
                if not data_entry.environmental.empty:
                    environmental_frame = _normalize_columns(data_entry.environmental)
                if not data_entry.screens.empty:
                    screens_frame = _normalize_columns(data_entry.screens)
                if not data_entry.gradients.empty:
                    gradients_frame = _normalize_columns(data_entry.gradients)

            if data_entry is not None and not has_native:
                collars_frame = _normalize_columns(data_entry.collars)
                lithology_frame = _normalize_columns(data_entry.lithology)
            else:
                missing = [
                    sheet
                    for sheet in (collars_name, lithology_name)
                    if self._find_sheet(workbook.sheet_names, sheet) is None
                ]
                if missing:
                    raise ValueError(f"Missing required sheet(s): {', '.join(missing)}")

                collars_frame = _normalize_columns(pd.read_excel(workbook, sheet_name=collars_name))
                lithology_frame = _normalize_columns(pd.read_excel(workbook, sheet_name=lithology_name))
                # Empty header-only native tabs: fall back to Data Entry geology when present.
                if (
                    data_entry is not None
                    and not data_entry.collars.empty
                    and not data_entry.lithology.empty
                    and not self._frame_has_hole_rows(collars_frame)
                ):
                    collars_frame = _normalize_columns(data_entry.collars)
                    lithology_frame = _normalize_columns(data_entry.lithology)

        collars, collar_errors = self._parse_collars(collars_frame)
        lithologies, lithology_errors = self._parse_lithologies(lithology_frame, collars)
        water_levels: list[WaterLevel] = []
        water_errors: list[str] = []
        deviation_readings: list[DeviationReading] = []
        correlation_overrides: list[CorrelationOverride] = []
        environmental_readings: list[EnvironmentalReading] = []
        environmental_errors: list[str] = []
        faults: list[Fault] = []
        unconformities: list[Unconformity] = []
        screen_intervals: list[ScreenInterval] = []
        vertical_gradients: list[VerticalGradient] = []
        screen_errors: list[str] = []
        gradient_errors: list[str] = []
        if water_df is not None:
            water_levels, water_errors = self._parse_water_levels(
                _normalize_columns(water_df), collars
            )
        elif workbook is not None:
            if water_frame is None:
                water_sheet = self._find_sheet(workbook.sheet_names, self.WATER_SHEET)
                if water_sheet:
                    water_frame = _normalize_columns(pd.read_excel(workbook, sheet_name=water_sheet))
            if water_frame is not None and not water_frame.empty:
                water_levels, water_errors = self._parse_water_levels(water_frame, collars)
            if environmental_frame is not None and not environmental_frame.empty:
                environmental_readings, environmental_errors = self._parse_environmental_dataframe(
                    environmental_frame, collars
                )
            else:
                environmental_readings, environmental_errors = self._parse_environmental_sheet(
                    workbook, collars
                )
            if screens_frame is not None and not screens_frame.empty:
                screen_intervals, screen_errors = self._parse_screens_dataframe(screens_frame, collars)
            else:
                screen_intervals, screen_errors = self._parse_screens_sheet(workbook, collars)
            if gradients_frame is not None and not gradients_frame.empty:
                vertical_gradients, gradient_errors = self._parse_gradients_dataframe(
                    gradients_frame, collars
                )
            else:
                vertical_gradients, gradient_errors = self._parse_gradients_sheet(workbook, collars)
            deviation_readings = self._parse_deviation_sheet(workbook)
            correlation_overrides = self._parse_correlation_sheet(workbook)
            faults = self._parse_fault_sheet(workbook)
            unconformities = self._parse_unconformity_sheet(workbook)

        if lithology_aliases:
            lithologies = self._apply_lithology_aliases(lithologies, lithology_aliases)

        errors = tuple(
            collar_errors
            + lithology_errors
            + water_errors
            + screen_errors
            + gradient_errors
            + environmental_errors
        )
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
        if not aliases:
            return lithologies
        from ai_quality import normalize_lithology_code

        updated: list[Lithology] = []
        for lithology in lithologies:
            new_code = normalize_lithology_code(lithology.lithology_code, aliases)
            if new_code == lithology.lithology_code:
                updated.append(lithology)
            else:
                updated.append(
                    Lithology(
                        hole_id=lithology.hole_id,
                        from_depth=lithology.from_depth,
                        to_depth=lithology.to_depth,
                        lithology_code=new_code,
                        hatch_pattern=lithology.hatch_pattern,
                        unit_order=lithology.unit_order,
                    )
                )
        return updated

    @staticmethod
    def _frame_has_hole_rows(frame: pd.DataFrame) -> bool:
        if frame is None or frame.empty or "hole_id" not in frame.columns:
            return False
        for value in frame["hole_id"]:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            text = str(value).strip()
            if text and text.lower() != "nan":
                return True
        return False

    @staticmethod
    def _find_sheet(sheet_names: list[str], target: str) -> str | None:
        target_key = target.strip().lower()
        for name in sheet_names:
            if name.strip().lower() == target_key:
                return name
        return None

    @staticmethod
    def _blank_hole_id(value: object) -> bool:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return True
        text = str(value).strip()
        return text == "" or text.lower() == "nan"

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
            if self._blank_hole_id(row.hole_id):
                continue
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
            if self._blank_hole_id(row.hole_id):
                continue
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
        collar_by_id = {collar.hole_id: collar for collar in collars}
        columns = set(df.columns)
        if "hole_id" not in columns:
            raise ValueError("Water sheet missing columns: hole_id")
        if not columns.intersection(WATER_VALUE_COLUMNS):
            raise ValueError("Water sheet requires hole_id plus depth or elevation_masl")

        for index, row in enumerate(df.itertuples(index=True)):
            row_num = int(row.Index) + 2
            if self._blank_hole_id(row.hole_id):
                continue
            try:
                hole_id = str(row.hole_id).strip()
                depth_raw = getattr(row, "depth", None)
                masl_raw = getattr(row, "elevation_masl", None)
                has_depth = depth_raw is not None and not pd.isna(depth_raw) and str(depth_raw).strip() != ""
                has_masl = masl_raw is not None and not pd.isna(masl_raw) and str(masl_raw).strip() != ""
                if has_depth and has_masl:
                    raise ValueError("provide depth or elevation_masl, not both")
                if not has_depth and not has_masl:
                    raise ValueError("depth or elevation_masl is required")
                elevation_masl: float | None = None
                if has_masl:
                    elevation_masl = float(masl_raw)
                    collar = collar_by_id.get(hole_id)
                    if collar is None:
                        raise ValueError(f"unknown hole_id '{hole_id}' for elevation_masl")
                    if elevation_masl > collar.elevation:
                        raise ValueError(
                            f"elevation_masl {elevation_masl} is above collar elevation "
                            f"{collar.elevation} (artesian / above-collar water level)"
                        )
                    depth = collar.elevation - elevation_masl
                else:
                    depth = float(depth_raw)
                payload: dict[str, object] = {
                    "hole_id": hole_id,
                    "depth": depth,
                    "elevation_masl": elevation_masl,
                }
                for col in ("series_id", "series_label", "color", "marker"):
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
    ) -> tuple[list[ScreenInterval], list[str]]:
        sheet = self._find_sheet(workbook.sheet_names, self.SCREENS_SHEET)
        if not sheet:
            return [], []
        frame = _normalize_columns(pd.read_excel(workbook, sheet_name=sheet))
        return self._parse_screens_dataframe(frame, collars)

    def _parse_gradients_sheet(
        self,
        workbook: pd.ExcelFile,
        collars: list[Collar],
    ) -> tuple[list[VerticalGradient], list[str]]:
        sheet = self._find_sheet(workbook.sheet_names, self.GRADIENTS_SHEET)
        if not sheet:
            return [], []
        frame = _normalize_columns(pd.read_excel(workbook, sheet_name=sheet))
        return self._parse_gradients_dataframe(frame, collars)

    def _parse_screens_dataframe(
        self,
        frame: pd.DataFrame,
        collars: list[Collar],
    ) -> tuple[list[ScreenInterval], list[str]]:
        if not SCREEN_COLUMNS.issubset(frame.columns):
            return [], []
        valid_hole_ids = {collar.hole_id for collar in collars}
        intervals: list[ScreenInterval] = []
        errors: list[str] = []
        for row_num, row in enumerate(frame.itertuples(index=False), start=2):
            if self._blank_hole_id(getattr(row, "hole_id", None)):
                continue
            try:
                interval = ScreenInterval.model_validate(row._asdict())
            except Exception as exc:
                errors.append(f"Screens row {row_num}: {exc}")
                continue
            if valid_hole_ids and interval.hole_id not in valid_hole_ids:
                errors.append(f"Screens row {row_num}: unknown hole_id '{interval.hole_id}'")
                continue
            intervals.append(interval)
        return intervals, errors

    def _parse_gradients_dataframe(
        self,
        frame: pd.DataFrame,
        collars: list[Collar],
    ) -> tuple[list[VerticalGradient], list[str]]:
        if not GRADIENT_COLUMNS.issubset(frame.columns):
            return [], []
        valid_hole_ids = {collar.hole_id for collar in collars}
        gradients: list[VerticalGradient] = []
        errors: list[str] = []
        for row_num, row in enumerate(frame.itertuples(index=False), start=2):
            if self._blank_hole_id(getattr(row, "hole_id", None)):
                continue
            try:
                gradient = VerticalGradient.model_validate(row._asdict())
            except Exception as exc:
                errors.append(f"Gradients row {row_num}: {exc}")
                continue
            if valid_hole_ids and gradient.hole_id not in valid_hole_ids:
                errors.append(f"Gradients row {row_num}: unknown hole_id '{gradient.hole_id}'")
                continue
            gradients.append(gradient)
        return gradients, errors

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

    def _parse_environmental_dataframe(
        self,
        frame: pd.DataFrame,
        collars: list[Collar],
    ) -> tuple[list[EnvironmentalReading], list[str]]:
        columns = set(frame.columns)
        if not {"hole_id", "parameter", "value"}.issubset(columns):
            return [], []
        has_depth = "depth" in columns
        has_interval = "from_depth" in columns and "to_depth" in columns
        if not has_depth and not has_interval:
            return [], []
        valid_hole_ids = {collar.hole_id for collar in collars}
        readings: list[EnvironmentalReading] = []
        errors: list[str] = []
        for row_num, row in enumerate(frame.itertuples(index=False), start=2):
            if self._blank_hole_id(getattr(row, "hole_id", None)):
                continue
            payload = row._asdict()
            if "unit" not in payload:
                payload["unit"] = ""
            if "value_label" not in payload:
                payload["value_label"] = ""
            try:
                has_point = (
                    has_depth
                    and "depth" in payload
                    and not pd.isna(payload["depth"])
                    and str(payload["depth"]).strip() != ""
                )
                has_interval_values = (
                    has_interval
                    and "from_depth" in payload
                    and "to_depth" in payload
                    and not pd.isna(payload["from_depth"])
                    and not pd.isna(payload["to_depth"])
                    and str(payload["from_depth"]).strip() != ""
                    and str(payload["to_depth"]).strip() != ""
                )
                if has_point:
                    payload["from_depth"] = None
                    payload["to_depth"] = None
                elif has_interval_values:
                    payload["depth"] = None
                else:
                    continue
                if payload.get("value_label") in ("", None) or pd.isna(payload.get("value_label")):
                    payload["value_label"] = ""
                reading = EnvironmentalReading.model_validate(payload)
            except Exception as exc:
                errors.append(f"Environmental row {row_num}: {exc}")
                continue
            if valid_hole_ids and reading.hole_id not in valid_hole_ids:
                errors.append(
                    f"Environmental row {row_num}: unknown hole_id '{reading.hole_id}'"
                )
                continue
            readings.append(reading)
        return readings, errors

    def _parse_environmental_sheet(
        self,
        workbook: pd.ExcelFile,
        collars: list[Collar],
    ) -> tuple[list[EnvironmentalReading], list[str]]:
        sheet = self._find_sheet(workbook.sheet_names, "Environmental")
        if not sheet:
            return [], []
        frame = _normalize_columns(pd.read_excel(workbook, sheet_name=sheet))
        return self._parse_environmental_dataframe(frame, collars)

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
