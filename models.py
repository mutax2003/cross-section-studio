"""Immutable Pydantic schemas for borehole datasets."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

COLLAR_COLUMNS = {"hole_id", "easting", "northing", "elevation", "total_depth"}
LITHOLOGY_COLUMNS = {"hole_id", "from_depth", "to_depth", "lithology_code"}
LITHOLOGY_OPTIONAL_COLUMNS = {"hatch_pattern", "unit_order"}
WATER_COLUMNS = {"hole_id"}
WATER_VALUE_COLUMNS = frozenset({"depth", "elevation_masl"})
WATER_OPTIONAL_COLUMNS = frozenset({"series_id", "series_label", "color", "marker", "depth", "elevation_masl"})
SCREEN_COLUMNS = {"hole_id", "from_depth", "to_depth"}
GRADIENT_COLUMNS = {"hole_id", "direction"}

InterpretationMode = Literal["borehole_only", "interpolated", "correlation_lines"]


class Collar(BaseModel, frozen=True):
    hole_id: str
    easting: float
    northing: float
    elevation: float
    total_depth: float
    elevation_datum: str | None = None
    inclination_deg: float | None = None
    azimuth_deg: float | None = None

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
        text = str(value).strip()
        if not text:
            return None
        return int(text)

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
    elevation_masl: float | None = None
    series_id: str = "default"
    series_label: str = ""
    color: str | None = None
    marker: str | None = None

    @field_validator("hole_id", mode="before")
    @classmethod
    def strip_hole_id(cls, value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            raise ValueError("hole_id is required")
        return str(value).strip()

    @field_validator("series_id", mode="before")
    @classmethod
    def default_series_id(cls, value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "default"
        text = str(value).strip()
        return text or "default"

    @field_validator("series_label", mode="before")
    @classmethod
    def strip_series_label(cls, value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        return str(value).strip()

    @field_validator("color", "marker", mode="before")
    @classmethod
    def optional_string(cls, value: object) -> str | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = str(value).strip()
        return text or None

    @field_validator("depth")
    @classmethod
    def validate_depth(cls, value: float) -> float:
        if value < 0:
            raise ValueError("depth must be non-negative")
        return value


class ScreenInterval(BaseModel, frozen=True):
    hole_id: str
    from_depth: float
    to_depth: float

    @field_validator("hole_id", mode="before")
    @classmethod
    def strip_hole_id(cls, value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            raise ValueError("hole_id is required")
        return str(value).strip()

    @model_validator(mode="after")
    def validate_depths(self) -> ScreenInterval:
        if self.to_depth <= self.from_depth:
            raise ValueError("to_depth must be greater than from_depth")
        return self


class VerticalGradient(BaseModel, frozen=True):
    hole_id: str
    direction: str = "up"

    @field_validator("hole_id", mode="before")
    @classmethod
    def strip_hole_id(cls, value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            raise ValueError("hole_id is required")
        return str(value).strip()

    @field_validator("direction", mode="before")
    @classmethod
    def normalize_direction(cls, value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "up"
        text = str(value).strip().lower()
        if text in {"up", "u", "↑"}:
            return "up"
        if text in {"down", "d", "↓"}:
            return "down"
        return "up"


class DeviationReading(BaseModel, frozen=True):
    hole_id: str
    depth: float
    inclination_deg: float
    azimuth_deg: float

    @field_validator("hole_id", mode="before")
    @classmethod
    def strip_hole_id(cls, value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            raise ValueError("hole_id is required")
        return str(value).strip()


class CorrelationOverride(BaseModel, frozen=True):
    """Manual geologist pairing of units between adjacent holes."""

    left_hole_id: str
    right_hole_id: str
    left_unit_order: int
    right_unit_order: int

    @field_validator("left_hole_id", "right_hole_id", mode="before")
    @classmethod
    def strip_ids(cls, value: object) -> str:
        return str(value).strip()


class Fault(BaseModel, frozen=True):
    """Fault trace in profile plane (x_profile, elevation)."""

    name: str
    trace_points: list[tuple[float, float]] = Field(min_length=2)


class Unconformity(BaseModel, frozen=True):
    """Unconformity surface in profile plane."""

    name: str
    elevation_profile: list[tuple[float, float]] = Field(min_length=2)


class EnvironmentalReading(BaseModel, frozen=True):
    """Environmental / lab sample on a point depth or depth interval."""

    hole_id: str
    parameter: str
    value: float
    depth: float | None = None
    from_depth: float | None = None
    to_depth: float | None = None
    unit: str = ""
    value_label: str = ""

    @field_validator("hole_id", "parameter", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> str:
        return str(value).strip()

    @model_validator(mode="after")
    def validate_depth_fields(self) -> EnvironmentalReading:
        has_point = self.depth is not None
        has_interval = self.from_depth is not None or self.to_depth is not None
        if has_point and has_interval:
            raise ValueError("provide depth or from_depth/to_depth, not both")
        if not has_point and (self.from_depth is None or self.to_depth is None):
            raise ValueError("depth or from_depth and to_depth are required")
        if has_point and self.depth is not None and self.depth < 0:
            raise ValueError("depth must be non-negative")
        if has_interval and self.from_depth is not None and self.to_depth is not None:
            if self.from_depth < 0 or self.to_depth < 0:
                raise ValueError("depths must be non-negative")
            if self.to_depth < self.from_depth:
                raise ValueError("to_depth must be >= from_depth")
        return self

    @property
    def sample_depth(self) -> float:
        if self.depth is not None:
            return self.depth
        assert self.from_depth is not None and self.to_depth is not None
        return (self.from_depth + self.to_depth) / 2.0

    @property
    def display_label(self) -> str:
        if self.value_label:
            return self.value_label
        unit_suffix = f" {self.unit}" if self.unit else ""
        return f"{self.value:g}{unit_suffix}"


class RasterLogStrip(BaseModel, frozen=True):
    """Optional raster log column placeholder for section-sheet rendering."""

    hole_id: str
    depth_top: float
    depth_bottom: float
    label: str = "Raster log"
    image_bytes: bytes | None = None

    @field_validator("hole_id", mode="before")
    @classmethod
    def strip_hole_id(cls, value: object) -> str:
        return str(value).strip()


class ConsultingTitleBlock(BaseModel, frozen=True):
    """Report-sheet metadata for consulting cross-section layout."""

    section_label: str = ""
    transect_start_label: str = ""
    transect_end_label: str = ""
    transect_start_primary: str = ""
    transect_start_secondary: str = ""
    transect_end_primary: str = ""
    transect_end_secondary: str = ""
    map_scale: str = "1:1000"
    figure_number: str = ""
    project_number: str = ""
    source: str = ""
    date: str = ""
    notes: tuple[str, ...] = ()
    drawn_by: str = ""
    revised: str = ""
    prepared_for: str = ""
    prepared_by: str = ""
    logo_prepared_for_bytes: bytes | None = None
    logo_prepared_by_bytes: bytes | None = None
    screen_legend_label: str = "SCREENED INTERVAL"
    y_axis_label: str = "ELEVATION ABOVE SEA LEVEL (MASL)"
    scale_bar_m: float = 30.0
    show_gradient_legend: bool = False


class SectionFigureMetadata(BaseModel, frozen=True):
    """Cartographic and geologic metadata for figure footers."""

    coordinate_reference: str = ""
    elevation_datum: str = ""
    vertical_exaggeration: float = 1.0
    transect_azimuth_deg: float | None = None
    hole_count: int = 0
    max_offset_m: float = 0.0
    uses_placeholder_elevation: bool = False


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
    screen_intervals: tuple[ScreenInterval, ...] = ()
    vertical_gradients: tuple[VerticalGradient, ...] = ()
    deviation_readings: tuple[DeviationReading, ...] = ()
    correlation_overrides: tuple[CorrelationOverride, ...] = ()
    faults: tuple[Fault, ...] = ()
    unconformities: tuple[Unconformity, ...] = ()
    environmental_readings: tuple[EnvironmentalReading, ...] = ()


# Re-exports for backward-compatible imports
from parse_ops import (  # noqa: E402
    apply_unit_order_fix,
    assign_missing_unit_orders,
    geology_sheet_counts,
    holes_with_duplicate_lithology_codes,
    lithologies_by_hole,
    lithology_has_unit_order_column,
    parse_bundle_from_json,
    parse_result_to_json_bundle,
    subset_json_bundle,
    subset_parse_result,
)


def __getattr__(name: str):
    """Lazy re-export to avoid models ↔ parsing circular import on cold start."""
    if name == "DataParser":
        from parsing import DataParser as _DataParser

        return _DataParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

