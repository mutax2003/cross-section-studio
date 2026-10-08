"""Immutable Pydantic schemas for borehole datasets."""

from __future__ import annotations

import math
import re
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

COLLAR_COLUMNS = {"hole_id", "easting", "northing", "elevation", "total_depth"}
LITHOLOGY_COLUMNS = {"hole_id", "from_depth", "to_depth", "lithology_code"}
LITHOLOGY_OPTIONAL_COLUMNS = {"hatch_pattern", "unit_order"}
WATER_COLUMNS = {"hole_id"}
WATER_VALUE_COLUMNS = frozenset({"depth", "elevation_masl"})
WATER_STATUS_COLUMNS = frozenset({"status"})
WATER_OPTIONAL_COLUMNS = frozenset(
    {
        "series_id",
        "series_label",
        "color",
        "marker",
        "depth",
        "elevation_masl",
        "connect_group",
        "status",
    }
)
SCREEN_COLUMNS = {"hole_id", "from_depth", "to_depth"}
GRADIENT_COLUMNS = {"hole_id", "direction"}
MAX_WATER_SERIES = 4

InterpretationMode = Literal["borehole_only", "interpolated", "correlation_lines"]


# XML 1.0 disallows these control characters; matplotlib's SVG backend would
# emit them verbatim and produce unparseable documents.
_XML_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean_text(value: object) -> str:
    text = _XML_ILLEGAL.sub("", str(value)).strip()
    if text.startswith("="):
        # A leading '=' is a spreadsheet formula; written back out by the
        # cleaned export it would execute in Excel.
        raise ValueError(f"text cannot start with '=' (formula): {text[:40]!r}")
    return text


_COLLAR_LIMITS = {"easting": 1e9, "northing": 1e9, "elevation": 1e5, "total_depth": 1e5}


class Collar(BaseModel, frozen=True):
    hole_id: str
    easting: float
    northing: float
    elevation: float
    total_depth: float
    elevation_datum: str | None = None
    inclination_deg: float | None = None
    azimuth_deg: float | None = None
    stick_up_m: float | None = None

    @field_validator("hole_id", mode="before")
    @classmethod
    def strip_hole_id(cls, value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            raise ValueError("hole_id is required")
        return _clean_text(value)

    @field_validator("elevation_datum", mode="before")
    @classmethod
    def blank_datum_to_none(cls, value: object) -> str | None:
        # Excel hands back NaN for a cleared cell and int for e.g. "2013".
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = _clean_text(value)
        return text or None

    @field_validator("easting", "northing", "elevation", "total_depth")
    @classmethod
    def require_finite(cls, value: float, info) -> float:
        if not math.isfinite(value):
            raise ValueError(f"{info.field_name} must be a finite number")
        # Real coordinates (UTM / local grids) and elevations are far inside
        # these; larger values are typos and overflowed the section maths.
        limit = _COLLAR_LIMITS[info.field_name]
        if abs(value) > limit:
            raise ValueError(f"{info.field_name} {value:g} is out of range (limit ±{limit:g})")
        return value

    @field_validator("total_depth")
    @classmethod
    def validate_total_depth(cls, value: float) -> float:
        if value < 0:
            raise ValueError("total_depth must be non-negative")
        return value

    @field_validator("inclination_deg", "azimuth_deg", "stick_up_m", mode="before")
    @classmethod
    def blank_optional_to_none(cls, value: object, info) -> object:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isnan(float(value)):
                return None
            if math.isinf(float(value)):
                raise ValueError(f"{info.field_name} must be a finite number")
        return value

    @field_validator("stick_up_m")
    @classmethod
    def validate_stick_up(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if value < 0:
            raise ValueError("stick_up_m must be non-negative")
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
    def strip_strings(cls, value: object, info) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            raise ValueError("required string field is missing")
        text = _clean_text(value)
        if not text:
            raise ValueError(f"{info.field_name} is blank")
        return text

    @field_validator("from_depth", "to_depth")
    @classmethod
    def require_finite(cls, value: float, info) -> float:
        if not math.isfinite(value):
            raise ValueError(f"{info.field_name} must be a finite number")
        return value

    @field_validator("hatch_pattern", mode="before")
    @classmethod
    def optional_string(cls, value: object) -> str | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = _clean_text(value)
        return text or None

    @field_validator("unit_order", mode="before")
    @classmethod
    def optional_unit_order(cls, value: object) -> int | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, bool):
            raise ValueError("unit_order must be an integer")
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError("unit_order must be a whole number")
            return int(value)
        text = _clean_text(value)
        if not text:
            return None
        try:
            number = float(text)
        except ValueError as exc:
            raise ValueError("unit_order must be an integer") from exc
        if not number.is_integer():
            raise ValueError("unit_order must be a whole number")
        return int(number)

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
    connect_group: str = ""
    status: str = "measured"

    @field_validator("hole_id", mode="before")
    @classmethod
    def strip_hole_id(cls, value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            raise ValueError("hole_id is required")
        return _clean_text(value)

    @field_validator("series_id", mode="before")
    @classmethod
    def default_series_id(cls, value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "default"
        text = _clean_text(value)
        return text or "default"

    @field_validator("series_label", "connect_group", mode="before")
    @classmethod
    def strip_series_label(cls, value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        return _clean_text(value)

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "measured"
        text = _clean_text(value).lower()
        if not text:
            return "measured"
        if text in {"dry", "d"}:
            return "dry"
        if text in {"nm", "n/m", "not measured", "not_measured", "ns", "not sampled", "not_sampled"}:
            return "nm"
        if text in {"measured", "meas", "m", "wet"}:
            return "measured"
        raise ValueError(
            f"status must be measured, dry, or nm (got {value!r})"
        )

    @field_validator("color", "marker", mode="before")
    @classmethod
    def optional_string(cls, value: object) -> str | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = _clean_text(value)
        return text or None

    @field_validator("depth")
    @classmethod
    def validate_depth(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("depth must be a finite number")
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
        return _clean_text(value)

    @field_validator("from_depth", "to_depth")
    @classmethod
    def require_finite(cls, value: float, info) -> float:
        if not math.isfinite(value):
            raise ValueError(f"{info.field_name} must be a finite number")
        return value

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
        return _clean_text(value)

    @field_validator("direction", mode="before")
    @classmethod
    def normalize_direction(cls, value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "up"
        text = _clean_text(value).lower()
        if text in {"up", "u", "↑"}:
            return "up"
        if text in {"down", "d", "↓"}:
            return "down"
        if not text:
            return "up"
        raise ValueError(f"direction must be up or down (got {value!r})")


class DeviationReading(BaseModel, frozen=True):
    hole_id: str
    depth: float
    inclination_deg: float
    azimuth_deg: float

    @field_validator("depth", "inclination_deg", "azimuth_deg")
    @classmethod
    def require_finite(cls, value: float, info) -> float:
        # A blank survey cell would otherwise NaN the whole hole's geometry.
        if not math.isfinite(value):
            raise ValueError(f"{info.field_name} must be a finite number")
        return value

    @field_validator("hole_id", mode="before")
    @classmethod
    def strip_hole_id(cls, value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            raise ValueError("hole_id is required")
        return _clean_text(value)


class CorrelationOverride(BaseModel, frozen=True):
    """Manual geologist pairing of units between adjacent holes."""

    left_hole_id: str
    right_hole_id: str
    left_unit_order: int
    right_unit_order: int

    @field_validator("left_hole_id", "right_hole_id", mode="before")
    @classmethod
    def strip_ids(cls, value: object) -> str:
        return _clean_text(value)


class Fault(BaseModel, frozen=True):
    """Fault trace in profile plane (x_profile, elevation)."""

    name: str
    trace_points: list[tuple[float, float]] = Field(min_length=2)


class Unconformity(BaseModel, frozen=True):
    """Unconformity surface in profile plane."""

    name: str
    elevation_profile: list[tuple[float, float]] = Field(min_length=2)


# Fixed label colours a logger may pick in the workbook (one hex each in
# render_theme); blue is reserved for groundwater elevations.
LABEL_COLOR_NAMES: tuple[str, ...] = ("green", "red", "black", "orange")


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
    label_color: str = ""  # "", green, red, black or orange (fixed hex each)

    @field_validator("hole_id", "parameter", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> str:
        return _clean_text(value)

    @field_validator("label_color", mode="before")
    @classmethod
    def normalise_label_color(cls, value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        text = _clean_text(value).casefold()
        if text == "" or text in LABEL_COLOR_NAMES:
            return text
        if text == "blue":
            raise ValueError("label_color 'blue' is reserved for groundwater elevations")
        raise ValueError(f"label_color must be green, red, black or orange (got {text!r})")

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
        return _clean_text(value)


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


class WorkbookSectionSpec(BaseModel, frozen=True):
    """Optional Sections sheet row: label + ordered holes for multi-transect batch."""

    label: str
    hole_ids: tuple[str, ...]

    @field_validator("label", mode="before")
    @classmethod
    def strip_label(cls, value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            raise ValueError("section_label is required")
        text = _clean_text(value)
        if not text:
            raise ValueError("section_label is required")
        if any(ch in text for ch in ("\n", "\r", "|")):
            raise ValueError("section_label cannot contain newlines or '|'")
        return text

    @field_validator("hole_ids", mode="before")
    @classmethod
    def coerce_hole_ids(cls, value: object) -> tuple[str, ...]:
        if value is None:
            raise ValueError("hole_ids is required")
        if isinstance(value, str):
            raise ValueError("hole_ids must be a sequence of hole IDs")
        holes = tuple(_clean_text(item) for item in value if _clean_text(item))
        if len(holes) < 2:
            raise ValueError("section requires at least two hole_ids")
        duplicates = sorted({h for h in holes if holes.count(h) > 1})
        if duplicates:
            raise ValueError(f"hole_id listed more than once: {', '.join(duplicates)}")
        for hole in holes:
            if any(ch in hole for ch in (",", ";", "|", "\n", "\r")) or "→" in hole or "->" in hole:
                raise ValueError(
                    f"hole_id {hole!r} cannot contain separators (, ; | →) or newlines"
                )
        return holes


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
    section_specs: tuple[WorkbookSectionSpec, ...] = ()


# Re-exports for backward-compatible imports
from parse_ops import (  # noqa: E402
    apply_unit_order_fix,  # noqa: F401
    assign_missing_unit_orders,  # noqa: F401
    format_section_specs_as_batch_text,  # noqa: F401
    geology_sheet_counts,  # noqa: F401
    holes_with_duplicate_lithology_codes,  # noqa: F401
    lithologies_by_hole,  # noqa: F401
    lithology_has_unit_order_column,  # noqa: F401
    parse_bundle_from_json,  # noqa: F401
    parse_result_to_json_bundle,  # noqa: F401
    subset_json_bundle,  # noqa: F401
    subset_parse_result,  # noqa: F401
)


def __getattr__(name: str):
    """Lazy re-export to avoid models ↔ parsing circular import on cold start."""
    if name == "DataParser":
        from parsing import DataParser as _DataParser

        return _DataParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

