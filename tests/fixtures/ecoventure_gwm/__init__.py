"""Backward-compatible re-exports — prefer `gwm_reference` in application code."""

from gwm_reference import (
    GWM_TRANSECTS,
    TransectSpec,
    build_parse_result,
    build_subset,
    write_fixture_workbook,
)
from gwm_reference.fixtures import (
    COLLAR_META,
    GW_MASL,
    SILT_HOLES,
)

# Legacy names used by early tests
from gwm_reference.fixtures import write_fixture_workbook as _write_wb

ALL_HOLE_IDS = tuple(COLLAR_META.keys())

__all__ = [
    "GWM_TRANSECTS",
    "TransectSpec",
    "build_parse_result",
    "build_subset",
    "write_fixture_workbook",
    "ALL_HOLE_IDS",
    "COLLAR_META",
    "GW_MASL",
    "SILT_HOLES",
]
