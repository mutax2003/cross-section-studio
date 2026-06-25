"""Shared pytest fixtures and E2E pipeline helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from io import BytesIO
from typing import Sequence

import pandas as pd
import pytest

from models import Collar, Lithology, Transect
from pipeline import build_cross_section
from stratigraphy import GeologicalPolygon


@pytest.fixture
def axis_transect() -> Transect:
    return Transect(points=[(0.0, 0.0), (100.0, 0.0), (100.0, 50.0)])


@pytest.fixture
def axis_collars() -> list[Collar]:
    return [
        Collar(hole_id="BH-01", easting=10.0, northing=5.0, elevation=100.0, total_depth=20.0),
        Collar(hole_id="BH-02", easting=60.0, northing=-3.0, elevation=102.0, total_depth=25.0),
    ]


@pytest.fixture
def axis_lithologies() -> list[Lithology]:
    return [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Sandstone"),
        Lithology(hole_id="BH-01", from_depth=5.0, to_depth=15.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=6.0, lithology_code="Sandstone"),
        Lithology(hole_id="BH-02", from_depth=6.0, to_depth=18.0, lithology_code="Clay"),
    ]


def make_workbook_bytes(
    collars: list[dict],
    lithology: list[dict],
    *,
    collars_sheet: str = "Collars",
    lithology_sheet: str = "Lithology",
) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(collars).to_excel(writer, sheet_name=collars_sheet, index=False)
        pd.DataFrame(lithology).to_excel(writer, sheet_name=lithology_sheet, index=False)
    return buffer.getvalue()


def run_pipeline(
    collars: Sequence[Collar],
    lithologies: Sequence[Lithology],
    transect_points: Sequence[tuple[float, float]],
    *,
    vertical_exaggeration: float = 1.0,
    show_hatches: bool = True,
    show_legend: bool = True,
    interpretation_mode: str = "interpolated",
    allow_pinch_outs: bool = True,
) -> tuple[pd.DataFrame, list[GeologicalPolygon], bytes]:
    projected, polygons, svg_bytes, _, _ = build_cross_section(
        collars,
        lithologies,
        transect_points,
        vertical_exaggeration=vertical_exaggeration,
        show_hatches=show_hatches,
        show_legend=show_legend,
        interpretation_mode=interpretation_mode,  # type: ignore[arg-type]
        allow_pinch_outs=allow_pinch_outs,
    )
    return projected, polygons, svg_bytes


def make_field_export_bytes(
    rows: list[dict],
    *,
    extra_sheets: dict[str, pd.DataFrame] | None = None,
) -> bytes:
    sheet_rows = [
        {
            "Label": row["hole_id"],
            "Depth": row["depth"],
            "Lithology": row["lithology"],
            "Lat": row["lat"],
            "Long": row["long"],
        }
        for row in rows
    ]
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(sheet_rows).to_excel(writer, sheet_name="Lithology", index=False)
        if extra_sheets:
            for name, frame in extra_sheets.items():
                frame.to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()


def assert_valid_svg(svg_bytes: bytes) -> None:
    assert len(svg_bytes) > 100
    lowered = svg_bytes.lower()
    assert b"<svg" in lowered or b"<?xml" in lowered
