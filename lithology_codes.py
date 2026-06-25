"""Shared lithology code extraction for pipeline and renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import pandas as pd

if TYPE_CHECKING:
    from stratigraphy import GeologicalPolygon


def collect_lithology_codes(
    projected: pd.DataFrame,
    polygons: Sequence["GeologicalPolygon"],
) -> list[str]:
    codes: set[str] = {polygon.lithology_code for polygon in polygons}
    if not projected.empty:
        for raw in projected["lithology_code"].unique():
            code = str(raw).strip()
            if code and code.lower() != "nan":
                codes.add(code)
    return sorted(codes)
