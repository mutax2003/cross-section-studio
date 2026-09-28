"""Parse Cross_Section_Chlorides.xlsx (or packaged JSON fallback) into environmental readings.

Client Fig 6/7 callouts use interval mid-depths with compact red labels and
**mg/kg** in the legend (not point averages with mg/L).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from models import EnvironmentalReading
from paths import cross_section_chlorides_workbook

_PARAMETER = "Chloride"
_UNIT = "mg/kg"
_PACKAGE_DIR = Path(__file__).resolve().parent
_JSON_FIXTURE = _PACKAGE_DIR / "chloride_readings.json"
_ND_PATTERN = re.compile(
    r"^<\s*([0-9]+(?:\.[0-9]+)?)\s*(?:mg/?[Lk]g?)?(?:\s*\(ND\))?$",
    flags=re.IGNORECASE,
)
_NUMERIC_PATTERN = re.compile(r"^([0-9]+(?:\.[0-9]+)?)")
# Point-average columns (legacy). Prefer From/To intervals when present.
_TRANSECT_POINT_COLUMNS: dict[str, tuple[str, str, str]] = {
    "A_A": ("Borehole", "Avg", "Cl"),
    "B_B": ("Borehole.1", "Avg.1", "Cl.1"),
}
_TRANSECT_INTERVAL_COLUMNS: dict[str, tuple[str, str, str, str]] = {
    "A_A": ("Borehole", "Cl", "From", "To"),
    "B_B": ("Borehole.1", "Cl.1", "From.1", "To.1"),
}


def normalize_compound_hole_id(raw: object) -> str:
    """Normalize BH11/BH24-11 and BH11 / BH24-11 to a single display id."""
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return ""
    return re.sub(r"\s*/\s*", " / ", text)


def parse_chloride_value(raw: object) -> tuple[float, str]:
    """Return numeric plotting value and compact display label (no unit suffix)."""
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        raise ValueError("empty chloride value")
    nd_match = _ND_PATTERN.match(text)
    if nd_match:
        limit = float(nd_match.group(1))
        return limit, f"<{limit:g}"
    numeric_match = _NUMERIC_PATTERN.match(text)
    if numeric_match:
        value = float(numeric_match.group(1))
        return value, f"{value:g}"
    raise ValueError(f"unrecognized chloride value: {text!r}")


@lru_cache(maxsize=4)
def _load_chloride_frame(path_str: str, mtime_ns: int) -> pd.DataFrame:
    del mtime_ns
    return pd.read_excel(path_str, sheet_name="Chloride", header=1)


def _reading_from_dict(row: dict[str, object]) -> EnvironmentalReading:
    depth = row.get("depth")
    from_depth = row.get("from_depth")
    to_depth = row.get("to_depth")
    kwargs: dict[str, object] = {
        "hole_id": str(row["hole_id"]),
        "parameter": str(row.get("parameter") or _PARAMETER),
        "value": float(row["value"]),
        "unit": str(row.get("unit") or _UNIT),
        "value_label": str(row.get("value_label") or ""),
    }
    if from_depth is not None and to_depth is not None and depth is None:
        kwargs["from_depth"] = float(from_depth)
        kwargs["to_depth"] = float(to_depth)
    else:
        kwargs["depth"] = float(depth) if depth is not None else None
    return EnvironmentalReading(**kwargs)  # type: ignore[arg-type]


@lru_cache(maxsize=1)
def _load_json_fixture(path_str: str, mtime_ns: int) -> dict[str, tuple[EnvironmentalReading, ...]]:
    del mtime_ns
    payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
    return {
        transect_id: tuple(_reading_from_dict(row) for row in rows)
        for transect_id, rows in payload.items()
    }


def _interval_rows_for_transect(
    frame: pd.DataFrame, transect_id: str
) -> tuple[EnvironmentalReading, ...]:
    hole_col, cl_col, from_col, to_col = _TRANSECT_INTERVAL_COLUMNS[transect_id]
    if not all(col in frame.columns for col in (hole_col, cl_col, from_col, to_col)):
        return ()
    readings: list[EnvironmentalReading] = []
    for record in frame.to_dict(orient="records"):
        hole_id = normalize_compound_hole_id(record.get(hole_col))
        if not hole_id:
            continue
        try:
            from_depth = float(record[from_col])  # type: ignore[arg-type]
            to_depth = float(record[to_col])  # type: ignore[arg-type]
            value, value_label = parse_chloride_value(record[cl_col])
        except (TypeError, ValueError, KeyError):
            continue
        if to_depth < from_depth:
            from_depth, to_depth = to_depth, from_depth
        readings.append(
            EnvironmentalReading(
                hole_id=hole_id,
                parameter=_PARAMETER,
                value=value,
                from_depth=from_depth,
                to_depth=to_depth,
                unit=_UNIT,
                value_label=value_label,
            )
        )
    return tuple(readings)


def _point_rows_for_transect(
    frame: pd.DataFrame, transect_id: str
) -> tuple[EnvironmentalReading, ...]:
    if transect_id not in _TRANSECT_POINT_COLUMNS:
        raise ValueError(f"unknown transect_id: {transect_id}")
    hole_col, avg_col, cl_col = _TRANSECT_POINT_COLUMNS[transect_id]
    subset = frame[[hole_col, avg_col, cl_col]].dropna(subset=[hole_col, avg_col, cl_col])
    readings: list[EnvironmentalReading] = []
    for hole_raw, avg_raw, cl_raw in subset.itertuples(index=False, name=None):
        hole_id = normalize_compound_hole_id(hole_raw)
        if not hole_id:
            continue
        try:
            avg_depth = float(avg_raw)
            value, value_label = parse_chloride_value(cl_raw)
        except (TypeError, ValueError):
            continue
        readings.append(
            EnvironmentalReading(
                hole_id=hole_id,
                parameter=_PARAMETER,
                value=value,
                depth=avg_depth,
                unit=_UNIT,
                value_label=value_label,
            )
        )
    return tuple(readings)


def _rows_for_transect(frame: pd.DataFrame, transect_id: str) -> tuple[EnvironmentalReading, ...]:
    if transect_id not in _TRANSECT_INTERVAL_COLUMNS:
        raise ValueError(f"unknown transect_id: {transect_id}")
    intervals = _interval_rows_for_transect(frame, transect_id)
    if intervals:
        return intervals
    return _point_rows_for_transect(frame, transect_id)


def packaged_chloride_fixture_path() -> Path:
    """Path to the CI-vendored chloride JSON (xlsx preferred when present)."""
    return _JSON_FIXTURE


def load_chloride_readings(
    workbook: Path | None = None,
    *,
    transect_id: str | None = None,
) -> tuple[EnvironmentalReading, ...] | dict[str, tuple[EnvironmentalReading, ...]]:
    """Load chloride environmental readings from workbook or packaged JSON fallback."""
    path = workbook or cross_section_chlorides_workbook()
    if path.exists():
        frame = _load_chloride_frame(str(path), path.stat().st_mtime_ns)
        if transect_id is not None:
            return _rows_for_transect(frame, transect_id)
        return {key: _rows_for_transect(frame, key) for key in _TRANSECT_INTERVAL_COLUMNS}

    if not _JSON_FIXTURE.is_file():
        raise FileNotFoundError(
            f"Chloride workbook not found: {path}; also missing packaged fixture {_JSON_FIXTURE}"
        )
    by_transect = _load_json_fixture(str(_JSON_FIXTURE), _JSON_FIXTURE.stat().st_mtime_ns)
    if transect_id is not None:
        if transect_id not in by_transect:
            raise ValueError(f"unknown transect_id: {transect_id}")
        return by_transect[transect_id]
    return dict(by_transect)
