"""Parse Cross_Section_Chlorides.xlsx (or packaged JSON fallback) into environmental readings."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from models import EnvironmentalReading
from paths import cross_section_chlorides_workbook

_PARAMETER = "Chloride"
_UNIT = "mg/L"
_PACKAGE_DIR = Path(__file__).resolve().parent
_JSON_FIXTURE = _PACKAGE_DIR / "chloride_readings.json"
_ND_PATTERN = re.compile(
    r"^<\s*([0-9]+(?:\.[0-9]+)?)\s*(?:mg/?L)?(?:\s*\(ND\))?$",
    flags=re.IGNORECASE,
)
_NUMERIC_PATTERN = re.compile(r"^([0-9]+(?:\.[0-9]+)?)")
_TRANSECT_COLUMNS: dict[str, tuple[str, str, str]] = {
    "A_A": ("Borehole", "Avg", "Cl"),
    "B_B": ("Borehole.1", "Avg.1", "Cl.1"),
}


def normalize_compound_hole_id(raw: object) -> str:
    """Normalize BH11/BH24-11 and BH11 / BH24-11 to a single display id."""
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return ""
    return re.sub(r"\s*/\s*", " / ", text)


def parse_chloride_value(raw: object) -> tuple[float, str]:
    """Return numeric plotting value and display label for a chloride cell."""
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        raise ValueError("empty chloride value")
    nd_match = _ND_PATTERN.match(text)
    if nd_match:
        limit = float(nd_match.group(1))
        return limit, f"<{limit:g} {_UNIT}"
    numeric_match = _NUMERIC_PATTERN.match(text)
    if numeric_match:
        value = float(numeric_match.group(1))
        return value, f"{value:g} {_UNIT}"
    raise ValueError(f"unrecognized chloride value: {text!r}")


@lru_cache(maxsize=4)
def _load_chloride_frame(path_str: str, mtime_ns: int) -> pd.DataFrame:
    del mtime_ns
    return pd.read_excel(path_str, sheet_name="Chloride", header=1)


def _reading_from_dict(row: dict[str, object]) -> EnvironmentalReading:
    return EnvironmentalReading(
        hole_id=str(row["hole_id"]),
        parameter=str(row.get("parameter") or _PARAMETER),
        value=float(row["value"]),
        depth=float(row["depth"]) if row.get("depth") is not None else None,
        unit=str(row.get("unit") or _UNIT),
        value_label=str(row.get("value_label") or ""),
    )


@lru_cache(maxsize=1)
def _load_json_fixture(path_str: str, mtime_ns: int) -> dict[str, tuple[EnvironmentalReading, ...]]:
    del mtime_ns
    payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
    return {
        transect_id: tuple(_reading_from_dict(row) for row in rows)
        for transect_id, rows in payload.items()
    }


def _rows_for_transect(frame: pd.DataFrame, transect_id: str) -> tuple[EnvironmentalReading, ...]:
    if transect_id not in _TRANSECT_COLUMNS:
        raise ValueError(f"unknown transect_id: {transect_id}")
    hole_col, avg_col, cl_col = _TRANSECT_COLUMNS[transect_id]
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
        return {key: _rows_for_transect(frame, key) for key in _TRANSECT_COLUMNS}

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
