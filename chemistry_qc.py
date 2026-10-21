"""Chemistry QA helpers: non-detects against guidelines, unit consistency.

Engine module (no Streamlit). Used by the Validate summary
(``ai_quality.summarize_environmental_readings``) and the chemistry overlay
(``renderer_chemistry``) so the warning text is identical in both places.

Threshold limits carry no unit of their own (they are "in the parameter's own
units"), so nothing here converts units: a parameter reported in more than one
unit is flagged for the user to fix in the workbook instead of being silently
converted.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from render_profiles import (
    ChemistryThresholdPair,
    chemistry_parameter_key,
    chemistry_threshold_pair_for,
)


def unit_key(unit: object) -> str:
    """Comparison key for a unit spelling: ``"µg/L"``, ``"ug/l"`` and ``"μg / L"`` match.

    Only spelling is normalised (case, spaces, micro sign); different units
    (mg/L vs µg/L, ppb vs µg/L) stay different.
    """
    text = str(unit or "").strip().casefold()
    text = text.replace("µ", "u").replace("μ", "u")
    return "".join(text.split())


def mixed_unit_parameters(readings: Iterable[object]) -> dict[str, tuple[str, ...]]:
    """Parameters reported in more than one unit: name -> unit spellings (first seen each).

    Blank units are ignored (unknown, not a second unit). Parameter names
    collapse case-insensitively and keep their first spelling.
    """
    names: dict[str, str] = {}
    units: dict[str, dict[str, str]] = {}
    for reading in readings:
        name = str(getattr(reading, "parameter", "") or "").strip()
        key = chemistry_parameter_key(name)
        unit = str(getattr(reading, "unit", "") or "").strip()
        if not key or not unit:
            continue
        names.setdefault(key, name)
        units.setdefault(key, {}).setdefault(unit_key(unit), unit)
    return {
        names[key]: tuple(spellings.values())
        for key, spellings in sorted(units.items())
        if len(spellings) > 1
    }


def mixed_unit_warnings(readings: Iterable[object]) -> list[str]:
    """One Validate warning per parameter reported in more than one unit."""
    return [
        f"{parameter} is reported in more than one unit ({', '.join(units)}) — values and "
        "threshold limits are compared as typed, not converted; use one unit for "
        f"{parameter} in the Environmental sheet."
        for parameter, units in mixed_unit_parameters(readings).items()
    ]


def rdl_exceeds_guideline(reading: object, green_max: float | None) -> bool:
    """True for a non-detect whose detection limit is above the guideline (green limit)."""
    if green_max is None or getattr(reading, "detected", True):
        return False
    return float(getattr(reading, "value")) > float(green_max)


def _with_unit(number: float, unit: str) -> str:
    return f"{float(number):g} {unit}" if unit else f"{float(number):g}"


def rdl_above_guideline_note(reading: object, green_max: float) -> str:
    """``"BH-3 Benzene at 2.5 m: detection limit 5 µg/L exceeds guideline 1 µg/L — result inconclusive"``."""
    unit = str(getattr(reading, "unit", "") or "").strip()
    hole_id = getattr(reading, "hole_id", "")
    parameter = getattr(reading, "parameter", "")
    depth = float(getattr(reading, "sample_depth"))
    return (
        f"{hole_id} {parameter} at {depth:g} m: detection limit "
        f"{_with_unit(getattr(reading, 'value'), unit)} exceeds guideline "
        f"{_with_unit(green_max, unit)} — result inconclusive"
    )


def rdl_above_guideline_notes(
    readings: Iterable[object],
    thresholds: Mapping[str, ChemistryThresholdPair] | None,
) -> list[str]:
    """Notes for non-detects whose detection limit exceeds the parameter's own green limit.

    Only parameters with an explicit green limit in ``thresholds`` are checked
    (Validate uses the workbook ``green_max`` values; the figure resolves its
    own limits per parameter and calls :func:`rdl_above_guideline_note`).
    """
    notes: list[str] = []
    for reading in readings:
        green_max, _orange = chemistry_threshold_pair_for(
            str(getattr(reading, "parameter", "")), thresholds
        )
        if rdl_exceeds_guideline(reading, green_max):
            assert green_max is not None
            notes.append(rdl_above_guideline_note(reading, green_max))
    return notes
