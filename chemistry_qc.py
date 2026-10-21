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

import re
from collections.abc import Iterable, Mapping
from typing import Literal

from render_profiles import (
    ChemistryThresholdPair,
    chemistry_parameter_key,
    chemistry_threshold_pair_for,
)

# --- Lab qualifier flags (Environmental ``qualifier`` column) -------------------
#
# Policy: a "U"-family flag (U, UJ, UB, ...), "<", ND / NDL / BDL / "not
# detected" mark a non-detect (value = detection limit). Detected-data flags
# (J estimated, B blank contamination, E over calibration range, D dilution,
# N / NJ tentatively identified, P, Q, H, M, X, * ...) keep the value as
# measured. R (rejected) is NOT dropped: the value is plotted as entered and
# Validate warns "R-qualified (rejected) result plotted" so the user decides --
# silently deleting a lab result is worse than flagging it. Text that is none
# of these is kept (value treated as entered) with a Validate warning instead
# of rejecting the whole row.

QualifierKind = Literal["non_detect", "detected", "unknown"]

QUALIFIER_NON_DETECT = frozenset(
    {"<", "nd", "n.d.", "n.d", "ndl", "u", "bdl", "not detected", "non-detect", "non detect",
     "below detection limit", "below detection", "<rdl", "<dl", "<mdl", "no", "n", "false", "0"}
)
QUALIFIER_DETECTED = frozenset({"=", "detected", "yes", "y", "true", "1", "d"})
# Letters of common detected-data lab flags (EPA CLP / Canadian lab reports).
_LAB_FLAG_LETTERS = frozenset("jbedhnpqmxr*")
_TOKEN_NON_DETECT = frozenset({"nd", "n.d.", "ndl", "bdl", "u"})
_TOKEN_SPLIT_RE = re.compile(r"[,;/\s]+")


def _lab_flag_token(token: str) -> QualifierKind | None:
    core = token.rstrip("+-")
    if token in _TOKEN_NON_DETECT or token.startswith("<"):
        return "non_detect"
    if not core or len(core) > 4:
        return None
    if core.startswith("u") and set(core[1:]) <= _LAB_FLAG_LETTERS:
        return "non_detect"
    if set(core) <= _LAB_FLAG_LETTERS:
        return "detected"
    return None


def classify_lab_qualifier(text: object) -> tuple[QualifierKind | None, bool]:
    """``(kind, rejected)`` of a qualifier cell; ``(None, False)`` when blank.

    ``kind`` is ``"non_detect"``, ``"detected"`` or ``"unknown"``; ``rejected``
    is True for an R flag (``R``, ``UR``, ``J,R`` ...), see the policy above.
    """
    compact = " ".join(str(text or "").strip().casefold().split())
    if not compact:
        return None, False
    if compact in QUALIFIER_NON_DETECT or compact.replace(" ", "") in QUALIFIER_NON_DETECT:
        return "non_detect", False
    if compact.startswith("<"):
        return "non_detect", False
    if compact in QUALIFIER_DETECTED:
        return "detected", False
    tokens = [token for token in _TOKEN_SPLIT_RE.split(compact) if token]
    kinds = [_lab_flag_token(token) for token in tokens]
    if not tokens or any(kind is None for kind in kinds):
        return "unknown", False
    rejected = any("r" in token.rstrip("+-") for token in tokens)
    if "non_detect" in kinds:
        return "non_detect", rejected
    return "detected", rejected


def _sample_where(reading: object) -> str:
    hole_id = getattr(reading, "hole_id", "")
    parameter = getattr(reading, "parameter", "")
    depth = float(getattr(reading, "sample_depth"))
    return f"{hole_id} {parameter} at {depth:g} m"


def lab_qualifier_warnings(readings: Iterable[object]) -> list[str]:
    """Validate warnings for R-qualified (rejected) and unrecognised qualifier flags."""
    warnings: list[str] = []
    for reading in readings:
        text = str(getattr(reading, "qualifier", "") or "").strip()
        if not text:
            continue
        kind, rejected = classify_lab_qualifier(text)
        if rejected:
            warnings.append(
                f"{_sample_where(reading)}: R-qualified (rejected) result plotted "
                f"(qualifier '{text}') — the lab rejected this value; confirm it or remove the row."
            )
        elif kind == "unknown":
            warnings.append(
                f"{_sample_where(reading)}: qualifier '{text}' not recognised — value plotted as "
                "entered; use < / ND / U for a non-detect or a lab flag such as J."
            )
    return warnings


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
