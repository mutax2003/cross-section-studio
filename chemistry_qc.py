"""Chemistry QA helpers: non-detects against guidelines, unit consistency.

Engine module (no Streamlit). Used by the Validate summary
(``ai_quality.summarize_environmental_readings``) and the chemistry overlay
(``renderer_chemistry``) so the warning text is identical in both places.

Units: threshold limits carry no unit of their own -- they are in the
parameter's *main* unit, the unit most of its readings use
(:func:`dominant_units`). A parameter reported in more than one unit is
harmonised for the figure by :func:`harmonize_units`: a reading in a known
convertible unit (ng/L <-> ug/L <-> mg/L <-> g/L, ug/kg <-> mg/kg = ug/g <-> g/kg,
ppb <-> ppm, and ppm / ppb as mg/L / mg/kg when the parameter's other units are
all aqueous or all solid) is shown and threshold-coloured in the main unit, and
the figure QA notes say so ("MW-04 Chloride 45000 ug/L shown as 45 mg/L
(converted to the parameter's main unit)"). A reading whose unit cannot be
converted (mg/kg beside mg/L, an unknown unit) keeps its own value and unit on
the figure, is never threshold-coloured, and gets a QA note. Nothing is
converted silently; Validate still asks for one unit per parameter.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
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
     "below detection limit", "below detection", "<rdl", "<dl", "<mdl", "no", "false", "0"}
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
    """One Validate warning per parameter reported in more than one unit.

    Says what the figure does with the odd readings: converted to the main
    unit (named in the figure QA notes), or printed with their own unit and
    left out of the threshold colours when no conversion is known.
    """
    readings = list(readings)
    harmonized = harmonize_units(readings)
    warnings: list[str] = []
    for parameter, units in mixed_unit_parameters(readings).items():
        key = chemistry_parameter_key(parameter)
        main = harmonized.dominant.get(key, "")
        stuck = sorted(
            {
                str(getattr(reading, "unit", "") or "").strip()
                for reading in harmonized.unconverted
                if chemistry_parameter_key(str(getattr(reading, "parameter", ""))) == key
            }
        )
        if stuck:
            handling = (
                f"the figure converts what it can to {main} (its main unit); {', '.join(stuck)} "
                f"cannot be converted to {main}, so those values print with their own unit and "
                "are not threshold-coloured"
            )
        else:
            handling = (
                f"the figure converts them to {main} (its main unit) and says so in the QA notes; "
                f"threshold limits are read as {main}"
            )
        warnings.append(
            f"{parameter} is reported in more than one unit ({', '.join(units)}) — {handling}. "
            f"Use one unit for {parameter} in the Environmental sheet."
        )
    return warnings


# --- Unit conversion (figure) -------------------------------------------------
#
# unit_key spelling -> (family, factor to the family's base unit). "ratio" is
# ppm / ppb: mass per mass or (dilute water) mass per volume, so it joins the
# aqueous or the solid family only when the parameter's other units say which.
# "ppt" is left out on purpose: parts per thousand or per trillion.

_UNIT_FAMILIES: dict[str, tuple[str, float]] = {
    "g/l": ("aqueous", 1e3),
    "mg/l": ("aqueous", 1.0),
    "ug/l": ("aqueous", 1e-3),
    "mcg/l": ("aqueous", 1e-3),
    "ng/l": ("aqueous", 1e-6),
    "g/kg": ("solid", 1e3),
    "mg/kg": ("solid", 1.0),
    "ug/g": ("solid", 1.0),
    "mcg/g": ("solid", 1.0),
    "ug/kg": ("solid", 1e-3),
    "mcg/kg": ("solid", 1e-3),
    "ng/g": ("solid", 1e-3),
    "ng/kg": ("solid", 1e-6),
    "ppm": ("ratio", 1.0),
    "ppb": ("ratio", 1e-3),
}


def unit_family(unit: object) -> tuple[str, float] | None:
    """``(family, factor)`` of a known unit (``"ug/L"`` -> ``("aqueous", 0.001)``), else None."""
    return _UNIT_FAMILIES.get(unit_key(unit))


def convert_unit(
    value: float,
    from_unit: object,
    to_unit: object,
    *,
    ratio_family: str | None = None,
) -> float | None:
    """``value`` in ``from_unit`` expressed in ``to_unit``, or None when not convertible.

    Same spelling key: unchanged. ppm / ppb convert to the aqueous or solid
    family only when ``ratio_family`` names it (the parameter's other units
    are all of that family): 1 ppm = 1 mg/L (dilute water) = 1 mg/kg.
    """
    if unit_key(from_unit) == unit_key(to_unit):
        return float(value)
    source, target = unit_family(from_unit), unit_family(to_unit)
    if source is None or target is None:
        return None
    (source_family, source_factor), (target_family, target_factor) = source, target
    if source_family != target_family and (
        ratio_family is None or {source_family, target_family} != {"ratio", ratio_family}
    ):
        return None
    # 12 significant digits: 0.003 * 1000 prints "3", not "3.0000000000000004".
    return float(f"{float(value) * source_factor / target_factor:.12g}")


def dominant_units(readings: Iterable[object]) -> dict[str, str]:
    """Main unit per parameter key: the unit most readings use (first seen wins a tie).

    Blank units are ignored; spellings collapse by :func:`unit_key` and keep
    the first spelling seen. Parameters without any unit are left out.
    """
    counts: dict[str, dict[str, int]] = {}
    spellings: dict[str, dict[str, str]] = {}
    for reading in readings:
        key = chemistry_parameter_key(str(getattr(reading, "parameter", "") or ""))
        unit = str(getattr(reading, "unit", "") or "").strip()
        if not key or not unit:
            continue
        ukey = unit_key(unit)
        spellings.setdefault(key, {}).setdefault(ukey, unit)
        per = counts.setdefault(key, {})
        per[ukey] = per.get(ukey, 0) + 1
    result: dict[str, str] = {}
    for key, per in counts.items():
        best = max(per.values())
        # Insertion order: the first unit seen with the top count.
        ukey = next(u for u, n in per.items() if n == best)
        result[key] = spellings[key][ukey]
    return result


@dataclass
class UnitHarmonization:
    """Result of :func:`harmonize_units` (see the module docstring)."""

    # Same order as the input; converted readings carry the main unit.
    readings: list = field(default_factory=list)
    # Parameter key (chemistry_parameter_key) -> main unit spelling.
    dominant: dict[str, str] = field(default_factory=dict)
    # Readings (as returned) whose unit could not be converted.
    unconverted: list = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def dominant_unit_for(self, parameter: str) -> str:
        return self.dominant.get(chemistry_parameter_key(parameter), "")

    def is_unconverted(self, reading: object) -> bool:
        return any(reading is item for item in self.unconverted)


def harmonize_units(readings: Iterable[object]) -> UnitHarmonization:
    """Readings in each parameter's main unit, plus figure QA notes.

    A reading in another, convertible unit is replaced by a copy in the main
    unit (value / detection limit converted; ``value_label`` text typed in the
    workbook is kept as typed). A reading in a unit that cannot be converted
    is returned unchanged and listed in ``unconverted``: the figure prints it
    with its own unit and does not threshold-colour it. Readings without a
    unit, or of a parameter with a single unit, are returned unchanged.
    """
    readings = list(readings)
    dominant = dominant_units(readings)
    families: dict[str, set[str]] = {}
    for reading in readings:
        key = chemistry_parameter_key(str(getattr(reading, "parameter", "") or ""))
        family = unit_family(getattr(reading, "unit", ""))
        if key and family is not None and family[0] != "ratio":
            families.setdefault(key, set()).add(family[0])
    result = UnitHarmonization(dominant=dominant)
    for reading in readings:
        key = chemistry_parameter_key(str(getattr(reading, "parameter", "") or ""))
        unit = str(getattr(reading, "unit", "") or "").strip()
        main = dominant.get(key, "")
        if not unit or not main or unit_key(unit) == unit_key(main):
            result.readings.append(reading)
            continue
        matrix = families.get(key, set())
        ratio_family = next(iter(matrix)) if len(matrix) == 1 else None
        value = float(getattr(reading, "value"))
        where = f"{getattr(reading, 'hole_id', '')} {getattr(reading, 'parameter', '')}"
        converted = convert_unit(value, unit, main, ratio_family=ratio_family)
        if converted is None:
            result.readings.append(reading)
            result.unconverted.append(reading)
            note = (
                f"{where} {value:g} {unit} printed with its own unit — {unit} cannot be "
                f"converted to {main} (the parameter's main unit), so it is not threshold-coloured"
            )
        else:
            result.readings.append(
                reading.model_copy(update={"value": converted, "unit": main})  # type: ignore[attr-defined]
            )
            prefix = "" if getattr(reading, "detected", True) else "<"
            note = (
                f"{where} {prefix}{value:g} {unit} shown as {prefix}{converted:g} {main} "
                "(converted to the parameter's main unit)"
            )
        if note not in result.notes:
            result.notes.append(note)
    return result


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
