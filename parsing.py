"""Native Excel workbook parsing (DataParser)."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable, Sequence
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

from models import (
    COLLAR_COLUMNS,
    GRADIENT_COLUMNS,
    LITHOLOGY_COLUMNS,
    LITHOLOGY_OPTIONAL_COLUMNS,
    SCREEN_COLUMNS,
    WATER_STATUS_COLUMNS,
    WATER_VALUE_COLUMNS,
    Collar,
    CorrelationOverride,
    DeviationReading,
    EnvironmentalReading,
    Fault,
    Lithology,
    ParseResult,
    ScreenInterval,
    Unconformity,
    VerticalGradient,
    WaterLevel,
    WorkbookSectionSpec,
    _clean_text,
    check_bounded_number,
    reject_bool_number,
)

logger = logging.getLogger(__name__)

WorkbookSource = str | Path | BinaryIO | BytesIO | pd.ExcelFile


# A 5 MB sheet of a million empty-but-formatted rows takes ~18 s to read and
# scales linearly; real projects are a few thousand rows.
MAX_WORKBOOK_ROWS = 200_000


class WorkbookTooLargeError(ValueError):
    """A sheet exceeds MAX_WORKBOOK_ROWS (usually blank formatted rows)."""


def check_workbook_row_counts(source: WorkbookSource, *, limit: int = MAX_WORKBOOK_ROWS) -> None:
    """Reject oversized sheets cheaply (openpyxl read-only) before pandas loads them."""
    if isinstance(source, pd.ExcelFile):
        return
    import openpyxl

    if hasattr(source, "seek"):
        source.seek(0)
    try:
        book = openpyxl.load_workbook(source, read_only=True, data_only=True)
    except Exception:
        return  # let the real reader report format problems
    try:
        for sheet in book.worksheets:
            rows = sheet.max_row
            if rows is None or rows <= limit:
                # The <dimension> tag is only a claim: it can be missing (write-
                # only generators) or understated (a tampered A1:E2 on 600k rows
                # parsed for 45 s). Count real rows, stopping one past the limit.
                reset = getattr(sheet, "reset_dimensions", None)
                if callable(reset):
                    reset()
                rows = sum(1 for _ in sheet.iter_rows(min_row=1, max_row=limit + 1))
            if rows > limit:
                size = f"{rows:,} rows" if rows > limit + 1 else f"more than {limit:,} rows"
                raise WorkbookTooLargeError(
                    f"Sheet '{sheet.title}' has {size} (limit {limit:,}). "
                    "Delete the empty formatted rows below the data and upload again."
                )
    finally:
        book.close()
        if hasattr(source, "seek"):
            source.seek(0)


def _close_owned_workbook(workbook: pd.ExcelFile | None) -> None:
    """Close an ExcelFile this module opened (releases the Windows file lock)."""
    if workbook is None:
        return
    try:
        workbook.close()
    except Exception:  # pragma: no cover - closing must never mask a parse result
        logger.debug("Failed to close workbook handle", exc_info=True)


def _as_excel(source: WorkbookSource) -> pd.ExcelFile:
    if isinstance(source, pd.ExcelFile):
        return source
    check_workbook_row_counts(source)
    if hasattr(source, "seek"):
        source.seek(0)
    return pd.ExcelFile(source)


# --- Data-sheet reads that keep TRUE/FALSE cells ---------------------------------
# ``pd.read_excel`` infers an int/float column when the other cells are numeric,
# and even with ``dtype=object`` it folds a bool onto an equal number already
# seen in the column (``1`` then ``TRUE`` reads as two ``1``s), so a TRUE/FALSE
# typed in a number column silently became 1 / 0 before the row models could
# reject it (``models.reject_bool_number``). Every pandas Excel reader turns a
# native boolean cell into a Python bool in its cell grid (openpyxl ``'b'``,
# xlrd ``XL_CELL_BOOLEAN``, calamine, odf ``boolean``, pyxlsb), so the grid is
# marked while pandas reads it — one pass, whatever header / blank rows pandas
# skips — and the marks are turned back into bools in the frame.
_BOOL_MARKS = {True: "\x00xs-bool:TRUE\x00", False: "\x00xs-bool:FALSE\x00"}
_MARK_TO_BOOL = {mark: value for value, mark in _BOOL_MARKS.items()}
# Readers whose cell grid carries booleans as Python bools.
BOOL_AWARE_EXCEL_ENGINES = frozenset({"openpyxl", "xlrd", "calamine", "odf", "pyxlsb"})
_BOOL_UNAVAILABLE_ATTR = "_xs_bool_detection_unavailable"
_SHEET_CACHE_ATTR = "_xs_data_sheet_frames"


def _mark_bool_cells(data: list) -> bool:
    """Replace every bool in a pandas cell grid by its mark (in place); True when any."""
    found = False
    for index, row in enumerate(data):
        if not any(isinstance(value, (bool, np.bool_)) for value in row):
            continue
        found = True
        data[index] = [
            _BOOL_MARKS[bool(value)] if isinstance(value, (bool, np.bool_)) else value
            for value in row
        ]
    return found


def _unmark(value: object) -> object:
    if isinstance(value, str) and "\x00xs-bool:" in value:
        exact = _MARK_TO_BOOL.get(value)
        if exact is not None:
            return exact
        # A deduplicated header ("TRUE.1"): pandas appended to the mark.
        for mark, flag in _MARK_TO_BOOL.items():
            value = value.replace(mark, str(flag))
    return value


def _restore_marked_bools(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [_unmark(name) for name in frame.columns]
    marks = list(_MARK_TO_BOOL)
    for position in range(frame.shape[1]):
        column = frame.iloc[:, position]
        if column.dtype != object:
            continue
        hits = column.isin(marks).to_numpy()
        if not hits.any():
            continue
        values = column.to_numpy(dtype=object, copy=True)
        values[hits] = [_MARK_TO_BOOL[value] for value in values[hits]]
        frame.isetitem(position, values)
    return frame


def _note_bool_detection_unavailable(workbook: pd.ExcelFile) -> None:
    engine = str(getattr(workbook, "engine", None) or "unknown")
    if getattr(workbook, _BOOL_UNAVAILABLE_ATTR, None) is None:
        logger.warning(
            "TRUE/FALSE cell detection is unavailable for the %s Excel reader; a TRUE/FALSE "
            "in a number column may read as 1 / 0",
            engine,
        )
    try:
        setattr(workbook, _BOOL_UNAVAILABLE_ATTR, engine)
    except Exception:  # pragma: no cover - an object that refuses attributes
        pass


def bool_cell_detection_note(workbook: pd.ExcelFile | None) -> str | None:
    """Import note when this workbook's reader could not report TRUE/FALSE cells."""
    engine = getattr(workbook, _BOOL_UNAVAILABLE_ATTR, None) if workbook is not None else None
    if not engine:
        return None
    return (
        f"TRUE/FALSE cells cannot be detected in this file type (read with the {engine} "
        "reader): a TRUE/FALSE typed in a number column may have been read as 1 / 0 instead "
        "of being flagged. Save the workbook as .xlsx to have such cells reported."
    )


def _read_marked_frame(
    workbook: pd.ExcelFile, sheet_name: str, header: int | None
) -> pd.DataFrame:
    reader = getattr(workbook, "_reader", None)
    get_sheet_data = getattr(reader, "get_sheet_data", None)
    engine = getattr(workbook, "engine", None)
    if not callable(get_sheet_data) or engine not in BOOL_AWARE_EXCEL_ENGINES:
        _note_bool_detection_unavailable(workbook)
        return pd.read_excel(workbook, sheet_name=sheet_name, header=header, dtype=object)
    marked = False

    def marking_get_sheet_data(*args: object, **kwargs: object) -> list:
        nonlocal marked
        data = get_sheet_data(*args, **kwargs)
        if isinstance(data, list):
            marked = _mark_bool_cells(data) or marked
        return data

    reader.get_sheet_data = marking_get_sheet_data  # type: ignore[union-attr]
    try:
        frame = pd.read_excel(workbook, sheet_name=sheet_name, header=header, dtype=object)
    finally:
        reader.__dict__.pop("get_sheet_data", None)
    return _restore_marked_bools(frame) if marked else frame


def read_data_sheet(
    workbook: pd.ExcelFile, sheet_name: str, *, header: int | None = 0
) -> pd.DataFrame:
    """One data sheet with every cell kept as the type Excel stored (TRUE/FALSE kept as bools).

    ``dtype=object`` keeps ints, floats, dates and text as they are (the row
    models coerce and validate each value); boolean cells survive as bools
    (see the note above). ``header=None`` reads a layout sheet such as Data
    Entry. Each sheet is read once per workbook: later reads (a probe, then
    the parse) get a copy of the cached frame. A reader that cannot report
    booleans records an import note (:func:`bool_cell_detection_note`).
    """
    cache = getattr(workbook, _SHEET_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        try:
            setattr(workbook, _SHEET_CACHE_ATTR, cache)
        except Exception:  # pragma: no cover - an object that refuses attributes
            pass
    key = (sheet_name, header)
    frame = cache.get(key)
    if frame is None:
        frame = _read_marked_frame(workbook, sheet_name, header)
        cache[key] = frame
    return frame.copy()


def _short_error(exc: Exception) -> str:
    """First useful line of a pydantic ValidationError, else str(exc)."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            first = errors()[0]
            field = ".".join(str(part) for part in first.get("loc", ()))
            message = str(first.get("msg", "")).removeprefix("Value error, ")
            return f"{field}: {message}" if field else message
        except Exception:  # pragma: no cover - defensive
            pass
    return str(exc)


# --- Plain-language row errors -------------------------------------------------
# Every message starts "<Sheet> row <n>" (the Excel row number) so users can jump
# straight to the cell; :func:`skipped_row_hole_ids` relies on that shape.

_NUMBER_ERROR_TYPES = frozenset(
    {"float_parsing", "float_type", "int_parsing", "int_type", "int_from_float", "finite_number"}
)
_ROW_PREFIX_RE = re.compile(
    r"^(?P<sheet>[A-Z][A-Za-z ]+?) row \d+(?: \((?P<paren>[^)]+)\))?[,:] (?P<rest>.*)$"
)
_UNKNOWN_HOLE_RE = re.compile(r"unknown hole_id '([^']*)'")
_LABEL_COLOUR_RE = re.compile(r"label_color must be green, red, black or orange \(got '(.*)'\)")


def _is_blank_cell(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def _cell_text(value: object, limit: int = 30) -> str:
    if isinstance(value, float) and value.is_integer():
        text = str(int(value))
    else:
        text = str(value).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _number_example(value: object) -> str:
    match = re.search(r"-?\d+(?:[.,]\d+)?", str(value))
    return match.group(0).replace(",", ".") if match else "4"


def _first_error(exc: Exception) -> tuple[str, str, object, str]:
    """(field, pydantic error type, offending input, message) for *exc*."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            first = errors()[0]
            field = ".".join(str(part) for part in first.get("loc", ()))
            message = str(first.get("msg", "")).removeprefix("Value error, ")
            return field, str(first.get("type", "")), first.get("input"), message
        except Exception:  # pragma: no cover - defensive
            pass
    return "", "value_error", None, str(exc)


def _row_where(sheet: str, row_num: int, hole_id: object) -> str:
    where = f"{sheet} row {row_num}"
    if not _is_blank_cell(hole_id):
        where += f" ({_cell_text(hole_id, 20)})"
    return where


# Lab results reported as text. Non-detects ("<10", "ND (0.5)") are accepted by
# _apply_detection below; what still reaches the row error is ">1000"-style text.
_NON_DETECT_RE = re.compile(r"^\s*([<>]\s*[\d.]+|n\.?\s*d\.?|bdl|not detected)\s*$", re.IGNORECASE)

# Non-detect text the Environmental ``value`` accepts: a "<" or ND-style word,
# optional limit words, then the reporting detection limit (RDL) and an
# optional unit: "<5", "< 5", "<RDL 5", "<RDL=0.5", "ND (0.5)", "ND<0.5",
# "n.d. 0.5", "BDL 2", "<0.5 mg/L".
_ND_PREFIX_RE = re.compile(
    r"^\s*(?:<|n\.?\s*d\.?(?![a-z])|ndl(?![a-z])|bdl(?![a-z])|not\s+detected(?![a-z])"
    r"|below\s+detection(?:\s+limit)?(?![a-z]))",
    re.IGNORECASE,
)
_ND_LIMIT_WORDS_RE = re.compile(
    r"^(?:[\s<(=:,]|(?:rdl|mdl|edl|dl|lod|loq|rl|pql)(?![a-z]))*", re.IGNORECASE
)
_ND_NUMBER_RE = re.compile(r"^(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?", re.IGNORECASE)
_ND_UNIT_RE = re.compile(r"^[a-z\u00b5\u03bc%/][a-z0-9\u00b5\u03bc%/. ]{0,14}$", re.IGNORECASE)

# Optional ``qualifier`` (or ``detected``) column: lab-style flags. The
# qualifier column accepts lab flags too (``chemistry_qc.classify_lab_qualifier``:
# U-family = non-detect; J, B, E ... = detected; R = rejected, plotted with a
# Validate warning; unrecognised text = kept with a Validate warning).
_QUALIFIER_NON_DETECT = frozenset(
    {"<", "nd", "n.d.", "n.d", "ndl", "u", "bdl", "not detected", "non-detect", "<rdl", "<dl",
     "<mdl", "no", "n", "false", "0"}
)
_QUALIFIER_DETECTED = frozenset({"=", "detected", "yes", "y", "true", "1", "d"})


def _non_detect_value(raw: object) -> tuple[float, str] | None:
    """``(detection limit, unit text)`` for non-detect text such as ``"<5"``.

    ``None`` when *raw* is not non-detect text (numbers, ``">1000"``, junk);
    ``ValueError`` for a non-detect without a usable limit (bare ``"ND"``).
    """
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    prefix = _ND_PREFIX_RE.match(text)
    if not prefix:
        return None
    rest = text[prefix.end():]
    rest = rest[_ND_LIMIT_WORDS_RE.match(rest).end():]  # type: ignore[union-attr]
    shown = _cell_text(text)
    if not rest.strip():
        raise ValueError(
            f"value = '{shown}' — a non-detect needs its detection limit: enter it as "
            "'<0.5' (or 0.5 in value with < in the qualifier column)."
        )
    number = _ND_NUMBER_RE.match(rest)
    if not number:
        raise ValueError(
            f"value = '{shown}' — write a non-detect as '<' and the detection limit, "
            "e.g. <0.5 (use a dot for decimals)."
        )
    tail = rest[number.end():].strip().rstrip(")").strip()
    if tail and not _ND_UNIT_RE.match(tail):
        raise ValueError(
            f"value = '{shown}' — write a non-detect as '<' and the detection limit, "
            "e.g. <0.5 (use a dot for decimals)."
        )
    return float(number.group(0)), tail


def _qualifier_text(raw: object) -> str:
    """Cell text of a qualifier / detected cell ("" when blank)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    if isinstance(raw, bool):
        return str(raw).casefold()
    if isinstance(raw, (int, float)):
        return f"{raw:g}"
    return str(raw).strip()


def _detect_flag(raw: object, column: str) -> bool | None:
    """``True`` detected / ``False`` non-detect from a qualifier cell; ``None`` if blank.

    The ``qualifier`` column also takes lab flags (``classify_lab_qualifier``):
    R-qualified and unrecognised text return ``None`` (the value text decides;
    Validate warns). The ``detected`` column stays a strict yes / no.
    """
    if isinstance(raw, bool):
        return raw
    text = _qualifier_text(raw).casefold()
    if not text:
        return None
    compact = " ".join(text.split())
    # In the qualifier column a bare "N" is the lab flag (tentatively
    # identified, detected); in the detected column it means "no".
    lab_flag_n = column == "qualifier" and compact == "n"
    if not lab_flag_n and (
        compact in _QUALIFIER_NON_DETECT or compact.replace(" ", "") in _QUALIFIER_NON_DETECT
    ):
        return False
    if compact in _QUALIFIER_DETECTED:
        return True
    if column == "qualifier":
        from chemistry_qc import classify_lab_qualifier

        kind, rejected = classify_lab_qualifier(compact)
        if kind == "non_detect":
            return False
        if kind == "detected" and not rejected:
            return True
        return None
    raise ValueError(
        f"{column} = '{_cell_text(raw)}' — use < (or ND) for a non-detect, or leave it "
        "blank for a measured value."
    )


def _apply_detection(payload: dict) -> None:
    """Fold non-detect text in ``value`` and the qualifier column into ``value`` + ``detected``.

    ``value`` keeps the detection limit as a number; ``detected`` is False for a
    non-detect. A qualifier that contradicts the value text is an error, as is a
    unit in the value text that disagrees with the ``unit`` column.
    """
    flag: bool | None = None
    qualifier_text = ""
    for column in ("qualifier", "detected"):
        if column not in payload:
            continue
        raw = payload.pop(column)
        if column == "qualifier":
            # Kept on the reading: Validate warns on R / unrecognised flags.
            qualifier_text = _qualifier_text(raw)
        column_flag = _detect_flag(raw, column)
        if column_flag is None:
            continue
        if flag is not None and flag != column_flag:
            raise ValueError(
                "qualifier and detected disagree — fill one of them (< or ND for a non-detect)."
            )
        flag = column_flag
    if qualifier_text:
        payload["qualifier"] = qualifier_text
    non_detect = _non_detect_value(payload.get("value"))
    if non_detect is None:
        payload["detected"] = True if flag is None else flag
        return
    shown = _cell_text(payload.get("value"))
    if flag is True:
        raise ValueError(
            f"value = '{shown}' is a non-detect but the qualifier says detected — "
            "clear the qualifier or enter the measured number."
        )
    limit, unit_text = non_detect
    payload["value"] = limit
    payload["detected"] = False
    if unit_text:
        from chemistry_qc import unit_key

        unit = str(payload.get("unit") or "").strip()
        if not unit:
            payload["unit"] = unit_text
        elif unit_key(unit) != unit_key(unit_text):
            raise ValueError(
                f"value = '{shown}' is in {unit_text} but unit = '{unit}' — use one unit "
                "(values are not converted)."
            )


def _row_error(sheet: str, row_num: int, exc: Exception, *, hole_id: object = None) -> str:
    """One actionable sentence for a rejected row: sheet, Excel row, bad value, fix."""
    field, kind, value, message = _first_error(exc)
    where = _row_where(sheet, row_num, hole_id)
    lowered = message.lower()

    unknown = _UNKNOWN_HOLE_RE.search(message)
    if unknown:
        return _unknown_hole_error(sheet, row_num, unknown.group(1))

    colour = _LABEL_COLOUR_RE.search(message)
    if colour:
        return (
            f"{where}, label_color = '{_cell_text(colour.group(1))}' — use green, red, black "
            "or orange, or leave it blank."
        )
    if "label_color 'blue' is reserved" in message:
        return (
            f"{where}, label_color = 'blue' — blue is reserved for groundwater; "
            "use green, red, black or orange."
        )

    if message.startswith(("value = ", "qualifier = ", "detected = ", "date = ")):
        # Non-detect / qualifier / date messages already name the cell and the fix.
        return f"{where}, {message}"
    if "is a TRUE/FALSE cell" in message:
        return f"{where}, {message}."

    whole_number = "must be an integer" in lowered or "whole number" in lowered
    if field and (
        kind in _NUMBER_ERROR_TYPES or "must be a finite number" in lowered or whole_number
    ):
        if _is_blank_cell(value):
            return f"{where}: {field} is empty — enter a number (e.g. {_number_example(value)})."
        if whole_number:
            return f"{where}, {field} = '{_cell_text(value)}' — enter a whole number (e.g. 2)."
        text = _cell_text(value)
        if field == "value" and _NON_DETECT_RE.match(text):
            # Lab non-detects: keep the number for plotting, the text for the label.
            return (
                f"{where}, value = '{text}' — enter the number in value (e.g. 10) and "
                f"put '{text}' in the value_label column to print it as written."
            )
        return (
            f"{where}, {field} = '{text}' — enter a number only "
            f"(e.g. {_number_example(value)})."
        )

    if "to_depth must be" in lowered and "from_depth" in lowered:
        top = value.get("from_depth") if isinstance(value, dict) else None
        base = value.get("to_depth") if isinstance(value, dict) else None
        if top is not None and base is not None:
            return (
                f"{where}: from_depth {_cell_text(top)} is deeper than to_depth "
                f"{_cell_text(base)} — swap them or fix the typo."
            )
        return f"{where}: from_depth is deeper than to_depth — swap them or fix the typo."

    if "non-negative" in lowered:
        subject = field or "depth"
        shown = f" ({_cell_text(value)})" if field and not _is_blank_cell(value) else ""
        if not field and isinstance(value, dict):
            for name in ("from_depth", "to_depth", "depth"):
                number = value.get(name)
                if isinstance(number, (int, float)) and number < 0:
                    subject, shown = name, f" ({_cell_text(number)})"
                    break
        return f"{where}: {subject}{shown} can't be negative — enter 0 or more."

    if kind == "missing" or "is required" in lowered or "is missing" in lowered:
        return f"{where}: {field or 'a required value'} is empty — fill it in."
    if field and lowered.endswith(" is blank"):
        return f"{where}: {field} is empty — fill it in."

    if "formula" in lowered:
        return (
            f"{where}: {field or 'a cell'} holds a formula — paste it as a value "
            "(Paste Special → Values)."
        )

    if field and not message.startswith(field):
        return f"{where}, {field}: {message}"
    return f"{where}: {message}"


def hole_id_match_key(hole_id: object) -> str:
    """Case- and whitespace-insensitive key (``MW-01`` ~ ``mw-01`` ~ ``MW -01``)."""
    return re.sub(r"\s+", "", str(hole_id)).casefold()


def _case_variant_of(hole_id: object, known_ids: Iterable[str] | None) -> str | None:
    """The Collars ID that *hole_id* matches only when case / spaces are ignored."""
    if not known_ids:
        return None
    key = hole_id_match_key(hole_id)
    return next((known for known in sorted(known_ids) if hole_id_match_key(known) == key), None)


def _unknown_hole_error(
    sheet: str,
    row_num: int,
    hole_id: object,
    known_ids: Iterable[str] | None = None,
    *,
    detail: str = "",
) -> str:
    """Row references a hole that is not in Collars (keeps the Excel row number).

    When Collars has the same ID in another case / spacing (``mw-01`` vs
    ``MW-01``) the message names it: IDs must match exactly.
    """
    hole = _cell_text(hole_id, 20)
    variant = _case_variant_of(hole_id, known_ids)
    fix = (
        f"Collars has {variant} — hole IDs must match exactly (case and spaces)."
        if variant is not None
        else "add it to Collars or fix the spelling."
    )
    return f"{sheet} row {row_num}: {hole} is in {sheet} but not in Collars{detail} — {fix}"


def _missing_columns_message(sheet: str, missing: object) -> str:
    names = ", ".join(sorted(str(name) for name in missing))  # type: ignore[union-attr]
    noun = "column" if "," not in names else "columns"
    return (
        f"The {sheet} sheet is missing the {names} {noun} — add the header to row 1 "
        "(see Help → Workbook & data entry)."
    )


def skipped_row_hole_ids(errors: Sequence[str]) -> dict[str, set[str]]:
    """Hole IDs whose rows were skipped on import, keyed by sheet name.

    Reads the "<Sheet> row <n> (<hole>)" / "<Sheet> row <n>: <hole> is in …"
    shapes produced by :func:`_row_error` and :func:`_unknown_hole_error`.
    """
    holes: dict[str, set[str]] = {}
    for message in errors:
        match = _ROW_PREFIX_RE.match(str(message))
        if not match:
            continue
        sheet = match.group("sheet")
        hole = match.group("paren")
        if hole is None:
            rest = match.group("rest")
            marker = f" is in {sheet} but not in Collars"
            if marker in rest:
                hole = rest.split(marker, 1)[0]
        if hole:
            holes.setdefault(sheet, set()).add(hole.strip())
    return holes


_COLUMN_SPELLINGS = {"label_colour": "label_color", "yellow_max": "orange_max"}

# pandas reads an Excel error cell (#N/A, #DIV/0! ...) and a formula saved
# without its result (files written by scripts / web tools) as a blank cell.
_EMPTY_CELL_ERROR_RE = re.compile(
    r"^(?P<where>(?P<sheet>[A-Z][A-Za-z ]+?) row (?P<row>\d+)(?: \([^)]*\))?): "
    r"(?P<field>[a-z_]+) is empty — "
)
_MAX_EXPLAINED_CELLS = 200


def _header_key(value: object) -> str:
    name = re.sub(r"\s+", "_", str(value).strip().lower())
    return _COLUMN_SPELLINGS.get(name, name)


def _cells_by_row(sheet: object, wanted: dict[int, set[int]]) -> dict[tuple[int, int], object]:
    """``{(row, col): value}`` for the wanted 1-based rows / 0-based columns (one pass)."""
    found: dict[tuple[int, int], object] = {}
    last = max(wanted)
    for row_num, values in enumerate(sheet.iter_rows(values_only=True), start=1):  # type: ignore[attr-defined]
        if row_num > last:
            break
        for col in wanted.get(row_num, ()):
            found[(row_num, col)] = values[col] if col < len(values) else None
    return found


def _explain_empty_cells(
    errors: list[str],
    workbook: pd.ExcelFile | None,
    native_sheets: dict[str, str],
) -> list[str]:
    """Reword "<field> is empty" row errors whose cell is not actually empty.

    An Excel error value (``#N/A``, ``#DIV/0!``) and a formula whose result was
    never cached both reach pandas as a blank. The cached-value workbook names
    the error; formulas are read from a second, formula-mode load (only when
    such an error exists). Anything unexpected leaves the message unchanged.
    """
    targets: dict[str, list[tuple[int, re.Match[str]]]] = {}
    for index, message in enumerate(errors):
        match = _EMPTY_CELL_ERROR_RE.match(message)
        if match and match.group("sheet") in native_sheets:
            targets.setdefault(match.group("sheet"), []).append((index, match))
    book = getattr(workbook, "book", None) if workbook is not None else None
    if not targets or book is None or not hasattr(book, "sheetnames"):
        return errors
    explained = list(errors)
    raw_book = None
    try:
        for label, items in targets.items():
            sheet_name = DataParser._find_sheet(list(book.sheetnames), native_sheets[label])
            if sheet_name is None:
                continue
            sheet = book[sheet_name]
            header = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True), ())
            columns = {_header_key(name): col for col, name in enumerate(header) if name is not None}
            wanted: dict[int, set[int]] = {}
            located: list[tuple[int, re.Match[str], int, int]] = []
            for index, match in items[:_MAX_EXPLAINED_CELLS]:
                col = columns.get(match.group("field"))
                if col is None:
                    continue
                row = int(match.group("row"))
                wanted.setdefault(row, set()).add(col)
                located.append((index, match, row, col))
            if not wanted:
                continue
            cached = _cells_by_row(sheet, wanted)
            formula_rows: dict[int, set[int]] = {}
            for _index, _match, row, col in located:
                if cached.get((row, col)) is None:
                    formula_rows.setdefault(row, set()).add(col)
            raw: dict[tuple[int, int], object] = {}
            if formula_rows:
                if raw_book is None:
                    raw_book = _load_formula_workbook(workbook)
                if raw_book is not None and sheet_name in raw_book.sheetnames:
                    raw = _cells_by_row(raw_book[sheet_name], formula_rows)
            for index, match, row, col in located:
                where, field = match.group("where"), match.group("field")
                value = cached.get((row, col))
                if isinstance(value, str) and value.strip().startswith("#"):
                    explained[index] = (
                        f"{where}: {field} holds the Excel error value {value.strip()} — fix "
                        "the formula in Excel or type the number."
                    )
                    continue
                formula = raw.get((row, col))
                if isinstance(formula, str) and formula.startswith("="):
                    explained[index] = (
                        f"{where}: {field} is a formula whose result was not cached (the file "
                        "was saved by a program that does not calculate formulas) — open and "
                        "save it in Excel, or paste the values (Paste Special → Values)."
                    )
    except Exception:  # pragma: no cover - explanation must never break parsing
        logger.debug("Could not inspect blank cells", exc_info=True)
        return errors
    finally:
        if raw_book is not None:
            raw_book.close()
    return explained


def _load_formula_workbook(workbook: pd.ExcelFile | None) -> object | None:
    """The same workbook loaded with formulas (not cached values), or ``None``."""
    source = getattr(workbook, "_io", None)
    if source is None:
        return None
    import openpyxl

    if hasattr(source, "seek"):
        source.seek(0)
    try:
        return openpyxl.load_workbook(source, read_only=True, data_only=False)
    except Exception:
        return None


def _blank_cell(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return isinstance(value, str) and not value.strip()


def _thickness_from_depth_to_product(
    depth_to_product: object,
    water_depth: float | None,
    product_thickness: object,
) -> float | None:
    """Apparent LNAPL thickness from an optional Water ``depth_to_product`` cell.

    ``depth`` is the depth to WATER (oil/water interface) and
    ``depth_to_product`` the depth to the air/product interface, both on the
    same datum (bgs or btoc): thickness = depth − depth_to_product. Returns
    ``None`` when the cell is blank (``product_thickness_m`` is then used as
    given). A thickness given as well must agree within 5 mm.
    """
    if _blank_cell(depth_to_product):
        return None
    try:
        dtp = float(depth_to_product)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"depth_to_product must be a number (got {depth_to_product!r})") from exc
    if not math.isfinite(dtp) or dtp < 0:
        raise ValueError("depth_to_product must be a finite, non-negative depth")
    if water_depth is None:
        raise ValueError(
            "depth_to_product needs the water depth in depth (oil/water interface); "
            "with elevation_masl give product_thickness_m instead"
        )
    if dtp > water_depth:
        raise ValueError(
            f"depth_to_product {dtp:g} m is below the water depth {water_depth:g} m — depth is "
            "the depth to water (oil/water interface), depth_to_product the shallower depth to "
            "the top of the product"
        )
    thickness = water_depth - dtp
    if not _blank_cell(product_thickness):
        try:
            given = float(product_thickness)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"product_thickness_m must be a number (got {product_thickness!r})"
            ) from exc
        if abs(given - thickness) > 0.005:
            raise ValueError(
                f"product_thickness_m {given:g} m disagrees with depth − depth_to_product "
                f"({thickness:.3f} m) — give one or the other"
            )
    return thickness


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Same collapse as FormatDetector, so a header it accepts cannot fail here.
    renamed = {col: re.sub(r"\s+", "_", str(col).strip().lower()) for col in df.columns}
    # The product's own text says "colour"; accept that spelling in headers too.
    renamed = {col: _COLUMN_SPELLINGS.get(name, name) for col, name in renamed.items()}
    df = df.rename(columns=renamed)
    if "hole_id" in df.columns:
        # A blank cell makes pandas read numeric IDs as float: 1 -> 1.0 -> "1.0",
        # which then matches nothing in Collars.
        # Built as an object Series: ``Series.map`` re-infers the dtype, so a
        # sheet read with dtype=object (1, <blank>, 2) came back as floats.
        ids = df["hole_id"]
        if isinstance(ids, pd.Series):
            df["hole_id"] = pd.Series(
                [_integral_id_to_text(value) for value in ids], index=df.index, dtype=object
            )
        else:  # two headers collapsed to hole_id: the duplicate check reports it
            df["hole_id"] = ids.map(_integral_id_to_text)
    return df


def _integral_id_to_text(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) and not pd.isna(value) and value.is_integer():
        return str(int(value))
    return value


class DataParser:
    """Reads Excel workbooks and validates rows against borehole schemas."""

    COLLARS_SHEET = "Collars"
    LITHOLOGY_SHEET = "Lithology"
    WATER_SHEET = "Water"
    SCREENS_SHEET = "Screens"
    GRADIENTS_SHEET = "Gradients"
    SECTIONS_SHEET = "Sections"

    def parse_file(
        self,
        source: WorkbookSource,
        *,
        collars_df: pd.DataFrame | None = None,
        lithology_df: pd.DataFrame | None = None,
        collars_sheet: str | None = None,
        lithology_sheet: str | None = None,
        lithology_aliases: dict[str, str] | None = None,
        water_df: pd.DataFrame | None = None,
        workbook: pd.ExcelFile | None = None,
    ) -> ParseResult:
        water_frame = None
        environmental_frame = None
        screens_frame = None
        gradients_frame = None
        # Row errors from these sheet labels point at native workbook tabs
        # (label -> real sheet name), so blank-cell errors can be explained.
        native_sheets: dict[str, str] = {}
        data_entry_precedence_warnings: list[str] = []
        data_entry = None
        supplied_geology = collars_df is not None and lithology_df is not None
        _owned_workbook = None
        if workbook is None:
            try:
                workbook = _as_excel(source)
                _owned_workbook = workbook if workbook is not source else None
            except Exception:
                if not supplied_geology:
                    raise
                workbook = None
        try:
            if supplied_geology:
                collars_frame = _normalize_columns(collars_df)
                lithology_frame = _normalize_columns(lithology_df)
            else:
                data_entry_sheet = self._find_sheet(workbook.sheet_names, "Data Entry")
                collars_name = collars_sheet or self.COLLARS_SHEET
                lithology_name = lithology_sheet or self.LITHOLOGY_SHEET
                has_native = (
                    self._find_sheet(workbook.sheet_names, collars_name) is not None
                    and self._find_sheet(workbook.sheet_names, lithology_name) is not None
                )
                if data_entry_sheet:
                    from workbook_template import parse_data_entry_sheet

                    entry_frame = read_data_sheet(workbook, data_entry_sheet, header=None)
                    data_entry = parse_data_entry_sheet(entry_frame)
                    if not data_entry.environmental.empty:
                        environmental_frame = _normalize_columns(data_entry.environmental)

                # The compatibility Data Entry sheet stands in for Collars +
                # Lithology only when it holds geology rows. The template's
                # copy holds Project metadata only, so a template with its
                # Lithology tab deleted loaded as 0 holes with "health OK".
                data_entry_has_geology = (
                    data_entry is not None
                    and not data_entry.collars.empty
                    and not data_entry.lithology.empty
                )
                if data_entry_has_geology and not has_native:
                    collars_frame = _normalize_columns(data_entry.collars)
                    lithology_frame = _normalize_columns(data_entry.lithology)
                else:
                    missing = [
                        sheet
                        for sheet in (collars_name, lithology_name)
                        if self._find_sheet(workbook.sheet_names, sheet) is None
                    ]
                    if missing:
                        raise ValueError(f"Missing required sheet(s): {', '.join(missing)}")

                    # Read by the sheet's actual name: detection is case-insensitive.
                    collars_tab = self._find_sheet(workbook.sheet_names, collars_name)
                    lithology_tab = self._find_sheet(workbook.sheet_names, lithology_name)
                    collars_frame = _normalize_columns(read_data_sheet(workbook, collars_tab))
                    lithology_frame = _normalize_columns(read_data_sheet(workbook, lithology_tab))
                    native_sheets["Collars"] = self._find_sheet(workbook.sheet_names, collars_name)
                    native_sheets["Lithology"] = self._find_sheet(
                        workbook.sheet_names, lithology_name
                    )
                    # Empty header-only native tabs: fall back to Data Entry geology when present.
                    if (
                        data_entry is not None
                        and not data_entry.collars.empty
                        and not data_entry.lithology.empty
                        and not self._frame_has_hole_rows(collars_frame)
                    ):
                        collars_frame = _normalize_columns(data_entry.collars)
                        lithology_frame = _normalize_columns(data_entry.lithology)
                        native_sheets.clear()

            collars, collar_errors = self._parse_collars(collars_frame)
            lithologies, lithology_errors = self._parse_lithologies(lithology_frame, collars)

            if data_entry is not None:
                # Data Entry overlay rows win over native optional tabs, but only
                # when they reference the winning collar universe — otherwise the
                # native tabs stay in effect (and no misleading warning is emitted).
                valid_ids = {collar.hole_id for collar in collars}

                def _data_entry_matches(frame: pd.DataFrame, label: str) -> bool:
                    ids = {
                        _clean_text(value)
                        for value in frame.get("hole_id", [])
                        if not self._blank_hole_id(value)
                    }
                    if not valid_ids or bool(ids & valid_ids):
                        return True
                    data_entry_precedence_warnings.append(
                        f"Data Entry {label} rows ignored: hole_ids do not match Collars "
                        f"({', '.join(sorted(ids)) or 'blank'})."
                    )
                    return False

                if not data_entry.water.empty:
                    candidate = _normalize_columns(data_entry.water)
                    if _data_entry_matches(candidate, "water"):
                        water_frame = candidate
                        if self._native_optional_sheet_has_rows(workbook, self.WATER_SHEET):
                            data_entry_precedence_warnings.append(
                                "Native Water sheet was ignored because Data Entry supplied water rows."
                            )
                if not data_entry.screens.empty:
                    candidate = _normalize_columns(data_entry.screens)
                    if _data_entry_matches(candidate, "screen"):
                        screens_frame = candidate
                        if self._native_optional_sheet_has_rows(workbook, self.SCREENS_SHEET):
                            data_entry_precedence_warnings.append(
                                "Native Screens sheet was ignored because Data Entry supplied screen rows."
                            )
                if not data_entry.gradients.empty:
                    candidate = _normalize_columns(data_entry.gradients)
                    if _data_entry_matches(candidate, "gradient"):
                        gradients_frame = candidate
                        if self._native_optional_sheet_has_rows(workbook, self.GRADIENTS_SHEET):
                            data_entry_precedence_warnings.append(
                                "Native Gradients sheet was ignored because Data Entry supplied gradient rows."
                            )
            water_levels: list[WaterLevel] = []
            water_errors: list[str] = []
            deviation_readings: list[DeviationReading] = []
            correlation_overrides: list[CorrelationOverride] = []
            environmental_readings: list[EnvironmentalReading] = []
            environmental_errors: list[str] = []
            faults: list[Fault] = []
            unconformities: list[Unconformity] = []
            screen_intervals: list[ScreenInterval] = []
            vertical_gradients: list[VerticalGradient] = []
            screen_errors: list[str] = []
            gradient_errors: list[str] = []
            structure_errors: list[str] = []
            section_specs: list[WorkbookSectionSpec] = []
            if water_df is not None:
                water_levels, water_errors = self._parse_water_levels(
                    _normalize_columns(water_df), collars
                )
            elif workbook is not None:
                if water_frame is None:
                    water_sheet = self._find_sheet(workbook.sheet_names, self.WATER_SHEET)
                    if water_sheet:
                        water_frame = _normalize_columns(read_data_sheet(workbook, water_sheet))
                        native_sheets["Water"] = water_sheet
                if water_frame is not None and not water_frame.empty:
                    water_levels, water_errors = self._parse_water_levels(water_frame, collars)
                if environmental_frame is not None and not environmental_frame.empty:
                    environmental_readings, environmental_errors = self._parse_environmental_dataframe(
                        environmental_frame, collars
                    )
                else:
                    environmental_readings, environmental_errors = self._parse_environmental_sheet(
                        workbook, collars
                    )
                    native_sheets["Environmental"] = "Environmental"
                field_readings, field_errors = self._parse_field_data_environmental_sheet(
                    workbook, collars
                )
                environmental_readings.extend(field_readings)
                environmental_errors.extend(field_errors)
                if screens_frame is not None and not screens_frame.empty:
                    screen_intervals, screen_errors = self._parse_screens_dataframe(screens_frame, collars)
                else:
                    screen_intervals, screen_errors = self._parse_screens_sheet(workbook, collars)
                    native_sheets["Screens"] = self.SCREENS_SHEET
                if gradients_frame is not None and not gradients_frame.empty:
                    vertical_gradients, gradient_errors = self._parse_gradients_dataframe(
                        gradients_frame, collars
                    )
                else:
                    vertical_gradients, gradient_errors = self._parse_gradients_sheet(workbook, collars)
                    native_sheets["Gradients"] = self.GRADIENTS_SHEET
                deviation_readings = self._parse_deviation_sheet(workbook, structure_errors, collars)
                native_sheets["Deviations"] = "Deviations"
                correlation_overrides = self._parse_correlation_sheet(workbook)
                faults = self._parse_fault_sheet(workbook, structure_errors)
                unconformities = self._parse_unconformity_sheet(workbook, structure_errors)

            if workbook is not None:
                section_specs = self._parse_sections_sheet(workbook, collars, structure_errors)

            if lithology_aliases:
                lithologies = self._apply_lithology_aliases(lithologies, lithology_aliases)

            errors = tuple(
                _explain_empty_cells(
                    collar_errors
                    + lithology_errors
                    + water_errors
                    + screen_errors
                    + gradient_errors
                    + environmental_errors
                    + structure_errors,
                    workbook,
                    native_sheets,
                )
                + data_entry_precedence_warnings
            )
            if errors:
                for message in errors:
                    logger.warning("Parse issue: %s", message)

            return ParseResult(
                collars=tuple(collars),
                lithologies=tuple(lithologies),
                errors=errors,
                water_levels=tuple(water_levels),
                screen_intervals=tuple(screen_intervals),
                vertical_gradients=tuple(vertical_gradients),
                deviation_readings=tuple(deviation_readings),
                correlation_overrides=tuple(correlation_overrides),
                faults=tuple(faults),
                unconformities=tuple(unconformities),
                environmental_readings=tuple(environmental_readings),
                section_specs=tuple(section_specs),
            )
        finally:
            _close_owned_workbook(_owned_workbook)

    def _apply_lithology_aliases(
        self,
        lithologies: list[Lithology],
        aliases: dict[str, str],
    ) -> list[Lithology]:
        if not aliases:
            return lithologies
        from ai_quality import normalize_lithology_code

        updated: list[Lithology] = []
        for lithology in lithologies:
            new_code = normalize_lithology_code(lithology.lithology_code, aliases)
            if new_code == lithology.lithology_code:
                updated.append(lithology)
            else:
                updated.append(
                    Lithology(
                        hole_id=lithology.hole_id,
                        from_depth=lithology.from_depth,
                        to_depth=lithology.to_depth,
                        lithology_code=new_code,
                        hatch_pattern=lithology.hatch_pattern,
                        unit_order=lithology.unit_order,
                    )
                )
        return updated

    @staticmethod
    def _frame_has_hole_rows(frame: pd.DataFrame) -> bool:
        if frame is None or frame.empty or "hole_id" not in frame.columns:
            return False
        for value in frame["hole_id"]:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            text = str(value).strip()
            if text and text.lower() != "nan":
                return True
        return False

    def _native_optional_sheet_has_rows(self, workbook: pd.ExcelFile, sheet_name: str) -> bool:
        """True when a native optional tab exists and has at least one hole_id row."""
        sheet = self._find_sheet(workbook.sheet_names, sheet_name)
        if not sheet:
            return False
        frame = _normalize_columns(read_data_sheet(workbook, sheet))
        return self._frame_has_hole_rows(frame)

    @staticmethod
    def _find_sheet(sheet_names: list[str], target: str) -> str | None:
        target_key = target.strip().lower()
        for name in sheet_names:
            if name.strip().lower() == target_key:
                return name
        return None

    _failed_collar_ids: set[str] = set()

    def _collar_failed(self, hole_id: object) -> bool:
        return str(hole_id).strip() in self._failed_collar_ids

    @staticmethod
    def _row_has_values(row: object, fields: tuple[str, ...]) -> bool:
        """True when any named cell holds data (so the row is not just padding)."""
        for field in fields:
            value = getattr(row, field, None)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            if str(value).strip():
                return True
        return False

    @staticmethod
    def _blank_hole_id(value: object) -> bool:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return True
        text = str(value).strip()
        return text == "" or text.lower() == "nan"

    def _parse_collars(self, df: pd.DataFrame) -> tuple[list[Collar], list[str]]:
        errors: list[str] = []
        collars: list[Collar] = []
        seen_ids: set[str] = set()

        missing_cols = COLLAR_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(_missing_columns_message("Collars", missing_cols))
        optional_cols = {
            "elevation_datum",
            "inclination_deg",
            "azimuth_deg",
            "stick_up_m",
        } & set(df.columns)
        self._failed_collar_ids = set()

        for index, row in enumerate(df.itertuples(index=True)):
            row_num = int(row.Index) + 2
            if self._blank_hole_id(row.hole_id):
                if self._row_has_values(row, ("easting", "northing", "elevation", "total_depth")):
                    errors.append(
                        f"Collars row {row_num}: hole_id is blank (a formula without a "
                        "saved value reads as blank)"
                    )
                continue
            try:
                payload: dict[str, object] = {
                    "hole_id": row.hole_id,
                    "easting": row.easting,
                    "northing": row.northing,
                    "elevation": row.elevation,
                    "total_depth": row.total_depth,
                }
                if "elevation_datum" in optional_cols:
                    payload["elevation_datum"] = row.elevation_datum
                if "inclination_deg" in optional_cols:
                    payload["inclination_deg"] = row.inclination_deg
                if "azimuth_deg" in optional_cols:
                    payload["azimuth_deg"] = row.azimuth_deg
                if "stick_up_m" in optional_cols:
                    payload["stick_up_m"] = row.stick_up_m
                collar = Collar.model_validate(payload)
            except Exception as exc:
                errors.append(_row_error("Collars", row_num, exc, hole_id=getattr(row, "hole_id", None)))
                # Child rows of a rejected collar get no "unknown hole_id"
                # cascade: the collar error already explains them.
                self._failed_collar_ids.add(str(row.hole_id).strip())
                continue

            if collar.hole_id in seen_ids:
                errors.append(
                    f"Collars row {row_num}: duplicate hole_id {collar.hole_id} — "
                    "keep one row or rename the hole"
                )
                continue

            seen_ids.add(collar.hole_id)
            collars.append(collar)

        return collars, errors

    def _parse_lithologies(
        self,
        df: pd.DataFrame,
        collars: list[Collar],
    ) -> tuple[list[Lithology], list[str]]:
        errors: list[str] = []
        lithologies: list[Lithology] = []
        valid_hole_ids = {collar.hole_id for collar in collars}

        missing_cols = LITHOLOGY_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(_missing_columns_message("Lithology", missing_cols))

        optional_cols = LITHOLOGY_OPTIONAL_COLUMNS & set(df.columns)
        for index, row in enumerate(df.itertuples(index=True)):
            row_num = int(row.Index) + 2
            if self._blank_hole_id(row.hole_id):
                if self._row_has_values(row, ("from_depth", "to_depth", "lithology_code")):
                    errors.append(f"Lithology row {row_num}: hole_id is blank")
                continue
            try:
                payload: dict[str, object] = {
                    "hole_id": row.hole_id,
                    "from_depth": row.from_depth,
                    "to_depth": row.to_depth,
                    "lithology_code": row.lithology_code,
                }
                if "hatch_pattern" in optional_cols:
                    payload["hatch_pattern"] = row.hatch_pattern
                if "unit_order" in optional_cols:
                    payload["unit_order"] = row.unit_order
                lithology = Lithology.model_validate(payload)
            except Exception as exc:
                errors.append(_row_error("Lithology", row_num, exc, hole_id=getattr(row, "hole_id", None)))
                continue

            if valid_hole_ids and lithology.hole_id not in valid_hole_ids:
                if not self._collar_failed(lithology.hole_id):
                    errors.append(
                        _unknown_hole_error("Lithology", row_num, lithology.hole_id, valid_hole_ids)
                    )
                continue

            lithologies.append(lithology)

        return lithologies, errors

    def _parse_water_levels(
        self,
        df: pd.DataFrame,
        collars: list[Collar],
    ) -> tuple[list[WaterLevel], list[str]]:
        errors: list[str] = []
        levels: list[WaterLevel] = []
        valid_hole_ids = {collar.hole_id for collar in collars}
        collar_by_id = {collar.hole_id: collar for collar in collars}
        columns = set(df.columns)
        if "hole_id" not in columns:
            raise ValueError(_missing_columns_message("Water", {"hole_id"}))
        if not columns.intersection(WATER_VALUE_COLUMNS | WATER_STATUS_COLUMNS):
            raise ValueError(
                "Water sheet requires hole_id plus depth, elevation_masl, or status (dry/nm)"
            )

        for index, row in enumerate(df.itertuples(index=True)):
            row_num = int(row.Index) + 2
            if self._blank_hole_id(row.hole_id):
                continue
            try:
                hole_id = _clean_text(row.hole_id)
                status_raw = getattr(row, "status", None)
                status_text = (
                    ""
                    if status_raw is None or (isinstance(status_raw, float) and pd.isna(status_raw))
                    else str(status_raw).strip().lower()
                )
                is_non_measured = status_text in {
                    "dry",
                    "d",
                    "nm",
                    "n/m",
                    "not measured",
                    "not_measured",
                    "ns",
                    "not sampled",
                    "not_sampled",
                }
                depth_raw = getattr(row, "depth", None)
                masl_raw = getattr(row, "elevation_masl", None)
                has_depth = depth_raw is not None and not pd.isna(depth_raw) and str(depth_raw).strip() != ""
                has_masl = masl_raw is not None and not pd.isna(masl_raw) and str(masl_raw).strip() != ""
                if has_depth and has_masl:
                    raise ValueError("provide depth or elevation_masl, not both")
                if not has_depth and not has_masl and not is_non_measured:
                    raise ValueError("depth or elevation_masl is required")
                elevation_masl: float | None = None
                if has_masl:
                    # Checked before the depth conversion so the row error names
                    # elevation_masl, not the derived depth.
                    elevation_masl = check_bounded_number(
                        float(reject_bool_number(masl_raw, "elevation_masl")),  # type: ignore[arg-type]
                        "elevation_masl",
                    )
                    collar = collar_by_id.get(hole_id)
                    if collar is None:
                        if self._collar_failed(hole_id):
                            continue
                        errors.append(
                            _unknown_hole_error("Water", row_num, hole_id, valid_hole_ids)
                        )
                        continue
                    # A head above the collar is a (flowing) artesian condition,
                    # not an error: the head keeps its elevation and Validate
                    # reports it as information. depth stays non-negative.
                    depth = max(0.0, collar.elevation - elevation_masl)
                elif has_depth:
                    depth = float(reject_bool_number(depth_raw, "depth"))  # type: ignore[arg-type]
                else:
                    depth = 0.0
                payload: dict[str, object] = {
                    "hole_id": hole_id,
                    "depth": depth,
                    "elevation_masl": elevation_masl,
                }
                for col in (
                    "series_id",
                    "series_label",
                    "color",
                    "marker",
                    "connect_group",
                    "status",
                    "depth_reference",
                    "date",
                    "product_thickness_m",
                    "product_density",
                ):
                    if hasattr(row, col):
                        payload[col] = getattr(row, col)
                thickness_from_dtp = _thickness_from_depth_to_product(
                    getattr(row, "depth_to_product", None),
                    depth if has_depth else None,
                    payload.get("product_thickness_m"),
                )
                if thickness_from_dtp is not None:
                    payload["product_thickness_m"] = thickness_from_dtp
                level = WaterLevel.model_validate(payload)
            except Exception as exc:
                errors.append(_row_error("Water", row_num, exc, hole_id=getattr(row, "hole_id", None)))
                continue
            if valid_hole_ids and level.hole_id not in valid_hole_ids:
                if not self._collar_failed(level.hole_id):
                    errors.append(_unknown_hole_error("Water", row_num, level.hole_id, valid_hole_ids))
                continue
            levels.append(level)

        return levels, errors

    def _parse_screens_sheet(
        self,
        workbook: pd.ExcelFile,
        collars: list[Collar],
    ) -> tuple[list[ScreenInterval], list[str]]:
        sheet = self._find_sheet(workbook.sheet_names, self.SCREENS_SHEET)
        if not sheet:
            return [], []
        frame = _normalize_columns(read_data_sheet(workbook, sheet))
        return self._parse_screens_dataframe(frame, collars)

    def _parse_gradients_sheet(
        self,
        workbook: pd.ExcelFile,
        collars: list[Collar],
    ) -> tuple[list[VerticalGradient], list[str]]:
        sheet = self._find_sheet(workbook.sheet_names, self.GRADIENTS_SHEET)
        if not sheet:
            return [], []
        frame = _normalize_columns(read_data_sheet(workbook, sheet))
        return self._parse_gradients_dataframe(frame, collars)

    def _parse_screens_dataframe(
        self,
        frame: pd.DataFrame,
        collars: list[Collar],
    ) -> tuple[list[ScreenInterval], list[str]]:
        if not SCREEN_COLUMNS.issubset(frame.columns):
            return [], []
        valid_hole_ids = {collar.hole_id for collar in collars}
        intervals: list[ScreenInterval] = []
        errors: list[str] = []
        for row_num, row in enumerate(frame.itertuples(index=False), start=2):
            if self._blank_hole_id(getattr(row, "hole_id", None)):
                continue
            try:
                interval = ScreenInterval.model_validate(row._asdict())
            except Exception as exc:
                errors.append(_row_error("Screens", row_num, exc, hole_id=getattr(row, "hole_id", None)))
                continue
            if valid_hole_ids and interval.hole_id not in valid_hole_ids:
                if not self._collar_failed(interval.hole_id):
                    errors.append(_unknown_hole_error("Screens", row_num, interval.hole_id, valid_hole_ids))
                continue
            intervals.append(interval)
        return intervals, errors

    def _parse_gradients_dataframe(
        self,
        frame: pd.DataFrame,
        collars: list[Collar],
    ) -> tuple[list[VerticalGradient], list[str]]:
        if not GRADIENT_COLUMNS.issubset(frame.columns):
            return [], []
        valid_hole_ids = {collar.hole_id for collar in collars}
        gradients: list[VerticalGradient] = []
        errors: list[str] = []
        for row_num, row in enumerate(frame.itertuples(index=False), start=2):
            if self._blank_hole_id(getattr(row, "hole_id", None)):
                continue
            try:
                gradient = VerticalGradient.model_validate(row._asdict())
            except Exception as exc:
                errors.append(_row_error("Gradients", row_num, exc, hole_id=getattr(row, "hole_id", None)))
                continue
            if valid_hole_ids and gradient.hole_id not in valid_hole_ids:
                if not self._collar_failed(gradient.hole_id):
                    errors.append(_unknown_hole_error("Gradients", row_num, gradient.hole_id, valid_hole_ids))
                continue
            gradients.append(gradient)
        return gradients, errors

    def _parse_deviation_sheet(
        self,
        workbook: pd.ExcelFile,
        errors: list[str],
        collars: Sequence[Collar] | None = None,
    ) -> list[DeviationReading]:
        """Deviations rows; errors carry the Excel row number like other sheets.

        Rows for holes not in Collars are not used by the section (the survey
        has no collar to hang from): one message per hole names them.
        """
        sheet = self._find_sheet(workbook.sheet_names, "Deviations")
        if not sheet:
            return []
        frame = _normalize_columns(read_data_sheet(workbook, sheet))
        required = {"hole_id", "depth", "inclination_deg", "azimuth_deg"}
        if not required.issubset(frame.columns):
            return []
        valid_hole_ids = {collar.hole_id for collar in collars or ()}
        readings: list[DeviationReading] = []
        orphan_rows: dict[str, list[int]] = {}
        for row_num, row in enumerate(frame.itertuples(index=False), start=2):
            if self._blank_hole_id(row.hole_id):
                if self._row_has_values(row, ("depth", "inclination_deg", "azimuth_deg")):
                    errors.append(f"Deviations row {row_num}: hole_id is blank")
                continue
            try:
                reading = DeviationReading.model_validate(
                    {
                        "hole_id": row.hole_id,
                        "depth": row.depth,
                        "inclination_deg": row.inclination_deg,
                        "azimuth_deg": row.azimuth_deg,
                    }
                )
            except Exception as exc:
                errors.append(_row_error("Deviations", row_num, exc, hole_id=row.hole_id))
                continue
            if valid_hole_ids and reading.hole_id not in valid_hole_ids:
                if not self._collar_failed(reading.hole_id):
                    orphan_rows.setdefault(reading.hole_id, []).append(row_num)
                continue
            readings.append(reading)
        for hole_id, rows in orphan_rows.items():
            noun = "row" if len(rows) == 1 else "rows"
            errors.append(
                _unknown_hole_error(
                    "Deviations",
                    rows[0],
                    hole_id,
                    valid_hole_ids,
                    detail=f" (survey of {len(rows)} {noun} not used)",
                )
            )
        return readings

    def _parse_correlation_sheet(self, workbook: pd.ExcelFile) -> list[CorrelationOverride]:
        sheet = self._find_sheet(workbook.sheet_names, "Correlations")
        if not sheet:
            return []
        frame = _normalize_columns(read_data_sheet(workbook, sheet))
        required = {"left_hole_id", "right_hole_id", "left_unit_order", "right_unit_order"}
        if not required.issubset(frame.columns):
            return []
        overrides: list[CorrelationOverride] = []
        for row in frame.itertuples(index=False):
            try:
                overrides.append(CorrelationOverride.model_validate(row._asdict()))
            except Exception as exc:
                logger.warning("Skipping correlation override row: %s", exc)
        return overrides

    def _parse_environmental_dataframe(
        self,
        frame: pd.DataFrame,
        collars: list[Collar],
    ) -> tuple[list[EnvironmentalReading], list[str]]:
        columns = set(frame.columns)
        if not {"hole_id", "parameter", "value"}.issubset(columns):
            return [], []
        has_depth = "depth" in columns
        has_interval = "from_depth" in columns and "to_depth" in columns
        if not has_depth and not has_interval:
            return [], []
        valid_hole_ids = {collar.hole_id for collar in collars}
        readings: list[EnvironmentalReading] = []
        errors: list[str] = []
        for row_num, row in enumerate(frame.itertuples(index=False), start=2):
            if self._blank_hole_id(getattr(row, "hole_id", None)):
                continue
            payload = row._asdict()
            # Blank text cells read back as NaN (e.g. from the cleaned export);
            # the model expects strings.
            for text_field in ("unit", "value_label", "label_color"):
                raw = payload.get(text_field)
                if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                    payload[text_field] = ""
            try:
                has_point = (
                    has_depth
                    and "depth" in payload
                    and not pd.isna(payload["depth"])
                    and str(payload["depth"]).strip() != ""
                )
                has_interval_values = (
                    has_interval
                    and "from_depth" in payload
                    and "to_depth" in payload
                    and not pd.isna(payload["from_depth"])
                    and not pd.isna(payload["to_depth"])
                    and str(payload["from_depth"]).strip() != ""
                    and str(payload["to_depth"]).strip() != ""
                )
                if has_point and has_interval_values:
                    raise ValueError("provide depth or from_depth/to_depth, not both")
                if has_point:
                    payload["from_depth"] = None
                    payload["to_depth"] = None
                elif has_interval_values:
                    payload["depth"] = None
                else:
                    raise ValueError("depth or from_depth and to_depth are required")
                value_raw = payload.get("value")
                if value_raw is None or pd.isna(value_raw) or str(value_raw).strip() == "":
                    raise ValueError("value is required")
                if payload.get("value_label") in ("", None) or pd.isna(payload.get("value_label")):
                    payload["value_label"] = ""
                # "<5" / "ND (5)" in value, or a qualifier column: value keeps the
                # detection limit, detected=False.
                _apply_detection(payload)
                reading = EnvironmentalReading.model_validate(payload)
            except Exception as exc:
                errors.append(_row_error("Environmental", row_num, exc, hole_id=getattr(row, "hole_id", None)))
                continue
            if valid_hole_ids and reading.hole_id not in valid_hole_ids:
                if not self._collar_failed(reading.hole_id):
                    errors.append(
                        _unknown_hole_error("Environmental", row_num, reading.hole_id, valid_hole_ids)
                    )
                continue
            readings.append(reading)
        return readings, errors

    def _parse_environmental_sheet(
        self,
        workbook: pd.ExcelFile,
        collars: list[Collar],
    ) -> tuple[list[EnvironmentalReading], list[str]]:
        sheet = self._find_sheet(workbook.sheet_names, "Environmental")
        if not sheet:
            return [], []
        frame = _normalize_columns(read_data_sheet(workbook, sheet))
        return self._parse_environmental_dataframe(frame, collars)

    def _parse_field_data_environmental_dataframe(
        self,
        frame: pd.DataFrame,
        collars: list[Collar],
    ) -> tuple[list[EnvironmentalReading], list[str]]:
        """Map Field Data OVA/EC columns into EnvironmentalReading rows (append path)."""
        # Lazy import avoids circular load: ingestion → models → parsing.
        from ingestion import parse_depth_interval

        columns = set(frame.columns)
        hole_col = next(
            (name for name in ("label", "hole_id", "hole", "bh_id") if name in columns),
            None,
        )
        if hole_col is None:
            return [], []
        has_ova = "ova" in columns
        has_ec = "ec" in columns
        if not has_ova and not has_ec:
            return [], []
        depth_col = next(
            (name for name in ("depth", "depth_interval") if name in columns),
            None,
        )
        # Optional label colour: per-parameter columns win, else a shared
        # ``label_color`` applies to both OVA and EC (``label_colour`` is
        # already folded by _normalize_columns). Absent → "" as before.
        shared_color_col = "label_color" if "label_color" in columns else None
        ova_color_col = "ova_label_color" if "ova_label_color" in columns else shared_color_col
        ec_color_col = "ec_label_color" if "ec_label_color" in columns else shared_color_col

        def _color_value(payload: dict[str, object], column: str | None) -> str:
            if column is None:
                return ""
            raw = payload.get(column)
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                return ""
            return _clean_text(raw)

        valid_hole_ids = {collar.hole_id for collar in collars}
        readings: list[EnvironmentalReading] = []
        errors: list[str] = []
        for row_num, row in enumerate(frame.itertuples(index=False), start=2):
            payload = row._asdict()
            hole_raw = payload.get(hole_col)
            if self._blank_hole_id(hole_raw):
                continue
            hole_id = _clean_text(hole_raw)

            ova_value: float | None = None
            ec_value: float | None = None
            for column in ("ova", "ec"):
                try:
                    # A TRUE/FALSE cell is not a reading of 1 / 0.
                    reject_bool_number(payload.get(column), column)
                except ValueError as exc:
                    errors.append(_row_error("Field Data", row_num, exc, hole_id=hole_id))
                    payload[column] = None
            if has_ova:
                parsed_ova = pd.to_numeric(payload.get("ova"), errors="coerce")
                if not pd.isna(parsed_ova):
                    ova_value = float(parsed_ova)
            if has_ec:
                parsed_ec = pd.to_numeric(payload.get("ec"), errors="coerce")
                if not pd.isna(parsed_ec):
                    ec_value = float(parsed_ec)
            if ova_value is None and ec_value is None:
                continue

            if depth_col is None:
                errors.append(
                    f"Field Data row {row_num}: missing depth interval for hole_id '{hole_id}'"
                )
                continue
            try:
                from_depth, to_depth = parse_depth_interval(payload.get(depth_col))
            except Exception as exc:
                errors.append(_row_error("Field Data", row_num, exc, hole_id=hole_id))
                continue

            if valid_hole_ids and hole_id not in valid_hole_ids:
                errors.append(_unknown_hole_error("Field Data", row_num, hole_id, valid_hole_ids))
                continue

            if ova_color_col is not None and ova_color_col == ec_color_col:
                # Shared colour column: validate once so a bad cell yields a
                # single error for the row rather than one per parameter.
                try:
                    EnvironmentalReading.normalise_label_color(
                        _color_value(payload, ova_color_col)
                    )
                except Exception as exc:
                    errors.append(_row_error("Field Data", row_num, exc, hole_id=hole_id))
                    continue

            if ova_value is not None:
                try:
                    readings.append(
                        EnvironmentalReading.model_validate(
                            {
                                "hole_id": hole_id,
                                "parameter": "OVA",
                                "value": ova_value,
                                "from_depth": from_depth,
                                "to_depth": to_depth,
                                "unit": "ppm",
                                "label_color": _color_value(payload, ova_color_col),
                            }
                        )
                    )
                except Exception as exc:
                    errors.append(_row_error("Field Data", row_num, exc, hole_id=hole_id))
            if ec_value is not None:
                try:
                    readings.append(
                        EnvironmentalReading.model_validate(
                            {
                                "hole_id": hole_id,
                                "parameter": "EC",
                                "value": ec_value,
                                "from_depth": from_depth,
                                "to_depth": to_depth,
                                "unit": "",
                                "label_color": _color_value(payload, ec_color_col),
                            }
                        )
                    )
                except Exception as exc:
                    errors.append(_row_error("Field Data", row_num, exc, hole_id=hole_id))
        return readings, errors

    def _parse_field_data_environmental_sheet(
        self,
        workbook: pd.ExcelFile,
        collars: list[Collar],
    ) -> tuple[list[EnvironmentalReading], list[str]]:
        from ingestion import get_cached_field_data_frame

        frame = get_cached_field_data_frame(workbook)
        if frame is None:
            return [], []
        return self._parse_field_data_environmental_dataframe(_normalize_columns(frame), collars)

    def _parse_fault_sheet(self, workbook: pd.ExcelFile, errors: list[str]) -> list[Fault]:
        sheet = self._find_sheet(workbook.sheet_names, "Faults")
        if not sheet:
            return []
        frame = _normalize_columns(read_data_sheet(workbook, sheet))
        if not {"name", "x_profile", "elevation"}.issubset(frame.columns):
            return []
        frame = frame.dropna(how="all", subset=["name", "x_profile", "elevation"])
        faults: dict[str, list[tuple[float, float]]] = {}
        for row in frame.itertuples(index=False):
            if pd.isna(row.x_profile) or pd.isna(row.elevation):
                errors.append(f"Faults sheet: blank x_profile/elevation for fault {row.name!r}")
                continue
            faults.setdefault(str(row.name), []).append((float(row.x_profile), float(row.elevation)))
        return [
            Fault(name=name, trace_points=points)
            for name, points in faults.items()
            if len(points) >= 2
        ]

    def _parse_unconformity_sheet(
        self, workbook: pd.ExcelFile, errors: list[str]
    ) -> list[Unconformity]:
        sheet = self._find_sheet(workbook.sheet_names, "Unconformities")
        if not sheet:
            return []
        frame = _normalize_columns(read_data_sheet(workbook, sheet))
        if not {"name", "x_profile", "elevation"}.issubset(frame.columns):
            return []
        frame = frame.dropna(how="all", subset=["name", "x_profile", "elevation"])
        surfaces: dict[str, list[tuple[float, float]]] = {}
        for row in frame.itertuples(index=False):
            if pd.isna(row.x_profile) or pd.isna(row.elevation):
                errors.append(
                    f"Unconformities sheet: blank x_profile/elevation for surface {row.name!r}"
                )
                continue
            surfaces.setdefault(str(row.name), []).append(
                (float(row.x_profile), float(row.elevation))
            )
        return [
            Unconformity(name=name, elevation_profile=points)
            for name, points in surfaces.items()
            if len(points) >= 2
        ]

    @staticmethod
    def _split_section_hole_ids(raw: object) -> tuple[str, ...]:
        """Split hole_ids on comma, semicolon, or → (also accepts ASCII ->)."""
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return ()
        text = str(raw).strip()
        if not text:
            return ()
        normalized = text.replace("→", ",").replace("->", ",").replace(";", ",")
        return tuple(part.strip() for part in normalized.split(",") if part.strip())

    def _parse_sections_sheet(
        self,
        workbook: pd.ExcelFile,
        collars: list[Collar],
        errors: list[str] | None = None,
    ) -> list[WorkbookSectionSpec]:
        """Parse optional Sections sheet; log-and-skip bad rows (not folded into errors)."""
        sheet = self._find_sheet(workbook.sheet_names, self.SECTIONS_SHEET)
        if not sheet:
            return []
        frame = _normalize_columns(read_data_sheet(workbook, sheet))
        if "section_label" not in frame.columns or "hole_ids" not in frame.columns:
            logger.warning(
                "Sections sheet missing required columns section_label and hole_ids; skipped"
            )
            return []
        collar_ids = {collar.hole_id for collar in collars}
        specs: list[WorkbookSectionSpec] = []
        for row_num, row in enumerate(frame.itertuples(index=False), start=2):
            label_raw = getattr(row, "section_label", None)
            holes_raw = getattr(row, "hole_ids", None)
            if (
                label_raw is None
                or (isinstance(label_raw, float) and pd.isna(label_raw))
                or str(label_raw).strip() == ""
            ) and (
                holes_raw is None
                or (isinstance(holes_raw, float) and pd.isna(holes_raw))
                or str(holes_raw).strip() == ""
            ):
                continue
            hole_ids = self._split_section_hole_ids(holes_raw)
            try:
                spec = WorkbookSectionSpec.model_validate(
                    {"label": label_raw, "hole_ids": hole_ids}
                )
            except Exception as exc:
                message = f"Sections row {row_num}: {_short_error(exc)}"
                logger.warning("Skipping %s", message)
                if errors is not None:
                    errors.append(message)
                continue
            if collar_ids:
                unknown = [hole_id for hole_id in spec.hole_ids if hole_id not in collar_ids]
                if unknown:
                    message = (
                        f"Sections row {row_num} ({spec.label}): unknown collar(s) "
                        f"{', '.join(unknown)} — check the spelling against Collars"
                    )
                    logger.warning("Skipping %s", message)
                    if errors is not None:
                        errors.append(message)
                    continue
            specs.append(spec)
        return specs
