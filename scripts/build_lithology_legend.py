"""Build data/bh_log_lithology_legend.json from the client legend workbook.

Source of truth: ``data/BH Log Lithology Legend.xlsx`` (client legend 261023),
Sheet1 columns F-I ("New 2026 Colours"): F = group colour hex, H = lithology,
I = hatch ("solid", "Sand hatch .", "Silty hatch /", "Loam hatch +",
"Gravel hatch"). A hex in column F applies to its own row and every row below
it until the next hex (the group's base colour); column G's cell fill must
agree. Columns A-C ("Original Colours") are the superseded scheme and are
ignored.

Topsoil is on the 261023 legend sheet but not in the workbook, so it comes
from :data:`EXPLICIT_ENTRIES`, together with the legacy codes older workbooks
still use (never shown on the 261023 legend, kept so they do not turn grey).

    python scripts/build_lithology_legend.py           # rewrite the JSON
    python scripts/build_lithology_legend.py --check   # exit 1 if it is stale

``tests/test_lithology_scheme_261023.py`` fails when the committed JSON and
this script's output differ.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "BH Log Lithology Legend.xlsx"
OUTPUT = ROOT / "data" / "bh_log_lithology_legend.json"

SCHEME_ID = "261023"
SCHEMA_VERSION = 2

# Hatch families of the 261023 legend. ``hatch`` is the matplotlib hatch
# string the app draws (hatch_patterns.py replaces those characters with the
# template marks); ``cad`` is the AutoCAD pattern on the CAD legend
# (Cross_Section_Litho_Legend_261002_CB.dwg, layer XSEC-LITH-HATCH / -GRVL-HATCH,
# colour #323232).
HATCH_FAMILIES: dict[str, dict[str, object]] = {
    "solid": {"hatch": "", "label": "solid", "cad": {"pattern": "SOLID"}},
    "sandy": {
        "hatch": ".",
        "label": "Sand hatch .",
        "cad": {"pattern": "DOTS", "scale": 0.75, "angle_deg": 0},
    },
    "silty": {
        "hatch": "/",
        "label": "Silty hatch /",
        "cad": {"pattern": "DASH", "scale": 0.5, "angle_deg": 45},
    },
    "loam": {
        "hatch": "+",
        "label": "Loam hatch +",
        "cad": {"pattern": "CROSS", "scale": 0.35, "angle_deg": 0},
    },
    "gravel": {
        "hatch": "O",
        "label": "Gravel hatch",
        "cad": {"pattern": "GRAVEL", "scale": 0.15, "angle_deg": 0},
    },
}

# Swatch order on the 261023 legend sheet (two rows of 13, read left to right).
LEGEND_ORDER: tuple[str, ...] = (
    "Clay", "Silt", "Sand", "Loamy Sand", "Sandy Clay", "Silty Clay", "Organics",
    "Topsoil", "Loam", "Silty Loam", "Clay Loam", "Sandy Clay Loam", "Silty Clay Loam",
    "Fill", "Gravel", "Sand and Gravel", "Drilling Waste", "Other", "No Recovery",
    "Bentonite", "Coal", "Refuse", "Siltstone", "Sandstone", "Mudstone",
)  # fmt: skip

# Entries that are not rows of the workbook.
EXPLICIT_ENTRIES: dict[str, dict[str, object]] = {
    # On the 261023 legend sheet, not in the workbook: colour from the legend
    # PDF / DWG layer XSEC-LITH-TOPSOIL (same as before).
    "Topsoil": {"color": "#534230", "hatch_family": "solid", "source": "261023 legend sheet"},
    # Legacy codes: not on 261023; kept so older workbooks keep their colour.
    "Silty Sand": {"color": "#FFE39F", "hatch_family": "silty", "source": "legacy", "legacy": True},
    "Bedrock": {"color": "#4A4A4A", "hatch": "xxx", "source": "legacy", "legacy": True},
    "Limestone": {"color": "#E0E4E8", "hatch": "..", "source": "legacy", "legacy": True},
    "Shale": {"color": "#5C6B7A", "hatch": "\\\\", "source": "legacy", "legacy": True},
    "Flare Pit Material": {"color": "#6D4C41", "hatch": "**", "source": "legacy", "legacy": True},
    "Sand and Clay": {"color": "#E8B84A", "hatch": "/.", "source": "legacy", "legacy": True},
}

_HEX = re.compile(r"^#?([0-9A-Fa-f]{6})$")


def _hex(value: object) -> str | None:
    match = _HEX.match(str(value or "").strip())
    return f"#{match.group(1).upper()}" if match else None


def hatch_family(label: object) -> str:
    """Workbook hatch text -> family key ("Silty hatch /" -> "silty")."""
    text = str(label or "").strip().casefold()
    if text in {"", "solid", "none", "plain"}:
        return "solid"
    for family in ("sand", "silty", "loam", "gravel"):
        if text.startswith(family):
            return "sandy" if family == "sand" else family
    raise ValueError(f"Unknown hatch {label!r} in the lithology legend workbook")


def read_workbook_rows(path: Path = WORKBOOK) -> list[dict[str, object]]:
    """Lithology rows of the "New 2026 Colours" table (F-I), group colour filled down."""
    import openpyxl

    book = openpyxl.load_workbook(path, data_only=True)
    sheet = book.worksheets[0]
    header_row = None
    for row in sheet.iter_rows(min_col=6, max_col=9):
        if str(row[2].value or "").strip().casefold() == "lithology":
            header_row = row[0].row
            break
    if header_row is None:
        raise ValueError(f"No 'Lithology' header in columns F-I of {path}")
    rows: list[dict[str, object]] = []
    group_colour: str | None = None
    for row in sheet.iter_rows(min_row=header_row + 1, min_col=6, max_col=9):
        hex_cell, fill_cell, name_cell, hatch_cell = row
        name = str(name_cell.value or "").strip()
        colour = _hex(hex_cell.value)
        if colour:
            group_colour = colour
        if not name:
            continue
        if group_colour is None:
            raise ValueError(f"{name!r} (row {name_cell.row}) has no group colour above it")
        fill = fill_cell.fill
        if fill is not None and fill.fill_type == "solid":
            fill_hex = _hex(str(fill.fgColor.rgb or "")[-6:])
            if fill_hex and fill_hex != group_colour:
                raise ValueError(
                    f"{name!r} (row {name_cell.row}): column G fill {fill_hex} "
                    f"disagrees with group colour {group_colour}"
                )
        rows.append(
            {
                "name": name,
                "color": group_colour,
                "hatch_family": hatch_family(hatch_cell.value),
                "row": name_cell.row,
            }
        )
    return rows


def build_scheme(path: Path = WORKBOOK) -> dict[str, object]:
    """The versioned scheme written to ``data/bh_log_lithology_legend.json``."""
    lithologies: dict[str, dict[str, object]] = {}
    for row in read_workbook_rows(path):
        family = str(row["hatch_family"])
        lithologies[str(row["name"])] = {
            "color": row["color"],
            "hatch": HATCH_FAMILIES[family]["hatch"],
            "hatch_family": family,
            "source": f"{path.name} row {row['row']}",
        }
    for name, entry in EXPLICIT_ENTRIES.items():
        if name in lithologies:
            raise ValueError(f"{name!r} is in the workbook; drop it from EXPLICIT_ENTRIES")
        resolved = dict(entry)
        family = resolved.get("hatch_family")
        if family is not None:
            resolved["hatch"] = HATCH_FAMILIES[str(family)]["hatch"]
        lithologies[name] = {
            key: resolved[key]
            for key in ("color", "hatch", "hatch_family", "source", "legacy")
            if key in resolved
        }
    missing = [name for name in LEGEND_ORDER if name not in lithologies]
    if missing:
        raise ValueError(f"Legend 261023 lithologies missing from the scheme: {missing}")
    legacy = [name for name, entry in lithologies.items() if entry.get("legacy")]
    unordered = [name for name in lithologies if name not in LEGEND_ORDER and name not in legacy]
    if unordered:
        raise ValueError(f"Add {unordered} to LEGEND_ORDER (or mark them legacy)")
    return {
        "scheme_id": SCHEME_ID,
        "schema_version": SCHEMA_VERSION,
        "title": f"Cross Section Litho Legend {SCHEME_ID}",
        "generated_by": "scripts/build_lithology_legend.py",
        "source_workbook": f"data/{path.name}",
        "hatch_line_color": "#323232",
        "hatch_families": HATCH_FAMILIES,
        "legend_order": list(LEGEND_ORDER),
        "legacy_codes": legacy,
        "lithologies": {name: lithologies[name] for name in [*LEGEND_ORDER, *legacy]},
    }


def render_json(scheme: dict[str, object]) -> str:
    return json.dumps(scheme, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workbook", type=Path, default=WORKBOOK)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true", help="exit 1 if the output is out of date")
    args = parser.parse_args(argv)
    text = render_json(build_scheme(args.workbook))
    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if current != text:
            print(f"{args.output} is out of date; run scripts/build_lithology_legend.py")
            return 1
        print(f"{args.output} is up to date (scheme {SCHEME_ID})")
        return 0
    args.output.write_text(text, encoding="utf-8")
    count = len(json.loads(text)["lithologies"])
    print(f"Wrote {args.output} (scheme {SCHEME_ID}, {count} lithologies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
