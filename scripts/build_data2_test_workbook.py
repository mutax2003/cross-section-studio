"""Build native test workbooks from ``data/Data2`` chlorides + BH Log lithology names.

Sources (Advantage Phase 2 / 09-36-055-02 W4M):
  - ``data/Data2/Cross_Section_Chlorides.xlsx`` — Cl (mg/L) at From–To depth intervals
  - ``data/Data2/BH Log Lithology Legend.xlsx`` — lithology display names
  - Reference PDFs Fig 6 A–A' / Fig 7 B–B' (layout only; not parsed)

Writes:
  - ``data/Data2/Cross_Section_Test_AA_Lithology_Chlorides.xlsx``
  - ``data/Data2/Cross_Section_Test_BB_Lithology_Chlorides.xlsx``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from advantage_p2_reference.chlorides import (  # noqa: E402
    normalize_compound_hole_id,
    parse_chloride_value,
)
from advantage_p2_reference.transects import ADVANTAGE_P2_TRANSECTS  # noqa: E402

DATA2 = ROOT / "data" / "Data2"
CHLORIDES = DATA2 / "Cross_Section_Chlorides.xlsx"
LEGEND = DATA2 / "BH Log Lithology Legend.xlsx"

_UNIT = "mg/L"
_PARAMETER = "Chloride"
_ELEVATION_MASL = 635.0

# Chloride sheet columns after header=1 (A–A' left block, B–B' right block).
_TRANSECT_INTERVAL_COLUMNS: dict[str, tuple[str, str, str, str]] = {
    "A_A": ("Borehole", "Cl", "From", "To"),
    "B_B": ("Borehole.1", "Cl.1", "From.1", "To.1"),
}

_OUTPUT_BY_TRANSECT: dict[str, Path] = {
    "A_A": DATA2 / "Cross_Section_Test_AA_Lithology_Chlorides.xlsx",
    "B_B": DATA2 / "Cross_Section_Test_BB_Lithology_Chlorides.xlsx",
}

_SECTION_META: dict[str, dict[str, str]] = {
    "A_A": {
        "section_title": "A - A' WITH CHLORIDE INTERVALS",
        "transect_start": "A / NORTHWEST",
        "transect_end": "A' / SOUTHEAST",
        "figure": "Fig 6",
    },
    "B_B": {
        "section_title": "B - B' WITH CHLORIDE INTERVALS",
        "transect_start": "B / NORTHWEST",
        "transect_end": "B' / SOUTHEAST",
        "figure": "Fig 7",
    },
}


def _legend_lithology_names() -> list[str]:
    frame = pd.read_excel(LEGEND, header=None)
    # Row 1 is header Area/Colour/Lithology/RGB; names start at row 2
    names: list[str] = []
    for raw in frame.iloc[2:, 2].tolist():
        text = str(raw).strip()
        if text and text.lower() != "nan":
            names.append(text)
    return names


def _chloride_intervals(path: Path, transect_id: str) -> pd.DataFrame:
    """Parse From–To chloride intervals for one transect side of the sheet."""
    if transect_id not in _TRANSECT_INTERVAL_COLUMNS:
        raise ValueError(f"unknown transect_id: {transect_id}")
    hole_col, cl_col, from_col, to_col = _TRANSECT_INTERVAL_COLUMNS[transect_id]
    frame = pd.read_excel(path, sheet_name="Chloride", header=1)
    # Use dict rows — dotted names like Borehole.1 are not valid itertuples attrs.
    rows: list[dict[str, object]] = []
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
        rows.append(
            {
                "hole_id": hole_id,
                "parameter": _PARAMETER,
                "value": value,
                "from_depth": from_depth,
                "to_depth": to_depth,
                "unit": _UNIT,
                "value_label": value_label,
            }
        )
    return pd.DataFrame(rows)


def _lithology_stack(hole_id: str, total_depth: float, legend: list[str]) -> list[dict[str, object]]:
    """Synthetic BH Log–style sticks covering 0 → total_depth (not from PDFs)."""

    def pick(*candidates: str) -> str:
        for name in candidates:
            if name in legend:
                return name
        return candidates[0]

    topsoil = pick("Organics", "Loam")
    sand = pick("Sand", "Loamy Sand", "Sand and Gravel")
    transition = pick("Clay Loam", "Silty Clay Loam", "Sandy Clay Loam")
    clay = pick("Clay", "Silty Clay", "Sandy Clay")
    silt = pick("Silt", "Silty Loam")

    seed = sum(ord(ch) for ch in hole_id) % 3
    shallow_sand = 1.5 + 0.5 * seed
    mid = 4.0 + seed
    deep = min(total_depth, 9.0 + seed)

    layers = [
        (0.0, min(0.4, total_depth), topsoil),
        (min(0.4, total_depth), min(shallow_sand, total_depth), sand),
        (min(shallow_sand, total_depth), min(mid, total_depth), transition),
        (min(mid, total_depth), min(deep, total_depth), clay),
    ]
    if deep < total_depth:
        layers.append((deep, total_depth, silt if seed else clay))

    rows: list[dict[str, object]] = []
    order = 1
    for from_depth, to_depth, code in layers:
        if to_depth <= from_depth + 1e-9:
            continue
        rows.append(
            {
                "hole_id": hole_id,
                "from_depth": round(from_depth, 2),
                "to_depth": round(to_depth, 2),
                "lithology_code": code,
                "unit_order": order,
            }
        )
        order += 1
    return rows


def build(transect_id: str = "A_A") -> Path:
    if transect_id not in ADVANTAGE_P2_TRANSECTS:
        raise ValueError(f"unknown transect_id: {transect_id}")
    if not CHLORIDES.is_file():
        raise FileNotFoundError(CHLORIDES)
    if not LEGEND.is_file():
        raise FileNotFoundError(LEGEND)

    legend = _legend_lithology_names()
    environmental = _chloride_intervals(CHLORIDES, transect_id)
    if environmental.empty:
        raise RuntimeError(f"No {transect_id} chloride intervals parsed from Data2 workbook")

    spec = ADVANTAGE_P2_TRANSECTS[transect_id]
    meta = _SECTION_META[transect_id]
    output = _OUTPUT_BY_TRANSECT[transect_id]

    hole_ids: list[str] = list(spec.hole_ids)
    for hole_id in environmental["hole_id"].unique():
        if hole_id not in hole_ids:
            hole_ids.append(str(hole_id))

    max_by_hole = environmental.groupby("hole_id")["to_depth"].max().to_dict()
    collars: list[dict[str, object]] = []
    lithology: list[dict[str, object]] = []
    eastings = list(spec.profile_eastings)
    while len(eastings) < len(hole_ids):
        eastings.append(eastings[-1] + 45.0)

    for hole_id, easting in zip(hole_ids, eastings, strict=True):
        total_depth = float(max(15.0, max_by_hole.get(hole_id, 15.0) + 1.0))
        collars.append(
            {
                "hole_id": hole_id,
                "easting": float(easting),
                "northing": 0.0,
                "elevation": _ELEVATION_MASL,
                "total_depth": total_depth,
            }
        )
        lithology.extend(_lithology_stack(hole_id, total_depth, legend))

    hole_set = {row["hole_id"] for row in collars}
    environmental = environmental[environmental["hole_id"].isin(hole_set)].reset_index(drop=True)

    project = pd.DataFrame(
        [
            {"field": "client_name", "value": "C-GROUP ENERGY INC."},
            {"field": "prepared_by", "value": "ECOVENTURE"},
            {"field": "project_number", "value": "100/09-36-055-02 W4M"},
            {"field": "section_title", "value": meta["section_title"]},
            {"field": "report_date", "value": "06/24/26"},
            {"field": "drawn_by", "value": "SL/JG"},
            {
                "field": "data_source",
                "value": "data/Data2 chlorides + BH Log lithology (synthetic sticks)",
            },
            {"field": "transect_start", "value": meta["transect_start"]},
            {"field": "transect_end", "value": meta["transect_end"]},
            {"field": "vertical_exaggeration", "value": "5"},
            {
                "field": "notes",
                "value": (
                    f"Test workbook from Data2 ({meta['figure']}). Chloride From–To "
                    "intervals from Cross_Section_Chlorides.xlsx. Lithology sticks are "
                    "synthetic using BH Log legend names (PDFs not digitized)."
                ),
            },
        ]
    )

    DATA2.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        project.to_excel(writer, sheet_name="Project", index=False)
        pd.DataFrame(collars).to_excel(writer, sheet_name="Collars", index=False)
        pd.DataFrame(lithology).to_excel(writer, sheet_name="Lithology", index=False)
        environmental.to_excel(writer, sheet_name="Environmental", index=False)
        pd.DataFrame(
            {
                "note": [
                    "Upload this workbook in Cross Section Studio (native Collars + Lithology).",
                    "Environmental sheet: Chloride (mg/L) with from_depth/to_depth intervals.",
                    f"Select holes in {transect_id.replace('_', '–')} order on Configure; "
                    "enable Chloride parameter plotting.",
                    f"Source chlorides: {CHLORIDES.name}",
                    f"Lithology names from: {LEGEND.name}",
                    "Reference figures: Fig 6 A–A' / Fig 7 B–B' PDFs in data/Data2 (not parsed).",
                ]
            }
        ).to_excel(writer, sheet_name="Instructions", index=False)

    return output


def build_all() -> list[Path]:
    return [build(transect_id) for transect_id in ("A_A", "B_B")]


def _smoke_ingest(path: Path) -> None:
    from ingestion import ingest_workbook

    parse_result, report = ingest_workbook(path)
    print(f"Wrote {path}")
    print(
        f"Ingest: {len(parse_result.collars)} holes, "
        f"{len(parse_result.lithologies)} lithology intervals, "
        f"{len(parse_result.environmental_readings)} chloride samples, "
        f"profile={report.profile_id}"
    )
    codes = sorted({lit.lithology_code for lit in parse_result.lithologies})
    print(f"Lithology codes: {', '.join(codes)}")
    print(f"Holes: {', '.join(c.hole_id for c in parse_result.collars)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transect",
        choices=("A_A", "B_B", "all"),
        default="all",
        help="Which transect workbook(s) to build (default: all)",
    )
    args = parser.parse_args(argv)

    paths = build_all() if args.transect == "all" else [build(args.transect)]
    for path in paths:
        _smoke_ingest(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
