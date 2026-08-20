"""Build native test workbooks from ``data/Data2`` chlorides + lithology.

Sources (Advantage Phase 2 / 09-36-055-02 W4M):
  - ``data/Data2/Cross_Section_Chlorides.xlsx`` — Cl (mg/kg) at From–To depth intervals
  - Optional ``--lithology`` CSV (hole_id, from_depth, to_depth, lithology_code)
  - Digitized PDF sticks when CSV absent (``advantage_p2_reference.lithology_digitized``)

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
from advantage_p2_reference.fixtures import load_lithology_csv  # noqa: E402
from advantage_p2_reference.lithology_digitized import (  # noqa: E402
    digitized_lithology,
    digitized_total_depth,
)
from advantage_p2_reference.transects import ADVANTAGE_P2_TRANSECTS  # noqa: E402

DATA2 = ROOT / "data" / "Data2"
CHLORIDES = DATA2 / "Cross_Section_Chlorides.xlsx"

_UNIT = "mg/kg"
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
        "transect_start": "A / WEST",
        "transect_end": "A' / EAST",
        "figure": "Fig 6",
    },
    "B_B": {
        "section_title": "B - B' WITH CHLORIDE INTERVALS",
        "transect_start": "B / SOUTH",
        "transect_end": "B' / NORTH",
        "figure": "Fig 7",
    },
}


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


def _lithology_rows_for_hole(
    transect_id: str,
    hole_id: str,
    total_depth: float,
    csv_by_hole: dict | None,
) -> list[dict[str, object]]:
    if csv_by_hole and hole_id in csv_by_hole:
        return [
            {
                "hole_id": interval.hole_id,
                "from_depth": interval.from_depth,
                "to_depth": interval.to_depth,
                "lithology_code": interval.lithology_code,
                "unit_order": interval.unit_order,
            }
            for interval in csv_by_hole[hole_id]
        ]
    digitized = digitized_lithology(transect_id, hole_id)
    if digitized:
        return [
            {
                "hole_id": interval.hole_id,
                "from_depth": interval.from_depth,
                "to_depth": interval.to_depth,
                "lithology_code": interval.lithology_code,
                "unit_order": interval.unit_order,
            }
            for interval in digitized
        ]
    return [
        {
            "hole_id": hole_id,
            "from_depth": 0.0,
            "to_depth": total_depth,
            "lithology_code": "Clay",
            "unit_order": 1,
        }
    ]


def build(
    transect_id: str = "A_A",
    *,
    lithology_csv: Path | None = None,
) -> Path:
    if transect_id not in ADVANTAGE_P2_TRANSECTS:
        raise ValueError(f"unknown transect_id: {transect_id}")
    if not CHLORIDES.is_file():
        raise FileNotFoundError(CHLORIDES)

    csv_by_hole = load_lithology_csv(lithology_csv) if lithology_csv else None
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
        eastings.append(eastings[-1] + 5.0)

    for hole_id, easting in zip(hole_ids, eastings, strict=True):
        total_depth = float(
            max(
                digitized_total_depth(transect_id, hole_id, fallback=15.0),
                max_by_hole.get(hole_id, 15.0) + 0.5,
            )
        )
        collars.append(
            {
                "hole_id": hole_id,
                "easting": float(easting),
                "northing": 0.0,
                "elevation": _ELEVATION_MASL,
                "total_depth": total_depth,
            }
        )
        lithology.extend(
            _lithology_rows_for_hole(transect_id, hole_id, total_depth, csv_by_hole)
        )

    hole_set = {row["hole_id"] for row in collars}
    environmental = environmental[environmental["hole_id"].isin(hole_set)].reset_index(drop=True)
    lithology_source = (
        f"CSV {lithology_csv.name}" if lithology_csv else "digitized pdf_extract_p2 sticks"
    )

    project = pd.DataFrame(
        [
            {"field": "client_name", "value": "WHITECAP RESOURCES INC."},
            {"field": "prepared_by", "value": "ECOVENTURE"},
            {"field": "project_number", "value": "100/09-36-055-02 W4M"},
            {"field": "section_title", "value": meta["section_title"]},
            {"field": "report_date", "value": "06/24/26"},
            {"field": "drawn_by", "value": "SL"},
            {
                "field": "data_source",
                "value": f"data/Data2 chlorides + {lithology_source}",
            },
            {"field": "transect_start", "value": meta["transect_start"]},
            {"field": "transect_end", "value": meta["transect_end"]},
            {"field": "vertical_exaggeration", "value": "1"},
            {"field": "figure_preset", "value": "p2_chemistry_sticks"},
            {
                "field": "notes",
                "value": (
                    f"Test workbook from Data2 ({meta['figure']}). Chloride From–To "
                    f"intervals (mg/kg). Lithology: {lithology_source}."
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
                    "Output style Project.figure_preset = p2_chemistry_sticks.",
                    "Environmental sheet: Chloride (mg/kg) with from_depth/to_depth intervals.",
                    f"Select holes in {transect_id.replace('_', '–')} order on Configure.",
                    f"Source chlorides: {CHLORIDES.name}",
                    f"Lithology: {lithology_source}",
                ]
            }
        ).to_excel(writer, sheet_name="Instructions", index=False)

    return output


def build_all(*, lithology_csv: Path | None = None) -> list[Path]:
    return [build(transect_id, lithology_csv=lithology_csv) for transect_id in ("A_A", "B_B")]


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
    parser.add_argument(
        "--lithology",
        type=Path,
        default=None,
        help="Optional lithology CSV (hole_id, from_depth, to_depth, lithology_code[, unit_order])",
    )
    args = parser.parse_args(argv)

    paths = (
        build_all(lithology_csv=args.lithology)
        if args.transect == "all"
        else [build(args.transect, lithology_csv=args.lithology)]
    )
    for path in paths:
        _smoke_ingest(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
