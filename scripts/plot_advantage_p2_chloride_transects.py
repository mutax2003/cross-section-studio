"""Export Advantage Phase 2 chloride cross-section figures (Figs 6–7).

Prefers generated workbooks under ``data/Data2/`` when present; otherwise uses
``advantage_p2_reference.fixtures.build_parse_result`` (digitized lithology +
interval chlorides).

Writes SVG under ``data/advantage_p2_transects/`` and PNG/SVG under
``data/Data2/test_output/`` using parity filenames expected by
``scripts/compare_figure_parity.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

from advantage_p2_reference.fixtures import build_parse_result
from advantage_p2_reference.transects import ADVANTAGE_P2_TRANSECTS
from ingestion import ingest_workbook
from models import subset_parse_result
from pipeline import build_cross_section

DEFAULT_OUTPUT_DIR = ROOT / "data" / "advantage_p2_transects"
PARITY_OUTPUT_DIR = ROOT / "data" / "Data2" / "test_output"
_WORKBOOK_BY_TRANSECT: dict[str, Path] = {
    "A_A": ROOT / "data" / "Data2" / "Cross_Section_Test_AA_Lithology_Chlorides.xlsx",
    "B_B": ROOT / "data" / "Data2" / "Cross_Section_Test_BB_Lithology_Chlorides.xlsx",
}

_PARITY_STEM: dict[str, str] = {
    "A_A": "fig6_aa_lithology_chlorides",
    "B_B": "fig7_bb_lithology_chlorides",
}


def _load_from_workbook(transect_id: str, workbook: Path):
    spec = ADVANTAGE_P2_TRANSECTS[transect_id]
    parse_result, _report = ingest_workbook(workbook)
    subset = subset_parse_result(parse_result, spec.hole_ids)
    # Re-order collars to transect sequence and assign profile eastings.
    by_id = {collar.hole_id: collar for collar in subset.collars}
    collars = []
    for hole_id, easting in zip(spec.hole_ids, spec.profile_eastings, strict=True):
        collar = by_id.get(hole_id)
        if collar is None:
            raise ValueError(f"{workbook.name}: missing collar {hole_id}")
        collars.append(collar.model_copy(update={"easting": easting, "northing": 0.0}))
    hole_set = set(spec.hole_ids)
    lithologies = tuple(lit for lit in subset.lithologies if lit.hole_id in hole_set)
    environmental = tuple(
        reading for reading in subset.environmental_readings if reading.hole_id in hole_set
    )
    return spec, subset.model_copy(
        update={
            "collars": tuple(collars),
            "lithologies": lithologies,
            "environmental_readings": environmental,
        }
    )


def generate_transect(
    transect_id: str,
    output_dir: Path,
    *,
    parity_dir: Path,
    workbook: Path | None = None,
) -> Path:
    workbook_path = workbook or _WORKBOOK_BY_TRANSECT.get(transect_id)
    if workbook_path is not None and workbook_path.is_file():
        spec, parse_result = _load_from_workbook(transect_id, workbook_path)
    else:
        spec, parse_result = build_parse_result(transect_id)
    transect_points = [
        (collar.easting, collar.northing) for collar in parse_result.collars
    ]
    result = build_cross_section(
        parse_result.collars,
        parse_result.lithologies,
        transect_points,
        vertical_exaggeration=spec.vertical_exaggeration,
        title=spec.title_block.section_label,
        render_layout="consulting_section",
        consulting_title_block=spec.title_block,
        environmental_readings=parse_result.environmental_readings,
        environmental_parameters=("Chloride",),
        show_parameter_labels=True,
        parameter_interpolate_segments=spec.parameter_interpolate_segments,
        interpretation_mode=spec.interpretation_mode,  # type: ignore[arg-type]
        elevation_mode=spec.elevation_mode,
        export_formats=frozenset({"svg", "png"}),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    parity_dir.mkdir(parents=True, exist_ok=True)

    svg_path = output_dir / f"{spec.output_stem}.svg"
    svg_path.write_bytes(result.svg_bytes)

    parity_stem = _PARITY_STEM.get(transect_id, spec.output_stem)
    (parity_dir / f"{parity_stem}.svg").write_bytes(result.svg_bytes)
    if result.png_bytes:
        (parity_dir / f"{parity_stem}.png").write_bytes(result.png_bytes)
        (output_dir / f"{spec.output_stem}.png").write_bytes(result.png_bytes)
    return svg_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Advantage P2 chloride transect figures")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--parity-dir", type=Path, default=PARITY_OUTPUT_DIR)
    parser.add_argument(
        "--transect",
        choices=tuple(ADVANTAGE_P2_TRANSECTS),
        default=None,
        help="Export one transect only (default: all)",
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=None,
        help="Optional native workbook override (otherwise Data2 test workbook if present)",
    )
    args = parser.parse_args()

    transect_ids = (args.transect,) if args.transect else tuple(ADVANTAGE_P2_TRANSECTS)
    for transect_id in transect_ids:
        path = generate_transect(
            transect_id,
            args.output_dir,
            parity_dir=args.parity_dir,
            workbook=args.workbook,
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
