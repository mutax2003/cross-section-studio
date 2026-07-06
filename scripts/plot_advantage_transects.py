"""Generate Advantage Phase 2 cross-section SVGs for transects A, B, and C."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

from models import ConsultingTitleBlock, DataParser, subset_parse_result
from pipeline import build_cross_section
from render_profiles import LayoutMode

DEFAULT_WORKBOOK = ROOT / "data" / "advantage_phase2_platform.xlsx"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "advantage_transects"

TRANSECTS: dict[str, tuple[str, ...]] = {
    "A_EW": ("BH26-18", "BH26-03", "BH26-01", "BH26-02", "BH26-05"),
    "B_NS": ("BH26-08", "BH26-09", "BH26-10", "BH26-21"),
    "C_diagonal": ("BH26-16", "BH26-06", "BH26-22", "BH26-01"),
}

VERTICAL_EXAGGERATION = 8.0


def generate_transect_svg(
    workbook: Path,
    transect_id: str,
    hole_ids: tuple[str, ...],
    output_dir: Path,
    *,
    vertical_exaggeration: float = VERTICAL_EXAGGERATION,
    render_layout: LayoutMode = "section_sheet",
    section_label: str = "",
    map_scale: str = "1:1000",
    project_number: str = "",
    source: str = "",
) -> Path:
    parse_result = DataParser().parse_file(workbook)
    collar_lookup = {collar.hole_id: collar for collar in parse_result.collars}

    missing = [hole_id for hole_id in hole_ids if hole_id not in collar_lookup]
    if missing:
        raise ValueError(f"Transect {transect_id}: unknown hole(s) {missing}")

    subset = subset_parse_result(parse_result, hole_ids)
    transect_points = [
        (collar_lookup[hole_id].easting, collar_lookup[hole_id].northing)
        for hole_id in hole_ids
    ]
    title = (
        f"Advantage Phase 2 ESA — Transect {transect_id.replace('_', ' ')}\n"
        f"{' → '.join(hole_ids)} | VE {vertical_exaggeration:.0f}x | Datum: relative (100 m placeholder)"
    )
    consulting_title_block = None
    if render_layout == "consulting_section":
        consulting_title_block = ConsultingTitleBlock(
            section_label=section_label or transect_id.replace("_", " "),
            map_scale=map_scale,
            project_number=project_number,
            source=source,
        )
    result = build_cross_section(
        subset.collars,
        subset.lithologies,
        transect_points,
        vertical_exaggeration=vertical_exaggeration,
        title=title,
        render_layout=render_layout,
        consulting_title_block=consulting_title_block,
        screen_intervals=subset.screen_intervals,
        vertical_gradients=subset.vertical_gradients,
        water_levels=subset.water_levels,
        show_legend=render_layout != "consulting_section",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"cross_section_{transect_id.lower()}.svg"
    output_path.write_bytes(result.svg_bytes)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Advantage transect cross-section SVGs")
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ve", type=float, default=VERTICAL_EXAGGERATION)
    parser.add_argument(
        "--layout",
        choices=("section_sheet", "consulting_section", "chart"),
        default="section_sheet",
        help="Render layout profile",
    )
    parser.add_argument("--section-label", default="", help="Consulting section label (B-B')")
    parser.add_argument("--map-scale", default="1:1000", help="Consulting map scale text")
    parser.add_argument("--project-number", default="", help="Consulting project number")
    parser.add_argument("--source", default="", help="Consulting data source")
    args = parser.parse_args()

    if not args.workbook.exists():
        raise FileNotFoundError(
            f"Platform workbook not found: {args.workbook}\n"
            "Run: python scripts/convert_advantage_export.py"
        )

    for transect_id, hole_ids in TRANSECTS.items():
        path = generate_transect_svg(
            args.workbook,
            transect_id,
            hole_ids,
            args.output_dir,
            vertical_exaggeration=args.ve,
            render_layout=args.layout,
            section_label=args.section_label,
            map_scale=args.map_scale,
            project_number=args.project_number,
            source=args.source,
        )
        print(f"Wrote {path} ({len(hole_ids)} holes)")


if __name__ == "__main__":
    main()
