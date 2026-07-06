"""Generate EcoVenture GWM cross-section figures (PDF references 3–6)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

from models import DataParser, subset_parse_result
from pipeline import build_cross_section
from gwm_reference import GWM_TRANSECTS, build_parse_result, write_fixture_workbook

logger = logging.getLogger(__name__)

DEFAULT_WORKBOOK = ROOT / "data" / "fixtures" / "ecoventure_gwm_16-29.xlsx"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "ecoventure_gwm"


def _load_from_workbook(workbook: Path, transect_id: str):
    spec = GWM_TRANSECTS[transect_id]
    parse_result = DataParser().parse_file(workbook)
    collar_lookup = {collar.hole_id: collar for collar in parse_result.collars}
    missing = [hole_id for hole_id in spec.hole_ids if hole_id not in collar_lookup]
    if missing:
        raise ValueError(f"unknown hole(s) {missing}")
    subset = subset_parse_result(parse_result, spec.hole_ids)
    transect_points = [
        (collar_lookup[hole_id].easting, collar_lookup[hole_id].northing)
        for hole_id in spec.hole_ids
    ]
    return spec, subset, transect_points


def generate_transect(
    transect_id: str,
    output_dir: Path,
    *,
    workbook: Path | None = None,
    use_fixture: bool = True,
) -> tuple[Path, Path | None]:
    spec = GWM_TRANSECTS[transect_id]
    if workbook is not None and workbook.exists():
        try:
            spec, subset, transect_points = _load_from_workbook(workbook, transect_id)
        except (ValueError, OSError) as exc:
            if not use_fixture:
                raise
            logger.warning("Workbook load failed (%s); using golden fixture for %s", exc, transect_id)
            spec, subset = build_parse_result(transect_id)
            transect_points = [(c.easting, c.northing) for c in subset.collars]
    elif use_fixture:
        spec, subset = build_parse_result(transect_id)
        transect_points = [(c.easting, c.northing) for c in subset.collars]
    else:
        raise FileNotFoundError(
            f"No workbook at {workbook} and fixture fallback disabled. "
            "Use --fixture or provide a valid workbook with --use-workbook."
        )

    result = build_cross_section(
        subset.collars,
        subset.lithologies,
        transect_points,
        vertical_exaggeration=spec.vertical_exaggeration,
        render_layout="consulting_section",
        consulting_title_block=spec.title_block,
        screen_intervals=subset.screen_intervals,
        water_levels=subset.water_levels,
        show_legend=False,
        show_hatches=True,
        export_formats=frozenset({"svg", "png"}),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = spec.output_stem or f"fig_{spec.figure_number}_{transect_id.lower()}"
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"
    svg_path.write_bytes(result.svg_bytes)
    if result.png_bytes:
        png_path.write_bytes(result.png_bytes)
        return svg_path, png_path
    return svg_path, None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Export EcoVenture GWM cross-section figures")
    parser.add_argument(
        "--transect",
        choices=(*GWM_TRANSECTS.keys(), "all"),
        default="all",
        help="Transect to render (A_A = Fig 3, etc.)",
    )
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--use-workbook",
        action="store_true",
        help="Load collar/lithology data from workbook (default: golden fixture geometry)",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        default=True,
        help="Fall back to golden fixture when workbook load fails (default)",
    )
    parser.add_argument(
        "--no-fixture",
        action="store_false",
        dest="fixture",
        help="Do not fall back to fixture on workbook errors",
    )
    parser.add_argument(
        "--write-workbook",
        action="store_true",
        help="Write data/fixtures/ecoventure_gwm_16-29.xlsx from golden data",
    )
    args = parser.parse_args()

    if args.write_workbook:
        path = write_fixture_workbook(args.workbook)
        logger.info("Wrote workbook %s", path)

    workbook = args.workbook if args.use_workbook and args.workbook.exists() else None
    if args.use_workbook and not args.workbook.exists():
        logger.warning("Workbook not found at %s; using fixtures", args.workbook)

    transect_ids = list(GWM_TRANSECTS.keys()) if args.transect == "all" else [args.transect]
    for transect_id in transect_ids:
        svg_path, png_path = generate_transect(
            transect_id,
            args.output_dir,
            workbook=workbook,
            use_fixture=args.fixture,
        )
        logger.info("Wrote %s", svg_path)
        if png_path is not None:
            logger.info("Wrote %s", png_path)


if __name__ == "__main__":
    main()
