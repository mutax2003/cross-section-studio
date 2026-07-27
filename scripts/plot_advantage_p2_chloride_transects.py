"""Export Advantage Phase 2 chloride cross-section SVGs (Figs 6–7)."""

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
from pipeline import build_cross_section

DEFAULT_OUTPUT_DIR = ROOT / "data" / "advantage_p2_transects"


def generate_transect_svg(transect_id: str, output_dir: Path) -> Path:
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
        parameter_interpolate_segments=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{spec.output_stem}.svg"
    output_path.write_bytes(result.svg_bytes)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Advantage P2 chloride transect SVGs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--transect",
        choices=tuple(ADVANTAGE_P2_TRANSECTS),
        default=None,
        help="Export one transect only (default: all)",
    )
    args = parser.parse_args()

    transect_ids = (args.transect,) if args.transect else tuple(ADVANTAGE_P2_TRANSECTS)
    for transect_id in transect_ids:
        path = generate_transect_svg(transect_id, args.output_dir)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
