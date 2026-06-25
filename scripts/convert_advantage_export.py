"""Convert Advantage Phase 2 field export to platform Collars/Lithology workbook."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion import export_platform_workbook, parse_depth_interval

DEFAULT_SOURCE = Path(
    r"C:\Users\Andrew Liu\Downloads"
    r"\Boreholes_Advantage_2026_Phase_2_ESA_10011_24_110_08_W6M_062426.xlsx"
)
DEFAULT_OUTPUT = ROOT / "data" / "advantage_phase2_platform.xlsx"
DEFAULT_ELEVATION_M = 100.0

# Re-export for backward-compatible tests
COORDINATE_OFFSETS_M: dict[str, tuple[float, float]] = {
    "BH26-15": (0.5, 0.0),
}


def convert_advantage_export(
    source: Path,
    output: Path,
    *,
    elevation_m: float = DEFAULT_ELEVATION_M,
):
    return export_platform_workbook(
        source,
        output,
        profile_id="field_export_v1",
        override_id="advantage_phase2_2026",
        elevation_m=elevation_m,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Advantage export to platform workbook")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--elevation", type=float, default=DEFAULT_ELEVATION_M)
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Source workbook not found: {args.source}")

    collars, lithology = convert_advantage_export(
        args.source,
        args.output,
        elevation_m=args.elevation,
    )
    print(f"Wrote {args.output}")
    print(f"  Collars: {len(collars)} holes")
    print(f"  Lithology intervals: {len(lithology)}")
    print(f"  Elevation datum: uniform {args.elevation} m (Option B placeholder)")
    if COORDINATE_OFFSETS_M:
        print(f"  Coordinate offsets applied: {COORDINATE_OFFSETS_M}")


if __name__ == "__main__":
    main()
