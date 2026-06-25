"""Generic workbook conversion CLI using import profiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion import FormatDetector, export_platform_workbook, list_profiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert field export to platform Collars/Lithology workbook")
    parser.add_argument("--source", type=Path, required=True, help="Source .xlsx workbook")
    parser.add_argument("--output", type=Path, required=True, help="Output .xlsx path")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Import profile id (default: auto-detect)",
    )
    parser.add_argument(
        "--override",
        type=str,
        default=None,
        help="Profile override id (e.g. advantage_phase2_2026)",
    )
    parser.add_argument("--elevation", type=float, default=None, help="Default collar elevation (m)")
    parser.add_argument("--target-crs", type=str, default=None, help="Target CRS EPSG code")
    parser.add_argument("--list-profiles", action="store_true", help="List available import profiles")
    args = parser.parse_args()

    if args.list_profiles:
        for profile in list_profiles():
            print(f"{profile.id}: {profile.label}")
        return

    if not args.source.exists():
        raise FileNotFoundError(f"Source workbook not found: {args.source}")

    profile_id = args.profile
    if profile_id is None:
        detection = FormatDetector().detect(args.source)
        profile_id = detection.profile_id
        print(f"Detected format: {detection.label} ({detection.confidence:.0%})")

    collars, lithology = export_platform_workbook(
        args.source,
        args.output,
        profile_id=profile_id,
        override_id=args.override,
        elevation_m=args.elevation,
        target_crs=args.target_crs,
    )
    print(f"Wrote {args.output}")
    print(f"  Profile: {profile_id}" + (f" + override {args.override}" if args.override else ""))
    print(f"  Collars: {len(collars)} holes")
    print(f"  Lithology intervals: {len(lithology)}")


if __name__ == "__main__":
    main()
