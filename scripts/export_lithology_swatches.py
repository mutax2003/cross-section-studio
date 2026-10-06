"""Write one PNG swatch per lithology for the Survey123 borehole-log export.

Survey123 renders each lithology from a fixed-size PNG in a zip; this script
draws every canonical code with the app's own colour and hatch so field logs
and cross-sections match. Size is a parameter because the required pixel
dimensions are set by the form, not by this program.

    python scripts/export_lithology_swatches.py --width 64 --height 64 --out dist/swatches
    python scripts/export_lithology_swatches.py --zip dist/lithology_swatches.zip
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from constants import (  # noqa: E402
    HATCH_LINE_COLOR,
    POLYGON_EDGE_COLOR,
    USGS_LITHOLOGY_COLORS,
    get_lithology_style,
)
from hatch_patterns import install as install_template_hatches  # noqa: E402

# Same marks as the app's figures (template 261002 stipple, dashes, plus marks).
install_template_hatches()

_DPI = 100


def swatch_filename(code: str) -> str:
    """Filesystem-safe, stable name: 'Sand and Gravel' -> 'sand_and_gravel.png'."""
    return re.sub(r"[^a-z0-9]+", "_", code.lower()).strip("_") + ".png"


def render_swatch_png(code: str, *, width_px: int, height_px: int, border: bool = True) -> bytes:
    """PNG bytes for one lithology at exactly width_px x height_px."""
    from io import BytesIO

    style = get_lithology_style(code)
    # A border on a tiny swatch is all border: drop it below 8 px.
    border = border and min(width_px, height_px) >= 8
    with plt.rc_context({"hatch.color": HATCH_LINE_COLOR, "hatch.linewidth": 0.65}):
        fig = plt.figure(figsize=(width_px / _DPI, height_px / _DPI), dpi=_DPI)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=style.color,
                hatch=style.hatch or None,
                edgecolor=POLYGON_EDGE_COLOR if border else "none",
                linewidth=1.0 if border else 0.0,
            )
        )
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=_DPI, facecolor=style.color)
        plt.close(fig)
    return buffer.getvalue()


def export_swatches(
    out_dir: Path | None,
    *,
    width_px: int,
    height_px: int,
    zip_path: Path | None = None,
    codes: list[str] | None = None,
) -> list[str]:
    names: list[str] = []
    selected = codes or sorted(USGS_LITHOLOGY_COLORS)
    if zip_path is not None:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) if zip_path else None
    try:
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
        for code in selected:
            name = swatch_filename(code)
            payload = render_swatch_png(code, width_px=width_px, height_px=height_px)
            if out_dir is not None:
                (out_dir / name).write_bytes(payload)
            if archive is not None:
                archive.writestr(name, payload)
            names.append(name)
    finally:
        if archive is not None:
            archive.close()
    return names


def resolve_codes(requested: list[str] | None) -> list[str]:
    """Canonical codes for --code values (case-insensitive); unknown names are an error."""
    if not requested:
        return sorted(USGS_LITHOLOGY_COLORS)
    lookup = {code.casefold(): code for code in USGS_LITHOLOGY_COLORS}
    unknown = [name for name in requested if name.strip().casefold() not in lookup]
    if unknown:
        raise SystemExit(
            f"Unknown lithology code(s): {', '.join(unknown)}. "
            f"Known codes: {', '.join(sorted(USGS_LITHOLOGY_COLORS))}"
        )
    return [lookup[name.strip().casefold()] for name in requested]


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive pixel size")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--width", type=_positive_int, default=64, help="swatch width in pixels (default 64)")
    parser.add_argument("--height", type=_positive_int, default=64, help="swatch height in pixels (default 64)")
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "lithology_swatches")
    parser.add_argument("--zip", type=Path, default=None, help="also write all swatches to this zip")
    parser.add_argument("--code", action="append", help="export only this code (repeatable)")
    parser.add_argument(
        "--sheet",
        type=Path,
        default=None,
        help="also write a legend sheet PNG (swatch, name, colour and pattern per row)",
    )
    args = parser.parse_args()
    names = export_swatches(
        args.out,
        width_px=args.width,
        height_px=args.height,
        zip_path=args.zip,
        codes=resolve_codes(args.code),
    )
    print(f"Wrote {len(names)} swatch(es) at {args.width}x{args.height} px to {args.out}")
    if args.sheet:
        from lithology_legend_sheet import build_legend_sheet_png

        args.sheet.parent.mkdir(parents=True, exist_ok=True)
        args.sheet.write_bytes(build_legend_sheet_png(resolve_codes(args.code)))
        print(f"Legend sheet: {args.sheet}")
    if args.zip:
        print(f"Zip: {args.zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
