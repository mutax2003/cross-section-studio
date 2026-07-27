"""Extract first-page PDF figures to PNG for figure-parity references.

Requires PyMuPDF (``pip install pymupdf``). Not needed at app runtime — commit the
PNG outputs and CI compares those files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (pdf_glob_or_name under data/, output_dir, output_png_name)
GWM_EXTRACTS: tuple[tuple[str, str], ...] = (
    (
        "2605011D_09_16-29-057-19 W4M_25-GWM_Fig 3 Cross Section A-A'.pdf",
        "2605011D_09_16-29-057-19 W4M_25-GWM_Fig 3 Cross Section A-A'.png",
    ),
    (
        "2605011D_09_16-29-057-19 W4M_25-GWM_Fig 4 Cross Section B-B'.pdf",
        "2605011D_09_16-29-057-19 W4M_25-GWM_Fig 4 Cross Section B-B'.png",
    ),
    (
        "2605011D_09_16-29-057-19 W4M_25-GWM_Fig 5 Cross Section C-C'.pdf",
        "2605011D_09_16-29-057-19 W4M_25-GWM_Fig 5 Cross Section C-C'.png",
    ),
    (
        "2605011D_09_16-29-057-19 W4M_25-GWM_Fig 6 Cross Section D-D'.pdf",
        "2605011D_09_16-29-057-19 W4M_25-GWM_Fig 6 Cross Section D-D'.png",
    ),
)

P2_EXTRACTS: tuple[tuple[str, str], ...] = (
    (
        "260624G_100_09-36-055-02_W4M_P2_Fig 6 Cross Section_A-A'.pdf",
        "fig_6_cross_section_a_a.png",
    ),
    (
        "260624G_100_09-36-055-02_W4M_P2_Fig 7 Cross Section_B-B'.pdf",
        "fig_7_cross_section_b_b.png",
    ),
)


def _extract_page(pdf_path: Path, png_path: Path, *, dpi: float) -> tuple[int, int]:
    import fitz

    doc = fitz.open(pdf_path)
    try:
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(png_path)
        return int(pix.width), int(pix.height)
    finally:
        doc.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("gwm", "p2", "all"),
        default="all",
        help="Which reference suite to extract",
    )
    parser.add_argument("--dpi", type=float, default=300.0, help="Raster DPI (default 300)")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data",
        help="Directory containing source PDFs",
    )
    args = parser.parse_args()

    try:
        import fitz  # noqa: F401
    except ImportError:
        print("PyMuPDF required: pip install pymupdf", file=sys.stderr)
        return 1

    jobs: list[tuple[Path, Path]] = []
    if args.suite in ("gwm", "all"):
        out_dir = args.data_dir / "pdf_extract"
        for pdf_name, png_name in GWM_EXTRACTS:
            jobs.append((args.data_dir / pdf_name, out_dir / png_name))
    if args.suite in ("p2", "all"):
        out_dir = args.data_dir / "pdf_extract_p2"
        for pdf_name, png_name in P2_EXTRACTS:
            jobs.append((args.data_dir / pdf_name, out_dir / png_name))

    for pdf_path, png_path in jobs:
        if not pdf_path.is_file():
            print(f"MISSING PDF: {pdf_path}", file=sys.stderr)
            return 1
        width, height = _extract_page(pdf_path, png_path, dpi=args.dpi)
        print(f"Wrote {png_path.relative_to(ROOT)} ({width}x{height} @ {args.dpi:g} dpi)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
