"""Compare platform-generated GWM figures against reference PDF extracts.

Structural parity checks (dimensions, optional MSE). SSIM when scikit-image is installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REFERENCE_DIR = ROOT / "data" / "pdf_extract"
GENERATED_DIR = ROOT / "data" / "ecoventure_gwm"

FIGURE_PAIRS: dict[str, tuple[str, str]] = {
    "fig_3": (
        "2605011D_09_16-29-057-19 W4M_25-GWM_Fig 3 Cross Section A-A'.png",
        "fig_3_cross_section_a_a.png",
    ),
    "fig_4": (
        "2605011D_09_16-29-057-19 W4M_25-GWM_Fig 4 Cross Section B-B'.png",
        "fig_4_cross_section_b_b.png",
    ),
    "fig_5": (
        "2605011D_09_16-29-057-19 W4M_25-GWM_Fig 5 Cross Section C-C'.png",
        "fig_5_cross_section_c_c.png",
    ),
    "fig_6": (
        "2605011D_09_16-29-057-19 W4M_25-GWM_Fig 6 Cross Section D-D'.png",
        "fig_6_cross_section_d_d.png",
    ),
}


def _load_grayscale(path: Path):
    from PIL import Image
    import numpy as np

    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float64)


def _resize_to_match(reference, generated):
    from PIL import Image
    import numpy as np

    ref_img = Image.fromarray(reference.astype("uint8"))
    gen_img = Image.fromarray(generated.astype("uint8"))
    gen_resized = gen_img.resize(ref_img.size, Image.Resampling.BILINEAR)
    return reference, np.asarray(gen_resized, dtype=np.float64)


def compare_pair(reference: Path, generated: Path, *, mse_threshold: float) -> dict[str, object]:
    if not reference.is_file():
        raise FileNotFoundError(f"Reference missing: {reference}")
    if not generated.is_file():
        raise FileNotFoundError(f"Generated missing: {generated}")

    ref = _load_grayscale(reference)
    gen = _load_grayscale(generated)
    ref, gen = _resize_to_match(ref, gen)
    mse = float(((ref - gen) ** 2).mean())
    result: dict[str, object] = {
        "reference": str(reference),
        "generated": str(generated),
        "reference_size": (int(ref.shape[1]), int(ref.shape[0])),
        "generated_size": (int(gen.shape[1]), int(gen.shape[0])),
        "mse": mse,
        "mse_ok": mse <= mse_threshold,
    }
    try:
        from skimage.metrics import structural_similarity as ssim  # type: ignore[import-untyped]

        score = float(ssim(ref, gen, data_range=255.0))
        result["ssim"] = score
        result["ssim_ok"] = score >= 0.25
    except ImportError:
        result["ssim"] = None
        result["ssim_ok"] = None
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ecoventure_gwm PNGs to pdf_extract references")
    parser.add_argument(
        "--figure",
        choices=(*FIGURE_PAIRS.keys(), "all"),
        default="all",
        help="Figure to compare",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=REFERENCE_DIR,
        help="Directory with reference PNG extracts",
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=GENERATED_DIR,
        help="Directory with platform-generated PNGs",
    )
    parser.add_argument(
        "--mse-threshold",
        type=float,
        default=2500.0,
        help="Fail when mean squared error exceeds this value (resized to reference)",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Print results but always exit 0",
    )
    args = parser.parse_args()

    keys = list(FIGURE_PAIRS.keys()) if args.figure == "all" else [args.figure]
    failures: list[str] = []
    for key in keys:
        ref_name, gen_name = FIGURE_PAIRS[key]
        try:
            metrics = compare_pair(
                args.reference_dir / ref_name,
                args.generated_dir / gen_name,
                mse_threshold=args.mse_threshold,
            )
        except FileNotFoundError as exc:
            print(f"SKIP {key}: {exc}")
            failures.append(str(exc))
            continue
        mse = metrics["mse"]
        ssim_val = metrics.get("ssim")
        line = f"{key}: mse={mse:.1f} mse_ok={metrics['mse_ok']}"
        if ssim_val is not None:
            line += f" ssim={ssim_val:.3f} ssim_ok={metrics['ssim_ok']}"
        print(line)
        if not metrics["mse_ok"]:
            failures.append(f"{key} mse={mse:.1f}")

    if failures and not args.warn_only:
        print(f"FAILED: {len(failures)} issue(s)", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
