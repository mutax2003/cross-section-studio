"""Compare platform-generated figures against reference PNG extracts.

Hard-fails missing pairs and empty/invalid dimensions. MSE can soft-fail via
``--warn-only`` (softens MSE only). SSIM is always soft (reported, never fails).
Optional ``--suite p2`` / ``all`` compares Advantage P2 pairs when fixtures exist.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REFERENCE_DIR = ROOT / "data" / "pdf_extract"
GENERATED_DIR = ROOT / "data" / "ecoventure_gwm"
P2_REFERENCE_DIR = ROOT / "data" / "pdf_extract_p2"
P2_GENERATED_DIR = ROOT / "data" / "Data2" / "test_output"

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

P2_FIGURE_PAIRS: dict[str, tuple[str, str]] = {
    "p2_fig_6": (
        "fig_6_cross_section_a_a.png",
        "fig6_aa_lithology_chlorides.png",
    ),
    "p2_fig_7": (
        "fig_7_cross_section_b_b.png",
        "fig7_bb_lithology_chlorides.png",
    ),
}


def _load_grayscale(path: Path):
    from PIL import Image
    import numpy as np

    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float64)
    if arr.size == 0 or min(arr.shape[:2]) < 1:
        raise ValueError(f"empty or invalid image: {path}")
    return arr


def _resize_to_match(reference, generated):
    from PIL import Image
    import numpy as np

    ref_img = Image.fromarray(reference.astype("uint8"))
    gen_img = Image.fromarray(generated.astype("uint8"))
    gen_resized = gen_img.resize(ref_img.size, Image.Resampling.BILINEAR)
    return reference, np.asarray(gen_resized, dtype=np.float64)


def compare_pair(
    reference: Path,
    generated: Path,
    *,
    mse_threshold: float,
    require_same_size: bool = False,
    size_tolerance_px: int = 0,
) -> dict[str, object]:
    if not reference.is_file():
        raise FileNotFoundError(f"Reference missing: {reference}")
    if not generated.is_file():
        raise FileNotFoundError(f"Generated missing: {generated}")

    ref_native = _load_grayscale(reference)
    gen_native = _load_grayscale(generated)
    ref_wh = (int(ref_native.shape[1]), int(ref_native.shape[0]))
    gen_wh = (int(gen_native.shape[1]), int(gen_native.shape[0]))
    size_ok = (
        abs(ref_wh[0] - gen_wh[0]) <= size_tolerance_px
        and abs(ref_wh[1] - gen_wh[1]) <= size_tolerance_px
    )
    # Always require non-empty valid dimensions (structural).
    dims_valid = min(ref_wh) > 0 and min(gen_wh) > 0

    ref, gen = _resize_to_match(ref_native, gen_native)
    mse = float(((ref - gen) ** 2).mean())
    result: dict[str, object] = {
        "reference": str(reference),
        "generated": str(generated),
        "native_reference_size": ref_wh,
        "native_generated_size": gen_wh,
        "size_ok": size_ok,
        "dims_valid": dims_valid,
        "mse": mse,
        "mse_ok": mse <= mse_threshold,
    }
    if require_same_size and not size_ok:
        result["mse_ok"] = False
    try:
        from skimage.metrics import structural_similarity as ssim  # type: ignore[import-untyped]

        score = float(ssim(ref, gen, data_range=255.0))
        result["ssim"] = score
        result["ssim_ok"] = score >= 0.25
    except ImportError:
        result["ssim"] = None
        result["ssim_ok"] = None
    return result


def _pairs_available(
    pairs: dict[str, tuple[str, str]],
    reference_dir: Path,
    generated_dir: Path,
) -> bool:
    return all(
        (reference_dir / ref_name).is_file() and (generated_dir / gen_name).is_file()
        for ref_name, gen_name in pairs.values()
    )


def _run_suite(
    *,
    label: str,
    pairs: dict[str, tuple[str, str]],
    keys: list[str],
    reference_dir: Path,
    generated_dir: Path,
    mse_threshold: float,
    require_same_size: bool,
    size_tolerance_px: int,
    warn_mse: bool,
) -> list[str]:
    failures: list[str] = []
    for key in keys:
        if key not in pairs:
            continue
        ref_name, gen_name = pairs[key]
        try:
            metrics = compare_pair(
                reference_dir / ref_name,
                generated_dir / gen_name,
                mse_threshold=mse_threshold,
                require_same_size=require_same_size,
                size_tolerance_px=size_tolerance_px,
            )
        except FileNotFoundError as exc:
            print(f"MISSING {label}/{key}: {exc}", file=sys.stderr)
            failures.append(f"missing:{label}/{key}")
            continue
        except ValueError as exc:
            print(f"INVALID {label}/{key}: {exc}", file=sys.stderr)
            failures.append(f"invalid:{label}/{key}")
            continue

        mse = metrics["mse"]
        ssim_val = metrics.get("ssim")
        native_ref = metrics["native_reference_size"]
        native_gen = metrics["native_generated_size"]
        line = (
            f"{label}/{key}: size {native_gen} vs ref {native_ref} "
            f"size_ok={metrics['size_ok']} mse={mse:.1f} mse_ok={metrics['mse_ok']}"
        )
        if ssim_val is not None:
            line += f" ssim={ssim_val:.3f} ssim_ok={metrics['ssim_ok']}"
            if metrics.get("ssim_ok") is False:
                print(f"WARN {label}/{key}: ssim below threshold (soft)", file=sys.stderr)
        print(line)

        if not metrics.get("dims_valid", True):
            failures.append(f"{label}/{key} invalid dimensions")
        if require_same_size and not metrics["size_ok"]:
            failures.append(
                f"{label}/{key} size {native_gen} != {native_ref}"
            )
        if not metrics["mse_ok"]:
            msg = f"{label}/{key} mse={mse:.1f}"
            if warn_mse:
                print(f"WARN {msg} (soft)", file=sys.stderr)
            else:
                failures.append(msg)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare platform PNGs to reference extracts")
    parser.add_argument(
        "--suite",
        choices=("gwm", "p2", "all"),
        default="gwm",
        help="Which figure suite to compare (p2 skipped when fixtures absent)",
    )
    parser.add_argument(
        "--figure",
        choices=(*FIGURE_PAIRS.keys(), *P2_FIGURE_PAIRS.keys(), "all"),
        default="all",
        help="Figure key within the selected suite(s)",
    )
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR)
    parser.add_argument("--generated-dir", type=Path, default=GENERATED_DIR)
    parser.add_argument("--p2-reference-dir", type=Path, default=P2_REFERENCE_DIR)
    parser.add_argument("--p2-generated-dir", type=Path, default=P2_GENERATED_DIR)
    parser.add_argument(
        "--mse-threshold",
        type=float,
        default=2500.0,
        help="Fail when mean squared error exceeds this value (resized to reference)",
    )
    parser.add_argument(
        "--require-same-size",
        action="store_true",
        help="Hard-fail when native PNG dimensions differ (default: warn via size_ok only)",
    )
    parser.add_argument(
        "--size-tolerance-px",
        type=int,
        default=0,
        help="Pixel tolerance when --require-same-size is set",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Soften MSE failures to warnings; missing/invalid pairs still fail",
    )
    args = parser.parse_args()

    failures: list[str] = []
    run_gwm = args.suite in ("gwm", "all")
    run_p2 = args.suite in ("p2", "all")

    if run_gwm:
        gwm_keys = list(FIGURE_PAIRS.keys()) if args.figure == "all" else [args.figure]
        failures.extend(
            _run_suite(
                label="gwm",
                pairs=FIGURE_PAIRS,
                keys=gwm_keys,
                reference_dir=args.reference_dir,
                generated_dir=args.generated_dir,
                mse_threshold=args.mse_threshold,
                require_same_size=args.require_same_size,
                size_tolerance_px=args.size_tolerance_px,
                warn_mse=args.warn_only,
            )
        )

    if run_p2:
        if not _pairs_available(P2_FIGURE_PAIRS, args.p2_reference_dir, args.p2_generated_dir):
            print("P2 parity skipped — reference/generated PNG fixtures absent")
        else:
            p2_keys = list(P2_FIGURE_PAIRS.keys()) if args.figure == "all" else [args.figure]
            failures.extend(
                _run_suite(
                    label="p2",
                    pairs=P2_FIGURE_PAIRS,
                    keys=p2_keys,
                    reference_dir=args.p2_reference_dir,
                    generated_dir=args.p2_generated_dir,
                    mse_threshold=args.mse_threshold,
                    require_same_size=args.require_same_size,
                    size_tolerance_px=args.size_tolerance_px,
                    warn_mse=args.warn_only,
                )
            )

    if failures:
        print(f"FAILED: {len(failures)} issue(s)", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
