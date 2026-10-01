from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from export_lithology_swatches import (  # noqa: E402
    export_swatches,
    render_swatch_png,
    swatch_filename,
)
from PIL import Image  # noqa: E402

from constants import USGS_LITHOLOGY_COLORS  # noqa: E402


def test_swatch_is_exact_pixel_size_and_filled_with_the_lithology_colour() -> None:
    png = render_swatch_png("Sand", width_px=100, height_px=40, border=False)
    image = Image.open(io.BytesIO(png)).convert("RGB")
    assert image.size == (100, 40)
    assert image.getpixel((50, 20)) == (0xFF, 0xE3, 0x9F)  # Sand base colour between the dots
    hatched = Image.open(io.BytesIO(render_swatch_png("Sandy Clay", width_px=64, height_px=64))).convert("RGB")
    assert len(set(hatched.getdata())) > 2  # hatch dots present


def test_export_writes_every_code_to_folder_and_zip(tmp_path: Path) -> None:
    names = export_swatches(tmp_path / "out", width_px=32, height_px=32, zip_path=tmp_path / "s.zip")
    assert len(names) == len(USGS_LITHOLOGY_COLORS)
    assert swatch_filename("Sand and Gravel") == "sand_and_gravel.png" in names
    assert (tmp_path / "out" / "silty_sand.png").exists()
    with zipfile.ZipFile(tmp_path / "s.zip") as archive:
        assert sorted(archive.namelist()) == sorted(names)


def test_cli_rejects_unknown_codes_and_tiny_swatches_show_the_fill(tmp_path: Path) -> None:
    import pytest
    from export_lithology_swatches import export_swatches, resolve_codes

    assert resolve_codes(["sand", "Silty Sand"]) == ["Sand", "Silty Sand"]
    with pytest.raises(SystemExit, match="Unknown lithology code"):
        resolve_codes(["Bogus"])
    png = render_swatch_png("Clay", width_px=2, height_px=2)  # plain unit: no border, no hatch
    assert Image.open(io.BytesIO(png)).convert("RGB").getpixel((0, 0)) == (0x96, 0x72, 0x59)
    # Zip parent folders are created rather than crashing.
    export_swatches(None, width_px=8, height_px=8, zip_path=tmp_path / "new" / "dir" / "s.zip", codes=["Coal"])
    assert (tmp_path / "new" / "dir" / "s.zip").exists()
