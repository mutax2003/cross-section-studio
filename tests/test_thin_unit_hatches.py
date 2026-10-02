"""Thin lithology intervals get a densified hatch so they do not read as plain fill."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

from constants import get_lithology_style
from models import Collar, Lithology
from render_profiles import SECTION_SHEET_PROFILE
from renderer import CrossSectionRenderer
from renderer_common import (
    THIN_UNIT_MAX_DENSIFY,
    THIN_UNIT_MIN_DENSIFY,
    THIN_UNIT_MIN_HEIGHT_IN,
    ThinUnitHatchCollection,
    densify_hatch,
    thin_unit_densify_factor,
)
from tests.conftest import assert_valid_svg, run_pipeline

matplotlib.use("Agg")

THIN_TOP = 5.0
THIN_BOTTOM = 5.3
THICK_TOP = 8.0
THICK_BOTTOM = 13.0


def _section_inputs(thin_top: float = THIN_TOP) -> tuple[list[Collar], list[Lithology]]:
    thin_bottom = thin_top + (THIN_BOTTOM - THIN_TOP)
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=15.0),
        Collar(hole_id="BH-02", easting=60.0, northing=0.0, elevation=100.0, total_depth=15.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=thin_top, lithology_code="Clay"),
        Lithology(
            hole_id="BH-01", from_depth=thin_top, to_depth=thin_bottom, lithology_code="Sand"
        ),
        Lithology(
            hole_id="BH-01", from_depth=thin_bottom, to_depth=THICK_TOP, lithology_code="Clay"
        ),
        Lithology(
            hole_id="BH-01", from_depth=THICK_TOP, to_depth=THICK_BOTTOM, lithology_code="Sand"
        ),
        Lithology(hole_id="BH-01", from_depth=THICK_BOTTOM, to_depth=15.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=15.0, lithology_code="Clay"),
    ]
    return collars, lithologies


def _render(
    *,
    show_hatches: bool = True,
    thin_min_height_in: float | None = None,
    thin_top: float = THIN_TOP,
):
    collars, lithologies = _section_inputs(thin_top)
    projected, polygons, _ = run_pipeline(
        collars,
        lithologies,
        [(0.0, 0.0), (60.0, 0.0)],
        show_hatches=show_hatches,
        show_legend=False,
    )
    renderer = CrossSectionRenderer(
        show_hatches=show_hatches,
        show_legend=False,
        render_profile=SECTION_SHEET_PROFILE,
    )
    if thin_min_height_in is not None:
        renderer.thin_unit_min_height_in = thin_min_height_in
    figure = renderer.render(polygons, projected, collar_depths={"BH-01": 15.0, "BH-02": 15.0})
    return renderer, figure


def _sand_collection(figure) -> ThinUnitHatchCollection:
    ax = figure.axes[0]
    sand = get_lithology_style("Sand")
    matches = [
        collection
        for collection in ax.collections
        if isinstance(collection, ThinUnitHatchCollection) and collection.base_hatch == sand.hatch
    ]
    assert len(matches) == 1, "expected one Sand column collection"
    return matches[0]


def _rect_index_by_height(collection: ThinUnitHatchCollection) -> tuple[int, int]:
    """Return (thin_index, thick_index) from the data-space rect heights."""
    heights = [abs(path.vertices[3, 1] - path.vertices[0, 1]) for path in collection._paths]
    assert len(heights) == 2
    thin = int(np.argmin(heights))
    thick = 1 - thin
    assert heights[thin] < heights[thick]
    return thin, thick


def _thin_rect_dark_pixels(
    figure, collection: ThinUnitHatchCollection, dpi: int
) -> tuple[int, int]:
    """Count hatch-coloured pixels inside the thin rect (inset to skip the border)."""
    figure.dpi = dpi
    figure.canvas.draw()
    buffer = np.asarray(figure.canvas.buffer_rgba())
    height_px = buffer.shape[0]
    thin_index, _ = _rect_index_by_height(collection)
    verts_px = collection.get_transform().transform(collection._paths[thin_index].vertices)
    x0, x1 = verts_px[:, 0].min(), verts_px[:, 0].max()
    y0, y1 = verts_px[:, 1].min(), verts_px[:, 1].max()
    row0, row1 = int(round(height_px - y1)) + 2, int(round(height_px - y0)) - 2
    col0, col1 = int(round(x0)) + 3, int(round(x1)) - 3
    region = buffer[row0:row1, col0:col1, :3].astype(int)
    assert region.size > 0
    fill_rgb = np.array([int(get_lithology_style("Sand").color[i : i + 2], 16) for i in (1, 3, 5)])
    dark = int((region.sum(axis=2) < fill_rgb.sum() - 60).sum())
    return dark, int(region.shape[0] * region.shape[1])


@pytest.mark.parametrize(
    ("hatch", "expected"),
    [(".", ".."), ("/", "//"), ("O", "OO"), ("+", "++"), ("", ""), (None, "")],
)
def test_densify_hatch_repeats_sparse_marks(hatch, expected) -> None:
    assert densify_hatch(hatch) == expected
    assert densify_hatch(hatch, 3) == (expected[:1] * 3 if expected else "")


def test_densify_factor_follows_interval_height() -> None:
    assert thin_unit_densify_factor(0.5) == 1
    assert thin_unit_densify_factor(THIN_UNIT_MIN_HEIGHT_IN) == 1
    just_under = THIN_UNIT_MIN_HEIGHT_IN - 1e-3
    assert thin_unit_densify_factor(just_under) == THIN_UNIT_MIN_DENSIFY
    # 0.3 m in a 15 m hole on the section sheet is ~0.08 in: needs 3 rows/0.167 in.
    assert thin_unit_densify_factor(0.08) == 3
    assert thin_unit_densify_factor(0.01) == THIN_UNIT_MAX_DENSIFY
    assert thin_unit_densify_factor(0.0) == THIN_UNIT_MAX_DENSIFY
    assert thin_unit_densify_factor(float("nan")) == 1
    # Factors never fall below 2 or exceed the cap.
    for height in np.linspace(1e-4, THIN_UNIT_MIN_HEIGHT_IN - 1e-4, 50):
        assert (
            THIN_UNIT_MIN_DENSIFY
            <= thin_unit_densify_factor(float(height))
            <= THIN_UNIT_MAX_DENSIFY
        )


def test_thin_sand_interval_gets_denser_hatch_than_thick_one() -> None:
    _, figure = _render()
    collection = _sand_collection(figure)
    figure.canvas.draw()
    thin_index, thick_index = _rect_index_by_height(collection)
    heights_in = collection._interval_heights_in()
    assert heights_in is not None
    assert heights_in[thin_index] < THIN_UNIT_MIN_HEIGHT_IN < heights_in[thick_index]
    resolved = collection.resolved_hatches
    base = get_lithology_style("Sand").hatch
    assert resolved[thick_index] == base
    expected_factor = thin_unit_densify_factor(float(heights_in[thin_index]))
    assert expected_factor >= THIN_UNIT_MIN_DENSIFY
    assert resolved[thin_index] == densify_hatch(base, expected_factor)
    assert len(resolved[thin_index]) > len(resolved[thick_index])
    assert set(resolved[thin_index]) == set(base), "densified hatch keeps the legend's mark"
    # The collection is left on its base hatch after drawing, so a second draw is stable.
    assert collection.get_hatch() == base
    matplotlib.pyplot.close(figure)


def test_thin_hatch_resolution_is_stable_across_dpi() -> None:
    _, figure = _render()
    collection = _sand_collection(figure)
    thin_index, thick_index = _rect_index_by_height(collection)
    outcomes = []
    for dpi in (72, 150, 300):
        figure.dpi = dpi
        figure.canvas.draw()
        outcomes.append(tuple(collection.resolved_hatches))
    assert len(set(outcomes)) == 1
    assert len(outcomes[0][thin_index]) > 1
    assert outcomes[0][thick_index] == "."
    matplotlib.pyplot.close(figure)


def test_thin_densification_can_be_disabled_via_threshold() -> None:
    _, figure = _render(thin_min_height_in=0.0)
    collection = _sand_collection(figure)
    figure.canvas.draw()
    assert collection.resolved_hatches == [".", "."]
    matplotlib.pyplot.close(figure)


def test_show_hatches_false_leaves_no_hatch_on_either_interval() -> None:
    renderer, figure = _render(show_hatches=False)
    ax = figure.axes[0]
    assert not any(isinstance(c, ThinUnitHatchCollection) for c in ax.collections)
    assert all(c.get_hatch() in (None, "") for c in ax.collections)
    figure.canvas.draw()
    assert all(c.get_hatch() in (None, "") for c in ax.collections)
    svg_bytes = renderer.to_svg_bytes(figure)
    assert_valid_svg(svg_bytes)
    matplotlib.pyplot.close(figure)


def test_thin_interval_raster_shows_hatch_marks_at_150_dpi() -> None:
    """Whatever the hatch phase, the 0.3 m bed shows marks; the sparse hatch does not."""
    dense_counts: list[int] = []
    plain_counts: list[int] = []
    for thin_top in (3.0, 3.4, 4.1, 4.3, 4.7):
        _, figure = _render(thin_top=thin_top)
        dark, total = _thin_rect_dark_pixels(figure, _sand_collection(figure), dpi=150)
        matplotlib.pyplot.close(figure)
        assert total > 0
        dense_counts.append(dark)

        _, plain_figure = _render(thin_min_height_in=0.0, thin_top=thin_top)
        plain_dark, _ = _thin_rect_dark_pixels(
            plain_figure, _sand_collection(plain_figure), dpi=150
        )
        matplotlib.pyplot.close(plain_figure)
        plain_counts.append(plain_dark)

    assert min(dense_counts) >= 20, f"thin bed lost its hatch at some phase: {dense_counts}"
    assert min(plain_counts) < min(dense_counts), (
        f"sparse hatch should miss the thin bed at some phase: plain={plain_counts} dense={dense_counts}"
    )


def test_thin_hatch_section_exports_valid_svg_and_png() -> None:
    renderer, figure = _render()
    svg_bytes = renderer.to_svg_bytes(figure)
    assert_valid_svg(svg_bytes)
    png_bytes = renderer.to_png_bytes(figure, dpi=150)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    matplotlib.pyplot.close(figure)
