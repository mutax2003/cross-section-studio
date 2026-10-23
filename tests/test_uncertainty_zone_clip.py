"""Wider-spacing uncertainty zones stay inside the drawn fence and are keyed.

Figure QA: data/advantage_phase2_platform.xlsx showed a pale amber,
flat-bottomed band at ~87 m under BH26-12…BH26-16 that was no stratigraphy
polygon. It was ``_draw_uncertainty_zones``: a rectangle from the section's
deepest TD to its highest collar across every hole pair spaced past
``uncertainty_spacing_m``, so it hung below the fence under short holes. The
zone is now clipped to the union of that pair's fence polygons and the outside
legend keys it only when one is drawn.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.ops import unary_union

from models import Collar, Lithology
from render_profiles import CHART_PROFILE, SECTION_SHEET_PROFILE
from renderer import UNCERTAINTY_LEGEND_LABEL, CrossSectionRenderer
from renderer_common import LEGEND_LABEL_MAX_CHARS, fit_label
from tests.conftest import run_pipeline

ZONE_GID = "uncertainty-zone"
LEGEND_TEXT = fit_label(UNCERTAINTY_LEGEND_LABEL, max_chars=LEGEND_LABEL_MAX_CHARS)


def _inputs(*, wide: bool) -> tuple[list[Collar], list[Lithology], list[tuple[float, float]]]:
    """Three holes: the first pair spaced past (or within) the 80 m threshold.

    BH-01 and BH-02 are short (10 m); BH-03 is deep (30 m), so a rectangle
    down to the section's deepest TD would show below the fence between the
    first pair.
    """
    x2 = 150.0 if wide else 50.0
    x3 = x2 + 50.0
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=x2, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-03", easting=x3, northing=0.0, elevation=100.0, total_depth=30.0),
    ]
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=4.0, lithology_code="Clay"),
        Lithology(hole_id="BH-01", from_depth=4.0, to_depth=10.0, lithology_code="Sand"),
        Lithology(hole_id="BH-02", from_depth=0.0, to_depth=5.0, lithology_code="Clay"),
        Lithology(hole_id="BH-02", from_depth=5.0, to_depth=10.0, lithology_code="Sand"),
        Lithology(hole_id="BH-03", from_depth=0.0, to_depth=5.0, lithology_code="Clay"),
        Lithology(hole_id="BH-03", from_depth=5.0, to_depth=12.0, lithology_code="Sand"),
        Lithology(hole_id="BH-03", from_depth=12.0, to_depth=30.0, lithology_code="Gravel"),
    ]
    return collars, lithologies, [(0.0, 0.0), (x3, 0.0)]


def _render(profile, *, wide: bool):
    collars, lithologies, transect = _inputs(wide=wide)
    projected, polygons, _svg = run_pipeline(
        collars, lithologies, transect, render_layout=str(profile.layout)
    )
    renderer = CrossSectionRenderer(render_profile=profile)
    figure = renderer.render(polygons, projected, collar_depths={c.hole_id: c.total_depth for c in collars})
    return renderer, figure, polygons


def _zone_collections(figure) -> list:
    return [c for ax in figure.axes for c in ax.collections if c.get_gid() == ZONE_GID]


def _legend_labels(figure) -> list[str]:
    labels: list[str] = []
    for ax in figure.axes:
        legend = ax.get_legend()
        if legend is not None:
            labels += [t.get_text() for t in legend.get_texts()]
    return labels


@pytest.mark.parametrize("profile", [SECTION_SHEET_PROFILE, CHART_PROFILE], ids=["sheet", "chart"])
def test_zone_clipped_to_pair_fence(profile) -> None:
    renderer, figure, polygons = _render(profile, wide=True)
    zones = _zone_collections(figure)
    assert len(zones) == 1, "one wider-spacing zone is drawn"
    verts = np.vstack([np.asarray(p.vertices) for p in zones[0].get_paths()])
    ve = renderer.vertical_exaggeration
    pair = [p.polygon for p in polygons if set(p.hole_pair) == {"BH-01", "BH-02"}]
    assert pair, "the wide pair has fence polygons to clip to"
    x0, y0, x1, y1 = unary_union(pair).bounds
    eps = 1e-6
    assert verts[:, 0].min() >= x0 - eps and verts[:, 0].max() <= x1 + eps
    assert verts[:, 1].min() >= y0 * ve - eps and verts[:, 1].max() <= y1 * ve + eps
    # Never the old flat bottom at the section's deepest TD (BH-03, 70 m RL).
    deepest = 70.0 * ve
    assert verts[:, 1].min() > deepest + 1.0
    # The zone covers the pair's x span (it is the whole fence there).
    assert verts[:, 0].min() == pytest.approx(0.0, abs=1e-6)
    assert verts[:, 0].max() == pytest.approx(150.0, abs=1e-6)
    assert LEGEND_TEXT in _legend_labels(figure)
    assert renderer._has_uncertainty_zone is True


@pytest.mark.parametrize("profile", [SECTION_SHEET_PROFILE, CHART_PROFILE], ids=["sheet", "chart"])
def test_no_zone_no_legend_entry(profile) -> None:
    renderer, figure, _polygons = _render(profile, wide=False)
    assert _zone_collections(figure) == []
    assert LEGEND_TEXT not in _legend_labels(figure)
    assert renderer._has_uncertainty_zone is False


def test_zone_absent_from_svg_when_spacing_is_close() -> None:
    collars, lithologies, transect = _inputs(wide=False)
    _projected, _polygons, svg = run_pipeline(collars, lithologies, transect)
    text = svg.decode("utf-8", errors="ignore")
    assert ZONE_GID not in text
    assert "Wider hole spacing" not in text


def test_zone_in_svg_when_spacing_is_wide() -> None:
    collars, lithologies, transect = _inputs(wide=True)
    _projected, _polygons, svg = run_pipeline(collars, lithologies, transect)
    text = svg.decode("utf-8", errors="ignore")
    assert ZONE_GID in text
    assert "Wider hole spacing" in text
