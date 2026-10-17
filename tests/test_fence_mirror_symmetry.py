"""Reversing a transect mirrors the fence.

Fuzz-found sections where GEOS's floating overlay clipped the pinch-out wedges
differently for the reversed transect: three wedges from one hole tapering to
a shared tip (seed 7308), and a millimetre-thin sliver below the lowest match
tiled against the matched fill's base (seeds 7440, 7428). Overlap clipping now
runs on a fixed precision grid (``stratigraphy.CLIP_PRECISION_GRID_M``), where
the overlay is exact, and overlap detection ignores the grid's slivers.
"""

from __future__ import annotations

from collections import defaultdict

import pytest
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

from models import Collar, Lithology
from pipeline import compute_section_geometry
from stratigraphy import GeologicalPolygon, detect_polygon_overlaps

_CASES = {
    # Three Bedrock wedges from H1 with no match in H0: all taper to one tip,
    # so their edges coincide exactly and the clip of one against another left
    # a ring joined by a float-noise neck in one direction only.
    "shared_tip_wedges": (
        dict(interpretation_mode="correlation_lines", allow_pinch_outs=True),
        [(21.57482727434849, -0.9843156670516187), (229.9641897711506, 0.9414668434331421)],
        [
            Collar(hole_id="H0", easting=90.48608021726385, northing=0.7092845506452163, elevation=100.0, total_depth=1591.552748685516),
            Collar(hole_id="H1", easting=79.74192085865019, northing=0.7921929590231727, elevation=198.2915117390999, total_depth=1343.193207707914),
        ],
        [
            Lithology(hole_id="H0", from_depth=0.0, to_depth=1223.90809, lithology_code="Gravel", unit_order=30),
            Lithology(hole_id="H0", from_depth=1223.90809, to_depth=1299.31264, lithology_code="Clay", unit_order=46),
            Lithology(hole_id="H1", from_depth=52.2721, to_depth=111.60377, lithology_code="Bedrock", unit_order=26),
            Lithology(hole_id="H1", from_depth=181.54612, to_depth=181.55378, lithology_code="Bedrock", unit_order=28),
            Lithology(hole_id="H1", from_depth=181.55378, to_depth=301.45626, lithology_code="Bedrock", unit_order=29),
        ],
    ),
    # A 4 mm Sand sliver under the matched Clay: tiled against the Clay base
    # (its top edge *is* the fill's base), the floating overlay reported it
    # wholly inside the fill for the reversed transect (kept 0%).
    "sliver_under_matched_fill": (
        dict(interpretation_mode="correlation_lines", allow_pinch_outs=True),
        [(45.0669339990083, 31.241052204386), (981.9951794482488, 974.8331100415581)],
        [
            Collar(hole_id="H0", easting=229.7711500475152, northing=337.2886012067603, elevation=100.0, total_depth=35.54929011811961),
            Collar(hole_id="H1", easting=317.0946079417317, northing=31.241052204386, elevation=154.8082765882723, total_depth=1275.682135154258),
        ],
        [
            Lithology(hole_id="H1", from_depth=359.68655, to_depth=453.37838, lithology_code="Clay", unit_order=23),
            Lithology(hole_id="H1", from_depth=453.37838, to_depth=544.63666, lithology_code="Clay", unit_order=24),
            Lithology(hole_id="H0", from_depth=0.0, to_depth=1.65525, lithology_code="Clay", unit_order=6),
            Lithology(hole_id="H0", from_depth=1.65525, to_depth=1.65909, lithology_code="Sand", unit_order=17),
        ],
    ),
    "sliver_till_under_matched_till": (
        dict(interpretation_mode="correlation_lines", allow_pinch_outs=True),
        None,
        [
            Collar(hole_id="H0", easting=487.7930375632001, northing=982.4368912683664, elevation=99.0638531500523, total_depth=2638.612358043051),
            Collar(hole_id="H1", easting=237.7556168414363, northing=311.6072257491865, elevation=239.2627123275914, total_depth=14.21152734362621),
        ],
        [
            Lithology(hole_id="H1", from_depth=0.0, to_depth=0.55792, lithology_code="Till", unit_order=5),
            Lithology(hole_id="H1", from_depth=0.93274, to_depth=0.93569, lithology_code="Till", unit_order=7),
            Lithology(hole_id="H0", from_depth=0.0, to_depth=718.83841, lithology_code="Till", unit_order=5),
            Lithology(hole_id="H0", from_depth=718.83841, to_depth=1599.48011, lithology_code="Silt", unit_order=37),
        ],
    ),
}


def _transect_length(points) -> float:
    return sum(
        ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False)
    )


def _unit_unions(geometry, length: float, *, mirror: bool):
    groups = defaultdict(list)
    for polygon in geometry.polygons:
        shape = polygon.polygon
        if mirror:
            shape = affinity.scale(shape, xfact=-1.0, yfact=1.0, origin=(length / 2.0, 0.0))
        groups[(frozenset(polygon.hole_pair), polygon.lithology_code, polygon.is_pinch_out)].append(shape)
    return {key: unary_union(shapes) for key, shapes in groups.items()}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_reversed_transect_mirrors_the_fence(name: str) -> None:
    kwargs, transect, collars, lithologies = _CASES[name]
    transect = transect or [(c.easting, c.northing) for c in collars]
    forward = compute_section_geometry(collars, lithologies, transect, **kwargs)
    backward = compute_section_geometry(collars, lithologies, list(reversed(transect)), **kwargs)
    length = _transect_length(transect)
    fwd = _unit_unions(forward, length, mirror=False)
    rev = _unit_unions(backward, length, mirror=True)
    assert set(fwd) == set(rev)
    for key, shape in fwd.items():
        mirrored = rev[key]
        difference = shape.symmetric_difference(mirrored).area
        assert difference <= 1e-4 * max(shape.area, mirrored.area) + 1e-4, (key, shape.area, mirrored.area)
    assert len(forward.overlap_warnings) == len(backward.overlap_warnings)
    # Clipping on the grid never reports the pair as self-overlapping.
    assert not [w for w in forward.overlap_warnings if " / " in w]
    assert not [w for w in backward.overlap_warnings if " / " in w]


def _pair(left: Polygon, right: Polygon) -> list[GeologicalPolygon]:
    return [
        GeologicalPolygon("Sand", left, ("A", "B"), is_pinch_out=False),
        GeologicalPolygon("Clay", right, ("A", "B"), is_pinch_out=True),
    ]


def test_overlap_detection_ignores_grid_slivers_but_not_real_overlaps() -> None:
    """A node half a grid step off the shared edge makes a 1e-7 m² sliver on a
    kilometre-long contact: a precision artefact, not a correlation conflict."""
    left = Polygon([(0.0, 0.0), (1000.0, 0.0), (1000.0, 1000.0), (0.0, 1000.0)])
    sliver = Polygon([(1000.0, 0.0), (2000.0, 0.0), (2000.0, 1000.0), (1000.0, 1000.0), (1000.0 - 5e-10, 500.0)])
    assert left.intersection(sliver).area > 1e-9  # the old fixed floor would have reported it
    assert detect_polygon_overlaps(_pair(left, sliver)) == []
    real = affinity.translate(sliver, xoff=-0.01)
    assert len(detect_polygon_overlaps(_pair(left, real))) == 1


def test_extreme_coordinates_still_build_a_fence() -> None:
    """Coordinates near 5e8 m made GEOS snap-rounding reject some overlays
    ("Ring edge missing"); the floating fallback must keep the fence buildable."""
    import random

    from models import Collar, Lithology
    from pipeline import compute_section_geometry

    rng = random.Random(7264)
    codes = ["Clay", "Sand", "Silt", "Till", "Gravel", "Bedrock"]
    collars = [
        Collar(hole_id=f"H{i}", easting=5e8 + i * 37.3, northing=1e6 + rng.uniform(-3, 3),
               elevation=99000 + rng.uniform(-40, 40), total_depth=60)
        for i in range(8)
    ]
    rows = []
    for collar in collars:
        depth = 0.0
        for _ in range(rng.randint(3, 7)):
            top = depth
            depth = min(depth + rng.uniform(0.004, 15), 60)
            rows.append(Lithology(hole_id=collar.hole_id, from_depth=round(top, 5), to_depth=round(depth, 5),
                                  lithology_code=rng.choice(codes)))
            if depth >= 60:
                break
    line = [(collars[0].easting, collars[0].northing), (collars[-1].easting, collars[-1].northing)]
    geometry = compute_section_geometry(collars, rows, line)
    assert geometry.polygons
