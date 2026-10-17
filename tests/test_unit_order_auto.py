"""Tests for automatic unit_order assignment."""

from __future__ import annotations

from ai_quality import _hole_quality_issues
from models import Collar, Lithology, ParseResult, apply_unit_order_fix, assign_missing_unit_orders


def test_assign_missing_unit_orders_for_duplicate_codes() -> None:
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=3.0, lithology_code="Clay"),
        Lithology(hole_id="BH-01", from_depth=3.0, to_depth=6.0, lithology_code="Sand"),
        Lithology(hole_id="BH-01", from_depth=6.0, to_depth=10.0, lithology_code="Clay"),
    ]
    updated, messages = assign_missing_unit_orders(lithologies)
    orders = sorted((lit.from_depth, lit.unit_order) for lit in updated if lit.hole_id == "BH-01")
    assert orders == [(0.0, 1), (3.0, 2), (6.0, 3)]
    assert messages


def test_apply_unit_order_fix_clears_qa_error() -> None:
    collar = Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0)
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=3.0, lithology_code="Clay"),
        Lithology(hole_id="BH-01", from_depth=3.0, to_depth=6.0, lithology_code="Sand"),
        Lithology(hole_id="BH-01", from_depth=6.0, to_depth=10.0, lithology_code="Clay"),
    ]
    issues_before = _hole_quality_issues(collar, lithologies)
    assert any(issue.code == "duplicate_lithology_no_unit_order" for issue in issues_before)

    fixed = apply_unit_order_fix(
        ParseResult(
            collars=(collar,),
            lithologies=tuple(lithologies),
            errors=(),
        )
    )
    issues_after = _hole_quality_issues(collar, list(fixed.lithologies))
    assert not any(issue.code == "duplicate_lithology_no_unit_order" for issue in issues_after)


# ---------------------------------------------------------------------------
# Site-wide alignment (parse_ops.site_unit_sequence)
# ---------------------------------------------------------------------------


def _holes(spec: dict[str, list[str]]) -> list[Lithology]:
    """Build 1 m intervals per hole from top-down code lists."""
    return [
        Lithology(hole_id=hole, from_depth=float(i), to_depth=float(i + 1), lithology_code=code)
        for hole, codes in spec.items()
        for i, code in enumerate(codes)
    ]


def _orders(lithologies) -> dict[str, list[tuple[str, int | None]]]:
    out: dict[str, list[tuple[str, int | None]]] = {}
    for lit in sorted(lithologies, key=lambda item: (item.hole_id, item.from_depth)):
        out.setdefault(lit.hole_id, []).append((lit.lithology_code, lit.unit_order))
    return out


def _assert_unique_and_monotonic(orders) -> None:
    for hole, seq in orders.items():
        numbers = [order for _code, order in seq]
        assert numbers == sorted(numbers) and len(set(numbers)) == len(numbers), hole


def test_unit_missing_from_one_hole_does_not_shift_deeper_numbers() -> None:
    updated, messages = assign_missing_unit_orders(
        _holes(
            {
                "N": ["Topsoil", "Silt", "Clay", "Sand", "Clay"],
                "S": ["Topsoil", "Clay", "Sand", "Clay"],
            }
        )
    )
    orders = _orders(updated)
    assert orders["N"] == [("Topsoil", 1), ("Silt", 2), ("Clay", 3), ("Sand", 4), ("Clay", 5)]
    assert orders["S"] == [("Topsoil", 1), ("Clay", 3), ("Sand", 4), ("Clay", 5)]
    assert len(messages) == 1 and "assigned unit_order" in messages[0]


def test_hole_without_repeats_is_numbered_too_so_keys_match() -> None:
    """A hole with unique codes would key on code alone and miss numbered neighbours."""
    updated, _ = assign_missing_unit_orders(
        _holes({"A": ["Clay", "Sand", "Clay"], "B": ["Clay", "Sand"]})
    )
    assert _orders(updated)["B"] == [("Clay", 1), ("Sand", 2)]


def test_no_repeats_anywhere_leaves_unit_order_blank() -> None:
    updated, messages = assign_missing_unit_orders(
        _holes({"A": ["Clay", "Sand"], "B": ["Clay", "Silt", "Sand"]})
    )
    assert messages == ()
    assert all(lit.unit_order is None for lit in updated)


def test_pinched_sand_between_two_clays_keeps_the_aquifer_aligned() -> None:
    """Where the upper Sand pinches out its two Clays merge into one logged Clay."""
    updated, messages = assign_missing_unit_orders(
        _holes(
            {
                "W": ["Clay", "Sand", "Clay", "Aquifer", "Clay"],
                "M": ["Clay", "Aquifer", "Clay"],
                "E": ["Clay", "Sand", "Clay", "Aquifer", "Gravel", "Clay"],
            }
        )
    )
    orders = _orders(updated)
    aquifer = {hole: dict(seq)["Aquifer"] for hole, seq in orders.items()}
    assert len(set(aquifer.values())) == 1
    base_clay = {hole: seq[-1][1] for hole, seq in orders.items()}
    assert len(set(base_clay.values())) == 1
    assert orders["E"][4] == ("Gravel", aquifer["E"] + 1)
    _assert_unique_and_monotonic(orders)
    assert len(messages) == 1


def test_split_unit_does_not_shift_deeper_units() -> None:
    """Consecutive same-code rows are one unit; the extra piece gets the next number."""
    updated, _ = assign_missing_unit_orders(
        _holes({"A": ["Clay", "Clay", "Sand", "Clay"], "B": ["Clay", "Sand", "Clay"]})
    )
    orders = _orders(updated)
    assert orders["A"] == [("Clay", 1), ("Clay", 2), ("Sand", 3), ("Clay", 4)]
    assert orders["B"] == [("Clay", 1), ("Sand", 3), ("Clay", 4)]


def test_contradictory_order_follows_the_majority_and_is_reported() -> None:
    updated, messages = assign_missing_unit_orders(
        _holes(
            {
                "A": ["Clay", "Silt", "Sand", "Clay"],
                "B": ["Clay", "Silt", "Sand", "Clay"],
                "C": ["Clay", "Sand", "Silt", "Clay"],
            }
        )
    )
    orders = _orders(updated)
    assert orders["A"] == orders["B"]
    assert dict(orders["A"])["Silt"] < dict(orders["A"])["Sand"]
    # C's Sand still lines up; its Silt cannot and becomes its own unit.
    assert dict(orders["C"])["Sand"] == dict(orders["A"])["Sand"]
    _assert_unique_and_monotonic(orders)
    assert any(message.startswith("C: Silt could not be lined up") for message in messages)


def test_site_unit_sequence_places_one_hole_units_by_depth() -> None:
    from parse_ops import site_unit_sequence

    codes, units_by_hole, unaligned = site_unit_sequence(
        {
            "A": [("Top", 0.0), ("Silt", 1.0), ("Base", 3.0)],
            "B": [("Top", 0.0), ("Sand", 2.0), ("Base", 3.0)],
        }
    )
    assert codes == ("Top", "Silt", "Sand", "Base")
    assert units_by_hole == {"A": [0, 1, 3], "B": [0, 2, 3]}
    assert unaligned == {}


def test_explicit_unit_orders_are_kept_and_reused_for_blank_holes() -> None:
    lithologies = [
        *(
            Lithology(
                hole_id="A",
                from_depth=float(i),
                to_depth=float(i + 1),
                lithology_code=code,
                unit_order=order,
            )
            for i, (code, order) in enumerate([("Clay", 10), ("Sand", 20), ("Clay", 30)])
        ),
        *_holes({"B": ["Clay", "Silt", "Sand", "Clay"]}),
    ]
    updated, _ = assign_missing_unit_orders(lithologies)
    orders = _orders(updated)
    assert orders["A"] == [("Clay", 10), ("Sand", 20), ("Clay", 30)]
    assert orders["B"] == [("Clay", 10), ("Silt", 31), ("Sand", 20), ("Clay", 30)]


def test_pinched_sand_section_keeps_aquifer_continuous() -> None:
    """End to end: stratigraphy's pairwise re-match alone cannot untangle the merged
    Clay where the upper Sand pinches out; the site-wide numbering can."""
    from pipeline import compute_section_geometry

    spec = {
        "W": [("Clay", 2), ("Sand", 1), ("Clay", 2), ("Aquifer", 2), ("Clay", 3)],
        "M": [("Clay", 5), ("Aquifer", 2), ("Clay", 3)],
        "E": [("Clay", 2), ("Sand", 1), ("Clay", 2), ("Aquifer", 2), ("Clay", 3)],
    }
    collars: list[Collar] = []
    lithologies: list[Lithology] = []
    for index, (hole, layers) in enumerate(spec.items()):
        depth = 0.0
        for code, thickness in layers:
            lithologies.append(
                Lithology(
                    hole_id=hole,
                    from_depth=depth,
                    to_depth=depth + thickness,
                    lithology_code=code,
                )
            )
            depth += thickness
        collars.append(
            Collar(
                hole_id=hole,
                easting=100.0 * index,
                northing=0.0,
                elevation=100.0,
                total_depth=depth,
            )
        )
    lithologies, _ = assign_missing_unit_orders(lithologies)
    geometry = compute_section_geometry(
        collars,
        lithologies,
        [(0.0, 0.0), (200.0, 0.0)],
        allow_pinch_outs=True,
        fail_on_overlaps=False,
        warn_on_correlation_gaps=False,
    )
    continuous = [p for p in geometry.polygons if not p.is_pinch_out]
    assert sum(p.lithology_code == "Aquifer" for p in continuous) == 2
    assert not any(p.lithology_code == "Aquifer" and p.is_pinch_out for p in geometry.polygons)
    assert not geometry.overlap_pairs


def test_auto_unit_order_ignores_sheet_row_order() -> None:
    from models import Lithology
    from parse_ops import assign_missing_unit_orders

    def hole(hole_id, top_order):
        return [
            Lithology(hole_id=hole_id, from_depth=0, to_depth=1, lithology_code="Clay", unit_order=top_order),
            Lithology(hole_id=hole_id, from_depth=1, to_depth=2, lithology_code="Sand"),
            Lithology(hole_id=hole_id, from_depth=2, to_depth=3, lithology_code="Clay"),
        ]

    def orders(rows):
        result, _ = assign_missing_unit_orders(rows)
        return sorted((r.hole_id, r.from_depth, r.unit_order) for r in result)

    first = orders(hole("A", 1) + hole("B", 5) + hole("C", None))
    assert first == orders(hole("B", 5) + hole("A", 1) + hole("C", None))
    assert first == orders(hole("C", None) + hole("B", 5) + hole("A", 1))
