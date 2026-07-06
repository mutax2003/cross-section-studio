"""Tests for automatic unit_order assignment."""

from __future__ import annotations

from models import Collar, Lithology, ParseResult, assign_missing_unit_orders, apply_unit_order_fix
from ai_quality import _hole_quality_issues


def test_assign_missing_unit_orders_for_duplicate_codes() -> None:
    lithologies = [
        Lithology(hole_id="BH-01", from_depth=0.0, to_depth=3.0, lithology_code="Clay"),
        Lithology(hole_id="BH-01", from_depth=3.0, to_depth=6.0, lithology_code="Sand"),
        Lithology(hole_id="BH-01", from_depth=6.0, to_depth=10.0, lithology_code="Clay"),
    ]
    updated, messages = assign_missing_unit_orders(lithologies)
    orders = sorted(
        (lit.from_depth, lit.unit_order)
        for lit in updated
        if lit.hole_id == "BH-01"
    )
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
