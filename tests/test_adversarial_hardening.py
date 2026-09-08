"""Hardening tests for figure retain memo and Sections validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app_services import (
    _digest_text,
    _store_figure_memo,
    clear_service_memos,
)
from models import WorkbookSectionSpec


def test_workbook_section_spec_rejects_label_injection() -> None:
    with pytest.raises(ValidationError):
        WorkbookSectionSpec(label="A-A'\nEvil | X", hole_ids=("BH-01", "BH-02"))
    with pytest.raises(ValidationError):
        WorkbookSectionSpec(label="A|B", hole_ids=("BH-01", "BH-02"))


def test_workbook_section_spec_rejects_hole_separators() -> None:
    with pytest.raises(ValidationError):
        WorkbookSectionSpec(label="A-A'", hole_ids=("BH-01,extra", "BH-02"))


def test_figure_memo_overwrite_closes_previous() -> None:
    clear_service_memos()
    closed: list[object] = []

    class _Fig:
        pass

    fig1, fig2 = _Fig(), _Fig()

    import app_services as services

    original = services._close_figure_bundle

    def _tracking_close(bundle: dict[str, object] | None) -> None:
        if bundle and bundle.get("figure") is not None:
            closed.append(bundle["figure"])
        # Do not call matplotlib on fake figures.

    services._close_figure_bundle = _tracking_close  # type: ignore[assignment]
    try:
        key = ("local", _digest_text("a"), _digest_text("b"))
        _store_figure_memo(key, {"figure": fig1})
        _store_figure_memo(key, {"figure": fig2})
        assert fig1 in closed
        assert services._figure_memo[key]["figure"] is fig2
    finally:
        clear_service_memos()
        services._close_figure_bundle = original  # type: ignore[assignment]


def test_figure_memo_evicts_per_session_cap() -> None:
    clear_service_memos()
    closed: list[object] = []

    class _Fig:
        pass

    import app_services as services

    original = services._close_figure_bundle

    def _tracking_close(bundle: dict[str, object] | None) -> None:
        if bundle and bundle.get("figure") is not None:
            closed.append(bundle["figure"])

    services._close_figure_bundle = _tracking_close  # type: ignore[assignment]
    try:
        fig_a, fig_b, fig_c = _Fig(), _Fig(), _Fig()
        key_a = ("sess-1", _digest_text("s1"), _digest_text("r1"))
        key_b = ("sess-1", _digest_text("s2"), _digest_text("r2"))
        key_c = ("sess-1", _digest_text("s3"), _digest_text("r3"))
        other = ("sess-2", _digest_text("s1"), _digest_text("r1"))
        _store_figure_memo(key_a, {"figure": fig_a})
        _store_figure_memo(key_b, {"figure": fig_b})
        _store_figure_memo(other, {"figure": _Fig()})
        _store_figure_memo(key_c, {"figure": fig_c})
        assert fig_a in closed
        assert key_a not in services._figure_memo
        assert key_b in services._figure_memo
        assert key_c in services._figure_memo
        assert other in services._figure_memo
    finally:
        clear_service_memos()
        services._close_figure_bundle = original  # type: ignore[assignment]


def test_memo_lock_is_reentrant() -> None:
    import app_services as services

    with services._memo_lock:
        with services._memo_lock:
            assert services._memo_lock._is_owned()  # type: ignore[attr-defined]
