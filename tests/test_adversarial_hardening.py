"""Hardening tests for figure retain memo and Sections validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app_services import (
    _digest_text,
    _request_memo_key,
    _run_build_cross_section,
    _store_figure_memo,
    clear_service_memos,
)
from models import Collar, Lithology, ParseResult, WorkbookSectionSpec
from section_build_request import SectionBuildRequest


def _minimal_subset() -> ParseResult:
    return ParseResult(
        collars=(
            Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
            Collar(hole_id="BH-02", easting=10.0, northing=0.0, elevation=100.0, total_depth=10.0),
        ),
        lithologies=(
            Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Sand"),
            Lithology(hole_id="BH-02", from_depth=0.0, to_depth=10.0, lithology_code="Sand"),
        ),
        errors=(),
    )


def _minimal_request() -> SectionBuildRequest:
    return SectionBuildRequest(transect_points=((0.0, 0.0), (10.0, 0.0)))


def _request_key(subset_json: str, request_json: str) -> tuple[str, str, str]:
    return _request_memo_key(subset_json, request_json)


def _patch_redraw_path(
    monkeypatch: pytest.MonkeyPatch,
    *,
    png: bytes = b"REDRAW_PNG",
    pdf: bytes = b"REDRAW_PDF",
    svg: bytes = b"",
    retain_figures: bool = False,
) -> dict[str, int]:
    import app_services as services

    calls: dict[str, int] = {"render": 0}
    retained_figs: list[object] = []

    def _fake_render(*_args: object, **kwargs: object) -> SimpleNamespace:
        calls["render"] += 1
        formats = frozenset(kwargs.get("export_formats") or ())
        bundle = None
        if retain_figures and formats == frozenset({"svg"}):
            fig = object()
            retained_figs.append(fig)

            class _Renderer:
                def export_figure_bytes(
                    self, figure: object, export_formats: frozenset[str], **_k: object
                ):
                    return b"", png, pdf

            bundle = {
                "figure": fig,
                "renderer": _Renderer(),
                "polygons": [],
                "projected": object(),
            }
        return SimpleNamespace(
            svg_bytes=svg,
            png_bytes=png,
            pdf_bytes=pdf,
            polygons=(),
            lithology_codes=(),
            overlap_warnings=(),
            retained_export_bundle=bundle,
        )

    monkeypatch.setattr(services, "render_cross_section_from_geometry", _fake_render)
    monkeypatch.setattr(services, "_resolve_section_geometry", lambda *_a, **_k: object())
    monkeypatch.setattr(services, "_apply_section_geometry_qa", lambda geo, _req: geo)
    monkeypatch.setattr(
        services,
        "_build_section_kwargs",
        lambda *_a, **_k: (SimpleNamespace(), "correlated", ()),
    )
    monkeypatch.setattr(
        services,
        "_cached_section_inputs",
        lambda _s, _r: (_minimal_subset(), _minimal_request()),
    )
    calls["retained_figs"] = retained_figs  # type: ignore[assignment]
    return calls


def _tracking_close(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    import app_services as services

    closed: list[object] = []

    def _close(bundle: dict[str, object] | None) -> None:
        if bundle and bundle.get("figure") is not None:
            closed.append(bundle["figure"])

    monkeypatch.setattr(services, "_close_figure_bundle", _close)
    return closed


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

    def _tracking_close_fn(bundle: dict[str, object] | None) -> None:
        if bundle and bundle.get("figure") is not None:
            closed.append(bundle["figure"])
        # Do not call matplotlib on fake figures.

    services._close_figure_bundle = _tracking_close_fn  # type: ignore[assignment]
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

    def _tracking_close_fn(bundle: dict[str, object] | None) -> None:
        if bundle and bundle.get("figure") is not None:
            closed.append(bundle["figure"])

    services._close_figure_bundle = _tracking_close_fn  # type: ignore[assignment]
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


def test_retain_prepare_happy_uses_export_no_redraw(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_service_memos()
    import app_services as services

    closed = _tracking_close(monkeypatch)
    redraw = _patch_redraw_path(monkeypatch)
    export_calls: list[frozenset[str]] = []

    class _Renderer:
        def export_figure_bytes(self, figure: object, formats: frozenset[str], **_kwargs: object):
            export_calls.append(frozenset(formats))
            return b"", b"RETAIN_PNG", b"RETAIN_PDF"

    fig = object()
    subset_json = "subset-retain-happy"
    request_json = "request-retain-happy"
    key = _request_key(subset_json, request_json)
    _store_figure_memo(
        key,
        {
            "figure": fig,
            "renderer": _Renderer(),
            "polygons": [],
            "projected": object(),
            "polygon_count": 2,
            "lithology_codes_tuple": ("Sand",),
            "overlap_warnings": ("gap",),
        },
    )

    try:
        png, pdf = services.cached_build_section_exports(subset_json, request_json)
        assert png == b"RETAIN_PNG"
        assert pdf == b"RETAIN_PDF"
        assert export_calls == [frozenset({"png", "pdf"})]
        assert redraw["render"] == 0
        assert fig in closed
        assert closed.count(fig) == 1
        export_key = (*key, frozenset({"png", "pdf"}))
        packed = (b"", b"RETAIN_PNG", b"RETAIN_PDF", 2, ("Sand",), ("gap",))
        assert services._export_memo[export_key] == packed
        assert key not in services._figure_memo
    finally:
        clear_service_memos()


def test_retain_prepare_corrupt_keys_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_service_memos()
    import app_services as services

    closed = _tracking_close(monkeypatch)
    redraw = _patch_redraw_path(monkeypatch)

    fig = object()
    subset_json = "subset-retain-corrupt"
    request_json = "request-retain-corrupt"
    key = _request_key(subset_json, request_json)
    # Missing renderer / polygons / projected — incomplete retain bundle.
    _store_figure_memo(key, {"figure": fig})

    try:
        png, pdf = services.cached_build_section_exports(subset_json, request_json)
        assert png == b"REDRAW_PNG"
        assert pdf == b"REDRAW_PDF"
        assert redraw["render"] == 1
        assert fig in closed
        assert closed.count(fig) == 1
        assert key not in services._figure_memo
    finally:
        clear_service_memos()


def test_retain_prepare_export_raises_falls_back_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_service_memos()
    closed = _tracking_close(monkeypatch)
    redraw = _patch_redraw_path(monkeypatch)

    class _BoomRenderer:
        def export_figure_bytes(self, *_args: object, **_kwargs: object):
            raise RuntimeError("export failed")

    fig = object()
    subset_json = "subset-retain-boom"
    request_json = "request-retain-boom"
    key = _request_key(subset_json, request_json)
    _store_figure_memo(
        key,
        {
            "figure": fig,
            "renderer": _BoomRenderer(),
            "polygons": [],
            "projected": object(),
        },
    )

    try:
        import app_services as services

        png, pdf = services.cached_build_section_exports(subset_json, request_json)
        assert png == b"REDRAW_PNG"
        assert pdf == b"REDRAW_PDF"
        assert redraw["render"] == 1
        assert fig in closed
        assert closed.count(fig) == 1
    finally:
        clear_service_memos()


def test_close_figure_bundle_swallows_errors_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app_services as services

    clear_service_memos()
    fig = object()
    bundle: dict[str, object] = {"figure": fig}
    closes: list[object] = []

    def _boom_close(figure: object) -> None:
        closes.append(figure)
        raise RuntimeError("plt.close failed")

    monkeypatch.setattr(
        "matplotlib.pyplot.close",
        _boom_close,
        raising=False,
    )
    # Import path used inside _close_figure_bundle.
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "close", _boom_close)

    services._close_figure_bundle(bundle)
    assert bundle.get("figure") is None
    assert closes == [fig]
    # Second close must not re-enter plt.close or raise.
    services._close_figure_bundle(bundle)
    assert closes == [fig]


def test_retain_prepare_none_figure_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_service_memos()
    redraw = _patch_redraw_path(monkeypatch)

    subset_json = "subset-retain-none-fig"
    request_json = "request-retain-none-fig"
    key = _request_key(subset_json, request_json)

    class _Renderer:
        def export_figure_bytes(self, *_a: object, **_k: object):
            raise AssertionError("must not export with None figure")

    _store_figure_memo(
        key,
        {
            "figure": None,
            "renderer": _Renderer(),
            "polygons": [],
            "projected": object(),
        },
    )
    try:
        import app_services as services

        png, pdf = services.cached_build_section_exports(subset_json, request_json)
        assert png == b"REDRAW_PNG"
        assert pdf == b"REDRAW_PDF"
        assert redraw["render"] == 1
        assert key not in services._figure_memo
    finally:
        clear_service_memos()


def test_retain_prepare_non_bytes_falls_back_no_memo_pollution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_service_memos()
    import app_services as services

    closed = _tracking_close(monkeypatch)
    redraw = _patch_redraw_path(monkeypatch)

    class _BadRenderer:
        def export_figure_bytes(self, *_a: object, **_k: object):
            return b"", None, "not-bytes"

    fig = object()
    subset_json = "subset-retain-nonbytes"
    request_json = "request-retain-nonbytes"
    key = _request_key(subset_json, request_json)
    export_key = (*key, frozenset({"png", "pdf"}))
    _store_figure_memo(
        key,
        {
            "figure": fig,
            "renderer": _BadRenderer(),
            "polygons": [],
            "projected": object(),
        },
    )
    try:
        png, pdf = services.cached_build_section_exports(subset_json, request_json)
        assert png == b"REDRAW_PNG"
        assert pdf == b"REDRAW_PDF"
        assert redraw["render"] == 1
        assert fig in closed
        # Redraw may populate export memo; must not store the bad None/"not-bytes" retain.
        packed = services._export_memo.get(export_key)
        assert packed is not None
        assert packed[1] == b"REDRAW_PNG"
        assert packed[2] == b"REDRAW_PDF"
        assert isinstance(packed[1], bytes)
    finally:
        clear_service_memos()


def test_retain_fail_skips_stale_export_memo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Failed retain must redraw — not silently return a pre-Generate export memo hit."""
    clear_service_memos()
    import app_services as services

    closed = _tracking_close(monkeypatch)
    redraw = _patch_redraw_path(monkeypatch, png=b"FRESH_PNG", pdf=b"FRESH_PDF")

    class _BoomRenderer:
        def export_figure_bytes(self, *_a: object, **_k: object):
            raise RuntimeError("export failed")

    fig = object()
    subset_json = "subset-retain-stale-memo"
    request_json = "request-retain-stale-memo"
    key = _request_key(subset_json, request_json)
    export_key = (*key, frozenset({"png", "pdf"}))
    stale = (b"", b"STALE_PNG", b"STALE_PDF", 9, ("Old",), ("stale",))
    with services._memo_lock:
        services._export_memo[export_key] = stale
    _store_figure_memo(
        key,
        {
            "figure": fig,
            "renderer": _BoomRenderer(),
            "polygons": [],
            "projected": object(),
        },
    )
    try:
        png, pdf = services.cached_build_section_exports(subset_json, request_json)
        assert png == b"FRESH_PNG"
        assert pdf == b"FRESH_PDF"
        assert (png, pdf) != (stale[1], stale[2])
        assert redraw["render"] == 1
        assert fig in closed
        packed = services._export_memo[export_key]
        assert packed[1] == b"FRESH_PNG"
        assert packed[2] == b"FRESH_PDF"
    finally:
        clear_service_memos()


def test_retain_success_skips_memo_if_figure_reappeared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent Generate re-retain: do not overwrite export memo with the older export."""
    clear_service_memos()
    import app_services as services

    closed = _tracking_close(monkeypatch)
    redraw = _patch_redraw_path(monkeypatch)

    subset_json = "subset-retain-race"
    request_json = "request-retain-race"
    key = _request_key(subset_json, request_json)
    export_key = (*key, frozenset({"png", "pdf"}))
    newer_fig = object()

    class _Renderer:
        def export_figure_bytes(self, figure: object, formats: frozenset[str], **_k: object):
            # Simulate Generate storing a newer figure mid-export.
            _store_figure_memo(
                key,
                {
                    "figure": newer_fig,
                    "renderer": self,
                    "polygons": [],
                    "projected": object(),
                },
            )
            return b"", b"OLD_RETAIN_PNG", b"OLD_RETAIN_PDF"

    fig = object()
    _store_figure_memo(
        key,
        {
            "figure": fig,
            "renderer": _Renderer(),
            "polygons": [],
            "projected": object(),
            "polygon_count": 1,
            "lithology_codes_tuple": ("Sand",),
            "overlap_warnings": (),
        },
    )
    try:
        png, pdf = services.cached_build_section_exports(subset_json, request_json)
        assert png == b"OLD_RETAIN_PNG"
        assert pdf == b"OLD_RETAIN_PDF"
        assert redraw["render"] == 0
        assert fig in closed
        # Newer live figure kept; export memo must not be polluted with the older retain.
        assert export_key not in services._export_memo
        assert services._figure_memo[key]["figure"] is newer_fig
    finally:
        clear_service_memos()


def test_retain_prepare_wrong_formats_skips_retain_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_run_build_cross_section`` must not consume retain (Prepare-only path)."""
    clear_service_memos()
    import app_services as services

    closed = _tracking_close(monkeypatch)
    redraw = _patch_redraw_path(monkeypatch)

    class _Renderer:
        def export_figure_bytes(self, *_a: object, **_k: object):
            raise AssertionError("retain export must not run from draw path")

    fig = object()
    subset = _minimal_subset()
    request = _minimal_request()
    subset_json = "subset-retain-fmt"
    request_json = "request-retain-fmt"
    key = _request_key(subset_json, request_json)
    _store_figure_memo(
        key,
        {
            "figure": fig,
            "renderer": _Renderer(),
            "polygons": [],
            "projected": object(),
        },
    )
    try:
        packed = _run_build_cross_section(
            subset,
            request,
            export_formats=frozenset({"png", "pdf"}),
            subset_json=subset_json,
            request_json=request_json,
        )
        assert packed[1] == b"REDRAW_PNG"
        assert redraw["render"] == 1
        assert fig not in closed
        assert key in services._figure_memo
    finally:
        clear_service_memos()

def test_prepare_exports_retain_bypasses_warm_streamlit_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public Prepare wrapper prefers live retain even when Streamlit draw cache is warm."""
    clear_service_memos()
    import app_services as services

    closed = _tracking_close(monkeypatch)
    draw_calls = {"n": 0}

    def _stale_draw(_subset_json: str, _request_json: str) -> tuple[bytes, bytes]:
        draw_calls["n"] += 1
        return b"STALE_STREAMLIT_PNG", b"STALE_STREAMLIT_PDF"

    monkeypatch.setattr(services, "_cached_build_section_exports_draw", _stale_draw)

    class _Renderer:
        def export_figure_bytes(self, figure: object, formats: frozenset[str], **_kwargs: object):
            return b"", b"LIVE_RETAIN_PNG", b"LIVE_RETAIN_PDF"

    fig = object()
    subset_json = "subset-warm-st-bypass"
    request_json = "request-warm-st-bypass"
    key = _request_key(subset_json, request_json)
    _store_figure_memo(
        key,
        {
            "figure": fig,
            "renderer": _Renderer(),
            "polygons": [],
            "projected": object(),
            "polygon_count": 3,
            "lithology_codes_tuple": ("Clay",),
            "overlap_warnings": (),
        },
    )
    try:
        png, pdf = services.cached_build_section_exports(subset_json, request_json)
        assert png == b"LIVE_RETAIN_PNG"
        assert pdf == b"LIVE_RETAIN_PDF"
        assert draw_calls["n"] == 0
        assert fig in closed
        assert key not in services._figure_memo
        export_key = (*key, frozenset({"png", "pdf"}))
        assert services._export_memo[export_key][1] == b"LIVE_RETAIN_PNG"
    finally:
        clear_service_memos()


def test_prepare_exports_retain_works_when_streamlit_clear_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Correctness must not require ``_invalidate_streamlit_prepare_caches``."""
    clear_service_memos()
    import app_services as services

    closed = _tracking_close(monkeypatch)
    draw_calls = {"n": 0}

    def _stale_draw(_subset_json: str, _request_json: str) -> tuple[bytes, bytes]:
        draw_calls["n"] += 1
        return b"WARM_PNG", b"WARM_PDF"

    monkeypatch.setattr(services, "_cached_build_section_exports_draw", _stale_draw)
    monkeypatch.setattr(services, "_invalidate_streamlit_prepare_caches", lambda: None)

    class _Renderer:
        def export_figure_bytes(self, *_a: object, **_k: object):
            return b"", b"RETAIN_AFTER_NOOP_CLEAR_PNG", b"RETAIN_AFTER_NOOP_CLEAR_PDF"

    fig = object()
    subset_json = "subset-clear-noop"
    request_json = "request-clear-noop"
    key = _request_key(subset_json, request_json)
    _store_figure_memo(
        key,
        {
            "figure": fig,
            "renderer": _Renderer(),
            "polygons": [],
            "projected": object(),
            "polygon_count": 1,
            "lithology_codes_tuple": ("Sand",),
            "overlap_warnings": (),
        },
    )
    try:
        # Simulate Generate's post-retain clear being a noop; Prepare must still consume.
        services._invalidate_streamlit_prepare_caches()
        png, pdf = services.cached_build_section_exports(subset_json, request_json)
        assert png == b"RETAIN_AFTER_NOOP_CLEAR_PNG"
        assert pdf == b"RETAIN_AFTER_NOOP_CLEAR_PDF"
        assert draw_calls["n"] == 0
        assert fig in closed
        assert key not in services._figure_memo
    finally:
        clear_service_memos()


def test_prepare_second_call_uses_export_memo_not_stale_streamlit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After retain consume, a repeat Prepare must not resurrect pre-Generate Streamlit bytes."""
    clear_service_memos()
    import app_services as services

    closed = _tracking_close(monkeypatch)
    draw_calls = {"n": 0}

    def _stale_draw(_subset_json: str, _request_json: str) -> tuple[bytes, bytes]:
        draw_calls["n"] += 1
        return b"STALE_SECOND_PNG", b"STALE_SECOND_PDF"

    monkeypatch.setattr(services, "_cached_build_section_exports_draw", _stale_draw)
    monkeypatch.setattr(services, "_invalidate_streamlit_prepare_caches", lambda: None)

    class _Renderer:
        def export_figure_bytes(self, *_a: object, **_k: object):
            return b"", b"RETAIN_SEED_PNG", b"RETAIN_SEED_PDF"

    fig = object()
    subset_json = "subset-second-prepare"
    request_json = "request-second-prepare"
    key = _request_key(subset_json, request_json)
    _store_figure_memo(
        key,
        {
            "figure": fig,
            "renderer": _Renderer(),
            "polygons": [],
            "projected": object(),
            "polygon_count": 1,
            "lithology_codes_tuple": ("Sand",),
            "overlap_warnings": (),
        },
    )
    try:
        first = services.cached_build_section_exports(subset_json, request_json)
        second = services.cached_build_section_exports(subset_json, request_json)
        assert first == (b"RETAIN_SEED_PNG", b"RETAIN_SEED_PDF")
        assert second == first
        assert draw_calls["n"] == 0
        assert fig in closed
        assert key not in services._figure_memo
    finally:
        clear_service_memos()


def test_prepare_failed_retain_bypasses_stale_streamlit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed retain must uncached-redraw — not return warm Streamlit when clear is a noop."""
    clear_service_memos()
    import app_services as services

    closed = _tracking_close(monkeypatch)
    draw_calls = {"n": 0}
    uncached_calls = {"n": 0}

    def _stale_draw(_subset_json: str, _request_json: str) -> tuple[bytes, bytes]:
        draw_calls["n"] += 1
        return b"STALE_AFTER_FAIL_PNG", b"STALE_AFTER_FAIL_PDF"

    def _fresh_uncached(_subset_json: str, _request_json: str) -> tuple[bytes, bytes]:
        uncached_calls["n"] += 1
        return b"FRESH_UNCACHED_PNG", b"FRESH_UNCACHED_PDF"

    monkeypatch.setattr(services, "_cached_build_section_exports_draw", _stale_draw)
    monkeypatch.setattr(services, "_build_prepare_exports_uncached", _fresh_uncached)
    monkeypatch.setattr(services, "_invalidate_streamlit_prepare_caches", lambda: None)

    fig = object()
    subset_json = "subset-fail-bypass-st"
    request_json = "request-fail-bypass-st"
    key = _request_key(subset_json, request_json)
    # Incomplete bundle → retain pop + export fail.
    _store_figure_memo(key, {"figure": fig})
    try:
        png, pdf = services.cached_build_section_exports(subset_json, request_json)
        assert png == b"FRESH_UNCACHED_PNG"
        assert pdf == b"FRESH_UNCACHED_PDF"
        assert uncached_calls["n"] == 1
        assert draw_calls["n"] == 0
        assert fig in closed
        assert key not in services._figure_memo
    finally:
        clear_service_memos()


def test_png_pdf_wrappers_delegate_to_public_exports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PNG/PDF helpers must not call the inner Streamlit draw cache directly."""
    clear_service_memos()
    import app_services as services

    closed = _tracking_close(monkeypatch)
    draw_calls = {"n": 0}

    def _stale_draw(_subset_json: str, _request_json: str) -> tuple[bytes, bytes]:
        draw_calls["n"] += 1
        return b"WRAPPER_STALE_PNG", b"WRAPPER_STALE_PDF"

    monkeypatch.setattr(services, "_cached_build_section_exports_draw", _stale_draw)

    class _Renderer:
        def export_figure_bytes(self, *_a: object, **_k: object):
            return b"", b"WRAPPER_RETAIN_PNG", b"WRAPPER_RETAIN_PDF"

    fig = object()
    subset_json = "subset-wrapper-path"
    request_json = "request-wrapper-path"
    key = _request_key(subset_json, request_json)
    _store_figure_memo(
        key,
        {
            "figure": fig,
            "renderer": _Renderer(),
            "polygons": [],
            "projected": object(),
            "polygon_count": 1,
            "lithology_codes_tuple": ("Sand",),
            "overlap_warnings": (),
        },
    )
    try:
        assert services.cached_build_section_png(subset_json, request_json) == b"WRAPPER_RETAIN_PNG"
        # Figure already consumed; PDF wrapper must use retain-seeded export memo, not Streamlit.
        assert services.cached_build_section_pdf(subset_json, request_json) == b"WRAPPER_RETAIN_PDF"
        assert draw_calls["n"] == 0
        assert fig in closed
    finally:
        clear_service_memos()


def test_generate_live_memo_returns_streamlit_svg_without_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``_figure_memo`` already has a live figure, Generate must not re-draw."""
    clear_service_memos()
    import app_services as services

    draw_calls = {"n": 0}
    uncached_calls = {"n": 0}

    def _warm_draw(_s: str, _r: str) -> tuple[bytes, bytes, bytes, int, tuple[str, ...], tuple[str, ...]]:
        draw_calls["n"] += 1
        return b"LIVE_MEMO_SVG", b"", b"", 2, ("Sand",), ()

    def _uncached(_s: str, _r: str) -> tuple[bytes, bytes, bytes, int, tuple[str, ...], tuple[str, ...]]:
        uncached_calls["n"] += 1
        return b"SHOULD_NOT_RUN", b"", b"", 0, (), ()

    monkeypatch.setattr(services, "_cached_build_section_svg_draw", _warm_draw)
    monkeypatch.setattr(services, "_build_section_svg_uncached", _uncached)

    fig = object()
    subset_json = "subset-live-memo-gen"
    request_json = "request-live-memo-gen"
    key = _request_key(subset_json, request_json)
    _store_figure_memo(
        key,
        {
            "figure": fig,
            "renderer": object(),
            "polygons": [],
            "projected": object(),
        },
    )
    try:
        packed = services.cached_build_section(subset_json, request_json)
        assert packed[0] == b"LIVE_MEMO_SVG"
        assert draw_calls["n"] == 1
        assert uncached_calls["n"] == 0
        assert services._figure_memo[key]["figure"] is fig
    finally:
        clear_service_memos()


def test_generate_warm_svg_cache_empty_memo_re_retains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warm Streamlit SVG + empty memo must still uncached re-retain; return cached SVG."""
    clear_service_memos()
    import app_services as services

    closed = _tracking_close(monkeypatch)
    redraw = _patch_redraw_path(
        monkeypatch,
        png=b"RETAIN_PNG",
        pdf=b"RETAIN_PDF",
        svg=b"FRESH_SVG",
        retain_figures=True,
    )
    monkeypatch.setattr(
        services,
        "_cached_section_inputs",
        lambda _s, _r: (_minimal_subset(), _minimal_request()),
    )

    warm_calls = {"n": 0}

    def _warm_svg(
        _subset_json: str, _request_json: str
    ) -> tuple[bytes, bytes, bytes, int, tuple[str, ...], tuple[str, ...]]:
        warm_calls["n"] += 1
        return b"WARM_CACHED_SVG", b"", b"", 1, ("Clay",), ()

    monkeypatch.setattr(services, "_cached_build_section_svg_draw", _warm_svg)

    subset_json = "subset-warm-svg-empty-memo"
    request_json = "request-warm-svg-empty-memo"
    key = _request_key(subset_json, request_json)
    try:
        packed = services.cached_build_section(subset_json, request_json)
        assert packed[0] == b"WARM_CACHED_SVG"
        assert warm_calls["n"] == 1
        assert redraw["render"] == 1
        assert key in services._figure_memo
        assert services._figure_memo[key]["figure"] is not None
        assert services._figure_memo[key]["figure"] in redraw["retained_figs"]
        assert not closed  # re-retain must leave figure open for Prepare
    finally:
        clear_service_memos()


def test_generate_miss_failed_retain_does_not_double_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streamlit miss that already attempted retain must not force a second uncached rebuild."""
    clear_service_memos()
    import app_services as services

    uncached_calls = {"n": 0}

    def _miss_no_retain(
        _subset_json: str, _request_json: str
    ) -> tuple[bytes, bytes, bytes, int, tuple[str, ...], tuple[str, ...]]:
        # Real ``@st.cache_data`` miss path notes the body flag before retain.
        services._note_svg_draw_body_ran()
        return b"MISS_NO_RETAIN_SVG", b"", b"", 0, (), ()

    def _uncached(
        _subset_json: str, _request_json: str
    ) -> tuple[bytes, bytes, bytes, int, tuple[str, ...], tuple[str, ...]]:
        uncached_calls["n"] += 1
        return b"DOUBLE_DRAW", b"", b"", 0, (), ()

    monkeypatch.setattr(services, "_cached_build_section_svg_draw", _miss_no_retain)
    monkeypatch.setattr(services, "_build_section_svg_uncached", _uncached)

    subset_json = "subset-miss-no-double"
    request_json = "request-miss-no-double"
    key = _request_key(subset_json, request_json)
    try:
        packed = services.cached_build_section(subset_json, request_json)
        assert packed[0] == b"MISS_NO_RETAIN_SVG"
        assert uncached_calls["n"] == 0
        assert key not in services._figure_memo
        assert not services._svg_draw_body_ran()
    finally:
        clear_service_memos()


def test_svg_draw_body_flag_reset_hygiene(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """clear_service_memos and Generate exit both leave ``_svg_draw_body_ran()`` False."""
    clear_service_memos()
    import app_services as services

    services._note_svg_draw_body_ran()
    assert services._svg_draw_body_ran()
    clear_service_memos()
    assert not services._svg_draw_body_ran()

    noted_during_draw = {"value": False}

    def _miss_notes(
        _subset_json: str, _request_json: str
    ) -> tuple[bytes, bytes, bytes, int, tuple[str, ...], tuple[str, ...]]:
        services._note_svg_draw_body_ran()
        noted_during_draw["value"] = services._svg_draw_body_ran()
        return b"FLAG_HYGIENE_SVG", b"", b"", 0, (), ()

    monkeypatch.setattr(services, "_cached_build_section_svg_draw", _miss_notes)
    monkeypatch.setattr(
        services,
        "_build_section_svg_uncached",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("unexpected uncached")),
    )

    try:
        packed = services.cached_build_section("subset-flag-hygiene", "request-flag-hygiene")
        assert packed[0] == b"FLAG_HYGIENE_SVG"
        assert noted_during_draw["value"] is True
        assert not services._svg_draw_body_ran()
    finally:
        clear_service_memos()

    def _raise_after_note(
        _subset_json: str, _request_json: str
    ) -> tuple[bytes, bytes, bytes, int, tuple[str, ...], tuple[str, ...]]:
        services._note_svg_draw_body_ran()
        raise RuntimeError("forced generate failure")

    monkeypatch.setattr(services, "_cached_build_section_svg_draw", _raise_after_note)
    with pytest.raises(RuntimeError, match="forced generate failure"):
        services.cached_build_section("subset-flag-raise", "request-flag-raise")
    assert not services._svg_draw_body_ran()


def test_generate_live_figure_stolen_during_warm_draw_re_retains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent Prepare pop during warm SVG hit must still uncached re-retain (TOCTOU)."""
    clear_service_memos()
    import app_services as services

    closed = _tracking_close(monkeypatch)
    redraw = _patch_redraw_path(
        monkeypatch,
        png=b"RETAIN_PNG",
        pdf=b"RETAIN_PDF",
        svg=b"FRESH_SVG",
        retain_figures=True,
    )
    monkeypatch.setattr(
        services,
        "_cached_section_inputs",
        lambda _s, _r: (_minimal_subset(), _minimal_request()),
    )

    subset_json = "subset-live-stolen"
    request_json = "request-live-stolen"
    key = _request_key(subset_json, request_json)
    stolen = object()
    _store_figure_memo(
        key,
        {
            "figure": stolen,
            "renderer": object(),
            "polygons": [],
            "projected": object(),
        },
    )

    def _warm_steal(
        _subset_json: str, _request_json: str
    ) -> tuple[bytes, bytes, bytes, int, tuple[str, ...], tuple[str, ...]]:
        # Simulate Prepare consuming the live figure between check and return.
        with services._memo_lock:
            popped = services._figure_memo.pop(key, None)
        services._close_figure_bundle(popped)
        return b"WARM_AFTER_STEAL_SVG", b"", b"", 1, ("Sand",), ()

    monkeypatch.setattr(services, "_cached_build_section_svg_draw", _warm_steal)

    try:
        packed = services.cached_build_section(subset_json, request_json)
        assert packed[0] == b"WARM_AFTER_STEAL_SVG"
        assert redraw["render"] == 1
        assert key in services._figure_memo
        assert services._figure_memo[key]["figure"] is not None
        assert services._figure_memo[key]["figure"] is not stolen
        assert stolen in closed
        assert services._figure_memo[key]["figure"] not in closed
    finally:
        clear_service_memos()


def test_generate_prepare_generate_leaves_live_figure_second_prepare_zero_redraw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate→Prepare→Generate re-retains; next Prepare uses retain with no full redraw."""
    clear_service_memos()
    import app_services as services

    closed = _tracking_close(monkeypatch)
    redraw = _patch_redraw_path(
        monkeypatch,
        png=b"PREPARE_PNG",
        pdf=b"PREPARE_PDF",
        svg=b"GEN_SVG",
        retain_figures=True,
    )
    monkeypatch.setattr(
        services,
        "_cached_section_inputs",
        lambda _s, _r: (_minimal_subset(), _minimal_request()),
    )

    # Simulate Streamlit SVG cache: cold miss retains once; later hits return bytes only.
    svg_cache: dict[str, object] = {"packed": None}

    def _svg_draw(
        subset_json: str, request_json: str
    ) -> tuple[bytes, bytes, bytes, int, tuple[str, ...], tuple[str, ...]]:
        if svg_cache["packed"] is None:
            svg_cache["packed"] = services._build_section_svg_uncached(subset_json, request_json)
        return svg_cache["packed"]  # type: ignore[return-value]

    monkeypatch.setattr(services, "_cached_build_section_svg_draw", _svg_draw)
    # Prepare draw must not run if retain / export memo succeed.
    monkeypatch.setattr(
        services,
        "_cached_build_section_exports_draw",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("unexpected Streamlit Prepare draw")),
    )

    subset_json = "subset-gen-prep-gen"
    request_json = "request-gen-prep-gen"
    key = _request_key(subset_json, request_json)
    try:
        first = services.cached_build_section(subset_json, request_json)
        assert first[0] == b"GEN_SVG"
        assert key in services._figure_memo
        assert services._figure_memo[key]["figure"] is not None
        assert redraw["render"] == 1

        png1, pdf1 = services.cached_build_section_exports(subset_json, request_json)
        assert png1 == b"PREPARE_PNG"
        assert pdf1 == b"PREPARE_PDF"
        assert key not in services._figure_memo
        assert redraw["render"] == 1  # retain export, no redraw

        second = services.cached_build_section(subset_json, request_json)
        assert second[0] == b"GEN_SVG"  # prefer warm SVG bytes
        assert key in services._figure_memo
        assert services._figure_memo[key]["figure"] is not None
        assert redraw["render"] == 2  # uncached re-retain after warm SVG hit

        # Live memo → Generate must not rebuild again.
        third = services.cached_build_section(subset_json, request_json)
        assert third[0] == b"GEN_SVG"
        assert redraw["render"] == 2

        png2, pdf2 = services.cached_build_section_exports(subset_json, request_json)
        assert png2 == b"PREPARE_PNG"
        assert pdf2 == b"PREPARE_PDF"
        assert redraw["render"] == 2  # second Prepare via retain, zero full redraw
        assert key not in services._figure_memo
        assert len(closed) == 2  # each Prepare closes its retained figure
    finally:
        clear_service_memos()
