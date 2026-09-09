"""Cached ingest and render services."""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import replace
from io import BytesIO
from typing import Any, Callable

import streamlit as st

from ingestion import ingest_workbook
from models import (
    CorrelationOverride,
    DeviationReading,
    ParseResult,
    SectionFigureMetadata,
    Transect,
)
from pipeline import (
    ALL_EXPORT_FORMATS,
    SectionGeometry,
    compute_section_geometry,
    render_cross_section_from_geometry,
    filter_projected_for_interpolation,
    validate_interpretation_mode,
)
from projection import off_transect_warnings, project_boreholes
from section_build_request import SectionBuildRequest
from stratigraphy import (
    CorrelationPairSummary,
    build_stratigraphy,
    detect_polygon_overlaps,
    preview_correlation_health,
)
from transect_planner import recommend_transects

_memo_lock = threading.RLock()

_GEOMETRY_MEMO_MAX = 8
_geometry_memo: OrderedDict[tuple[str, str, str], SectionGeometry] = OrderedDict()

_GEOMETRY_JSON_MEMO_MAX = 32
_geometry_json_memo: OrderedDict[str, str] = OrderedDict()

_EXPORT_MEMO_MAX = 6
_ExportResult = tuple[bytes, bytes, bytes, int, tuple[str, ...], tuple[str, ...]]
_export_memo: OrderedDict[tuple[str, str, str, frozenset[str]], _ExportResult] = OrderedDict()

_FIGURE_MEMO_MAX = 16
_FIGURE_MEMO_MAX_PER_SESSION = 2
_figure_memo: OrderedDict[tuple[str, str, str], dict[str, object]] = OrderedDict()

# Distinguishes Streamlit ``_cached_build_section_svg_draw`` cache hits (body skipped)
# from misses (body ran + retain already attempted). Prevents a second full redraw when
# a miss leaves the memo empty (failed retain) while still allowing warm-hit re-retain.
_svg_draw_body_flag = threading.local()


def _reset_svg_draw_body_flag() -> None:
    _svg_draw_body_flag.ran = False


def _note_svg_draw_body_ran() -> None:
    _svg_draw_body_flag.ran = True


def _svg_draw_body_ran() -> bool:
    return bool(getattr(_svg_draw_body_flag, "ran", False))


def clear_service_memos() -> None:
    """Clear process-local geometry/export/figure memos (tests / long-running workers).

    Also resets the Generate SVG body thread-local flag. Does not clear Streamlit
    ``@st.cache_data`` entries.
    """
    with _memo_lock:
        _geometry_memo.clear()
        _geometry_json_memo.clear()
        _export_memo.clear()
        while _figure_memo:
            _key, bundle = _figure_memo.popitem(last=False)
            _close_figure_bundle(bundle)
    _reset_svg_draw_body_flag()


def _memo_session_id() -> str:
    """Isolate process memos per Streamlit session when a script context exists."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        if ctx is not None and getattr(ctx, "session_id", None):
            return str(ctx.session_id)
    except Exception:
        pass
    return "local"


def _digest_text(text: str) -> str:
    import hashlib

    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _request_memo_key(subset_json: str, request_json: str) -> tuple[str, str, str]:
    """Session-scoped key for ``_figure_memo`` / ``_export_memo`` request identity."""
    return (
        _memo_session_id(),
        _digest_text(subset_json),
        _digest_text(request_json),
    )


def _close_figure_bundle(bundle: dict[str, object] | None) -> None:
    """Close a retained matplotlib figure; never raise (Prepare fallback must stay open).

    Clears ``bundle["figure"]`` after the close attempt so a second call is a no-op
    (avoids double ``plt.close`` / double-tracking in tests and eviction races).
    """
    if not bundle:
        return
    figure = bundle.get("figure")
    if figure is None:
        return
    # Drop the handle first so concurrent/duplicate close is idempotent even if plt.close hangs/fails.
    bundle["figure"] = None
    try:
        from matplotlib import pyplot as plt

        plt.close(figure)
    except Exception:
        # Figure may already be closed or a non-matplotlib test double; never abort callers.
        pass


_REQUIRED_RETAINED_KEYS = frozenset({"figure", "renderer", "polygons", "projected"})


def _try_export_from_retained_figure(
    retained: dict[str, object],
    formats: frozenset[str],
) -> _ExportResult | None:
    """Export PNG/PDF from a retained Generate figure, or ``None`` to force redraw.

    Validates required bundle keys, wraps ``export_figure_bytes`` so any failure
    (missing keys, ``AttributeError``, export raise, non-bytes payloads) yields
    ``None``, and always closes the retained figure exactly once via
    ``_close_figure_bundle`` (close errors must not override the return / fallback).
    """
    try:
        if not _REQUIRED_RETAINED_KEYS.issubset(retained.keys()):
            return None
        figure = retained["figure"]
        renderer = retained["renderer"]
        if figure is None or renderer is None:
            return None
        _svg, png_bytes, pdf_bytes = renderer.export_figure_bytes(  # type: ignore[union-attr]
            figure,
            formats,
            polygons=retained["polygons"],  # type: ignore[arg-type]
            projected_df=retained["projected"],  # type: ignore[arg-type]
            collar_depths=retained.get("collar_depths"),  # type: ignore[arg-type]
            water_levels=retained.get("water_levels"),  # type: ignore[arg-type]
            lithology_codes=retained.get("lithology_codes"),  # type: ignore[arg-type]
            qa_lines=tuple(retained.get("qa_lines") or ()),  # type: ignore[arg-type]
        )
        # Reject non-bytes so we never memoize None / str and greenwash a bad retain.
        if not isinstance(png_bytes, (bytes, bytearray)) or not isinstance(
            pdf_bytes, (bytes, bytearray)
        ):
            return None
        return (
            b"",
            bytes(png_bytes),
            bytes(pdf_bytes),
            int(retained.get("polygon_count") or 0),
            tuple(retained.get("lithology_codes_tuple") or ()),  # type: ignore[arg-type]
            tuple(retained.get("overlap_warnings") or ()),  # type: ignore[arg-type]
        )
    except Exception:
        return None
    finally:
        _close_figure_bundle(retained)


def _session_figure_count(session_id: str) -> int:
    return sum(1 for key in _figure_memo if key[0] == session_id)


def _store_figure_memo(key: tuple[str, str, str], bundle: dict[str, object]) -> None:
    with _memo_lock:
        existing = _figure_memo.get(key)
        if existing is not None and existing is not bundle:
            _close_figure_bundle(existing)
        session_id = key[0]
        while (
            _session_figure_count(session_id) >= _FIGURE_MEMO_MAX_PER_SESSION
            and key not in _figure_memo
        ):
            for old_key in list(_figure_memo):
                if old_key[0] == session_id:
                    _close_figure_bundle(_figure_memo.pop(old_key))
                    break
            else:
                break
        while len(_figure_memo) >= _FIGURE_MEMO_MAX and key not in _figure_memo:
            _old_key, old_bundle = _figure_memo.popitem(last=False)
            _close_figure_bundle(old_bundle)
        _figure_memo[key] = bundle
        _figure_memo.move_to_end(key)


def _drop_export_memo_for_request(session_id: str, subset_digest: str, request_digest: str) -> None:
    with _memo_lock:
        stale = [
            key
            for key in _export_memo
            if key[0] == session_id and key[1] == subset_digest and key[2] == request_digest
        ]
        for key in stale:
            _export_memo.pop(key, None)


def _invalidate_streamlit_prepare_caches() -> None:
    """Best-effort clear of Streamlit Prepare draw caches after SVG retain / Prepare.

    Optional alone for correctness: ``cached_build_section_exports`` prefers live
    ``_figure_memo``, then process ``_export_memo``, and on failed retain forces an
    uncached redraw so a noop ``.clear()`` cannot greenwash pre-Generate bytes.
    Clearing still matters to drop stale draw entries from memory.
    """
    try:
        _cached_build_section_exports_draw.clear()
    except Exception:
        pass
    try:
        cached_build_section_bundle.clear()
    except Exception:
        pass


def _consume_retained_prepare_exports(
    subset_json: str,
    request_json: str,
) -> tuple[tuple[bytes, bytes] | None, bool]:
    """Sole retain consume for Prepare PNG+PDF (pop → export → seed memo).

    Returns ``((png, pdf) | None, force_uncached_draw)``:
    - ``((png, pdf), False)`` on successful retain (seeds ``_export_memo`` when safe;
      best-effort Streamlit clear)
    - ``(None, True)`` when a retain existed but export failed — caller must bypass
      ``@st.cache_data`` so a noop clear cannot serve pre-Generate draw bytes
    - ``(None, False)`` when no retained figure
    """
    request_key = _request_memo_key(subset_json, request_json)
    formats = frozenset({"png", "pdf"})
    export_key = (*request_key, formats)

    with _memo_lock:
        retained = _figure_memo.pop(request_key, None)
    if retained is None:
        return None, False

    packed = _try_export_from_retained_figure(retained, formats)
    if packed is None:
        _drop_export_memo_for_request(request_key[0], request_key[1], request_key[2])
        _invalidate_streamlit_prepare_caches()
        return None, True

    with _memo_lock:
        # Newer Generate may have re-retained under request_key while we
        # exported; skip memo write so the next Prepare prefers that figure.
        if request_key not in _figure_memo:
            _export_memo[export_key] = packed
            while len(_export_memo) > _EXPORT_MEMO_MAX:
                _export_memo.popitem(last=False)
    # Drop any warm Streamlit draw left after a failed Generate-time clear.
    _invalidate_streamlit_prepare_caches()
    return (packed[1], packed[2]), False


def _lookup_prepare_export_memo(
    subset_json: str,
    request_json: str,
) -> tuple[bytes, bytes] | None:
    """Process-local PNG/PDF seeded by retain/draw; consulted before Streamlit draw."""
    export_key = (*_request_memo_key(subset_json, request_json), frozenset({"png", "pdf"}))
    with _memo_lock:
        packed = _export_memo.get(export_key)
        if packed is None:
            return None
        _export_memo.move_to_end(export_key)
        return packed[1], packed[2]


def _memo_get(
    store: OrderedDict,
    key: Any,
    factory: Callable[[], Any],
    *,
    max_entries: int,
) -> Any:
    with _memo_lock:
        cached = store.get(key)
        if cached is not None:
            store.move_to_end(key)
            return cached
    value = factory()
    with _memo_lock:
        store[key] = value
        store.move_to_end(key)
        while len(store) > max_entries:
            store.popitem(last=False)
        return store[key]


def _apply_section_geometry_qa(
    geometry: SectionGeometry,
    request: SectionBuildRequest,
) -> SectionGeometry:
    """Apply request QA flags after geometry cache hit (polygons unchanged)."""
    if request.fail_on_overlaps and geometry.overlap_pairs:
        raise ValueError(
            f"Polygon overlap detected ({len(geometry.overlap_pairs)} pair(s)); "
            "resolve correlation or set fail_on_overlaps=False to export."
        )
    if not request.warn_on_correlation_gaps:
        filtered_warnings = tuple(
            message
            for message in geometry.overlap_warnings
            if not message.startswith("Correlation gap ")
        )
        if filtered_warnings != geometry.overlap_warnings:
            geometry = replace(geometry, overlap_warnings=filtered_warnings)
    return geometry


@st.cache_data(show_spinner="Parsing workbook...", ttl=3600, max_entries=8)
def cached_ingest_workbook(
    file_bytes: bytes,
    profile_id: str | None,
    override_id: str | None,
    elevation_m: float | None,
    target_crs: str | None,
    aliases_json: str,
    auto_assign_unit_order: bool,
) -> tuple[ParseResult, Any]:
    aliases = json.loads(aliases_json)
    return ingest_workbook(
        BytesIO(file_bytes),
        profile_id=profile_id,
        override_id=override_id,
        elevation_m=elevation_m,
        target_crs=target_crs,
        lithology_aliases=aliases,
        auto_assign_unit_order=auto_assign_unit_order,
    )


@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def cached_recommend_transects(
    collars: tuple[Any, ...],
    lithologies: tuple[Any, ...],
    top_n: int,
) -> list:
    return recommend_transects(collars, lithologies, top_n=top_n)


def _build_section_kwargs(
    subset: ParseResult,
    request: SectionBuildRequest,
) -> tuple[SectionFigureMetadata, str, tuple[CorrelationOverride, ...]]:
    elevation_datum = next(
        (collar.elevation_datum for collar in subset.collars if collar.elevation_datum),
        "Collar RL",
    )
    if request.elevation_mode == "relative":
        elevation_datum = "Depth below collar (relative)"
    figure_metadata = request.figure_metadata or SectionFigureMetadata(
        coordinate_reference=request.coordinate_reference,
        elevation_datum=elevation_datum,
        vertical_exaggeration=request.vertical_exaggeration,
        hole_count=len(subset.collars),
        uses_placeholder_elevation=request.uses_placeholder_elevation
        and request.elevation_mode == "absolute",
    )
    mode = validate_interpretation_mode(request.interpretation_mode)
    overrides = tuple(request.correlation_overrides) + tuple(subset.correlation_overrides)
    return figure_metadata, mode, overrides


@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def cached_parse_subset(subset_json: str) -> ParseResult:
    """Parse ``ParseResult`` JSON once; Generate/Prepare/geometry share this cache."""
    return ParseResult.model_validate_json(subset_json)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def cached_parse_request(request_json: str) -> SectionBuildRequest:
    """Parse ``SectionBuildRequest`` JSON once; Generate/Prepare/geometry share this cache."""
    return SectionBuildRequest.model_validate_json(request_json)  # type: ignore[attr-defined]


def _cached_section_inputs(
    subset_json: str, request_json: str
) -> tuple[ParseResult, SectionBuildRequest]:
    return cached_parse_subset(subset_json), cached_parse_request(request_json)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def cached_compute_section_geometry(
    subset_json: str, geometry_request_json: str
) -> SectionGeometry:
    """``SectionGeometry`` shared by Generate SVG and Prepare deliverables (PNG/PDF).

    Streamlit ``@st.cache_data`` serializes the returned object. Prefer
    ``_resolve_section_geometry`` on hot paths so process-local LRU skips unpickle.
    Cache key is geometry-scoped JSON from ``SectionBuildRequest.geometry_cache_payload()``
    (``geometry_request_json``), so cosmetic / render-only fields do not bust this cache.
    """
    subset = cached_parse_subset(subset_json)
    payload = json.loads(geometry_request_json)
    mode = validate_interpretation_mode(str(payload["interpretation_mode"]))
    # Configure / Generate already merge workbook overrides into the request payload.
    # Prefer payload overrides; fall back to subset when the payload omitted them.
    payload_overrides = tuple(
        CorrelationOverride.model_validate(item)
        for item in payload.get("correlation_overrides", ())
    )
    overrides = payload_overrides or tuple(subset.correlation_overrides)
    deviations = tuple(
        DeviationReading.model_validate(item)
        for item in payload.get("deviation_readings", ())
    ) or tuple(subset.deviation_readings)
    transect_points = tuple(
        (float(point[0]), float(point[1])) for point in payload["transect_points"]
    )
    return compute_section_geometry(
        subset.collars,
        subset.lithologies,
        transect_points,
        offset_warning_m=float(payload["offset_warning_m"]),
        interpretation_mode=mode,
        allow_pinch_outs=bool(payload["allow_pinch_outs"]),
        max_offset_for_interpolation_m=payload.get("max_offset_for_interpolation_m"),
        correlation_overrides=overrides,
        deviation_readings=deviations,
        warn_on_correlation_gaps=True,
        fail_on_overlaps=False,
    )


def _resolve_section_geometry(subset_json: str, geometry_request_json: str) -> SectionGeometry:
    """Process-local LRU over ``cached_compute_section_geometry`` (avoids pickle on hits)."""
    geo_key = (
        _memo_session_id(),
        _digest_text(subset_json),
        _digest_text(geometry_request_json),
    )
    return _memo_get(
        _geometry_memo,
        geo_key,
        lambda: cached_compute_section_geometry(subset_json, geometry_request_json),
        max_entries=_GEOMETRY_MEMO_MAX,
    )


def _geometry_request_json(request: SectionBuildRequest, request_digest: str | None = None) -> str:
    """Serialize ``geometry_cache_payload`` once per request digest when possible."""
    if request_digest is not None:
        with _memo_lock:
            cached = _geometry_json_memo.get(request_digest)
            if cached is not None:
                _geometry_json_memo.move_to_end(request_digest)
                return cached
    geometry_json = json.dumps(request.geometry_cache_payload(), sort_keys=True)
    if request_digest is not None:
        with _memo_lock:
            _geometry_json_memo[request_digest] = geometry_json
            while len(_geometry_json_memo) > _GEOMETRY_JSON_MEMO_MAX:
                _geometry_json_memo.popitem(last=False)
    return geometry_json


def _run_build_cross_section(
    subset: ParseResult,
    request: SectionBuildRequest,
    *,
    export_formats: frozenset[str],
    subset_json: str,
    request_json: str | None = None,
) -> tuple[bytes, bytes, bytes, int, tuple[str, ...], tuple[str, ...]]:
    """Draw / export only — does not consume ``_figure_memo`` retain for png+pdf.

    Prepare retain consume is solely via ``cached_build_section_exports`` →
    ``_consume_retained_prepare_exports``. SVG Generate still retains here when
    ``export_formats == {svg}``.
    """
    formats = frozenset(export_formats)
    export_key: tuple[str, str, str, frozenset[str]] | None = None
    request_key: tuple[str, str, str] | None = None
    request_digest: str | None = None
    if request_json is not None:
        request_key = _request_memo_key(subset_json, request_json)
        request_digest = request_key[2]
        export_key = (*request_key, formats)

    # SVG Generate must redraw to re-retain; process ``_export_memo`` is for Prepare
    # png+pdf (and full-bundle) reuse — never short-circuit an SVG retain rebuild.
    retain = request_key is not None and formats == frozenset({"svg"})
    if export_key is not None and not retain:
        with _memo_lock:
            cached = _export_memo.get(export_key)
            if cached is not None:
                _export_memo.move_to_end(export_key)
                return cached
    figure_metadata, mode, _overrides = _build_section_kwargs(subset, request)
    geometry_json = _geometry_request_json(request, request_digest)
    geometry = _apply_section_geometry_qa(
        _resolve_section_geometry(subset_json, geometry_json),
        request,
    )
    result = render_cross_section_from_geometry(
        geometry,
        request.transect_points,
        vertical_exaggeration=request.vertical_exaggeration,
        show_hatches=request.show_hatches,
        show_legend=request.show_legend,
        title=request.section_title,
        interpretation_mode=mode,
        water_levels=request.water_levels or subset.water_levels or None,
        uncertainty_spacing_m=request.uncertainty_spacing_m,
        uncertainty_offset_m=request.uncertainty_offset_m,
        faults=request.faults or subset.faults,
        unconformities=request.unconformities or subset.unconformities,
        environmental_readings=request.environmental_readings or subset.environmental_readings,
        figure_metadata=figure_metadata,
        show_ground_surface=request.show_ground_surface,
        interpolate_water_table=request.interpolate_water_table,
        show_water_elevation_labels=request.show_water_elevation_labels,
        show_water_legend=request.show_water_legend,
        show_dry_well_nm=request.show_dry_well_nm,
        water_interpolate_across_gaps=request.water_interpolate_across_gaps,
        environmental_parameters=request.environmental_parameters,
        show_parameter_labels=request.show_parameter_labels,
        parameter_interpolate_segments=request.parameter_interpolate_segments,
        parameter_interpolate_across_gaps=request.parameter_interpolate_across_gaps,
        parameter_draw_markers=request.parameter_draw_markers,
        parameter_marker_size=request.parameter_marker_size,
        parameter_draw_leaders=request.parameter_draw_leaders,
        parameter_label_include_units=request.parameter_label_include_units,
        column_header_detail=request.column_header_detail,
        show_scale_bar=request.show_scale_bar,
        show_ve_annotation=request.show_ve_annotation,
        show_parameter_legend_text=request.show_parameter_legend_text,
        export_font_family=request.export_font_family,
        export_font_size=request.export_font_size,
        selected_water_series_ids=request.selected_water_series_ids or None,
        water_line_solid=request.water_line_solid,
        legend_ncol=request.legend_ncol,
        chemistry_color_mode=request.chemistry_color_mode,
        chemistry_threshold_green_max=request.chemistry_threshold_green_max,
        chemistry_threshold_yellow_max=request.chemistry_threshold_yellow_max,
        render_layout=request.render_layout,
        track_width_m=request.track_width_m,
        auto_fit_track_width=request.auto_fit_track_width,
        elevation_mode=request.elevation_mode,
        raster_log_strips=request.raster_log_strips,
        export_formats=formats,
        consulting_title_block=request.consulting_title_block,
        screen_intervals=request.screen_intervals or subset.screen_intervals,
        vertical_gradients=request.vertical_gradients or subset.vertical_gradients,
        export_framing=request.export_framing,
        close_figure=not retain,
    )
    if retain and request_key is not None:
        bundle = result.retained_export_bundle
        if bundle is not None:
            bundle["polygon_count"] = len(result.polygons)
            bundle["lithology_codes_tuple"] = tuple(result.lithology_codes)
            bundle["overlap_warnings"] = result.overlap_warnings
            _drop_export_memo_for_request(
                request_key[0], request_key[1], request_key[2]
            )
            _store_figure_memo(request_key, bundle)
            _invalidate_streamlit_prepare_caches()
    packed: _ExportResult = (
        result.svg_bytes,
        result.png_bytes,
        result.pdf_bytes,
        len(result.polygons),
        result.lithology_codes,
        result.overlap_warnings,
    )
    if export_key is not None:
        with _memo_lock:
            _export_memo[export_key] = packed
            while len(_export_memo) > _EXPORT_MEMO_MAX:
                _export_memo.popitem(last=False)
    return packed


@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def cached_build_section_bundle(
    subset_json: str,
    request_json: str,
) -> tuple[bytes, bytes, bytes, int, tuple[str, ...], tuple[str, ...]]:
    """One-shot SVG+PNG+PDF (scripts / full export). Prefer SVG-first Generate + Prepare exports."""
    subset, request = _cached_section_inputs(subset_json, request_json)
    return _run_build_cross_section(
        subset,
        request,
        export_formats=ALL_EXPORT_FORMATS,
        subset_json=subset_json,
        request_json=request_json,
    )


def _has_live_retained_figure(subset_json: str, request_json: str) -> bool:
    """True when ``_figure_memo`` holds a non-``None`` figure for this Generate key."""
    request_key = _request_memo_key(subset_json, request_json)
    with _memo_lock:
        bundle = _figure_memo.get(request_key)
        if bundle is None:
            return False
        return bundle.get("figure") is not None


def _build_section_svg_uncached(
    subset_json: str,
    request_json: str,
) -> tuple[bytes, bytes, bytes, int, tuple[str, ...], tuple[str, ...]]:
    """SVG Generate draw without Streamlit cache (re-retain / cold-path escape hatch)."""
    subset, request = _cached_section_inputs(subset_json, request_json)
    svg, _png, _pdf, count, codes, warnings = _run_build_cross_section(
        subset,
        request,
        export_formats=frozenset({"svg"}),
        subset_json=subset_json,
        request_json=request_json,
    )
    return svg, b"", b"", count, codes, warnings


# ---------------------------------------------------------------------------
# Generate / Prepare figure state machine
#
#   Generate (``cached_build_section``):
#     SVG draw → retain live figure in ``_figure_memo`` (``close_figure=False``).
#     Warm Streamlit SVG hit + empty memo → uncached re-retain (body-ran / TOCTOU).
#
#   Prepare (``cached_build_section_exports``):
#     1. ``_consume_retained_prepare_exports`` — sole pop/export of live figure
#     2. Failed retain → uncached redraw (never warm Streamlit after noop clear)
#     3. Process ``_export_memo`` (retain-seeded bytes)
#     4. Streamlit ``_cached_build_section_exports_draw``
#
#   ``_run_build_cross_section`` draws / memoizes only; it does not consume retain.
# ---------------------------------------------------------------------------


def cached_build_section(
    subset_json: str,
    request_json: str,
) -> tuple[bytes, bytes, bytes, int, tuple[str, ...], tuple[str, ...]]:
    """Generate path: SVG with live figure retain even on Streamlit cache hits.

    Not ``@st.cache_data`` — after Prepare consumes ``_figure_memo``, a same-key
    Streamlit SVG hit must still re-retain. Order:
    1. Call Streamlit SVG draw (miss retains inside the body; hit returns bytes only)
    2. Live ``_figure_memo`` after draw → return those SVG bytes (no second rebuild)
    3. Empty memo after a warm hit (body skipped) → uncached retain rebuild for
       side-effect only; still return the warm SVG bytes
    4. Empty memo after a miss (body already ran) → do **not** rebuild again
       (avoids double-draw / session-cap thrash when retain failed)

    Re-checks memo after the draw so a concurrent Prepare that pops a live figure
    during a warm hit still falls through to uncached re-retain (TOCTOU).

    The body-ran flag is always cleared on exit (return or raise) so process-local
    retain-machine state does not leak across Generate calls.
    """
    _reset_svg_draw_body_flag()
    try:
        packed = _cached_build_section_svg_draw(subset_json, request_json)
        if _has_live_retained_figure(subset_json, request_json):
            return packed
        # Warm Streamlit hit skipped retain — rebuild uncached for side-effect only.
        # Skip when the draw body already ran (miss path attempted retain once).
        if not _svg_draw_body_ran():
            _build_section_svg_uncached(subset_json, request_json)
        return packed
    finally:
        _reset_svg_draw_body_flag()


@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def _cached_build_section_svg_draw(
    subset_json: str,
    request_json: str,
) -> tuple[bytes, bytes, bytes, int, tuple[str, ...], tuple[str, ...]]:
    """Inner Generate SVG draw cache; retain runs only on Streamlit miss."""
    _note_svg_draw_body_ran()
    return _build_section_svg_uncached(subset_json, request_json)


def cached_build_section_png(
    subset_json: str,
    request_json: str,
) -> bytes:
    """PNG-only Prepare; delegates to ``cached_build_section_exports`` (shared draw cache)."""
    png, _pdf = cached_build_section_exports(subset_json, request_json)
    return png


def cached_build_section_pdf(
    subset_json: str,
    request_json: str,
) -> bytes:
    """PDF-only Prepare; delegates to ``cached_build_section_exports`` (shared draw cache)."""
    _png, pdf = cached_build_section_exports(subset_json, request_json)
    return pdf


def _build_prepare_exports_uncached(
    subset_json: str,
    request_json: str,
) -> tuple[bytes, bytes]:
    """One matplotlib PNG+PDF draw without Streamlit cache (failed-retain escape hatch)."""
    subset, request = _cached_section_inputs(subset_json, request_json)
    _svg, png, pdf, _count, _codes, _warnings = _run_build_cross_section(
        subset,
        request,
        export_formats=frozenset({"png", "pdf"}),
        subset_json=subset_json,
        request_json=request_json,
    )
    return png, pdf


def cached_build_section_exports(
    subset_json: str,
    request_json: str,
) -> tuple[bytes, bytes]:
    """Prepare both: retain → process export memo → Streamlit draw (uncached on fail).

    Not ``@st.cache_data`` — retain consume must run even when Streamlit has a warm
    draw cache hit. Order:
    1. Live ``_figure_memo`` retain (``_consume_retained_prepare_exports``)
    2. On failed retain: uncached redraw (never trust warm Streamlit after noop clear)
    3. Process ``_export_memo`` (retain-seeded bytes outrank stale Streamlit)
    4. ``_cached_build_section_exports_draw``
    """
    retained, force_uncached = _consume_retained_prepare_exports(subset_json, request_json)
    if retained is not None:
        return retained
    if force_uncached:
        return _build_prepare_exports_uncached(subset_json, request_json)
    memoized = _lookup_prepare_export_memo(subset_json, request_json)
    if memoized is not None:
        return memoized
    return _cached_build_section_exports_draw(subset_json, request_json)


@st.cache_data(show_spinner="Preparing PNG/PDF exports...", ttl=3600, max_entries=8)
def _cached_build_section_exports_draw(
    subset_json: str,
    request_json: str,
) -> tuple[bytes, bytes]:
    """Inner Prepare draw cache: one matplotlib draw for PNG+PDF on retain miss."""
    return _build_prepare_exports_uncached(subset_json, request_json)


def preflight_correlation_health(
    subset: ParseResult,
    transect_points: tuple[tuple[float, float], ...],
    *,
    interpretation_mode: str,
    allow_pinch_outs: bool,
    correlation_overrides: tuple[CorrelationOverride, ...],
    offset_warning_m: float = 50.0,
    max_offset_for_interpolation_m: float | None = None,
) -> list[CorrelationPairSummary]:
    """Project transect subset and summarize correlation match rates (UI preflight only)."""
    if interpretation_mode == "borehole_only":
        return []
    projected = project_boreholes(
        subset.collars,
        subset.lithologies,
        Transect(points=list(transect_points)),
        offset_warning_m=offset_warning_m,
        deviation_readings=subset.deviation_readings or None,
    )
    if projected.empty:
        return []
    try:
        projected = filter_projected_for_interpolation(
            projected, max_offset_for_interpolation_m
        )
    except ValueError:
        return []
    return preview_correlation_health(
        projected,
        allow_pinch_outs=allow_pinch_outs,
        correlation_overrides=correlation_overrides,
    )


def preflight_polygon_overlap_warnings(
    subset: ParseResult,
    transect_points: tuple[tuple[float, float], ...],
    *,
    interpretation_mode: str,
    allow_pinch_outs: bool,
    correlation_overrides: tuple[CorrelationOverride, ...],
    offset_warning_m: float = 50.0,
    max_offset_for_interpolation_m: float | None = None,
) -> tuple[str, ...]:
    """Return configure-step warnings when fence polygons overlap (engine-only)."""
    if interpretation_mode == "borehole_only":
        return ()
    projected = project_boreholes(
        subset.collars,
        subset.lithologies,
        Transect(points=list(transect_points)),
        offset_warning_m=offset_warning_m,
        deviation_readings=subset.deviation_readings or None,
    )
    if projected.empty or len(projected["hole_id"].unique()) < 2:
        return ()
    try:
        projected = filter_projected_for_interpolation(
            projected, max_offset_for_interpolation_m
        )
    except ValueError as exc:
        return (str(exc),)
    if len(projected["hole_id"].unique()) < 2:
        return ()
    polygons = build_stratigraphy(
        projected,
        allow_pinch_outs=allow_pinch_outs,
        correlation_overrides=correlation_overrides,
    )
    overlaps = detect_polygon_overlaps(polygons)
    if not overlaps:
        return ()
    return (
        (
            f"Polygon overlap: {len(overlaps)} inter-hole contact conflict(s) detected — "
            "review correlation before export."
        ),
    )


@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def cached_configure_preflight(
    subset_json: str,
    transect_points_json: str,
    interpretation_mode: str,
    allow_pinch_outs: bool,
    correlation_overrides_json: str,
    offset_warning_m: float,
    max_offset_for_interpolation_m: float = 50.0,
    check_overlaps: bool = True,
) -> tuple[tuple[str, ...], tuple[CorrelationPairSummary, ...]]:
    """Cached Configure-step preflight; warms Generate geometry cache when possible."""
    subset = cached_parse_subset(subset_json)
    transect_points: tuple[tuple[float, float], ...] = tuple(
        tuple(point) for point in json.loads(transect_points_json)
    )
    overrides = tuple(
        CorrelationOverride.model_validate(item)
        for item in json.loads(correlation_overrides_json)
    )
    transect = Transect(points=list(transect_points))
    warnings = tuple(
        off_transect_warnings(
            subset.collars,
            transect,
            offset_warning_m,
        )
    )
    if interpretation_mode == "borehole_only":
        return warnings, ()

    preflight_request = SectionBuildRequest(
        transect_points=transect_points,
        interpretation_mode=interpretation_mode,  # type: ignore[arg-type]
        allow_pinch_outs=allow_pinch_outs,
        correlation_overrides=overrides,
        offset_warning_m=offset_warning_m,
        max_offset_for_interpolation_m=max_offset_for_interpolation_m,
    )
    geometry_json = json.dumps(
        preflight_request.geometry_cache_payload(),
        sort_keys=True,
    )
    try:
        geometry = _resolve_section_geometry(subset_json, geometry_json)
    except ValueError as exc:
        message = str(exc)
        if message.startswith("No lithology intervals were projected"):
            return warnings, ()
        return warnings + (message,), ()

    overlap_extra: tuple[str, ...] = ()
    if check_overlaps and geometry.overlap_pairs:
        overlap_extra = (
            (
                f"Polygon overlap: {len(geometry.overlap_pairs)} inter-hole contact conflict(s) detected — "
                "review correlation before export."
            ),
        )

    # Geometry stage already collected pair summaries during build_stratigraphy
    # (cached_compute_section_geometry always enables correlation-gap collection).
    summaries = geometry.correlation_summaries
    if not summaries:
        try:
            filtered = filter_projected_for_interpolation(
                geometry.projected,
                max_offset_for_interpolation_m,
            )
        except ValueError:
            return warnings + overlap_extra, ()
        summaries = tuple(
            preview_correlation_health(
                filtered,
                allow_pinch_outs=allow_pinch_outs,
                correlation_overrides=overrides,
            )
        )
    return warnings + overlap_extra, summaries
