"""Configure step: transect selection helpers, correlation assist, section Q&A."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from app_common import (
    _active_transect_selection,
    _build_assistant,
    _render_lithology_legend,
    _section_facts,
    _session_correlation_overrides,
    _sidebar_heading,
    safe_lithology_index,
)
from app_services import cached_configure_preflight, cached_recommend_transects
from models import CorrelationOverride, ParseResult, subset_parse_result


@dataclass(frozen=True)
class ConfigureState:
    selected_holes: list[str]
    coordinate_text: str
    transect_selection: tuple[tuple[str, ...], tuple[tuple[float, float], ...]] | None
    can_generate: bool
    blocking: bool
    has_warnings: bool
    override_warnings: bool
    placeholder_blocks_interp: bool
    elevation_mode: str
    fail_on_overlaps: bool
    has_overlap_warnings: bool
    environmental_parameters: tuple[str, ...] = ()
    show_parameter_labels: bool = True
    parameter_interpolate_segments: bool = True


def render_transect_sidebar(parse_result: ParseResult, hole_ids: list[str], transect_mode: str) -> tuple[list[str], str]:
    st.divider()
    _sidebar_heading("Stratigraphy legend")
    legend_codes = st.session_state.section_lithology_codes or st.session_state.unique_lithology_codes
    _render_lithology_legend(legend_codes)

    _sidebar_heading("Transect selection")
    render_nl_transect_input(hole_ids)
    selected_holes: list[str] = []
    coordinate_text = ""
    if transect_mode == "Recommended":
        if st.session_state.transect_candidates is None:
            st.session_state.transect_candidates = cached_recommend_transects(
                parse_result.collars,
                parse_result.lithologies,
                3,
            )
        candidates = st.session_state.transect_candidates
        if candidates:
            labels = [
                f"{' → '.join(candidate.hole_ids)} (score {candidate.score:.1f})"
                for candidate in candidates
            ]
            choice = st.selectbox("Recommended transects", options=labels, index=0)
            selected_holes = list(candidates[labels.index(choice)].hole_ids)
        else:
            st.info("Not enough holes for recommendations.")
    elif transect_mode == "By hole sequence":
        if "hole_sequence_multiselect" not in st.session_state:
            st.session_state.hole_sequence_multiselect = hole_ids[: min(4, len(hole_ids))]
        selected_holes = st.multiselect(
            "Hole sequence",
            options=hole_ids,
            key="hole_sequence_multiselect",
        )
    else:
        default_coords = "\n".join(
            f"{collar.easting} {collar.northing}"
            for collar in parse_result.collars[: min(4, len(parse_result.collars))]
        )
        coordinate_text = st.text_area(
            "Transect coordinates (easting northing per line)",
            value=default_coords,
            height=160,
        )
    return selected_holes, coordinate_text


def render_configure_step(
    parse_result: ParseResult,
    *,
    transect_mode: str,
    selected_holes: list[str],
    coordinate_text: str,
    offset_warning_m: float,
    interpretation_mode: str,
    allow_pinch_outs: bool,
    quality_report,
    import_report,
    is_consulting_layout: bool = False,
    max_offset_for_interpolation_m: float | None = None,
) -> ConfigureState:
    st.subheader("Configure")
    st.caption("Choose elevation mode, transect readiness, and export gates before generating.")
    all_hole_ids = [collar.hole_id for collar in parse_result.collars]
    _render_plan_minimap(parse_result, selected_holes, transect_mode)
    if transect_mode == "By hole sequence":
        _render_hole_sequence_order(all_hole_ids)
        selected_holes = list(st.session_state.get("hole_sequence_multiselect") or selected_holes)
    blocking = quality_report is not None and quality_report.has_blocking_errors
    elevation_mode = st.session_state.get("elevation_mode", "absolute")
    if import_report and import_report.uses_placeholder_elevation:
        elevation_mode = st.radio(
            "Elevation mode",
            options=["absolute", "relative"],
            format_func=lambda value: {
                "absolute": "Absolute RL (requires surveyed collar elevation)",
                "relative": "Relative depth below collar (placeholder RL OK)",
            }[value],
            horizontal=True,
            key="elevation_mode",
        )
    placeholder_blocks_interp = (
        import_report is not None
        and import_report.uses_placeholder_elevation
        and elevation_mode == "absolute"
        and interpretation_mode in {"interpolated", "correlation_lines"}
    )
    if placeholder_blocks_interp:
        st.error(
            "All collar elevations use the profile placeholder. Set site elevation in the sidebar "
            "or ingest survey RL before generating an interpreted section."
        )
    has_warnings = quality_report is not None and quality_report.warning_count > 0
    warnings_default = not is_consulting_layout
    override_warnings = st.checkbox(
        "Allow generate with warnings",
        value=warnings_default,
        help="Consulting report preset defaults to blocking export when QA warnings are present.",
    )
    if has_warnings and not override_warnings:
        st.warning(
            "QA warnings are present. Enable **Allow generate with warnings** above to proceed, "
            "or resolve them in Validate."
        )
    fail_on_overlaps = st.checkbox(
        "Block export on polygon overlaps",
        value=is_consulting_layout,
        help="When enabled, generation fails if inter-hole fence polygons overlap.",
    )

    environmental_parameters: tuple[str, ...] = ()
    show_parameter_labels = True
    parameter_interpolate_segments = True
    subset_ready = False
    has_overlap_warnings = False

    preflight_selection = _active_transect_selection(
        parse_result,
        transect_mode,
        selected_holes,
        coordinate_text,
        offset_warning_m,
    )
    if preflight_selection is not None:
        active_ids, active_points = preflight_selection
        section_caption = _transect_section_caption(active_ids)
        st.caption(
            f"Transect: **{' → '.join(active_ids)}** ({len(active_ids)} holes)"
        )
        if section_caption:
            st.caption(f"Section line: **{section_caption}**")
        subset_preflight = None
        try:
            subset_preflight = subset_parse_result(
                parse_result,
                active_ids,
                lithology_index=safe_lithology_index(parse_result),
            )
        except Exception as exc:
            st.error(f"Could not build transect subset for preflight: {exc}")
            subset_preflight = None

        if subset_preflight is None:
            st.caption("Transect subset unavailable — fix selection or workbook data before generating.")
        else:
            subset_ready = True
            available_params = sorted(
                {reading.parameter for reading in subset_preflight.environmental_readings}
            )
            if available_params:
                st.markdown("**Lab / environmental parameters**")
                options_sig = tuple(available_params)
                if st.session_state.get("_env_params_options_sig") != options_sig:
                    previous = list(st.session_state.get("environmental_parameters_multiselect") or [])
                    st.session_state.environmental_parameters_multiselect = [
                        p for p in previous if p in available_params
                    ] or list(available_params)
                    st.session_state._env_params_options_sig = options_sig
                environmental_parameters = tuple(
                    st.multiselect(
                        "Parameters to plot on section",
                        options=available_params,
                        help="Markers and optional fence lines at sample depths (Environmental sheet).",
                        key="environmental_parameters_multiselect",
                    )
                )
                if not environmental_parameters:
                    st.caption("No parameters selected — environmental markers will not be plotted.")
                show_parameter_labels = st.toggle(
                    "Show parameter value labels",
                    value=True,
                    key="show_parameter_labels_toggle",
                )
                parameter_interpolate_segments = st.toggle(
                    "Interpolate parameter between adjacent holes",
                    value=True,
                    key="parameter_interpolate_segments_toggle",
                )
            elif parse_result.environmental_readings:
                st.caption(
                    "Environmental readings exist but none fall on the current transect holes."
                )
            else:
                st.caption(
                    "Add an **Environmental** sheet (`hole_id`, `parameter`, `value`, `depth` or "
                    "`from_depth`/`to_depth`) to plot lab data by depth. See docs/workbook-format.md."
                )
            correlation_overrides = _session_correlation_overrides() + subset_preflight.correlation_overrides
            max_interp_m = float(
                max_offset_for_interpolation_m
                if max_offset_for_interpolation_m is not None
                else offset_warning_m
            )
            overrides_payload = tuple(item.model_dump() for item in correlation_overrides)
            preflight_json_key = (
                tuple(active_ids),
                tuple(active_points),
                interpretation_mode,
                allow_pinch_outs,
                overrides_payload,
                offset_warning_m,
                max_interp_m,
                st.session_state.get("file_hash"),
                st.session_state.get("parse_signature"),
                fail_on_overlaps,
            )
            if st.session_state.get("_preflight_json_key") == preflight_json_key:
                subset_json = st.session_state["_preflight_subset_json"]
                overrides_json = st.session_state["_preflight_overrides_json"]
            else:
                subset_json = subset_preflight.model_dump_json()
                overrides_json = json.dumps(list(overrides_payload))
                st.session_state._preflight_json_key = preflight_json_key
                st.session_state._preflight_subset_json = subset_json
                st.session_state._preflight_overrides_json = overrides_json
            preflight_warnings, pair_summaries = cached_configure_preflight(
                subset_json,
                json.dumps(list(active_points)),
                interpretation_mode,
                allow_pinch_outs,
                overrides_json,
                offset_warning_m,
                max_interp_m,
                check_overlaps=fail_on_overlaps,
            )
            for message in preflight_warnings:
                st.warning(message)
            has_overlap_warnings = any("Polygon overlap" in message for message in preflight_warnings)
            if pair_summaries:
                with st.expander("Correlation health preview", expanded=False):
                    for summary in pair_summaries:
                        st.write(
                            f"**{summary.left_hole_id} → {summary.right_hole_id}**: "
                            f"{summary.matched_count} matched, "
                            f"pinch-out candidates {summary.pinch_out_candidates}, "
                            f"match rate {summary.match_rate:.0%}"
                        )
                        if summary.left_only_codes or summary.right_only_codes:
                            st.caption(
                                f"Left only: {', '.join(summary.left_only_codes) or '—'} · "
                                f"Right only: {', '.join(summary.right_only_codes) or '—'}"
                            )
                    low_match = [s for s in pair_summaries if s.match_rate < 0.5]
                    if low_match:
                        st.info(
                            "Low match rate — consider borehole-only or correlation-lines mode for review."
                        )
                    render_correlation_assist(pair_summaries, subset_preflight, active_ids)

            if len(active_ids) >= 2:
                render_manual_correlation_overrides(active_ids, subset_preflight)

            if _session_correlation_overrides():
                st.info(
                    "Correlation styling locked — manual overrides are active and will persist when the transect changes."
                )

            render_section_qa(subset_preflight, active_ids, preflight_warnings)
    else:
        st.info(
            "Select holes under **Transect selection** in the sidebar "
            "(By hole sequence / Recommended / coordinates)."
        )

    can_generate = (
        parse_result is not None
        and preflight_selection is not None
        and subset_ready
        and not blocking
        and not placeholder_blocks_interp
        and (override_warnings or not has_warnings)
        and (not fail_on_overlaps or not has_overlap_warnings)
    )
    if blocking:
        st.error("Resolve data errors before generating a cross-section.")
    elif fail_on_overlaps and has_overlap_warnings:
        st.error(
            "Polygon overlaps detected. Resolve correlation or disable "
            "'Block export on polygon overlaps' after manual review."
        )

    return ConfigureState(
        selected_holes=selected_holes,
        coordinate_text=coordinate_text,
        transect_selection=preflight_selection,
        can_generate=can_generate,
        blocking=blocking,
        has_warnings=has_warnings,
        override_warnings=override_warnings,
        placeholder_blocks_interp=placeholder_blocks_interp,
        elevation_mode=str(elevation_mode),
        fail_on_overlaps=fail_on_overlaps,
        has_overlap_warnings=has_overlap_warnings,
        environmental_parameters=environmental_parameters,
        show_parameter_labels=show_parameter_labels,
        parameter_interpolate_segments=parameter_interpolate_segments,
    )


def _transect_section_caption(hole_ids: Sequence[str]) -> str:
    if len(hole_ids) < 2:
        return ""
    return f"A–A′ ({hole_ids[0]} → {hole_ids[-1]})"


def _render_plan_minimap(
    parse_result: ParseResult,
    selected_holes: list[str],
    transect_mode: str,
) -> None:
    """Plan-view collar scatter (gINT/Strater-style fence context)."""
    if not parse_result.collars:
        return
    selected_set = set(selected_holes)
    rows = []
    for collar in parse_result.collars:
        rows.append(
            {
                "easting": collar.easting,
                "northing": collar.northing,
                "hole_id": collar.hole_id,
                "selected": collar.hole_id in selected_set,
            }
        )
    df = pd.DataFrame(rows)
    st.markdown("**Plan view (collar locations)**")
    chart_df = df.rename(columns={"easting": "Easting", "northing": "Northing"})
    color_col = "selected" if transect_mode != "Recommended" else None
    if color_col and chart_df["selected"].any():
        st.scatter_chart(
            chart_df,
            x="Easting",
            y="Northing",
            color="selected",
        )
    else:
        st.scatter_chart(chart_df, x="Easting", y="Northing")
    st.caption("Collar positions from workbook — transect follows hole order or line geometry.")


def _render_hole_sequence_order(hole_ids: list[str]) -> None:
    """Numbered hole order with Up/Down (first-class fence sequence)."""
    if "hole_sequence_multiselect" not in st.session_state:
        st.session_state.hole_sequence_multiselect = hole_ids[: min(4, len(hole_ids))]
    sequence: list[str] = list(st.session_state.hole_sequence_multiselect)
    if not sequence:
        return
    st.markdown("**Hole order (fence sequence)**")
    for index, hole_id in enumerate(sequence):
        col_num, col_label, col_up, col_down = st.columns([0.4, 3, 0.5, 0.5])
        with col_num:
            st.markdown(f"**{index + 1}.**")
        with col_label:
            st.markdown(hole_id)
        with col_up:
            if st.button("↑", key=f"hole_order_up_{index}", disabled=index == 0):
                sequence[index - 1], sequence[index] = sequence[index], sequence[index - 1]
                st.session_state.hole_sequence_multiselect = sequence
                st.rerun()
        with col_down:
            if st.button("↓", key=f"hole_order_down_{index}", disabled=index >= len(sequence) - 1):
                sequence[index + 1], sequence[index] = sequence[index], sequence[index + 1]
                st.session_state.hole_sequence_multiselect = sequence
                st.rerun()
    st.caption(f"Section A–A′: **{sequence[0]} → {sequence[-1]}**")


def render_manual_correlation_overrides(active_ids: Sequence[str], subset: ParseResult) -> None:
    with st.expander("Manual correlation overrides", expanded=False):
        st.caption("Pair units between adjacent holes (unit_order on each stick).")
        pair_index = 0
        session_overrides: list[CorrelationOverride] = list(_session_correlation_overrides())
        for left_id, right_id in zip(active_ids, active_ids[1:]):
            left_units = sorted(
                {
                    lith.unit_order
                    for lith in subset.lithologies
                    if lith.hole_id == left_id and lith.unit_order is not None
                }
            )
            right_units = sorted(
                {
                    lith.unit_order
                    for lith in subset.lithologies
                    if lith.hole_id == right_id and lith.unit_order is not None
                }
            )
            if not left_units or not right_units:
                continue
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                left_order = st.selectbox(
                    f"{left_id} unit",
                    options=left_units,
                    key=f"corr_left_{pair_index}",
                )
            with col2:
                right_order = st.selectbox(
                    f"{right_id} unit",
                    options=right_units,
                    key=f"corr_right_{pair_index}",
                )
            with col3:
                if st.button("Link", key=f"corr_link_{pair_index}"):
                    session_overrides.append(
                        CorrelationOverride(
                            left_hole_id=left_id,
                            right_hole_id=right_id,
                            left_unit_order=int(left_order),
                            right_unit_order=int(right_order),
                        )
                    )
                    st.session_state.session_correlation_overrides = session_overrides
                    st.rerun()
            pair_index += 1
        if session_overrides:
            st.json([item.model_dump() for item in session_overrides])
            if st.button("Clear manual overrides"):
                st.session_state.session_correlation_overrides = None
                st.rerun()


def render_nl_transect_input(hole_ids: list[str]) -> None:
    """Sidebar natural-language transect control."""
    nl_text = st.text_input(
        "Natural-language transect",
        value=st.session_state.get("nl_transect_text", ""),
        key="nl_transect_text",
        placeholder="Section B-B' through MW-01, MW-03, MW-07",
        help="Parses hole IDs from text. Geometry still comes from collar coordinates.",
    )
    if st.button("Apply NL transect", key="apply_nl_transect") and nl_text.strip():
        parsed = _build_assistant().parse_transect_request(nl_text, hole_ids)
        if parsed is None:
            st.warning("Could not find at least two known hole IDs in that request.")
        else:
            st.session_state.hole_sequence_multiselect = list(parsed.hole_ids)
            st.session_state.pending_transect_mode = "By hole sequence"
            if parsed.section_label:
                st.session_state.consulting_section_label = parsed.section_label
            st.success(
                f"Transect: {' → '.join(parsed.hole_ids)}"
                + (f" ({parsed.section_label})" if parsed.section_label else "")
            )
            st.rerun()


def render_correlation_assist(
    pair_summaries: Sequence,
    subset: ParseResult,
    active_ids: Sequence[str],
) -> None:
    """AI correlation link suggestions inside correlation health expander."""
    if st.button("Suggest correlation links", key="suggest_correlation_links"):
        st.session_state.ai_correlation_suggestions = (
            _build_assistant().suggest_correlation_overrides(
                pair_summaries,
                subset.lithologies,
                active_ids,
            )
        )
    corr_suggestions = st.session_state.get("ai_correlation_suggestions") or ()
    if not corr_suggestions:
        return
    st.markdown("**Suggested overrides** (review before apply)")
    for suggestion in corr_suggestions:
        st.write(
            f"{suggestion.left_hole_id} unit {suggestion.left_unit_order} ↔ "
            f"{suggestion.right_hole_id} unit {suggestion.right_unit_order} "
            f"({suggestion.confidence:.0%}) — {suggestion.rationale}"
        )
    if st.button("Accept correlation suggestions", key="accept_corr_suggestions"):
        session_overrides = list(_session_correlation_overrides())
        existing = {
            (
                item.left_hole_id,
                item.right_hole_id,
                item.left_unit_order,
                item.right_unit_order,
            )
            for item in session_overrides
        }
        for suggestion in corr_suggestions:
            override = suggestion.to_override()
            key = (
                override.left_hole_id,
                override.right_hole_id,
                override.left_unit_order,
                override.right_unit_order,
            )
            if key not in existing:
                session_overrides.append(override)
        st.session_state.session_correlation_overrides = session_overrides
        st.session_state.ai_correlation_suggestions = None
        st.rerun()


def render_section_qa(
    subset: ParseResult,
    active_ids: Sequence[str],
    preflight_warnings: Sequence[str],
) -> None:
    """Section Q&A expander grounded on active-transect facts."""
    with st.expander("Section Q&A", expanded=False):
        st.caption(
            "Answers use active-transect facts only (holes, water, NM, thicknesses, offsets)."
        )
        qa_question = st.text_input(
            "Ask about this section",
            key="section_qa_question",
            placeholder="Which wells are NM? What is clay thickness at MW-01?",
        )
        if st.button("Ask", key="section_qa_ask") and qa_question.strip():
            offset_map: dict[str, float] = {
                collar.hole_id: 0.0 for collar in subset.collars
            }
            for message in preflight_warnings:
                parts = message.split(" is ", 1)
                if len(parts) == 2 and parts[0] in offset_map:
                    try:
                        offset_map[parts[0]] = float(parts[1].split(" m ", 1)[0])
                    except ValueError:
                        pass
            st.session_state.section_qa_answer = _build_assistant().answer_section_question(
                qa_question,
                _section_facts(subset, active_ids, offsets_m=offset_map),
            )
        if st.session_state.get("section_qa_answer"):
            st.write(st.session_state.section_qa_answer)
