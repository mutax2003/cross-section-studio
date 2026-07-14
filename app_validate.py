"""Validate step: Data Health, fix plan, lithology mapping, sheet roles."""

from __future__ import annotations

from io import BytesIO
from typing import Any, Callable, Sequence

import pandas as pd
import streamlit as st

from ai_quality import save_lithology_alias
from ai_assistant import AIAssistant
from app_common import (
    _apply_auto_unit_order_fix,
    _build_assistant,
    _health_status_label,
    _mapping_rows,
    _metric_tone,
    _render_metric_card,
    llm_assist_status_caption,
    llm_suggestions_available,
)
from ingestion import NATIVE_PROFILE_ID


def _render_fix_step_actions(
    step: Any,
    *,
    parse_result: Any,
    import_report: Any,
    key_prefix: str,
) -> None:
    """Render one-click actions for a fix-plan step."""
    if step.action_id == "auto_unit_order":
        if st.button(
            "Fix layer order automatically",
            key=f"{key_prefix}_unit_order",
            type="primary" if step.blocks_generate else "secondary",
        ):
            _apply_auto_unit_order_fix(
                parse_result,
                import_report,
                success_message="Assigned layer order from depth — review QA below.",
            )
    elif step.action_id == "relative_elevation":
        if st.button(
            "Switch to relative depth mode",
            key=f"{key_prefix}_relative",
            type="primary" if step.blocks_generate else "secondary",
        ):
            st.session_state.elevation_mode = "relative"
            st.success("Elevation mode set to relative depth below collar.")
            st.rerun()
    elif step.action_id == "manual_unit_order":
        st.caption("Edit the Lithology sheet so each layer order (unit_order) is unique per hole.")
    elif step.action_id == "manual_intervals":
        st.caption("Correct from_depth/to_depth in the Lithology sheet.")


def _column_rename_checklist(suggestions: dict[str, Sequence[Any]]) -> str:
    """Markdown checklist for workbook header renames (advisory)."""
    lines = [
        "### Column rename checklist",
        "Update headers in the source workbook, then re-upload:",
        "",
    ]
    for sheet, items in suggestions.items():
        if not items:
            continue
        lines.append(f"**{sheet}**")
        for item in items:
            source = getattr(item, "source_column", None) or item.get("source_column", "")
            canonical = getattr(item, "canonical_column", None) or item.get("canonical_column", "")
            confidence = getattr(item, "confidence", None)
            if confidence is None and isinstance(item, dict):
                confidence = item.get("confidence", 0.0)
            conf_txt = f" ({float(confidence):.0%})" if confidence is not None else ""
            lines.append(f"- [ ] `{source}` → `{canonical}`{conf_txt}")
        lines.append("")
    return "\n".join(lines).strip()


def _sheet_role_checklist(roles: Sequence[Any]) -> str:
    """Markdown checklist for renaming sheets to platform roles."""
    lines = [
        "### Sheet rename checklist (advisory)",
        "Rename sheets in Excel to match platform roles, then re-upload:",
        "",
    ]
    for item in roles:
        role = item.role
        if role in {"unknown", ""}:
            continue
        lines.append(f"- [ ] `{item.sheet_name}` → role **{role}** ({item.confidence:.0%}) — {item.rationale}")
    if len(lines) <= 3:
        lines.append("- No high-confidence renames suggested.")
    return "\n".join(lines)


def _render_validate_details(
    *,
    parse_result: Any,
    import_report: Any,
    quality_report: Any,
    mapping_proposal: Any,
    unique_lithologies: list[str],
    scope_hole_ids: tuple[str, ...],
    scope_caption: str,
    missing_col: str,
    get_assistant: Callable[[], AIAssistant],
    show_blocking_fix_coach: bool,
) -> None:
    """Heavy Validate panels: optional sheets, fix coach, mapping, metrics, issues table."""
    if parse_result.water_levels:
        from ai_quality import summarize_water_levels

        water_summary = summarize_water_levels(
            parse_result.collars, parse_result.water_levels, scope_hole_ids
        )
        with st.expander("Groundwater", expanded=bool(water_summary.warnings)):
            st.caption(scope_caption)
            if water_summary.series:
                st.markdown(f"| Series | Label | Holes | {missing_col} |")
                st.markdown("| --- | --- | ---: | --- |")
                for series in water_summary.series:
                    missing = ", ".join(series.missing_hole_ids) if series.missing_hole_ids else "—"
                    st.markdown(
                        f"| `{series.series_id}` | {series.series_label or '—'} | "
                        f"{series.hole_count} | {missing} |"
                    )
            if water_summary.holes_without_any_water:
                st.warning(
                    "No water reading: "
                    + ", ".join(water_summary.holes_without_any_water)
                )
            for message in water_summary.warnings:
                st.warning(message)

    if parse_result.screen_intervals:
        from ai_quality import summarize_screen_intervals

        screen_summary = summarize_screen_intervals(
            parse_result.collars,
            parse_result.screen_intervals,
            scope_hole_ids,
        )
        if screen_summary.warnings:
            with st.expander("Screens", expanded=True):
                st.caption(scope_caption)
                for message in screen_summary.warnings:
                    st.warning(message)

    if parse_result.environmental_readings:
        from ai_quality import summarize_environmental_readings

        env_summary = summarize_environmental_readings(
            parse_result.collars,
            parse_result.environmental_readings,
            scope_hole_ids,
        )
        collar_ids = {collar.hole_id for collar in parse_result.collars}
        orphan_env_ids = sorted(
            {
                reading.hole_id
                for reading in parse_result.environmental_readings
                if reading.hole_id not in collar_ids
            }
        )
        with st.expander("Environmental / Lab", expanded=bool(env_summary.warnings) or bool(orphan_env_ids)):
            st.caption(scope_caption)
            if orphan_env_ids:
                st.warning(
                    "Environmental readings with no matching collar: "
                    + ", ".join(orphan_env_ids)
                )
            if env_summary.parameters:
                st.markdown(f"| Parameter | Holes | Depth range (m) | {missing_col} |")
                st.markdown("| --- | ---: | --- | --- |")
                for param in env_summary.parameters:
                    depth_range = (
                        f"{param.min_depth:.1f}–{param.max_depth:.1f}"
                        if param.min_depth is not None and param.max_depth is not None
                        else "—"
                    )
                    missing = ", ".join(param.missing_hole_ids) if param.missing_hole_ids else "—"
                    st.markdown(
                        f"| `{param.parameter}` | {param.hole_count} | {depth_range} | {missing} |"
                    )
            if env_summary.holes_without_any_readings:
                st.warning(
                    "No environmental reading: "
                    + ", ".join(env_summary.holes_without_any_readings)
                )
            for message in env_summary.warnings:
                st.warning(message)

    if quality_report.issues:
        st.markdown("**Fix coach**")
        cached_plan = st.session_state.get("qa_fix_plan")
        if cached_plan:
            fix_plan = cached_plan
        else:
            fix_plan = AIAssistant(None).suggest_fix_plan(quality_report.issues)
            st.session_state.qa_fix_plan = fix_plan
        blocking_steps = [step for step in fix_plan if step.blocks_generate]
        if show_blocking_fix_coach and quality_report.has_blocking_errors:
            st.error("Resolve these issues before generating a cross-section.")
            for index, step in enumerate(blocking_steps):
                st.markdown(f"**{step.summary}**")
                st.caption(step.action)
                _render_fix_step_actions(
                    step,
                    parse_result=parse_result,
                    import_report=import_report,
                    key_prefix=f"fix_blocking_{index}",
                )
        with st.expander("Full fix plan", expanded=False):
            if st.button("Build / refresh fix plan", key="build_fix_plan"):
                st.session_state.qa_fix_plan = get_assistant().suggest_fix_plan(
                    quality_report.issues
                )
                st.rerun()
            plan_steps = st.session_state.get("qa_fix_plan") or fix_plan
            blocking_codes = {step.issue_code for step in blocking_steps}
            for index, step in enumerate(plan_steps):
                badge = "blocks generate" if step.blocks_generate else "advisory"
                st.markdown(
                    f"**{index + 1}. [{step.issue_code}]** ({badge}) — {step.summary}"
                )
                st.caption(step.action)
                if step.blocks_generate and step.issue_code in blocking_codes:
                    continue
                _render_fix_step_actions(
                    step,
                    parse_result=parse_result,
                    import_report=import_report,
                    key_prefix=f"fix_plan_{index}",
                )
        st.divider()

    if import_report is not None:
        with st.expander("Import report", expanded=import_report.profile_id != NATIVE_PROFILE_ID):
            st.markdown(
                f"**Profile:** {import_report.profile_label} (`{import_report.profile_id}`)"
            )
            st.markdown(
                f"**Holes:** {import_report.hole_count} · "
                f"**Intervals:** {import_report.lithology_interval_count}"
            )
            if import_report.suggested_utm_crs:
                st.caption(f"Suggested CRS: `{import_report.suggested_utm_crs}`")
            if import_report.uses_placeholder_elevation:
                st.warning(
                    "All collars use placeholder elevation — set sidebar RL, use relative elevation mode, "
                    "or ingest Field Data RL before absolute interpreted sections."
                )
            if import_report.optional_sheets_detected:
                st.caption(
                    "Optional sheets: " + ", ".join(import_report.optional_sheets_detected)
                )
            if import_report.geology_sheet_counts:
                counts = ", ".join(
                    f"{key}={value}"
                    for key, value in import_report.geology_sheet_counts.items()
                    if value > 0
                )
                if counts:
                    st.caption(f"Loaded geology records: {counts}")
            if import_report.unit_order_auto_assigned:
                st.info("unit_order auto-assigned from depth sequence for holes with duplicate lithology codes.")
            elif import_report.lithology_has_unit_order_column:
                st.caption("Lithology sheet includes unit_order values.")
            else:
                st.caption("No unit_order column detected — auto-assign runs when duplicate codes exist.")
            if import_report.normalized_lithology_count:
                st.caption(
                    f"Normalized {import_report.normalized_lithology_count} "
                    "lithology code(s) via aliases."
                )
            if import_report.coordinate_offsets_applied:
                st.caption(
                    "Coordinate offsets applied: "
                    + ", ".join(
                        f"{hole} ({de:.1f}, {dn:.1f} m)"
                        for hole, (de, dn) in import_report.coordinate_offsets_applied.items()
                    )
                )
            for warning in import_report.warnings:
                st.warning(warning)

    if mapping_proposal is not None:
        with st.expander("Column mapping proposal", expanded=False):
            st.markdown("**Collars**")
            st.dataframe(
                pd.DataFrame(_mapping_rows(mapping_proposal.collar_column_mappings)),
                width="stretch",
                hide_index=True,
            )
            st.markdown("**Lithology**")
            st.dataframe(
                pd.DataFrame(_mapping_rows(mapping_proposal.lithology_column_mappings)),
                width="stretch",
                hide_index=True,
            )

            if mapping_proposal.low_confidence_mappings:
                if llm_suggestions_available():
                    if st.button("Get AI mapping suggestions (headers only)", key="ai_column_map"):
                        assistant = get_assistant()
                        collar_suggestions = assistant.suggest_column_mappings(
                            mapping_proposal, sheet="collars"
                        )
                        lithology_suggestions = assistant.suggest_column_mappings(
                            mapping_proposal, sheet="lithology"
                        )
                        st.session_state.ai_column_suggestions = {
                            "collars": collar_suggestions,
                            "lithology": lithology_suggestions,
                        }
                else:
                    st.caption(
                        "AI header mapping needs **Enable LLM suggestions** and an API key "
                        "(sidebar → AI Assist). Fuzzy mapping above already applied at ingest."
                    )

            stored = st.session_state.get("ai_column_suggestions")
            if stored:
                st.info("Review the rename checklist and update the spreadsheet if needed.")
                st.code(_column_rename_checklist(stored), language="markdown")

    if st.session_state.file_bytes and st.button(
        "Detect sheet roles (advisory)",
        key="detect_sheet_roles",
    ):
        workbook = pd.ExcelFile(BytesIO(st.session_state.file_bytes))
        headers_by_sheet = {
            name: [str(col) for col in pd.read_excel(workbook, sheet_name=name, nrows=0).columns]
            for name in workbook.sheet_names
        }
        st.session_state.ai_sheet_roles = get_assistant().suggest_sheet_roles(
            workbook.sheet_names,
            headers_by_sheet,
        )
    if st.session_state.get("ai_sheet_roles"):
        with st.expander("Sheet role detective (advisory)", expanded=False):
            st.caption(
                "Does not change import. Rename sheets in Excel to match platform roles, then re-upload."
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "sheet": item.sheet_name,
                            "role": item.role,
                            "confidence": f"{item.confidence:.0%}",
                            "rationale": item.rationale,
                        }
                        for item in st.session_state.ai_sheet_roles
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
            st.markdown(_sheet_role_checklist(st.session_state.ai_sheet_roles))

    health_tone = _metric_tone(quality_report.error_count, quality_report.warning_count)
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        _render_metric_card(
            import_report.hole_count if import_report else len(parse_result.collars),
            "Boreholes",
            health_tone,
        )
    with m2:
        _render_metric_card(
            import_report.lithology_interval_count if import_report else len(parse_result.lithologies),
            "Intervals",
            "ok",
        )
    with m3:
        _render_metric_card(len(unique_lithologies), "Lithology units", "ok")
    with m4:
        _render_metric_card(
            quality_report.error_count,
            "Errors",
            _metric_tone(quality_report.error_count, quality_report.warning_count, errors_only=True),
        )
    with m5:
        _render_metric_card(
            quality_report.warning_count,
            "Warnings",
            "warn" if quality_report.warning_count else "ok",
        )

    if quality_report.unmapped_lithologies:
        with st.expander("Lithology code mapping", expanded=False):
            st.caption("Map field codes to canonical USGS-style names used for correlation.")
            if st.button("Suggest aliases", key="suggest_lithology_aliases"):
                st.session_state.ai_lithology_suggestions = (
                    get_assistant().suggest_lithology_mappings(
                        quality_report.unmapped_lithologies
                    )
                )
            suggestions = st.session_state.get("ai_lithology_suggestions") or ()
            if suggestions:
                for suggestion in suggestions:
                    st.write(
                        f"`{suggestion.source_code}` → `{suggestion.canonical_code}` "
                        f"({suggestion.confidence:.0%}) — {suggestion.rationale}"
                    )
                if st.button("Accept all suggested aliases", key="accept_all_lith_aliases"):
                    for suggestion in suggestions:
                        save_lithology_alias(
                            suggestion.source_code,
                            suggestion.canonical_code,
                        )
                    st.session_state.ai_lithology_suggestions = None
                    st.session_state.lithology_aliases = None
                    st.session_state.parse_signature = None
                    st.session_state.parse_result = None
                    st.rerun()
            for source_code in quality_report.unmapped_lithologies:
                suggested = next(
                    (
                        item.canonical_code
                        for item in suggestions
                        if item.source_code == source_code
                    ),
                    source_code,
                )
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.text(source_code)
                with col2:
                    canonical = st.text_input(
                        "Canonical",
                        value=suggested,
                        key=f"map_{source_code}",
                        label_visibility="collapsed",
                    )
                with col3:
                    if st.button("Save", key=f"save_map_{source_code}"):
                        save_lithology_alias(source_code, canonical)
                        st.session_state.lithology_aliases = None
                        st.session_state.parse_signature = None
                        st.session_state.parse_result = None
                        st.rerun()

    if quality_report.normalized_lithology_count:
        st.caption(
            f"Normalized {quality_report.normalized_lithology_count} lithology code(s) via aliases."
        )

    if quality_report.issues:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "severity": issue.severity,
                        "code": issue.code,
                        "hole_id": issue.hole_id or "",
                        "message": issue.message,
                    }
                    for issue in quality_report.issues
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    if st.button("Generate QA narrative", key="generate_qa_narrative"):
        st.session_state.qa_narrative = get_assistant().explain_quality_issues(
            quality_report.issues
        )

    if st.session_state.qa_narrative:
        st.markdown("**QA Summary**")
        st.write(st.session_state.qa_narrative)

    for message in parse_result.errors:
        st.warning(message)


def render_validate_step() -> None:
    """Render Data Health panel when a parse result is in session."""
    if st.session_state.parse_result is None:
        return
    parse_result = st.session_state.parse_result
    import_report = st.session_state.import_report
    quality_report = st.session_state.quality_report
    mapping_proposal = st.session_state.mapping_proposal
    unique_lithologies = list(st.session_state.unique_lithology_codes)
    if quality_report is None:
        st.warning("No quality report available. Re-parse the workbook.")
        return

    st.subheader("Data Health")
    st.caption(llm_assist_status_caption())

    active_transect = st.session_state.get("transect_selection")
    scope_hole_ids: tuple[str, ...]
    if (
        isinstance(active_transect, (tuple, list))
        and len(active_transect) >= 1
        and isinstance(active_transect[0], (tuple, list))
    ):
        scope_hole_ids = tuple(str(hole_id) for hole_id in active_transect[0])
        scope_caption = (
            f"Coverage scoped to active transect (**{' → '.join(scope_hole_ids)}**)."
        )
        missing_col = "Missing on transect"
    else:
        scope_hole_ids = tuple(collar.hole_id for collar in parse_result.collars)
        scope_caption = (
            "Coverage is **site-wide** (all collars). Select a transect in Configure "
            "to see missing holes for the section only."
        )
        missing_col = "Missing (site-wide)"

    compact = not quality_report.has_blocking_errors
    status_label = _health_status_label(quality_report.error_count, quality_report.warning_count)
    if compact:
        st.markdown(
            '<div class="next-step-coach" tabindex="0">'
            "<strong>Next:</strong> pick holes in sidebar <strong>Transect selection</strong> "
            "→ Configure → <strong>Generate Cross-Section</strong> "
            "(SVG ready; Prepare for PNG/PDF)."
            "</div>",
            unsafe_allow_html=True,
        )
        if quality_report.warning_count or quality_report.info_count:
            st.markdown(
                f"**{status_label}** — "
                f"**{quality_report.error_count} errors**, "
                f"**{quality_report.warning_count} warnings**, "
                f"**{quality_report.info_count} info**"
            )
        else:
            st.markdown(f"**Data health OK** — no blocking issues detected.")
        details_loaded = st.session_state.get("validate_details_loaded")
        if quality_report.warning_count and not details_loaded:
            top_warnings = [
                issue
                for issue in quality_report.issues
                if issue.severity == "warning"
            ][:3]
            for issue in top_warnings:
                st.warning(f"**{issue.severity}** — {issue.message}")
            st.caption(
                "Enable **Allow generate with warnings** in Configure to proceed, "
                "or open Data Health details below to review all issues."
            )

    assistant: AIAssistant | None = None
    if quality_report.has_blocking_errors:
        assistant = _build_assistant()

    def get_assistant() -> AIAssistant:
        nonlocal assistant
        if assistant is None:
            assistant = _build_assistant()
        return assistant

    detail_kwargs = dict(
        parse_result=parse_result,
        import_report=import_report,
        quality_report=quality_report,
        mapping_proposal=mapping_proposal,
        unique_lithologies=unique_lithologies,
        scope_hole_ids=scope_hole_ids,
        scope_caption=scope_caption,
        missing_col=missing_col,
        get_assistant=get_assistant,
    )

    if compact:
        with st.expander("Data Health details", expanded=st.session_state.get("validate_details_expanded", False)):
            if st.session_state.get("validate_details_loaded"):
                _render_validate_details(**detail_kwargs, show_blocking_fix_coach=False)
            elif st.button("Load data health details", key="validate_load_details"):
                st.session_state.validate_details_loaded = True
                st.session_state.validate_details_expanded = True
                st.rerun()
    else:
        st.markdown(
            f"**{status_label}** — "
            f"**{quality_report.error_count} errors**, "
            f"**{quality_report.warning_count} warnings**, "
            f"**{quality_report.info_count} info**"
        )
        _render_validate_details(**detail_kwargs, show_blocking_fix_coach=True)
