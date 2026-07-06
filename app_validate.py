"""Validate step: Data Health, fix plan, lithology mapping, sheet roles."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st

from ai_quality import save_lithology_alias
from app_common import (
    _apply_auto_unit_order_fix,
    _build_assistant,
    _health_emoji,
    _mapping_rows,
    _metric_tone,
    _render_metric_card,
)
from ingestion import NATIVE_PROFILE_ID


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
    assistant = _build_assistant()

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
                if st.button("Get AI mapping suggestions (headers only)"):
                    collar_suggestions = assistant.suggest_column_mappings(
                        mapping_proposal, sheet="collars"
                    )
                    lithology_suggestions = assistant.suggest_column_mappings(
                        mapping_proposal, sheet="lithology"
                    )
                    if collar_suggestions or lithology_suggestions:
                        st.info("Review AI suggestions below and update the spreadsheet if needed.")
                        st.json(
                            {
                                "collars": [s.__dict__ for s in collar_suggestions],
                                "lithology": [s.__dict__ for s in lithology_suggestions],
                            }
                        )

    if st.session_state.file_bytes and st.button("Detect sheet roles", key="detect_sheet_roles"):
        workbook = pd.ExcelFile(BytesIO(st.session_state.file_bytes))
        headers_by_sheet = {
            name: [str(col) for col in pd.read_excel(workbook, sheet_name=name, nrows=0).columns]
            for name in workbook.sheet_names
        }
        st.session_state.ai_sheet_roles = assistant.suggest_sheet_roles(
            workbook.sheet_names,
            headers_by_sheet,
        )
    if st.session_state.get("ai_sheet_roles"):
        with st.expander("Sheet role suggestions", expanded=False):
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
            st.caption("Advisory only — rename sheets or map columns in the source workbook.")

    hole_ids = list(st.session_state.hole_ids)

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

    blocking_errors = [
        issue for issue in quality_report.issues if issue.severity == "error"
    ]
    if blocking_errors:
        with st.expander("Fix and proceed", expanded=True):
            if any(issue.code == "duplicate_lithology_no_unit_order" for issue in blocking_errors):
                st.markdown(
                    "Duplicate lithology codes need distinct **`unit_order`** values (1 = shallowest) "
                    "so units can correlate across holes."
                )
                if st.button("Auto-assign unit_order from depth", key="fix_unit_order"):
                    _apply_auto_unit_order_fix(
                        parse_result,
                        import_report,
                        success_message="Assigned unit_order from depth — review QA below.",
                    )
            if any(issue.code == "placeholder_elevation" for issue in blocking_errors):
                st.markdown(
                    "Set **Default collar elevation** in Workbook import, add RL to Field Data, "
                    "or choose **Relative elevation** below the transect before generating."
                )

    if quality_report.unmapped_lithologies:
        with st.expander("Lithology code mapping", expanded=False):
            st.caption("Map field codes to canonical USGS-style names used for correlation.")
            if st.button("Suggest aliases", key="suggest_lithology_aliases"):
                st.session_state.ai_lithology_suggestions = (
                    assistant.suggest_lithology_mappings(
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

    st.markdown(
        f"{_health_emoji(quality_report.error_count, quality_report.warning_count)} "
        f"**{quality_report.error_count} errors**, "
        f"**{quality_report.warning_count} warnings**, "
        f"**{quality_report.info_count} info**"
    )
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

    narrative_cols = st.columns(2)
    with narrative_cols[0]:
        if st.button("Generate QA narrative"):
            st.session_state.qa_narrative = assistant.explain_quality_issues(
                quality_report.issues
            )
    with narrative_cols[1]:
        if quality_report.issues and st.button("Build fix plan", key="build_fix_plan"):
            st.session_state.qa_fix_plan = assistant.suggest_fix_plan(
                quality_report.issues
            )

    if st.session_state.qa_narrative:
        st.markdown("**QA Summary**")
        st.write(st.session_state.qa_narrative)

    if st.session_state.get("qa_fix_plan"):
        with st.expander("Fix-path coach", expanded=True):
            for index, step in enumerate(st.session_state.qa_fix_plan):
                badge = "blocks generate" if step.blocks_generate else "advisory"
                st.markdown(
                    f"**{index + 1}. [{step.issue_code}]** ({badge}) — {step.summary}"
                )
                st.caption(step.action)
                if step.action_id == "auto_unit_order":
                    if st.button(
                        "Run auto-assign unit_order",
                        key=f"fix_plan_unit_order_{index}",
                    ):
                        _apply_auto_unit_order_fix(
                            parse_result,
                            import_report,
                            success_message="Assigned unit_order from depth.",
                        )
                elif step.action_id == "relative_elevation":
                    if st.button(
                        "Switch to relative elevation",
                        key=f"fix_plan_relative_{index}",
                    ):
                        st.session_state.elevation_mode = "relative"
                        st.success("Elevation mode set to relative.")
                        st.rerun()

    for message in parse_result.errors:
        st.warning(message)
