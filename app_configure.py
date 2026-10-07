"""Configure step: transect selection helpers, correlation assist, section Q&A."""

from __future__ import annotations

import json
import math
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
    render_full_lithology_legend,
    safe_lithology_index,
)
from app_services import cached_configure_preflight, cached_recommend_transects
from app_upload import queue_session_values
from batch_export import validate_batch_transect_lines
from models import Collar, CorrelationOverride, ParseResult, subset_parse_result
from render_profiles import ChemistryLabelStyle
from ui_output_presets import resolve_output_preset

OVERRIDE_WARNINGS_LABEL = "Generate even if data checks found warnings"
FAIL_ON_OVERLAPS_LABEL = "Stop if matched layers overlap"
MATCH_LAYERS_BY_HAND_LABEL = "Match layers by hand"
# Interpretation radio labels in the sidebar; hints must quote them exactly.
OBSERVED_LOGS_ONLY_LABEL = "Observed logs only"
CONTACT_LINES_ONLY_LABEL = "Contact lines only"
LOW_MATCH_HINT = (
    "Few layers match between some holes. To review the logs without joining layers, "
    f"choose '{OBSERVED_LOGS_ONLY_LABEL}' or '{CONTACT_LINES_ONLY_LABEL}' under "
    "Interpretation in the sidebar."
)
OVERLAP_GATE_MESSAGE = (
    f"Some matched layers overlap. Check '{MATCH_LAYERS_BY_HAND_LABEL}' or untick "
    f"'{FAIL_ON_OVERLAPS_LABEL}'."
)


def override_warnings_label(warning_count: int) -> str:
    """Checkbox label with the current warning count next to it."""
    if warning_count <= 0:
        return OVERRIDE_WARNINGS_LABEL
    noun = "warning" if warning_count == 1 else "warnings"
    return f"{OVERRIDE_WARNINGS_LABEL} ({warning_count} {noun})"


# Logging gaps draw grey ("Not logged") and are flagged in Validate; real logs
# often have no-recovery intervals, so they warn without locking Generate.
NON_GATING_WARNING_CODES = frozenset({"depth_gap"})


def gating_warning_count(quality_report) -> int:
    """Warnings that need the override tick before Generate (coverage gaps excluded)."""
    if quality_report is None:
        return 0
    return sum(
        1
        for issue in quality_report.issues
        if issue.severity == "warning" and issue.code not in NON_GATING_WARNING_CODES
    )


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
    selected_water_series_ids: tuple[str, ...] = ()
    chemistry_color_mode: str = "black"
    chemistry_threshold_green_max: float | None = None
    chemistry_threshold_yellow_max: float | None = None
    chemistry_label_style: ChemistryLabelStyle = "plain"
    placeholder_blocks_masl_water: bool = False

    @property
    def blocked_reason(self) -> str | None:
        """Why Generate is disabled, in the user's terms; None when it can run."""
        if self.can_generate:
            return None
        if self.blocking:
            return "fix the blocking data errors in Validate"
        if self.placeholder_blocks_interp:
            return (
                "collar elevations are missing — enter the site elevation under "
                "Data → Workbook format, or switch to depth below ground"
            )
        if self.placeholder_blocks_masl_water:
            return (
                "water levels are given as elevations but collar elevations are "
                "missing — add surveyed elevations or switch to depth below ground"
            )
        if self.fail_on_overlaps and self.has_overlap_warnings:
            return (
                "some matched layers overlap — check 'Match layers by hand' "
                "or untick 'Stop if matched layers overlap' in Configure"
            )
        if self.has_warnings and not self.override_warnings:
            return "tick 'Generate even if data checks found warnings' in Configure"
        if self.transect_selection is None:
            return "choose a section line in Configure"
        return "resolve the Configure / Validate issues"


def render_transect_sidebar(parse_result: ParseResult, hole_ids: list[str], transect_mode: str) -> tuple[list[str], str]:
    st.divider()
    _sidebar_heading("Stratigraphy legend")
    legend_codes = st.session_state.section_lithology_codes or st.session_state.unique_lithology_codes
    _render_lithology_legend(legend_codes)
    render_full_lithology_legend()

    _sidebar_heading("Section line")
    specs = tuple(getattr(parse_result, "section_specs", ()) or ())
    if specs:
        _render_workbook_section_picker(specs)
    render_nl_transect_input(hole_ids)
    selected_holes: list[str] = []
    coordinate_text = ""
    if transect_mode == "Recommended":
        candidates = load_transect_candidates(parse_result)
        if candidates:
            described = describe_transect_candidates(candidates, parse_result.collars)
            labels = [label for label, _candidate in described]
            choice = st.selectbox(
                "Suggested section lines",
                options=labels,
                index=0,
                help=(
                    "Best first. Offset is how far a hole sits off a straight line from "
                    "the first hole to the last; small offsets give a truer section. "
                    "Lines with only 2 holes show little between them and are listed last."
                ),
            )
            selected_holes = list(described[labels.index(choice)][1].hole_ids)
        else:
            st.info("Not enough holes to suggest a section line.")
    elif transect_mode == "By hole sequence":
        if "hole_sequence_multiselect" not in st.session_state:
            # A workbook with a Sections tab starts on its first section.
            st.session_state.hole_sequence_multiselect = (
                [h for h in specs[0].hole_ids if h in hole_ids]
                if specs
                else default_hole_sequence(hole_ids)
            )
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
            "Section line points (easting northing, one point per line)",
            value=default_coords,
            height=160,
        )
    return selected_holes, coordinate_text


# Candidates fetched for both the Recommended picker and "Fill from
# recommended" in the batch panel; more than are shown so 2-hole lines can be
# pushed below better ones.
RECOMMENDED_CANDIDATES_FETCH = 8
MIN_USEFUL_SECTION_HOLES = 3


def load_transect_candidates(parse_result: ParseResult) -> list:
    """Suggested section lines, computed on first use and kept for the session."""
    if st.session_state.get("transect_candidates") is None:
        st.session_state.transect_candidates = cached_recommend_transects(
            parse_result.collars,
            parse_result.lithologies,
            RECOMMENDED_CANDIDATES_FETCH,
        )
    return list(st.session_state.transect_candidates or [])


def max_offset_from_straight_m(
    collars: Sequence[Collar],
    hole_ids: Sequence[str],
) -> float:
    """Largest distance of any hole from the straight line first hole → last hole."""
    lookup = {collar.hole_id: collar for collar in collars}
    points = [(lookup[h].easting, lookup[h].northing) for h in hole_ids if h in lookup]
    if len(points) < 3:
        return 0.0
    (x0, y0), (x1, y1) = points[0], points[-1]
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length == 0.0:
        return max(math.hypot(x - x0, y - y0) for x, y in points)
    return max(abs(dy * (x - x0) - dx * (y - y0)) / length for x, y in points)


def _format_metres(value: float) -> str:
    return f"{value:.0f} m" if value >= 10 else f"{value:.1f} m".replace(".0 m", " m")


def describe_transect_candidates(
    candidates: Sequence,
    collars: Sequence[Collar],
) -> list[tuple[str, object]]:
    """Plain-language label per suggested line, lines with < 3 holes ranked last.

    e.g. ``"MW-01 → MW-05 · 4 holes · 120 m long · max offset 6 m"``.
    """
    ranked = sorted(
        candidates,
        key=lambda candidate: len(candidate.hole_ids) < MIN_USEFUL_SECTION_HOLES,
    )
    described: list[tuple[str, object]] = []
    seen: set[str] = set()
    for candidate in ranked:
        hole_ids = tuple(candidate.hole_ids)
        parts = [f"{hole_ids[0]} → {hole_ids[-1]}", f"{len(hole_ids)} holes"]
        length = getattr(candidate, "length_m", None)
        if length:
            parts.append(f"{_format_metres(float(length))} long")
        if len(hole_ids) < MIN_USEFUL_SECTION_HOLES:
            parts.append("only 2 holes")
        else:
            parts.append(f"max offset {_format_metres(max_offset_from_straight_m(collars, hole_ids))}")
        label = " · ".join(parts)
        if label in seen:  # same ends and size: name the middle holes
            label = f"{label} (via {', '.join(hole_ids[1:-1])})"
        seen.add(label)
        described.append((label, candidate))
    return described


def _render_workbook_section_picker(specs) -> None:
    """Drive the on-screen preview from the workbook's Sections tab.

    One drop-down per named section line: choosing one sets the hole order,
    switches to "By hole sequence" and labels the sheet, so a moved line only
    needs its row in the Sections tab changed. "Custom" leaves the manual
    controls as they are.
    """
    # Options are row indices (None = Custom) so a row labelled "Custom" or a
    # duplicate label can still be chosen; duplicates show their row number.
    seen: dict[str, int] = {}
    display: dict[int | None, str] = {None: "Custom"}
    for index, spec in enumerate(specs):
        seen[spec.label] = seen.get(spec.label, 0) + 1
        display[index] = spec.label if seen[spec.label] == 1 else f"{spec.label} (row {index + 2})"
    choice = st.selectbox(
        "Section (from workbook Sections tab)",
        options=list(display),
        index=0,
        format_func=lambda key: display[key],
        key="workbook_section_choice",
        help=(
            "Each row of the Sections tab is one section line. Pick one to preview it; "
            "use 'Several section lines (batch ZIP)' under Configure to export every section."
        ),
    )
    if choice is None:
        st.session_state.pop("_workbook_section_applied", None)
        return
    spec = specs[choice]
    known = set(st.session_state.get("hole_ids") or [])
    holes = [hole for hole in spec.hole_ids if hole in known]
    if len(holes) < 2:
        st.warning(f"Section {spec.label!r} needs at least two holes that exist in Collars.")
        return
    already = st.session_state.get("_workbook_section_applied") == choice
    if already and list(st.session_state.get("hole_sequence_multiselect") or []) == holes:
        return  # applied and untouched
    if already:
        # The user edited the hole list after picking this section: say so
        # rather than pretending the caption still describes the row.
        st.caption(f"Hole order edited by hand; re-select {display[choice]!r} to restore the row.")
        return
    st.session_state.hole_sequence_multiselect = holes
    st.session_state.pending_transect_mode = "By hole sequence"
    queue_consulting_section_label(spec.label)
    st.session_state["_workbook_section_applied"] = choice
    st.rerun()


def queue_consulting_section_label(label: str | None) -> None:
    """Set the sheet label on the NEXT run, before its text_input exists.

    Assigning the widget key directly raises once the sidebar has drawn the
    "Section label" box (consulting layouts), so it goes through the pending
    project seed that the sidebar applies first. None restores the default.
    """
    if label is None:
        st.session_state["_reset_consulting_section_label"] = True
        return
    # Both the consulting sheet label and the figure/file title: exports name
    # files and metadata from section_title, which otherwise kept the
    # previous section's name.
    queue_session_values(consulting_section_label=label, section_title=label)


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
    prefer_chemistry: bool = False,
    show_parameter_labels_default: bool | None = None,
    parameter_interpolate_segments_default: bool | None = None,
    elevation_mode_default: str | None = None,
) -> ConfigureState:
    st.subheader("Configure", anchor=False)
    st.caption("Check the section line, elevations and data-check settings before generating.")
    all_hole_ids = [collar.hole_id for collar in parse_result.collars]
    if transect_mode == "By hole sequence":
        if "hole_sequence_multiselect" not in st.session_state:
            st.session_state.hole_sequence_multiselect = default_hole_sequence(all_hole_ids)
        selected_holes = list(st.session_state.get("hole_sequence_multiselect") or selected_holes)
    preflight_selection = _active_transect_selection(
        parse_result,
        transect_mode,
        selected_holes,
        coordinate_text,
        offset_warning_m,
    )
    _render_plan_minimap(parse_result, preflight_selection)
    if transect_mode == "By hole sequence":
        _render_hole_sequence_order(all_hole_ids)
    blocking = quality_report is not None and quality_report.has_blocking_errors
    output_preset_key = st.session_state.get("output_preset")
    if (
        elevation_mode_default is not None
        and st.session_state.get("_synced_elevation_preset") != output_preset_key
    ):
        st.session_state.elevation_mode = elevation_mode_default
        st.session_state._synced_elevation_preset = output_preset_key
    elevation_mode = st.session_state.get("elevation_mode", elevation_mode_default or "absolute")
    if elevation_mode_default is not None:
        st.caption(
            "Elevation mode locked by output preset: "
            f"**{'relative (mbgs)' if elevation_mode_default == 'relative' else 'absolute (MASL)'}**."
        )
        elevation_mode = elevation_mode_default
        st.session_state.elevation_mode = elevation_mode_default
    elif import_report and import_report.uses_placeholder_elevation:
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
    placeholder_blocks_masl_water = False
    if (
        import_report is not None
        and import_report.uses_placeholder_elevation
        and elevation_mode == "absolute"
        and parse_result.water_levels
    ):
        placeholder_blocks_masl_water = any(
            level.elevation_masl is not None for level in parse_result.water_levels
        )
    if placeholder_blocks_interp:
        st.error(
            "Collar elevations are missing. Enter the site elevation under "
            "Data → Workbook format, or switch to depth below ground."
        )
    if placeholder_blocks_masl_water:
        st.error(
            "Water levels are given as elevations, but collar elevations are missing. "
            "Add surveyed elevations or switch to depth below ground."
        )
    warning_count = gating_warning_count(quality_report)
    has_warnings = warning_count > 0
    warnings_default = not is_consulting_layout
    if "override_warnings_checkbox" not in st.session_state:
        st.session_state["override_warnings_checkbox"] = warnings_default
    # The count sits in the label; the warnings themselves are listed in
    # Validate on the same screen, so they are not repeated here.
    override_warnings = st.checkbox(
        override_warnings_label(warning_count),
        key="override_warnings_checkbox",
        help=(
            "Data checks are listed under Validate. The consulting report style starts "
            "with this off so warnings are reviewed before a client figure is made."
        ),
    )
    if "fail_on_overlaps_checkbox" not in st.session_state:
        st.session_state["fail_on_overlaps_checkbox"] = is_consulting_layout
    fail_on_overlaps = st.checkbox(
        FAIL_ON_OVERLAPS_LABEL,
        key="fail_on_overlaps_checkbox",
        help=(
            "When ticked, Generate stops if the shaded layers drawn between two holes "
            "cross over each other."
        ),
    )

    environmental_parameters: tuple[str, ...] = ()
    show_parameter_labels = True
    parameter_interpolate_segments = True
    selected_water_series_ids: tuple[str, ...] = ()
    chemistry_color_mode: str = "black"
    chemistry_threshold_green_max: float | None = None
    chemistry_threshold_yellow_max: float | None = None
    chemistry_label_style = "plain"
    subset_ready = False
    has_overlap_warnings = False

    if preflight_selection is not None:
        active_ids, active_points = preflight_selection
        section_label = st.session_state.get("consulting_section_label") or "A-A'"
        st.caption(
            f"Section line {section_label}: **{' → '.join(active_ids)}** ({len(active_ids)} holes)"
        )
        subset_preflight = None
        try:
            subset_preflight = subset_parse_result(
                parse_result,
                active_ids,
                lithology_index=safe_lithology_index(parse_result),
            )
        except Exception as exc:
            st.error(f"Could not read the holes on this section line: {exc}")
            subset_preflight = None

        if subset_preflight is None:
            st.caption("Fix the section line or the workbook data before generating.")
        else:
            subset_ready = True
            available_params = sorted(
                {reading.parameter for reading in subset_preflight.environmental_readings}
            )
            if available_params:
                st.markdown("**Lab / environmental parameters**")
                options_sig = tuple(available_params)
                preset_chem_sig = (prefer_chemistry, options_sig)
                if st.session_state.get("_env_params_options_sig") != options_sig:
                    previous = list(st.session_state.get("environmental_parameters_multiselect") or [])
                    kept = [p for p in previous if p in available_params]
                    if prefer_chemistry:
                        chloride_like = [
                            p
                            for p in available_params
                            if "chlor" in p.lower() or p.lower() in {"cl", "cl-"}
                        ]
                        st.session_state.environmental_parameters_multiselect = (
                            kept or chloride_like or list(available_params)
                        )
                    else:
                        st.session_state.environmental_parameters_multiselect = (
                            kept or list(available_params)
                        )
                    st.session_state._env_params_options_sig = options_sig
                elif (
                    prefer_chemistry
                    and st.session_state.get("_env_params_preset_sig") != preset_chem_sig
                ):
                    chloride_like = [
                        p
                        for p in available_params
                        if "chlor" in p.lower() or p.lower() in {"cl", "cl-"}
                    ]
                    if chloride_like:
                        st.session_state.environmental_parameters_multiselect = chloride_like
                    st.session_state._env_params_preset_sig = preset_chem_sig
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
                label_default = (
                    True
                    if show_parameter_labels_default is None
                    else show_parameter_labels_default
                )
                segments_default = (
                    True
                    if parameter_interpolate_segments_default is None
                    else parameter_interpolate_segments_default
                )
                # Seed through session state only (no value= on the widgets):
                # passing both makes Streamlit warn on every rerun.
                if "show_parameter_labels_toggle" not in st.session_state:
                    st.session_state.show_parameter_labels_toggle = label_default
                if "parameter_interpolate_segments_toggle" not in st.session_state:
                    st.session_state.parameter_interpolate_segments_toggle = segments_default
                if prefer_chemistry:
                    if st.session_state.get("_chem_toggle_preset") != prefer_chemistry:
                        st.session_state.show_parameter_labels_toggle = label_default
                        st.session_state.parameter_interpolate_segments_toggle = (
                            segments_default
                        )
                        st.session_state._chem_toggle_preset = prefer_chemistry
                show_parameter_labels = st.toggle(
                    "Show parameter value labels",
                    key="show_parameter_labels_toggle",
                    disabled=prefer_chemistry and show_parameter_labels_default is not None,
                )
                parameter_interpolate_segments = st.toggle(
                    "Interpolate parameter between adjacent holes",
                    key="parameter_interpolate_segments_toggle",
                    disabled=(
                        prefer_chemistry and parameter_interpolate_segments_default is not None
                    ),
                )
                if prefer_chemistry:
                    if show_parameter_labels_default is not None:
                        show_parameter_labels = show_parameter_labels_default
                    if parameter_interpolate_segments_default is not None:
                        parameter_interpolate_segments = (
                            parameter_interpolate_segments_default
                        )
                st.markdown("**Parameter label colour**")
                mode_labels = {
                    "black": "All black",
                    "red": "All red (client P2 style)",
                    "threshold": "Green / orange / red thresholds",
                }
                # The output style sets the starting colour (Chemistry columns
                # prints red like the client's P2 figures); switching style
                # resets it, otherwise the user's choice is kept.
                output_preset = str(st.session_state.get("output_preset", "section_sheet"))
                style_mode = resolve_output_preset(output_preset).chemistry_color_mode
                if (
                    "chemistry_color_mode_radio" not in st.session_state
                    or st.session_state.get("_chem_color_mode_preset") != output_preset
                ):
                    st.session_state["chemistry_color_mode_radio"] = mode_labels.get(
                        style_mode, mode_labels["black"]
                    )
                    st.session_state["_chem_color_mode_preset"] = output_preset
                color_mode_choice = st.radio(
                    "Chemistry value labels",
                    options=list(mode_labels.values()),
                    key="chemistry_color_mode_radio",
                    help=(
                        "Black or red labels for data presentation. Threshold mode colours "
                        "each value green, orange, or red using site-specific limits set below. "
                        "A colour given in the workbook's label_color column always wins."
                    ),
                )
                chemistry_color_mode = next(
                    (mode for mode, label in mode_labels.items() if label == color_mode_choice),
                    "black",
                )
                chemistry_threshold_green_max = None
                chemistry_threshold_yellow_max = None
                if chemistry_color_mode == "threshold":
                    threshold_cols = st.columns(2)
                    with threshold_cols[0]:
                        chemistry_threshold_green_max = float(
                            st.number_input(
                                "Green ≤",
                                min_value=0.0,
                                value=100.0,
                                step=1.0,
                                key="chemistry_threshold_green_max",
                            )
                        )
                    with threshold_cols[1]:
                        chemistry_threshold_yellow_max = float(
                            st.number_input(
                                "Orange ≤",
                                min_value=0.0,
                                value=250.0,
                                step=1.0,
                                key="chemistry_threshold_yellow_max",
                            )
                        )
                    if chemistry_threshold_yellow_max < chemistry_threshold_green_max:
                        st.warning("Orange threshold should be ≥ green threshold.")
                if any(r.label_color for r in parse_result.environmental_readings):
                    st.caption(
                        "Rows with a **label_color** (green / red / black / orange) in the "
                        "Environmental sheet keep that colour; the setting above applies to the rest."
                    )
                style_labels = {
                    "plain": "Plain coloured text",
                    "box": "White box behind the label",
                    "dot": "Coloured dot, black text",
                    "stroke": "Outlined (white halo) text",
                }
                style_choice = st.selectbox(
                    "Label readability over hatched fills",
                    options=list(style_labels),
                    format_func=lambda key: style_labels[key],
                    index=0,
                    key="chemistry_label_style_select",
                    help="Keeps coloured values legible over hatched lithology fills.",
                )
                chemistry_label_style = style_choice
            elif parse_result.environmental_readings:
                st.caption(
                    "Lab results exist, but none are for holes on this section line."
                )

            water_levels = subset_preflight.water_levels
            if water_levels:
                from models import MAX_WATER_SERIES

                series_options: list[str] = []
                series_labels: dict[str, str] = {}
                for level in water_levels:
                    sid = level.series_id or "default"
                    if sid not in series_labels:
                        series_options.append(sid)
                        series_labels[sid] = level.series_label or sid
                if series_options:
                    st.markdown("**Groundwater series**")
                    options_sig = tuple(series_options)
                    if st.session_state.get("_water_series_options_sig") != options_sig:
                        previous = list(
                            st.session_state.get("water_series_multiselect") or []
                        )
                        kept = [s for s in previous if s in series_options]
                        st.session_state.water_series_multiselect = (
                            kept or series_options[:MAX_WATER_SERIES]
                        )
                        st.session_state._water_series_options_sig = options_sig
                    selected_water_series_ids = tuple(
                        st.multiselect(
                            "Water-level series to plot (max 4)",
                            options=series_options,
                            format_func=lambda sid: (
                                f"{series_labels.get(sid, sid)} ({sid})"
                                if series_labels.get(sid, sid) != sid
                                else sid
                            ),
                            help=(
                                "Each series uses a blue inverted triangle. "
                                "Use Water.connect_group so shallow and deep nests "
                                "are not joined by the same dashed line."
                            ),
                            key="water_series_multiselect",
                            max_selections=MAX_WATER_SERIES,
                        )
                    )
                    if not selected_water_series_ids:
                        st.caption(
                            "No water series selected — all groundwater series will be plotted. "
                            "Select specific series to limit the markers."
                        )
            else:
                st.caption(
                    "Add an **Environmental** sheet (`hole_id`, `parameter`, `value`, `depth` or "
                    "`from_depth`/`to_depth`) to plot lab data by depth. See Help → Workbook and data entry."
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
                # While Validate shows blocking errors, keep the heavy review
                # panels folded: the data has to be fixed first.
                with st.expander(
                    "Layer matching between holes" + (" (fix data errors first)" if blocking else ""),
                    expanded=not blocking,
                ):
                    table_rows = [
                        {
                            "Left": summary.left_hole_id,
                            "Right": summary.right_hole_id,
                            "Matched": summary.matched_count,
                            "Left only": ", ".join(summary.left_only_codes) or "—",
                            "Right only": ", ".join(summary.right_only_codes) or "—",
                            "Pinch-outs": summary.pinch_out_candidates,
                            "Match rate": f"{summary.match_rate:.0%}",
                        }
                        for summary in pair_summaries
                    ]
                    st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)
                    low_match = [s for s in pair_summaries if s.match_rate < 0.5]
                    if low_match:
                        st.info(LOW_MATCH_HINT)
                    _render_quick_correlation_links(pair_summaries, subset_preflight)
                    render_correlation_assist(pair_summaries, subset_preflight, active_ids)

            if len(active_ids) >= 2:
                render_manual_correlation_overrides(active_ids, subset_preflight)

            if _session_correlation_overrides():
                st.info(
                    "Layers matched by hand are in use and stay in place when you change "
                    "the section line."
                )

            render_section_qa(subset_preflight, active_ids, preflight_warnings)
    else:
        st.info(
            "Choose at least two holes under **Section line** in the sidebar "
            "(pick holes in order, use a suggested line, or enter points)."
        )

    can_generate = (
        parse_result is not None
        and preflight_selection is not None
        and subset_ready
        and not blocking
        and not placeholder_blocks_interp
        and not placeholder_blocks_masl_water
        and (override_warnings or not has_warnings)
        and (not fail_on_overlaps or not has_overlap_warnings)
    )
    if blocking:
        st.error("Fix the data errors listed under Validate before generating a section.")
    elif fail_on_overlaps and has_overlap_warnings:
        st.error(OVERLAP_GATE_MESSAGE)

    _render_batch_section_lines(
        preflight_selection,
        known_hole_ids=all_hole_ids,
        collapsed_for_errors=blocking,
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
        placeholder_blocks_masl_water=placeholder_blocks_masl_water,
        elevation_mode=str(elevation_mode),
        fail_on_overlaps=fail_on_overlaps,
        has_overlap_warnings=has_overlap_warnings,
        environmental_parameters=environmental_parameters,
        show_parameter_labels=show_parameter_labels,
        parameter_interpolate_segments=parameter_interpolate_segments,
        selected_water_series_ids=selected_water_series_ids,
        chemistry_color_mode=chemistry_color_mode,
        chemistry_threshold_green_max=chemistry_threshold_green_max,
        chemistry_threshold_yellow_max=chemistry_threshold_yellow_max,
        chemistry_label_style=chemistry_label_style,
    )


BATCH_EXPANDER_KEY = "batch_section_lines_expander"
_SECTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _batch_workbook_specs() -> tuple:
    report = st.session_state.get("import_report")
    specs = tuple(getattr(report, "section_specs", ()) or ()) if report is not None else ()
    if not specs:
        parse_result = st.session_state.get("parse_result")
        if parse_result is not None:
            specs = tuple(getattr(parse_result, "section_specs", ()) or ())
    return specs


def recommended_batch_text(candidates: Sequence, limit: int = 8) -> str:
    """``A-A' | …`` lines for the suggested section lines (3+ holes first)."""
    ranked = sorted(
        candidates,
        key=lambda candidate: len(candidate.hole_ids) < MIN_USEFUL_SECTION_HOLES,
    )
    lines: list[str] = []
    for idx, candidate in enumerate(ranked[:limit]):
        letter = _SECTION_LETTERS[idx] if idx < len(_SECTION_LETTERS) else str(idx + 1)
        lines.append(f"{letter}-{letter}' | {', '.join(candidate.hole_ids)}")
    return "\n".join(lines)


def _render_batch_section_lines(
    preflight_selection,
    *,
    known_hole_ids: Sequence[str],
    collapsed_for_errors: bool,
) -> None:
    """Editor for several section lines, checked line by line as typed."""
    label = "Several section lines (batch ZIP)"
    if collapsed_for_errors:
        label += " (fix data errors first)"
    # Keyed so the panel keeps its open/closed state across the reruns its
    # own buttons trigger.
    with st.expander(label, expanded=False, key=BATCH_EXPANDER_KEY, on_change="rerun"):
        st.caption(
            "One section line per row: name, then '|', then hole IDs in order "
            "(e.g. A-A' | MW-01, MW-02, MW-03). Prepare batch ZIP on Generate redraws "
            "each line as its own section."
        )
        workbook_specs = _batch_workbook_specs()
        if (
            workbook_specs
            and not str(st.session_state.get("batch_transect_specs", "")).strip()
            and not st.session_state.get("_batch_specs_seeded_from_sections")
        ):
            from parse_ops import format_section_specs_as_batch_text

            st.session_state["batch_transect_specs"] = format_section_specs_as_batch_text(
                workbook_specs
            )
            st.session_state["_batch_specs_seeded_from_sections"] = True
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("Add current section line", key="batch_add_current"):
                if preflight_selection is None:
                    st.warning("Choose a section line first.")
                else:
                    hole_ids, _pts = preflight_selection
                    line_label = (
                        str(st.session_state.get("consulting_section_label") or "").strip()
                        or f"{hole_ids[0]}→{hole_ids[-1]}"
                    )
                    line = f"{line_label} | {', '.join(hole_ids)}"
                    existing = str(st.session_state.get("batch_transect_specs", "")).rstrip()
                    st.session_state["batch_transect_specs"] = (
                        f"{existing}\n{line}".strip() if existing else line
                    )
                    st.rerun()
        with col_b:
            if st.button("Fill from suggested lines", key="batch_fill_recommended"):
                parse_result = st.session_state.get("parse_result")
                candidates = load_transect_candidates(parse_result) if parse_result is not None else []
                if not candidates:
                    st.warning("Not enough holes to suggest section lines.")
                else:
                    st.session_state["batch_transect_specs"] = recommended_batch_text(candidates)
                    st.rerun()
        with col_c:
            if st.button("Load from workbook Sections", key="batch_load_sections"):
                if not workbook_specs:
                    st.warning("The workbook has no Sections tab rows.")
                else:
                    from parse_ops import format_section_specs_as_batch_text

                    st.session_state["batch_transect_specs"] = format_section_specs_as_batch_text(
                        workbook_specs
                    )
                    st.session_state["_batch_specs_seeded_from_sections"] = True
                    st.rerun()
        text = st.text_area(
            "Section lines",
            key="batch_transect_specs",
            placeholder="A-A' | BH-01, BH-02, BH-03\nB-B' | BH-08, BH-09, BH-10",
            height=120,
            help=(
                "Each row becomes its own section in the batch ZIP. An empty box is filled "
                "once from the workbook Sections tab; use Load from workbook Sections to refresh."
            ),
        )
        _render_batch_line_status(text, known_hole_ids)


def _render_batch_line_status(text: str, known_hole_ids: Sequence[str]) -> None:
    statuses = validate_batch_transect_lines(str(text or ""), known_hole_ids)
    if not statuses:
        return
    ready = [status for status in statuses if status.ok]
    problems = [status for status in statuses if not status.ok]
    summary = f"{len(ready)} of {len(statuses)} section lines ready."
    if not problems:
        st.success(summary)
        return
    st.warning(
        summary
        + " Lines with problems are skipped when you prepare the batch ZIP:\n\n"
        + "\n".join(f"- {status.message}" for status in problems)
    )


IN_SECTION = "On section line"
NOT_IN_SECTION = "Not on section line"
_IN_SECTION_COLOUR = "#2E6B4F"  # app primary green
_NOT_IN_SECTION_COLOUR = "#9AA5B1"
_LINE_COLOUR = "#C0392B"
PLAN_VIEW_WIDTH_PX = 640
_PLAN_VIEW_HEIGHT_RANGE_PX = (240, 480)


def _render_plan_minimap(
    parse_result: ParseResult,
    selection: tuple[tuple[str, ...], tuple[tuple[float, float], ...]] | None,
) -> None:
    """Plan view: every collar labelled, the section line drawn in hole order."""
    if not parse_result.collars:
        return
    active_ids = list(selection[0]) if selection else []
    line_points = list(selection[1]) if selection else []
    order = {hole_id: index + 1 for index, hole_id in enumerate(active_ids)}
    chart_df = pd.DataFrame(
        [
            {
                "Easting": collar.easting,
                "Northing": collar.northing,
                "hole_id": collar.hole_id,
                "section": IN_SECTION if collar.hole_id in order else NOT_IN_SECTION,
                "order": order.get(collar.hole_id),
            }
            for collar in parse_result.collars
        ]
    )
    st.markdown("**Plan view (collar locations)**")
    st.altair_chart(
        _plan_view_chart(chart_df, "section" if active_ids else None, line_points),
        width="content",
    )
    if active_ids:
        st.caption(
            f"Red line: the section line, {active_ids[0]} → {active_ids[-1]}. "
            "Green holes are on it; grey holes are not. Both axes use the same scale."
        )
    else:
        st.caption("Choose at least two holes to draw the section line. Both axes use the same scale.")


def equal_scale_domains(
    eastings: Sequence[float],
    northings: Sequence[float],
    *,
    width_px: int = PLAN_VIEW_WIDTH_PX,
    height_range_px: tuple[int, int] = _PLAN_VIEW_HEIGHT_RANGE_PX,
) -> tuple[list[float], list[float], int]:
    """Axis domains and chart height so 1 m east = 1 m north on screen.

    Returns ``(x_domain, y_domain, height_px)``: the height follows the data's
    shape within ``height_range_px``, then the shorter axis is widened so both
    axes have the same metres per pixel.
    """
    x_low, x_high = float(min(eastings)), float(max(eastings))
    y_low, y_high = float(min(northings)), float(max(northings))
    x_pad = max((x_high - x_low) * 0.08, 5.0)
    y_pad = max((y_high - y_low) * 0.08, 5.0)
    x_low, x_high = x_low - x_pad, x_high + x_pad
    y_low, y_high = y_low - y_pad, y_high + y_pad
    x_span, y_span = x_high - x_low, y_high - y_low
    min_h, max_h = height_range_px
    height_px = int(round(min(max(width_px * y_span / x_span, min_h), max_h)))
    metres_per_px = max(x_span / width_px, y_span / height_px)
    x_mid, y_mid = (x_low + x_high) / 2, (y_low + y_high) / 2
    x_half = metres_per_px * width_px / 2
    y_half = metres_per_px * height_px / 2
    return [x_mid - x_half, x_mid + x_half], [y_mid - y_half, y_mid + y_half], height_px


def _plan_view_chart(
    chart_df: pd.DataFrame,
    color_col: str | None,
    line_points: Sequence[tuple[float, float]] = (),
):
    """Collars on equal-scale axes fitted to the data, labelled by hole ID.

    st.scatter_chart anchors both axes at zero, so UTM collars (500000 E,
    4500000 N) collapse into one dot in a corner; independent axis fits would
    distort the section line's angle and spacing.
    """
    import altair as alt

    all_e = list(chart_df["Easting"]) + [point[0] for point in line_points]
    all_n = list(chart_df["Northing"]) + [point[1] for point in line_points]
    x_domain, y_domain, height_px = equal_scale_domains(all_e, all_n)
    x = alt.X(
        "Easting:Q",
        scale=alt.Scale(domain=x_domain, nice=False, zero=False),
        axis=alt.Axis(format="d", tickCount=6),
    )
    y = alt.Y(
        "Northing:Q",
        scale=alt.Scale(domain=y_domain, nice=False, zero=False),
        axis=alt.Axis(format="d", tickCount=6),
    )
    tooltip = ["hole_id", "Easting", "Northing"]
    points = alt.Chart(chart_df).mark_circle(size=90, opacity=1).encode(x=x, y=y, tooltip=tooltip)
    if color_col:
        points = points.encode(
            color=alt.Color(
                f"{color_col}:N",
                title=None,
                scale=alt.Scale(
                    domain=[IN_SECTION, NOT_IN_SECTION],
                    range=[_IN_SECTION_COLOUR, _NOT_IN_SECTION_COLOUR],
                ),
                legend=alt.Legend(orient="top"),
            )
        )
    labels = (
        alt.Chart(chart_df)
        .mark_text(align="left", baseline="bottom", dx=6, dy=-4, fontSize=11, color="#33414E")
        .encode(x=x, y=y, text="hole_id:N")
    )
    layers = []
    if len(line_points) >= 2:
        line_df = pd.DataFrame(
            [
                {"Easting": easting, "Northing": northing, "step": index}
                for index, (easting, northing) in enumerate(line_points)
            ]
        )
        layers.append(
            alt.Chart(line_df)
            .mark_line(color=_LINE_COLOUR, strokeWidth=2)
            .encode(x=x, y=y, order="step:Q")
        )
    layers += [points, labels]
    return alt.layer(*layers).properties(width=PLAN_VIEW_WIDTH_PX, height=height_px)


# A workbook with this many holes or fewer is usually one section listed in
# order (e.g. a B-B' test workbook), so start with all of them; larger
# multi-transect workbooks start on the first few holes.
ALL_HOLES_DEFAULT_MAX = 10
PARTIAL_DEFAULT_HOLES = 4


def default_hole_sequence(hole_ids: list[str]) -> list[str]:
    """Initial "By hole sequence" selection when the workbook has no Sections tab."""
    if len(hole_ids) <= ALL_HOLES_DEFAULT_MAX:
        return list(hole_ids)
    return list(hole_ids[:PARTIAL_DEFAULT_HOLES])

def _render_hole_sequence_order(hole_ids: list[str]) -> None:
    """Numbered hole order with Up/Down (first-class fence sequence)."""
    if "hole_sequence_multiselect" not in st.session_state:
        st.session_state.hole_sequence_multiselect = default_hole_sequence(hole_ids)
    sequence: list[str] = list(st.session_state.hole_sequence_multiselect)
    if not sequence:
        return
    st.markdown("**Hole order along the section line**")
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
    section_label = st.session_state.get("consulting_section_label") or "A-A'"
    st.caption(f"Section {section_label}: **{sequence[0]} → {sequence[-1]}**")


def _unit_orders_for_code(subset: ParseResult, hole_id: str, code: str) -> list[int]:
    return sorted(
        {
            lith.unit_order
            for lith in subset.lithologies
            if lith.hole_id == hole_id
            and lith.lithology_code == code
            and lith.unit_order is not None
        }
    )


def _render_quick_correlation_links(pair_summaries: Sequence, subset: ParseResult) -> None:
    """One-click session overrides for same-code left/right-only pairs."""
    candidates: list[tuple[str, str, str, int, int]] = []
    for summary in pair_summaries:
        shared = set(summary.left_only_codes) & set(summary.right_only_codes)
        # Also offer same-name codes that appear only on one side vs the other
        # when both holes have that code with unit_order (manual re-link).
        for code in sorted(set(summary.left_only_codes) | set(summary.right_only_codes)):
            left_orders = _unit_orders_for_code(subset, summary.left_hole_id, code)
            right_orders = _unit_orders_for_code(subset, summary.right_hole_id, code)
            if left_orders and right_orders:
                candidates.append(
                    (
                        summary.left_hole_id,
                        summary.right_hole_id,
                        code,
                        left_orders[0],
                        right_orders[0],
                    )
                )
        for code in shared:
            left_orders = _unit_orders_for_code(subset, summary.left_hole_id, code)
            right_orders = _unit_orders_for_code(subset, summary.right_hole_id, code)
            if left_orders and right_orders:
                candidates.append(
                    (
                        summary.left_hole_id,
                        summary.right_hole_id,
                        code,
                        left_orders[0],
                        right_orders[0],
                    )
                )
    # Dedupe while preserving order
    seen: set[tuple[str, str, int, int]] = set()
    unique: list[tuple[str, str, str, int, int]] = []
    for item in candidates:
        key = (item[0], item[1], item[3], item[4])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    if not unique:
        return
    st.markdown("**Quick matches**")
    st.caption("Join the same layer code in two neighbouring holes with one click.")
    session_overrides: list[CorrelationOverride] = list(_session_correlation_overrides())
    for index, (left_id, right_id, code, left_order, right_order) in enumerate(unique[:12]):
        label = f"{left_id} layer {left_order} ↔ {right_id} layer {right_order} ({code})"
        if st.button(f"Match {label}", key=f"quick_corr_{index}"):
            session_overrides.append(
                CorrelationOverride(
                    left_hole_id=left_id,
                    right_hole_id=right_id,
                    left_unit_order=left_order,
                    right_unit_order=right_order,
                )
            )
            st.session_state.session_correlation_overrides = session_overrides
            st.rerun()


def render_manual_correlation_overrides(active_ids: Sequence[str], subset: ParseResult) -> None:
    with st.expander(MATCH_LAYERS_BY_HAND_LABEL, expanded=False):
        st.caption(
            "Pick a layer in each of two neighbouring holes and click Link to join them "
            "on the section. Layer numbers are the unit order from the workbook."
        )
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
                    f"{left_id} layer",
                    options=left_units,
                    key=f"corr_left_{pair_index}",
                )
            with col2:
                right_order = st.selectbox(
                    f"{right_id} layer",
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
            st.markdown(
                "**Matched by hand:** "
                + "; ".join(
                    f"{item.left_hole_id} layer {item.left_unit_order} ↔ "
                    f"{item.right_hole_id} layer {item.right_unit_order}"
                    for item in session_overrides
                )
            )
            if st.button("Clear layers matched by hand"):
                st.session_state.session_correlation_overrides = None
                st.rerun()


def render_nl_transect_input(hole_ids: list[str]) -> None:
    """Sidebar control: describe the section line in words."""
    nl_text = st.text_input(
        "Describe the section line in words",
        value=st.session_state.get("nl_transect_text", ""),
        key="nl_transect_text",
        placeholder="Section B-B' through MW-01, MW-03, MW-07",
        help="Picks out the hole IDs you name, in order. Positions still come from the collar coordinates.",
    )
    if st.button("Use this section line", key="apply_nl_transect") and nl_text.strip():
        parsed = _build_assistant().parse_transect_request(nl_text, hole_ids)
        if parsed is None:
            st.warning("Name at least two hole IDs from the Collars tab, e.g. MW-01, MW-03.")
        else:
            st.session_state.hole_sequence_multiselect = list(parsed.hole_ids)
            st.session_state.pending_transect_mode = "By hole sequence"
            if parsed.section_label:
                queue_consulting_section_label(parsed.section_label)
            st.success(
                f"Section line: {' → '.join(parsed.hole_ids)}"
                + (f" ({parsed.section_label})" if parsed.section_label else "")
            )
            st.rerun()


def render_correlation_assist(
    pair_summaries: Sequence,
    subset: ParseResult,
    active_ids: Sequence[str],
) -> None:
    """AI correlation link suggestions inside correlation health expander."""
    if st.button("Suggest layer matches", key="suggest_correlation_links"):
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
    st.markdown("**Suggested layer matches** (review before accepting)")
    for suggestion in corr_suggestions:
        st.write(
            f"{suggestion.left_hole_id} layer {suggestion.left_unit_order} ↔ "
            f"{suggestion.right_hole_id} layer {suggestion.right_unit_order} "
            f"({suggestion.confidence:.0%}) — {suggestion.rationale}"
        )
    if st.button("Accept suggested matches", key="accept_corr_suggestions"):
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
    """Section Q&A expander grounded on the active section line's facts."""
    with st.expander("Ask about this section", expanded=False):
        st.caption(
            "Answers use only the holes on this section line: depths, water levels, "
            "layer thicknesses and offsets."
        )
        qa_question = st.text_input(
            "Your question",
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
