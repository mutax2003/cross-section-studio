"""Streamlit application for borehole cross-section generation."""

from __future__ import annotations

import base64
import json
import logging
import re
import traceback
from io import BytesIO
from typing import Any, Sequence

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ai_assistant import AIAssistant, OpenAIProvider
from ai_quality import (
    analyze_parsed_data,
    load_lithology_aliases,
    save_lithology_alias,
)
from ingestion import (
    FormatDetector,
    ImportReport,
    NATIVE_PROFILE_ID,
    ingest_workbook,
    list_profiles,
)
from constants import USGS_LITHOLOGY_HATCHES, get_lithology_style
from models import Collar, Lithology, ParseResult, Transect, WaterLevel, subset_parse_result
from pipeline import DEFAULT_UNCERTAINTY_SPACING_M, build_cross_section, validate_interpretation_mode
from projection import DEFAULT_OFFSET_WARNING_M, off_transect_warnings, suggest_offset_threshold_m
from transect_planner import recommend_transects
from ui_helpers import (
    legend_hatch_background,
    svg_display_height,
    svg_is_valid,
    transect_cache_key,
    workflow_stage,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Cross Section Studio",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

_APP_CSS = """
<style>
    :root {
        --brand-dark: #1e3a2f;
        --brand-mid: #2e6b4f;
        --brand-light: #3d8b5f;
        --surface: #ffffff;
        --border: #e2e8f0;
        --text: #1e293b;
        --muted: #64748b;
    }
    .app-hero {
        background: linear-gradient(135deg, var(--brand-dark) 0%, var(--brand-mid) 55%, var(--brand-light) 100%);
        padding: 1.35rem 1.5rem 1.1rem;
        border-radius: 14px;
        margin-bottom: 0.85rem;
        color: #f8fafc;
        box-shadow: 0 8px 24px rgba(30, 58, 47, 0.18);
    }
    .app-hero h1 { color: #f8fafc !important; margin: 0; font-size: 1.7rem; letter-spacing: -0.02em; }
    .app-hero p { margin: 0.35rem 0 0; opacity: 0.93; font-size: 0.92rem; line-height: 1.45; }
    .workflow {
        display: flex;
        gap: 0.45rem;
        flex-wrap: wrap;
        margin: 0.75rem 0 0.25rem;
    }
    .workflow-step {
        flex: 1 1 8rem;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.22);
        border-radius: 10px;
        padding: 0.45rem 0.65rem;
        font-size: 0.78rem;
        color: #e2e8f0;
        text-align: center;
    }
    .workflow-step.active {
        background: rgba(255,255,255,0.95);
        color: var(--brand-dark);
        font-weight: 600;
        border-color: transparent;
    }
    .workflow-step.done { opacity: 0.88; }
    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.8rem 0.9rem;
        text-align: center;
        min-height: 4.5rem;
    }
    .metric-card .value { font-size: 1.45rem; font-weight: 700; color: var(--text); line-height: 1.2; }
    .metric-card .label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.15rem; }
    .metric-card.ok { border-color: #86efac; background: linear-gradient(180deg, #f0fdf4 0%, #fff 100%); }
    .metric-card.warn { border-color: #fcd34d; background: linear-gradient(180deg, #fffbeb 0%, #fff 100%); }
    .metric-card.error { border-color: #fca5a5; background: linear-gradient(180deg, #fef2f2 0%, #fff 100%); }
    .section-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.15rem 0.65rem;
        margin-top: 0.65rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }
    .profile-header {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.35rem 0 0.85rem;
    }
    .chip {
        display: inline-block;
        padding: 0.22rem 0.55rem;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 600;
        border: 1px solid var(--border);
        background: #f8fafc;
        color: #334155;
    }
    .chip.brand { background: #ecfdf5; border-color: #a7f3d0; color: #065f46; }
    .chip.warn { background: #fffbeb; border-color: #fde68a; color: #92400e; }
    .legend-swatch {
        display: inline-block;
        width: 24px;
        height: 17px;
        border: 1px solid #334155;
        border-radius: 4px;
        margin-right: 8px;
        vertical-align: middle;
        background-size: 6px 6px, 6px 6px;
        background-position: 0 0, 3px 3px;
    }
    .legend-row { margin: 0.38rem 0; font-size: 0.84rem; color: #334155; line-height: 1.35; }
    .welcome-card {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px dashed #cbd5e1;
        border-radius: 14px;
        padding: 1.75rem 1.5rem;
        color: #475569;
    }
    .welcome-card h3 { margin: 0 0 0.5rem; color: #1e293b; }
    .welcome-steps { text-align: left; margin: 1rem auto 0; max-width: 34rem; }
    .welcome-steps li { margin: 0.35rem 0; }
    .stale-banner {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e40af;
        border-radius: 10px;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.75rem;
        font-size: 0.88rem;
    }
    .sidebar-section-title {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        margin: 0.35rem 0 0.15rem;
        font-weight: 700;
    }
    .svg-frame {
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
        background: #fff;
    }
    div[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    div[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        font-weight: 600;
    }
</style>
"""
st.markdown(_APP_CSS, unsafe_allow_html=True)

_WORKFLOW_LABELS = ("Upload", "Validate", "Configure", "Generate")


def _render_workflow_stepper(stage: int) -> None:
    steps_html = []
    for index, label in enumerate(_WORKFLOW_LABELS):
        if index < stage:
            css_class = "workflow-step done"
        elif index == stage:
            css_class = "workflow-step active"
        else:
            css_class = "workflow-step"
        steps_html.append(f'<div class="{css_class}">{index + 1}. {label}</div>')
    st.markdown(f'<div class="workflow">{"".join(steps_html)}</div>', unsafe_allow_html=True)


def _render_hero(stage: int) -> None:
    st.markdown(
        """
<div class="app-hero">
  <h1>Cross Section Studio</h1>
  <p>Upload borehole data, validate stratigraphy, and generate publication-ready profiles with USGS-style fills and hatch patterns.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    _render_workflow_stepper(stage)


def _metric_tone(error_count: int, warning_count: int, *, errors_only: bool = False) -> str:
    if error_count > 0:
        return "error"
    if errors_only:
        return "ok"
    if warning_count > 0:
        return "warn"
    return "ok"


def _render_metric_card(value: str | int, label: str, tone: str = "ok") -> None:
    st.markdown(
        f'<div class="metric-card {tone}"><div class="value">{value}</div><div class="label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def _render_profile_chips(
    *,
    interpretation_mode: str,
    vertical_exaggeration: float,
    hole_count: int | None,
    polygon_count: int | None,
    is_stale: bool,
) -> None:
    mode_label = (
        "Observed only"
        if interpretation_mode == "borehole_only"
        else "Interpolated fence"
    )
    chips = [
        f'<span class="chip brand">{mode_label}</span>',
        f'<span class="chip">VE {vertical_exaggeration:g}×</span>',
    ]
    if hole_count is not None:
        chips.append(f'<span class="chip">{hole_count} boreholes</span>')
    if polygon_count is not None and interpretation_mode == "interpolated":
        chips.append(f'<span class="chip">{polygon_count} polygons</span>')
    if is_stale:
        chips.append('<span class="chip warn">Settings changed</span>')
    st.markdown(f'<div class="profile-header">{"".join(chips)}</div>', unsafe_allow_html=True)


def _sidebar_heading(title: str) -> None:
    st.markdown(f'<div class="sidebar-section-title">{title}</div>', unsafe_allow_html=True)


DEFAULT_SESSION: dict[str, Any] = {
    "svg_bytes": None,
    "mapping_proposal": None,
    "quality_report": None,
    "parse_result": None,
    "qa_narrative": None,
    "file_bytes": None,
    "transect_candidates": None,
    "lithology_aliases": None,
    "import_report": None,
    "detection_result": None,
    "section_lithology_codes": None,
    "hole_ids": [],
    "unique_lithology_codes": [],
    "render_cache_key": None,
    "polygon_overlap_warnings": [],
    "suggested_offset_m": 50.0,
    "section_polygon_count": None,
    "section_hole_count": None,
    "collars_json": None,
    "lithologies_json": None,
    "water_levels_json": None,
}
for key, value in DEFAULT_SESSION.items():
    if key not in st.session_state:
        st.session_state[key] = value


def _get_aliases() -> dict[str, str]:
    if st.session_state.lithology_aliases is None:
        st.session_state.lithology_aliases = load_lithology_aliases()
    return st.session_state.lithology_aliases


def _render_lithology_legend(codes: list[str]) -> None:
    if not codes:
        st.caption("Legend appears after you generate a cross-section.")
        return
    rows = []
    for code in sorted(codes):
        style = get_lithology_style(code)
        hatch = USGS_LITHOLOGY_HATCHES.get(code, "..")
        hatch_bg = legend_hatch_background(hatch)
        rows.append(
            f'<div class="legend-row">'
            f'<span class="legend-swatch" style="background-color:{style.color};'
            f'background-image:{hatch_bg};"></span>'
            f"<strong>{code}</strong>"
            f"</div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def _display_svg(svg_bytes: bytes) -> None:
    """Render SVG in Streamlit (st.image does not support SVG via PIL)."""
    if not svg_is_valid(svg_bytes):
        st.error("Renderer produced invalid or empty SVG output.")
        return
    text = svg_bytes.decode("utf-8", errors="replace").strip()
    frame_height = svg_display_height(svg_bytes)
    st.markdown('<div class="svg-frame">', unsafe_allow_html=True)
    try:
        components.html(
            f'<div style="width:100%;overflow-x:auto;line-height:0;">{text}</div>',
            height=frame_height,
            scrolling=True,
        )
    except Exception as exc:
        logger.warning("components.html failed, using base64 fallback: %s", exc)
        encoded = base64.b64encode(svg_bytes).decode("ascii")
        st.markdown(
            f'<img src="data:image/svg+xml;base64,{encoded}" '
            'style="width:100%;height:auto;display:block;" alt="Cross-section profile" />',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_overlap_warnings(warnings: Sequence[str]) -> None:
    if not warnings:
        return
    if len(warnings) == 1:
        st.warning(f"Polygon overlap: {warnings[0]}")
        return
    with st.expander(
        f"{len(warnings)} polygon overlaps detected on section",
        expanded=True,
    ):
        for message in warnings:
            st.write(message)


def _active_transect_selection(
    parse_result: ParseResult,
    transect_mode: str,
    selected_holes: list[str],
    coordinate_text: str,
) -> tuple[tuple[str, ...], tuple[tuple[float, float], ...]] | None:
    collar_lookup = {collar.hole_id: collar for collar in parse_result.collars}
    if transect_mode == "By coordinates":
        try:
            transect_points = tuple(_parse_coordinate_lines(coordinate_text))
        except ValueError:
            return None
        return tuple(collar.hole_id for collar in parse_result.collars), transect_points
    if len(selected_holes) < 2:
        return None
    missing = [hole_id for hole_id in selected_holes if hole_id not in collar_lookup]
    if missing:
        return None
    transect_points = tuple(
        (collar_lookup[hole_id].easting, collar_lookup[hole_id].northing)
        for hole_id in selected_holes
    )
    return tuple(selected_holes), transect_points


def _sidebar_render_cache_key(
    parse_result: ParseResult,
    transect_mode: str,
    selected_holes: list[str],
    coordinate_text: str,
    vertical_exaggeration: float,
    offset_warning_m: float,
    show_hatches: bool,
    show_legend: bool,
    section_title: str,
    interpretation_mode: str,
    allow_pinch_outs: bool,
    uncertainty_spacing_m: float,
    uncertainty_offset_m: float,
) -> str | None:
    selection = _active_transect_selection(
        parse_result,
        transect_mode,
        selected_holes,
        coordinate_text,
    )
    if selection is None:
        return None
    active_hole_ids, transect_points = selection
    return _transect_cache_key(
        active_hole_ids,
        transect_points,
        vertical_exaggeration,
        offset_warning_m,
        show_hatches,
        show_legend,
        section_title,
        interpretation_mode,
        allow_pinch_outs,
        uncertainty_spacing_m,
        uncertainty_offset_m,
    )


def _transect_cache_key(
    hole_ids: tuple[str, ...],
    transect_points: tuple[tuple[float, float], ...],
    vertical_exaggeration: float,
    offset_warning_m: float,
    show_hatches: bool,
    show_legend: bool,
    section_title: str,
    interpretation_mode: str,
    allow_pinch_outs: bool,
    uncertainty_spacing_m: float,
    uncertainty_offset_m: float,
) -> str:
    return transect_cache_key(
        hole_ids,
        transect_points,
        vertical_exaggeration,
        offset_warning_m,
        show_hatches,
        show_legend,
        section_title,
        interpretation_mode,
        allow_pinch_outs,
        uncertainty_spacing_m,
        uncertainty_offset_m,
    )


def _parse_coordinate_lines(text: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = re.split(r"[,\s]+", line)
        if len(parts) != 2:
            raise ValueError(f"Line {line_number}: expected 'easting northing' pair")
        points.append((float(parts[0]), float(parts[1])))
    if len(points) < 2:
        raise ValueError("At least two coordinate pairs are required")
    return points


def _health_emoji(error_count: int, warning_count: int) -> str:
    if error_count > 0:
        return "🔴"
    if warning_count > 0:
        return "🟡"
    return "🟢"


def _build_assistant() -> AIAssistant:
    api_key = st.session_state.get("openai_api_key", "").strip()
    if api_key:
        return AIAssistant(OpenAIProvider(api_key=api_key))
    return AIAssistant(None)


def _mapping_rows(mappings: tuple) -> list[dict[str, str]]:
    return [
        {
            "source": mapping.source_column,
            "canonical": mapping.canonical_column,
            "confidence": f"{mapping.confidence:.0%}",
        }
        for mapping in mappings
    ]


def _parse_result_json_bundle(parse_result: ParseResult) -> tuple[str, str, str]:
    return (
        json.dumps([collar.model_dump() for collar in parse_result.collars]),
        json.dumps([lit.model_dump() for lit in parse_result.lithologies]),
        json.dumps([level.model_dump() for level in parse_result.water_levels]),
    )


def _parse_uploaded_workbook(
    file_bytes: bytes,
    *,
    profile_id: str | None,
    override_id: str | None,
    elevation_m: float | None,
    target_crs: str | None,
) -> tuple[ParseResult, ImportReport]:
    return _cached_ingest_workbook(
        file_bytes,
        profile_id,
        override_id,
        elevation_m,
        target_crs,
        json.dumps(_get_aliases(), sort_keys=True),
    )


@st.cache_data(show_spinner="Parsing workbook...")
def _cached_ingest_workbook(
    file_bytes: bytes,
    profile_id: str | None,
    override_id: str | None,
    elevation_m: float | None,
    target_crs: str | None,
    aliases_json: str,
) -> tuple[ParseResult, ImportReport]:
    aliases = json.loads(aliases_json)
    return ingest_workbook(
        BytesIO(file_bytes),
        profile_id=profile_id,
        override_id=override_id,
        elevation_m=elevation_m,
        target_crs=target_crs,
        lithology_aliases=aliases,
    )


@st.cache_data(show_spinner=False)
def _cached_subset_json(
    collars_json: str,
    lithologies_json: str,
    water_levels_json: str,
    hole_ids: tuple[str, ...],
) -> tuple[str, str, str]:
    parse_result = ParseResult(
        collars=tuple(Collar.model_validate(item) for item in json.loads(collars_json)),
        lithologies=tuple(Lithology.model_validate(item) for item in json.loads(lithologies_json)),
        errors=(),
        water_levels=tuple(
            WaterLevel.model_validate(item) for item in json.loads(water_levels_json)
        ),
    )
    subset = subset_parse_result(parse_result, hole_ids)
    return _parse_result_json_bundle(subset)


@st.cache_data(show_spinner=False)
def _cached_recommend_transects(
    collars_json: str,
    lithologies_json: str,
    top_n: int,
) -> list:
    collars = [Collar.model_validate(item) for item in json.loads(collars_json)]
    lithologies = [Lithology.model_validate(item) for item in json.loads(lithologies_json)]
    return recommend_transects(collars, lithologies, top_n=top_n)


@st.cache_data(show_spinner=False)
def _cached_render_section(
    collars_json: str,
    lithologies_json: str,
    transect_points: tuple[tuple[float, float], ...],
    vertical_exaggeration: float,
    show_hatches: bool,
    show_legend: bool,
    section_title: str,
    interpretation_mode: str,
    allow_pinch_outs: bool,
    water_levels_json: str,
    offset_warning_m: float,
    uncertainty_spacing_m: float,
    uncertainty_offset_m: float,
) -> tuple[bytes, int, tuple[str, ...], tuple[str, ...]]:
    collars = tuple(Collar.model_validate(item) for item in json.loads(collars_json))
    lithologies = tuple(Lithology.model_validate(item) for item in json.loads(lithologies_json))
    water_levels = tuple(
        WaterLevel.model_validate(item) for item in json.loads(water_levels_json)
    )
    mode = validate_interpretation_mode(interpretation_mode)
    _, polygons, svg_bytes, lithology_codes, overlap_warnings = build_cross_section(
        collars,
        lithologies,
        transect_points,
        vertical_exaggeration=vertical_exaggeration,
        show_hatches=show_hatches,
        show_legend=show_legend,
        title=section_title,
        interpretation_mode=mode,
        allow_pinch_outs=allow_pinch_outs,
        water_levels=water_levels or None,
        offset_warning_m=offset_warning_m,
        uncertainty_spacing_m=uncertainty_spacing_m,
        uncertainty_offset_m=uncertainty_offset_m,
    )
    return svg_bytes, len(polygons), lithology_codes, overlap_warnings


def _generate_cross_section(
    parse_result: ParseResult,
    transect_points: list[tuple[float, float]],
    hole_ids: tuple[str, ...],
    vertical_exaggeration: float,
    offset_warning_m: float,
    *,
    show_hatches: bool = True,
    show_legend: bool = True,
    section_title: str = "Borehole Cross-Section",
    interpretation_mode: str = "interpolated",
    allow_pinch_outs: bool = True,
    uncertainty_spacing_m: float = DEFAULT_UNCERTAINTY_SPACING_M,
    uncertainty_offset_m: float = DEFAULT_OFFSET_WARNING_M,
) -> tuple[bytes, int, list[str], list[str]]:
    subset = subset_parse_result(parse_result, hole_ids)
    if len(subset.collars) < 2:
        raise ValueError("Select at least two boreholes with collar and lithology data")
    if st.session_state.collars_json is None:
        (
            st.session_state.collars_json,
            st.session_state.lithologies_json,
            st.session_state.water_levels_json,
        ) = _parse_result_json_bundle(parse_result)

    transect = Transect(points=transect_points)
    warnings = off_transect_warnings(subset.collars, transect, offset_warning_m)
    if warnings:
        if len(warnings) == 1:
            st.warning(warnings[0])
        else:
            with st.expander(
                f"{len(warnings)} boreholes exceed transect offset threshold ({offset_warning_m:.0f} m)",
                expanded=False,
            ):
                for message in warnings:
                    st.write(message)

    collars_json, lithologies_json, water_levels_json = _cached_subset_json(
        st.session_state.collars_json,
        st.session_state.lithologies_json,
        st.session_state.water_levels_json,
        hole_ids,
    )
    svg_bytes, polygon_count, lithology_codes, overlap_warnings = _cached_render_section(
        collars_json,
        lithologies_json,
        tuple(transect_points),
        vertical_exaggeration,
        show_hatches,
        show_legend,
        section_title,
        interpretation_mode,
        allow_pinch_outs,
        water_levels_json,
        offset_warning_m,
        uncertainty_spacing_m,
        uncertainty_offset_m,
    )
    return svg_bytes, polygon_count, list(lithology_codes), list(overlap_warnings)


with st.sidebar:
    _sidebar_heading("Data source")
    uploaded = st.file_uploader(
        "Borehole database (.xlsx)",
        type=["xlsx"],
        help="Native Collars/Lithology workbook or field export with Lat/Long",
    )

    st.divider()
    _sidebar_heading("Cross-section style")
    interpretation_mode = st.radio(
        "Interpretation",
        options=["interpolated", "borehole_only"],
        format_func=lambda value: (
            "Interpolated fence diagram"
            if value == "interpolated"
            else "Observed only (no inter-hole fill)"
        ),
        help="Observed only shows logged intervals on borehole sticks without correlation.",
    )
    allow_pinch_outs = st.toggle(
        "Allow pinch-out wedges",
        value=True,
        disabled=interpretation_mode == "borehole_only",
        help="When off, units present in only one hole are not inferred between holes.",
    )
    show_hatches = st.toggle("Hatch patterns", value=True)
    show_legend = st.toggle("Legend on chart", value=True)
    section_title = st.text_input("Section title", value="Borehole Cross-Section")

    st.divider()
    _sidebar_heading("Transect")
    transect_mode = st.radio(
        "Definition mode",
        options=["By hole sequence", "By coordinates", "Recommended"],
        label_visibility="collapsed",
    )
    vertical_exaggeration = st.slider(
        "Vertical exaggeration",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
    )
    offset_warning_m = st.number_input(
        "Transect offset warning (m)",
        min_value=1.0,
        value=float(st.session_state.get("suggested_offset_m", 50.0)),
        step=5.0,
        help="Warn when a selected borehole is farther than this from the transect line",
    )
    override_warnings = st.checkbox("Allow generate with warnings", value=True)

    st.divider()
    with st.expander("Interpretation uncertainty", expanded=False):
        uncertainty_spacing_m = st.number_input(
            "Uncertain zone spacing (m)",
            min_value=10.0,
            value=float(DEFAULT_UNCERTAINTY_SPACING_M),
            step=10.0,
            disabled=interpretation_mode == "borehole_only",
            help="Shade inter-hole zones when adjacent boreholes are farther apart than this.",
        )
        uncertainty_offset_m = st.number_input(
            "Uncertain zone offset (m)",
            min_value=1.0,
            value=float(st.session_state.get("suggested_offset_m", 50.0)),
            step=5.0,
            disabled=interpretation_mode == "borehole_only",
            help="Shade inter-hole zones when either borehole exceeds this transect offset.",
        )

    with st.expander("Workbook import", expanded=False):
        profile_options = {profile.id: profile.label for profile in list_profiles()}
        profile_options[NATIVE_PROFILE_ID] = "Native platform (Collars + Lithology)"
        auto_profile = st.session_state.get("detection_result")
        default_profile_index = 0
        profile_ids = ["auto"] + list(profile_options.keys())
        if auto_profile is not None and auto_profile.profile_id in profile_options:
            default_profile_index = profile_ids.index(auto_profile.profile_id)
        selected_profile_key = st.selectbox(
            "Import profile",
            options=profile_ids,
            format_func=lambda key: "Auto-detect" if key == "auto" else profile_options[key],
            index=min(default_profile_index, len(profile_ids) - 1),
        )
        override_id = st.text_input(
            "Profile override (optional)",
            value="advantage_phase2_2026" if auto_profile and not auto_profile.is_native else "",
            help="e.g. advantage_phase2_2026 for per-site coordinate offsets",
        ).strip() or None
        default_elevation_m = st.number_input(
            "Default collar elevation (m)",
            min_value=0.0,
            value=100.0,
            step=1.0,
            help="Used for field exports without RL/elevation column",
        )
        target_crs = st.text_input(
            "Target CRS (EPSG)",
            value="EPSG:32611",
            help="UTM zone for Lat/Long field exports (Alberta default: 32611)",
        ).strip() or None

    with st.expander("AI Assist (optional)", expanded=False):
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API key",
            type="password",
            help="Headers-only column mapping and QA narratives. Leave blank for local checks only.",
        )
        enable_ai = st.checkbox("Enable LLM suggestions", value=False)

_render_hero(
    workflow_stage(
        has_upload=uploaded is not None,
        has_parse_result=st.session_state.parse_result is not None,
        has_profile=st.session_state.svg_bytes is not None,
    )
)

parse_result: ParseResult | None = st.session_state.parse_result
hole_ids: list[str] = list(st.session_state.hole_ids)
quality_report = st.session_state.quality_report
mapping_proposal = st.session_state.mapping_proposal

if uploaded is None:
    st.markdown(
        """
<div class="welcome-card">
  <h3>Get started in four steps</h3>
  <p>Upload an Excel workbook in the sidebar to inspect stratigraphy, validate data health, and build a cross-section profile.</p>
  <ol class="welcome-steps">
    <li><strong>Upload</strong> a native <em>Collars + Lithology</em> workbook or a field export with <em>Lat/Long</em>.</li>
    <li><strong>Validate</strong> import mapping, lithology aliases, and QA issues in Data Health.</li>
    <li><strong>Configure</strong> interpretation mode, transect holes, and vertical exaggeration in the sidebar.</li>
    <li><strong>Generate</strong> the profile and download publication-ready SVG.</li>
  </ol>
</div>
""",
        unsafe_allow_html=True,
    )
else:
    file_bytes = uploaded.getvalue()
    if st.session_state.file_bytes != file_bytes:
        st.session_state.file_bytes = file_bytes
        st.session_state.parse_result = None
        st.session_state.quality_report = None
        st.session_state.transect_candidates = None
        st.session_state.import_report = None
        st.session_state.svg_bytes = None
        st.session_state.polygon_overlap_warnings = []
        st.session_state.render_cache_key = None
        st.session_state.section_lithology_codes = None
        st.session_state.section_polygon_count = None
        st.session_state.section_hole_count = None
        st.session_state.collars_json = None
        st.session_state.lithologies_json = None
        st.session_state.water_levels_json = None
        st.session_state.qa_narrative = None
        try:
            st.session_state.detection_result = FormatDetector().detect(BytesIO(file_bytes))
        except Exception as exc:
            st.session_state.detection_result = None
            st.error(f"Failed to inspect workbook: {exc}")

    detection = st.session_state.detection_result
    if detection is not None:
        st.caption(
            f"Detected format: **{detection.label}** "
            f"({detection.confidence:.0%} confidence)"
        )
        if detection.profile_id != NATIVE_PROFILE_ID:
            st.info(
                "Field Data sheet (if present) is not used for stratigraphy — "
                "OVA overlay is planned for a future release."
            )

    profile_id = None if selected_profile_key == "auto" else selected_profile_key
    apply_mapping = st.button("Parse workbook", type="secondary")
    auto_parse = detection is not None and st.session_state.parse_result is None
    needs_parse = apply_mapping or auto_parse

    if needs_parse:
        try:
            parse_result, import_report = _parse_uploaded_workbook(
                file_bytes,
                profile_id=profile_id,
                override_id=override_id,
                elevation_m=default_elevation_m,
                target_crs=target_crs,
            )
            mapping_proposal = import_report.mapping_proposal
            quality_report = analyze_parsed_data(
                parse_result.collars,
                parse_result.lithologies,
                mapping_proposal=mapping_proposal,
                aliases=_get_aliases(),
            )
            hole_ids = [collar.hole_id for collar in parse_result.collars]
            st.session_state.parse_result = parse_result
            st.session_state.import_report = import_report
            st.session_state.mapping_proposal = mapping_proposal
            st.session_state.quality_report = quality_report
            st.session_state.hole_ids = hole_ids
            st.session_state.unique_lithology_codes = sorted(
                {lit.lithology_code for lit in parse_result.lithologies}
            )
            st.session_state.suggested_offset_m = suggest_offset_threshold_m(parse_result.collars)
            (
                st.session_state.collars_json,
                st.session_state.lithologies_json,
                st.session_state.water_levels_json,
            ) = _parse_result_json_bundle(parse_result)
            st.session_state.transect_candidates = None
            st.session_state.svg_bytes = None
            st.session_state.polygon_overlap_warnings = []
            st.session_state.render_cache_key = None
            st.session_state.section_lithology_codes = None
            st.session_state.section_polygon_count = None
            st.session_state.section_hole_count = None
        except Exception as exc:
            st.error(f"Failed to parse workbook: {exc}")
            parse_result = None

    if st.session_state.parse_result is not None:
        parse_result = st.session_state.parse_result
        import_report = st.session_state.import_report
        quality_report = st.session_state.quality_report
        mapping_proposal = st.session_state.mapping_proposal
        hole_ids = list(st.session_state.hole_ids)
        unique_lithologies = list(st.session_state.unique_lithology_codes)

        st.subheader("Data Health")
        assistant = _build_assistant() if enable_ai else AIAssistant(None)

        if import_report is not None:
            with st.expander("Import report", expanded=import_report.profile_id != NATIVE_PROFILE_ID):
                st.markdown(
                    f"**Profile:** {import_report.profile_label} (`{import_report.profile_id}`)"
                )
                st.markdown(
                    f"**Holes:** {import_report.hole_count} · "
                    f"**Intervals:** {import_report.lithology_interval_count}"
                )
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

                if enable_ai and assistant.enabled and mapping_proposal.low_confidence_mappings:
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

        hole_ids = list(st.session_state.hole_ids)

        health_tone = _metric_tone(quality_report.error_count, quality_report.warning_count)
        m1, m2, m3, m4 = st.columns(4)
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
        if quality_report.unmapped_lithologies:
            st.warning(
                "Unmapped lithology codes: "
                + ", ".join(quality_report.unmapped_lithologies)
            )
            if enable_ai and assistant.enabled and st.button("Suggest lithology aliases"):
                suggestions = assistant.suggest_lithology_mappings(
                    quality_report.unmapped_lithologies
                )
                for suggestion in suggestions:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(
                            f"`{suggestion.source_code}` → `{suggestion.canonical_code}` "
                            f"({suggestion.confidence:.0%}) — {suggestion.rationale}"
                        )
                    with col2:
                        if st.button("Save alias", key=f"alias_{suggestion.source_code}"):
                            save_lithology_alias(
                                suggestion.source_code,
                                suggestion.canonical_code,
                            )
                            st.session_state.lithology_aliases = None
                            st.success(f"Saved alias for {suggestion.source_code}")
                            st.rerun()

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

        if st.button("Generate QA narrative"):
            st.session_state.qa_narrative = assistant.explain_quality_issues(
                quality_report.issues
            )

        if st.session_state.qa_narrative:
            st.markdown("**QA Summary**")
            st.write(st.session_state.qa_narrative)

        for message in parse_result.errors:
            st.warning(message)

selected_holes: list[str] = []
coordinate_text = ""

if parse_result is not None:
    with st.sidebar:
        st.divider()
        _sidebar_heading("Stratigraphy legend")
        legend_codes = st.session_state.section_lithology_codes or st.session_state.unique_lithology_codes
        _render_lithology_legend(legend_codes)

        _sidebar_heading("Transect selection")
        if transect_mode == "Recommended":
            if st.session_state.transect_candidates is None:
                if st.session_state.collars_json and st.session_state.lithologies_json:
                    st.session_state.transect_candidates = _cached_recommend_transects(
                        st.session_state.collars_json,
                        st.session_state.lithologies_json,
                        3,
                    )
                else:
                    st.session_state.transect_candidates = recommend_transects(
                        list(parse_result.collars),
                        list(parse_result.lithologies),
                        top_n=3,
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
            selected_holes = st.multiselect(
                "Hole sequence",
                options=hole_ids,
                default=hole_ids[: min(4, len(hole_ids))],
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

    blocking = quality_report is not None and quality_report.has_blocking_errors
    has_warnings = quality_report is not None and quality_report.warning_count > 0
    can_generate = parse_result is not None and not blocking and (override_warnings or not has_warnings)

    if blocking:
        st.error("Resolve data errors before generating a cross-section.")
    elif has_warnings and not override_warnings:
        st.warning("Warnings detected. Enable 'Allow generate with warnings' in the sidebar to proceed.")

    gen_col1, gen_col2 = st.columns([1, 3])
    with gen_col1:
        generate_clicked = st.button(
            "Generate Cross-Section",
            type="primary",
            disabled=not can_generate,
            width="stretch",
        )
    with gen_col2:
        if can_generate:
            st.caption("Builds the profile from sidebar style and transect settings.")
        elif blocking:
            st.caption("Fix blocking QA errors before generating.")
        else:
            st.caption("Enable warnings override in the sidebar or resolve QA issues.")

    if generate_clicked:
        try:
            selection = _active_transect_selection(
                parse_result,
                transect_mode,
                selected_holes,
                coordinate_text,
            )
            if selection is None:
                if transect_mode == "By coordinates":
                    raise ValueError("Enter at least two valid coordinate pairs for the transect")
                raise ValueError("Select at least two holes for the transect")
            active_hole_ids, transect_points = selection
            transect_point_list = list(transect_points)

            cache_key = _sidebar_render_cache_key(
                parse_result,
                transect_mode,
                selected_holes,
                coordinate_text,
                vertical_exaggeration,
                offset_warning_m,
                show_hatches,
                show_legend,
                section_title,
                interpretation_mode,
                allow_pinch_outs,
                uncertainty_spacing_m,
                uncertainty_offset_m,
            )

            svg_bytes, polygon_count, lithology_codes, overlap_warnings = _generate_cross_section(
                parse_result,
                transect_point_list,
                active_hole_ids,
                vertical_exaggeration,
                offset_warning_m,
                show_hatches=show_hatches,
                show_legend=show_legend,
                section_title=section_title,
                interpretation_mode=interpretation_mode,
                allow_pinch_outs=allow_pinch_outs,
                uncertainty_spacing_m=uncertainty_spacing_m,
                uncertainty_offset_m=uncertainty_offset_m,
            )
            st.session_state.svg_bytes = svg_bytes
            st.session_state.section_lithology_codes = lithology_codes
            st.session_state.section_polygon_count = polygon_count
            st.session_state.section_hole_count = len(active_hole_ids)
            st.session_state.polygon_overlap_warnings = overlap_warnings
            st.session_state.render_cache_key = cache_key
            if interpretation_mode == "borehole_only":
                st.success(
                    f"Generated observed-data cross-section across {len(active_hole_ids)} boreholes."
                )
            else:
                st.success(
                    f"Generated cross-section with {polygon_count} geological polygons "
                    f"across {len(active_hole_ids)} boreholes."
                )
        except Exception as exc:
            logger.exception("Cross-section generation failed")
            st.error(str(exc))
            with st.expander("Error details"):
                st.code(traceback.format_exc())

if st.session_state.svg_bytes is not None:
    is_stale = False
    if st.session_state.render_cache_key is not None and parse_result is not None:
        current_key = _sidebar_render_cache_key(
            parse_result,
            transect_mode,
            selected_holes,
            coordinate_text,
            vertical_exaggeration,
            offset_warning_m,
            show_hatches,
            show_legend,
            section_title,
            interpretation_mode,
            allow_pinch_outs,
            uncertainty_spacing_m,
            uncertainty_offset_m,
        )
        is_stale = current_key is not None and current_key != st.session_state.render_cache_key

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Cross-Section Profile")
    _render_profile_chips(
        interpretation_mode=interpretation_mode,
        vertical_exaggeration=vertical_exaggeration,
        hole_count=st.session_state.section_hole_count,
        polygon_count=st.session_state.section_polygon_count,
        is_stale=is_stale,
    )
    if is_stale:
        st.markdown(
            '<div class="stale-banner">Sidebar settings changed — click <strong>Generate Cross-Section</strong> to refresh the profile.</div>',
            unsafe_allow_html=True,
        )
    _render_overlap_warnings(st.session_state.polygon_overlap_warnings)
    _display_svg(st.session_state.svg_bytes)
    st.markdown("</div>", unsafe_allow_html=True)

    dl_col1, dl_col2 = st.columns([1, 3])
    with dl_col1:
        st.download_button(
            label="Download SVG",
            data=st.session_state.svg_bytes,
            file_name="cross_section.svg",
            mime="image/svg+xml",
            type="primary",
            width="stretch",
        )
    with dl_col2:
        st.caption("Vector output suitable for reports, GIS overlays, and further editing.")

with st.expander("Excel format"):
    st.markdown(
        """
**Native platform:** `Collars` + `Lithology` sheets with `hole_id`, `easting`, `northing`, etc.

**Field export:** single `Lithology` sheet with `Label`, `Depth` (e.g. `0.00-2.00m`), `Lithology`, `Lat`, `Long` — auto-converted to UTM on import.

Column names are fuzzy-matched for native workbooks (e.g. `BH`, `RL`, `TD`, `CLY`).
        """
    )
