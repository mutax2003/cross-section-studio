"""Upload welcome card and workbook parse flow."""

from __future__ import annotations

import hashlib
import logging
from io import BytesIO

import streamlit as st

from app_common import _parse_signature_key, _parse_uploaded_workbook
from app_state import clear_section_output_state
from ingestion import FormatDetector, NATIVE_PROFILE_ID
from models import ParseResult, lithologies_by_hole
from projection import suggest_offset_threshold_m

logger = logging.getLogger(__name__)


def render_welcome_card() -> None:
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


def handle_workbook_upload(
    uploaded,
    *,
    selected_profile_key: str,
    override_id: str | None,
    default_elevation_m: float,
    target_crs: str | None,
) -> ParseResult | None:
    """Detect format, parse workbook when needed, return current parse result."""
    file_bytes = uploaded.getvalue()
    if st.session_state.file_bytes != file_bytes:
        st.session_state.file_bytes = file_bytes
        st.session_state.file_hash = hashlib.sha256(file_bytes).hexdigest()[:24]
        st.session_state.parse_result = None
        st.session_state.quality_report = None
        st.session_state.transect_candidates = None
        st.session_state.import_report = None
        clear_section_output_state()
        st.session_state.polygon_overlap_warnings = []
        st.session_state.section_lithology_codes = None
        st.session_state.section_polygon_count = None
        st.session_state.section_hole_count = None
        st.session_state.lithology_index = None
        st.session_state.parse_signature = None
        st.session_state.qa_narrative = None
        st.session_state.transect_selection_key = None
        st.session_state.transect_selection = None
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
    parse_signature = _parse_signature_key(
        profile_id=profile_id,
        override_id=override_id,
        elevation_m=default_elevation_m,
        target_crs=target_crs,
        file_hash=st.session_state.file_hash,
    )
    apply_mapping = st.button("Parse workbook", type="secondary")
    auto_parse = (
        detection is not None
        and st.session_state.parse_result is None
        and st.session_state.parse_signature != parse_signature
    )
    if apply_mapping or auto_parse:
        try:
            parse_result, import_report = _parse_uploaded_workbook(
                file_bytes,
                profile_id=profile_id,
                override_id=override_id,
                elevation_m=default_elevation_m,
                target_crs=target_crs,
                auto_assign_unit_order=bool(st.session_state.get("auto_assign_unit_order", True)),
            )
            mapping_proposal = import_report.mapping_proposal
            quality_report = import_report.quality_report
            if quality_report is None:
                raise RuntimeError("Import report missing quality analysis")
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
            st.session_state._apply_suggested_offset = True
            st.session_state.lithology_index = lithologies_by_hole(parse_result.lithologies)
            st.session_state.transect_candidates = None
            clear_section_output_state()
            st.session_state.transect_selection_key = None
            st.session_state.transect_selection = None
            st.session_state.polygon_overlap_warnings = []
            st.session_state.render_cache_key = None
            st.session_state.section_lithology_codes = None
            st.session_state.section_polygon_count = None
            st.session_state.section_hole_count = None
            st.session_state.parse_signature = parse_signature
            st.caption(
                f"Suggested transect offset threshold: **{st.session_state.suggested_offset_m:.0f} m** "
                "(applied to sidebar warnings)."
            )
            st.rerun()
        except Exception as exc:
            logger.exception("Workbook parse failed")
            st.session_state.parse_result = None
            st.session_state.import_report = None
            st.session_state.quality_report = None
            st.session_state.parse_signature = None
            st.error(f"Failed to parse workbook: {exc}")
            return None

    return st.session_state.parse_result
