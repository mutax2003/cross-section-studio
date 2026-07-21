"""Upload welcome card and workbook parse flow."""

from __future__ import annotations

import hashlib
import logging
from io import BytesIO

import streamlit as st

from app_common import _parse_signature_key, _parse_uploaded_workbook
from app_state import (
    DEFAULT_SESSION,
    SESSION_PARSE_KEYS,
    clear_ai_session_state,
    clear_section_output_state,
)
from ingestion import NATIVE_PROFILE_ID, FormatDetector
from models import ParseResult, lithologies_by_hole
from ops_audit import audit_event
from paths import sample_boreholes_workbook
from projection import suggest_offset_threshold_m

logger = logging.getLogger(__name__)

_TEMPLATE_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_TEMPLATE_FALLBACK_NAME = "Cross_Section_Input_Template.xlsx"


class _BytesUpload:
    """Minimal upload shim for session-stored workbook bytes."""

    def __init__(self, data: bytes, name: str) -> None:
        self._data = data
        self.name = name

    def getvalue(self) -> bytes:
        return self._data


def clear_workbook_session() -> None:
    """Clear parse/session workbook state and any leftover section SVG/PNG/PDF."""
    for key in SESSION_PARSE_KEYS:
        if key in DEFAULT_SESSION:
            st.session_state[key] = DEFAULT_SESSION[key]
        elif key in st.session_state:
            st.session_state[key] = None
    clear_section_output_state()
    clear_ai_session_state()
    st.session_state.uploaded_name = DEFAULT_SESSION.get("uploaded_name")
    st.session_state.transect_candidates = None
    st.session_state.suggested_offset_m = DEFAULT_SESSION.get("suggested_offset_m", 50.0)


def input_template_download_payload() -> tuple[bytes, str] | None:
    """Return ``(xlsx_bytes, file_name)`` for the data-entry template, or None."""
    from paths import cross_section_input_template

    template_path = cross_section_input_template()
    if template_path.exists():
        return template_path.read_bytes(), template_path.name
    try:
        import workbook_template as wt
    except ImportError:
        return None
    builder = getattr(wt, "build_input_template_bytes", None)
    if not callable(builder):
        return None
    try:
        return builder(), _TEMPLATE_FALLBACK_NAME
    except Exception:
        logger.exception("Failed to build input template bytes")
        return None


def render_input_template_download(*, key: str, help: str | None = None) -> None:
    """Offer a template download button; fall back to caption if unavailable."""
    payload = input_template_download_payload()
    if payload is None:
        st.caption("Input template not found — run `python scripts/build_input_template.py`.")
        return
    data, file_name = payload
    st.download_button(
        "Download template (data entry)",
        data=data,
        file_name=file_name,
        mime=_TEMPLATE_MIME,
        key=key,
        help=help
        or (
            "Fill Collars and Lithology in Excel, then upload via the sidebar. "
            "Includes Instructions, Project, optional Water/Screens/Gradients/Environmental."
        ),
    )


def render_workbook_recovery(*, key_prefix: str = "recovery") -> None:
    """Clear / retry controls when workbook bytes exist but parse_result is missing."""
    cols = st.columns([1, 3])
    with cols[0]:
        if st.button("Clear workbook", key=f"{key_prefix}_clear_workbook"):
            clear_workbook_session()
            st.rerun()
    with cols[1]:
        st.caption(
            "Parse failed or did not complete — leftover section output was cleared. "
            "Clear the workbook, fix the file, or try the sample project again."
        )


def load_sample_workbook() -> None:
    """Load the bundled sample workbook into session for demo use.

    Sets ``file_bytes`` and clears detection/parse so the next
    ``handle_workbook_upload`` pass re-detects and parses (same path as a
    fresh file upload). Without clearing detection, sample load would skip
    parse because bytes already match session state.
    """
    sample_path = sample_boreholes_workbook()
    if not sample_path.exists():
        raise FileNotFoundError(
            f"Sample workbook not found at {sample_path}. "
            "Run: python scripts/generate_sample_data.py"
        )
    data = sample_path.read_bytes()
    st.session_state.file_bytes = data
    st.session_state.uploaded_name = sample_path.name
    st.session_state.file_hash = hashlib.sha256(data).hexdigest()[:24]
    st.session_state.parse_result = None
    st.session_state.parse_signature = None
    st.session_state.detection_result = None
    st.session_state.import_report = None
    st.session_state.quality_report = None
    clear_ai_session_state()
    clear_section_output_state()


def render_welcome_card() -> None:
    st.markdown(
        """
<div class="welcome-card">
  <h3>Get started in four steps</h3>
  <p><strong>Enter data in Excel</strong> (download the template), then <strong>upload</strong> the workbook
  in the sidebar. Or try the sample project to skip prep.</p>
  <ol class="welcome-steps">
    <li><strong>Enter</strong> — Download the multi-tab template and fill <em>Collars</em> + <em>Lithology</em> (optional Water, Screens, …).</li>
    <li><strong>Upload</strong> — Use <em>Upload Excel workbook</em> in the sidebar Data source section.</li>
    <li><strong>Validate &amp; Configure</strong> — Review Data Health, then pick transect holes and style.</li>
    <li><strong>Generate</strong> — SVG is ready immediately; Prepare PNG/PDF for deliverables.</li>
  </ol>
</div>
""",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    with cols[0]:
        if st.button("Try sample project", type="primary", key="try_sample_project"):
            try:
                load_sample_workbook()
                st.rerun()
            except FileNotFoundError as exc:
                st.error(str(exc))
    with cols[1]:
        render_input_template_download(key="download_input_template")
    st.caption(
        "Already have an .xlsx? Skip the template — open **Upload Excel workbook** in the sidebar."
    )


_PENDING_PROJECT_SEED_KEY = "_pending_project_seed"


def _seed_consulting_fields_from_project_metadata(project: dict[str, str]) -> None:
    """Queue Project metadata for sidebar widgets (applied before widgets on next run).

    Streamlit forbids writing widget keys after the widget is instantiated; sidebar
    runs before upload parse, so we stash values and apply them at sidebar start.
    """
    if not project:
        return
    pending: dict[str, str] = {}
    mapping = {
        "client_name": "consulting_prepared_for",
        "prepared_by": "consulting_prepared_by",
        "project_number": "consulting_project_number",
        "section_title": "consulting_section_label",
        "report_date": "consulting_date",
        "drawn_by": "consulting_drawn_by",
        "data_source": "consulting_source",
        "map_scale": "consulting_map_scale",
        "notes": "consulting_notes",
    }
    for source_key, session_key in mapping.items():
        value = str(project.get(source_key, "")).strip()
        if value:
            pending[session_key] = value
    section_title = str(project.get("section_title", "")).strip()
    if section_title:
        pending["section_title"] = section_title
        pending.setdefault("consulting_section_label", section_title)
    start = str(project.get("transect_start", "")).strip()
    if start:
        parts = [part.strip() for part in start.split("/", 1)]
        pending["consulting_start_label"] = start
        pending["consulting_start_primary"] = parts[0]
        if len(parts) > 1:
            pending["consulting_start_secondary"] = parts[1]
    end = str(project.get("transect_end", "")).strip()
    if end:
        parts = [part.strip() for part in end.split("/", 1)]
        pending["consulting_end_label"] = end
        pending["consulting_end_primary"] = parts[0]
        if len(parts) > 1:
            pending["consulting_end_secondary"] = parts[1]
    ve = str(project.get("vertical_exaggeration", "")).strip()
    if ve:
        pending["_pending_vertical_exaggeration"] = ve
    if pending:
        st.session_state[_PENDING_PROJECT_SEED_KEY] = pending


def apply_pending_project_seed() -> None:
    """Apply queued Project metadata before sidebar widgets are created."""
    pending = st.session_state.pop(_PENDING_PROJECT_SEED_KEY, None)
    if not isinstance(pending, dict):
        return
    ve_raw = pending.pop("_pending_vertical_exaggeration", None)
    for key, value in pending.items():
        st.session_state[key] = value
    if ve_raw is not None:
        try:
            st.session_state.vertical_exaggeration = float(ve_raw)
        except ValueError:
            pass


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
    bytes_changed = st.session_state.file_bytes != file_bytes
    if bytes_changed:
        st.session_state.file_bytes = file_bytes
        st.session_state.file_hash = hashlib.sha256(file_bytes).hexdigest()[:24]
        st.session_state.parse_result = None
        st.session_state.quality_report = None
        st.session_state.transect_candidates = None
        st.session_state.import_report = None
        clear_ai_session_state()
        clear_section_output_state()
        st.session_state.polygon_overlap_warnings = []
        st.session_state.section_lithology_codes = None
        st.session_state.section_polygon_count = None
        st.session_state.section_hole_count = None
        st.session_state.lithology_index = None
        st.session_state.parse_signature = None
        st.session_state.transect_selection_key = None
        st.session_state.transect_selection = None
        st.session_state.detection_result = None

    # Re-detect when bytes change OR sample/menubar load cleared detection while
    # leaving file_bytes already set (otherwise parse never runs).
    if bytes_changed or st.session_state.get("detection_result") is None:
        try:
            st.session_state.detection_result = FormatDetector().detect(BytesIO(file_bytes))
            st.session_state.pop("upload_banner_error", None)
            st.session_state.pop("upload_banner_success", None)
            st.session_state.pop("upload_banner_info", None)
            st.session_state.pop("upload_banner_caption", None)
        except Exception as exc:
            st.session_state.detection_result = None
            clear_section_output_state()
            st.session_state.upload_banner_error = f"Failed to inspect workbook: {exc}"
            st.session_state.pop("upload_banner_success", None)
            st.session_state.pop("upload_banner_caption", None)

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
    should_parse = (
        detection is not None
        and st.session_state.parse_signature != parse_signature
    )
    if should_parse:
        try:
            with st.spinner("Reading workbook…"):
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
            clear_ai_session_state()
            clear_section_output_state()
            st.session_state.transect_selection_key = None
            st.session_state.transect_selection = None
            st.session_state.polygon_overlap_warnings = []
            st.session_state.render_cache_key = None
            st.session_state.section_lithology_codes = None
            st.session_state.section_polygon_count = None
            st.session_state.section_hole_count = None
            st.session_state.parse_signature = parse_signature
            project_metadata = getattr(import_report, "project_metadata", {}) or {}
            info_parts: list[str] = []
            if project_metadata:
                _seed_consulting_fields_from_project_metadata(project_metadata)
                info_parts.append(
                    "Seeded consulting report fields from Project metadata."
                )
            st.session_state.upload_banner_success = (
                f"Loaded **{len(hole_ids)}** boreholes and "
                f"**{len(parse_result.lithologies)}** lithology intervals."
            )
            st.session_state.upload_banner_info = " ".join(info_parts) if info_parts else None
            st.session_state.upload_banner_caption = (
                f"Suggested transect offset threshold: **{st.session_state.suggested_offset_m:.0f} m** "
                "(applied to sidebar warnings)."
            )
            st.session_state.pop("upload_banner_error", None)
            audit_event(
                "workbook_parsed",
                workbook=str(st.session_state.get("uploaded_name") or "upload"),
                hole_count=len(hole_ids),
                lithology_count=len(parse_result.lithologies),
            )
            st.rerun()
        except Exception as exc:
            logger.exception("Workbook parse failed")
            st.session_state.parse_result = None
            st.session_state.import_report = None
            st.session_state.quality_report = None
            st.session_state.parse_signature = None
            clear_section_output_state()
            st.session_state.upload_banner_error = f"Failed to parse workbook: {exc}"
            st.session_state.pop("upload_banner_success", None)
            st.session_state.pop("upload_banner_info", None)
            st.session_state.pop("upload_banner_caption", None)
            return None

    error_banner = st.session_state.pop("upload_banner_error", None)
    if error_banner:
        st.error(error_banner)
    success_banner = st.session_state.pop("upload_banner_success", None)
    if success_banner:
        st.success(success_banner)
    info_banner = st.session_state.pop("upload_banner_info", None)
    if info_banner:
        st.info(info_banner)
    caption_banner = st.session_state.pop("upload_banner_caption", None)
    if caption_banner:
        st.caption(caption_banner)

    return st.session_state.parse_result
