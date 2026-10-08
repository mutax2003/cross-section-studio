"""Upload welcome card and workbook parse flow."""

from __future__ import annotations

import hashlib
import logging
import zipfile
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


def _forget_previous_transect() -> None:
    """A new workbook must not inherit the last one's hole order or sheet label."""
    st.session_state.pop("hole_sequence_multiselect", None)
    st.session_state.pop("workbook_section_choice", None)
    st.session_state.pop("_workbook_section_applied", None)
    st.session_state["_reset_consulting_section_label"] = True
    # Project-seeded title block fields belong to the workbook being dropped.
    st.session_state["_reset_project_seed"] = True


def clear_workbook_session() -> None:
    """Clear parse/session workbook state and any leftover section SVG/PNG/PDF."""
    _forget_previous_transect()
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
    st.session_state["workbook_uploader_key"] = (
        st.session_state.get("workbook_uploader_key", 0) + 1
    )


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
        st.caption("The data-entry template isn't included in this install — use **Try sample project** instead.")
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


def _friendly_workbook_error(exc: Exception) -> str:
    """Plain-language upload/parse failure with a next step (raw text goes in a details expander)."""
    text = str(exc)
    lowered = text.lower()
    if isinstance(exc, zipfile.BadZipFile) or "format cannot be determined" in lowered or "not a zip file" in lowered:
        return (
            "This file isn't a readable Excel workbook (.xlsx). It may be a renamed CSV or .xls "
            "file, or damaged. Open it in Excel, choose **Save As → Excel Workbook (.xlsx)**, "
            "then upload it again."
        )
    if "Could not detect a supported workbook format" in text:
        found = text.split("Sheets found:", 1)[-1].strip() if "Sheets found:" in text else ""
        sheets = f" (sheets in this file: {found})" if found else ""
        found_names = {name.strip().strip("[]'\" ").casefold() for name in found.split(",")}
        if "collars" in found_names:
            # Collars is there, so the missing piece is the lithology log.
            return (
                f"No **Lithology** sheet was found{sheets}. Add a Lithology sheet with hole_id, "
                "from_depth, to_depth and lithology_code, or start from **Download template**. "
                "See Help → Workbook and data entry."
            )
        return (
            f"No **Collars** sheet was found{sheets}. Add a Collars sheet with hole_id, easting, "
            "northing, elevation and total_depth, or start from **Download template**. "
            "See Help → Workbook and data entry."
        )
    if "sheet is missing the" in text:
        # Already plain language from the parser: names the sheet and the column.
        return text
    if "Missing required sheet" in text:
        missing = {
            name.strip().casefold() for name in text.split(":", 1)[-1].split(",") if name.strip()
        }
        if missing == {"lithology"}:
            return (
                "No **Lithology** sheet was found. Add a Lithology sheet with hole_id, "
                "from_depth, to_depth and lithology_code, or start from **Download template**. "
                "See Help → Workbook and data entry."
            )
        if missing == {"collars"}:
            return (
                "No **Collars** sheet was found. Add a Collars sheet with hole_id, easting, "
                "northing, elevation and total_depth, or start from **Download template**. "
                "See Help → Workbook and data entry."
            )
        return f"{text}. Add the missing sheet(s), or start from **Download template**."
    if "rows (limit" in text:
        return f"The workbook is too large to read: {text}"
    if "xlrd" in lowered or "ole2" in lowered or "encrypted" in lowered:
        return (
            "This looks like an old .xls or a password-protected workbook. Open it in Excel, "
            "remove the password if any, and **Save As → Excel Workbook (.xlsx)**."
        )
    if "io.excel." in lowered:
        return (
            "This file isn't an Excel workbook (it may be a ZIP or another format renamed to "
            ".xlsx). Save it from Excel as an .xlsx workbook and upload again."
        )
    return (
        "The workbook couldn't be read. Check that Collars and Lithology use the template "
        "headers, then upload it again."
    )


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def upload_headline(
    *,
    hole_count: int,
    interval_count: int,
    skipped_count: int,
    error_count: int,
) -> tuple[str, str]:
    """``("success" | "warning", text)`` for the banner shown after a workbook loads.

    Green only when no row was dropped and no data check failed; otherwise the
    headline says what went wrong and points to Validate, where the row list lives.
    """
    # Same total as the Data health tile, which counts skipped rows as errors.
    total = skipped_count + error_count
    if total:
        detail = f" ({_plural(skipped_count, 'row')} skipped)" if skipped_count else ""
        return (
            "warning",
            f"**Loaded with problems: {_plural(total, 'data error')}{detail} — see Validate.** "
            f"({_plural(hole_count, 'borehole')} and "
            f"{_plural(interval_count, 'lithology interval')} loaded.)",
        )
    return (
        "success",
        f"Loaded **{hole_count}** boreholes and **{interval_count}** lithology intervals.",
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
            "The workbook couldn't be loaded. Fix the file in Excel and upload it again, "
            "clear it, or try the sample project."
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
        logger.warning(
            "Sample workbook not found at %s (run scripts/generate_sample_data.py)", sample_path
        )
        raise FileNotFoundError(
            "The sample project isn't included in this install. Upload your own workbook, "
            "or download the template to start one."
        )
    data = sample_path.read_bytes()
    # The demo must never open on a blocked Generate: the sample has pinch-out
    # overlaps, which the consulting preset blocks by default.
    queue_session_values(fail_on_overlaps_checkbox=False)
    st.session_state.file_bytes = data
    st.session_state.uploaded_name = sample_path.name
    st.session_state.file_hash = hashlib.sha256(data).hexdigest()[:24]
    st.session_state.batch_transect_specs = ""
    st.session_state._batch_specs_seeded_from_sections = False
    st.session_state.pop("workbook_section_choice", None)
    st.session_state.pop("_workbook_section_applied", None)
    _forget_previous_transect()
    st.session_state.parse_result = None
    st.session_state.parse_signature = None
    st.session_state.detection_result = None
    st.session_state.import_report = None
    st.session_state.quality_report = None
    clear_ai_session_state()
    clear_section_output_state()
    st.session_state["workbook_uploader_key"] = (
        st.session_state.get("workbook_uploader_key", 0) + 1
    )



DESTRUCTIVE_PROMPTS = {
    "clear": ("Clear workbook", "Clear the workbook and discard the generated section?"),
    "sample": ("Load sample", "Load the sample project and discard the generated section?"),
}


def run_destructive(action: str) -> None:
    if action == "clear":
        clear_workbook_session()
        st.rerun()
    try:
        load_sample_workbook()
        st.rerun()
    except FileNotFoundError as exc:
        st.error(str(exc))


def request_destructive(action: str) -> None:
    """Act immediately unless it would discard a generated section; then confirm.

    Shared by the sidebar, File menu and Alt+Shift+O so no entry point can
    discard a section silently; the sidebar renders the pending prompt.
    """
    if st.session_state.get("svg_bytes") is None:
        run_destructive(action)
        return
    st.session_state["_pending_destructive"] = action


def render_welcome_card() -> None:
    st.markdown(
        """
<div class="welcome-card">
  <h3>Get started in four steps</h3>
  <p><strong>Enter data in Excel</strong> (download the template), then <strong>upload</strong> the workbook
  in the sidebar. Or try the sample project to skip prep.</p>
  <ol class="welcome-steps">
    <li><strong>Enter</strong> — Download the multi-tab template and fill <em>Collars</em> + <em>Lithology</em> (optional Water, Screens, …).</li>
    <li><strong>Upload</strong> — Use <em>Upload Excel workbook</em> in the sidebar Data section.</li>
    <li><strong>Validate &amp; Configure</strong> — Review data health, then pick the holes on the section line and the style.</li>
    <li><strong>Generate</strong> — SVG is ready immediately; Prepare deliverables for PNG/PDF/Word/package.</li>
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
    figure_raw = str(
        project.get("figure_preset") or project.get("section_style") or ""
    ).strip()
    if figure_raw:
        from ui_output_presets import normalize_figure_preset

        resolved = normalize_figure_preset(figure_raw)
        if resolved:
            pending["output_preset"] = resolved
    if pending:
        st.session_state[_PENDING_PROJECT_SEED_KEY] = pending


# Widget keys the Project tab (or the Section picker) seeds; cleared together
# when the workbook goes away so nothing leaks onto the next one.
PROJECT_SEEDED_KEYS: tuple[str, ...] = (
    "consulting_prepared_for",
    "consulting_prepared_by",
    "consulting_project_number",
    "consulting_section_label",
    "consulting_date",
    "consulting_drawn_by",
    "consulting_source",
    "consulting_map_scale",
    "consulting_notes",
    "consulting_start_label",
    "consulting_start_primary",
    "consulting_start_secondary",
    "consulting_end_label",
    "consulting_end_primary",
    "consulting_end_secondary",
    "section_title",
)


def queue_session_values(**values: object) -> None:
    """Set widget-backed session values on the NEXT run, before widgets exist.

    Writing a widget key after its widget was drawn raises; the sidebar
    applies this queue first thing each run.
    """
    pending = st.session_state.get(_PENDING_PROJECT_SEED_KEY)
    if not isinstance(pending, dict):
        pending = {}
    pending.update(values)
    st.session_state[_PENDING_PROJECT_SEED_KEY] = pending


def apply_pending_project_seed() -> None:
    """Apply queued Project metadata before sidebar widgets are created."""
    if st.session_state.pop("_reset_project_seed", False):
        # The sidebar keeps a non-widget copy of these fields (so they survive
        # output-style switches); the new workbook must not inherit it.
        store = st.session_state.get("_sidebar_widget_store")
        for key in PROJECT_SEEDED_KEYS:
            st.session_state.pop(key, None)
            if isinstance(store, dict):
                store.pop(key, None)
    pending = st.session_state.pop(_PENDING_PROJECT_SEED_KEY, None)
    if not isinstance(pending, dict):
        return
    ve_raw = pending.pop("_pending_vertical_exaggeration", None)
    if "output_preset" in pending:
        st.session_state.pop("_synced_output_preset", None)
    for key, value in pending.items():
        st.session_state[key] = value
    if ve_raw is not None:
        from ui_output_presets import normalize_ve_choice

        # "5" -> exact 5x; "auto" (or anything unreadable) -> Auto (fit page).
        st.session_state.vertical_exaggeration = normalize_ve_choice(ve_raw)


def handle_workbook_upload(
    uploaded,
    *,
    selected_profile_key: str,
    override_id: str | None,
    default_elevation_m: float | None,
    target_crs: str | None,
) -> ParseResult | None:
    """Detect format, parse workbook when needed, return current parse result."""
    file_bytes = uploaded.getvalue()
    bytes_changed = st.session_state.file_bytes != file_bytes
    if bytes_changed:
        st.session_state.file_bytes = file_bytes
        st.session_state.file_hash = hashlib.sha256(file_bytes).hexdigest()[:24]
        st.session_state.batch_transect_specs = ""
        st.session_state._batch_specs_seeded_from_sections = False
        st.session_state.pop("workbook_section_choice", None)
        st.session_state.pop("_workbook_section_applied", None)
        _forget_previous_transect()
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
            st.session_state.pop("upload_banner_problem", None)
            st.session_state.pop("upload_banner_info", None)
            st.session_state.pop("upload_banner_caption", None)
        except Exception as exc:
            st.session_state.detection_result = None
            clear_section_output_state()
            st.session_state.upload_banner_error = _friendly_workbook_error(exc)
            st.session_state.upload_banner_error_detail = str(exc)
            st.session_state.pop("upload_banner_success", None)
            st.session_state.pop("upload_banner_problem", None)
            st.session_state.pop("upload_banner_caption", None)

    detection = st.session_state.detection_result
    if detection is not None:
        if st.session_state.get("svg_bytes") is None:
            st.caption(
                f"Detected format: **{detection.label}** "
                f"({detection.confidence:.0%} confidence)"
            )
        # Field exports only: the native and Data Entry templates have no
        # Field Data sheet, so the note only confused them.
        if detection.profile_id != NATIVE_PROFILE_ID and not detection.is_native:
            st.info(
                "Field Data sheet (if present) is not used for stratigraphy. "
                "OVA/EC columns map to environmental readings — select them on Configure."
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
                from workbook_template import _sample_project

                sample = _sample_project()
                stale = [
                    key
                    for key in ("client_name", "project_number", "report_date", "section_title")
                    if str(project_metadata.get(key, "")).strip() == sample.get(key, "")
                ]
                # A real project can share a number format or date with the
                # sample; only the client name or section title identify a
                # Project tab that was never filled in.
                if stale and {"client_name", "section_title"} & set(stale):
                    quoted = ", ".join(f"{key} = '{sample[key]}'" for key in stale)
                    st.session_state.upload_banner_caution = (
                        "The Project tab still holds the template's sample values "
                        f"({quoted}); they will print on the title block — update them "
                        "before issuing figures."
                    )
            level, headline = upload_headline(
                hole_count=len(hole_ids),
                interval_count=len(parse_result.lithologies),
                skipped_count=len(parse_result.errors),
                error_count=int(getattr(quality_report, "error_count", 0) or 0),
            )
            if level == "success":
                st.session_state.upload_banner_success = headline
            else:
                st.session_state.upload_banner_problem = headline
            st.session_state.upload_banner_info = " ".join(info_parts) if info_parts else None
            st.session_state.upload_banner_caption = (
                f"Suggested section line offset limit: **{st.session_state.suggested_offset_m:.0f} m** "
                "(holes farther than this are flagged)."
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
            st.session_state.upload_banner_error = _friendly_workbook_error(exc)
            st.session_state.upload_banner_error_detail = str(exc)
            st.session_state.pop("upload_banner_success", None)
            st.session_state.pop("upload_banner_problem", None)
            st.session_state.pop("upload_banner_info", None)
            st.session_state.pop("upload_banner_caption", None)
            # Fall through: the banner block below is what shows the error.

    error_banner = st.session_state.pop("upload_banner_error", None)
    if error_banner:
        st.error(error_banner)
        detail = st.session_state.pop("upload_banner_error_detail", None)
        if detail:
            with st.expander("Technical details"):
                st.code(detail, language=None)
    success_banner = st.session_state.pop("upload_banner_success", None)
    if success_banner:
        st.success(success_banner)
    problem_banner = st.session_state.pop("upload_banner_problem", None)
    if problem_banner:
        # The row-by-row list is shown once, on Validate — not repeated here.
        st.warning(problem_banner)
    caution = st.session_state.pop("upload_banner_caution", None)
    if caution:
        st.warning(caution)
    info_banner = st.session_state.pop("upload_banner_info", None)
    if info_banner:
        st.info(info_banner)
    caption_banner = st.session_state.pop("upload_banner_caption", None)
    if caption_banner:
        st.caption(caption_banner)

    return st.session_state.parse_result
