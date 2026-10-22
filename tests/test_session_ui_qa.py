"""Regressions from browser QA of session handling and deliverable metadata.

1. Sign out left the workbook loaded: signing back in showed "Loaded: …",
   the holes and plan view, and Generate rebuilt the previous section.
2. Switching workbooks (chemistry → none → chemistry) left "Parameters to
   plot" empty, so the PDF and batch ZIP silently had no lab values.
3. Metadata recorded auto VE as blank and could omit the label colour mode.
4. The "Next: choose holes…" hint stayed after a section was generated.
5. Lab values left off the figure were only an info-styled note.
6. File names carried apostrophes; batch sheets dropped the title text.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_chemistry_per_parameter_thresholds import _two_parameter_workbook_bytes

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample_boreholes.xlsx"
PASSWORD = "qa-pass"


@pytest.fixture(scope="module")
def sample_bytes() -> bytes:
    if not SAMPLE.exists():
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_sample_data.py")],
            cwd=ROOT,
            check=True,
        )
    return SAMPLE.read_bytes()


def _app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)


def _sign_in(at) -> None:
    at.text_input[0].set_value(PASSWORD)
    [b for b in at.button if b.label == "Sign in"][0].click().run()
    assert not at.exception
    assert at.session_state["_auth_ok"] is True


def _generate(at) -> None:
    for box in at.checkbox:
        if "Generate even if data checks found warnings" in (box.label or ""):
            box.set_value(True)
    at.run()
    [b for b in at.button if b.label == "Generate section"][0].click().run()
    assert not at.exception
    assert at.session_state["svg_bytes"]


def _captions(at) -> str:
    return "\n".join(str(c.value) for c in at.caption)


# --- 1. Sign out ---------------------------------------------------------------------


def test_sign_out_wipes_workbook_figure_and_typed_fields(monkeypatch, sample_bytes) -> None:
    import ops_auth

    monkeypatch.setenv("CROSS_SECTION_AUTH_PASSWORD", PASSWORD)
    store = ops_auth._auth_throttle_store()
    store["failures"].clear()
    store["lock_until"].clear()
    at = _app()
    at.run()
    _sign_in(at)
    at.file_uploader[0].upload("site.xlsx", sample_bytes).run()
    assert at.session_state["parse_result"] is not None
    at.text_input(key="section_title").set_value("CONFIDENTIAL SITE").run()
    _generate(at)
    assert "Loaded: site.xlsx" in _captions(at)
    uploader_key = int(at.session_state["workbook_uploader_key"])
    logo_key = int(at.session_state["logo_uploader_key"])

    [b for b in at.button if b.label == "Sign out"][0].click().run()
    assert not at.exception
    assert at.session_state["_auth_ok"] is False
    assert at.text_input[0].label == "Password"

    _sign_in(at)
    state = at.session_state
    # Workbook, parse, figure and every prepared deliverable are gone.
    for key in ("file_bytes", "parse_result", "svg_bytes", "png_bytes", "pdf_bytes", "uploaded_name"):
        assert key not in state or state[key] is None, key
    assert list(state["hole_ids"]) == []
    # Upload widgets get fresh keys, so the browser's copy is not re-read.
    assert int(state["workbook_uploader_key"]) == uploader_key + 1
    assert int(state["logo_uploader_key"]) == logo_key + 1
    # No "Loaded:", no Generate, nothing typed survives.
    assert "Loaded:" not in _captions(at)
    assert not [b for b in at.button if b.label == "Generate section"]
    assert at.text_input(key="section_title").value != "CONFIDENTIAL SITE"
    assert "hole_sequence_multiselect" not in state
    # A fresh upload works after the wipe.
    at.file_uploader[0].upload("other.xlsx", sample_bytes).run()
    assert not at.exception
    assert at.session_state["parse_result"] is not None


def test_logout_wipe_keeps_only_the_allowlist_and_bumps_upload_keys() -> None:
    from ops_auth import _clear_sensitive_session_on_logout

    session = {
        "_auth_ok": False,
        "keyboard_shortcuts_enabled": False,
        "workbook_uploader_key": 3,
        "logo_uploader_key": 1,
        "file_bytes": b"xlsx",
        "consulting_project_number": "P-123",
        "consulting_logo_for": object(),
        "environmental_parameters_multiselect": ["Chloride"],
        "water_series_multiselect": ["default"],
        "_widget_store": {"consulting_prepared_for": "Client"},
        "_llm_api_key_runtime_groq": "gsk_live",
        "any_future_key": "data",
    }
    _clear_sensitive_session_on_logout(session)
    assert session == {
        "_auth_ok": False,
        "keyboard_shortcuts_enabled": False,
        "workbook_uploader_key": 4,
        "logo_uploader_key": 2,
    }


def test_drop_session_memos_closes_this_sessions_retained_figures() -> None:
    import app_services

    app_services.clear_service_memos()
    closed: list[str] = []

    class _Figure:
        def __init__(self, name: str) -> None:
            self.name = name

    import matplotlib.pyplot as plt

    original_close = plt.close
    try:
        plt.close = lambda figure: closed.append(figure.name)  # type: ignore[assignment]
        app_services._figure_memo[("me", "s", "r")] = {"figure": _Figure("mine")}
        app_services._figure_memo[("other", "s", "r")] = {"figure": _Figure("theirs")}
        app_services._export_memo[("me", "s", "r", frozenset({"png"}))] = (b"",) * 3 + (0, (), ())
        app_services._geometry_memo[("me", "s", "r")] = object()  # type: ignore[assignment]
        removed = app_services.drop_session_memos("me")
    finally:
        plt.close = original_close
    assert removed == 3
    assert closed == ["mine"]
    assert list(app_services._figure_memo) == [("other", "s", "r")]
    assert not app_services._export_memo and not app_services._geometry_memo
    app_services.clear_service_memos()


# --- 2. Chemistry selection across workbooks -----------------------------------------


def _chem_params(at) -> list[str]:
    return list(at.multiselect(key="environmental_parameters_multiselect").value)


def test_switching_workbooks_reseeds_the_lab_parameters(sample_bytes) -> None:
    chem = _two_parameter_workbook_bytes()
    at = _app()
    at.run()
    at.file_uploader[0].upload("chem.xlsx", chem).run()
    assert not at.exception
    first = _chem_params(at)
    assert first, "a workbook with lab data must start with parameters selected"

    # Narrow the pick, then switch to a workbook without lab data and back.
    at.multiselect(key="environmental_parameters_multiselect").set_value(["Benzene"]).run()
    at.file_uploader[0].upload("plain.xlsx", sample_bytes).run()
    assert not at.exception
    at.file_uploader[0].upload("chem.xlsx", chem).run()
    assert not at.exception
    # A new workbook seeds its own default, never an empty / stale pick.
    assert _chem_params(at) == first
    assert at.session_state["environmental_parameters_multiselect"] == first


def test_deselecting_every_parameter_warns_with_the_count() -> None:
    at = _app()
    at.run()
    at.file_uploader[0].upload("chem.xlsx", _two_parameter_workbook_bytes()).run()
    at.multiselect(key="environmental_parameters_multiselect").set_value([]).run()
    assert not at.exception
    assert any("No parameters selected" in w.value and "6 lab value" in w.value for w in at.warning)


# --- 3. Metadata -----------------------------------------------------------------------


def test_metadata_records_auto_ve_as_drawn_and_always_names_the_colour_mode() -> None:
    from ui_helpers import export_metadata_payload

    auto = export_metadata_payload(
        section_title="A-A'", preset_label="Chemistry + groundwater (elevation)",
        vertical_exaggeration=None, hole_count=3, transect_label=None, overlap_warnings=(),
        chemistry_color_mode=None, drawn_vertical_exaggeration="≈2.6×",
    )
    assert auto["vertical_exaggeration"] == "Auto (≈2.6×)"
    assert auto["vertical_exaggeration_drawn"] == "≈2.6×"
    assert auto["chemistry_label_colour_mode"] == "black"
    unknown = export_metadata_payload(
        section_title="S", preset_label=None, vertical_exaggeration=None, hole_count=1,
        transect_label=None, overlap_warnings=(),
    )
    assert unknown["vertical_exaggeration"] == "Auto (fit to page)"
    fixed = export_metadata_payload(
        section_title="S", preset_label=None, vertical_exaggeration=5.0, hole_count=1,
        transect_label=None, overlap_warnings=(), chemistry_color_mode="red",
        drawn_vertical_exaggeration="5×",
    )
    assert fixed["vertical_exaggeration"] == 5.0
    assert fixed["chemistry_label_colour_mode"] == "red"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("APPROX. SCALE 1:649 VERTICAL EXAGGERATION ≈2×", "≈2×"),
        ("APPROX. SCALE 1:1230 5× VERTICAL EXAGGERATION", "5×"),
        ("CRS: EPSG:32611 | Datum: Collar RL | VE: ≈2.6× | Holes: 4", "≈2.6×"),
        ("Distance V.E. ≈2.2×", "≈2.2×"),
        ("APPROX. SCALE 1:681 NO VERTICAL EXAGGERATION", "1×"),
        ("No V.E. (1×)", "1×"),
        ("VERTICAL EXAGGERATION: AUTO", None),
        ("", None),
    ],
)
def test_ve_wording_matches_every_caption_style(text: str, expected: str | None) -> None:
    from ui_helpers import ve_wording_from_text

    assert ve_wording_from_text(text) == expected


def test_drawn_ve_is_read_from_a_real_auto_ve_figure() -> None:
    from models import Collar, Lithology
    from pipeline import build_cross_section
    from ui_helpers import drawn_vertical_exaggeration

    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=60.0, northing=0.0, elevation=99.0, total_depth=10.0),
    ]
    lithologies = [
        Lithology(hole_id=h, from_depth=0.0, to_depth=10.0, lithology_code="Clay")
        for h in ("BH-01", "BH-02")
    ]
    result = build_cross_section(
        collars, lithologies, [(0.0, 0.0), (60.0, 0.0)],
        vertical_exaggeration=None, show_ve_annotation=True,
        export_formats=frozenset({"svg", "pdf"}),
    )
    from_pdf = drawn_vertical_exaggeration(pdf_bytes=result.pdf_bytes)
    from_svg = drawn_vertical_exaggeration(svg_bytes=result.svg_bytes)
    assert from_svg is not None and from_svg.endswith("×")
    assert from_pdf in (None, from_svg) or from_pdf.endswith("×")


# --- 4. Next-step hint -------------------------------------------------------------


def test_next_step_hint_is_gone_once_a_section_is_generated(sample_bytes) -> None:
    at = _app()
    at.run()
    at.file_uploader[0].upload("site.xlsx", sample_bytes).run()
    hole_ids = list(at.session_state["hole_ids"])
    at.session_state["hole_sequence_multiselect"] = hole_ids[:4]
    at.run()
    coach = '<div class="next-step-coach"'
    assert any(coach in str(m.value) for m in at.markdown)
    _generate(at)
    assert not any(coach in str(m.value) for m in at.markdown)
    at.run()
    assert not any(coach in str(m.value) for m in at.markdown)


# --- 5. Lab values left off ----------------------------------------------------------


def test_lab_values_left_off_get_a_counted_warning_headline() -> None:
    from app_common import lab_values_left_off_markdown

    notes = [
        "Chemistry labels: 33 value(s) not printed - no room beside their borehole at this "
        "scale (BH23-01: <5, <5). Markers are still drawn; a wider page or fewer boreholes "
        "per section shows them.",
        "Map scale 1:1000 doesn't fit a letter landscape page.",
    ]
    text = lab_values_left_off_markdown(notes)
    assert text.startswith("**33 lab values were left off the figure**")
    assert "Page size" in text and "fewer boreholes" in text
    assert lab_values_left_off_markdown(notes[1:]) == ""


def test_off_line_lab_values_are_counted_with_the_fix() -> None:
    from app_configure import off_line_lab_values_message
    from models import EnvironmentalReading, ParseResult

    readings = [
        EnvironmentalReading(hole_id=h, parameter="Chloride", value=v, depth=1.0)
        for h, v in (("BH-01", 10.0), ("BH-09", 20.0), ("BH-09", 30.0), ("BH-10", 5.0))
    ]
    result = ParseResult(collars=[], lithologies=[], errors=[], environmental_readings=readings)
    text = off_line_lab_values_message(result, ["BH-01", "BH-02"], offset_warning_m=50.0)
    assert text.startswith("**3 lab values will be left off**")
    assert "2 hole(s) more than 50 m" in text and "BH-09, BH-10" in text
    assert "Warn when a hole is this far off the line" in text
    assert off_line_lab_values_message(result, ["BH-01", "BH-09", "BH-10"], offset_warning_m=50.0) == ""


# --- 6. File names and batch titles ----------------------------------------------------


def test_every_export_name_is_quote_free() -> None:
    from export_framing import build_export_filename
    from ui_helpers import sanitize_filename

    assert build_export_filename(
        pattern="section_title",
        section_title="B - B' WITH CHLORIDE INTERVALS",
        transect_label="B-B' BH24-12→BH23-01",
        include_transect_label=True,
    ) == "B-B_WITH_CHLORIDE_INTERVALS_BH24-12_BH23-01"
    assert sanitize_filename("Cross Sections A-A', B-B′ WITH CL") == "Cross_Sections_A-A_B-B_WITH_CL"


def test_batch_sheet_for_the_base_line_matches_the_single_section_sheet() -> None:
    from batch_export import BatchTransectSpec, prepare_batch_section_request
    from models import Collar, ConsultingTitleBlock, Lithology, ParseResult
    from section_build_request import SectionBuildRequest

    title = "A - A' WITH CHLORIDE AVERAGES"
    collars = [
        Collar(hole_id=h, easting=float(i) * 40.0, northing=0.0, elevation=100.0, total_depth=10.0)
        for i, h in enumerate(("MW-01", "MW-02", "MW-03"))
    ]
    lithologies = [
        Lithology(hole_id=c.hole_id, from_depth=0.0, to_depth=10.0, lithology_code="Clay")
        for c in collars
    ]
    parse_result = ParseResult(collars=collars, lithologies=lithologies, errors=[])
    base = SectionBuildRequest(
        transect_points=((0.0, 0.0), (80.0, 0.0)),
        section_title=title,
        consulting_title_block=ConsultingTitleBlock(section_label=title),
    )
    _, same = prepare_batch_section_request(
        parse_result, base, BatchTransectSpec(label="A-A'", hole_ids=("MW-01", "MW-02", "MW-03"))
    )
    assert same.section_title == title
    assert same.consulting_title_block.section_label == title
    _, other = prepare_batch_section_request(
        parse_result, base, BatchTransectSpec(label="B-B'", hole_ids=("MW-01", "MW-03"))
    )
    # The title text after the label is kept on the sheet, not just "B-B'".
    assert other.section_title == "B - B' WITH CHLORIDE AVERAGES"
    assert other.consulting_title_block.section_label == "B - B' WITH CHLORIDE AVERAGES"


def test_logout_allowlist_names_the_real_shortcut_setting() -> None:
    from app_menubar import SHORTCUTS_ENABLED_KEY
    from ops_auth import _LOGOUT_KEEP_KEYS

    assert SHORTCUTS_ENABLED_KEY in _LOGOUT_KEEP_KEYS
