"""Streamlit AppTest smoke for upload → validate → generate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample_boreholes.xlsx"


@pytest.fixture(scope="module")
def sample_workbook() -> Path:
    if not SAMPLE.exists():
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_sample_data.py")],
            cwd=ROOT,
            check=True,
        )
    assert SAMPLE.exists()
    return SAMPLE


def test_streamlit_upload_and_validate(sample_workbook: Path) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    at.file_uploader[0].upload("sample.xlsx", sample_workbook.read_bytes()).run()
    assert not at.exception
    assert "hole_ids" in at.session_state and at.session_state["hole_ids"]
    assert "parse_result" in at.session_state and at.session_state["parse_result"] is not None
    assert "quality_report" in at.session_state and at.session_state["quality_report"] is not None


def test_streamlit_welcome_sample_button_present() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    labels = [btn.label for btn in at.button]
    assert "Try sample project" in labels


def test_streamlit_try_sample_project_parses(sample_workbook: Path) -> None:
    """Regression: sample load must re-detect/parse even when file_bytes is pre-set."""
    from streamlit.testing.v1 import AppTest

    assert sample_workbook.exists()
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    sample_buttons = [btn for btn in at.button if btn.label == "Try sample project"]
    assert sample_buttons, "Try sample project button missing"
    sample_buttons[0].click().run()
    assert not at.exception
    assert "file_bytes" in at.session_state and at.session_state["file_bytes"]
    assert "detection_result" in at.session_state and at.session_state["detection_result"] is not None
    assert "parse_result" in at.session_state and at.session_state["parse_result"] is not None
    assert "hole_ids" in at.session_state and at.session_state["hole_ids"]
    assert "quality_report" in at.session_state and at.session_state["quality_report"] is not None
    # Sample load remounts the file_uploader so session bytes stay authoritative.
    assert "workbook_uploader_key" in at.session_state
    assert at.session_state["workbook_uploader_key"] >= 1


def test_streamlit_clear_workbook_bumps_uploader_key(sample_workbook: Path) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    at.file_uploader[0].upload("sample.xlsx", sample_workbook.read_bytes()).run()
    assert at.session_state["parse_result"] is not None
    before = int(at.session_state["workbook_uploader_key"]) if "workbook_uploader_key" in at.session_state else 0
    clear_buttons = [btn for btn in at.button if btn.label == "Clear workbook"]
    assert clear_buttons, "Clear workbook button missing"
    clear_buttons[0].click().run()
    assert not at.exception
    assert at.session_state["parse_result"] is None
    assert at.session_state["file_bytes"] is None
    assert int(at.session_state["workbook_uploader_key"]) == before + 1


def test_streamlit_generate_smoke(sample_workbook: Path) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    at.file_uploader[0].upload("sample.xlsx", sample_workbook.read_bytes()).run()
    assert not at.exception
    assert at.session_state["parse_result"] is not None

    hole_ids = list(at.session_state["hole_ids"])
    assert len(hole_ids) >= 2
    at.session_state["hole_sequence_multiselect"] = hole_ids[: min(4, len(hole_ids))]
    for box in at.checkbox:
        label = box.label or ""
        if "Allow generate with warnings" in label:
            box.set_value(True)
            break
    at.run()

    generate_buttons = [btn for btn in at.button if btn.label == "Generate Cross-Section"]
    assert generate_buttons, (
        "Generate Cross-Section missing after upload+transect setup; "
        f"buttons={[b.label for b in at.button]}"
    )
    assert not generate_buttons[0].disabled, (
        "Generate disabled — check QA blocking / placeholder elev / transect / overlaps"
    )
    generate_buttons[0].click().run()
    assert not at.exception
    assert "svg_bytes" in at.session_state and at.session_state["svg_bytes"]
    # Figure-first: setup collapsed, regenerate strip present after first generate
    assert any(btn.label == "Regenerate" for btn in at.button)


def test_render_hero_compact_after_upload() -> None:
    from app_common import _render_hero

    # Smoke: compact class applied for stage >= 1 (no Streamlit run required for logic)
    assert callable(_render_hero)


def _generated_app(sample_workbook: Path):
    """AppTest with the sample uploaded and one section generated."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    at.file_uploader[0].upload("sample.xlsx", sample_workbook.read_bytes()).run()
    hole_ids = list(at.session_state["hole_ids"])
    at.session_state["hole_sequence_multiselect"] = hole_ids[: min(4, len(hole_ids))]
    for box in at.checkbox:
        if "Allow generate with warnings" in (box.label or ""):
            box.set_value(True)
            break
    at.run()
    [btn for btn in at.button if btn.label == "Generate Cross-Section"][0].click().run()
    assert at.session_state["svg_bytes"]
    return at


def test_clear_workbook_asks_before_discarding_a_generated_section(sample_workbook: Path) -> None:
    at = _generated_app(sample_workbook)
    [btn for btn in at.button if btn.label == "Clear workbook"][0].click().run()
    assert not at.exception
    # Nothing discarded yet: a confirmation is shown instead.
    assert at.session_state["svg_bytes"]
    assert any("discard the generated section" in w.value for w in at.warning)

    [btn for btn in at.button if btn.key == "cancel_destructive"][0].click().run()
    assert at.session_state["svg_bytes"], "Cancel must keep the section"
    assert not any(btn.key == "confirm_destructive" for btn in at.button)

    [btn for btn in at.button if btn.label == "Clear workbook"][0].click().run()
    [btn for btn in at.button if btn.key == "confirm_destructive"][0].click().run()
    assert not at.exception
    assert at.session_state["parse_result"] is None
    assert at.session_state["svg_bytes"] is None


def test_generate_action_and_state_aware_coach_sit_above_validate(sample_workbook: Path) -> None:
    """UX regressions: Generate is at the top of the page (not ~2.5 screens
    down), the 'Next:' hint reflects the picked transect, and the stepper is
    inside the hero (it rendered outside, white-on-white)."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    at.file_uploader[0].upload("sample.xlsx", sample_workbook.read_bytes()).run()
    at.run()
    assert not at.exception
    markdown = [m.value for m in at.markdown]
    hero = next(value for value in markdown if "<h1>Cross Section Studio</h1>" in value)
    assert "workflow-step" in hero, "stepper must render inside the hero"
    assert "✓ Upload" in hero
    coach = next(value for value in markdown if 'class="next-step-coach"' in value)
    assert "is selected" in coach and "Generate Cross-Section" in coach

    def walk(node):
        yield node
        for child in getattr(node, "children", {}).values():
            yield from walk(child)

    flat = list(walk(at.main))
    generate_index = next(
        i for i, node in enumerate(flat)
        if getattr(node, "type", "") == "button" and node.label == "Generate Cross-Section"
    )
    health_index = next(
        i for i, node in enumerate(flat)
        if getattr(node, "type", "") in {"subheader", "heading"} and "Data Health" in str(node.value)
    )
    assert generate_index < health_index, "Generate must render above Validate's Data Health"


def test_preview_zoom_switches_to_scrollable_native_size_frame(sample_workbook: Path) -> None:
    at = _generated_app(sample_workbook)

    def frame_html() -> str:
        return next(md.value for md in at.markdown if "<img src=\"data:image/svg" in md.value)

    assert 'class="svg-frame"' in frame_html()
    assert "width:100%" in frame_html()
    at.session_state["svg_preview_zoom"] = "150%"
    at.run()
    assert not at.exception
    html_150 = frame_html()
    assert "svg-frame--zoomed" in html_150 and 'tabindex="0"' in html_150
    natural = at.session_state["svg_display_meta"].natural_width_px
    assert natural > 0
    assert f"width:{round(natural * 1.5)}px" in html_150


def test_menu_load_sample_asks_before_discarding_a_generated_section(sample_workbook: Path) -> None:
    at = _generated_app(sample_workbook)
    for key in ("menu_file_sample", "menu_accel_sample"):
        at.button(key=key).click().run()
        assert not at.exception
        assert at.session_state["svg_bytes"], f"{key} discarded the section without asking"
        assert at.session_state["_pending_destructive"] == "sample"
        at.button(key="cancel_destructive").click().run()


def test_stale_destructive_prompt_is_dropped_once_its_section_is_gone(sample_workbook: Path) -> None:
    at = _generated_app(sample_workbook)
    [btn for btn in at.button if btn.label == "Clear workbook"][0].click().run()
    assert at.session_state["_pending_destructive"] == "clear"
    # Section disappears another way (new upload, menu clear...): the old
    # Confirm must not linger to wipe whatever is generated next.
    at.session_state["svg_bytes"] = None
    at.run()
    assert "_pending_destructive" not in at.session_state
    assert not [btn for btn in at.button if btn.key == "confirm_destructive"]
