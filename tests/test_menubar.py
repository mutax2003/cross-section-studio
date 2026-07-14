"""Tests for Windows-style menubar helpers and help files."""

from __future__ import annotations

from pathlib import Path

from app_menubar import (
    ACCEL_LABELS,
    ACCEL_GENERATE,
    _SHORTCUT_ROWS,
    help_topics_on_disk,
    keyboard_shortcuts_help_body,
    load_help_markdown,
    shortcut_reference_table,
)
from paths import help_dir, help_topic_path


def test_help_dir_contains_operator_topics() -> None:
    topics = set(help_topics_on_disk())
    assert {"getting-started", "keyboard-shortcuts", "workbook-quick", "about", "generate-exports"} <= topics
    assert "README" not in topics
    assert help_dir().is_dir()


def test_load_help_markdown_getting_started() -> None:
    text = load_help_markdown("getting-started")
    assert "Upload" in text
    assert "Validate" in text
    assert "Generate" in text


def test_keyboard_shortcuts_single_source() -> None:
    body = keyboard_shortcuts_help_body()
    dialog = load_help_markdown("keyboard-shortcuts")
    assert body == dialog
    for keys, action in _SHORTCUT_ROWS:
        assert keys in body
        assert action in body
    on_disk = (help_dir() / "keyboard-shortcuts.md").read_text(encoding="utf-8")
    for keys, _action in _SHORTCUT_ROWS:
        assert keys in on_disk
    assert ACCEL_GENERATE in ACCEL_LABELS
    assert len(ACCEL_LABELS) == 6
    assert "Ctrl+G" in shortcut_reference_table()


def test_help_topic_path_rejects_traversal() -> None:
    assert help_topic_path("../secrets") is None
    assert help_topic_path("") is None
    assert help_topic_path("a/b") is None
    assert help_topic_path("getting-started") == (help_dir() / "getting-started.md").resolve()
    assert help_topic_path("missing-topic.md") == (help_dir() / "missing-topic.md").resolve()


def test_load_help_missing_topic_message() -> None:
    text = load_help_markdown("does-not-exist-xyz")
    assert "not found" in text.lower()


def test_streamlit_menubar_present() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    labels = [btn.label for btn in at.button]
    assert any(label == "File" or "File" in str(label) for label in labels) or any(
        "accel" in str(label) for label in labels
    )
