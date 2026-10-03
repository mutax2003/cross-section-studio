"""Tests for Windows-style menubar helpers and help files."""

from __future__ import annotations

from pathlib import Path

from app_menubar import (
    _SHORTCUT_ROWS,
    ACCEL_GENERATE,
    ACCEL_LABELS,
    help_topics_on_disk,
    keyboard_shortcuts_help_body,
    load_help_markdown,
    shortcut_reference_table,
)
from paths import help_dir, help_topic_path


def test_help_dir_contains_operator_topics() -> None:
    topics = set(help_topics_on_disk())
    assert {
        "getting-started",
        "keyboard-shortcuts",
        "workbook-quick",
        "about",
        "generate-exports",
        "consulting-ux",
    } <= topics
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
    assert "Alt+Shift+G" in shortcut_reference_table()


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
    menus = [menu.label for menu in at.menu_button]
    assert menus[:4] == ["File", "Edit", "View", "Help"]
    assert {btn.key for btn in at.button} >= {f"menu_accel_{a}" for a in ("sample", "generate")}


def test_shortcuts_avoid_browser_reserved_combos() -> None:
    from app_menubar import SHORTCUTS

    codes = [s.code for s in SHORTCUTS]
    assert len(set(codes)) == len(codes)
    for shortcut in SHORTCUTS:
        assert shortcut.keys.startswith("Alt+Shift+"), shortcut
        assert shortcut.code.startswith("Key"), "match event.code so macOS Option works"
    table = shortcut_reference_table()
    for reserved in ("Ctrl+", "F1", "Cmd+"):
        assert reserved not in table


def test_bridge_uses_bubble_phase_and_only_prevents_handled_combos() -> None:
    from app_menubar import _BRIDGE_JS, _bridge_config, menu_items

    assert "addEventListener('keydown', onKey, false)" in _BRIDGE_JS
    assert (
        "e.code" in _BRIDGE_JS
        and "e.key ===" not in _BRIDGE_JS.split("function onKey")[1].split("}")[0]
    )
    # preventDefault only after a configured code matched.
    on_key = _BRIDGE_JS.split("function onKey")[1].split("function onMenuChoose")[0]
    assert on_key.index("if (!label) return;") < on_key.index("preventDefault")
    on = _bridge_config(enabled=True, menus=menu_items({}))
    assert on["codes"]["KeyG"] == "☰accel·generate"
    assert "Generate section (Alt+Shift+G)" in on["aria"]
    assert on["aria"]["Generate section (Alt+Shift+G)"] == "Alt+Shift+G"
    off_state = {"keyboard_shortcuts_enabled": False}
    off = _bridge_config(enabled=False, menus=menu_items(off_state))
    assert off["codes"] == {} and off["aria"] == {}
    assert "Generate section" in [label for label, _ in menu_items(off_state)["File"]]


def test_view_menu_shows_state_and_output_style_locks() -> None:
    from app_menubar import menu_items, view_toggles

    chart = {"output_preset": "chart", "show_hatches": False, "show_legend": True}
    labels = [label for label, _ in menu_items(chart)["View"]]
    assert "Hatch patterns: off (Alt+Shift+H)" in labels
    assert "Legend on chart: on" in labels
    assert "Keyboard shortcuts: on" in labels

    consulting = {"output_preset": "gwm_fence", "show_legend": True, "show_ground_surface": False}
    toggles = {t.action: t for t in view_toggles(consulting)}
    assert toggles["legend"].set_by_output_style and not toggles["legend"].on
    assert toggles["ground"].set_by_output_style and toggles["ground"].on
    actions = dict(menu_items(consulting)["View"])
    assert actions["Legend on chart: off (set by output style)"] == "locked_legend"
    assert actions["Ground surface: on (set by output style)"] == "locked_ground"
