"""Windows-style menubar, keyboard accelerators, and help dialogs."""

from __future__ import annotations

import streamlit as st

from app_state import clear_ai_session_state, clear_section_output_state
from app_upload import load_sample_workbook
from paths import cross_section_input_template, help_topic_path
from ui_output_presets import resolve_output_preset

# Unique button labels for the zero-height accelerator bridge (JS clicks these).
ACCEL_SAMPLE = "☰accel·sample"
ACCEL_GENERATE = "☰accel·generate"
ACCEL_CLEAR = "☰accel·clear"
ACCEL_HATCHES = "☰accel·hatches"
ACCEL_HELP_KEYS = "☰accel·help_keys"
ACCEL_HELP_START = "☰accel·help_start"

ACCEL_LABELS: tuple[str, ...] = (
    ACCEL_SAMPLE,
    ACCEL_GENERATE,
    ACCEL_CLEAR,
    ACCEL_HATCHES,
    ACCEL_HELP_KEYS,
    ACCEL_HELP_START,
)

MENU_HELP_TOPIC_KEY = "_menu_help_topic"

_SHORTCUT_ROWS: tuple[tuple[str, str], ...] = (
    ("Ctrl+Shift+O", "Load sample project"),
    ("Ctrl+G", "Generate / regenerate cross-section"),
    ("Ctrl+Shift+C", "Clear section output"),
    ("Ctrl+H", "Toggle hatch patterns"),
    ("Ctrl+/ or F1", "Keyboard shortcuts help"),
    ("Ctrl+Shift+/", "Getting started help"),
)


def shortcut_reference_table() -> str:
    """Markdown table of documented accelerators (canonical shortcut list)."""
    lines = ["| Shortcut | Action |", "| --- | --- |"]
    for keys, action in _SHORTCUT_ROWS:
        lines.append(f"| `{keys}` | {action} |")
    return "\n".join(lines)


def keyboard_shortcuts_help_body() -> str:
    """Single source of truth for Help → Keyboard shortcuts."""
    return (
        "# Keyboard shortcuts\n\n"
        + shortcut_reference_table()
        + "\n\nShortcuts apply when the app window has focus. "
        "On some browsers or hosts, system or browser bindings may take priority.\n"
    )


def load_help_markdown(topic: str) -> str:
    """Load a help topic from ``docs/help/{topic}.md`` (safe basename only)."""
    stem = str(topic).strip().removesuffix(".md")
    if stem == "keyboard-shortcuts":
        return keyboard_shortcuts_help_body()
    path = help_topic_path(stem)
    if path is None or not path.is_file():
        return f"Help topic `{topic}` was not found."
    return path.read_text(encoding="utf-8")


def _set_help_topic(topic: str) -> None:
    st.session_state[MENU_HELP_TOPIC_KEY] = topic


def _is_consulting_layout() -> bool:
    preset = resolve_output_preset(str(st.session_state.get("output_preset", "section_sheet")))
    return preset.render_layout == "consulting_section"


@st.dialog("Help")
def _help_dialog(topic: str) -> None:
    st.markdown(load_help_markdown(topic))
    if st.button("Close", key=f"help_dialog_close_{topic}"):
        st.rerun()


def _menu_item(label: str, *, key: str, shortcut: str | None = None) -> bool:
    clicked = st.button(label, key=key, width="stretch")
    if shortcut:
        st.caption(shortcut)
    return clicked


def render_menubar() -> None:
    """Render File / Edit / View / Help popovers and process menu intents."""
    st.markdown(
        '<div class="app-menubar" aria-label="Application menu">',
        unsafe_allow_html=True,
    )
    c_file, c_edit, c_view, c_help, c_hint = st.columns([1, 1, 1, 1, 4])

    with c_file:
        with st.popover("File", use_container_width=True):
            st.caption(
                "Enter data in Excel (Help → Workbook & data entry), then "
                "**Upload Excel workbook** in the sidebar. Load sample skips prep."
            )
            if _menu_item("Load sample project", key="menu_file_sample", shortcut="Ctrl+Shift+O"):
                try:
                    load_sample_workbook()
                    st.rerun()
                except FileNotFoundError as exc:
                    st.error(str(exc))
            template_path = cross_section_input_template()
            if template_path.is_file():
                st.download_button(
                    "Download template",
                    data=template_path.read_bytes(),
                    file_name=template_path.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="menu_file_download_template",
                    width="stretch",
                )
            if _menu_item("Clear section output", key="menu_file_clear", shortcut="Ctrl+Shift+C"):
                clear_section_output_state()
                st.success("Cleared generated SVG/PNG/PDF.")
                st.rerun()
            if _menu_item("Generate cross-section", key="menu_file_generate", shortcut="Ctrl+G"):
                st.session_state["_regenerate_requested"] = True
                st.rerun()
            st.caption("Geology is not edited in-app — use the Excel template, then the sidebar uploader.")

    with c_edit:
        with st.popover("Edit", use_container_width=True):
            st.caption("Session assist state")
            if _menu_item("Clear AI suggestions", key="menu_edit_clear_ai"):
                clear_ai_session_state()
                st.success("Cleared AI narratives and suggestions.")
                st.rerun()
            st.caption("Transect, style, and title fields stay in the sidebar.")

    with c_view:
        with st.popover("View", use_container_width=True):
            st.caption("Display toggles")
            if _menu_item("Toggle hatch patterns", key="menu_view_hatches", shortcut="Ctrl+H"):
                st.session_state["show_hatches"] = not bool(
                    st.session_state.get("show_hatches", True)
                )
                st.rerun()
            if _menu_item("Toggle legend on chart", key="menu_view_legend"):
                if _is_consulting_layout():
                    st.info("Consulting layout owns the footer legend — switch output style to toggle.")
                else:
                    st.session_state["show_legend"] = not bool(
                        st.session_state.get("show_legend", True)
                    )
                    st.rerun()
            if _menu_item("Toggle ground surface", key="menu_view_ground"):
                st.session_state["show_ground_surface"] = not bool(
                    st.session_state.get("show_ground_surface", True)
                )
                st.rerun()

    with c_help:
        with st.popover("Help", use_container_width=True):
            if _menu_item("Getting started", key="menu_help_start", shortcut="Ctrl+Shift+/"):
                _set_help_topic("getting-started")
                st.rerun()
            if _menu_item("Generate & exports", key="menu_help_exports"):
                _set_help_topic("generate-exports")
                st.rerun()
            if _menu_item("Consulting UX (gINT/Strater)", key="menu_help_consulting_ux"):
                _set_help_topic("consulting-ux")
                st.rerun()
            if _menu_item("Keyboard shortcuts", key="menu_help_keys", shortcut="Ctrl+/"):
                _set_help_topic("keyboard-shortcuts")
                st.rerun()
            if _menu_item("Workbook & data entry", key="menu_help_workbook"):
                _set_help_topic("workbook-quick")
                st.rerun()
            if _menu_item("About", key="menu_help_about"):
                _set_help_topic("about")
                st.rerun()

    with c_hint:
        st.caption("Menus · F1 or Ctrl+/ for shortcuts")

    st.markdown("</div>", unsafe_allow_html=True)

    # One-shot topic so native dialog dismiss does not reopen forever.
    topic = st.session_state.pop(MENU_HELP_TOPIC_KEY, None)
    if isinstance(topic, str) and topic:
        if hasattr(st, "dialog"):
            _help_dialog(topic)
        else:
            with st.expander(f"Help — {topic}", expanded=True):
                st.markdown(load_help_markdown(topic))

    _render_accelerator_buttons()
    # Parent-document listener persists across Streamlit reruns; avoid remounting the iframe.
    if not st.session_state.get("_shortcut_bridge_mounted"):
        _inject_shortcut_bridge()
        st.session_state["_shortcut_bridge_mounted"] = True


def _render_accelerator_buttons() -> None:
    """Hidden buttons targeted by the keyboard bridge (must stay in the DOM)."""
    st.markdown('<div class="app-menu-accels" aria-hidden="true">', unsafe_allow_html=True)
    b1, b2, b3, b4, b5, b6 = st.columns(6)
    with b1:
        if st.button(ACCEL_SAMPLE, key="menu_accel_sample"):
            try:
                load_sample_workbook()
            except FileNotFoundError as exc:
                st.session_state["_menu_accel_error"] = str(exc)
            st.rerun()
    with b2:
        if st.button(ACCEL_GENERATE, key="menu_accel_generate"):
            st.session_state["_regenerate_requested"] = True
            st.rerun()
    with b3:
        if st.button(ACCEL_CLEAR, key="menu_accel_clear"):
            clear_section_output_state()
            st.rerun()
    with b4:
        if st.button(ACCEL_HATCHES, key="menu_accel_hatches"):
            st.session_state["show_hatches"] = not bool(
                st.session_state.get("show_hatches", True)
            )
            st.rerun()
    with b5:
        if st.button(ACCEL_HELP_KEYS, key="menu_accel_help_keys"):
            _set_help_topic("keyboard-shortcuts")
            st.rerun()
    with b6:
        if st.button(ACCEL_HELP_START, key="menu_accel_help_start"):
            _set_help_topic("getting-started")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    err = st.session_state.pop("_menu_accel_error", None)
    if err:
        st.error(err)


def _inject_shortcut_bridge() -> None:
    """Ctrl/Cmd accelerators via a 1px ``st.iframe`` HTML bridge (height must be > 0)."""
    mapping = {
        "sample": ACCEL_SAMPLE,
        "generate": ACCEL_GENERATE,
        "clear": ACCEL_CLEAR,
        "hatches": ACCEL_HATCHES,
        "help_keys": ACCEL_HELP_KEYS,
        "help_start": ACCEL_HELP_START,
    }
    labels_js = {
        key: label.replace("\\", "\\\\").replace("'", "\\'") for key, label in mapping.items()
    }
    st.iframe(
        f"""
<script>
(function () {{
  const LABELS = {{
    sample: '{labels_js["sample"]}',
    generate: '{labels_js["generate"]}',
    clear: '{labels_js["clear"]}',
    hatches: '{labels_js["hatches"]}',
    help_keys: '{labels_js["help_keys"]}',
    help_start: '{labels_js["help_start"]}'
  }};
  function isEditableTarget(el) {{
    if (!el) return false;
    const tag = (el.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select') return true;
    if (el.isContentEditable) return true;
    const role = (el.getAttribute && el.getAttribute('role')) || '';
    if (role === 'textbox' || role === 'combobox' || role === 'searchbox') return true;
    return false;
  }}
  function clickLabel(label) {{
    const root = window.parent.document;
    const buttons = root.querySelectorAll('button');
    for (const btn of buttons) {{
      const text = (btn.innerText || btn.textContent || '').trim();
      if (text === label) {{
        btn.click();
        return true;
      }}
    }}
    return false;
  }}
  function onKey(e) {{
    if (e.repeat) return;
    if (isEditableTarget(e.target)) return;
    const mod = e.ctrlKey || e.metaKey;
    const key = (e.key || '').toLowerCase();
    let action = null;
    if (e.key === 'F1') {{
      action = 'help_keys';
    }} else if (mod && e.shiftKey && key === 'o') {{
      action = 'sample';
    }} else if (mod && !e.shiftKey && key === 'g') {{
      action = 'generate';
    }} else if (mod && e.shiftKey && key === 'c') {{
      action = 'clear';
    }} else if (mod && !e.shiftKey && key === 'h') {{
      action = 'hatches';
    }} else if (mod && !e.shiftKey && (key === '/' || key === '?')) {{
      action = 'help_keys';
    }} else if (mod && e.shiftKey && (key === '/' || key === '?')) {{
      action = 'help_start';
    }}
    if (!action) return;
    e.preventDefault();
    e.stopPropagation();
    clickLabel(LABELS[action]);
  }}
  const doc = window.parent.document;
  function disarmAccelButtons() {{
    const buttons = doc.querySelectorAll('button');
    for (const btn of buttons) {{
      const text = (btn.innerText || btn.textContent || '').trim();
      if (text.indexOf('☰accel·') === 0) {{
        btn.tabIndex = -1;
        btn.setAttribute('aria-hidden', 'true');
      }}
    }}
  }}
  disarmAccelButtons();
  if (doc._cssMenuAccelBound) return;
  doc.addEventListener('keydown', onKey, true);
  doc._cssMenuAccelBound = true;
}})();
</script>
""",
        height=1,
    )


def help_topics_on_disk() -> tuple[str, ...]:
    """Basenames (without .md) available under docs/help for tests."""
    from paths import help_dir

    root = help_dir()
    if not root.is_dir():
        return ()
    return tuple(sorted(path.stem for path in root.glob("*.md") if path.stem != "README"))
