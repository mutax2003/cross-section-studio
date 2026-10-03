"""Menubar (File / Edit / View / Help), keyboard shortcuts, and help dialogs.

Menus are ``st.menu_button`` widgets: they expose ``role="menu"`` /
``role="menuitem"``, open with Enter or Space, move with the arrow keys and
return focus to their trigger on Esc. Shortcuts use Alt+Shift+letter (Option+
Shift on macOS) so they never shadow browser bindings such as Ctrl+G (find
next), Ctrl+H (history) or F1 (browser help).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import streamlit as st

from app_state import clear_ai_session_state, clear_section_output_state
from app_upload import render_input_template_download, request_destructive
from paths import help_topic_path
from ui_output_presets import resolve_output_preset

# Unique button labels for the hidden accelerator buttons (the bridge clicks these).
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
MENU_TEMPLATE_KEY = "_menu_download_template"
SHORTCUTS_ENABLED_KEY = "keyboard_shortcuts_enabled"
SET_BY_OUTPUT_STYLE = "(set by output style)"


@dataclass(frozen=True)
class Shortcut:
    """One accelerator: action id, ``KeyboardEvent.code``, display text, action label."""

    action: str
    code: str
    keys: str
    label: str


# Alt+Shift+letter, matched on ``event.code`` so macOS Option (which changes
# ``event.key`` to a symbol such as "©") still works.
SHORTCUTS: tuple[Shortcut, ...] = (
    Shortcut("sample", "KeyO", "Alt+Shift+O", "Load sample project"),
    Shortcut("generate", "KeyG", "Alt+Shift+G", "Generate section"),
    Shortcut("clear", "KeyC", "Alt+Shift+C", "Clear section output"),
    Shortcut("hatches", "KeyH", "Alt+Shift+H", "Hatch patterns on or off"),
    Shortcut("help_keys", "KeyK", "Alt+Shift+K", "Keyboard shortcuts help"),
    Shortcut("help_start", "KeyS", "Alt+Shift+S", "Getting started help"),
)
_SHORTCUT_BY_ACTION = {s.action: s for s in SHORTCUTS}
_ACCEL_LABEL_BY_ACTION = {
    "sample": ACCEL_SAMPLE,
    "generate": ACCEL_GENERATE,
    "clear": ACCEL_CLEAR,
    "hatches": ACCEL_HATCHES,
    "help_keys": ACCEL_HELP_KEYS,
    "help_start": ACCEL_HELP_START,
}

_SHORTCUT_ROWS: tuple[tuple[str, str], ...] = tuple((s.keys, s.label) for s in SHORTCUTS)


def shortcut_reference_table() -> str:
    """Markdown table of documented shortcuts (canonical shortcut list)."""
    lines = ["| Shortcut | Action |", "| --- | --- |"]
    for keys, action in _SHORTCUT_ROWS:
        lines.append(f"| `{keys}` | {action} |")
    return "\n".join(lines)


def keyboard_shortcuts_help_body() -> str:
    """Single source of truth for Help → Keyboard shortcuts."""
    return (
        "# Keyboard shortcuts\n\n"
        + shortcut_reference_table()
        + "\n\nOn macOS, Alt is the **Option** key. Shortcuts are ignored while "
        "you type in a text field. Turn them off with **View → Keyboard "
        "shortcuts** if they clash with your browser, screen reader or "
        "keyboard layout.\n\n"
        "Menus work from the keyboard: Tab to **File**, **Edit**, **View** or "
        "**Help**, press Enter or Space to open, use the arrow keys to move, "
        "Enter to choose and Esc to close.\n"
    )


def load_help_markdown(topic: str) -> str:
    """Load a help topic from ``docs/help/{topic}.md`` (safe basename only)."""
    stem = str(topic).strip().removesuffix(".md")
    if stem == "keyboard-shortcuts":
        return keyboard_shortcuts_help_body()
    path = help_topic_path(stem)
    if path is None or not path.is_file():
        return f"Help topic `{topic}` was not found."
    text = path.read_text(encoding="utf-8")
    if stem == "about":
        from app_version import about_version_markdown

        text = text.rstrip() + "\n\n" + about_version_markdown()
    return text


@st.dialog("Check for updates")
def _updates_dialog() -> None:
    from app_version import check_for_updates
    from desktop_updater import auto_install_allowed, download_and_schedule_install
    from paths import is_frozen

    with st.spinner("Checking for updates…"):
        result = check_for_updates()
    st.caption(f"Installed version: **{result.current_version}**")
    if result.error:
        st.warning(result.error)
        st.caption(f"Manifest: `{result.manifest_url}`")
        st.caption(
            "Set `CROSS_SECTION_UPDATE_MANIFEST_URL` to override the default "
            "GitHub Releases manifest URL."
        )
    elif result.update_available:
        st.success(f"Update available: **{result.latest_version}**")
        if result.notes:
            st.markdown(result.notes)
        if result.download_url:
            st.link_button(
                "Open download page",
                result.download_url,
                width="stretch",
            )
            st.caption(
                "Manual install: download the Windows zip, quit Cross Section Studio, "
                "then replace the install folder (keep `_internal` next to the `.exe`). "
                f"Expected SHA-256: `{result.sha256 or 'see release notes'}`."
            )
            if auto_install_allowed() and result.sha256:
                if st.button(
                    "Download and install (restart)",
                    key="menu_help_updates_install",
                    width="stretch",
                    type="primary",
                ):
                    status = st.empty()
                    try:

                        def _progress(message: str, fraction: float | None) -> None:
                            if fraction is None:
                                status.info(message)
                            else:
                                status.progress(fraction, text=message)

                        download_and_schedule_install(result, progress=_progress)
                        status.success(
                            "Update verified. The app will close and restart on the new build."
                        )
                        st.caption(
                            "If the window does not close, quit Cross Section Studio manually."
                        )
                        time.sleep(1.2)
                        os._exit(0)
                    except Exception as exc:  # noqa: BLE001 — show to operator
                        status.error(f"Update failed: {exc}")
            elif is_frozen() and not result.sha256:
                st.caption("Auto-install requires a SHA-256 in the release manifest.")
            elif not is_frozen():
                st.caption(
                    "Auto-install is available in the Windows desktop build "
                    "(or set `CROSS_SECTION_ALLOW_DEV_UPDATE=1`)."
                )
        else:
            st.info("A newer version is published but the manifest has no download URL.")
    else:
        st.info(
            f"You are on the latest published version"
            f"{f' ({result.latest_version})' if result.latest_version else ''}."
        )
    if st.button("Close", key="help_dialog_close_updates"):
        st.rerun()


def _set_help_topic(topic: str) -> None:
    st.session_state[MENU_HELP_TOPIC_KEY] = topic


def shortcuts_enabled(state: object | None = None) -> bool:
    """Whether the Alt+Shift shortcuts are on (session setting, default on)."""
    state = st.session_state if state is None else state
    return bool(state.get(SHORTCUTS_ENABLED_KEY, True))  # type: ignore[attr-defined]


@dataclass(frozen=True)
class ViewToggle:
    """State of one View-menu toggle as the current output style will draw it."""

    action: str
    name: str
    on: bool
    set_by_output_style: bool

    @property
    def menu_label(self) -> str:
        text = f"{self.name}: {'on' if self.on else 'off'}"
        return f"{text} {SET_BY_OUTPUT_STYLE}" if self.set_by_output_style else text


def view_toggles(state: object | None = None) -> tuple[ViewToggle, ...]:
    """Hatch / legend / ground toggles with their effective on/off state.

    Uses the same rules the build uses (``effective_render_options``) so a
    toggle the output style overrides is reported as set by the output style
    instead of silently doing nothing.
    """
    from app_build import effective_render_options

    state = st.session_state if state is None else state
    get = state.get  # type: ignore[attr-defined]
    preset = resolve_output_preset(str(get("output_preset", "section_sheet")))

    def effective(ground: bool, legend: bool):
        return effective_render_options(
            report_preset=preset.report_preset,
            render_layout=preset.render_layout,
            show_ground_surface=ground,
            track_width_m=3.0,
            show_legend=legend,
            interpolate_water_table=False,
            allow_pinch_outs=False,
            consulting_title_block=None,
            sample_figure_profile=preset.sample_figure_profile,
        )

    ground = bool(get("show_ground_surface", True))
    legend = bool(get("show_legend", True))
    current = effective(ground, legend)
    # The sidebar also locks ground surface for sample-figure styles.
    ground_locked = preset.sample_figure_profile or (
        effective(True, legend).show_ground_surface == effective(False, legend).show_ground_surface
    )
    legend_locked = effective(ground, True).show_legend == effective(ground, False).show_legend
    return (
        ViewToggle("hatches", "Hatch patterns", bool(get("show_hatches", True)), False),
        ViewToggle("legend", "Legend on chart", current.show_legend, legend_locked),
        ViewToggle("ground", "Ground surface", current.show_ground_surface, ground_locked),
    )


def _with_keys(label: str, action: str, enabled: bool) -> str:
    shortcut = _SHORTCUT_BY_ACTION.get(action)
    if enabled and shortcut is not None:
        return f"{label} ({shortcut.keys})"
    return label


def menu_items(state: object | None = None) -> dict[str, tuple[tuple[str, str], ...]]:
    """``{menu: ((option label, action id), ...)}`` for the current session state."""
    enabled = shortcuts_enabled(state)
    view: list[tuple[str, str]] = []
    for toggle in view_toggles(state):
        action = f"locked_{toggle.action}" if toggle.set_by_output_style else toggle.action
        view.append((_with_keys(toggle.menu_label, toggle.action, enabled), action))
    view.append((f"Keyboard shortcuts: {'on' if enabled else 'off'}", "shortcuts"))
    return {
        "File": (
            (_with_keys("Load sample project", "sample", enabled), "sample"),
            ("Download template", "template"),
            (_with_keys("Generate section", "generate", enabled), "generate"),
            (_with_keys("Clear section output", "clear", enabled), "clear"),
        ),
        "Edit": (("Clear AI suggestions", "clear_ai"),),
        "View": tuple(view),
        "Help": (
            (_with_keys("Getting started", "help_start", enabled), "help_start"),
            ("Generate and exports", "help_exports"),
            ("Consulting layouts (gINT, Strater)", "help_consulting"),
            (_with_keys("Keyboard shortcuts", "help_keys", enabled), "help_keys"),
            ("Workbook and data entry", "help_workbook"),
            ("About", "about"),
            ("Check for updates", "updates"),
        ),
    }


_HELP_TOPICS = {
    "help_start": "getting-started",
    "help_exports": "generate-exports",
    "help_consulting": "consulting-ux",
    "help_keys": "keyboard-shortcuts",
    "help_workbook": "workbook-quick",
    "about": "about",
}
_TOGGLE_KEYS = {
    "hatches": ("show_hatches", True),
    "legend": ("show_legend", True),
    "ground": ("show_ground_surface", True),
}
_MENU_HELP = {
    "File": (
        "Enter data in Excel (Help → Workbook and data entry), then upload it in "
        "the sidebar. Load sample project skips the prep."
    ),
    "Edit": "Transect, style and title fields stay in the sidebar.",
    "View": "Display settings. Items marked “set by output style” follow the sidebar Output style.",
    "Help": None,
}


def _run_action(action: str) -> None:
    """Perform one menu or shortcut action (shared by menus and the bridge)."""
    state = st.session_state
    if action == "sample":
        # Asks in the sidebar first when it would discard a section.
        try:
            request_destructive("sample")
        except FileNotFoundError as exc:
            state["_flash_error"] = str(exc)
    elif action == "template":
        state[MENU_TEMPLATE_KEY] = True
    elif action == "generate":
        state["_regenerate_requested"] = True
    elif action == "clear":
        clear_section_output_state()
        state["_flash_success"] = "Cleared the generated section."
    elif action == "clear_ai":
        clear_ai_session_state()
        state["_flash_success"] = "Cleared AI notes and suggestions."
    elif action in _TOGGLE_KEYS:
        key, default = _TOGGLE_KEYS[action]
        state[key] = not bool(state.get(key, default))
    elif action.startswith("locked_"):
        name = {"locked_legend": "Legend on chart", "locked_ground": "Ground surface"}.get(
            action, "This setting"
        )
        st.toast(
            f"{name} is set by the output style. Change Output style in the sidebar to control it."
        )
        return
    elif action == "shortcuts":
        state[SHORTCUTS_ENABLED_KEY] = not shortcuts_enabled(state)
        st.toast(f"Keyboard shortcuts {'on' if state[SHORTCUTS_ENABLED_KEY] else 'off'}.")
    elif action in _HELP_TOPICS:
        _set_help_topic(_HELP_TOPICS[action])
    elif action == "updates":
        state["_menu_check_updates"] = True
    else:
        return
    st.rerun()


@st.dialog("Help")
def _help_dialog(topic: str) -> None:
    st.markdown(load_help_markdown(topic))
    if st.button("Close", key=f"help_dialog_close_{topic}"):
        st.rerun()


@st.dialog("Download template")
def _template_dialog() -> None:
    st.markdown(
        "Fill in **Collars** and **Lithology** in Excel, then upload the workbook in the sidebar."
    )
    render_input_template_download(key="menu_file_download_template")
    if st.button("Close", key="template_dialog_close"):
        st.rerun()


def render_menubar() -> None:
    """Render the File / Edit / View / Help menus and process their actions."""
    st.session_state.setdefault(SHORTCUTS_ENABLED_KEY, True)
    menus = menu_items()
    enabled = shortcuts_enabled()
    # Keyed container: a raw <div> via st.markdown is auto-closed and wraps
    # nothing, so the .app-menubar chrome never applied to the menus.
    with st.container(key="app_menubar"):
        columns = st.columns([1, 1, 1, 1, 4])
        for column, (menu, items) in zip(columns, menus.items()):
            with column:
                labels = [label for label, _action in items]
                choice = st.menu_button(
                    menu,
                    labels,
                    key=f"menu_{menu.lower()}",
                    help=_MENU_HELP[menu],
                    type="tertiary",
                    width="stretch",
                )
                if choice is not None:
                    _run_action(dict(items)[choice])
        with columns[-1]:
            help_keys = _SHORTCUT_BY_ACTION["help_keys"].keys
            st.caption(
                f"Keyboard shortcuts: {help_keys} for the list"
                if enabled
                else "Keyboard shortcuts are off (View menu)"
            )

    # One-shot topic so native dialog dismiss does not reopen forever.
    topic = st.session_state.pop(MENU_HELP_TOPIC_KEY, None)
    if isinstance(topic, str) and topic:
        _help_dialog(topic)
    if st.session_state.pop(MENU_TEMPLATE_KEY, None):
        _template_dialog()
    if st.session_state.pop("_menu_check_updates", None):
        _updates_dialog()

    _render_accelerator_buttons()
    # Mount on EVERY run: an element not rendered in a run is removed by
    # Streamlit, and a detached iframe's listener can no longer reach the page
    # (shortcuts used to die after the first rerun). Same srcdoc each run, so
    # React keeps the node; the JS swaps in a fresh listener when it remounts.
    with st.container(key="shortcut_bridge"):
        _inject_shortcut_bridge(enabled=enabled, menus=menus)


def _render_accelerator_buttons() -> None:
    """Hidden buttons targeted by the keyboard bridge (must stay in the DOM).

    A keyed container gets a stable ``st-key-menu_accels`` class that the CSS
    visually hides (clip, not ``display:none``, so the bridge's text lookup
    and ``click()`` keep working). A raw ``<div>`` via ``st.markdown`` cannot
    wrap later elements — Streamlit auto-closes it immediately.
    """
    with st.container(key="menu_accels"):
        for column, (action, label) in zip(
            st.columns(len(_ACCEL_LABEL_BY_ACTION)), _ACCEL_LABEL_BY_ACTION.items()
        ):
            with column:
                if st.button(label, key=f"menu_accel_{action}"):
                    _run_action(action)


def _bridge_config(*, enabled: bool, menus: dict[str, tuple[tuple[str, str], ...]]) -> dict:
    """Data the bridge script needs: key codes, button labels, aria-keyshortcuts."""
    aria: dict[str, str] = {}
    if enabled:
        for items in menus.values():
            for label, action in items:
                shortcut = _SHORTCUT_BY_ACTION.get(action.removeprefix("locked_"))
                if shortcut is not None:
                    aria[label] = shortcut.keys
    return {
        "enabled": enabled,
        "codes": {s.code: _ACCEL_LABEL_BY_ACTION[s.action] for s in SHORTCUTS} if enabled else {},
        "aria": aria,
    }


_BRIDGE_JS = """
<script>
(function () {
  const CFG = __CFG__;
  const doc = window.parent.document;
  const MENU_TRIGGER = '.st-key-app_menubar [aria-haspopup="menu"]';

  function isEditableTarget(el) {
    if (!el) return false;
    const tag = (el.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select') return true;
    if (el.isContentEditable) return true;
    const role = (el.getAttribute && el.getAttribute('role')) || '';
    return role === 'textbox' || role === 'combobox' || role === 'searchbox';
  }
  function clickLabel(label) {
    for (const btn of doc.querySelectorAll('.st-key-menu_accels button')) {
      if ((btn.innerText || btn.textContent || '').trim() === label) {
        btn.click();
        return true;
      }
    }
    return false;
  }
  // Stable selector for an element Streamlit may re-create on rerun.
  function selectorFor(el) {
    if (!el || el === doc.body || !el.closest) return null;
    const keyed = el.closest('[class*="st-key-"]');
    const m = keyed && keyed.className.match(/(?:^|\\s)(st-key-[\\w-]+)/);
    return m ? '.' + m[1] + ' ' + el.tagName.toLowerCase() : null;
  }
  function rememberFocus(sel) {
    if (sel) doc._cssFocusReturn = { sel: sel, restoredAt: 0 };
  }

  // Shortcuts: bubble phase, Alt+Shift only, matched on event.code (macOS
  // Option changes event.key). Only combos we handle are prevented.
  function onKey(e) {
    if (e.repeat || e.defaultPrevented) return;
    if (!e.altKey || !e.shiftKey || e.ctrlKey || e.metaKey) return;
    if (isEditableTarget(e.target)) return;
    const label = CFG.codes[e.code];
    if (!label) return;
    e.preventDefault();
    rememberFocus(selectorFor(doc.activeElement));
    clickLabel(label);
  }

  // A menu item was chosen: come back to its menu button after the rerun
  // (and after any dialog it opened closes) instead of <body>.
  function onMenuChoose(e) {
    if (e.type === 'keydown' && e.key !== 'Enter' && e.key !== ' ') return;
    const item = e.target && e.target.closest && e.target.closest('[role="menuitem"]');
    if (!item) return;
    const trigger = doc.querySelector(MENU_TRIGGER + '[aria-expanded="true"]') || doc._cssLastMenu;
    rememberFocus(selectorFor(trigger));
  }
  // Safety net: if a menu opened from the keyboard but focus stayed on its
  // button (seen while the page is still settling), move it into the menu.
  function onTriggerKey(e) {
    if (['Enter', ' ', 'ArrowDown', 'ArrowUp'].indexOf(e.key) < 0) return;
    const trigger = e.target && e.target.closest && e.target.closest(MENU_TRIGGER);
    if (!trigger) return;
    [120, 350].forEach(function (ms) {
      setTimeout(function () {
        if (trigger.getAttribute('aria-expanded') !== 'true' || doc.activeElement !== trigger) return;
        const menu = doc.querySelector('[role="menu"]');
        if (!menu) return;
        const items = menu.querySelectorAll('[role="menuitem"]');
        const item = e.key === 'ArrowUp' ? items[items.length - 1] : items[0];
        (item || menu).focus();
      }, ms);
    });
  }
  function onFocusIn(e) {
    const t = e.target;
    if (t && t.matches && t.matches(MENU_TRIGGER)) doc._cssLastMenu = t;
    const p = doc._cssFocusReturn;
    if (!p || !t || t === doc.body || !t.closest) return;
    if (t.closest('[role="dialog"],[role="menu"],.st-key-shortcut_bridge')) return;
    if (t.matches(p.sel)) return;
    doc._cssFocusReturn = null;  // the user moved on; do not steal focus
  }
  function tick() {
    const p = doc._cssFocusReturn;
    if (!p) return;
    const running = !!doc.querySelector('[data-testid="stStatusWidget"]');
    const dialogOpen = !!doc.querySelector('[role="dialog"]');
    const ae = doc.activeElement;
    if (ae && ae !== doc.body) {
      if (p.restoredAt && !running && !dialogOpen && ae.matches(p.sel)
          && Date.now() - p.restoredAt > 1500) {
        doc._cssFocusReturn = null;
      }
      return;
    }
    if (dialogOpen) return;
    const target = doc.querySelector(p.sel);
    if (!target || target.disabled) return;
    target.focus({ preventScroll: true });
    p.restoredAt = Date.now();
  }

  let labelPending = false;
  function labelMenuItems() {
    labelPending = false;
    for (const el of doc.querySelectorAll('[role="menuitem"]')) {
      const keys = CFG.aria[(el.innerText || el.textContent || '').trim()];
      if (keys) {
        if (el.getAttribute('aria-keyshortcuts') !== keys) el.setAttribute('aria-keyshortcuts', keys);
      } else if (el.hasAttribute('aria-keyshortcuts')) {
        el.removeAttribute('aria-keyshortcuts');
      }
    }
  }
  function scheduleLabels() {
    if (labelPending) return;
    labelPending = true;
    setTimeout(labelMenuItems, 30);
  }

  // Replace anything a previous (possibly detached) bridge left behind.
  const old = doc._cssMenuBridge;
  if (old) {
    doc.removeEventListener('keydown', old.onKey, true);
    doc.removeEventListener('keydown', old.onKey, false);
    doc.removeEventListener('click', old.onMenuChoose, true);
    doc.removeEventListener('keydown', old.onMenuChoose, true);
    doc.removeEventListener('keydown', old.onTriggerKey, true);
    doc.removeEventListener('focusin', old.onFocusIn, true);
    try { old.observer.disconnect(); } catch (err) {}
    try { old.win.clearInterval(old.timer); } catch (err) {}
  }
  if (doc._cssMenuAccelHandler) {  // listener from older builds
    doc.removeEventListener('keydown', doc._cssMenuAccelHandler, true);
    doc._cssMenuAccelHandler = null;
  }
  if (CFG.enabled) doc.addEventListener('keydown', onKey, false);
  // Passive observers (capture so they still see events a widget stops).
  doc.addEventListener('click', onMenuChoose, true);
  doc.addEventListener('keydown', onMenuChoose, true);
  doc.addEventListener('keydown', onTriggerKey, true);
  doc.addEventListener('focusin', onFocusIn, true);
  const observer = new MutationObserver(scheduleLabels);
  observer.observe(doc.body, { childList: true, subtree: true });
  const timer = window.setInterval(tick, 200);
  doc._cssMenuBridge = { onKey, onMenuChoose, onTriggerKey, onFocusIn, observer, timer, win: window };
})();
</script>
"""


def _inject_shortcut_bridge(
    *, enabled: bool, menus: dict[str, tuple[tuple[str, str], ...]]
) -> None:
    """Keyboard bridge in a 1px ``st.iframe`` (height must be > 0), out of the tab order."""
    config = json.dumps(_bridge_config(enabled=enabled, menus=menus), ensure_ascii=False)
    st.iframe(_BRIDGE_JS.replace("__CFG__", config.replace("</", "<\\/")), height=1, tab_index=-1)


def help_topics_on_disk() -> tuple[str, ...]:
    """Basenames (without .md) available under docs/help for tests."""
    from paths import help_dir

    root = help_dir()
    if not root.is_dir():
        return ()
    return tuple(sorted(path.stem for path in root.glob("*.md") if path.stem != "README"))
