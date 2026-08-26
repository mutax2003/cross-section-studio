"""Optional password gate for shared Streamlit deployments."""

from __future__ import annotations

import hmac
import logging
import os
import time
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)

_MAX_AUTH_ATTEMPTS = 5
_LOCKOUT_SECONDS = 30.0
_AUTH_HMAC_KEY = b"css-auth"

# Sensitive session keys cleared on Sign out (shared-password / kiosk safety).
_LOGOUT_CLEAR_KEYS = (
    "file_bytes",
    "parse_result",
    "import_report",
    "quality_report",
    "mapping_proposal",
    "detection_result",
    "hole_ids",
    "unique_lithology_codes",
    "lithology_index",
    "parse_signature",
    "file_hash",
    "lithology_aliases",
    "render_cache_key",
    "polygon_overlap_warnings",
    "section_lithology_codes",
    "section_polygon_count",
    "section_hole_count",
    "transect_selection_key",
    "transect_selection",
    "svg_display_meta",
    "transect_candidates",
    "svg_bytes",
    "png_bytes",
    "pdf_bytes",
    "section_build_subset_json",
    "section_build_request_json",
    "uploaded_name",
    "qa_narrative",
    "qa_fix_plan",
    "ai_report_suggestion",
    "ai_lithology_suggestions",
    "ai_correlation_suggestions",
    "ai_sheet_roles",
    "ai_column_suggestions",
    "section_qa_answer",
    "ai_figure_caption",
)


def _auth_required() -> bool:
    return os.environ.get("CROSS_SECTION_AUTH_REQUIRED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _configured_password() -> str:
    env_password = os.environ.get("CROSS_SECTION_AUTH_PASSWORD", "").strip()
    if env_password:
        return env_password
    try:
        secrets = getattr(st, "secrets", None)
        if secrets is not None:
            for key in ("CROSS_SECTION_AUTH_PASSWORD", "cross_section_auth_password"):
                try:
                    value = str(secrets.get(key, "") or "").strip()
                except FileNotFoundError:
                    # No secrets.toml — expected for local/AppTest; skip silently.
                    value = ""
                except (AttributeError, KeyError, TypeError) as exc:
                    logger.warning("Auth secret %s unreadable: %s", key, exc)
                    value = ""
                if value:
                    return value
    except FileNotFoundError:
        pass
    except (AttributeError, KeyError, TypeError) as exc:
        logger.warning("Streamlit secrets unavailable for auth password: %s", exc)
    return ""


def _passwords_match(entered: str, password: str) -> bool:
    """Constant-time compare via fixed-length HMAC digests (no length oracle)."""
    left = hmac.digest(_AUTH_HMAC_KEY, entered.encode("utf-8"), "sha256")
    right = hmac.digest(_AUTH_HMAC_KEY, password.encode("utf-8"), "sha256")
    return hmac.compare_digest(left, right)


def _clear_sensitive_session_on_logout(session: Any | None = None) -> None:
    """Drop workbook / export / AI residuals so the next shared-password user starts clean."""
    target = session if session is not None else st.session_state
    for key in _LOGOUT_CLEAR_KEYS:
        if key in target:
            target[key] = None


def render_logout_control() -> None:
    """Show Sign out in the sidebar when the password gate is active."""
    if not _configured_password() or not st.session_state.get("_auth_ok"):
        return
    with st.sidebar:
        if st.button("Sign out", key="_auth_sign_out"):
            st.session_state["_auth_ok"] = False
            st.session_state.pop("_auth_failures", None)
            st.session_state.pop("_auth_lock_until", None)
            _clear_sensitive_session_on_logout()
            st.rerun()


def require_auth() -> None:
    """Stop the app until the shared password is entered (env/secrets-gated).

    When no password is configured, the gate is a no-op so Cloud demos work
    without secrets. ``CROSS_SECTION_AUTH_REQUIRED`` still fails closed if a
    password was expected but missing.
    """
    password = _configured_password()
    if not password:
        if _auth_required():
            st.title("Cross Section Studio")
            st.error(
                "This deployment requires authentication, but "
                "`CROSS_SECTION_AUTH_PASSWORD` is not set."
            )
            st.stop()
        return
    if st.session_state.get("_auth_ok"):
        return

    lock_until = float(st.session_state.get("_auth_lock_until") or 0.0)
    now = time.monotonic()
    locked = now < lock_until

    st.title("Cross Section Studio")
    st.caption("Authentication required for this deployment.")
    if locked:
        remaining = max(1, int(lock_until - now))
        st.warning(f"Too many failed attempts. Try again in {remaining}s.")
        st.stop()

    entered = st.text_input("Password", type="password", key="_auth_password_input")
    if st.button("Sign in", type="primary"):
        ok = _passwords_match(entered, password)
        if ok:
            st.session_state["_auth_ok"] = True
            st.session_state.pop("_auth_password_input", None)
            st.session_state.pop("_auth_failures", None)
            st.session_state.pop("_auth_lock_until", None)
            st.rerun()
        failures = int(st.session_state.get("_auth_failures") or 0) + 1
        st.session_state["_auth_failures"] = failures
        if failures >= _MAX_AUTH_ATTEMPTS:
            st.session_state["_auth_lock_until"] = time.monotonic() + _LOCKOUT_SECONDS
            st.session_state["_auth_failures"] = 0
            st.error("Invalid password. Account temporarily locked.")
        else:
            st.error("Invalid password.")
        st.session_state.pop("_auth_password_input", None)
    st.stop()
