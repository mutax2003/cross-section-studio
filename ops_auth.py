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
    "export_output_dir",
    "enable_ai_suggestions",
)
# Prefix-matched on Sign out: per-provider runtime LLM keys.
_LOGOUT_CLEAR_PREFIXES = ("_llm_api_key_runtime",)

_AUTH_WINDOW_SECONDS = 300.0


@st.cache_resource(show_spinner=False)
def _auth_throttle_store() -> dict[str, dict]:
    """Failed sign-ins per client, shared by every browser session.

    cache_resource lives for the server process (and survives Streamlit
    re-importing this module), so reconnecting cannot reset the count.
    Keyed by client IP, or one shared bucket when the runtime cannot tell
    clients apart.
    """
    return {"failures": {}, "lock_until": {}}


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
    for key in [k for k in list(target.keys()) if str(k).startswith(_LOGOUT_CLEAR_PREFIXES)]:
        target[key] = None


def _client_bucket() -> str:
    """Throttle key for the connecting client; one shared bucket if unknown."""
    try:
        address = getattr(st.context, "ip_address", None)
    except Exception:  # pragma: no cover - older runtimes
        address = None
    # Only a real address string separates clients; test doubles and None
    # share one bucket rather than each getting a fresh count.
    if isinstance(address, str) and address.strip():
        return address.strip()
    return "shared"


def _record_failed_attempt(bucket: str, now: float) -> float:
    """Record a failure; return the lock-until time (0.0 if still allowed)."""
    store = _auth_throttle_store()
    recent = [
        stamp for stamp in store["failures"].get(bucket, []) if now - stamp < _AUTH_WINDOW_SECONDS
    ]
    recent.append(now)
    if len(recent) >= _MAX_AUTH_ATTEMPTS:
        store["failures"][bucket] = []
        store["lock_until"][bucket] = now + _LOCKOUT_SECONDS
        return store["lock_until"][bucket]
    store["failures"][bucket] = recent
    return 0.0


def _lock_until(bucket: str) -> float:
    return float(_auth_throttle_store()["lock_until"].get(bucket, 0.0))


def _clear_throttle(bucket: str) -> None:
    store = _auth_throttle_store()
    store["failures"].pop(bucket, None)
    store["lock_until"].pop(bucket, None)


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

    bucket = _client_bucket()
    now = time.monotonic()
    lock_until = _lock_until(bucket)
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
            _clear_throttle(bucket)
            st.rerun()
        # Process-wide: a fresh browser session must not reset the count.
        if _record_failed_attempt(bucket, time.monotonic()):
            st.error("Invalid password. Account temporarily locked.")
        else:
            st.error("Invalid password.")
        st.session_state.pop("_auth_password_input", None)
    st.stop()
