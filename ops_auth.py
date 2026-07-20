"""Optional password gate for shared Streamlit deployments."""

from __future__ import annotations

import hmac
import os
import time

import streamlit as st

_MAX_AUTH_ATTEMPTS = 5
_LOCKOUT_SECONDS = 30.0


def _auth_required() -> bool:
    return os.environ.get("CROSS_SECTION_AUTH_REQUIRED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _configured_password() -> str:
    return os.environ.get("CROSS_SECTION_AUTH_PASSWORD", "").strip()


def render_logout_control() -> None:
    """Show Sign out in the sidebar when the password gate is active."""
    if not _configured_password() or not st.session_state.get("_auth_ok"):
        return
    with st.sidebar:
        if st.button("Sign out", key="_auth_sign_out"):
            st.session_state["_auth_ok"] = False
            st.session_state.pop("_auth_failures", None)
            st.session_state.pop("_auth_lock_until", None)
            st.rerun()


def require_auth() -> None:
    """Stop the app until the shared password is entered (env-gated)."""
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
        try:
            ok = hmac.compare_digest(entered.encode("utf-8"), password.encode("utf-8"))
        except ValueError:
            ok = False
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
