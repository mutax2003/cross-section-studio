"""Tests for production ops helpers."""

from __future__ import annotations

import hmac
from pathlib import Path

import pytest


def test_ops_auth_skips_when_env_unset(monkeypatch) -> None:
    monkeypatch.delenv("CROSS_SECTION_AUTH_PASSWORD", raising=False)
    monkeypatch.delenv("CROSS_SECTION_AUTH_REQUIRED", raising=False)
    from ops_auth import require_auth

    require_auth()


def test_ops_auth_required_without_password_is_documented() -> None:
    source = Path("ops_auth.py").read_text(encoding="utf-8")
    assert "CROSS_SECTION_AUTH_REQUIRED" in source
    assert "_MAX_AUTH_ATTEMPTS" in source
    assert "render_logout_control" in source


def test_ops_auth_uses_constant_time_compare() -> None:
    """Guard: password check must use hmac.compare_digest (timing-safe)."""
    source = Path("ops_auth.py").read_text(encoding="utf-8")
    assert "hmac.compare_digest" in source
    assert "entered == password" not in source
    assert "except ValueError" in source  # defensive for runtimes that raise on length mismatch
    assert hmac.compare_digest(b"secret", b"secret")
    assert not hmac.compare_digest(b"secret", b"wrong")
    # Unequal lengths must not authenticate (False or ValueError depending on Python).
    try:
        assert not hmac.compare_digest(b"short", b"longer-password")
    except ValueError:
        pass


def test_ops_logging_json_format(monkeypatch) -> None:
    monkeypatch.setenv("CROSS_SECTION_LOG_FORMAT", "json")
    from ops_logging import configure_logging

    configure_logging()


def test_audit_event_writes_under_user_data(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CROSS_SECTION_AUDIT_LOG", "nested/test_audit.log")
    monkeypatch.setattr("ops_audit.user_data_dir", lambda: tmp_path)
    monkeypatch.setattr("paths.user_data_dir", lambda: tmp_path)
    from ops_audit import audit_event

    audit_event("unit_test", detail="ok")
    log_path = tmp_path / "nested" / "test_audit.log"
    assert log_path.exists()
    assert "unit_test" in log_path.read_text(encoding="utf-8")


def test_audit_rejects_path_outside_user_data(tmp_path, monkeypatch) -> None:
    outside = tmp_path / "outside_audit.log"
    monkeypatch.setenv("CROSS_SECTION_AUDIT_LOG", str(outside))
    monkeypatch.setattr("ops_audit.user_data_dir", lambda: tmp_path / "allowed")
    monkeypatch.setattr("paths.user_data_dir", lambda: tmp_path / "allowed")
    from ops_audit import audit_event

    # Best-effort: should not create the outside file.
    audit_event("should_fail_closed", detail="x")
    assert not outside.exists()


def test_audit_event_tolerates_non_serializable_fields(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CROSS_SECTION_AUDIT_LOG", "nested/bad_fields.log")
    monkeypatch.setattr("ops_audit.user_data_dir", lambda: tmp_path)
    monkeypatch.setattr("paths.user_data_dir", lambda: tmp_path)
    from ops_audit import audit_event

    audit_event("unit_test", payload=object())
    log_path = tmp_path / "nested" / "bad_fields.log"
    assert not log_path.exists() or "unit_test" not in log_path.read_text(encoding="utf-8")


def test_redact_secrets_scrubs_api_keys() -> None:
    from ops_logging import redact_secrets

    text = redact_secrets("api_key=sk-abc123456789 password=hunter2 Bearer tokensecret")
    assert "sk-abc" not in text
    assert "hunter2" not in text
    assert "tokensecret" not in text
    assert "[redacted]" in text


def test_apm_traces_rate_clamped(monkeypatch) -> None:
    monkeypatch.setenv("SENTRY_TRACES_RATE", "2.5")
    from ops_apm import _traces_sample_rate

    assert _traces_sample_rate() == 1.0
    monkeypatch.setenv("SENTRY_TRACES_RATE", "not-a-float")
    assert _traces_sample_rate() == 0.1


def test_gemini_uses_header_not_query_key() -> None:
    source = Path("ai_assistant.py").read_text(encoding="utf-8")
    assert "x-goog-api-key" in source
    assert ":generateContent?key=" not in source


def test_paths_audit_log_rejects_absolute_outside(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("paths.user_data_dir", lambda: tmp_path / "allowed")
    monkeypatch.setenv("CROSS_SECTION_AUDIT_LOG", str(tmp_path / "outside.log"))
    from paths import audit_log_path

    with pytest.raises(PermissionError):
        audit_log_path()


def test_save_lithology_style_rejects_non_hex(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("constants.lithology_styles_path", lambda: tmp_path / "styles.json")
    from constants import save_lithology_style_override

    with pytest.raises(ValueError, match="Invalid lithology color"):
        save_lithology_style_override("Clay", "red; background:url(x)", "..")
    save_lithology_style_override("Clay", "#38220F", "---")
    assert "#38220F" in (tmp_path / "styles.json").read_text(encoding="utf-8")
