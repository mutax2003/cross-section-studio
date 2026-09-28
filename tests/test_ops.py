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
    """Guard: password check must use fixed-length HMAC digests + compare_digest."""
    from ops_auth import _passwords_match

    source = Path("ops_auth.py").read_text(encoding="utf-8")
    assert "hmac.compare_digest" in source
    assert "hmac.digest" in source
    assert "entered == password" not in source
    assert _passwords_match("secret", "secret")
    assert not _passwords_match("secret", "wrong")
    assert not _passwords_match("short", "longer-password")
    assert hmac.compare_digest(b"secret", b"secret")


def test_ops_auth_logout_clears_sensitive_session() -> None:
    from ops_auth import _clear_sensitive_session_on_logout

    session = {
        "_auth_ok": True,
        "file_bytes": b"xlsx",
        "parse_result": {"holes": 1},
        "svg_bytes": b"<svg/>",
        "png_bytes": b"png",
        "pdf_bytes": b"pdf",
        "section_build_request_json": "{}",
        "uploaded_name": "site.xlsx",
        "ai_report_suggestion": "secret narrative",
        "keep_me": "ok",
    }
    _clear_sensitive_session_on_logout(session)
    assert session["file_bytes"] is None
    assert session["parse_result"] is None
    assert session["svg_bytes"] is None
    assert session["png_bytes"] is None
    assert session["pdf_bytes"] is None
    assert session["section_build_request_json"] is None
    assert session["uploaded_name"] is None
    assert session["ai_report_suggestion"] is None
    assert session["keep_me"] == "ok"
    assert session["_auth_ok"] is True  # auth flag cleared by render_logout_control


def test_ops_auth_narrows_secrets_exceptions() -> None:
    source = Path("ops_auth.py").read_text(encoding="utf-8")
    assert "except (AttributeError, KeyError, TypeError)" in source
    assert "except FileNotFoundError" in source
    assert "logger.warning" in source
    assert "except Exception:" not in source.split("def _configured_password")[1].split("def _passwords_match")[0]


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

    query = redact_secrets("https://example.com/v1?key=sk-abc123456789&password=hunter2")
    assert "sk-abc" not in query
    assert "hunter2" not in query
    assert "[redacted]" in query


def test_redacting_filter_scrubs_exc_text() -> None:
    import logging

    from ops_logging import _RedactingFilter

    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="ok",
        args=(),
        exc_info=None,
    )
    record.exc_text = "Traceback: password=hunter2 api_key=sk-abc123456789"
    assert _RedactingFilter().filter(record) is True
    assert "hunter2" not in record.exc_text
    assert "sk-abc" not in record.exc_text
    assert "[redacted]" in record.exc_text


def test_audit_event_redacts_secret_field_values(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CROSS_SECTION_AUDIT_LOG", "nested/redact_audit.log")
    monkeypatch.setattr("ops_audit.user_data_dir", lambda: tmp_path)
    monkeypatch.setattr("paths.user_data_dir", lambda: tmp_path)
    from ops_audit import audit_event

    audit_event("unit_test", note="api_key=sk-abc123456789")
    log_path = tmp_path / "nested" / "redact_audit.log"
    body = log_path.read_text(encoding="utf-8")
    assert "sk-abc" not in body
    assert "[redacted]" in body
    assert "unit_test" in body


def test_apm_before_send_redacts_query_and_message() -> None:
    from ops_apm import _before_send

    event = {
        "message": "password=hunter2 failed",
        "request": {
            "query_string": "key=sk-abc123456789&token=sekrit",
            "headers": {"Authorization": "Bearer tokensecret"},
            "data": {"body": "drop-me"},
            "cookies": {"session": "x"},
        },
    }
    out = _before_send(event, {})
    assert out is not None
    assert "hunter2" not in out["message"]
    assert "[redacted]" in out["message"]
    assert "sk-abc" not in out["request"]["query_string"]
    assert "sekrit" not in out["request"]["query_string"]
    assert out["request"]["headers"]["Authorization"] == "[redacted]"
    assert "data" not in out["request"]
    assert "cookies" not in out["request"]


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


def test_paths_audit_log_rejects_relative_outside(tmp_path, monkeypatch) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setattr("paths.user_data_dir", lambda: allowed)
    monkeypatch.setenv("CROSS_SECTION_AUDIT_LOG", "../outside.log")
    from paths import audit_log_path

    with pytest.raises(PermissionError):
        audit_log_path()
    assert not (tmp_path / "outside.log").exists()


def test_save_lithology_style_rejects_non_hex(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("constants.lithology_styles_path", lambda: tmp_path / "styles.json")
    from constants import save_lithology_style_override

    with pytest.raises(ValueError, match="Invalid lithology color"):
        save_lithology_style_override("Clay", "red; background:url(x)", "..")
    save_lithology_style_override("Clay", "#38220F", "---")
    assert "#38220F" in (tmp_path / "styles.json").read_text(encoding="utf-8")
