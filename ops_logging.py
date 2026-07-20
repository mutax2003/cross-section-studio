"""Structured logging configuration for Cross Section Studio."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import UTC, datetime

_SECRET_PATTERNS = (
    (re.compile(r"(?i)\b(api[_-]?key|password|secret|token)\s*[=:]\s*\S+"), r"\1=[redacted]"),
    (re.compile(r"(?i)\bauthorization\s*[=:]\s*\S+"), "authorization=[redacted]"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+"), "Bearer [redacted]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"), "sk-[redacted]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{10,}"), "AIza[redacted]"),
    (re.compile(r"(?i)(x-goog-api-key\s*[:=]\s*)\S+"), r"\1[redacted]"),
)


def redact_secrets(text: str) -> str:
    """Best-effort scrub of credentials from log / error text."""
    redacted = text
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


class _RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        redacted = redact_secrets(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_secrets(record.getMessage()),
        }
        if record.exc_info:
            payload["exc_info"] = redact_secrets(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    level_name = os.environ.get("CROSS_SECTION_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        if not any(isinstance(item, _RedactingFilter) for item in root.filters):
            root.addFilter(_RedactingFilter())
        return
    handler = logging.StreamHandler(sys.stdout)
    if os.environ.get("CROSS_SECTION_LOG_FORMAT", "").lower() == "json":
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
    handler.addFilter(_RedactingFilter())
    root.addHandler(handler)
    root.addFilter(_RedactingFilter())
    root.setLevel(level)
