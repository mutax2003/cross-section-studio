"""Optional Sentry APM hook (enabled when SENTRY_DSN is set)."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _traces_sample_rate() -> float:
    raw = os.environ.get("SENTRY_TRACES_RATE", "0.1").strip() or "0.1"
    try:
        rate = float(raw)
    except ValueError:
        logger.warning("Invalid SENTRY_TRACES_RATE=%r; using 0.1", raw)
        return 0.1
    return max(0.0, min(rate, 1.0))


def _before_send(event: dict[str, Any], _hint: dict[str, Any]) -> dict[str, Any] | None:
    """Drop default PII-ish request bodies; keep stack frames for ops."""
    request = event.get("request")
    if isinstance(request, dict):
        request.pop("data", None)
        request.pop("cookies", None)
        headers = request.get("headers")
        if isinstance(headers, dict):
            for key in list(headers):
                lowered = str(key).lower()
                if lowered in {"authorization", "cookie", "x-api-key", "x-goog-api-key"}:
                    headers[key] = "[redacted]"
    return event


def init_apm() -> None:
    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        return
    try:
        import sentry_sdk
    except ImportError:
        logger.info("SENTRY_DSN set but sentry-sdk not installed; skipping APM")
        return
    sentry_sdk.init(
        dsn=dsn,
        traces_sample_rate=_traces_sample_rate(),
        send_default_pii=False,
        before_send=_before_send,
        environment=os.environ.get("SENTRY_ENVIRONMENT", "production"),
    )
    logger.info("Sentry APM initialized")
