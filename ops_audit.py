"""Append-only audit trail for uploads and exports."""

from __future__ import annotations

import json
import logging
import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paths import audit_log_path, user_data_dir

logger = logging.getLogger(__name__)


def _resolved_audit_path() -> Path:
    """Resolve audit path; refuse overrides outside the user data directory."""
    path = audit_log_path().expanduser()
    allowed_root = user_data_dir().resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        override = os.environ.get("CROSS_SECTION_AUDIT_LOG", "").strip()
        if override:
            raise PermissionError(
                f"CROSS_SECTION_AUDIT_LOG must stay under {allowed_root}"
            ) from exc
        raise
    return resolved


def audit_event(event: str, **fields: Any) -> None:
    """Record a JSON line audit event (best-effort)."""
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "event": event,
        **fields,
    }
    try:
        path = _resolved_audit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        created = not path.exists()
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if created:
            try:
                os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                # Windows may ignore restrictive chmod; still append-only JSONL.
                pass
    except (OSError, PermissionError, TypeError, ValueError) as exc:
        logger.warning("audit log write failed: %s", exc)
