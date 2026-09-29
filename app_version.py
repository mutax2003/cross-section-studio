"""Product version and release-manifest update checks (notify-only)."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from update_url_policy import validate_manifest_fetch_url as _validate_manifest_fetch_url
from update_url_policy import validate_update_download_url as _validate_update_download_url

# Default: GitHub Releases asset published beside the Windows zip.
DEFAULT_UPDATE_MANIFEST_URL = (
    "https://github.com/mutax2003/cross-section-studio/releases/latest/download/"
    "release-manifest.json"
)

_SEMVER_RE = re.compile(
    r"^\s*v?(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
    r"(?:[-+][0-9A-Za-z.-]+)?\s*$"
)


MAX_UPDATE_MANIFEST_BYTES = 1024 * 1024  # manifest is a small JSON file


def normalize_version_text(value: str | None) -> str:
    """Strip whitespace and a single optional leading ``v`` / ``V``.

    Must stay aligned with the PowerShell apply script and
    ``desktop_updater`` VERSION gate (``Trim()`` then ``-replace '^[vV]', ''``
    then ``Trim()``). Never use ``str.lstrip('vV')`` — that strips a character
    *set* (``vv1.0`` → ``1.0``), which drifts from the sidecar and can make
    the expected-version gate disagree with the zip ``VERSION`` file.
    """
    text = (value or "").strip()
    if text[:1] in {"v", "V"}:
        text = text[1:].strip()
    return text


def version_file_path() -> Path:
    """Path to the VERSION file (bundled next to app code when frozen)."""
    from paths import app_root

    return app_root() / "VERSION"


def get_version() -> str:
    """Return the installed product version string (semver without leading ``v``)."""
    path = version_file_path()
    if path.is_file():
        text = normalize_version_text(path.read_text(encoding="utf-8"))
        if text:
            return text
    return "0.0.0"


def parse_semver(version: str) -> tuple[int, int, int]:
    """Parse a semver-ish string into ``(major, minor, patch)``."""
    match = _SEMVER_RE.match(str(version or ""))
    if not match:
        raise ValueError(f"Invalid version string: {version!r}")
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
    )


def is_newer(remote: str, local: str) -> bool:
    """True when ``remote`` is a higher semver than ``local``."""
    return parse_semver(remote) > parse_semver(local)


def update_manifest_url() -> str:
    """Manifest URL from env, or the default GitHub Releases location."""
    env = os.environ.get("CROSS_SECTION_UPDATE_MANIFEST_URL", "").strip()
    return env or DEFAULT_UPDATE_MANIFEST_URL


class _AllowlistedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate every redirect hop against the manifest URL policy."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        _validate_manifest_fetch_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _urlopen_manifest(request: urllib.request.Request, *, timeout_s: float):
    """Open ``request`` with redirect hops re-checked against the allowlist."""
    opener = urllib.request.build_opener(_AllowlistedRedirectHandler)
    return opener.open(request, timeout=timeout_s)


@dataclass(frozen=True)
class UpdateCheckResult:
    """Outcome of a notify-only update check."""

    current_version: str
    latest_version: str | None
    update_available: bool
    download_url: str | None
    sha256: str | None
    notes: str | None
    manifest_url: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _normalize_manifest(payload: dict[str, Any], *, manifest_url: str) -> UpdateCheckResult:
    current = get_version()
    latest_raw = payload.get("version")
    if latest_raw is None:
        raise ValueError("manifest missing 'version'")
    latest = normalize_version_text(str(latest_raw))
    if not latest:
        raise ValueError("manifest 'version' is empty")
    download = payload.get("url") or payload.get("download_url")
    download_url = str(download).strip() if download else None
    # Fail closed before Help → Check for updates offers an Open-download link
    # or auto-install: zip hosts must pass the same HTTPS allowlist as downloads.
    if download_url:
        _validate_update_download_url(download_url)
    sha_raw = payload.get("sha256")
    sha256 = str(sha_raw).strip().lower() if sha_raw else None
    notes_raw = payload.get("notes") or payload.get("changelog")
    notes = str(notes_raw).strip() if notes_raw else None
    newer = is_newer(latest, current)
    return UpdateCheckResult(
        current_version=current,
        latest_version=latest,
        update_available=newer,
        download_url=download_url,
        sha256=sha256,
        notes=notes,
        manifest_url=manifest_url,
        error=None,
    )


def fetch_release_manifest(
    url: str | None = None,
    *,
    timeout_s: float = 8.0,
) -> dict[str, Any]:
    """Fetch and parse a JSON release manifest from ``url``.

    Remote URLs must be ``https`` to an allowlisted host; redirects are
    re-validated on each hop. ``file://`` is only for local tests when
    ``CROSS_SECTION_ALLOW_DEV_UPDATE=1`` (same gate as desktop zip downloads).
    """
    manifest_url = (url or update_manifest_url()).strip()
    if not manifest_url:
        raise ValueError("No update manifest URL configured")
    _validate_manifest_fetch_url(manifest_url)
    parsed = urlparse(manifest_url)
    if (parsed.scheme or "").lower() == "file":
        from urllib.parse import unquote
        from urllib.request import url2pathname

        path = Path(url2pathname(unquote(parsed.path)))
        if path.stat().st_size > MAX_UPDATE_MANIFEST_BYTES:
            raise ValueError(
                f"Update manifest too large (> {MAX_UPDATE_MANIFEST_BYTES} bytes)"
            )
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
    else:
        request = urllib.request.Request(
            manifest_url,
            headers={"Accept": "application/json", "User-Agent": "CrossSectionStudio-UpdateCheck"},
            method="GET",
        )
        with _urlopen_manifest(request, timeout_s=timeout_s) as response:  # noqa: S310
            raw = response.read(MAX_UPDATE_MANIFEST_BYTES + 1)
        if len(raw) > MAX_UPDATE_MANIFEST_BYTES:
            raise ValueError(
                f"Update manifest too large (> {MAX_UPDATE_MANIFEST_BYTES} bytes)"
            )
        payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be a JSON object")
    return payload


def check_for_updates(
    url: str | None = None,
    *,
    timeout_s: float = 8.0,
) -> UpdateCheckResult:
    """Compare installed version to the release manifest (notify-only; no download)."""
    manifest_url = (url or update_manifest_url()).strip()
    current = get_version()
    try:
        payload = fetch_release_manifest(manifest_url, timeout_s=timeout_s)
        return _normalize_manifest(payload, manifest_url=manifest_url)
    except urllib.error.HTTPError as exc:
        return UpdateCheckResult(
            current_version=current,
            latest_version=None,
            update_available=False,
            download_url=None,
            sha256=None,
            notes=None,
            manifest_url=manifest_url,
            error=f"HTTP {exc.code}: could not fetch update manifest",
        )
    except urllib.error.URLError as exc:
        return UpdateCheckResult(
            current_version=current,
            latest_version=None,
            update_available=False,
            download_url=None,
            sha256=None,
            notes=None,
            manifest_url=manifest_url,
            error=f"Network error: {exc.reason}",
        )
    except (OSError, ValueError, json.JSONDecodeError, TypeError, RecursionError) as exc:
        return UpdateCheckResult(
            current_version=current,
            latest_version=None,
            update_available=False,
            download_url=None,
            sha256=None,
            notes=None,
            manifest_url=manifest_url,
            error=str(exc),
        )


def about_version_markdown() -> str:
    """Markdown block appended to Help → About."""
    version = get_version()
    return (
        "## Version\n\n"
        f"Running **{version}**.\n\n"
        "Use **Help → Check for updates** to compare against the published release "
        "manifest. On the Windows desktop build you can open the download page or use "
        "**Download and install (restart)** (full zip, SHA-256 verified, sidecar replace).\n"
    )
