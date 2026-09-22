"""Shared HTTPS / file:// allowlist policy for update manifest and zip URLs.

Leaf module: no imports from ``app_version``, ``desktop_updater``, Streamlit,
or engine modules. Both callers re-export public validators for monkeypatch
compat.
"""

from __future__ import annotations

import os
from typing import Literal
from urllib.parse import urlparse

# Default fetch/download hosts (exact match, or DNS suffix match for multi-label
# entries). ``githubusercontent.com`` covers objects/release-assets/raw/gist CDN hosts.
DEFAULT_UPDATE_URL_HOST_ALLOWLIST: tuple[str, ...] = (
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
    "githubusercontent.com",
)

_UrlKind = Literal["download", "manifest"]


def env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def normalize_url_hostname(host: str | None) -> str:
    """Lowercase host and strip at most one trailing FQDN dot.

    Do **not** use ``str.rstrip('.')``: that collapses ``.`` / ``..`` (and any
    all-dot host) to empty, which previously made ``file://./…`` and
    ``file://../…`` look like authority-less local paths.
    """
    text = (host or "").strip().lower()
    if len(text) > 1 and text.endswith(".") and text.strip("."):
        text = text[:-1]
    return text


def _is_ip_literal_host(host: str) -> bool:
    """True when ``host`` is an IPv4/IPv6 literal (no brackets)."""
    import ipaddress

    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _dns_hostname_labels_ok(host: str) -> bool:
    """Reject empty DNS labels (``.github.com``, ``a..b.com``, trailing junk)."""
    if not host or host.startswith(".") or ".." in host:
        return False
    return all(label for label in host.split("."))


def _normalize_allowlist_entry(entry: str) -> str:
    """Normalize one allowlist entry (lowercase, one trailing FQDN dot)."""
    return normalize_url_hostname(entry.strip())


def update_url_host_allowlist() -> tuple[str, ...]:
    """Hosts allowed for update URL fetches (lowercase, no ports).

    Semantics for ``CROSS_SECTION_UPDATE_URL_ALLOWLIST``: when set to a
    non-empty comma-separated list, **that list replaces the defaults**
    (does not merge). When unset or blank, the built-in GitHub allowlist
    is used.

    Matching: exact host match always. DNS suffix match
    (``host.endswith("." + entry)``) only when ``entry`` contains a dot
    (two or more labels), so a footgun like ``ALLOWLIST=com`` cannot
    authorize every ``*.com`` host. IP literals match exactly only.
    """
    raw = os.environ.get("CROSS_SECTION_UPDATE_URL_ALLOWLIST", "").strip()
    if raw:
        hosts = tuple(
            h
            for h in (_normalize_allowlist_entry(part) for part in raw.split(","))
            if h
        )
        if hosts:
            return hosts
    return DEFAULT_UPDATE_URL_HOST_ALLOWLIST


def host_matches_allowlist(host: str, allowlist: tuple[str, ...]) -> bool:
    host = normalize_url_hostname(host)
    if not host:
        return False
    if _is_ip_literal_host(host):
        return any(host == entry for entry in allowlist if entry)
    if not _dns_hostname_labels_ok(host):
        return False
    for entry in allowlist:
        if not entry or _is_ip_literal_host(entry):
            # Domain host vs IP allowlist entry: never suffix-match.
            if entry and host == entry:
                return True
            continue
        if not _dns_hostname_labels_ok(entry):
            continue
        if host == entry:
            return True
        # Suffix match only for multi-label entries (blocks ALLOWLIST=com).
        if "." in entry and host.endswith("." + entry):
            return True
    return False


def validate_update_url(url: str, *, kind: _UrlKind = "download") -> None:
    """Raise ``ValueError`` if ``url`` is not an allowed update URL target.

    Production policy: ``https`` only, host must match the allowlist (see
    :func:`update_url_host_allowlist`). ``urlparse(...).hostname`` is used
    so ``userinfo@host`` cannot spoof the allowlist (``github.com@evil`` →
    ``evil``).

    Dev/test exception: ``file://`` is accepted only when
    ``CROSS_SECTION_ALLOW_DEV_UPDATE`` is enabled and the URL host is empty
    or a loopback name (``localhost`` / ``127.0.0.1`` / ``::1``). Weird
    authorities such as ``.`` / ``..`` are rejected. Frozen production
    installs never set that flag, so ``file://`` stays rejected — including
    as a redirect hop (blocks HTTPS→file escapes).

    ``kind`` selects error-message wording (``download`` vs ``manifest``)
    so callers keep their historical phrasing.
    """
    label = "download" if kind == "download" else "manifest"
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme == "file":
        if not env_flag_enabled("CROSS_SECTION_ALLOW_DEV_UPDATE"):
            raise ValueError(
                "file:// update URLs require CROSS_SECTION_ALLOW_DEV_UPDATE=1"
            )
        # hostname may raise on malformed IPv6 netloc — treat as rejected.
        try:
            host = normalize_url_hostname(parsed.hostname)
        except ValueError as exc:
            raise ValueError(
                f"file:// update URL host not allowed for local downloads: {exc}"
            ) from exc
        if host and host not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError(
                f"file:// update URL host not allowed for local downloads: {host!r}"
            )
        return
    if scheme != "https":
        raise ValueError(
            f"Update {label} URL must use https (got {scheme or 'missing'}://)"
        )
    try:
        host = normalize_url_hostname(parsed.hostname)
    except ValueError as exc:
        raise ValueError(f"Update {label} URL has invalid host ({exc})") from exc
    if not host:
        raise ValueError(f"Update {label} URL is missing a host")
    allowlist = update_url_host_allowlist()
    if not host_matches_allowlist(host, allowlist):
        raise ValueError(f"Update {label} host not allowlisted: {host}")


def validate_update_download_url(url: str) -> None:
    """Raise ``ValueError`` if ``url`` is not an allowed update download target."""
    validate_update_url(url, kind="download")


def validate_manifest_fetch_url(url: str) -> None:
    """Raise ``ValueError`` if ``url`` is not an allowed manifest fetch target."""
    validate_update_url(url, kind="manifest")
