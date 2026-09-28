"""Thin tests for shared update URL host allowlist policy."""

from __future__ import annotations

import pytest

from update_url_policy import (
    DEFAULT_UPDATE_URL_HOST_ALLOWLIST,
    host_matches_allowlist,
    update_url_host_allowlist,
    validate_manifest_fetch_url,
    validate_update_download_url,
)


def test_default_allowlist_hosts() -> None:
    assert "github.com" in DEFAULT_UPDATE_URL_HOST_ALLOWLIST
    assert "githubusercontent.com" in DEFAULT_UPDATE_URL_HOST_ALLOWLIST


def test_host_matches_exact_and_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CROSS_SECTION_UPDATE_URL_ALLOWLIST", raising=False)
    allow = update_url_host_allowlist()
    assert host_matches_allowlist("github.com", allow)
    assert host_matches_allowlist("objects.githubusercontent.com", allow)
    assert not host_matches_allowlist("evilgithub.com", allow)
    assert not host_matches_allowlist("github.com.evil.example", allow)


def test_env_allowlist_replaces_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CROSS_SECTION_UPDATE_URL_ALLOWLIST", "cdn.example.com")
    allow = update_url_host_allowlist()
    assert allow == ("cdn.example.com",)
    assert host_matches_allowlist("cdn.example.com", allow)
    assert not host_matches_allowlist("github.com", allow)


def test_single_label_allowlist_is_exact_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CROSS_SECTION_UPDATE_URL_ALLOWLIST", "com")
    allow = update_url_host_allowlist()
    assert host_matches_allowlist("com", allow)
    assert not host_matches_allowlist("evil.com", allow)


def test_validate_shims_share_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CROSS_SECTION_ALLOW_DEV_UPDATE", raising=False)
    monkeypatch.delenv("CROSS_SECTION_UPDATE_URL_ALLOWLIST", raising=False)
    validate_update_download_url("https://github.com/org/a.zip")
    validate_manifest_fetch_url("https://github.com/org/m.json")
    with pytest.raises(ValueError, match="not allowlisted"):
        validate_update_download_url("https://evil.example/a.zip")
    with pytest.raises(ValueError, match="not allowlisted"):
        validate_manifest_fetch_url("https://evil.example/m.json")
