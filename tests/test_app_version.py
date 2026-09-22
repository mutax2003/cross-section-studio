"""Tests for product version and notify-only update checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app_version import (
    DEFAULT_UPDATE_MANIFEST_URL,
    about_version_markdown,
    check_for_updates,
    fetch_release_manifest,
    get_version,
    is_newer,
    normalize_version_text,
    parse_semver,
    update_manifest_url,
)


def test_get_version_reads_repo_version_file() -> None:
    version = get_version()
    assert parse_semver(version)
    root_version = (Path(__file__).resolve().parents[1] / "VERSION").read_text(
        encoding="utf-8"
    ).strip()
    assert version == normalize_version_text(root_version)


def test_normalize_version_text_single_leading_v_only() -> None:
    """Must match PowerShell / desktop_updater gate (not str.lstrip('vV'))."""
    assert normalize_version_text("v1.2.3") == "1.2.3"
    assert normalize_version_text("V1.2.3") == "1.2.3"
    assert normalize_version_text("vv1.2.3") == "v1.2.3"
    assert normalize_version_text("  v0.1.1  ") == "0.1.1"


def test_parse_semver_and_compare() -> None:
    assert parse_semver("1.2.3") == (1, 2, 3)
    assert parse_semver("v0.1.0") == (0, 1, 0)
    assert is_newer("0.2.0", "0.1.0")
    assert not is_newer("0.1.0", "0.1.0")
    assert not is_newer("0.0.9", "0.1.0")
    with pytest.raises(ValueError):
        parse_semver("not-a-version")


def test_check_for_updates_from_file_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CROSS_SECTION_ALLOW_DEV_UPDATE", "1")
    monkeypatch.setenv("CROSS_SECTION_UPDATE_MANIFEST_URL", "")
    current = get_version()
    major, minor, patch = parse_semver(current)
    newer = f"{major}.{minor}.{patch + 1}"
    manifest = {
        "version": newer,
        "url": "https://github.com/org/CrossSectionStudio-win64.zip",
        "sha256": "abc123",
        "notes": "Test release",
    }
    path = tmp_path / "release-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    url = path.as_uri()
    result = check_for_updates(url)
    assert result.ok
    assert result.update_available
    assert result.latest_version == newer
    assert result.download_url.endswith(".zip")
    assert result.sha256 == "abc123"


def test_check_for_updates_up_to_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CROSS_SECTION_ALLOW_DEV_UPDATE", "1")
    current = get_version()
    path = tmp_path / "release-manifest.json"
    path.write_text(
        json.dumps({"version": current, "url": "https://github.com/org/app.zip"}),
        encoding="utf-8",
    )
    result = check_for_updates(path.as_uri())
    assert result.ok
    assert not result.update_available
    assert result.latest_version == current


def test_check_for_updates_rejects_non_allowlisted_download_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manifest download URL must pass the same allowlist before UI offers it."""
    monkeypatch.setenv("CROSS_SECTION_ALLOW_DEV_UPDATE", "1")
    monkeypatch.delenv("CROSS_SECTION_UPDATE_URL_ALLOWLIST", raising=False)
    current = get_version()
    major, minor, patch = parse_semver(current)
    newer = f"{major}.{minor}.{patch + 1}"
    path = tmp_path / "release-manifest.json"
    path.write_text(
        json.dumps(
            {
                "version": newer,
                "url": "https://evil.example/malware.zip",
                "sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    result = check_for_updates(path.as_uri())
    assert not result.ok
    assert result.error
    assert "not allowlisted" in result.error
    assert not result.update_available
    assert result.download_url is None


def test_check_for_updates_bad_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CROSS_SECTION_ALLOW_DEV_UPDATE", "1")
    path = tmp_path / "bad.json"
    path.write_text("{not-json", encoding="utf-8")
    result = check_for_updates(path.as_uri())
    assert not result.ok
    assert result.error
    assert not result.update_available


def test_fetch_release_manifest_rejects_http() -> None:
    with pytest.raises(ValueError, match="must use https"):
        fetch_release_manifest("http://github.com/org/release-manifest.json")


def test_fetch_release_manifest_rejects_non_allowlisted_host() -> None:
    with pytest.raises(ValueError, match="not allowlisted"):
        fetch_release_manifest("https://evil.example/release-manifest.json")


def test_fetch_release_manifest_allows_github_https(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"version": "9.9.9", "url": "https://github.com/org/app.zip", "sha256": "ab"}

    class _FakeResponse:
        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(payload).encode("utf-8")

    def fake_urlopen(request: object, *, timeout_s: float = 0.0) -> _FakeResponse:
        assert hasattr(request, "full_url")
        assert "github.com" in str(getattr(request, "full_url", ""))
        return _FakeResponse()

    monkeypatch.setattr("app_version._urlopen_manifest", fake_urlopen)
    got = fetch_release_manifest(
        "https://github.com/mutax2003/cross-section-studio/releases/latest/download/"
        "release-manifest.json"
    )
    assert got == payload


def test_fetch_release_manifest_rejects_file_without_dev_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CROSS_SECTION_ALLOW_DEV_UPDATE", raising=False)
    path = tmp_path / "release-manifest.json"
    path.write_text(json.dumps({"version": "1.0.0"}), encoding="utf-8")
    with pytest.raises(ValueError, match="ALLOW_DEV_UPDATE"):
        fetch_release_manifest(path.as_uri())


def test_fetch_release_manifest_file_uri_with_allow_dev(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CROSS_SECTION_ALLOW_DEV_UPDATE", "1")
    path = tmp_path / "release-manifest.json"
    path.write_text(json.dumps({"version": "1.0.0"}), encoding="utf-8")
    got = fetch_release_manifest(path.as_uri())
    assert got["version"] == "1.0.0"


def test_fetch_release_manifest_accepts_uppercase_file_scheme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scheme matching must be case-insensitive (not startswith('file:'))."""
    monkeypatch.setenv("CROSS_SECTION_ALLOW_DEV_UPDATE", "1")
    path = tmp_path / "release-manifest.json"
    path.write_text(json.dumps({"version": "2.0.0"}), encoding="utf-8")
    # Force uppercase scheme while keeping a valid local path.
    uri = path.as_uri()
    assert uri.startswith("file:")
    upper = "FILE:" + uri[5:]
    got = fetch_release_manifest(upper)
    assert got["version"] == "2.0.0"


def test_validate_manifest_fetch_url_blocks_host_spoofs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adversarial host checks: suffix tricks, userinfo, IPs, empty labels."""
    from app_version import _validate_manifest_fetch_url

    monkeypatch.delenv("CROSS_SECTION_ALLOW_DEV_UPDATE", raising=False)
    monkeypatch.delenv("CROSS_SECTION_UPDATE_URL_ALLOWLIST", raising=False)

    _validate_manifest_fetch_url(
        "https://github.com/org/repo/releases/latest/download/release-manifest.json"
    )
    _validate_manifest_fetch_url(
        "https://objects.githubusercontent.com/github-production-release-asset/1/m.json"
    )
    # Trailing FQDN dot is normalized, not a bypass.
    _validate_manifest_fetch_url("https://github.com./org/m.json")

    rejected = [
        ("https://evilgithub.com/m.json", "not allowlisted"),
        ("https://notgithub.com/m.json", "not allowlisted"),
        ("https://github.com.evil.example/m.json", "not allowlisted"),
        ("https://github.com@evil.example/m.json", "not allowlisted"),
        ("https://140.82.112.3/m.json", "not allowlisted"),
        ("https://127.0.0.1/m.json", "not allowlisted"),
        ("https://[::1]/m.json", "not allowlisted"),
        ("https://.github.com/m.json", "not allowlisted"),
        ("https://..github.com/m.json", "not allowlisted"),
        ("http://github.com/org/repo/m.json", "must use https"),
        ("ftp://github.com/m.json", "must use https"),
        ("file:///C:/tmp/m.json", "ALLOW_DEV_UPDATE"),
    ]
    for url, match in rejected:
        with pytest.raises(ValueError, match=match):
            _validate_manifest_fetch_url(url)


def test_allowlist_single_label_is_exact_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ALLOWLIST=com`` must not authorize every ``*.com`` host."""
    from app_version import _validate_manifest_fetch_url

    monkeypatch.delenv("CROSS_SECTION_ALLOW_DEV_UPDATE", raising=False)
    monkeypatch.setenv("CROSS_SECTION_UPDATE_URL_ALLOWLIST", "com")
    with pytest.raises(ValueError, match="not allowlisted"):
        _validate_manifest_fetch_url("https://evil.com/m.json")
    with pytest.raises(ValueError, match="not allowlisted"):
        _validate_manifest_fetch_url("https://github.com/m.json")
    _validate_manifest_fetch_url("https://com/m.json")


def test_file_url_rejects_dot_and_dotdot_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``rstrip('.')`` must not collapse ``file://./`` / ``file://../`` to empty."""
    from app_version import _validate_manifest_fetch_url

    monkeypatch.setenv("CROSS_SECTION_ALLOW_DEV_UPDATE", "1")
    _validate_manifest_fetch_url("file:///C:/tmp/m.json")
    _validate_manifest_fetch_url("file://localhost/C:/tmp/m.json")
    _validate_manifest_fetch_url("file://localhost./C:/tmp/m.json")
    _validate_manifest_fetch_url("file://127.0.0.1/C:/tmp/m.json")
    _validate_manifest_fetch_url("file://[::1]/C:/tmp/m.json")
    for url in (
        "file://./C:/tmp/m.json",
        "file://../C:/tmp/m.json",
        "file://evil.com/share/m.json",
        "file://github.com/m.json",
    ):
        with pytest.raises(ValueError, match="host not allowed"):
            _validate_manifest_fetch_url(url)


def test_redirect_handler_rejects_evil_hops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise ``_AllowlistedRedirectHandler`` itself (not just validate())."""
    from urllib.parse import urljoin
    from urllib.request import Request

    from app_version import _AllowlistedRedirectHandler

    monkeypatch.delenv("CROSS_SECTION_ALLOW_DEV_UPDATE", raising=False)
    monkeypatch.delenv("CROSS_SECTION_UPDATE_URL_ALLOWLIST", raising=False)

    handler = _AllowlistedRedirectHandler()
    req = Request(
        "https://github.com/org/repo/releases/latest/download/release-manifest.json"
    )

    class _DummyFP:
        def read(self) -> bytes:
            return b""

        def close(self) -> None:
            return None

    # Allowlisted CDN hop must be accepted.
    ok = urljoin(
        req.full_url, "https://objects.githubusercontent.com/asset/release-manifest.json"
    )
    new_req = handler.redirect_request(req, _DummyFP(), 302, "Found", {}, ok)
    assert new_req.full_url == ok

    evil_locations = [
        "https://evil.example/malware.json",
        "//evil.example/malware.json",
        "http://github.com/m.json",
        "https://evilgithub.com/m.json",
        "https://notgithub.com/m.json",
        "https://github.com@evil.example/m.json",
        "https://github.com.evil.example/m.json",
        "file:///C:/evil.json",
        "ftp://github.com/m.json",
    ]
    for loc in evil_locations:
        joined = urljoin(req.full_url, loc)
        with pytest.raises(ValueError):
            handler.redirect_request(req, _DummyFP(), 302, "Found", {}, joined)


def test_about_version_markdown_includes_running_version() -> None:
    text = about_version_markdown()
    assert get_version() in text
    assert "Check for updates" in text


def test_update_manifest_url_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CROSS_SECTION_UPDATE_MANIFEST_URL", "https://example.test/m.json")
    assert update_manifest_url() == "https://example.test/m.json"
    monkeypatch.delenv("CROSS_SECTION_UPDATE_MANIFEST_URL", raising=False)
    assert update_manifest_url() == DEFAULT_UPDATE_MANIFEST_URL


def test_load_help_about_includes_version() -> None:
    from app_menubar import load_help_markdown

    text = load_help_markdown("about")
    assert get_version() in text
    assert "## Version" in text
