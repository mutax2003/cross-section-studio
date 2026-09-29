"""Tests for full-zip desktop updater (download verify + apply swap)."""

from __future__ import annotations

import hashlib
import io
import shutil
import zipfile
from pathlib import Path

import pytest

from app_version import UpdateCheckResult
from desktop_updater import (
    APPLY_WAIT_PID_TIMEOUT_S,
    _acquire_apply_lock,
    _extract_zip_to,
    _normalize_version_text,
    _release_apply_lock,
    _require_expected_version,
    _sha256_file,
    _write_apply_script,
    apply_lock_path,
    apply_log_path,
    apply_update_from_zip,
    auto_install_allowed,
    download_update_zip,
    schedule_sidecar_apply,
    updates_dir,
)

TEST_VERSION = "0.1.1"


def _make_zip(
    path: Path,
    *,
    with_wrapper: bool = False,
    version: str | None = TEST_VERSION,
    extra_members: dict[str, bytes] | None = None,
) -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        if with_wrapper:
            archive.writestr("CrossSectionStudio/CrossSectionStudio.exe", b"MZ-new")
            archive.writestr("CrossSectionStudio/_internal/marker.txt", b"ok")
            if version is not None:
                archive.writestr("CrossSectionStudio/VERSION", version.encode("utf-8"))
        else:
            archive.writestr("CrossSectionStudio.exe", b"MZ-new")
            archive.writestr("_internal/marker.txt", b"ok")
            if version is not None:
                archive.writestr("VERSION", version.encode("utf-8"))
        if extra_members:
            for name, data in extra_members.items():
                archive.writestr(name, data)
    data = buf.getvalue()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_auto_install_allowed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CROSS_SECTION_ALLOW_DEV_UPDATE", raising=False)
    assert auto_install_allowed() is False
    monkeypatch.setenv("CROSS_SECTION_ALLOW_DEV_UPDATE", "1")
    assert auto_install_allowed() is True


def test_extract_flat_and_wrapped(tmp_path: Path) -> None:
    flat = tmp_path / "flat.zip"
    _make_zip(flat, with_wrapper=False)
    staging = tmp_path / "staging-flat"
    _extract_zip_to(staging, flat)
    assert (staging / "CrossSectionStudio.exe").is_file()
    assert (staging / "VERSION").read_text(encoding="utf-8") == TEST_VERSION

    wrapped = tmp_path / "wrapped.zip"
    _make_zip(wrapped, with_wrapper=True)
    staging2 = tmp_path / "staging-wrap"
    _extract_zip_to(staging2, wrapped)
    assert (staging2 / "CrossSectionStudio.exe").is_file()
    assert (staging2 / "_internal" / "marker.txt").is_file()
    assert (staging2 / "VERSION").read_text(encoding="utf-8") == TEST_VERSION


def test_extract_rejects_zip_slip(tmp_path: Path) -> None:
    evil_names = (
        "../outside.txt",
        r"..\outside.txt",
        "foo/../../outside.txt",
        "foo/../bar.txt",
        "/tmp/evil.txt",
        "C:/Windows/evil.txt",
        r"C:\Windows\evil.txt",
    )
    for evil_name in evil_names:
        evil = tmp_path / "evil.zip"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            archive.writestr("CrossSectionStudio.exe", b"MZ-new")
            archive.writestr("VERSION", TEST_VERSION.encode("utf-8"))
            archive.writestr(evil_name, b"pwned")
        evil.write_bytes(buf.getvalue())

        staging = tmp_path / f"staging-slip-{hash(evil_name) & 0xFFFF:x}"
        with pytest.raises(ValueError, match="Unsafe zip member"):
            _extract_zip_to(staging, evil)
        assert not (tmp_path / "outside.txt").exists()


def test_extract_rejects_symlink_member(tmp_path: Path) -> None:
    evil = tmp_path / "sym.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("CrossSectionStudio.exe", b"MZ-new")
        archive.writestr("VERSION", TEST_VERSION.encode("utf-8"))
        info = zipfile.ZipInfo("linkdir")
        info.create_system = 3
        info.external_attr = 0o120777 << 16
        archive.writestr(info, b"/tmp/outside")
    evil.write_bytes(buf.getvalue())
    with pytest.raises(ValueError, match="Unsafe zip member"):
        _extract_zip_to(tmp_path / "staging-sym", evil)


def test_normalize_version_matches_powershell_single_v() -> None:
    assert _normalize_version_text("v1.2.3") == "1.2.3"
    assert _normalize_version_text("V1.2.3") == "1.2.3"
    # Single leading v only (PS -replace '^[vV]'); not str.lstrip('vV').
    assert _normalize_version_text("vv1.2.3") == "v1.2.3"
    assert _normalize_version_text("  v0.1.1  ") == "0.1.1"
    with pytest.raises(ValueError, match="expected_version is required"):
        _require_expected_version("")
    with pytest.raises(ValueError, match="expected_version is required"):
        _require_expected_version("v")
    with pytest.raises(ValueError, match="expected_version is required"):
        _require_expected_version(None)


def test_write_apply_script_rejects_blank_expected_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    monkeypatch.setattr("paths.is_frozen", lambda: True)
    zip_path = tmp_path / "update.zip"
    _make_zip(zip_path)
    install = tmp_path / "CrossSectionStudio"
    install.mkdir()
    for blank in ("", "v", "   "):
        with pytest.raises(ValueError, match="expected_version is required"):
            _write_apply_script(
                zip_path=zip_path,
                target_install_dir=install,
                wait_pid=0,
                relaunch=False,
                expected_version=blank,
            )


def test_schedule_sidecar_rejects_blank_expected_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    monkeypatch.setattr("paths.is_frozen", lambda: True)
    zip_path = tmp_path / "update.zip"
    _make_zip(zip_path)
    with pytest.raises(ValueError, match="expected_version is required"):
        schedule_sidecar_apply(
            zip_path, expected_version="", install_root=tmp_path / "inst"
        )


def test_apply_rejects_version_only_under_wrapper_when_not_unwrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VERSION nested under a non-unwrapped folder must not satisfy the gate."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    monkeypatch.setattr("paths.is_frozen", lambda: True)
    install = tmp_path / "CrossSectionStudio"
    install.mkdir()
    (install / "CrossSectionStudio.exe").write_bytes(b"MZ-old")

    zip_path = tmp_path / "nested-only.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("CrossSectionStudio.exe", b"MZ-new")
        archive.writestr("nested/VERSION", TEST_VERSION.encode("utf-8"))
    zip_path.write_bytes(buf.getvalue())

    with pytest.raises(FileNotFoundError, match="missing VERSION"):
        apply_update_from_zip(
            zip_path,
            install,
            expected_version=TEST_VERSION,
            wait_pid=0,
            relaunch=False,
        )
    assert (install / "CrossSectionStudio.exe").read_bytes() == b"MZ-old"


def test_apply_update_swaps_install_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    monkeypatch.setattr("paths.is_frozen", lambda: True)
    install = tmp_path / "CrossSectionStudio"
    install.mkdir()
    (install / "CrossSectionStudio.exe").write_bytes(b"MZ-old")
    (install / "old.txt").write_text("old", encoding="utf-8")

    zip_path = tmp_path / "update.zip"
    _make_zip(zip_path)
    apply_update_from_zip(
        zip_path, install, expected_version=TEST_VERSION, wait_pid=0, relaunch=False
    )

    assert (install / "CrossSectionStudio.exe").read_bytes() == b"MZ-new"
    assert (install / "_internal" / "marker.txt").read_text(encoding="utf-8") == "ok"
    assert (install / "VERSION").read_text(encoding="utf-8") == TEST_VERSION
    backups = list(tmp_path.glob("CrossSectionStudio.bak-*"))
    assert len(backups) == 1
    assert (backups[0] / "old.txt").is_file()
    log_text = apply_log_path().read_text(encoding="utf-8")
    assert "SUCCESS:" in log_text
    assert str(install.resolve()) in log_text
    assert not apply_lock_path().exists()


def test_apply_update_rejects_version_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    monkeypatch.setattr("paths.is_frozen", lambda: True)
    install = tmp_path / "CrossSectionStudio"
    install.mkdir()
    (install / "CrossSectionStudio.exe").write_bytes(b"MZ-old")
    (install / "old.txt").write_text("old", encoding="utf-8")

    zip_path = tmp_path / "update.zip"
    _make_zip(zip_path, version="9.9.9")
    with pytest.raises(ValueError, match="VERSION mismatch"):
        apply_update_from_zip(
            zip_path,
            install,
            expected_version=TEST_VERSION,
            wait_pid=0,
            relaunch=False,
        )

    assert (install / "CrossSectionStudio.exe").read_bytes() == b"MZ-old"
    assert (install / "old.txt").read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob("CrossSectionStudio.bak-*"))
    log_text = apply_log_path().read_text(encoding="utf-8")
    assert "FAILURE:" in log_text
    assert "VERSION mismatch" in log_text
    assert not apply_lock_path().exists()


def test_apply_update_mid_swap_restores_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If staging→install fails after backup, restore install and log the reason."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    monkeypatch.setattr("paths.is_frozen", lambda: True)
    install = tmp_path / "CrossSectionStudio"
    install.mkdir()
    (install / "CrossSectionStudio.exe").write_bytes(b"MZ-old")
    (install / "old.txt").write_text("old", encoding="utf-8")

    zip_path = tmp_path / "update.zip"
    _make_zip(zip_path)

    real_move = shutil.move
    move_calls = {"n": 0}

    def flaky_move(src: str, dst: str, *args: object, **kwargs: object) -> str:
        move_calls["n"] += 1
        # 1st move: install → backup; 2nd: staging → install (simulate failure).
        if move_calls["n"] == 2:
            raise OSError("simulated mid-swap failure")
        return real_move(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "move", flaky_move)

    with pytest.raises(OSError, match="simulated mid-swap failure"):
        apply_update_from_zip(
            zip_path,
            install,
            expected_version=TEST_VERSION,
            wait_pid=0,
            relaunch=False,
        )

    assert install.is_dir()
    assert (install / "CrossSectionStudio.exe").read_bytes() == b"MZ-old"
    assert (install / "old.txt").read_text(encoding="utf-8") == "old"
    assert not (install / "_internal").exists()

    log_text = apply_log_path().read_text(encoding="utf-8")
    assert "FAILURE:" in log_text
    assert "simulated mid-swap failure" in log_text
    assert "restored backup" in log_text
    assert not apply_lock_path().exists()


def test_apply_update_partial_install_still_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partial staging→install that leaves install present must still roll back."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    monkeypatch.setattr("paths.is_frozen", lambda: True)
    install = tmp_path / "CrossSectionStudio"
    install.mkdir()
    (install / "CrossSectionStudio.exe").write_bytes(b"MZ-old")
    (install / "old.txt").write_text("old", encoding="utf-8")

    zip_path = tmp_path / "update.zip"
    _make_zip(zip_path)

    real_move = shutil.move
    move_calls = {"n": 0}

    def partial_then_fail(src: str, dst: str, *args: object, **kwargs: object) -> str:
        move_calls["n"] += 1
        if move_calls["n"] == 2:
            # Simulate a non-atomic move that creates a broken install tree.
            dst_path = Path(dst)
            dst_path.mkdir(parents=True, exist_ok=True)
            (dst_path / "CrossSectionStudio.exe").write_bytes(b"MZ-partial")
            raise OSError("simulated partial staging move")
        return real_move(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "move", partial_then_fail)

    with pytest.raises(OSError, match="simulated partial staging move"):
        apply_update_from_zip(
            zip_path,
            install,
            expected_version=TEST_VERSION,
            wait_pid=0,
            relaunch=False,
        )

    assert install.is_dir()
    assert (install / "CrossSectionStudio.exe").read_bytes() == b"MZ-old"
    assert (install / "old.txt").read_text(encoding="utf-8") == "old"
    log_text = apply_log_path().read_text(encoding="utf-8")
    assert "FAILURE:" in log_text
    assert "restored backup" in log_text
    broken = list(tmp_path.glob("CrossSectionStudio.broken-*"))
    assert len(broken) == 1
    assert (broken[0] / "CrossSectionStudio.exe").read_bytes() == b"MZ-partial"


def test_apply_update_restore_failure_uses_staging_degraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If backup→install fails, place staging so the user is not left with no install."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    monkeypatch.setattr("paths.is_frozen", lambda: True)
    install = tmp_path / "CrossSectionStudio"
    install.mkdir()
    (install / "CrossSectionStudio.exe").write_bytes(b"MZ-old")

    zip_path = tmp_path / "update.zip"
    _make_zip(zip_path)

    real_move = shutil.move
    move_calls = {"n": 0}

    def fail_second_and_restore(src: str, dst: str, *args: object, **kwargs: object) -> str:
        move_calls["n"] += 1
        src_path = Path(src)
        # 1: install→backup OK; 2: staging→install fails (staging remains);
        # 3: backup→install restore fails; 4+: allow staging→install degraded.
        if move_calls["n"] == 2:
            raise OSError("simulated mid-swap failure")
        if move_calls["n"] == 3 and src_path.name.startswith("CrossSectionStudio.bak-"):
            raise OSError("simulated restore failure")
        return real_move(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "move", fail_second_and_restore)

    with pytest.raises(OSError, match="simulated mid-swap failure"):
        apply_update_from_zip(
            zip_path,
            install,
            expected_version=TEST_VERSION,
            wait_pid=0,
            relaunch=False,
        )

    assert install.is_dir()
    assert (install / "CrossSectionStudio.exe").read_bytes() == b"MZ-new"
    log_text = apply_log_path().read_text(encoding="utf-8")
    assert "FAILURE:" in log_text
    assert "degraded recovery" in log_text
    assert "simulated restore failure" in log_text


def test_apply_log_path_absolute_when_localappdata_blank_or_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("paths.is_frozen", lambda: True)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    expected = (
        home / "AppData" / "Local" / "CrossSectionStudio" / "updates" / "apply_update.log"
    ).resolve()

    monkeypatch.setenv("LOCALAPPDATA", "")
    blank = apply_log_path()
    assert blank == expected
    assert blank.is_absolute()
    assert updates_dir() == expected.parent

    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    unset = apply_log_path()
    assert unset == expected


def test_concurrent_apply_rejected_by_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    monkeypatch.setattr("paths.is_frozen", lambda: True)
    install = tmp_path / "CrossSectionStudio"
    install.mkdir()
    (install / "CrossSectionStudio.exe").write_bytes(b"MZ-old")
    zip_path = tmp_path / "update.zip"
    _make_zip(zip_path)

    held = _acquire_apply_lock()
    try:
        with pytest.raises(RuntimeError, match="already in progress"):
            apply_update_from_zip(
                zip_path,
                install,
                expected_version=TEST_VERSION,
                wait_pid=0,
                relaunch=False,
            )
        assert (install / "CrossSectionStudio.exe").read_bytes() == b"MZ-old"
    finally:
        _release_apply_lock(held)

    apply_update_from_zip(
        zip_path, install, expected_version=TEST_VERSION, wait_pid=0, relaunch=False
    )
    assert (install / "CrossSectionStudio.exe").read_bytes() == b"MZ-new"

    # PID liveness probes must not poison Windows cert-store enumeration used by urllib.
    import urllib.request

    opener = urllib.request.build_opener()
    assert opener is not None


def test_write_apply_script_has_rollback_and_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    monkeypatch.setattr("paths.is_frozen", lambda: True)
    zip_path = tmp_path / "update.zip"
    _make_zip(zip_path)
    install = tmp_path / "CrossSectionStudio"
    install.mkdir()
    script = _write_apply_script(
        zip_path=zip_path,
        target_install_dir=install,
        wait_pid=0,
        relaunch=False,
        expected_version=TEST_VERSION,
    )
    body = script.read_text(encoding="utf-8")
    assert "try {" in body
    assert "catch {" in body
    assert "$backedUp" in body
    assert "Restore-BackupToInstall" in body
    assert "Quarantine-Path" in body
    assert "apply_update.log" in body
    assert "Write-ApplyLog" in body
    assert "Acquire-ApplyLock" in body
    assert "apply_update.lock" in body
    assert str(apply_lock_path()).replace("\\", "\\\\") in body or str(
        apply_lock_path()
    ) in body
    # PS/Python wait timeout must not drift.
    assert f"$waitTimeoutS = {int(APPLY_WAIT_PID_TIMEOUT_S)}" in body
    assert "Test-PidAlive" in body
    assert "$TargetPid" in body
    assert "degraded recovery" in body
    assert "Expand-ZipConfined" in body
    assert "Unsafe zip member path" in body
    assert "expected_version is required" in body
    assert f"$expectedVersion = '{TEST_VERSION}'" in body
    assert "VERSION mismatch after extract" in body
    # Segment-level .. rejection must stay in the PS sidecar (parity with Python).
    assert "$part -eq '..'" in body or '$part -eq ".."' in body


def test_download_update_zip_verifies_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    monkeypatch.setenv("CROSS_SECTION_ALLOW_DEV_UPDATE", "1")
    # Force frozen-style user_data_dir under LOCALAPPDATA for updates_dir.
    monkeypatch.setattr("paths.is_frozen", lambda: True)

    zip_path = tmp_path / "payload.zip"
    digest = _make_zip(zip_path)
    url = zip_path.as_uri()
    result = UpdateCheckResult(
        current_version="0.1.0",
        latest_version="0.1.1",
        update_available=True,
        download_url=url,
        sha256=digest,
        notes=None,
        manifest_url="file://manifest",
    )
    downloaded = download_update_zip(result, timeout_s=5.0)
    assert downloaded.zip_path.is_file()
    assert downloaded.sha256 == digest
    assert _sha256_file(downloaded.zip_path) == digest


def test_download_update_zip_rejects_bad_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    monkeypatch.setenv("CROSS_SECTION_ALLOW_DEV_UPDATE", "1")
    monkeypatch.setattr("paths.is_frozen", lambda: True)
    zip_path = tmp_path / "payload.zip"
    _make_zip(zip_path)
    result = UpdateCheckResult(
        current_version="0.1.0",
        latest_version="0.1.1",
        update_available=True,
        download_url=zip_path.as_uri(),
        sha256="0" * 64,
        notes=None,
        manifest_url="file://manifest",
    )
    with pytest.raises(ValueError, match="SHA-256"):
        download_update_zip(result, timeout_s=5.0)


def test_download_update_zip_rejects_blank_latest_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    monkeypatch.setenv("CROSS_SECTION_ALLOW_DEV_UPDATE", "1")
    monkeypatch.setattr("paths.is_frozen", lambda: True)
    zip_path = tmp_path / "payload.zip"
    digest = _make_zip(zip_path)
    for blank in (None, "", "v", "   "):
        result = UpdateCheckResult(
            current_version="0.1.0",
            latest_version=blank,  # type: ignore[arg-type]
            update_available=True,
            download_url=zip_path.as_uri(),
            sha256=digest,
            notes=None,
            manifest_url="file://manifest",
        )
        with pytest.raises(ValueError, match="expected_version is required"):
            download_update_zip(result, timeout_s=5.0)


def test_download_update_zip_rejects_missing_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    monkeypatch.setenv("CROSS_SECTION_ALLOW_DEV_UPDATE", "1")
    monkeypatch.setattr("paths.is_frozen", lambda: True)
    zip_path = tmp_path / "payload.zip"
    _make_zip(zip_path)
    for blank in (None, "", "   "):
        result = UpdateCheckResult(
            current_version="0.1.0",
            latest_version="0.1.1",
            update_available=True,
            download_url=zip_path.as_uri(),
            sha256=blank,
            notes=None,
            manifest_url="file://manifest",
        )
        with pytest.raises(ValueError, match="SHA-256 is required"):
            download_update_zip(result, timeout_s=5.0)


def test_download_update_zip_rejects_non_hex_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    monkeypatch.setenv("CROSS_SECTION_ALLOW_DEV_UPDATE", "1")
    monkeypatch.setattr("paths.is_frozen", lambda: True)
    zip_path = tmp_path / "payload.zip"
    _make_zip(zip_path)
    for bad in ("deadbeef", "g" * 64, "0" * 63, "0" * 65):
        result = UpdateCheckResult(
            current_version="0.1.0",
            latest_version="0.1.1",
            update_available=True,
            download_url=zip_path.as_uri(),
            sha256=bad,
            notes=None,
            manifest_url="file://manifest",
        )
        with pytest.raises(ValueError, match="64-character"):
            download_update_zip(result, timeout_s=5.0)


def test_download_update_zip_rejects_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    monkeypatch.delenv("CROSS_SECTION_ALLOW_DEV_UPDATE", raising=False)
    monkeypatch.setattr("paths.is_frozen", lambda: True)
    result = UpdateCheckResult(
        current_version="0.1.0",
        latest_version="0.1.1",
        update_available=True,
        download_url="http://github.com/org/repo/releases/download/v0.1.1/app.zip",
        sha256="a" * 64,
        notes=None,
        manifest_url="https://example.com/manifest.json",
    )
    with pytest.raises(ValueError, match="must use https"):
        download_update_zip(result, timeout_s=5.0)


def test_download_update_zip_rejects_non_allowlisted_https(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    monkeypatch.delenv("CROSS_SECTION_ALLOW_DEV_UPDATE", raising=False)
    monkeypatch.delenv("CROSS_SECTION_UPDATE_URL_ALLOWLIST", raising=False)
    monkeypatch.setattr("paths.is_frozen", lambda: True)
    result = UpdateCheckResult(
        current_version="0.1.0",
        latest_version="0.1.1",
        update_available=True,
        download_url="https://evil.example/malware.zip",
        sha256="a" * 64,
        notes=None,
        manifest_url="https://example.com/manifest.json",
    )
    with pytest.raises(ValueError, match="not allowlisted"):
        download_update_zip(result, timeout_s=5.0)


def test_download_update_zip_allows_github_https(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    monkeypatch.delenv("CROSS_SECTION_ALLOW_DEV_UPDATE", raising=False)
    monkeypatch.delenv("CROSS_SECTION_UPDATE_URL_ALLOWLIST", raising=False)
    monkeypatch.setattr("paths.is_frozen", lambda: True)

    zip_path = tmp_path / "payload.zip"
    digest = _make_zip(zip_path)
    payload = zip_path.read_bytes()

    class _FakeResponse:
        def __init__(self, data: bytes) -> None:
            self._buf = io.BytesIO(data)
            self.headers = {"Content-Length": str(len(data))}

        def read(self, size: int = -1) -> bytes:
            return self._buf.read(size)

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(request: object, *, timeout_s: float = 0.0) -> _FakeResponse:
        assert timeout_s > 0
        url = getattr(request, "full_url", None) or str(request)
        assert "objects.githubusercontent.com" in url
        return _FakeResponse(payload)

    monkeypatch.setattr("desktop_updater._urlopen_update", fake_urlopen)

    result = UpdateCheckResult(
        current_version="0.1.0",
        latest_version="0.1.1",
        update_available=True,
        download_url=(
            "https://objects.githubusercontent.com/github-production-release-asset/"
            "123/CrossSectionStudio-win64-v0.1.1.zip"
        ),
        sha256=digest,
        notes=None,
        manifest_url="https://github.com/org/repo/releases/latest/download/manifest.json",
    )
    downloaded = download_update_zip(result, timeout_s=5.0)
    assert downloaded.sha256 == digest
    assert downloaded.zip_path.is_file()


def test_download_update_zip_rejects_file_without_dev_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    monkeypatch.delenv("CROSS_SECTION_ALLOW_DEV_UPDATE", raising=False)
    monkeypatch.setattr("paths.is_frozen", lambda: True)
    zip_path = tmp_path / "payload.zip"
    digest = _make_zip(zip_path)
    result = UpdateCheckResult(
        current_version="0.1.0",
        latest_version="0.1.1",
        update_available=True,
        download_url=zip_path.as_uri(),
        sha256=digest,
        notes=None,
        manifest_url="file://manifest",
    )
    with pytest.raises(ValueError, match="ALLOW_DEV_UPDATE"):
        download_update_zip(result, timeout_s=5.0)


def test_download_update_zip_allowlist_env_replaces_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    monkeypatch.setenv("CROSS_SECTION_UPDATE_URL_ALLOWLIST", "cdn.example.com")
    monkeypatch.delenv("CROSS_SECTION_ALLOW_DEV_UPDATE", raising=False)
    monkeypatch.setattr("paths.is_frozen", lambda: True)

    # Default GitHub host must be rejected when env replaces the list.
    with pytest.raises(ValueError, match="not allowlisted"):
        download_update_zip(
            UpdateCheckResult(
                current_version="0.1.0",
                latest_version="0.1.1",
                update_available=True,
                download_url="https://github.com/org/repo/releases/download/v1/a.zip",
                sha256="a" * 64,
                notes=None,
                manifest_url="https://cdn.example.com/m.json",
            ),
            timeout_s=5.0,
        )

    zip_path = tmp_path / "payload.zip"
    digest = _make_zip(zip_path)
    payload = zip_path.read_bytes()

    class _FakeResponse:
        def __init__(self, data: bytes) -> None:
            self._buf = io.BytesIO(data)
            self.headers = {"Content-Length": str(len(data))}

        def read(self, size: int = -1) -> bytes:
            return self._buf.read(size)

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        "desktop_updater._urlopen_update",
        lambda request, *, timeout_s=0.0: _FakeResponse(payload),
    )
    downloaded = download_update_zip(
        UpdateCheckResult(
            current_version="0.1.0",
            latest_version="0.1.1",
            update_available=True,
            download_url="https://cdn.example.com/updates/app.zip",
            sha256=digest,
            notes=None,
            manifest_url="https://cdn.example.com/m.json",
        ),
        timeout_s=5.0,
    )
    assert downloaded.sha256 == digest


def test_validate_update_download_url_blocks_host_spoofs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adversarial host checks: suffix tricks, userinfo, IPs, empty labels."""
    from desktop_updater import validate_update_download_url

    monkeypatch.delenv("CROSS_SECTION_ALLOW_DEV_UPDATE", raising=False)
    monkeypatch.delenv("CROSS_SECTION_UPDATE_URL_ALLOWLIST", raising=False)

    validate_update_download_url(
        "https://github.com/org/repo/releases/download/v1/a.zip"
    )
    validate_update_download_url(
        "https://objects.githubusercontent.com/github-production-release-asset/1/a.zip"
    )
    # Trailing FQDN dot is normalized, not a bypass.
    validate_update_download_url("https://github.com./org/a.zip")

    rejected = [
        ("https://evilgithub.com/x.zip", "not allowlisted"),
        ("https://notgithub.com/x.zip", "not allowlisted"),
        ("https://github.com.evil.example/x.zip", "not allowlisted"),
        ("https://github.com@evil.example/x.zip", "not allowlisted"),
        ("https://140.82.112.3/x.zip", "not allowlisted"),
        ("https://127.0.0.1/x.zip", "not allowlisted"),
        ("https://[::1]/x.zip", "not allowlisted"),
        ("https://.github.com/x.zip", "not allowlisted"),
        ("https://..github.com/x.zip", "not allowlisted"),
        ("http://github.com/org/repo/a.zip", "must use https"),
        ("ftp://github.com/x.zip", "must use https"),
        ("file:///C:/tmp/x.zip", "ALLOW_DEV_UPDATE"),
    ]
    for url, match in rejected:
        with pytest.raises(ValueError, match=match):
            validate_update_download_url(url)


def test_allowlist_single_label_is_exact_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ALLOWLIST=com`` must not authorize every ``*.com`` host."""
    from desktop_updater import validate_update_download_url

    monkeypatch.delenv("CROSS_SECTION_ALLOW_DEV_UPDATE", raising=False)
    monkeypatch.setenv("CROSS_SECTION_UPDATE_URL_ALLOWLIST", "com")
    with pytest.raises(ValueError, match="not allowlisted"):
        validate_update_download_url("https://evil.com/x.zip")
    with pytest.raises(ValueError, match="not allowlisted"):
        validate_update_download_url("https://github.com/x.zip")
    # Exact single-label match still works if someone hosts there.
    validate_update_download_url("https://com/x.zip")


def test_file_url_rejects_dot_and_dotdot_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``rstrip('.')`` must not collapse ``file://./`` / ``file://../`` to empty."""
    from desktop_updater import validate_update_download_url

    monkeypatch.setenv("CROSS_SECTION_ALLOW_DEV_UPDATE", "1")
    validate_update_download_url("file:///C:/tmp/x.zip")
    validate_update_download_url("file://localhost/C:/tmp/x.zip")
    validate_update_download_url("file://localhost./C:/tmp/x.zip")
    validate_update_download_url("file://127.0.0.1/C:/tmp/x.zip")
    validate_update_download_url("file://[::1]/C:/tmp/x.zip")
    for url in (
        "file://./C:/tmp/x.zip",
        "file://../C:/tmp/x.zip",
        "file://evil.com/share/x.zip",
        "file://github.com/x.zip",
    ):
        with pytest.raises(ValueError, match="host not allowed"):
            validate_update_download_url(url)


def test_redirect_handler_rejects_evil_hops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise ``_AllowlistedRedirectHandler`` itself (not just validate())."""
    from urllib.parse import urljoin
    from urllib.request import Request

    from desktop_updater import _AllowlistedRedirectHandler

    monkeypatch.delenv("CROSS_SECTION_ALLOW_DEV_UPDATE", raising=False)
    monkeypatch.delenv("CROSS_SECTION_UPDATE_URL_ALLOWLIST", raising=False)

    handler = _AllowlistedRedirectHandler()
    req = Request("https://github.com/org/repo/releases/download/v1/a.zip")

    class _DummyFP:
        def read(self) -> bytes:
            return b""

        def close(self) -> None:
            return None

    # Allowlisted CDN hop must be accepted.
    ok = urljoin(req.full_url, "https://objects.githubusercontent.com/asset/a.zip")
    new_req = handler.redirect_request(req, _DummyFP(), 302, "Found", {}, ok)
    assert new_req.full_url == ok

    evil_locations = [
        "https://evil.example/malware.zip",
        "//evil.example/malware.zip",
        "http://github.com/x.zip",
        "https://evilgithub.com/x.zip",
        "https://notgithub.com/x.zip",
        "https://github.com@evil.example/x.zip",
        "https://github.com.evil.example/x.zip",
        "file:///C:/evil.zip",
        "ftp://github.com/x.zip",
    ]
    for loc in evil_locations:
        joined = urljoin(req.full_url, loc)
        with pytest.raises(ValueError):
            handler.redirect_request(req, _DummyFP(), 302, "Found", {}, joined)
