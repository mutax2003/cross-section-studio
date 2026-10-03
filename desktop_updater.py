"""Full-zip desktop update: download, verify SHA-256, sidecar replace + restart.

Only intended for the Windows PyInstaller onedir layout. Dev checkouts stay
notify-only unless ``CROSS_SECTION_ALLOW_DEV_UPDATE=1``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from app_version import UpdateCheckResult
from app_version import normalize_version_text as _normalize_version_text
from update_url_policy import env_flag_enabled as _env_flag_enabled
from update_url_policy import validate_update_download_url

# Monkeypatch / test compat: same function as ``app_version.normalize_version_text``.
normalize_version_text = _normalize_version_text

ProgressCallback = Callable[[str, float | None], None]

# Keep Python in-process apply and the PowerShell sidecar on the same clock.
APPLY_WAIT_PID_TIMEOUT_S = 180.0
APPLY_POST_EXIT_SLEEP_S = 1.0
APPLY_LOCK_NAME = "apply_update.lock"
APPLY_LOCK_STALE_S = 3600.0
MAX_UPDATE_ZIP_BYTES = 2 * 1024**3  # hard ceiling for the update zip download


@dataclass(frozen=True)
class DownloadResult:
    zip_path: Path
    version: str
    sha256: str
    bytes_written: int


def updates_dir() -> Path:
    """Writable updates folder (zip, apply script, apply_update.log, lock).

    Hardens against empty ``LOCALAPPDATA``: ``paths.user_data_dir()`` would
    otherwise resolve to a CWD-relative ``CrossSectionStudio`` when the env var
    is set but blank.
    """
    from paths import app_root, is_frozen

    if is_frozen():
        local = (os.environ.get("LOCALAPPDATA") or "").strip()
        if local:
            base = Path(local) / "CrossSectionStudio"
        else:
            # Unset or blank: use the Windows LocalAppData convention, not CWD.
            base = Path.home() / "AppData" / "Local" / "CrossSectionStudio"
    else:
        base = app_root() / "data"
    path = (base / "updates").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def install_dir() -> Path:
    """Directory that contains ``CrossSectionStudio.exe`` and ``_internal`` when frozen."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # Dev: allow override for tests / dry-runs only.
    override = os.environ.get("CROSS_SECTION_INSTALL_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    raise RuntimeError("install_dir() requires a frozen app or CROSS_SECTION_INSTALL_DIR")


def auto_install_allowed() -> bool:
    """True when in-app full-zip install is enabled for this process."""
    if getattr(sys, "frozen", False):
        return True
    return _env_flag_enabled("CROSS_SECTION_ALLOW_DEV_UPDATE")


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _disk_free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return int(usage.free)


def _require_expected_version(expected_version: str | None) -> str:
    """Return normalized expected version or raise (never allow a blank gate)."""
    want = _normalize_version_text(expected_version)
    if not want:
        raise ValueError("expected_version is required")
    return want


class _AllowlistedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate every redirect hop against the update URL policy."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        validate_update_download_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _urlopen_update(request: urllib.request.Request, *, timeout_s: float):
    """Open ``request`` with redirect hops re-checked against the allowlist."""
    opener = urllib.request.build_opener(_AllowlistedRedirectHandler)
    return opener.open(request, timeout=timeout_s)


def download_update_zip(
    result: UpdateCheckResult,
    *,
    progress: ProgressCallback | None = None,
    timeout_s: float = 120.0,
) -> DownloadResult:
    """Download the manifest zip into the user updates folder and verify SHA-256.

    Download URLs must be ``https`` to an allowlisted host (GitHub release
    assets by default). See :func:`validate_update_download_url`. Redirects
    are re-validated on each hop. ``file://`` is only for local tests when
    ``CROSS_SECTION_ALLOW_DEV_UPDATE=1``.
    """
    if not result.update_available:
        raise ValueError("No update available")
    if not result.download_url:
        raise ValueError("Manifest has no download URL")
    validate_update_download_url(result.download_url)
    expected = (result.sha256 or "").strip().lower()
    if not expected:
        raise ValueError("Manifest SHA-256 is required (blank or missing)")
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ValueError(
            "Manifest SHA-256 must be a 64-character lowercase hex digest"
        )
    # Never fall back to "unknown" — blank/missing latest_version must fail the
    # VERSION gate rather than schedule an apply that cannot match the zip.
    version = _require_expected_version(result.latest_version)
    dest = updates_dir() / f"CrossSectionStudio-win64-v{version}.zip"
    if dest.exists():
        dest.unlink()

    request = urllib.request.Request(
        result.download_url,
        headers={"User-Agent": "CrossSectionStudio-Updater"},
        method="GET",
    )
    if progress:
        progress("Connecting…", None)
    with _urlopen_update(request, timeout_s=timeout_s) as response:  # noqa: S310
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header and total_header.isdigit() else None
        if total is not None and total > MAX_UPDATE_ZIP_BYTES:
            raise ValueError(
                f"Update zip too large ({total} bytes; limit {MAX_UPDATE_ZIP_BYTES})"
            )
        if total is not None:
            free = _disk_free_bytes(updates_dir())
            # Need room for zip + extracted tree + brief backup.
            if free < int(total * 2.5):
                raise OSError(
                    f"Not enough free disk space under {updates_dir()} "
                    f"(need ~{total * 2.5 / 1e6:.0f} MB, have {free / 1e6:.0f} MB)"
                )
        written = 0
        try:
            with dest.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    written += len(chunk)
                    if written > MAX_UPDATE_ZIP_BYTES:
                        raise ValueError(
                            f"Update download exceeded {MAX_UPDATE_ZIP_BYTES} bytes; aborting"
                        )
                    if progress and total:
                        progress("Downloading…", min(1.0, written / total))
                    elif progress:
                        progress(f"Downloading… ({written // (1024 * 1024)} MB)", None)
        except BaseException:
            dest.unlink(missing_ok=True)
            raise

    if progress:
        progress("Verifying SHA-256…", None)
    digest = _sha256_file(dest)
    if digest != expected:
        dest.unlink(missing_ok=True)
        raise ValueError(
            f"SHA-256 mismatch for update zip (got {digest}, expected {expected})"
        )
    if progress:
        progress("Download verified", 1.0)
    try:
        from ops_audit import audit_event

        audit_event(
            "update_downloaded",
            version=version,
            sha256=digest,
            bytes=written,
        )
    except Exception:
        pass
    return DownloadResult(zip_path=dest, version=version, sha256=digest, bytes_written=written)


def _pid_alive(pid: int) -> bool:
    """Return True if ``pid`` appears to be a live process.

    On Windows, avoid ``os.kill(pid, 0)`` — it can leave the process unable to
    enumerate the system certificate store (``ssl.enum_certificates`` hangs),
    which then breaks any later ``urllib`` HTTPS/file opener bootstrap.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        # PROCESS_QUERY_LIMITED_INFORMATION — enough to probe existence.
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _wait_for_pid(pid: int, *, timeout_s: float = APPLY_WAIT_PID_TIMEOUT_S) -> None:
    if pid <= 0:
        return
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.5)
    # Proceed anyway after timeout — caller may have already exited.


def _zip_info_is_symlink(info: zipfile.ZipInfo) -> bool:
    """True when the member is marked as a Unix symlink (external_attr mode)."""
    # create_system 3 == Unix; S_IFLNK == 0o120000.
    if int(info.create_system) != 3:
        return False
    mode = (int(info.external_attr) >> 16) & 0o170000
    return mode == 0o120000


def _assert_zip_member_confined(staging: Path, member_name: str) -> Path:
    """Return the resolved destination if ``member_name`` stays under ``staging``.

    Rejects absolute / UNC paths, empty/``.``/``..`` segments, and resolve-based
    escapes (zip-slip). Segment rejection avoids check-vs-extract drift where
    ``foo/../bar`` resolves inside staging but extract implementations disagree.
    """
    name = member_name.replace("\\", "/")
    if (
        name.startswith("/")
        or name.startswith("//")
        or (len(name) >= 2 and name[1] == ":")
    ):
        raise ValueError(f"Unsafe zip member path: {member_name!r}")
    parts = name.split("/")
    # Trailing slash marks a directory entry ("foo/"); allow a final empty part.
    if parts and parts[-1] == "":
        parts = parts[:-1]
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Unsafe zip member path: {member_name!r}")
    staging_resolved = staging.resolve()
    dest = (staging / "/".join(parts)).resolve()
    try:
        dest.relative_to(staging_resolved)
    except ValueError as exc:
        raise ValueError(f"Unsafe zip member path: {member_name!r}") from exc
    if dest == staging_resolved:
        raise ValueError(f"Unsafe zip member path: {member_name!r}")
    return dest


def _extract_zip_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    dest: Path,
    *,
    staging_resolved: Path,
) -> None:
    """Write one zip member to ``dest`` (already confinement-checked)."""
    if info.is_dir() or info.filename.replace("\\", "/").endswith("/"):
        dest.mkdir(parents=True, exist_ok=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to write through a symlink parent planted by an earlier member.
    cursor: Path | None = dest.parent
    while cursor is not None:
        if cursor.is_symlink():
            raise ValueError(f"Unsafe zip member path: {info.filename!r}")
        if cursor == staging_resolved:
            break
        parent = cursor.parent
        cursor = None if parent == cursor else parent
    with archive.open(info) as source, dest.open("wb") as target:
        shutil.copyfileobj(source, target)
    if dest.is_symlink():
        dest.unlink(missing_ok=True)
        raise ValueError(f"Unsafe zip member path: {info.filename!r}")


def _extract_zip_to(staging: Path, zip_path: Path) -> None:
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    staging_resolved = staging.resolve()
    with zipfile.ZipFile(zip_path, "r") as archive:
        for info in archive.infolist():
            if _zip_info_is_symlink(info):
                raise ValueError(f"Unsafe zip member path: {info.filename!r}")
            dest = _assert_zip_member_confined(staging, info.filename)
            _extract_zip_member(
                archive, info, dest, staging_resolved=staging_resolved
            )
            # Re-validate after write (symlink / junction races).
            written = dest.resolve()
            try:
                written.relative_to(staging_resolved)
            except ValueError as exc:
                raise ValueError(f"Unsafe zip member path: {info.filename!r}") from exc
    # If the zip wrapped a single top-level folder, unwrap it.
    children = [p for p in staging.iterdir()]
    if len(children) == 1 and children[0].is_dir() and (children[0] / "CrossSectionStudio.exe").is_file():
        inner = children[0]
        for item in inner.iterdir():
            target = staging / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
        inner.rmdir()


def _require_staging_version(staging: Path, expected_version: str) -> None:
    """Require ``staging/VERSION`` (root only) to match ``expected_version``."""
    want = _require_expected_version(expected_version)
    version_path = staging / "VERSION"
    if version_path.is_symlink():
        raise ValueError("Update zip VERSION must be a regular file (not a symlink)")
    if not version_path.is_file():
        raise FileNotFoundError(
            f"Update zip is missing VERSION after extract ({staging})"
        )
    got = _normalize_version_text(version_path.read_text(encoding="utf-8"))
    if not got:
        raise ValueError("Update zip VERSION file is empty")
    if got != want:
        raise ValueError(
            f"VERSION mismatch after extract (got {got}, expected {want})"
        )


def apply_log_path() -> Path:
    """Persistent apply log under the user updates directory (always absolute)."""
    return updates_dir() / "apply_update.log"


def apply_lock_path() -> Path:
    """Exclusive apply lock under the user updates directory."""
    return updates_dir() / APPLY_LOCK_NAME


def _append_apply_log(message: str) -> None:
    """Append a timestamped line to ``apply_update.log`` (best-effort)."""
    try:
        path = apply_log_path()
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{ts}] {message}\n")
    except Exception:
        pass


def _acquire_apply_lock() -> TextIO:
    """Create an exclusive apply lock or raise if another apply holds it.

    Stale locks (dead PID or older than ``APPLY_LOCK_STALE_S``) are removed.
    """
    lock_path = apply_lock_path()
    for _ in range(2):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            handle = os.fdopen(fd, "w+", encoding="utf-8")
            handle.write(f"{os.getpid()}\n{time.time():.3f}\n")
            handle.flush()
            return handle
        except FileExistsError:
            stale = False
            try:
                text = lock_path.read_text(encoding="utf-8").strip().splitlines()
                holder = int(text[0]) if text else 0
                created = float(text[1]) if len(text) > 1 else 0.0
                if not _pid_alive(holder):
                    stale = True
                elif created and (time.time() - created) > APPLY_LOCK_STALE_S:
                    stale = True
            except Exception:
                # Unreadable / corrupt lock — treat as stale so apply can proceed.
                stale = True
            if stale:
                try:
                    lock_path.unlink(missing_ok=True)
                    continue
                except OSError as exc:
                    raise RuntimeError(
                        f"Another update apply holds {lock_path} (could not clear stale lock)"
                    ) from exc
            raise RuntimeError(
                f"Another update apply is already in progress ({lock_path})"
            ) from None
    raise RuntimeError(f"Could not acquire apply lock at {lock_path}")


def _release_apply_lock(handle: TextIO | None) -> None:
    if handle is None:
        return
    lock_path = apply_lock_path()
    try:
        handle.close()
    except Exception:
        pass
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:
        pass


def _quarantine_path(path: Path) -> Path | None:
    """Move ``path`` aside so a backup restore can reclaim the install name."""
    if not path.exists():
        return None
    parent = path.parent
    broken = parent / f"{path.name}.broken-{int(time.time())}"
    if broken.exists():
        shutil.rmtree(broken, ignore_errors=True)
        if broken.exists():
            broken.unlink(missing_ok=True)
    try:
        shutil.move(str(path), str(broken))
        return broken
    except OSError:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
        if path.exists():
            raise
        return None


def _restore_backup_to_install(
    backup: Path,
    target_install_dir: Path,
    *,
    staging: Path | None = None,
) -> str:
    """Restore install from backup; never leave the user with no install dir.

    If a partial/failed new tree occupies ``target_install_dir``, quarantine it
    first. If backup restore fails and ``staging`` still exists, move staging
    into place as degraded recovery so *some* install remains.

    Returns a short status phrase for the apply log. Raises only when no install
    directory could be put back.
    """
    if target_install_dir.exists():
        _quarantine_path(target_install_dir)
    try:
        shutil.move(str(backup), str(target_install_dir))
        return f"restored backup to {target_install_dir}"
    except OSError as restore_exc:
        if staging is not None and staging.exists() and not target_install_dir.exists():
            try:
                shutil.move(str(staging), str(target_install_dir))
                return (
                    f"backup restore failed ({restore_exc}); "
                    f"placed staging at {target_install_dir} as degraded recovery"
                )
            except OSError as staging_exc:
                raise RuntimeError(
                    f"backup restore failed ({restore_exc}); "
                    f"also failed to place staging ({staging_exc}); "
                    f"backup at {backup}"
                ) from staging_exc
        raise RuntimeError(
            f"backup restore failed ({restore_exc}); backup remains at {backup}"
        ) from restore_exc


def _ps_quote(value: object) -> str:
    """PowerShell single-quoted literal: inert (no ``$`` / backtick expansion; non-ASCII safe).

    PS grammar (spec 2.3.5.2) treats U+2018..U+201B as single-quote characters
    too, so those must be doubled alongside the ASCII apostrophe.
    """
    text = str(value)
    for quote in ("'", "\u2018", "\u2019", "\u201a", "\u201b"):
        text = text.replace(quote, quote * 2)
    return "'" + text + "'"


def _write_apply_script(
    *,
    zip_path: Path,
    target_install_dir: Path,
    wait_pid: int,
    relaunch: bool,
    expected_version: str,
    expected_sha256: str | None = None,
) -> Path:
    """Write a PowerShell apply script outside the install tree (avoids self-replace locks)."""
    script = updates_dir() / "apply_update.ps1"
    relaunch_exe = target_install_dir / "CrossSectionStudio.exe"
    zip_lit = _ps_quote(zip_path)
    install_lit = _ps_quote(target_install_dir)
    exe_lit = _ps_quote(relaunch_exe)
    log_lit = _ps_quote(apply_log_path())
    lock_lit = _ps_quote(apply_lock_path())
    expected_version_lit = _ps_quote(_require_expected_version(expected_version))
    expected_sha_lit = _ps_quote((expected_sha256 or "").strip().lower())
    wait_timeout_s = int(APPLY_WAIT_PID_TIMEOUT_S)
    post_sleep_s = int(APPLY_POST_EXIT_SLEEP_S)
    stale_s = int(APPLY_LOCK_STALE_S)
    relaunch_lit = "$true" if relaunch else "$false"
    body = f"""$ErrorActionPreference = "Stop"
$zip = {zip_lit}
$installDir = {install_lit}
$waitPid = {int(wait_pid)}
$logFile = {log_lit}
$lockFile = {lock_lit}
$expectedVersion = {expected_version_lit}
$expectedSha256 = {expected_sha_lit}
$staging = Join-Path (Split-Path -Parent $installDir) ((Split-Path -Leaf $installDir) + ".new")
$backup = Join-Path (Split-Path -Parent $installDir) ((Split-Path -Leaf $installDir) + ".bak-" + [int][double]::Parse((Get-Date -UFormat %s)))
$relaunch = {relaunch_lit}
$backedUp = $false
$lockHeld = $false
$waitTimeoutS = {wait_timeout_s}
$postSleepS = {post_sleep_s}
$lockStaleS = {stale_s}

function Write-ApplyLog([string]$message) {{
  $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  $line = "[$ts] $message"
  $dir = Split-Path -Parent $logFile
  if (-not (Test-Path -LiteralPath $dir)) {{
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
  }}
  Add-Content -LiteralPath $logFile -Value $line -Encoding utf8
}}

function Test-PidAlive([int]$TargetPid) {{
  if ($TargetPid -le 0) {{ return $false }}
  try {{
    Get-Process -Id $TargetPid -ErrorAction Stop | Out-Null
    return $true
  }} catch {{
    return $false
  }}
}}

function Acquire-ApplyLock {{
  $dir = Split-Path -Parent $lockFile
  if (-not (Test-Path -LiteralPath $dir)) {{
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
  }}
  for ($i = 0; $i -lt 2; $i++) {{
    try {{
      $fs = [System.IO.File]::Open($lockFile, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
      $bytes = [System.Text.Encoding]::UTF8.GetBytes(("{{0}}`n{{1:0.000}}`n" -f $PID, [double](Get-Date -UFormat %s)))
      $fs.Write($bytes, 0, $bytes.Length)
      $fs.Flush()
      return $fs
    }} catch {{
      $stale = $false
      if (Test-Path -LiteralPath $lockFile) {{
        try {{
          $lines = Get-Content -LiteralPath $lockFile -ErrorAction Stop
          $holder = 0
          $created = 0.0
          if ($lines.Count -ge 1) {{ [int]::TryParse($lines[0], [ref]$holder) | Out-Null }}
          if ($lines.Count -ge 2) {{ [double]::TryParse($lines[1], [ref]$created) | Out-Null }}
          if (-not (Test-PidAlive $holder)) {{ $stale = $true }}
          elseif (($created -gt 0) -and ((([double](Get-Date -UFormat %s)) - $created) -gt $lockStaleS)) {{ $stale = $true }}
        }} catch {{
          $stale = $true
        }}
      }}
      if ($stale) {{
        Remove-Item -LiteralPath $lockFile -Force -ErrorAction SilentlyContinue
        continue
      }}
      throw "Another update apply is already in progress ($lockFile)"
    }}
  }}
  throw "Could not acquire apply lock at $lockFile"
}}

function Release-ApplyLock($fs) {{
  if ($null -ne $fs) {{
    try {{ $fs.Close() }} catch {{}}
  }}
  Remove-Item -LiteralPath $lockFile -Force -ErrorAction SilentlyContinue
}}

function Quarantine-Path([string]$path) {{
  if (-not (Test-Path -LiteralPath $path)) {{ return }}
  $parent = Split-Path -Parent $path
  $leaf = Split-Path -Leaf $path
  $broken = Join-Path $parent ($leaf + ".broken-" + [int][double]::Parse((Get-Date -UFormat %s)))
  if (Test-Path -LiteralPath $broken) {{ Remove-Item -LiteralPath $broken -Recurse -Force }}
  try {{
    Move-Item -LiteralPath $path -Destination $broken -Force
  }} catch {{
    Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $path) {{ throw }}
  }}
}}

function Restore-BackupToInstall {{
  if (Test-Path -LiteralPath $installDir) {{
    Quarantine-Path $installDir
  }}
  try {{
    Move-Item -LiteralPath $backup -Destination $installDir
    return "restored backup to $installDir"
  }} catch {{
    $restoreErr = $_.Exception.Message
    if ((Test-Path -LiteralPath $staging) -and -not (Test-Path -LiteralPath $installDir)) {{
      try {{
        Move-Item -LiteralPath $staging -Destination $installDir
        return "backup restore failed ($restoreErr); placed staging at $installDir as degraded recovery"
      }} catch {{
        throw "backup restore failed ($restoreErr); also failed to place staging ($($_.Exception.Message)); backup at $backup"
      }}
    }}
    throw "backup restore failed ($restoreErr); backup remains at $backup"
  }}
}}

function Expand-ZipConfined([string]$zipPath, [string]$destDir) {{
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $stagingFull = [System.IO.Path]::GetFullPath($destDir)
  if (-not $stagingFull.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {{
    $stagingFull = $stagingFull + [System.IO.Path]::DirectorySeparatorChar
  }}
  $archive = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
  try {{
    foreach ($entry in $archive.Entries) {{
      $name = $entry.FullName -replace '\\','/'
      if ($name.StartsWith('/') -or $name.StartsWith('//') -or ($name.Length -ge 2 -and $name[1] -eq ':')) {{
        throw "Unsafe zip member path: $($entry.FullName)"
      }}
      $parts = @($name.Split('/') | Where-Object {{ $_ -ne '' }})
      if ($parts.Count -eq 0) {{
        throw "Unsafe zip member path: $($entry.FullName)"
      }}
      foreach ($part in $parts) {{
        if ($part -eq '.' -or $part -eq '..') {{
          throw "Unsafe zip member path: $($entry.FullName)"
        }}
      }}
      # Unix symlink bit in high 16 bits of external attributes (S_IFLNK = 0xA000).
      $mode = ($entry.ExternalAttributes -shr 16) -band 0xF000
      if ($mode -eq 0xA000) {{
        throw "Unsafe zip member path: $($entry.FullName)"
      }}
      $dest = [System.IO.Path]::GetFullPath((Join-Path $destDir ($parts -join [System.IO.Path]::DirectorySeparatorChar)))
      if (-not $dest.StartsWith($stagingFull, [System.StringComparison]::OrdinalIgnoreCase)) {{
        throw "Unsafe zip member path: $($entry.FullName)"
      }}
      if ($dest.TrimEnd('\\') -eq $stagingFull.TrimEnd('\\')) {{
        throw "Unsafe zip member path: $($entry.FullName)"
      }}
      if ($entry.FullName.EndsWith('/') -or $entry.FullName.EndsWith('\\')) {{
        if (-not (Test-Path -LiteralPath $dest)) {{
          New-Item -ItemType Directory -Path $dest -Force | Out-Null
        }}
        continue
      }}
      $parent = Split-Path -Parent $dest
      if (-not (Test-Path -LiteralPath $parent)) {{
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
      }}
      [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $dest, $true)
    }}
  }} finally {{
    $archive.Dispose()
  }}
}}

$lockStream = $null
try {{
  $lockStream = Acquire-ApplyLock
  $lockHeld = $true

  if ($waitPid -gt 0) {{
    $deadline = (Get-Date).AddSeconds($waitTimeoutS)
    while ((Get-Date) -lt $deadline) {{
      if (-not (Test-PidAlive $waitPid)) {{ break }}
      Start-Sleep -Milliseconds 500
    }}
    Start-Sleep -Seconds $postSleepS
  }}

  if (-not [string]::IsNullOrWhiteSpace($expectedSha256)) {{
    $gotSha = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($gotSha -ne $expectedSha256) {{
      throw "SHA-256 mismatch for update zip at apply time (got $gotSha, expected $expectedSha256)"
    }}
  }}

  if (Test-Path -LiteralPath $staging) {{ Remove-Item -LiteralPath $staging -Recurse -Force }}
  New-Item -ItemType Directory -Path $staging | Out-Null
  Expand-ZipConfined $zip $staging

  $children = @(Get-ChildItem -LiteralPath $staging)
  if ($children.Count -eq 1 -and $children[0].PSIsContainer) {{
    $innerExe = Join-Path $children[0].FullName "CrossSectionStudio.exe"
    if (Test-Path -LiteralPath $innerExe) {{
      $inner = $children[0].FullName
      Get-ChildItem -LiteralPath $inner | ForEach-Object {{
        Move-Item -LiteralPath $_.FullName -Destination (Join-Path $staging $_.Name) -Force
      }}
      Remove-Item -LiteralPath $inner -Force
    }}
  }}

  $exe = Join-Path $staging "CrossSectionStudio.exe"
  if (-not (Test-Path -LiteralPath $exe)) {{
    throw "Update zip missing CrossSectionStudio.exe"
  }}

  $versionFile = Join-Path $staging "VERSION"
  if (-not (Test-Path -LiteralPath $versionFile)) {{
    throw "Update zip missing VERSION file"
  }}
  if ([string]::IsNullOrWhiteSpace($expectedVersion)) {{
    throw "expected_version is required"
  }}
  $gotVersion = ((Get-Content -LiteralPath $versionFile -Raw).Trim() -replace '^[vV]', '').Trim()
  if ([string]::IsNullOrWhiteSpace($gotVersion)) {{
    throw "Update zip VERSION file is empty"
  }}
  if ($gotVersion -ne $expectedVersion) {{
    throw "VERSION mismatch after extract (got $gotVersion, expected $expectedVersion)"
  }}

  if (Test-Path -LiteralPath $installDir) {{
    if (Test-Path -LiteralPath $backup) {{ Remove-Item -LiteralPath $backup -Recurse -Force }}
    Move-Item -LiteralPath $installDir -Destination $backup
    $backedUp = $true
  }}
  Move-Item -LiteralPath $staging -Destination $installDir

  if ($relaunch) {{
    $newExe = {exe_lit}
    if (Test-Path -LiteralPath $newExe) {{
      Start-Process -FilePath $newExe -WorkingDirectory $installDir
    }}
  }}
  Write-ApplyLog "SUCCESS: Applied update to $installDir from $zip"
}} catch {{
  $err = $_.Exception.Message
  if ($backedUp -and (Test-Path -LiteralPath $backup)) {{
    try {{
      $restoreStatus = Restore-BackupToInstall
      Write-ApplyLog "FAILURE: $err; $restoreStatus"
    }} catch {{
      Write-ApplyLog "FAILURE: $err; also failed to restore backup: $($_.Exception.Message)"
    }}
  }} else {{
    Write-ApplyLog "FAILURE: $err"
  }}
  exit 1
}} finally {{
  if ($lockHeld) {{
    Release-ApplyLock $lockStream
  }}
}}
"""
    script.write_text(body, encoding="utf-8-sig")
    return script


def apply_update_from_zip(
    zip_path: Path,
    target_install_dir: Path,
    *,
    expected_version: str,
    wait_pid: int = 0,
    relaunch: bool = True,
    expected_sha256: str | None = None,
) -> Path:
    """Replace ``target_install_dir`` contents from ``zip_path`` after ``wait_pid`` exits.

    Preferred production path uses :func:`schedule_sidecar_apply` (external PowerShell)
    so the running EXE is not locked inside the folder being moved.

    After extract, ``staging/VERSION`` must match ``expected_version`` or apply fails
    (and restores from backup if install was already moved aside).

    If install→backup succeeds but staging→install fails, quarantine any partial
    install, restore backup→install, and append a FAILURE line to
    :func:`apply_log_path`. Concurrent applies are rejected via
    :func:`apply_lock_path`.
    """
    zip_path = zip_path.resolve()
    target_install_dir = target_install_dir.resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    expected_norm = _require_expected_version(expected_version)

    lock_handle: TextIO | None = None
    try:
        lock_handle = _acquire_apply_lock()
        _wait_for_pid(wait_pid)
        # Match PowerShell: only pause after a real parent-PID wait.
        if wait_pid > 0:
            time.sleep(APPLY_POST_EXIT_SLEEP_S)

        if expected_sha256:
            expected_sha = expected_sha256.strip().lower()
            got_sha = _sha256_file(zip_path)
            if got_sha != expected_sha:
                msg = (
                    f"SHA-256 mismatch for update zip at apply time "
                    f"(got {got_sha}, expected {expected_sha})"
                )
                # The detached fallback apply runs with stderr at DEVNULL —
                # the log is the only diagnostic channel (PS sidecar parity).
                _append_apply_log(f"FAILURE: {msg}")
                raise ValueError(msg)

        parent = target_install_dir.parent
        staging = parent / f"{target_install_dir.name}.new"
        backup = parent / f"{target_install_dir.name}.bak-{int(time.time())}"
        backed_up = False
        try:
            _extract_zip_to(staging, zip_path)
            exe = staging / "CrossSectionStudio.exe"
            if not exe.is_file():
                raise FileNotFoundError(
                    f"Update zip is missing CrossSectionStudio.exe after extract ({staging})"
                )
            _require_staging_version(staging, expected_norm)

            if target_install_dir.exists():
                if backup.exists():
                    shutil.rmtree(backup, ignore_errors=True)
                shutil.move(str(target_install_dir), str(backup))
                backed_up = True
            shutil.move(str(staging), str(target_install_dir))

            new_exe = target_install_dir / "CrossSectionStudio.exe"
            if relaunch and new_exe.is_file():
                subprocess.Popen(  # noqa: S603
                    [str(new_exe)],
                    cwd=str(target_install_dir),
                    close_fds=True,
                    creationflags=getattr(subprocess, "DETACHED_PROCESS", 0)
                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                )
            try:
                from ops_audit import audit_event

                audit_event(
                    "update_applied",
                    install_dir=str(target_install_dir),
                    backup=str(backup) if backup.exists() else "",
                    zip=str(zip_path),
                    version=expected_norm,
                )
            except Exception:
                pass
            _append_apply_log(
                f"SUCCESS: Applied update to {target_install_dir} from {zip_path}"
            )
            return target_install_dir
        except Exception as exc:
            if backed_up and backup.exists():
                try:
                    restore_status = _restore_backup_to_install(
                        backup, target_install_dir, staging=staging
                    )
                    _append_apply_log(f"FAILURE: {exc}; {restore_status}")
                except Exception as restore_exc:
                    _append_apply_log(
                        f"FAILURE: {exc}; also failed to restore backup: {restore_exc}"
                    )
            else:
                _append_apply_log(f"FAILURE: {exc}")
            raise
    finally:
        _release_apply_lock(lock_handle)


def schedule_sidecar_apply(
    zip_path: Path,
    *,
    expected_version: str,
    expected_sha256: str | None = None,
    install_root: Path | None = None,
) -> int:
    """Spawn detached PowerShell that applies the zip after this process exits.

    Returns the child PID. The caller should exit soon after.
    """
    expected_norm = _require_expected_version(expected_version)
    root = install_root or install_dir()
    wait_pid = os.getpid()
    script = _write_apply_script(
        zip_path=zip_path.resolve(),
        target_install_dir=root.resolve(),
        wait_pid=wait_pid,
        relaunch=True,
        expected_version=expected_norm,
        expected_sha256=expected_sha256,
    )
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if not powershell:
        # Fallback for tests / non-Windows: run in-process apply via python module.
        return _schedule_python_apply(zip_path, root, wait_pid, expected_norm, expected_sha256)

    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )
    proc = subprocess.Popen(  # noqa: S603
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
        ],
        cwd=str(updates_dir()),
        close_fds=True,
        creationflags=flags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )
    try:
        from ops_audit import audit_event

        audit_event(
            "update_scheduled",
            zip=str(zip_path),
            install_dir=str(root),
            sidecar_pid=proc.pid,
            script=str(script),
            version=expected_norm,
        )
    except Exception:
        pass
    return int(proc.pid)


def _schedule_python_apply(
    zip_path: Path,
    root: Path,
    wait_pid: int,
    expected_version: str,
    expected_sha256: str | None = None,
) -> int:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--apply-update",
        "--zip",
        str(zip_path),
        "--install-dir",
        str(root),
        "--wait-pid",
        str(wait_pid),
        "--expected-version",
        _require_expected_version(expected_version),
    ]
    if expected_sha256:
        cmd += ["--expected-sha256", expected_sha256.strip().lower()]
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        cwd=str(root),
        close_fds=True,
        creationflags=flags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )
    return int(proc.pid)


def download_and_schedule_install(
    result: UpdateCheckResult,
    *,
    progress: ProgressCallback | None = None,
) -> DownloadResult:
    """Download + verify, then schedule sidecar apply. Caller must exit the app."""
    if not auto_install_allowed():
        raise RuntimeError(
            "Auto-install is only enabled for the Windows desktop build "
            "(or set CROSS_SECTION_ALLOW_DEV_UPDATE=1 with CROSS_SECTION_INSTALL_DIR)."
        )
    downloaded = download_update_zip(result, progress=progress)
    schedule_sidecar_apply(
        downloaded.zip_path,
        expected_version=downloaded.version,
        expected_sha256=downloaded.sha256,
    )
    return downloaded


def parse_apply_argv(argv: list[str] | None = None) -> argparse.Namespace | None:
    """Return apply-update args if present; otherwise ``None``."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] != "--apply-update":
        return None
    parser = argparse.ArgumentParser(prog="desktop_updater")
    parser.add_argument("--apply-update", action="store_true")
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--install-dir", required=True, type=Path)
    parser.add_argument("--wait-pid", type=int, default=0)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--expected-sha256", default=None)
    parser.add_argument("--no-relaunch", action="store_true")
    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> int:
    parsed = parse_apply_argv(argv)
    if parsed is None:
        print(
            "Usage: desktop_updater.py --apply-update --zip PATH "
            "--install-dir PATH --expected-version VERSION",
            file=sys.stderr,
        )
        return 2
    apply_update_from_zip(
        parsed.zip,
        parsed.install_dir,
        expected_version=parsed.expected_version,
        wait_pid=parsed.wait_pid,
        relaunch=not parsed.no_relaunch,
        expected_sha256=parsed.expected_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
