"""Resolve application and user-writable paths (dev vs PyInstaller bundle)."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def app_root() -> Path:
    """Directory containing app code and bundled read-only data."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS"))  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


def user_data_dir() -> Path:
    """Writable per-user folder for aliases and future local settings."""
    if is_frozen():
        base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "CrossSectionStudio"
    else:
        base = app_root() / "data"
    base.mkdir(parents=True, exist_ok=True)
    return base


def import_profiles_dir() -> Path:
    return app_root() / "data" / "import_profiles"


def lithology_aliases_path() -> Path:
    bundled = app_root() / "data" / "lithology_aliases.json"
    if not is_frozen():
        return bundled
    user_file = user_data_dir() / "lithology_aliases.json"
    if not user_file.exists():
        if bundled.exists():
            shutil.copy2(bundled, user_file)
        else:
            user_file.write_text("{}\n", encoding="utf-8")
    return user_file


def lithology_styles_path() -> Path:
    return user_data_dir() / "lithology_styles.json"
