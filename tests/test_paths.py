"""Quick test for frozen-friendly resource paths."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import app_root, import_profiles_dir, lithology_aliases_path


def test_app_root_contains_app_py() -> None:
    assert (app_root() / "app.py").is_file()


def test_import_profiles_dir_exists() -> None:
    assert import_profiles_dir().is_dir()
    assert any(import_profiles_dir().glob("*.json"))


def test_lithology_aliases_path_readable() -> None:
    path = lithology_aliases_path()
    assert path.is_file()
