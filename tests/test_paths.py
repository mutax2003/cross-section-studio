"""Quick test for frozen-friendly resource paths."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import (
    app_root,
    cross_section_input_template,
    import_profiles_dir,
    lithology_aliases_path,
    sample_boreholes_workbook,
)


def test_app_root_contains_app_py() -> None:
    assert (app_root() / "app.py").is_file()


def test_import_profiles_dir_exists() -> None:
    assert import_profiles_dir().is_dir()
    assert any(import_profiles_dir().glob("*.json"))


def test_paths_help_topic_helpers() -> None:
    from paths import help_dir, help_topic_path

    assert help_dir().is_dir()
    path = help_topic_path("about")
    assert path is not None and path.is_file()
    assert help_topic_path(r"..\etc\passwd") is None
    assert help_topic_path("") is None
    assert help_topic_path("a/b") is None


def test_lithology_aliases_path_readable() -> None:
    path = lithology_aliases_path()
    assert path.is_file()


def test_cross_section_input_template_prefers_multi_tab() -> None:
    path = cross_section_input_template()
    if not path.exists():
        pytest.skip("Input template xlsx not present (gitignored; run build_input_template.py)")
    import pandas as pd

    names = set(pd.ExcelFile(path).sheet_names)
    assert {"Collars", "Lithology", "Instructions"}.issubset(names)


def test_sample_boreholes_workbook_path() -> None:
    path = sample_boreholes_workbook()
    assert path.name == "sample_boreholes.xlsx"
    assert path.parent == app_root() / "data"
