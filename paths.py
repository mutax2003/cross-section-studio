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


def bh_log_lithology_legend_xlsx_path() -> Path:
    """Source-of-truth BH Log lithology colour legend (Excel)."""
    return app_root() / "data" / "BH Log Lithology Legend.xlsx"


def bh_log_lithology_legend_path() -> Path:
    """Cached JSON export of the BH Log legend (fallback when Excel is absent)."""
    return app_root() / "data" / "bh_log_lithology_legend.json"


def cross_section_chlorides_workbook() -> Path:
    return app_root() / "data" / "Cross_Section_Chlorides.xlsx"


def _template_has_named_data_tabs(path: Path) -> bool:
    """True when the workbook is the multi-tab template (Collars + Lithology)."""
    try:
        import pandas as pd

        names = {str(name) for name in pd.ExcelFile(path).sheet_names}
        return {"Collars", "Lithology"}.issubset(names)
    except Exception:
        return False


def cross_section_input_template() -> Path:
    """Return the best available input template (prefer multi-tab, then newest mtime)."""
    root = app_root() / "data"
    primary = root / "Cross_Section_Input_Template.xlsx"
    candidates = sorted(
        root.glob("Cross_Section_Input_Template*.xlsx"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    multi_tab = [path for path in candidates if _template_has_named_data_tabs(path)]
    if multi_tab:
        return multi_tab[0]
    if candidates:
        return candidates[0]
    return primary


def build_input_template_path() -> Path:
    """Writable path for regenerating the input template (handles locked Excel files)."""
    root = app_root() / "data"
    primary = root / "Cross_Section_Input_Template.xlsx"
    try:
        if primary.exists():
            with primary.open("a"):
                pass
        return primary
    except OSError:
        return root / "Cross_Section_Input_Template_v3.xlsx"


def advantage_p2_reference_pdf_dir() -> Path:
    return app_root() / "data"


def advantage_platform_workbook() -> Path:
    return app_root() / "data" / "advantage_phase2_platform.xlsx"


def advantage_source_workbook() -> Path:
    return app_root() / "data" / "fixtures" / "advantage_phase2_source.xlsx"


def ecoventure_gwm_workbook() -> Path:
    return app_root() / "data" / "fixtures" / "ecoventure_gwm_16-29.xlsx"


def sample_boreholes_workbook() -> Path:
    """Bundled demo workbook for Try sample project (dev + frozen)."""
    return app_root() / "data" / "sample_boreholes.xlsx"


def audit_log_path() -> Path:
    """Writable audit log path; absolute overrides must stay under user_data_dir()."""
    override = os.environ.get("CROSS_SECTION_AUDIT_LOG")
    allowed_root = user_data_dir().resolve()
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_absolute():
            return (user_data_dir() / candidate).resolve()
        resolved = candidate.resolve()
        try:
            resolved.relative_to(allowed_root)
        except ValueError as exc:
            raise PermissionError(
                f"CROSS_SECTION_AUDIT_LOG must stay under {allowed_root}"
            ) from exc
        return resolved
    return allowed_root / "audit.log"


def help_dir() -> Path:
    """Bundled operator help markdown (``docs/help``)."""
    return app_root() / "docs" / "help"


def help_topic_path(topic: str) -> Path | None:
    """Resolve ``docs/help/{topic}.md``; reject path traversal."""
    raw = str(topic).strip()
    if not raw or ".." in raw or "/" in raw or "\\" in raw:
        return None
    name = Path(raw).name
    if name != raw:
        return None
    if not name.endswith(".md"):
        name = f"{name}.md"
    root = help_dir().resolve()
    path = (root / name).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    return path
