# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Cross Section Studio (Windows onedir distribution)."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

block_cipher = None
root = Path(SPECPATH)

datas: list[tuple[str, str]] = [
    (str(root / "app.py"), "."),
    (str(root / ".streamlit" / "config.toml"), ".streamlit"),
    (str(root / "data"), "data"),
]

for py_file in sorted(root.glob("*.py")):
    if py_file.name in {"launcher.py"}:
        continue
    datas.append((str(py_file), "."))

hiddenimports = [
    "paths",
    "section_build_request",
    "app_state",
    "app_services",
    "app_common",
    "app_build",
    "app_sidebar",
    "app_upload",
    "app_styles",
    "app_validate",
    "app_configure",
    "app_generate",
    "parse_ops",
    "parsing",
    "render_profiles",
    "render_theme",
    "report_export",
    "renderer_common",
    "renderer_consulting",
    "renderer_section_sheet",
    "renderer_chart",
    "gwm_reference",
    "gwm_reference.fixtures",
    "gwm_reference.transects",
    "app",
    "ingestion",
    "models",
    "pipeline",
    "projection",
    "renderer",
    "stratigraphy",
    "constants",
    "ui_helpers",
    "transect_planner",
    "ai_quality",
    "ai_assistant",
    "lithology_codes",
    "pydantic",
    "pandas",
    "openpyxl",
    "numpy",
    "shapely",
    "shapely.geometry",
    "pyproj",
    "rapidfuzz",
    "openai",
    "matplotlib.backends.backend_svg",
    "matplotlib.backends.backend_agg",
    "matplotlib.backends.backend_pdf",
]

binaries: list = []
for package in ("streamlit", "matplotlib", "altair"):
    try:
        collected = collect_all(package)
        datas += collected[0]
        binaries += collected[1]
        hiddenimports += collected[2]
    except Exception:
        pass

try:
    import pyproj

    proj_data = Path(pyproj.datadir.get_data_dir())
    if proj_data.is_dir():
        datas.append((str(proj_data), "pyproj/proj_dir"))
except Exception:
    pass

a = Analysis(
    ["launcher.py"],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "test", "tests"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CrossSectionStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="CrossSectionStudio",
)
