"""Multi-section workbook end to end on a realistic synthetic consulting site.

The client's real multi-section workbook is not available yet, so this builds
one shaped like a typical site investigation: 14 boreholes on three named
section lines (A-A' west→east, B-B' and C-C' north→south crossing A-A' at a
shared hole each), 261002-style lithology names, one logging gap, two water
level rounds, chloride readings scored against thresholds, and well screens.

It is written into the input template (so upload detection and the Project
tab behave as for a real filled-in template) and exercised through ingestion,
the batch ZIP builders and the Streamlit app.
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SECTIONS = {
    "A-A'": ("BH-01", "BH-02", "BH-03", "BH-04", "BH-05", "BH-06"),
    "B-B'": ("BH-07", "BH-08", "BH-03", "BH-09", "BH-10"),
    "C-C'": ("BH-11", "BH-12", "BH-05", "BH-13", "BH-14"),
}
SHARED_HOLES = {"BH-03": ("A-A'", "B-B'"), "BH-05": ("A-A'", "C-C'")}
GAP_HOLE = "BH-09"
GAP_LENGTH = 0.6  # metres of no recovery, deliberately not logged

E0, N0 = 512_000.0, 5_934_000.0


def _collars() -> list[dict[str, object]]:
    """UTM-style coordinates; ground falls gently to the east and south."""
    coords: dict[str, tuple[float, float]] = {}
    for index, hole in enumerate(SECTIONS["A-A'"]):
        coords[hole] = (E0 + 62.0 * index, N0 + 200.0 + (3.0 if index % 2 else -2.0))
    for hole, dn, de in zip(
        SECTIONS["B-B'"],
        (135.0, 68.0, 0.0, -61.0, -128.0),
        (4.0, -3.0, 0.0, 2.5, -4.0),
        strict=True,
    ):
        if hole not in coords:
            coords[hole] = (coords["BH-03"][0] + de, coords["BH-03"][1] + dn)
    for hole, dn, de in zip(
        SECTIONS["C-C'"],
        (142.0, 71.0, 0.0, -66.0, -133.0),
        (-3.5, 2.0, 0.0, -2.0, 3.0),
        strict=True,
    ):
        if hole not in coords:
            coords[hole] = (coords["BH-05"][0] + de, coords["BH-05"][1] + dn)
    rows: list[dict[str, object]] = []
    for number in range(1, 15):
        hole = f"BH-{number:02d}"
        easting, northing = coords[hole]
        elevation = 748.6 - 0.011 * (easting - E0) + 0.009 * (northing - N0 - 200.0)
        rows.append(
            {
                "hole_id": hole,
                "easting": round(easting, 2),
                "northing": round(northing, 2),
                "elevation": round(elevation, 2),
                "total_depth": 12.2 + (number % 4) * 0.9,
                "stick_up_m": 0.85 if number % 3 else "",
            }
        )
    return rows


def _lithology(collars: list[dict[str, object]]) -> list[dict[str, object]]:
    """Topsoil / Clay Loam / Silty Clay over a sand(-and-gravel) aquifer on clay till.

    Thicknesses vary smoothly with position; the upper Sand lens is absent in
    the west and south (a pinch-out), Silt shows in the north, and Gravel
    appears at the base of the aquifer in a few eastern holes.
    """
    rows: list[dict[str, object]] = []
    for collar in collars:
        hole = str(collar["hole_id"])
        number = int(hole[-2:])
        total = float(collar["total_depth"])
        de = (float(collar["easting"]) - E0) / 310.0  # 0 west .. 1 east
        dn = (float(collar["northing"]) - N0 - 200.0) / 140.0  # -1 south .. +1 north
        top = 0.3 if number % 2 else 0.2
        loam = round(top + 0.9 + 0.3 * de, 2)
        silty = round(loam + 2.1 + 0.5 * dn, 2)
        units: list[tuple[float, float, str]] = [
            (0.0, top, "Topsoil"),
            (top, loam, "Clay Loam"),
            (loam, silty, "Silty Clay"),
        ]
        cursor = silty
        if dn > 0.3:  # northern holes: silt band under the silty clay
            units.append((cursor, round(cursor + 0.8, 2), "Silt"))
            cursor = round(cursor + 0.8, 2)
        if de > 0.25 and dn > -0.6:  # upper sand lens pinches out west and south
            units.append((cursor, round(cursor + 1.1 + 0.6 * de, 2), "Sand"))
            cursor = round(cursor + 1.1 + 0.6 * de, 2)
        clay_to = round(cursor + 1.6 - 0.4 * de, 2)
        units.append((cursor, clay_to, "Clay"))
        aquifer_to = round(clay_to + 2.4 + 0.4 * dn, 2)
        units.append((clay_to, aquifer_to, "Sand and Gravel"))
        cursor = aquifer_to
        if de > 0.6:
            units.append((cursor, round(cursor + 0.7, 2), "Gravel"))
            cursor = round(cursor + 0.7, 2)
        units.append((cursor, total, "Clay"))
        for start, end, code in units:
            rows.append(
                {"hole_id": hole, "from_depth": start, "to_depth": end, "lithology_code": code}
            )
    # One logging gap (no recovery) in BH-09 at the base of the upper sand,
    # just above the clay: the last 0.6 m of the sand interval is not logged.
    patched: list[dict[str, object]] = []
    for row in rows:
        if row["hole_id"] == GAP_HOLE and row["lithology_code"] == "Sand":
            patched.append({**row, "to_depth": round(float(row["to_depth"]) - GAP_LENGTH, 2)})
        else:
            patched.append(row)
    return patched


def _water() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    wells = [f"BH-{n:02d}" for n in range(1, 15) if n not in (2, 12)]
    for index, hole in enumerate(wells):
        for series_id, label, base in (
            ("2026-05-14", "May 14, 2026", 3.1),
            ("2026-09-22", "Sept 22, 2026", 3.9),
        ):
            rows.append(
                {
                    "hole_id": hole,
                    "depth": round(base + 0.12 * (index % 5), 2),
                    "elevation_masl": "",
                    "status": "measured",
                    "series_id": series_id,
                    "series_label": label,
                    "connect_group": "",
                }
            )
    # One well dry on the autumn round.
    rows.append(
        {
            "hole_id": "BH-12",
            "depth": "",
            "elevation_masl": "",
            "status": "dry",
            "series_id": "2026-09-22",
            "series_label": "Sept 22, 2026",
            "connect_group": "",
        }
    )
    return rows


CHLORIDE_GUIDELINE = 120.0  # mg/kg — green ≤ 120, yellow ≤ 400, red above
CHLORIDE_YELLOW_MAX = 400.0


def _environmental() -> list[dict[str, object]]:
    values = {
        "BH-03": (45.0, 380.0, 1250.0),
        "BH-04": (22.0, 160.0, 95.0),
        "BH-05": (610.0, 2400.0, 870.0),
        "BH-08": (12.0, 30.0, 18.0),
        "BH-09": (88.0, 410.0, 140.0),
        "BH-12": (150.0, 520.0, 260.0),
        "BH-13": (1900.0, 3100.0, 780.0),
    }
    rows: list[dict[str, object]] = []
    for hole, triple in values.items():
        for depth, value in zip((0.75, 2.25, 4.5), triple, strict=True):
            rows.append(
                {
                    "hole_id": hole,
                    "parameter": "Chloride",
                    "value": value,
                    "depth": depth,
                    "from_depth": "",
                    "to_depth": "",
                    "unit": "mg/kg",
                    "value_label": "",
                    "label_color": "",
                }
            )
    return rows


def _screens(collars: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for collar in collars:
        number = int(str(collar["hole_id"])[-2:])
        if number in (2, 12):
            continue  # test holes, not completed as wells
        total = float(collar["total_depth"])
        rows.append(
            {
                "hole_id": collar["hole_id"],
                "from_depth": round(total - 4.5, 2),
                "to_depth": round(total - 1.5, 2),
            }
        )
    return rows


PROJECT_VALUES = {
    "client_name": "PRAIRIE RIDGE ENERGY LTD.",
    "prepared_by": "NORTHFIELD ENVIRONMENTAL",
    "project_number": "NE-24-0187",
    "section_title": "A - A' WITH CHLORIDE",
    "report_date": "09/30/26",
    "drawn_by": "AL",
    "data_source": "NORTHFIELD 2026",
    "map_scale": "1:2000",
    "coordinate_reference": "EPSG:26912",
    "transect_start": "A / WEST",
    "transect_end": "A' / EAST",
    "vertical_exaggeration": "5",
    "figure_preset": "consulting_report",
    "notes": "NOTE: masl DENOTES METRES ABOVE SEA LEVEL.",
}


def _replace_sheet_rows(sheet, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    """Keep the template header (and its hint column); replace the data rows."""
    header = [cell.value for cell in sheet[1][: len(columns)]]
    assert tuple(header) == columns, (sheet.title, header)
    if sheet.max_row > 1:
        sheet.delete_rows(2, sheet.max_row - 1)
    for row_index, row in enumerate(rows, start=2):
        for col_index, column in enumerate(columns, start=1):
            value = row.get(column, "")
            sheet.cell(row=row_index, column=col_index, value=None if value == "" else value)


# Site-wide stratigraphic column a geologist would enter in unit_order (the
# upper and lower Clay are different units, so they need their own numbers).
SITE_UNIT_ORDER = {
    "Topsoil": 1,
    "Clay Loam": 2,
    "Silty Clay": 3,
    "Silt": 4,
    "Sand": 5,
    "Clay": 6,  # upper clay; the lower clay below the aquifer is 9
    "Sand and Gravel": 7,
    "Gravel": 8,
}


def _site_unit_orders(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    seen_clay: set[str] = set()
    for row in rows:
        code = str(row["lithology_code"])
        order = SITE_UNIT_ORDER[code]
        if code == "Clay":
            hole = str(row["hole_id"])
            order = 9 if hole in seen_clay else 6
            seen_clay.add(hole)
        out.append({**row, "unit_order": order})
    return out


def build_site_workbook_bytes(*, site_unit_order: bool = True) -> bytes:
    """Fill the input template with the synthetic site.

    ``site_unit_order`` fills the Lithology ``unit_order`` column with one
    site-wide column; False leaves it blank (the app then numbers each hole's
    intervals by depth on upload).
    """
    import openpyxl

    from workbook_template import (
        COLLAR_COLUMNS,
        ENVIRONMENTAL_COLUMNS,
        GRADIENT_COLUMNS,
        LITHOLOGY_COLUMNS,
        SCREEN_COLUMNS,
        SECTION_COLUMNS,
        WATER_COLUMNS,
        build_input_template_bytes,
    )

    book = openpyxl.load_workbook(BytesIO(build_input_template_bytes()))
    collars = _collars()
    project = book["Project"]
    for row in project.iter_rows(min_row=2):
        key = row[0].value
        if key in PROJECT_VALUES:
            row[2].value = PROJECT_VALUES[key]
    _replace_sheet_rows(book["Collars"], COLLAR_COLUMNS, collars)
    _replace_sheet_rows(
        book["Lithology"],
        LITHOLOGY_COLUMNS,
        _site_unit_orders(_lithology(collars))
        if site_unit_order
        else [{**row, "unit_order": ""} for row in _lithology(collars)],
    )
    _replace_sheet_rows(book["Water"], WATER_COLUMNS, _water())
    _replace_sheet_rows(book["Environmental"], ENVIRONMENTAL_COLUMNS, _environmental())
    _replace_sheet_rows(book["Screens"], SCREEN_COLUMNS, _screens(collars))
    _replace_sheet_rows(book["Gradients"], GRADIENT_COLUMNS, [])
    _replace_sheet_rows(
        book["Sections"],
        SECTION_COLUMNS,
        [
            # Mixed separators, as people type them.
            {"section_label": "A-A'", "hole_ids": ", ".join(SECTIONS["A-A'"])},
            {"section_label": "B-B'", "hole_ids": " → ".join(SECTIONS["B-B'"])},
            {"section_label": "C-C'", "hole_ids": "; ".join(SECTIONS["C-C'"])},
        ],
    )
    buffer = BytesIO()
    book.save(buffer)
    return buffer.getvalue()


@pytest.fixture(scope="module")
def site_bytes() -> bytes:
    return build_site_workbook_bytes()


@pytest.fixture(scope="module")
def site_ingest(site_bytes: bytes):
    from ingestion import ingest_workbook

    return ingest_workbook(BytesIO(site_bytes))


# ---------------------------------------------------------------------------
# 1. Ingestion / parsing
# ---------------------------------------------------------------------------


def test_sections_tab_parses_three_transects_in_hole_order(site_ingest) -> None:
    parse_result, report = site_ingest
    assert report.profile_id == "data_entry_template"
    assert parse_result.errors == ()
    assert report.hole_count == 14
    labels = [spec.label for spec in parse_result.section_specs]
    assert labels == list(SECTIONS)
    for spec in parse_result.section_specs:
        assert spec.hole_ids == SECTIONS[spec.label], spec.label  # comma, → and ; all split
    assert [spec.label for spec in report.section_specs] == labels
    # Intersections: each shared hole is listed on both of its sections.
    for hole, members in SHARED_HOLES.items():
        on = {spec.label for spec in parse_result.section_specs if hole in spec.hole_ids}
        assert on == set(members), hole
    assert any("Sections sheet: loaded 3 transect" in warning for warning in report.warnings)


def test_site_data_loads_without_spurious_errors(site_ingest) -> None:
    parse_result, report = site_ingest
    # Explicit site-wide unit_order is kept as entered (no per-hole renumbering).
    assert not any("assigned unit_order" in warning for warning in report.warnings)
    assert {lith.lithology_code for lith in parse_result.lithologies} == {
        "Topsoil",
        "Clay Loam",
        "Silty Clay",
        "Silt",
        "Sand",
        "Clay",
        "Sand and Gravel",
        "Gravel",
    }
    qa = report.quality_report
    assert qa is not None and qa.error_count == 0 and not qa.unmapped_lithologies
    # The one deliberate logging gap is the only data-check finding.
    assert [(issue.code, issue.hole_id) for issue in qa.issues] == [("depth_gap", GAP_HOLE)]
    assert {level.series_id for level in parse_result.water_levels} == {"2026-05-14", "2026-09-22"}
    assert any(
        level.hole_id == "BH-12" and level.status == "dry" for level in parse_result.water_levels
    )
    chloride = [r for r in parse_result.environmental_readings if r.parameter == "Chloride"]
    assert len(chloride) == 21 and {r.unit for r in chloride} == {"mg/kg"}
    assert len(parse_result.screen_intervals) == 12
    assert report.project_metadata["client_name"] == PROJECT_VALUES["client_name"]


# ---------------------------------------------------------------------------
# 2. Batch export from the Sections tab
# ---------------------------------------------------------------------------


def _batch_specs(parse_result):
    from batch_export import parse_batch_transect_lines, split_batch_transect_lines
    from parse_ops import format_section_specs_as_batch_text

    text = format_section_specs_as_batch_text(parse_result.section_specs)
    specs, skipped = split_batch_transect_lines(text, [c.hole_id for c in parse_result.collars])
    assert skipped == []
    assert specs == parse_batch_transect_lines(text)
    assert [(spec.label, spec.hole_ids) for spec in specs] == list(SECTIONS.items())
    return specs


def _base_request(layout: str):
    from models import ConsultingTitleBlock
    from section_build_request import SectionBuildRequest

    consulting = layout == "consulting_section"
    return SectionBuildRequest(
        transect_points=((E0, N0), (E0 + 1.0, N0)),  # replaced per transect
        vertical_exaggeration=5.0,
        show_hatches=True,
        section_title=PROJECT_VALUES["section_title"],
        render_layout=layout,
        allow_pinch_outs=True,
        environmental_parameters=("Chloride",),
        chemistry_color_mode="threshold",
        chemistry_threshold_green_max=CHLORIDE_GUIDELINE,
        chemistry_threshold_yellow_max=CHLORIDE_YELLOW_MAX,
        selected_water_series_ids=("2026-05-14", "2026-09-22"),
        consulting_title_block=(
            ConsultingTitleBlock(
                section_label="A-A'",
                client_name=PROJECT_VALUES["client_name"],
                project_number=PROJECT_VALUES["project_number"],
            )
            if consulting
            else None
        ),
    )


def _pdf_pages(payload: bytes) -> int:
    from pypdf import PdfReader

    return len(PdfReader(BytesIO(payload)).pages)


@pytest.mark.parametrize("layout", ["consulting_section", "section_sheet"])
def test_batch_zip_builds_every_workbook_section(site_ingest, layout: str, tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    from batch_export import (
        build_batch_zip,
        build_multi_transect_exports,
        clear_batch_geometry_memo,
        export_binder_pdf,
    )

    parse_result, _report = site_ingest
    specs = _batch_specs(parse_result)
    clear_batch_geometry_memo()
    entries = build_multi_transect_exports(
        parse_result,
        _base_request(layout),
        specs,
        export_formats=frozenset({"png", "pdf", "svg"}),
    )
    assert [stem for stem, *_ in entries] == list(SECTIONS)
    pngs = [png for _stem, _svg, png, _pdf in entries]
    assert all(png.startswith(b"\x89PNG") for png in pngs)
    assert len(set(pngs)) == 3
    for label, svg, _png, pdf in entries:
        assert pdf.startswith(b"%PDF")
        text = svg.decode("utf-8")
        # Each sheet is titled with its own line, never another one's (the
        # base title "A - A' WITH CHLORIDE" is relabelled per line).
        assert label in text, label
        for other in SECTIONS:
            if other != label:
                assert other not in text, (label, other)
        for hole in SECTIONS[label]:
            assert hole in text, (label, hole)

    binder = export_binder_pdf([pdf for *_rest, pdf in entries], cover_title="Prairie Ridge")
    section_pages = [_pdf_pages(pdf) for *_rest, pdf in entries]
    assert _pdf_pages(binder) == 1 + sum(section_pages)

    zip_bytes = build_batch_zip(entries, binder_pdf=binder)
    with zipfile.ZipFile(BytesIO(zip_bytes)) as archive:
        names = set(archive.namelist())
        expected = {
            f"{stem}.{ext}" for stem in ("A-A", "B-B", "C-C") for ext in ("png", "pdf", "svg")
        }
        assert names == expected | {"report_binder.pdf", "README.txt"}
        assert "3 section line(s)" in archive.read("README.txt").decode("utf-8")
        archive.extractall(tmp_path)  # stems are filesystem-safe
    assert (tmp_path / "A-A.png").stat().st_size > 10_000


def _section_geometry(parse_result, holes):
    from batch_export import transect_points_from_collars
    from parse_ops import subset_parse_result
    from pipeline import compute_section_geometry

    subset = subset_parse_result(parse_result, holes)
    return compute_section_geometry(
        subset.collars,
        subset.lithologies,
        transect_points_from_collars(parse_result.collars, holes),
        allow_pinch_outs=False,
        fail_on_overlaps=False,
        warn_on_correlation_gaps=False,
    )


def _continuous_pairs(geometry, code: str) -> int:
    return sum(1 for p in geometry.polygons if p.lithology_code == code and not p.is_pinch_out)


@pytest.mark.parametrize("label", list(SECTIONS))
def test_site_wide_unit_order_correlates_units_across_each_section(site_ingest, label: str) -> None:
    """Units present in every hole join up between every pair of neighbours."""
    parse_result, _report = site_ingest
    geometry = _section_geometry(parse_result, SECTIONS[label])
    pairs = len(SECTIONS[label]) - 1
    for code in ("Topsoil", "Clay Loam", "Silty Clay", "Sand and Gravel"):
        assert _continuous_pairs(geometry, code) == pairs, (label, code)
    # Upper and lower clay both span the whole line.
    assert _continuous_pairs(geometry, "Clay") == 2 * pairs, label
    assert not geometry.overlap_pairs


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known gap: with no unit_order column, parse_ops.assign_missing_unit_orders numbers "
        "each hole 1..n by depth, and stratigraphy keys on (unit_order, code), so a unit "
        "present in only some holes (Silt, upper Sand, Gravel) shifts every deeper number "
        "and the aquifer stops correlating with its neighbours."
    ),
)
def test_auto_unit_order_still_correlates_the_aquifer() -> None:
    from ingestion import ingest_workbook

    parse_result, report = ingest_workbook(
        BytesIO(build_site_workbook_bytes(site_unit_order=False))
    )
    assert any("assigned unit_order" in warning for warning in report.warnings)
    for label, holes in SECTIONS.items():
        geometry = _section_geometry(parse_result, holes)
        assert _continuous_pairs(geometry, "Sand and Gravel") == len(holes) - 1, label


# ---------------------------------------------------------------------------
# 3. Streamlit: upload → Sections picker + batch lines → Generate → batch ZIP
# ---------------------------------------------------------------------------


def test_streamlit_upload_seeds_batch_lines_and_builds_zip(site_bytes: bytes) -> None:
    from streamlit.testing.v1 import AppTest

    from batch_export import clear_batch_geometry_memo

    clear_batch_geometry_memo()
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    at.file_uploader[0].upload("prairie_ridge.xlsx", site_bytes).run()
    assert not at.exception
    assert not any("template's sample values" in w.value for w in at.warning)
    at.run()
    assert not at.exception
    assert sorted(at.session_state["hole_ids"]) == [f"BH-{n:02d}" for n in range(1, 15)]

    picker = at.selectbox(key="workbook_section_choice")
    assert len(picker.options) == 4  # Custom + A-A', B-B', C-C'
    assert at.session_state["hole_sequence_multiselect"] == list(SECTIONS["A-A'"])

    expected_lines = "\n".join(f"{label} | {', '.join(holes)}" for label, holes in SECTIONS.items())
    # An empty batch box is seeded once from the Sections tab …
    assert str(at.session_state["batch_transect_specs"]) == expected_lines
    assert any(s.value == "3 of 3 section lines ready." for s in at.success)
    # … and Load from workbook Sections restores it after an edit.
    at.session_state["batch_transect_specs"] = "A-A' | BH-01, BH-02"
    at.run()
    at.button(key="batch_load_sections").click().run()
    assert not at.exception
    assert str(at.session_state["batch_transect_specs"]) == expected_lines

    # Preview B-B' from the picker (row index 1), then generate it.
    at.selectbox(key="workbook_section_choice").select(1).run()
    at.run()
    assert at.session_state["hole_sequence_multiselect"] == list(SECTIONS["B-B'"])
    for box in at.checkbox:
        if "Generate even if data checks found warnings" in (box.label or ""):
            box.set_value(True)  # the BH-09 logging gap is a warning
            break
    at.run()
    generate = [btn for btn in at.button if btn.label == "Generate section"]
    assert generate and not generate[0].disabled
    generate[0].click().run()
    assert not at.exception
    assert at.session_state["svg_bytes"]

    at.button(key="prepare_batch_zip").click().run()
    assert not at.exception
    package = at.session_state["batch_package_bytes"]
    assert package
    with zipfile.ZipFile(BytesIO(package)) as archive:
        names = set(archive.namelist())
    # Each file is named after its own section, never the previewed one
    # (B-B' here used to prefix every name: B-B_A-A.png).
    pngs = sorted(name for name in names if name.endswith(".png"))
    pdfs = sorted(name for name in names if name.endswith(".pdf") and name != "report_binder.pdf")
    assert len(pngs) == 3 and len(pdfs) == 3, names
    for stem in ("A-A", "B-B", "C-C"):
        assert any(name.endswith(f"{stem}.png") for name in pngs), names
    assert not any("B-B_" in name for name in names), names
    zip_name = at.session_state["_batch_zip_name"]
    assert all(label in zip_name for label in ("A-A", "B-B", "C-C")), zip_name
    assert {"README.txt", "report_binder.pdf"} <= names
    assert any("Batch ZIP ready: 3 section line" in s.value for s in at.success)
