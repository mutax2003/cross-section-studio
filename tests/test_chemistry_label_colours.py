"""Concentration label colours, readability styles and overlap (meeting 1 Oct 2026)."""

from __future__ import annotations

import itertools
import zipfile
from io import BytesIO

import pandas as pd
import pytest

from models import Collar, EnvironmentalReading, Lithology, WaterLevel
from render_profiles import CONSULTING_SECTION_PROFILE, SECTION_SHEET_PROFILE
from render_theme import CHEMISTRY_FIXED_COLORS, parameter_series_color
from renderer import CrossSectionRenderer
from tests.conftest import run_pipeline


def test_label_colour_names_are_fixed_and_blue_is_reserved() -> None:
    base = dict(hole_id="BH-01", parameter="Chloride", value=5.0, depth=1.0)
    assert EnvironmentalReading(**base, label_color=" Green ").label_color == "green"
    assert EnvironmentalReading(**base, label_color=float("nan")).label_color == ""
    with pytest.raises(ValueError, match="reserved for groundwater"):
        EnvironmentalReading(**base, label_color="blue")
    with pytest.raises(ValueError, match="green, red, black or orange"):
        EnvironmentalReading(**base, label_color="yellow")
    assert set(CHEMISTRY_FIXED_COLORS) == {"green", "red", "black", "orange"}


def test_parameter_colour_is_stable_across_sections() -> None:
    """Colours used to follow the order parameters appeared on a section."""
    assert parameter_series_color("Chloride") == parameter_series_color(" chloride ")
    assert all(parameter_series_color(p).upper() != "#2563EB" for p in ("Chloride", "Benzene", "BTEX"))


def test_template_has_a_label_colour_drop_down_and_round_trips_it() -> None:
    from openpyxl import load_workbook

    from ingestion import ingest_workbook
    from workbook_template import (
        ENVIRONMENTAL_SHEET,
        build_input_template_bytes,
        export_cleaned_workbook_bytes,
    )

    book = load_workbook(BytesIO(build_input_template_bytes()))
    sheet = book[ENVIRONMENTAL_SHEET]
    header = [cell.value for cell in sheet[1]]
    assert "label_color" in header
    validations = sheet.data_validations.dataValidation
    assert any("green,red,black,orange" in str(v.formula1) for v in validations)

    collars = pd.DataFrame(
        [{"hole_id": "BH-01", "easting": 0, "northing": 0, "elevation": 100, "total_depth": 10}]
    )
    lith = pd.DataFrame([{"hole_id": "BH-01", "from_depth": 0, "to_depth": 10, "lithology_code": "Clay"}])
    env = pd.DataFrame(
        [
            {"hole_id": "BH-01", "parameter": "Chloride", "value": 12.0, "depth": 2.0, "label_color": "red"},
            {"hole_id": "BH-01", "parameter": "Chloride", "value": 1.0, "depth": 4.0, "label_color": "blue"},
        ]
    )
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        collars.to_excel(writer, sheet_name="Collars", index=False)
        lith.to_excel(writer, sheet_name="Lithology", index=False)
        env.to_excel(writer, sheet_name="Environmental", index=False)
    buffer.seek(0)
    result, _ = ingest_workbook(buffer)
    assert [r.label_color for r in result.environmental_readings] == ["red"]
    assert any("Environmental row 3" in e and "reserved" in e for e in result.errors)

    exported = export_cleaned_workbook_bytes(result, project_metadata={})
    again, _ = ingest_workbook(BytesIO(exported))
    assert again.environmental_readings[0].label_color == "red"
    assert zipfile.is_zipfile(BytesIO(exported))


def _chemistry_section(n_readings: int = 8):
    ids = ["BH-01", "BH-02", "BH-03"]
    collars = [Collar(hole_id=h, easting=i * 6.0, northing=0.0, elevation=100.0, total_depth=12.0) for i, h in enumerate(ids)]
    lith = [Lithology(hole_id=h, from_depth=0.0, to_depth=12.0, lithology_code="Sandy Clay") for h in ids]
    readings = [
        EnvironmentalReading(
            hole_id=h, parameter="Chloride", value=float(100 + k * 37), depth=0.5 + k * (11.0 / n_readings),
            label_color=["green", "red", "orange", "black", ""][k % 5],
        )
        for h in ids
        for k in range(n_readings)
    ]
    water = [WaterLevel(hole_id=h, depth=3.0 + i) for i, h in enumerate(ids)]
    projected, polygons, _ = run_pipeline(collars, lith, [(0.0, 0.0), (12.0, 0.0)])
    return ids, projected, polygons, readings, water


def _render(style: str, profile=SECTION_SHEET_PROFILE, **extra):
    ids, projected, polygons, readings, water = _chemistry_section()
    renderer = CrossSectionRenderer(
        show_legend=True,
        render_profile=profile.model_copy(
            update={
                "show_parameter_markers": True,
                "show_parameter_labels": True,
                "chemistry_label_style": style,
                "show_water_elevation_labels": True,
                **extra,
            }
        ),
        environmental_readings=readings,
        environmental_parameters=("Chloride",),
    )
    figure = renderer.render(polygons, projected, collar_depths={h: 12.0 for h in ids}, water_levels=water)
    return renderer, figure


def test_workbook_colours_reach_the_figure_and_labels_never_overlap() -> None:
    from matplotlib.text import Text

    from renderer_water import _figure_renderer, _overlap_area

    renderer, figure = _render("plain")
    mpl_renderer = _figure_renderer(figure)
    figure.draw_without_rendering()
    labels = [(kind, ann) for kind, ann, _c in renderer._water_labels]
    kinds = {kind for kind, _ in labels}
    assert "chem" in kinds and "rl" in kinds
    colours = {ann.get_color().lower() for kind, ann in labels if kind == "chem"}
    assert {CHEMISTRY_FIXED_COLORS["green"].lower(), CHEMISTRY_FIXED_COLORS["red"].lower()} <= colours
    boxes = [Text.get_window_extent(ann, mpl_renderer) for _, ann in labels]
    overlaps = sum(1 for a, b in itertools.combinations(boxes, 2) if _overlap_area(a, b) > 0)
    assert overlaps == 0
    for ax in figure.axes:
        frame = ax.get_window_extent(mpl_renderer)
        for box in boxes:
            if box.x0 >= frame.x0 - 1 and box.x1 <= frame.x1 + 1:
                continue
        # every label stays on the page
        assert all(box.x0 >= figure.bbox.x0 and box.x1 <= figure.bbox.x1 for box in boxes)


def test_readability_styles_change_the_label_rendering() -> None:
    _, plain = _render("plain")
    _, boxed = _render("box")
    _, dotted = _render("dot")
    _, stroked = _render("stroke")

    def chem_texts(fig):
        return [t for ax in fig.axes for t in ax.texts if t.get_text().strip().isdigit()]

    # Plain keeps the translucent marker-mode box; "box" is fully opaque.
    assert all((t.get_bbox_patch() is None or t.get_bbox_patch().get_alpha() < 1.0) for t in chem_texts(plain))
    assert all(t.get_bbox_patch() is not None and t.get_bbox_patch().get_alpha() == 1.0 for t in chem_texts(boxed))
    assert all(t.get_path_effects() for t in chem_texts(stroked))
    assert all(t.get_color().lower() == "#111827" for t in chem_texts(dotted))  # black text, colour on the dot
    dots = [line for ax in dotted.axes for line in ax.lines if line.get_marker() == "o" and line.get_linestyle() == "None"]
    assert len(dots) >= len(chem_texts(dotted))


def test_consulting_layout_also_registers_chemistry_labels() -> None:
    renderer, _ = _render("box", profile=CONSULTING_SECTION_PROFILE)
    assert any(kind == "chem" for kind, _a, _c in renderer._water_labels)
