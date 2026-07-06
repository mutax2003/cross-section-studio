"""Transect registry for EcoVenture GWM reference figures."""

from __future__ import annotations

from dataclasses import dataclass

from models import ConsultingTitleBlock

PROJECT_NUMBER = "100/09-29 & 100/16-29-057-19 W4M"
PREPARED_FOR = "C-GROUP ENERGY INC."
PREPARED_BY = "ECOVENTURE"
SOURCE = "ECOVENTURE 2025"
REPORT_DATE = "05/11/26"
DRAWN_BY = "SL/JG"
REVISED = "EC 04/12/22-xsec"


@dataclass(frozen=True)
class TransectSpec:
    transect_id: str
    figure_number: str
    hole_ids: tuple[str, ...]
    profile_eastings: tuple[float, ...]
    title_block: ConsultingTitleBlock
    vertical_exaggeration: float = 5.0
    output_stem: str = ""

    def __post_init__(self) -> None:
        if len(self.hole_ids) != len(self.profile_eastings):
            raise ValueError(
                f"Transect {self.transect_id}: hole_ids and profile_eastings length mismatch"
            )
        if len(self.hole_ids) < 2:
            raise ValueError(f"Transect {self.transect_id}: requires at least two holes")


def _title_block(
    *,
    section_label: str,
    figure_number: str,
    map_scale: str,
    scale_bar_m: float,
    start_primary: str,
    start_secondary: str,
    end_primary: str,
    end_secondary: str,
    notes: tuple[str, ...] = ("NOTE: masl DENOTES METRES ABOVE SEA LEVEL.",),
) -> ConsultingTitleBlock:
    return ConsultingTitleBlock(
        section_label=section_label,
        transect_start_primary=start_primary,
        transect_start_secondary=start_secondary,
        transect_end_primary=end_primary,
        transect_end_secondary=end_secondary,
        map_scale=map_scale,
        scale_bar_m=scale_bar_m,
        figure_number=figure_number,
        project_number=PROJECT_NUMBER,
        source=SOURCE,
        date=REPORT_DATE,
        notes=notes,
        drawn_by=DRAWN_BY,
        revised=REVISED,
        prepared_for=PREPARED_FOR,
        prepared_by=PREPARED_BY,
        screen_legend_label="SCREENED INTERVAL",
        y_axis_label="ELEVATION ABOVE SEA LEVEL (MASL)",
        show_gradient_legend=False,
    )


GWM_TRANSECTS: dict[str, TransectSpec] = {
    "A_A": TransectSpec(
        transect_id="A_A",
        figure_number="3",
        hole_ids=(
            "MW18-18",
            "MW18-06B",
            "MW18-16",
            "BH18-05",
            "MW18-08D",
            "MW18-24",
        ),
        profile_eastings=(0.0, 32.0, 64.0, 96.0, 128.0, 160.0),
        title_block=_title_block(
            section_label="A - A' WITH GROUNDWATER LEVELS",
            figure_number="3",
            map_scale="1:1 000",
            scale_bar_m=30.0,
            start_primary="A",
            start_secondary="NORTHWEST",
            end_primary="A'",
            end_secondary="SOUTHEAST",
        ),
        output_stem="fig_3_cross_section_a_a",
    ),
    "B_B": TransectSpec(
        transect_id="B_B",
        figure_number="4",
        hole_ids=(
            "MW18-17",
            "BH18-03",
            "MW18-20",
            "BH18-08",
            "BH18-02",
            "MW18-08D",
        ),
        profile_eastings=(0.0, 116.0, 232.0, 348.0, 464.0, 580.0),
        title_block=_title_block(
            section_label="B - B'",
            figure_number="4",
            map_scale="1:1 500",
            scale_bar_m=40.0,
            start_primary="B",
            start_secondary="NORTH",
            end_primary="B'",
            end_secondary="SOUTH",
        ),
        output_stem="fig_4_cross_section_b_b",
    ),
    "C_C": TransectSpec(
        transect_id="C_C",
        figure_number="5",
        hole_ids=(
            "BH18-07",
            "MW18-20",
            "BH18-03",
            "BH18-04",
            "MW18-21",
            "MW18-16",
        ),
        profile_eastings=(0.0, 116.0, 232.0, 348.0, 464.0, 580.0),
        title_block=_title_block(
            section_label="C - C'",
            figure_number="5",
            map_scale="1:1 500",
            scale_bar_m=40.0,
            start_primary="C",
            start_secondary="NORTH",
            end_primary="C'",
            end_secondary="SOUTH",
        ),
        output_stem="fig_5_cross_section_c_c",
    ),
    "D_D": TransectSpec(
        transect_id="D_D",
        figure_number="6",
        hole_ids=(
            "MW18-19",
            "BH18-09",
            "MW18-22",
            "MW18-23",
        ),
        profile_eastings=(0.0, 180.0, 360.0, 540.0),
        title_block=_title_block(
            section_label="D - D'",
            figure_number="6",
            map_scale="1:1 500",
            scale_bar_m=40.0,
            start_primary="D",
            start_secondary="NORTHWEST",
            end_primary="D'",
            end_secondary="SOUTHEAST",
        ),
        output_stem="fig_6_cross_section_d_d",
    ),
}
