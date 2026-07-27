"""Transect registry for Advantage Phase 2 chloride cross-sections (Figs 6–7).

Chainage and presentation defaults mirror the client PDF extracts under
``data/pdf_extract_p2/`` (depth sticks, ~30 m span, VE = 1).
"""

from __future__ import annotations

from dataclasses import dataclass

from models import ConsultingTitleBlock

PROJECT_NUMBER = "100/09-36-055-02 W4M"
PREPARED_FOR = "C-GROUP ENERGY INC."
PREPARED_BY = "ECOVENTURE"
SOURCE = "ECOVENTURE 2026"
REPORT_DATE = "06/24/26"
DRAWN_BY = "SL"
REVISED = "EC 06/24/26-xsec"


@dataclass(frozen=True)
class TransectSpec:
    transect_id: str
    figure_number: str
    hole_ids: tuple[str, ...]
    profile_eastings: tuple[float, ...]
    title_block: ConsultingTitleBlock
    vertical_exaggeration: float = 1.0
    elevation_mode: str = "relative"
    interpretation_mode: str = "borehole_only"
    parameter_interpolate_segments: bool = False
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
        notes=("NOTE: mbgs DENOTES METRES BELOW GROUND SURFACE.",),
        drawn_by=DRAWN_BY,
        revised=REVISED,
        prepared_for=PREPARED_FOR,
        prepared_by=PREPARED_BY,
        screen_legend_label="SCREENED INTERVAL",
        y_axis_label="DEPTH (mbgs)",
        show_gradient_legend=False,
    )


ADVANTAGE_P2_TRANSECTS: dict[str, TransectSpec] = {
    "A_A": TransectSpec(
        transect_id="A_A",
        figure_number="6",
        hole_ids=(
            "BH23-10",
            "2017-BH05",
            "2017-BH18",
            "2017-BH11 / BH24-11",
            "2017-BH09",
            "2017-BH10",
            "BH23-07",
        ),
        # Approximate client Fig 6 chainage (distance axis ~0–32 m).
        profile_eastings=(1.0, 8.0, 11.0, 15.0, 25.0, 28.0, 32.0),
        title_block=_title_block(
            section_label="A-A'",
            figure_number="6",
            map_scale="1:200",
            scale_bar_m=5.0,
            start_primary="A",
            start_secondary="WEST",
            end_primary="A'",
            end_secondary="EAST",
        ),
        output_stem="fig_6_cross_section_a_a",
    ),
    "B_B": TransectSpec(
        transect_id="B_B",
        figure_number="7",
        hole_ids=(
            "BH24-12 / HA25-02",
            "BH23-04",
            "2017-BH11 / BH24-11",
            "BH23-03",
            "2017-BH12",
            "BH24-08",
            "BH23-01",
        ),
        # Compact ~30 m span matching client Fig 7 distance ticks.
        profile_eastings=(0.0, 5.0, 10.0, 15.0, 20.0, 26.0, 32.0),
        title_block=_title_block(
            section_label="B-B'",
            figure_number="7",
            map_scale="1:200",
            scale_bar_m=5.0,
            start_primary="B",
            start_secondary="SOUTH",
            end_primary="B'",
            end_secondary="NORTH",
        ),
        output_stem="fig_7_cross_section_b_b",
    ),
}
