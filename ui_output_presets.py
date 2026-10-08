"""Map user-facing output presets to render configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

InterpretationMode = Literal["interpolated", "correlation_lines", "borehole_only"]
ElevationMode = Literal["absolute", "relative"]

# Vertical exaggeration (VE) choice: "auto" fits the page and prints the VE
# the figure really ends up with; a number draws exactly that VE.
VE_AUTO = "auto"
VEChoice = float | Literal["auto"]
VE_CHOICES: tuple[VEChoice, ...] = (VE_AUTO, 1.0, 2.0, 5.0, 10.0, 20.0)
DEFAULT_VE_CHOICE: VEChoice = VE_AUTO


def normalize_ve_choice(value: object) -> VEChoice:
    """Session / workbook value -> ``"auto"`` or a positive float (bad input -> auto)."""
    if value is None:
        return VE_AUTO
    if isinstance(value, str):
        text = value.strip().lower().rstrip("x×").strip()
        if text in {"", "auto", "fit", "fit page", "auto (fit page)"}:
            return VE_AUTO
        value = text
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return VE_AUTO
    if not number > 0 or number != number or number == float("inf"):
        return VE_AUTO
    return number


def ve_choice_to_request(value: object) -> float | None:
    """VE for ``SectionBuildRequest`` / ``build_cross_section``: None = auto."""
    choice = normalize_ve_choice(value)
    return None if choice == VE_AUTO else float(choice)


def ve_choice_options(current: object = None) -> list[VEChoice]:
    """Select options; a custom workbook value (e.g. 3×) is kept in the list."""
    options = list(VE_CHOICES)
    choice = normalize_ve_choice(current)
    if choice != VE_AUTO and choice not in options:
        options.append(choice)
        options = [VE_AUTO, *sorted(o for o in options if o != VE_AUTO)]  # type: ignore[type-var]
    return options


def ve_choice_label(value: object) -> str:
    """Sidebar label: "Auto (fit page)", "1× (true scale)", "5×"."""
    choice = normalize_ve_choice(value)
    if choice == VE_AUTO:
        return "Auto (fit page)"
    if choice == 1.0:
        return "1× (true scale)"
    return f"{choice:g}×"


def ve_short_text(value: object) -> str:
    """Compact VE for chips / alt text: "auto" or "5×"."""
    choice = normalize_ve_choice(value)
    return "auto (fit page)" if choice == VE_AUTO else f"{choice:g}×"


@dataclass(frozen=True)
class OutputPresetConfig:
    render_layout: str
    report_preset: bool
    allow_pinch_outs: bool
    show_ground_surface: bool
    interpolate_water_table: bool
    show_legend: bool
    # Sample-figure profiles (GWM fence / P2 sticks). None = leave sidebar free.
    interpretation_mode: InterpretationMode | None = None
    elevation_mode: ElevationMode | None = None
    # None = leave the sidebar choice alone; "auto" = seed Auto (fit page,
    # still editable); a number = exact VE (locked on sample figures).
    vertical_exaggeration: VEChoice | None = None
    show_water_elevation_labels: bool | None = None
    show_water_legend: bool | None = None
    show_dry_well_nm: bool | None = None
    water_interpolate_across_gaps: bool | None = None
    # When True, consulting layout does not force water interpolation / pinch-outs off.
    sample_figure_profile: bool = False
    # Chemistry (Configure step defaults when preset is active).
    prefer_chemistry: bool = False
    show_parameter_labels: bool | None = None
    parameter_interpolate_segments: bool | None = None
    parameter_draw_markers: bool | None = None
    # Wave A drafting chrome (None = layout profile default).
    show_scale_bar: bool | None = None
    show_ve_annotation: bool | None = None
    show_parameter_legend_text: bool | None = None
    # Wave B: dashed CAD-style water connectors when False.
    water_line_solid: bool | None = None
    # Default chemistry value-label colour mode for the preset (None = black).
    # Workbook per-reading label_color still wins.
    chemistry_color_mode: str | None = None


OUTPUT_PRESET_LABELS: dict[str, str] = {
    "section_sheet": "Section sheet",
    "consulting_report": "Consulting report (title block)",
    "gwm_fence": "Groundwater fence (elevation + water levels)",
    "p2_chemistry_sticks": "Chemistry columns (depth + lab values)",
    "chemistry_gw": "Chemistry + groundwater (elevation)",
    "quick_preview": "Quick preview",
}

OUTPUT_PRESETS: dict[str, OutputPresetConfig] = {
    "section_sheet": OutputPresetConfig(
        render_layout="section_sheet",
        report_preset=True,
        allow_pinch_outs=False,
        show_ground_surface=True,
        interpolate_water_table=False,
        show_legend=True,
        show_scale_bar=False,
        show_ve_annotation=False,
        show_parameter_legend_text=False,
    ),
    "consulting_report": OutputPresetConfig(
        render_layout="consulting_section",
        report_preset=False,
        # Generic consulting builds force pinch-outs off (app_build); seed the
        # toggle to match so the sidebar shows what the figure uses.
        allow_pinch_outs=False,
        show_ground_surface=True,
        interpolate_water_table=True,
        show_legend=False,
        show_scale_bar=True,
    ),
    "gwm_fence": OutputPresetConfig(
        render_layout="consulting_section",
        report_preset=False,
        allow_pinch_outs=False,
        show_ground_surface=True,
        interpolate_water_table=True,
        show_legend=False,
        interpretation_mode="interpolated",
        elevation_mode="absolute",
        # Fit the page; the band prints the true (measured) VE.
        vertical_exaggeration=VE_AUTO,
        show_water_elevation_labels=True,
        show_water_legend=True,
        show_dry_well_nm=True,
        water_interpolate_across_gaps=False,
        sample_figure_profile=True,
        prefer_chemistry=False,
        show_scale_bar=True,
        water_line_solid=True,
    ),
    "p2_chemistry_sticks": OutputPresetConfig(
        render_layout="consulting_section",
        report_preset=False,
        allow_pinch_outs=False,
        show_ground_surface=True,
        interpolate_water_table=False,
        show_legend=False,
        interpretation_mode="borehole_only",
        elevation_mode="relative",
        # Client Figs 6/7: NO VERTICAL EXAGGERATION (drawn at exactly 1x).
        vertical_exaggeration=1.0,
        show_water_elevation_labels=False,
        show_water_legend=False,
        show_dry_well_nm=False,
        water_interpolate_across_gaps=False,
        sample_figure_profile=True,
        prefer_chemistry=True,
        show_parameter_labels=True,
        parameter_interpolate_segments=False,
        parameter_draw_markers=False,
        show_scale_bar=True,
        # Client Figs 6/7 print chloride values (and the legend sample) in red.
        chemistry_color_mode="red",
    ),
    "chemistry_gw": OutputPresetConfig(
        render_layout="consulting_section",
        report_preset=False,
        allow_pinch_outs=False,
        show_ground_surface=True,
        interpolate_water_table=True,
        show_legend=False,
        interpretation_mode="borehole_only",
        elevation_mode="absolute",
        # Fit the page; the band prints the true (measured) VE.
        vertical_exaggeration=VE_AUTO,
        show_water_elevation_labels=True,
        show_water_legend=True,
        show_dry_well_nm=True,
        water_interpolate_across_gaps=False,
        sample_figure_profile=True,
        prefer_chemistry=True,
        show_parameter_labels=True,
        parameter_interpolate_segments=False,
        parameter_draw_markers=False,
        show_scale_bar=True,
        water_line_solid=False,
    ),
    "quick_preview": OutputPresetConfig(
        render_layout="chart",
        report_preset=False,
        allow_pinch_outs=True,
        show_ground_surface=True,
        interpolate_water_table=False,
        show_legend=True,
        show_scale_bar=True,
    ),
}

FIGURE_PRESET_IDS: frozenset[str] = frozenset(
    {"gwm_fence", "p2_chemistry_sticks", "chemistry_gw"}
)


def resolve_output_preset(preset: str) -> OutputPresetConfig:
    return OUTPUT_PRESETS.get(preset, OUTPUT_PRESETS["section_sheet"])


def normalize_figure_preset(raw: str | None) -> str | None:
    """Map Project-sheet figure_preset / section_style to a known output preset id."""
    if raw is None:
        return None
    key = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "gwm": "gwm_fence",
        "gwm_fence": "gwm_fence",
        "ecoventure_gwm": "gwm_fence",
        "p2": "p2_chemistry_sticks",
        "p2_chemistry_sticks": "p2_chemistry_sticks",
        "p2_sticks": "p2_chemistry_sticks",
        "advantage_p2": "p2_chemistry_sticks",
        "chemistry_sticks": "p2_chemistry_sticks",
        "chemistry_gw": "chemistry_gw",
        "p2_gw": "chemistry_gw",
        "chemistry_and_groundwater": "chemistry_gw",
    }
    resolved = aliases.get(key, key)
    return resolved if resolved in OUTPUT_PRESETS else None


INTERPRETATION_LABELS: dict[str, str] = {
    "interpolated": "Connect layers between holes",
    "correlation_lines": "Contact lines only (no shading)",
    "borehole_only": "Observed logs only (no fill between holes)",
}


def output_preset_short_name(preset: str) -> str:
    """Display name without the parenthetical, for "Set by …" captions."""
    label = OUTPUT_PRESET_LABELS.get(preset, OUTPUT_PRESET_LABELS["section_sheet"])
    return label.split(" (", 1)[0]


@dataclass(frozen=True)
class SidebarVisibility:
    """Which sidebar controls an output style actually honours.

    Mirrors ``app_build.effective_render_options`` and the sample-figure locks so
    the sidebar can hide (not just disable) settings the figure ignores.
    """

    interpretation_editable: bool
    pinch_outs_editable: bool
    ground_surface_editable: bool
    vertical_exaggeration_editable: bool
    groundwater_editable: bool
    chart_legend_editable: bool
    label_detail_editable: bool
    parameter_text_block_editable: bool
    chemistry_marker_size_editable: bool
    title_block_shown: bool
    export_shown: bool


def sidebar_visibility(preset: str) -> SidebarVisibility:
    config = resolve_output_preset(preset)
    consulting = config.render_layout == "consulting_section"
    return SidebarVisibility(
        interpretation_editable=config.interpretation_mode is None,
        # Generic consulting forces pinch-outs off; sample figures fix them.
        pinch_outs_editable=not consulting,
        # report_preset (section sheet) and consulting layouts force it on.
        ground_surface_editable=not (config.report_preset or consulting),
        # Only an exact VE is locked; an "auto" preset seeds Auto but stays editable.
        vertical_exaggeration_editable=not (
            config.sample_figure_profile
            and config.vertical_exaggeration is not None
            and config.vertical_exaggeration != VE_AUTO
        ),
        # Generic consulting forces groundwater chrome; sample figures fix it.
        groundwater_editable=not consulting,
        # Consulting layouts put the legend in the title block.
        chart_legend_editable=not consulting,
        label_detail_editable=not consulting,
        parameter_text_block_editable=not consulting,
        # Chemistry presets draw lab values as labels without dots.
        chemistry_marker_size_editable=config.parameter_draw_markers is not False,
        title_block_shown=consulting,
        export_shown=config.render_layout != "chart",
    )


def _join_plain(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def locked_figure_summary(preset: str) -> str | None:
    """One caption line describing the layer/surface settings a style fixes."""
    config = resolve_output_preset(preset)
    visibility = sidebar_visibility(preset)
    parts: list[str] = []
    if not visibility.interpretation_editable and config.interpretation_mode is not None:
        parts.append(INTERPRETATION_LABELS[config.interpretation_mode].lower())
    if not visibility.pinch_outs_editable:
        parts.append(
            "layers that end between holes "
            + ("shown" if config.allow_pinch_outs and config.sample_figure_profile else "not drawn")
        )
    if not visibility.ground_surface_editable:
        parts.append("ground surface shown")
    if not visibility.vertical_exaggeration_editable and config.vertical_exaggeration is not None:
        parts.append(f"vertical exaggeration exactly {ve_short_text(config.vertical_exaggeration)}")
    if not visibility.chart_legend_editable:
        parts.append("lithology legend in the title block")
    if not parts:
        return None
    return f"Set by {output_preset_short_name(preset)}: {_join_plain(parts)}."


def locked_groundwater_summary(preset: str) -> str | None:
    """One caption line for the groundwater settings a style fixes (None = editable)."""
    config = resolve_output_preset(preset)
    if sidebar_visibility(preset).groundwater_editable:
        return None
    # Generic consulting forces the chrome on (see app_sidebar / app_build).
    forced = not config.sample_figure_profile

    def flag(value: bool | None) -> bool:
        return forced if value is None else bool(value)

    joined = True if forced else config.interpolate_water_table
    on: list[str] = []
    off: list[str] = []
    for name, value in (
        ("water level labels", flag(config.show_water_elevation_labels)),
        ("groundwater legend", flag(config.show_water_legend)),
        ("'not measured' markers for dry wells", flag(config.show_dry_well_nm)),
    ):
        (on if value else off).append(name)
    pieces = [
        "water table joined between wells" if joined else "water levels shown at wells only"
    ]
    if on:
        pieces.append(_join_plain(on) + " on")
    if off:
        pieces.append(_join_plain(off) + " off")
    return f"Set by {output_preset_short_name(preset)}: {'; '.join(pieces)}."
