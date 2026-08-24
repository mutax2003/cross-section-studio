"""Groundwater / water-table drawing mixin for CrossSectionRenderer."""

from __future__ import annotations

from typing import Sequence, TypedDict

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from models import WaterLevel
from render_theme import (
    CONSULTING_NM_COLOR,
    CONSULTING_WATER_COLOR,
    LABEL_COLOR,
    WATER_COLOR,
    consulting_gw_series_style,
)

_GW_MARKER_MAP = {
    "circle": "o",
    "triangle": "v",
    "diamond": "D",
    "plus": "P",
    "x": "x",
}


class WaterSeriesLegendEntry(TypedDict):
    series_id: str
    color: str
    marker: str
    level_label: str
    elevation_label: str


def _group_water_levels(
    water_levels: Sequence[WaterLevel],
    profile_lookup: dict[str, tuple[float, float]],
) -> dict[str, list[WaterLevel]]:
    groups: dict[str, list[WaterLevel]] = {}
    for level in water_levels:
        if level.hole_id not in profile_lookup:
            continue
        series_id = level.series_id or "default"
        groups.setdefault(series_id, []).append(level)
    return groups


def _connect_subgroups(
    levels: Sequence[WaterLevel],
) -> dict[str, list[WaterLevel]]:
    """Split a series into connect_group nests (blank = one shared polyline)."""
    subgroups: dict[str, list[WaterLevel]] = {}
    for level in levels:
        group_id = (level.connect_group or "").strip()
        subgroups.setdefault(group_id, []).append(level)
    return subgroups


class RendererWaterMixin:
    """Water-table markers, polylines, and compact GW legend."""

    def _water_elevation_label(self, level: WaterLevel, collar_rl: float) -> str:
        """Annotate water as RL (masl) in elevation mode, or depth (mbgs) in relative mode."""
        if self.profile.y_axis_mode == "depth_below_collar":
            return f"{level.depth:.2f} mbgs"
        water_rl = (
            float(level.elevation_masl)
            if level.elevation_masl is not None
            else collar_rl - level.depth
        )
        if self.profile.layout == "consulting_section":
            return f"{water_rl:.3f}"
        return f"{water_rl:.2f} m"

    def _water_legend_captions(
        self,
        series_id: str,
        label: str,
        default_label: str,
    ) -> tuple[str, str]:
        relative = self.profile.y_axis_mode == "depth_below_collar"
        datum = "mbgs" if relative else "masl"
        if series_id == "default" and not label:
            if relative:
                return "GROUNDWATER LEVEL (mbgs)", "GROUNDWATER DEPTH (mbgs)"
            return "GROUNDWATER LEVEL (masl)", "GROUNDWATER ELEVATION (masl)"
        display_label = (label or default_label or series_id).upper()
        if relative:
            return (
                f"GROUNDWATER LEVEL ({display_label})",
                f"GROUNDWATER DEPTH {datum} ({display_label})",
            )
        return (
            f"GROUNDWATER LEVEL ({display_label})",
            f"GROUNDWATER ELEVATION masl ({display_label})",
        )

    def _draw_water_table(
        self,
        ax,
        hole_summary: pd.DataFrame,
        water_levels: Sequence[WaterLevel],
        collar_lookup: dict[str, float],
        *,
        label_elevations: bool = False,
        label_dry_wells: bool = False,
        label_series_gaps: bool = False,
        water_color: str | None = None,
        profile_lookup: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if hole_summary.empty:
            return
        if not water_levels and not label_dry_wells and not label_series_gaps:
            self.water_series_legend = []
            return
        if profile_lookup is None:
            profile_lookup = self._profile_lookup(hole_summary, collar_lookup)
        series_groups = _group_water_levels(water_levels, profile_lookup)
        holes_with_any_water = {
            level.hole_id for levels in series_groups.values() for level in levels
        }
        fully_dry_nm_drawn: set[str] = set()
        if label_dry_wells:
            dry_lookup = {
                hole_id: profile
                for hole_id, profile in profile_lookup.items()
                if hole_id not in holes_with_any_water
            }
            if dry_lookup:
                dry_x = np.fromiter((p[0] for p in dry_lookup.values()), dtype=float, count=len(dry_lookup))
                dry_collars = np.fromiter((p[1] for p in dry_lookup.values()), dtype=float, count=len(dry_lookup))
                dry_y = self._plot_y_values(dry_collars - 1.0, dry_collars)
                for hole_id, x_profile, y in zip(dry_lookup.keys(), dry_x, dry_y, strict=True):
                    fully_dry_nm_drawn.add(str(hole_id))
                    ax.annotate(
                        "NM",
                        xy=(float(x_profile), float(y)),
                        xytext=(4, 0),
                        textcoords="offset points",
                        fontsize=8,
                        color=CONSULTING_NM_COLOR,
                        zorder=8,
                    )
        if not series_groups:
            return
        self.water_series_legend = []
        interpolate = self.interpolate_water_table or self.profile.interpolate_water_table_default
        use_segments = self.profile.water_interpolate_segments
        across_gaps = self.profile.water_interpolate_across_gaps
        default_water_color = (
            CONSULTING_WATER_COLOR
            if self.profile.layout == "consulting_section"
            else WATER_COLOR
        )
        profile_marker = _GW_MARKER_MAP.get(self.profile.water_symbol, self.profile.water_symbol)
        transect_hole_ids = hole_summary.sort_values("x_profile")["hole_id"].astype(str).tolist()
        transect_x = {
            str(row.hole_id): float(row.x_profile)
            for row in hole_summary.itertuples(index=False)
        }
        for series_index, (series_id, levels) in enumerate(sorted(series_groups.items())):
            first = levels[0]
            default_color, default_marker, default_label = consulting_gw_series_style(
                series_id,
                first.series_label,
                series_index=series_index,
            )
            color = first.color or water_color or default_color or default_water_color
            marker_key = (first.marker or default_marker or profile_marker).lower()
            marker = _GW_MARKER_MAP.get(marker_key, marker_key)
            label = first.series_label or default_label or series_id
            level_by_hole = {level.hole_id: level for level in levels}
            if label_series_gaps:
                # Fully dry holes: one NM only (skip if label_dry_wells already drew them,
                # or draw once across series when dry-well labeling is off).
                for hole_id in transect_hole_ids:
                    if hole_id in level_by_hole:
                        continue
                    if hole_id not in holes_with_any_water:
                        if label_dry_wells or hole_id in fully_dry_nm_drawn:
                            continue
                        fully_dry_nm_drawn.add(hole_id)
                    profile = profile_lookup.get(hole_id)
                    if profile is None:
                        continue
                    x_profile, collar_rl = profile
                    y = self._plot_y(collar_rl - 1.0, collar_rl)
                    ax.annotate(
                        "NM",
                        xy=(float(x_profile), float(y)),
                        xytext=(4, 0),
                        textcoords="offset points",
                        fontsize=8,
                        color=CONSULTING_NM_COLOR,
                        zorder=8,
                    )
            # Draw each connect_group nest separately so shallow/deep do not join.
            for _group_id, group_levels in _connect_subgroups(levels).items():
                level_by_id = {item.hole_id: item for item in group_levels}
                xs: list[float] = []
                water_rls: list[float] = []
                collars: list[float] = []
                measured_levels: list[WaterLevel] = []
                for hole_id in transect_hole_ids:
                    level = level_by_id.get(hole_id)
                    if level is None:
                        continue
                    profile = profile_lookup.get(hole_id)
                    if profile is None:
                        continue
                    x_profile, collar_rl = profile
                    xs.append(x_profile)
                    water_rls.append(collar_rl - level.depth)
                    collars.append(collar_rl)
                    measured_levels.append(level)
                if not xs:
                    continue
                xs_arr = np.asarray(xs, dtype=float)
                water_arr = np.asarray(water_rls, dtype=float)
                collar_arr = np.asarray(collars, dtype=float)
                ys = self._plot_y_values(water_arr, collar_arr)
                ax.scatter(xs_arr, ys, marker=marker, c=color, s=49, zorder=7)
                if label_elevations:
                    for x_profile, water_rl, y, level, collar_rl in zip(
                        xs_arr, water_arr, ys, measured_levels, collars, strict=True
                    ):
                        ax.annotate(
                            self._water_elevation_label(level, collar_rl),
                            xy=(float(x_profile), float(y)),
                            xytext=(4, -8),
                            textcoords="offset points",
                            fontsize=7,
                            color=color,
                            zorder=8,
                        )
                if len(xs_arr) >= 2 and interpolate:
                    gw_linestyle = "-" if self.profile.water_line_solid else "--"
                    if across_gaps:
                        x_dense = np.linspace(float(xs_arr.min()), float(xs_arr.max()), 100)
                        y_dense = np.interp(x_dense, xs_arr, ys)
                        ax.plot(
                            x_dense,
                            y_dense,
                            color=color,
                            linewidth=2.0,
                            linestyle=gw_linestyle,
                            zorder=6,
                        )
                    elif use_segments:
                        y_by_hole = {
                            level.hole_id: float(y)
                            for level, y in zip(measured_levels, ys, strict=True)
                        }
                        segments: list[np.ndarray] = []
                        for left_id, right_id in zip(
                            transect_hole_ids, transect_hole_ids[1:], strict=False
                        ):
                            if left_id not in y_by_hole or right_id not in y_by_hole:
                                continue
                            segments.append(
                                np.asarray(
                                    [
                                        [transect_x[left_id], y_by_hole[left_id]],
                                        [transect_x[right_id], y_by_hole[right_id]],
                                    ],
                                    dtype=float,
                                )
                            )
                        if segments:
                            collection = LineCollection(
                                segments,
                                colors=color,
                                linewidths=2.0,
                                linestyles=gw_linestyle,
                                zorder=6,
                            )
                            ax.add_collection(collection)
                    else:
                        ax.plot(
                            xs_arr,
                            ys,
                            color=color,
                            linewidth=2.0,
                            linestyle=gw_linestyle,
                            zorder=6,
                        )
            level_label_text, elevation_label_text = self._water_legend_captions(
                series_id, label, default_label
            )
            self.water_series_legend.append(
                {
                    "series_id": series_id,
                    "color": color,
                    "marker": marker,
                    "level_label": level_label_text,
                    "elevation_label": elevation_label_text,
                }
            )

    def _draw_compact_water_legend(self, ax) -> None:
        if not self.water_series_legend:
            return
        lines = []
        for entry in self.water_series_legend:
            lines.append(entry["level_label"])
        ax.text(
            0.01,
            0.01,
            " | ".join(lines),
            transform=ax.transAxes,
            fontsize=7,
            color=LABEL_COLOR,
            va="bottom",
            ha="left",
            zorder=20,
        )
