"""Groundwater / water-table drawing mixin for CrossSectionRenderer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.markers import MarkerStyle
from matplotlib.text import Text
from matplotlib.transforms import offset_copy

from hydro_metrics import (
    format_gradient_label,
    horizontal_gradients_along_profile,
    water_head_masl,
    water_status,
)
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


# Candidate label offsets (points) as (dx, dy, ha), sorted nearest-first from
# the default spot so a label takes the closest free position in any direction.
def _nearest_first(
    base: tuple[float, float],
    offsets: Sequence[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    return sorted(offsets, key=lambda item: ((item[0] - base[0]) ** 2 + (item[1] - base[1]) ** 2))


_DY_GRID = (0.0, -8.0, 8.0, -16.0, 16.0, -26.0, 26.0, -38.0, 38.0, -52.0, 52.0)
_SIDE_OFFSETS = [
    (dx, dy, "left" if dx > 0 else "right")
    for dx in (5.0, -5.0, 22.0, -22.0, 40.0, -40.0)
    for dy in _DY_GRID
]
_LABEL_CANDIDATES: dict[str, list[tuple[float, float, str]]] = {
    "rl": _nearest_first((5.0, -8.0), _SIDE_OFFSETS),
    "nm": _nearest_first((5.0, 0.0), _SIDE_OFFSETS),
    "gradient": _nearest_first(
        (0.0, 6.0),
        [(dx, dy, "center") for dx in (0.0, 20.0, -20.0, 40.0, -40.0, 60.0, -60.0) for dy in _DY_GRID],
    ),
}
# Placement order: RL values matter most, gradients least.
_LABEL_PRIORITY = {"rl": 0, "nm": 1, "gradient": 2}
_LEADER_THRESHOLD_PT = 14.0


def _overlap_area(a, b) -> float:
    width = min(a.x1, b.x1) - max(a.x0, b.x0)
    height = min(a.y1, b.y1) - max(a.y0, b.y0)
    return width * height if width > 0 and height > 0 else 0.0


def _outside_area(box, frame) -> float:
    inside_w = max(0.0, min(box.x1, frame.x1) - max(box.x0, frame.x0))
    inside_h = max(0.0, min(box.y1, frame.y1) - max(box.y0, frame.y0))
    return box.width * box.height - inside_w * inside_h


def _text_box(annotation, renderer):
    # Text-only extent: Annotation.get_window_extent includes a visible leader,
    # which would make a moved label block the whole strip back to its point.
    annotation.update_positions(renderer)
    return Text.get_window_extent(annotation, renderer)


# Hole-ID header candidates as (dx_pt, tier, ha): first shift alignment away
# from the neighbour, then stagger outward one or two text heights; negative
# tiers step inward for headers already at the page edge (consulting top row).
_HEADER_CANDIDATES = (
    (0.0, 0, "center"),
    (2.0, 0, "left"),
    (-2.0, 0, "right"),
    (0.0, 1, "center"),
    (2.0, 1, "left"),
    (-2.0, 1, "right"),
    (0.0, -1, "center"),
    (2.0, -1, "left"),
    (-2.0, -1, "right"),
    (0.0, 2, "center"),
)


def resolve_header_collisions(fig, headers) -> None:
    """Keep hole-ID column headers from fusing when holes are close together.

    Headers sit outside the plot, so they are never leadered: each stays above
    (or below) its own column, only shifting alignment or stepping one tier out.
    """
    headers = [text for text in headers if text.get_visible() and text.get_text().strip()]
    if len(headers) < 2:
        return
    renderer = _figure_renderer(fig)
    pad = renderer.points_to_pixels(2.0)
    for text in headers:
        # Re-runs (export resizes the page) must start from the drawn position,
        # not stack another offset on the previous pass's choice.
        if not hasattr(text, "_header_base"):
            text._header_base = (
                text.get_transform(),
                text.get_horizontalalignment(),
                text.get_rotation(),
            )
        base_transform, base_ha, base_rotation = text._header_base
        text.set_transform(base_transform)
        text.set_horizontalalignment(base_ha)
        text.set_rotation(base_rotation)
    page = fig.bbox
    obstacles = _header_obstacles(fig, renderer, 0.0, exclude={id(text) for text in headers})
    _stagger_headers(fig, renderer, headers, obstacles, page, pad)
    staggered_cost = _header_layout_cost(renderer, headers, obstacles, page, pad)
    if staggered_cost == 0.0:
        return
    # Dense well fields: no horizontal stagger fits, so try vertical headers
    # (CAD convention) and keep whichever layout overlaps less.
    staggered = [(t.get_transform(), t.get_horizontalalignment()) for t in headers]
    for text in headers:
        text.set_transform(text._header_base[0])
        text.set_horizontalalignment("center")
        text.set_rotation(90)
    if _header_layout_cost(renderer, headers, obstacles, page, pad) >= staggered_cost:
        for text, (transform, ha) in zip(headers, staggered):
            text.set_rotation(text._header_base[2])
            text.set_transform(transform)
            text.set_horizontalalignment(ha)


def _header_layout_cost(renderer, headers, obstacles, page, pad) -> float:
    boxes = [text.get_window_extent(renderer).padded(pad) for text in headers]
    cost = 0.0
    for index, box in enumerate(boxes):
        cost += sum(_overlap_area(box, other) for other in boxes[index + 1 :])
        cost += sum(_overlap_area(box, other) for other in obstacles)
        cost += 4.0 * _outside_area(box, page)
    return cost


def _stagger_headers(fig, renderer, headers, obstacles, page, pad) -> None:
    """Greedy pass: nudge alignment, then step headers outward by text-height tiers."""
    placed = list(obstacles)
    for text in sorted(headers, key=lambda item: item.get_window_extent(renderer).x0):
        base_transform = text._header_base[0]
        height_pt = text.get_window_extent(renderer).height * 72.0 / fig.dpi + 1.0
        outward = 1.0 if text.get_verticalalignment() == "bottom" else -1.0
        best: tuple[float, float, int, str] | None = None
        for index, (dx, tier, ha) in enumerate(_HEADER_CANDIDATES):
            text.set_transform(
                offset_copy(base_transform, fig=fig, x=dx, y=outward * tier * height_pt, units="points")
            )
            text.set_horizontalalignment(ha)
            box = text.get_window_extent(renderer).padded(pad)
            collision = sum(_overlap_area(box, other) for other in placed)
            outside = _outside_area(box, page)
            score = (collision + 4.0 * outside) * 1000.0 + index
            if best is None or score < best[0]:
                best = (score, dx, tier, ha)
            if collision == 0.0 and outside == 0.0:
                break
        assert best is not None
        _score, dx, tier, ha = best
        text.set_transform(
            offset_copy(base_transform, fig=fig, x=dx, y=outward * tier * height_pt, units="points")
        )
        text.set_horizontalalignment(ha)
        placed.append(text.get_window_extent(renderer).padded(pad))


def _drawn_tick_labels(axis, limits) -> list:
    """Major tick labels inside the view limits (matplotlib keeps but never
    draws labels for ticks past the axis ends)."""
    low, high = sorted(limits)
    span = (high - low) * 1e-9
    locs = axis.get_majorticklocs()
    labels = []
    for tick, loc in zip(axis.get_major_ticks(len(locs)), locs):
        if low - span <= loc <= high + span:
            labels.extend((tick.label1, tick.label2))
    return labels


def _header_obstacles(fig, renderer, pad, *, exclude: set[int]) -> list:
    """Tick labels, axis labels, titles and figure text a header must not cover."""
    fig.draw_without_rendering()  # settle tick label positions for this page size
    artists: list = list(fig.texts)
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        artists.extend(_drawn_tick_labels(ax.xaxis, ax.get_xlim()))
        artists.extend(_drawn_tick_labels(ax.yaxis, ax.get_ylim()))
        artists.extend((ax.xaxis.label, ax.yaxis.label, ax.title, ax._left_title, ax._right_title))
    boxes = []
    for artist in artists:
        if id(artist) in exclude or not artist.get_visible() or not artist.get_text().strip():
            continue
        boxes.append(artist.get_window_extent(renderer).padded(pad))
    return boxes

def _figure_renderer(fig):
    try:
        return fig.canvas.get_renderer()
    except AttributeError:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        return FigureCanvasAgg(fig).get_renderer()


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

    def _water_annotate(
        self,
        ax,
        text: str,
        xy: tuple[float, float],
        *,
        kind: str,
        color: str,
        fontsize: float,
        xytext: tuple[float, float],
        ha: str = "left",
    ):
        """Draw a water number and register it for collision avoidance."""
        annotation = ax.annotate(
            text,
            xy=xy,
            xytext=xytext,
            textcoords="offset points",
            fontsize=fontsize,
            color=color,
            ha=ha,
            va="center",
            zorder=9,
            bbox={"boxstyle": "square,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.85},
            # Leader exists from the start but stays hidden unless the label
            # is moved away from its point by the collision pass.
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.5, "shrinkA": 0, "shrinkB": 2},
        )
        annotation.arrow_patch.set_visible(False)
        if not hasattr(self, "_water_labels"):
            self._water_labels = []
        self._water_labels.append((kind, annotation, color))
        return annotation

    def _resolve_water_label_collisions(self, fig) -> None:
        """Move water numbers apart once the figure layout is final.

        Greedy placement: each label takes the nearest candidate offset that
        stays inside its axes and clears every label placed so far plus other
        axes text (chemistry values, annotations). Labels moved away from their
        point get a thin leader line, as on the client CAD figures.
        """
        labels = getattr(self, "_water_labels", None) or []
        if not labels:
            return
        renderer = _figure_renderer(fig)
        pad = renderer.points_to_pixels(1.5)
        water_artists = {id(annotation) for _kind, annotation, _color in labels}
        placed: list = []
        for ax in {annotation.axes for _kind, annotation, _color in labels}:
            for text in ax.texts:
                if id(text) not in water_artists and text.get_visible() and text.get_text().strip():
                    placed.append(text.get_window_extent(renderer).padded(pad))
        ordered = sorted(labels, key=lambda item: _LABEL_PRIORITY.get(item[0], 9))
        for kind, annotation, _color in ordered:
            frame = annotation.axes.get_window_extent(renderer)
            if not hasattr(annotation, "_water_base_xyann"):
                annotation._water_base_xyann = tuple(annotation.xyann)
            base = annotation._water_base_xyann
            best: tuple[float, tuple[float, float, str]] | None = None
            for index, (dx, dy, ha) in enumerate(_LABEL_CANDIDATES.get(kind, [(*base, "left")])):
                annotation.xyann = (dx, dy)
                annotation.set_horizontalalignment(ha)
                box = _text_box(annotation, renderer).padded(pad)
                collision = sum(_overlap_area(box, other) for other in placed)
                outside = _outside_area(box, frame)
                score = (collision + 4.0 * outside) * 1000.0 + index
                if best is None or score < best[0]:
                    best = (score, (dx, dy, ha))
                if collision == 0.0 and outside == 0.0:
                    break
            assert best is not None
            dx, dy, ha = best[1]
            annotation.xyann = (dx, dy)
            annotation.set_horizontalalignment(ha)
            placed.append(_text_box(annotation, renderer).padded(pad))
            moved = abs(dy - base[1]) > _LEADER_THRESHOLD_PT or abs(dx - base[0]) > _LEADER_THRESHOLD_PT
            annotation.arrow_patch.set_visible(moved)

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
        if not water_levels:
            # No water data at all: 'NM' would just label every hole (documented:
            # NM only when dry-well labeling is on AND water data exist).
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
                    self._water_annotate(
                        ax,
                        "NM",
                        (float(x_profile), float(y)),
                        kind="nm",
                        color=CONSULTING_NM_COLOR,
                        fontsize=8,
                        xytext=(4, 0),
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
            if self.profile.layout == "consulting_section":
                raw_marker = first.marker or default_marker or profile_marker
            else:
                raw_marker = first.marker or profile_marker or default_marker
            lowered = str(raw_marker).lower()
            marker = _GW_MARKER_MAP.get(lowered, raw_marker)
            if marker not in MarkerStyle.markers:
                marker = lowered if lowered in MarkerStyle.markers else "v"
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
                    self._water_annotate(
                        ax,
                        "NM",
                        (float(x_profile), float(y)),
                        kind="nm",
                        color=CONSULTING_NM_COLOR,
                        fontsize=8,
                        xytext=(4, 0),
                    )
            # Draw each connect_group nest separately so shallow/deep do not join.
            for group_id, group_levels in _connect_subgroups(levels).items():
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
                    status = water_status(level)
                    if status in {"dry", "nm"}:
                        y_nm = self._plot_y(collar_rl - 1.0, collar_rl)
                        self._water_annotate(
                            ax,
                            "NM" if status == "nm" else "DRY",
                            (float(x_profile), float(y_nm)),
                            kind="nm",
                            color=CONSULTING_NM_COLOR,
                            fontsize=8,
                            xytext=(4, 0),
                        )
                        continue
                    xs.append(x_profile)
                    water_rls.append(water_head_masl(level, collar_rl))
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
                        self._water_annotate(
                            ax,
                            self._water_elevation_label(level, collar_rl),
                            (float(x_profile), float(y)),
                            kind="rl",
                            color=color,
                            fontsize=7,
                            xytext=(4, -8),
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
                            if self._cad_svg_layers_enabled():
                                self._set_cad_gid(collection, "water")
                    else:
                        plotted = ax.plot(
                            xs_arr,
                            ys,
                            color=color,
                            linewidth=2.0,
                            linestyle=gw_linestyle,
                            zorder=6,
                        )
                        if self._cad_svg_layers_enabled() and plotted:
                            self._set_cad_gid(plotted[0], "water")
                    # Schematic horizontal i = Δh/Δx between adjacent measured heads.
                    gradient_segments = horizontal_gradients_along_profile(
                        measured_levels,
                        hole_order=transect_hole_ids,
                        x_by_hole=transect_x,
                        collar_rl_by_hole={
                            hid: float(profile_lookup[hid][1])
                            for hid in transect_hole_ids
                            if hid in profile_lookup
                        },
                        series_id=series_id,
                        connect_group=group_id,
                    )
                    adjacent_pairs = set(
                        zip(transect_hole_ids, transect_hole_ids[1:], strict=False)
                    )
                    for segment in gradient_segments:
                        if (
                            use_segments
                            and not across_gaps
                            and (segment.left_hole_id, segment.right_hole_id)
                            not in adjacent_pairs
                        ):
                            # Segments mode draws no water line across a dry/NM
                            # gap — do not float an i= label over the open gap.
                            continue
                        mid_collar = 0.5 * (
                            float(profile_lookup[segment.left_hole_id][1])
                            + float(profile_lookup[segment.right_hole_id][1])
                        )
                        mid_y = self._plot_y(segment.mid_head_masl, mid_collar)
                        self._water_annotate(
                            ax,
                            format_gradient_label(segment),
                            (segment.mid_x, float(mid_y)),
                            kind="gradient",
                            color=color,
                            fontsize=6,
                            xytext=(0, 6),
                            ha="center",
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
