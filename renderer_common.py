"""Shared geometry and style helpers for CrossSectionRenderer."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection

from constants import get_lithology_style
from models import ScreenInterval
from render_theme import TRACK_BORDER_COLOR


class RendererGeometryMixin:
    """Well extents, profile lookup, and lithology style resolution."""

    def _collar_values(self, hole_ids: pd.Series, fallback: pd.Series, collar_lookup: dict[str, float]) -> np.ndarray:
        mapped = hole_ids.astype(str).map(collar_lookup)
        return mapped.fillna(fallback).to_numpy(dtype=float)

    def _profile_lookup(
        self,
        hole_summary: pd.DataFrame,
        collar_lookup: dict[str, float] | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Map hole_id → (x_profile, collar_rl)."""
        if hole_summary.empty:
            return {}
        hole_ids = hole_summary["hole_id"].astype(str).to_numpy()
        x_values = hole_summary["x_profile"].to_numpy(dtype=float)
        if collar_lookup:
            collars = self._collar_values(
                hole_summary["hole_id"],
                hole_summary["collar_elevation"],
                collar_lookup,
            )
        else:
            collars = hole_summary["collar_elevation"].to_numpy(dtype=float)
        return {
            str(hole_id): (float(x_profile), float(collar_rl))
            for hole_id, x_profile, collar_rl in zip(hole_ids, x_values, collars, strict=True)
        }

    def _well_extents(
        self,
        hole_summary: pd.DataFrame,
        collar_depths: dict[str, float],
        collar_lookup: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return (x_profile, top_y, bottom_y) for well sticks/columns, or None if empty."""
        if hole_summary.empty:
            return None
        hole_ids = hole_summary["hole_id"].astype(str).to_numpy()
        x_values = hole_summary["x_profile"].to_numpy(dtype=float)
        collars = self._collar_values(
            hole_summary["hole_id"],
            hole_summary["collar_elevation"],
            collar_lookup,
        )
        row_bottoms = hole_summary["bottom_elevation"].to_numpy(dtype=float)
        td_values = np.fromiter(
            (collar_depths.get(str(hole_id), np.nan) for hole_id in hole_ids),
            dtype=float,
            count=len(hole_ids),
        )
        bottom_elev = np.where(np.isfinite(td_values), collars - td_values, row_bottoms)
        top_y = self._plot_y_values(collars, collars)
        bottom_y = self._plot_y_values(bottom_elev, collars)
        return x_values, top_y, bottom_y

    def _well_rect_geometry(
        self,
        hole_summary: pd.DataFrame,
        collar_depths: dict[str, float],
        collar_lookup: dict[str, float],
        track_half: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Return (x_left, y0, width, height) for filled/bordered well rectangles."""
        extents = self._well_extents(hole_summary, collar_depths, collar_lookup)
        if extents is None:
            return None
        x_values, top_y, bottom_y = extents
        heights = np.abs(top_y - bottom_y)
        mask = heights > 1e-9
        if not np.any(mask):
            return None
        x_values = x_values[mask]
        y0 = np.minimum(top_y[mask], bottom_y[mask])
        heights = heights[mask]
        return x_values - track_half, y0, np.full(len(x_values), track_half * 2.0), heights

    @staticmethod
    def _rect_verts(
        x_left: np.ndarray,
        y0: np.ndarray,
        widths: np.ndarray,
        heights: np.ndarray,
    ) -> np.ndarray:
        x_right = x_left + widths
        y_top = y0 + heights
        return np.stack(
            [
                np.column_stack([x_left, y0]),
                np.column_stack([x_right, y0]),
                np.column_stack([x_right, y_top]),
                np.column_stack([x_left, y_top]),
            ],
            axis=1,
        )

    def _add_rect_collection(
        self,
        ax,
        geometry: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        *,
        facecolors,
        edgecolors,
        linewidths: float,
        zorder: int,
        hatch: str | None = None,
        alpha: float | None = None,
    ) -> None:
        collection = PolyCollection(
            self._rect_verts(*geometry),
            facecolors=facecolors,
            edgecolors=edgecolors,
            linewidths=linewidths,
            hatch=hatch,
            alpha=alpha,
        )
        collection.set_zorder(zorder)
        ax.add_collection(collection)

    def _draw_lithology_interval_rects(
        self,
        ax,
        projected_df: pd.DataFrame,
        style_cache: dict,
        track_half_width: float,
        collar_lookup: dict[str, float],
        *,
        zorder: int = 5,
        alpha: float = 0.96,
    ) -> None:
        """Vectorized track lithology rectangles grouped by resolved style."""
        if projected_df.empty:
            return
        collars = self._collar_values(
            projected_df["hole_id"],
            projected_df["collar_elevation"],
            collar_lookup,
        )
        x_values = projected_df["x_profile"].to_numpy(dtype=float)
        tops = self._plot_y_values(
            projected_df["top_elevation"].to_numpy(dtype=float),
            collars,
        )
        bottoms = self._plot_y_values(
            projected_df["bottom_elevation"].to_numpy(dtype=float),
            collars,
        )
        y0 = np.minimum(tops, bottoms)
        heights = np.abs(tops - bottoms)
        mask = heights > 1e-9
        if not np.any(mask):
            return
        lithology_codes = projected_df["lithology_code"].astype(str).to_numpy()[mask]
        x_masked = x_values[mask]
        y0_masked = y0[mask]
        heights_masked = heights[mask]
        width = 2.0 * track_half_width
        for code in np.unique(lithology_codes):
            code_mask = lithology_codes == code
            style = self._resolve_style(str(code), style_cache)
            self._add_rect_collection(
                ax,
                (
                    x_masked[code_mask] - track_half_width,
                    y0_masked[code_mask],
                    np.full(int(code_mask.sum()), width),
                    heights_masked[code_mask],
                ),
                facecolors=style.color,
                edgecolors=style.edge_color,
                linewidths=0.6,
                zorder=zorder,
                hatch=style.hatch or None,
                alpha=alpha,
            )

    def _style_cache_for(self, lithology_codes: Sequence[str]) -> dict:
        consulting_palette = bool(getattr(self.profile, "use_consulting_palette", False))
        use_hatch = self.show_hatches
        return {
            code: get_lithology_style(
                code,
                use_hatch=use_hatch,
                consulting_palette=consulting_palette,
            )
            for code in lithology_codes
        }

    def _resolve_style(self, code: str, style_cache: dict):
        style = style_cache.get(code)
        if style is None:
            consulting = getattr(self.profile, "use_consulting_palette", False)
            style = get_lithology_style(
                code,
                use_hatch=self.show_hatches,
                consulting_palette=consulting,
            )
            style_cache[code] = style
        return style

    def _draw_screen_intervals(
        self,
        ax,
        hole_summary: pd.DataFrame,
        screen_intervals: Sequence[ScreenInterval],
        collar_lookup: dict[str, float],
        track_half: float,
        *,
        profile_lookup: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if not screen_intervals or hole_summary.empty:
            return
        if profile_lookup is None:
            profile_lookup = self._profile_lookup(hole_summary, collar_lookup)
        half = track_half * 0.92
        width = track_half * 1.84
        x_profiles: list[float] = []
        collar_rls: list[float] = []
        from_depths: list[float] = []
        to_depths: list[float] = []
        for interval in screen_intervals:
            profile = profile_lookup.get(interval.hole_id)
            if profile is None:
                continue
            x_profile, collar_rl = profile
            x_profiles.append(x_profile)
            collar_rls.append(collar_rl)
            from_depths.append(interval.from_depth)
            to_depths.append(interval.to_depth)
        if not x_profiles:
            return
        x_arr = np.asarray(x_profiles, dtype=float)
        collar_arr = np.asarray(collar_rls, dtype=float)
        from_arr = np.asarray(from_depths, dtype=float)
        to_arr = np.asarray(to_depths, dtype=float)
        top_ys = self._plot_y_values(collar_arr - from_arr, collar_arr)
        bottom_ys = self._plot_y_values(collar_arr - to_arr, collar_arr)
        heights = np.abs(top_ys - bottom_ys)
        mask = heights > 1e-9
        if not np.any(mask):
            return
        x_arr = x_arr[mask] - half
        y_arr = np.minimum(top_ys[mask], bottom_ys[mask])
        h_arr = heights[mask]
        self._add_rect_collection(
            ax,
            (x_arr, y_arr, np.full(int(mask.sum()), width), h_arr),
            facecolors="none",
            edgecolors=TRACK_BORDER_COLOR,
            linewidths=0.6,
            zorder=9,
            hatch="///",
        )

