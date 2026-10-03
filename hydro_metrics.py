"""Hydrogeology metrics for fence diagrams (leaf module — no UI widgets).

Horizontal hydraulic gradient between adjacent measured water levels along the
profile is a schematic annotation (Δh/Δx), not a calibrated flow model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from models import WaterLevel

# Warn when |Δh/Δx| exceeds this (dimensionless). Typical sand/gravel aquifers
# are often 1e-4–1e-2; values above ~0.1 are usually data or unit errors.
DEFAULT_ABSURD_HORIZONTAL_GRADIENT = 0.1


@dataclass(frozen=True)
class HorizontalGradientSegment:
    """Adjacent-hole schematic horizontal hydraulic gradient along the profile."""

    left_hole_id: str
    right_hole_id: str
    x_left: float
    x_right: float
    head_left_masl: float
    head_right_masl: float
    gradient: float  # (h_right - h_left) / (x_right - x_left)
    mid_x: float
    mid_head_masl: float

    @property
    def abs_gradient(self) -> float:
        return abs(self.gradient)

    @property
    def delta_x_m(self) -> float:
        return self.x_right - self.x_left

    @property
    def delta_h_m(self) -> float:
        return self.head_right_masl - self.head_left_masl


def water_status(level: WaterLevel) -> str:
    """Normalize WaterLevel.status to measured | dry | nm."""
    raw = str(getattr(level, "status", "measured") or "measured").strip().lower()
    if raw in {"dry", "d"}:
        return "dry"
    if raw in {"nm", "n/m", "not measured", "not_measured", "ns", "not sampled", "not_sampled"}:
        return "nm"
    return "measured"


def is_measured_water_level(level: WaterLevel) -> bool:
    return water_status(level) == "measured"


def water_head_masl(level: WaterLevel, collar_rl: float) -> float:
    """Head elevation (masl): prefer stored elevation_masl, else collar − depth."""
    if level.elevation_masl is not None:
        return float(level.elevation_masl)
    return float(collar_rl) - float(level.depth)


def horizontal_gradients_along_profile(
    water_levels: Sequence[WaterLevel],
    *,
    hole_order: Sequence[str],
    x_by_hole: Mapping[str, float],
    collar_rl_by_hole: Mapping[str, float],
    series_id: str | None = None,
    connect_group: str | None = None,
) -> tuple[HorizontalGradientSegment, ...]:
    """Compute Δh/Δx between consecutive measured holes along the transect.

    Unmeasured (dry/NM) holes are skipped, so a pair may span intermediate holes.

    Only levels with status ``measured`` participate. Optional ``series_id`` /
    ``connect_group`` filters match renderer nesting (blank connect_group = shared).
    """
    filtered: list[WaterLevel] = []
    for level in water_levels:
        if not is_measured_water_level(level):
            continue
        if series_id is not None and (level.series_id or "default") != series_id:
            continue
        if connect_group is not None:
            group = (level.connect_group or "").strip()
            if group != (connect_group or "").strip():
                continue
        if level.hole_id not in x_by_hole or level.hole_id not in collar_rl_by_hole:
            continue
        filtered.append(level)

    by_hole: dict[str, WaterLevel] = {}
    for level in filtered:
        by_hole[level.hole_id] = level  # last wins (matches plot behaviour)

    ordered_ids = [hid for hid in hole_order if hid in by_hole]
    segments: list[HorizontalGradientSegment] = []
    for left_id, right_id in zip(ordered_ids, ordered_ids[1:], strict=False):
        x_left = float(x_by_hole[left_id])
        x_right = float(x_by_hole[right_id])
        dx = x_right - x_left
        if abs(dx) < 1e-9:
            continue
        left = by_hole[left_id]
        right = by_hole[right_id]
        h_left = water_head_masl(left, float(collar_rl_by_hole[left_id]))
        h_right = water_head_masl(right, float(collar_rl_by_hole[right_id]))
        gradient = (h_right - h_left) / dx
        segments.append(
            HorizontalGradientSegment(
                left_hole_id=left_id,
                right_hole_id=right_id,
                x_left=x_left,
                x_right=x_right,
                head_left_masl=h_left,
                head_right_masl=h_right,
                gradient=gradient,
                mid_x=0.5 * (x_left + x_right),
                mid_head_masl=0.5 * (h_left + h_right),
            )
        )
    return tuple(segments)


def absurd_horizontal_gradient_warnings(
    segments: Sequence[HorizontalGradientSegment],
    *,
    threshold: float = DEFAULT_ABSURD_HORIZONTAL_GRADIENT,
    series_label: str = "",
) -> list[str]:
    """Validate-style warnings when |i| is unrealistically large for a fence annotation."""
    warnings: list[str] = []
    label = f" ({series_label})" if series_label else ""
    for segment in segments:
        if segment.abs_gradient <= threshold:
            continue
        warnings.append(
            f"Horizontal hydraulic gradient{label} between {segment.left_hole_id} and "
            f"{segment.right_hole_id}: |i|={segment.abs_gradient:.3f} "
            f"(Δh={segment.delta_h_m:.2f} m over Δx={segment.delta_x_m:.1f} m) "
            f"exceeds {threshold:g} — check units, elevations, or hole order"
        )
    return warnings


def format_gradient_label(segment: HorizontalGradientSegment) -> str:
    """Short profile annotation for schematic i = Δh/Δx."""
    return f"i={segment.gradient:.3g}"
