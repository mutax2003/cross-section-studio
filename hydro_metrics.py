"""Hydrogeology metrics for fence diagrams (leaf module — no UI widgets).

Horizontal hydraulic gradient between adjacent measured water levels along the
profile is a schematic annotation (Δh/Δx), not a calibrated flow model.

``water_head_masl`` is the single source of truth for a groundwater head: it
honours the reading's depth datum (below ground or below top of casing, using
the collar stick-up) and converts the along-hole depth to an elevation with the
hole's survey (``md_to_elevation``) so inclined holes plot at true vertical
depth. Renderer markers, labels, gradients and Validate QA all call it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from models import (
    DEFAULT_LNAPL_SPECIFIC_GRAVITY,
    Collar,
    DeviationReading,
    ScreenInterval,
    VerticalGradient,
    WaterLevel,
)

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


MdToElevation = Callable[[float], float]


def water_depth_along_hole_bgs(level: WaterLevel, stick_up_m: float | None = None) -> float:
    """Along-hole depth below ground (m) of a reading.

    ``btoc`` readings subtract the casing stick-up (missing stick-up counts as
    0 — Validate warns). Negative = water inside the stick-up, above ground.
    The stick-up is taken as measured along the hole axis (vertical for the
    usual near-vertical well casing).
    """
    depth = float(level.depth)
    if getattr(level, "depth_reference", "bgs") == "btoc":
        return depth - float(stick_up_m or 0.0)
    return depth


def water_head_masl(
    level: WaterLevel,
    collar_rl: float,
    *,
    stick_up_m: float | None = None,
    md_to_elevation: MdToElevation | None = None,
    lnapl_corrected: bool = True,
) -> float:
    """Head elevation (masl) of a water reading.

    * explicit ``elevation_masl`` is taken as the measured water elevation;
    * otherwise the along-hole depth below ground (``btoc`` minus stick-up) is
      converted with ``md_to_elevation`` (hole survey: collar − TVD) — vertical
      holes (``None``) use collar − depth, the legacy formula;
    * water above ground inside the casing (btoc depth < stick-up) sits
      ``stick_up − depth`` above the collar;
    * LNAPL: a reading with ``product_thickness_m > 0`` measured the oil/water
      interface, so the head is corrected upward by the product's vertical
      thickness × specific gravity (``lnapl_head_correction_m``) — also for an
      explicit ``elevation_masl``, which is the measured interface elevation,
      not an already-corrected head. ``lnapl_corrected=False`` returns the
      measured interface elevation (figure product band).
    """
    if level.elevation_masl is not None:
        head = float(level.elevation_masl)
    elif getattr(level, "depth_reference", "bgs") != "btoc" and md_to_elevation is None:
        head = float(collar_rl) - float(level.depth)
    else:
        md_below_ground = water_depth_along_hole_bgs(level, stick_up_m)
        if md_below_ground < 0.0 or md_to_elevation is None:
            head = float(collar_rl) - md_below_ground
        else:
            head = float(md_to_elevation(md_below_ground))
    if lnapl_corrected and has_lnapl(level):
        head += lnapl_head_correction_m(
            level,
            collar_rl,
            stick_up_m=stick_up_m,
            md_to_elevation=md_to_elevation,
            interface_masl=head,
        )
    return head


def water_depth_below_ground_m(
    level: WaterLevel,
    collar_rl: float,
    *,
    stick_up_m: float | None = None,
    md_to_elevation: MdToElevation | None = None,
) -> float:
    """Vertical depth of the head below ground surface (m) — the "mbgs" label value.

    Converts ``btoc`` readings to below ground and inclined-hole depths to true
    vertical depth; a plain below-ground reading on a vertical hole returns
    ``level.depth`` unchanged. LNAPL wells report the corrected head's depth
    (shallower than the measured oil/water interface).
    """
    if (
        level.elevation_masl is None
        and getattr(level, "depth_reference", "bgs") != "btoc"
        and md_to_elevation is None
        and not has_lnapl(level)
    ):
        return float(level.depth)
    head = water_head_masl(
        level, collar_rl, stick_up_m=stick_up_m, md_to_elevation=md_to_elevation
    )
    return float(collar_rl) - head


def horizontal_gradients_along_profile(
    water_levels: Sequence[WaterLevel],
    *,
    hole_order: Sequence[str],
    x_by_hole: Mapping[str, float],
    collar_rl_by_hole: Mapping[str, float],
    series_id: str | None = None,
    connect_group: str | None = None,
    stick_up_by_hole: Mapping[str, float] | None = None,
    md_to_elevation_by_hole: Mapping[str, MdToElevation] | None = None,
) -> tuple[HorizontalGradientSegment, ...]:
    """Compute Δh/Δx between consecutive measured holes along the transect.

    Unmeasured (dry/NM) holes are skipped, so a pair may span intermediate holes.
    Heads come from ``water_head_masl`` (stick-up for ``btoc`` readings and the
    hole survey for inclined holes when the optional maps are given).

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
        stick_ups = stick_up_by_hole or {}
        md_maps = md_to_elevation_by_hole or {}
        h_left = water_head_masl(
            left,
            float(collar_rl_by_hole[left_id]),
            stick_up_m=stick_ups.get(left_id),
            md_to_elevation=md_maps.get(left_id),
        )
        h_right = water_head_masl(
            right,
            float(collar_rl_by_hole[right_id]),
            stick_up_m=stick_ups.get(right_id),
            md_to_elevation=md_maps.get(right_id),
        )
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


# --- Nested / co-located wells: vertical hydraulic gradients ---------------------
#
# Sign convention: iv = (h_shallow − h_deep) / (z_shallow − z_deep), with z the
# screen mid-point elevation. The denominator is always positive (shallow
# screen above deep screen), so
#   iv > 0  → head falls with depth  → DOWNWARD flow (recharge),
#   iv < 0  → head rises with depth  → UPWARD flow (discharge).
# A user ``VerticalGradient`` arrow ``down`` / ``up`` has the same meaning.

# Collars within this plan distance (m) are treated as one well nest.
DEFAULT_NEST_DISTANCE_M = 5.0
# Head difference (m) at or below which a nest has no clear vertical direction:
# a drawn arrow is not "contradicted" by noise-level differences.
VERTICAL_GRADIENT_ARROW_TOLERANCE_M = 0.05

_SHALLOW_GROUP_TOKENS = frozenset({"s", "sh", "shallow", "upper", "u", "top"})
_MIDDLE_GROUP_TOKENS = frozenset({"i", "int", "intermediate", "m", "mid", "middle"})
_DEEP_GROUP_TOKENS = frozenset({"d", "dp", "deep", "lower", "l", "bottom", "base"})


@dataclass(frozen=True)
class NestedVerticalGradient:
    """Vertical hydraulic gradient between two screens of one well nest."""

    series_id: str
    shallow_hole_id: str
    deep_hole_id: str
    shallow_connect_group: str
    deep_connect_group: str
    head_shallow_masl: float
    head_deep_masl: float
    z_mid_shallow_masl: float  # screen mid-point elevation (measurement point)
    z_mid_deep_masl: float
    gradient: float  # iv; > 0 downward (recharge), < 0 upward (discharge)
    plan_distance_m: float

    @property
    def delta_h_m(self) -> float:
        """h_shallow − h_deep (m); positive = downward potential."""
        return self.head_shallow_masl - self.head_deep_masl

    @property
    def delta_z_m(self) -> float:
        return self.z_mid_shallow_masl - self.z_mid_deep_masl

    @property
    def direction(self) -> str:
        """``down`` / ``up`` flow, or ``none`` within the head tolerance."""
        if abs(self.delta_h_m) <= VERTICAL_GRADIENT_ARROW_TOLERANCE_M:
            return "none"
        return "down" if self.delta_h_m > 0 else "up"

    @property
    def shallow_label(self) -> str:
        return _well_point_label(self.shallow_hole_id, self.shallow_connect_group)

    @property
    def deep_label(self) -> str:
        return _well_point_label(self.deep_hole_id, self.deep_connect_group)

    @property
    def nest_label(self) -> str:
        return f"{self.shallow_label}/{self.deep_label}"

    def hole_ids(self) -> frozenset[str]:
        return frozenset({self.shallow_hole_id, self.deep_hole_id})


@dataclass(frozen=True)
class VerticalGradientAnalysis:
    """Nested-well gradients plus notes on readings that could not be paired."""

    gradients: tuple[NestedVerticalGradient, ...]
    notes: tuple[str, ...] = ()


def _well_point_label(hole_id: str, connect_group: str) -> str:
    return f"{hole_id} [{connect_group}]" if connect_group else hole_id


def _alpha_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    current = ""
    for char in text:
        if char.isalpha():
            current += char
            continue
        if current:
            tokens.append(current)
        current = ""
    if current:
        tokens.append(current)
    return tokens


def connect_group_depth_rank(name: str) -> float | None:
    """Depth rank (smaller = shallower) implied by a connect_group name, else ``None``.

    Recognises shallow / intermediate / deep words (``S``, ``shallow``,
    ``upper``, ``I``, ``mid``, ``D``, ``deep``, ``lower``…) and a trailing
    number (``1`` = shallowest). Anything else is ambiguous.
    """
    text = (name or "").strip().lower()
    if not text:
        return None
    for token in reversed(_alpha_tokens(text)):
        if token in _SHALLOW_GROUP_TOKENS:
            return 0.0
        if token in _MIDDLE_GROUP_TOKENS:
            return 1.0
        if token in _DEEP_GROUP_TOKENS:
            return 2.0
    digits = ""
    for char in reversed(text):
        if not char.isdigit():
            break
        digits = char + digits
    if digits:
        return 10.0 + float(int(digits))
    return None


def screen_elevation_range(
    screen: ScreenInterval,
    collar_rl: float,
    md_to_elevation: MdToElevation | None = None,
) -> tuple[float, float]:
    """``(top, bottom)`` elevation (masl) of a screen; surveyed holes use TVD."""
    if md_to_elevation is None:
        top = float(collar_rl) - float(screen.from_depth)
        bottom = float(collar_rl) - float(screen.to_depth)
    else:
        top = float(md_to_elevation(float(screen.from_depth)))
        bottom = float(md_to_elevation(float(screen.to_depth)))
    return (max(top, bottom), min(top, bottom))


def assign_screens_to_connect_groups(
    hole_id: str,
    screens: Sequence[ScreenInterval],
    connect_groups: Sequence[str],
) -> tuple[dict[str, ScreenInterval], str | None]:
    """Map each reading's connect_group to the screen it measures on one hole.

    * one screen → every group reads it;
    * several screens and one group → composite head, no mapping (no note);
    * several screens and as many groups whose names imply depth order
      (``S``/``D``, ``shallow``/``deep``, ``1``/``2``…) → shallowest name to
      the shallowest screen; otherwise no mapping and a note says why.
    """
    groups = sorted(set(connect_groups))
    ordered = sorted(screens, key=lambda item: (item.from_depth, item.to_depth))
    if not ordered or not groups:
        return {}, None
    if len(ordered) == 1:
        return {group: ordered[0] for group in groups}, None
    if len(groups) == 1:
        return {}, None
    ranks = {group: connect_group_depth_rank(group) for group in groups}
    known = [rank for rank in ranks.values() if rank is not None]
    if len(groups) != len(ordered) or len(known) != len(groups) or len(set(known)) != len(known):
        return {}, (
            f"{hole_id}: {len(ordered)} screens with readings in connect_group "
            + ", ".join(group or "(blank)" for group in groups)
            + " — cannot tell which screen each reading measures; use one group per "
            "screen named shallow/deep (S/D) or 1/2 for a vertical gradient."
        )
    by_rank = sorted(groups, key=lambda group: ranks[group] or 0.0)
    return dict(zip(by_rank, ordered, strict=True)), None


@dataclass(frozen=True)
class _WellPoint:
    hole_id: str
    connect_group: str
    head: float
    z_top: float
    z_bottom: float

    @property
    def z_mid(self) -> float:
        return 0.5 * (self.z_top + self.z_bottom)


def nested_vertical_gradients(
    water_levels: Sequence[WaterLevel],
    screen_intervals: Sequence[ScreenInterval],
    *,
    collar_rl_by_hole: Mapping[str, float],
    plan_xy_by_hole: Mapping[str, tuple[float, float]] | None = None,
    stick_up_by_hole: Mapping[str, float] | None = None,
    md_to_elevation_by_hole: Mapping[str, MdToElevation] | None = None,
    nest_distance_m: float = DEFAULT_NEST_DISTANCE_M,
) -> VerticalGradientAnalysis:
    """Vertical gradients between nested / co-located screened wells.

    Two measured readings of the same ``series_id`` (one survey round) pair up
    when they come from different holes whose collars are within
    ``nest_distance_m`` in plan, or from one hole with several screens whose
    readings are told apart by ``connect_group``. Both need a screen; the
    measurement elevation is the screen mid-point (survey-aware). Pairs whose
    screen elevations overlap are skipped (no vertical separation).

    ``iv = (h_shallow − h_deep) / (z_mid_shallow − z_mid_deep)`` — positive is
    downward flow (recharge), negative upward flow (discharge).
    """
    stick_ups = stick_up_by_hole or {}
    md_maps = md_to_elevation_by_hole or {}
    plan_xy = plan_xy_by_hole or {}
    screens_by_hole: dict[str, list[ScreenInterval]] = {}
    for screen in screen_intervals:
        screens_by_hole.setdefault(screen.hole_id, []).append(screen)

    # Last reading per series × hole × connect_group (matches the plot).
    readings: dict[tuple[str, str, str], WaterLevel] = {}
    for level in water_levels:
        if not is_measured_water_level(level):
            continue
        if level.hole_id not in collar_rl_by_hole or level.hole_id not in screens_by_hole:
            continue
        key = (level.series_id or "default", level.hole_id, (level.connect_group or "").strip())
        readings[key] = level

    groups_by_hole: dict[str, set[str]] = {}
    for _series, hole_id, group in readings:
        groups_by_hole.setdefault(hole_id, set()).add(group)
    notes: list[str] = []
    screen_map: dict[str, dict[str, ScreenInterval]] = {}
    for hole_id in sorted(groups_by_hole):
        mapping, note = assign_screens_to_connect_groups(
            hole_id, screens_by_hole[hole_id], sorted(groups_by_hole[hole_id])
        )
        screen_map[hole_id] = mapping
        if note:
            notes.append(note)

    points_by_series: dict[str, list[_WellPoint]] = {}
    for (series_id, hole_id, group), level in sorted(readings.items()):
        screen = screen_map.get(hole_id, {}).get(group)
        if screen is None:
            continue
        collar_rl = float(collar_rl_by_hole[hole_id])
        md_fn = md_maps.get(hole_id)
        head = water_head_masl(level, collar_rl, stick_up_m=stick_ups.get(hole_id), md_to_elevation=md_fn)
        z_top, z_bottom = screen_elevation_range(screen, collar_rl, md_fn)
        points_by_series.setdefault(series_id, []).append(
            _WellPoint(hole_id, group, head, z_top, z_bottom)
        )

    gradients: list[NestedVerticalGradient] = []
    for series_id in sorted(points_by_series):
        points = points_by_series[series_id]
        for index, first in enumerate(points):
            for second in points[index + 1 :]:
                if first.hole_id == second.hole_id:
                    distance = 0.0
                else:
                    xy_a = plan_xy.get(first.hole_id)
                    xy_b = plan_xy.get(second.hole_id)
                    if xy_a is None or xy_b is None:
                        continue
                    distance = ((xy_a[0] - xy_b[0]) ** 2 + (xy_a[1] - xy_b[1]) ** 2) ** 0.5
                    if distance > nest_distance_m + 1e-9:
                        continue
                # Overlapping (or touching) screens: no vertical separation.
                if min(first.z_top, second.z_top) >= max(first.z_bottom, second.z_bottom):
                    continue
                shallow, deep = (first, second) if first.z_mid > second.z_mid else (second, first)
                dz = shallow.z_mid - deep.z_mid
                if dz <= 1e-9:
                    continue
                gradients.append(
                    NestedVerticalGradient(
                        series_id=series_id,
                        shallow_hole_id=shallow.hole_id,
                        deep_hole_id=deep.hole_id,
                        shallow_connect_group=shallow.connect_group,
                        deep_connect_group=deep.connect_group,
                        head_shallow_masl=shallow.head,
                        head_deep_masl=deep.head,
                        z_mid_shallow_masl=shallow.z_mid,
                        z_mid_deep_masl=deep.z_mid,
                        gradient=(shallow.head - deep.head) / dz,
                        plan_distance_m=distance,
                    )
                )
    return VerticalGradientAnalysis(gradients=tuple(gradients), notes=tuple(notes))


def format_vertical_gradient_note(gradient: NestedVerticalGradient) -> str:
    """One-line QA / Validate description of a nested-well vertical gradient."""
    sense = {
        "down": "downward (recharge)",
        "up": "upward (discharge)",
    }.get(gradient.direction, "no clear vertical flow")
    series = "" if gradient.series_id == "default" else f" ({gradient.series_id})"
    return (
        f"Vertical gradient {gradient.nest_label}{series}: iv={gradient.gradient:+.3f} "
        f"{sense} — Δh={gradient.delta_h_m:+.2f} m over Δz={gradient.delta_z_m:.2f} m "
        "between screen mid-points"
    )


def vertical_gradient_arrow_conflicts(
    arrows: Sequence[VerticalGradient],
    gradients: Sequence[NestedVerticalGradient],
    *,
    tolerance_m: float = VERTICAL_GRADIENT_ARROW_TOLERANCE_M,
) -> list[str]:
    """Warnings where a user-entered gradient arrow contradicts the nest's heads.

    Only pairs with |Δh| > ``tolerance_m`` count; an arrow on a hole with no
    computed gradient is left alone.
    """
    warnings: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for arrow in arrows:
        direction = str(arrow.direction or "up").strip().lower()
        for gradient in gradients:
            if arrow.hole_id not in gradient.hole_ids():
                continue
            if abs(gradient.delta_h_m) <= tolerance_m:
                continue
            computed = "down" if gradient.delta_h_m > 0 else "up"
            key = (arrow.hole_id, direction, gradient.nest_label)
            if computed == direction or key in seen:
                continue
            seen.add(key)
            warnings.append(
                f"{arrow.hole_id}: vertical gradient arrow points {direction} but heads in "
                f"{gradient.nest_label} give {computed}ward flow "
                f"(h_shallow − h_deep = {gradient.delta_h_m:+.2f} m, iv={gradient.gradient:+.3f}) "
                "— check the arrow or the readings."
            )
    return warnings


# ---------------------------------------------------------------------------
# Plan-view groundwater flow from a planar head fit (h = a·E + b·N + c)
# ---------------------------------------------------------------------------
#
# Along-profile Δh/Δx is an *apparent* gradient: it equals the true gradient
# only when the section runs parallel to flow (|∇h|·cos θ otherwise). A
# least-squares plane through every measured well of a series gives the
# plan-view gradient vector, so the section's obliquity to flow can be stated.
# Screening estimate (the three-point problem generalised to n wells), not a
# calibrated flow model.

# Near-collinear wells: minor/major singular value ratio of the centred E/N
# matrix below this, or an rms spread across the best-fit line below
# ``COLLINEAR_MIN_SPREAD_M``, leaves the cross-line gradient undetermined.
COLLINEAR_SINGULAR_RATIO = 0.05
COLLINEAR_MIN_SPREAD_M = 1.0
# Below this |∇h| the head surface is treated as flat (no flow direction).
FLAT_GRADIENT = 1e-6
# Validate warnings: poor planarity.
POOR_FIT_R_SQUARED = 0.8
POOR_FIT_RESIDUAL_M = 0.5
# Section ↔ flow classification thresholds (degrees between flow and line).
ALONG_SECTION_MAX_DEG = 20.0
NEARLY_PERPENDICULAR_MIN_DEG = 70.0

_COMPASS_8 = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


@dataclass(frozen=True)
class HeadPoint:
    """One well's head for the plan-view fit (collar easting/northing)."""

    hole_id: str
    easting: float
    northing: float
    head_masl: float


@dataclass(frozen=True)
class HeadPlaneFit:
    """Planar head fit ``h = a·E + b·N + c`` for one series / connect_group nest.

    ``flow_azimuth_deg`` is the direction of *decreasing* head (−∇h), degrees
    clockwise from grid north; ``None`` when collinear or flat. ``r_squared``
    is ``None`` for the exact 3-point solution (and for perfectly flat heads);
    3-point residuals are 0 by construction.
    """

    series_id: str
    connect_group: str
    series_label: str
    hole_ids: tuple[str, ...]
    method: str  # "3-point" | "planar fit"
    dh_de: float
    dh_dn: float
    gradient: float
    flow_azimuth_deg: float | None
    r_squared: float | None
    max_abs_residual_m: float
    residuals_m: tuple[tuple[str, float], ...]
    collinear: bool
    singular_ratio: float

    @property
    def n_wells(self) -> int:
        return len(self.hole_ids)

    @property
    def usable(self) -> bool:
        """≥3 non-collinear wells with a finite gradient."""
        return self.n_wells >= 3 and not self.collinear and math.isfinite(self.gradient)

    @property
    def display_label(self) -> str:
        label = self.series_label or self.series_id
        return f"{label} [{self.connect_group}]" if self.connect_group else label

    @property
    def poor_fit(self) -> bool:
        """R² < 0.8 with ≥4 wells, or any residual > 0.5 m."""
        if not self.usable:
            return False
        low_r2 = (
            self.n_wells >= 4
            and self.r_squared is not None
            and self.r_squared < POOR_FIT_R_SQUARED
        )
        return low_r2 or self.max_abs_residual_m > POOR_FIT_RESIDUAL_M


@dataclass(frozen=True)
class SectionFlowRelation:
    """How a section line sits relative to the fitted plan-view flow direction."""

    section_azimuth_deg: float
    angle_deg: float  # 0–90 between flow and the (undirected) section line
    along_section_gradient: float  # |∇h| · cos θ — the apparent gradient magnitude
    classification: str  # "along section" | "oblique" | "nearly perpendicular"


def compass_point(azimuth_deg: float) -> str:
    """8-point compass name for an azimuth (degrees clockwise from north)."""
    index = int(((azimuth_deg % 360.0) + 22.5) // 45.0) % 8
    return _COMPASS_8[index]


def fit_head_plane(
    points: Sequence[HeadPoint],
    *,
    series_id: str = "default",
    connect_group: str = "",
    series_label: str = "",
    collinear_ratio: float = COLLINEAR_SINGULAR_RATIO,
    collinear_min_spread_m: float = COLLINEAR_MIN_SPREAD_M,
) -> HeadPlaneFit | None:
    """Least-squares plane through ≥3 well heads; ``None`` with fewer wells.

    Coordinates are centred before solving (UTM eastings would otherwise
    swamp the system). Three wells give the exact three-point solution.
    Near-collinear wells return a fit flagged ``collinear`` with a NaN
    gradient — the gradient across the line of wells is undetermined.
    """
    import numpy as np

    if len(points) < 3:
        return None
    hole_ids = tuple(point.hole_id for point in points)
    east = np.array([float(point.easting) for point in points])
    north = np.array([float(point.northing) for point in points])
    head = np.array([float(point.head_masl) for point in points])
    centred = np.column_stack([east - east.mean(), north - north.mean()])
    singular = np.linalg.svd(centred, compute_uv=False)
    major = float(singular[0])
    minor = float(singular[1])
    ratio = minor / major if major > 0.0 else 0.0
    rms_cross_spread = minor / math.sqrt(len(points))
    collinear = (
        major <= 0.0 or ratio < collinear_ratio or rms_cross_spread < collinear_min_spread_m
    )
    method = "3-point" if len(points) == 3 else "planar fit"
    if collinear:
        nan = float("nan")
        return HeadPlaneFit(
            series_id=series_id,
            connect_group=connect_group,
            series_label=series_label,
            hole_ids=hole_ids,
            method=method,
            dh_de=nan,
            dh_dn=nan,
            gradient=nan,
            flow_azimuth_deg=None,
            r_squared=None,
            max_abs_residual_m=nan,
            residuals_m=tuple((hid, nan) for hid in hole_ids),
            collinear=True,
            singular_ratio=ratio,
        )
    design = np.column_stack([centred, np.ones(len(points))])
    exact = len(points) == 3
    if exact:
        coeffs = np.linalg.solve(design, head)
    else:
        coeffs, *_ = np.linalg.lstsq(design, head, rcond=None)
    dh_de, dh_dn = float(coeffs[0]), float(coeffs[1])
    residuals = np.zeros(len(points)) if exact else head - design @ coeffs
    gradient = math.hypot(dh_de, dh_dn)
    azimuth: float | None = None
    if gradient > FLAT_GRADIENT:
        # Flow follows −∇h; atan2(east, north) is a bearing from grid north.
        azimuth = math.degrees(math.atan2(-dh_de, -dh_dn)) % 360.0
    r_squared: float | None = None
    if not exact:
        ss_tot = float(np.sum((head - head.mean()) ** 2))
        ss_res = float(np.sum(residuals**2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else None
    return HeadPlaneFit(
        series_id=series_id,
        connect_group=connect_group,
        series_label=series_label,
        hole_ids=hole_ids,
        method=method,
        dh_de=dh_de,
        dh_dn=dh_dn,
        gradient=gradient,
        flow_azimuth_deg=azimuth,
        r_squared=r_squared,
        max_abs_residual_m=float(np.max(np.abs(residuals))),
        residuals_m=tuple(
            (hid, float(value)) for hid, value in zip(hole_ids, residuals, strict=True)
        ),
        collinear=False,
        singular_ratio=ratio,
    )


def head_points_by_group(
    collars: Sequence[Collar],
    water_levels: Sequence[WaterLevel],
    *,
    deviation_readings: Sequence[DeviationReading] | None = None,
) -> dict[tuple[str, str], tuple[str, list[HeadPoint]]]:
    """Measured heads grouped by ``(series_id, connect_group)`` → ``(label, points)``.

    Same grouping as the renderer (blank connect_group = one shared nest);
    dry / NM readings are excluded and the last reading per hole wins. Heads
    come from ``water_head_masl`` (btoc stick-up; surveyed holes at TVD).
    Plan position is the collar easting/northing.
    """
    from projection import build_along_hole_tvd

    collar_by_id = {collar.hole_id: collar for collar in collars}
    tvd_tables = build_along_hole_tvd(collars, deviation_readings)
    grouped: dict[tuple[str, str], dict[str, WaterLevel]] = {}
    labels: dict[tuple[str, str], str] = {}
    for level in water_levels:
        if not is_measured_water_level(level) or level.hole_id not in collar_by_id:
            continue
        key = (level.series_id or "default", (level.connect_group or "").strip())
        grouped.setdefault(key, {})[level.hole_id] = level  # last wins
        if level.series_label and key not in labels:
            labels[key] = level.series_label
    result: dict[tuple[str, str], tuple[str, list[HeadPoint]]] = {}
    for key, by_hole in grouped.items():
        points: list[HeadPoint] = []
        for hole_id, level in by_hole.items():
            collar = collar_by_id[hole_id]
            rl = float(collar.elevation)
            table = tvd_tables.get(hole_id)
            md_to_elev = (
                (lambda md, _t=table, _rl=rl: _t.elevation_at(_rl, md)) if table else None
            )
            head = water_head_masl(
                level, rl, stick_up_m=collar.stick_up_m, md_to_elevation=md_to_elev
            )
            if not math.isfinite(head):
                continue
            points.append(
                HeadPoint(hole_id, float(collar.easting), float(collar.northing), head)
            )
        result[key] = (labels.get(key, ""), points)
    return result


def fit_head_planes(
    collars: Sequence[Collar],
    water_levels: Sequence[WaterLevel],
    *,
    deviation_readings: Sequence[DeviationReading] | None = None,
) -> tuple[HeadPlaneFit, ...]:
    """``fit_head_plane`` per series / nest with ≥3 measured wells (every collar given, not only on-section)."""
    fits: list[HeadPlaneFit] = []
    grouped = head_points_by_group(
        collars, water_levels, deviation_readings=deviation_readings
    )
    for (series_id, group), (label, points) in grouped.items():
        fit = fit_head_plane(
            points, series_id=series_id, connect_group=group, series_label=label
        )
        if fit is not None:
            fits.append(fit)
    return tuple(fits)


def section_flow_relation(
    fit: HeadPlaneFit, section_azimuth_deg: float | None
) -> SectionFlowRelation | None:
    """Angle between fitted flow and the section line, and the on-section gradient.

    The section line is undirected, so the angle folds to 0–90°. A section can
    only show the component ``|∇h| · cos θ`` (the apparent gradient). Returns
    ``None`` when the fit has no direction (collinear / flat) or the section
    bearing is unknown. Polyline sections use the overall start→end bearing
    (``projection.transect_azimuth_deg``).
    """
    if not fit.usable or fit.flow_azimuth_deg is None or section_azimuth_deg is None:
        return None
    diff = abs((fit.flow_azimuth_deg - float(section_azimuth_deg)) % 180.0)
    angle = min(diff, 180.0 - diff)
    if angle < ALONG_SECTION_MAX_DEG:
        classification = "along section"
    elif angle > NEARLY_PERPENDICULAR_MIN_DEG:
        classification = "nearly perpendicular"
    else:
        classification = "oblique"
    return SectionFlowRelation(
        section_azimuth_deg=float(section_azimuth_deg) % 360.0,
        angle_deg=angle,
        along_section_gradient=fit.gradient * math.cos(math.radians(angle)),
        classification=classification,
    )


def format_flow_direction(fit: HeadPlaneFit) -> str:
    """``"toward 128° (SE)"``, or why there is no direction."""
    if fit.collinear or not math.isfinite(fit.gradient):
        return "undetermined (collinear wells)"
    if fit.flow_azimuth_deg is None:
        return "flat (no measurable gradient)"
    return f"toward {fit.flow_azimuth_deg:.0f}° ({compass_point(fit.flow_azimuth_deg)})"


def head_plane_fit_warnings(
    fit: HeadPlaneFit,
    relation: SectionFlowRelation | None = None,
) -> list[str]:
    """Validate warnings: collinear wells, non-planar heads, flow ⟂ section."""
    label = fit.display_label
    if fit.collinear:
        return [
            f"Groundwater flow ({label}): wells {', '.join(fit.hole_ids)} are nearly "
            "collinear — plan-view flow direction cannot be fitted."
        ]
    warnings: list[str] = []
    if fit.poor_fit:
        r2 = f"R² = {fit.r_squared:.2f}, " if fit.r_squared is not None else ""
        warnings.append(
            f"Groundwater flow ({label}): heads not planar ({r2}max residual "
            f"{fit.max_abs_residual_m:.2f} m) — possible mounding, pumping, or mixed aquifers."
        )
    if relation is not None and relation.classification == "nearly perpendicular":
        warnings.append(
            f"Groundwater flow ({label}) is nearly perpendicular to the section line "
            f"({relation.angle_deg:.0f}°) — on-section gradient labels are apparent gradients."
        )
    return warnings


def groundwater_flow_note(
    fit: HeadPlaneFit,
    relation: SectionFlowRelation | None,
    *,
    include_series: bool = False,
) -> str | None:
    """One figure QA line for a usable fit; ``None`` for collinear wells."""
    if not fit.usable:
        return None
    series = f", {fit.display_label}" if include_series else ""
    text = f"Groundwater flow ({fit.method}, {fit.n_wells} wells{series}): "
    if fit.flow_azimuth_deg is None:
        return text + "heads are flat — no measurable plan-view gradient."
    text += f"{format_flow_direction(fit)}, i = {fit.gradient:.2g}"
    if fit.poor_fit:
        text += " (heads not planar — approximate)"
    if relation is None:
        return text + "."
    angle = f"{relation.angle_deg:.0f}°"
    if relation.classification == "along section":
        return text + f"; section runs along flow ({angle}) — gradients shown are close to true."
    if relation.classification == "nearly perpendicular":
        return text + (
            f"; section is nearly perpendicular to flow ({angle}) — "
            "gradients shown are apparent and say little about flow."
        )
    return text + f"; section is oblique ({angle}) — gradients shown are apparent."


def groundwater_flow_notes(
    collars: Sequence[Collar],
    water_levels: Sequence[WaterLevel],
    section_azimuth_deg: float | None,
    *,
    deviation_readings: Sequence[DeviationReading] | None = None,
) -> tuple[str, ...]:
    """Figure QA lines for every series / nest with ≥3 non-collinear wells."""
    fits = [
        fit
        for fit in fit_head_planes(collars, water_levels, deviation_readings=deviation_readings)
        if fit.usable
    ]
    include_series = len(fits) > 1
    notes: list[str] = []
    for fit in fits:
        note = groundwater_flow_note(
            fit,
            section_flow_relation(fit, section_azimuth_deg),
            include_series=include_series,
        )
        if note:
            notes.append(note)
    return tuple(notes)


# ---------------------------------------------------------------------------
# LNAPL (free product) head correction
#
# Field convention: ``WaterLevel.depth`` (or ``elevation_masl``) is the measured
# depth to WATER — the oil/water interface in the well — and
# ``product_thickness_m`` the apparent product thickness above it, along the
# hole on the same datum. The corrected (equivalent freshwater) head is
#
#     h = interface elevation + vertical product thickness × SG
#
# with SG = ``product_density`` (default 0.75). On a surveyed (inclined) hole
# the vertical thickness comes from the survey; elsewhere it equals the
# apparent thickness. Apparent thickness in a well overstates the formation
# thickness — the figure shows it as measured.
# ---------------------------------------------------------------------------

APPARENT_LNAPL_THICKNESS_WARN_M = 1.0


def lnapl_thickness_m(level: WaterLevel) -> float:
    """Apparent LNAPL thickness in the well (m); 0 when none was recorded."""
    value = getattr(level, "product_thickness_m", None)
    if value is None:
        return 0.0
    value = float(value)
    return value if math.isfinite(value) and value > 0.0 else 0.0


def has_lnapl(level: WaterLevel) -> bool:
    """True when the reading records measurable free product (thickness > 0)."""
    return lnapl_thickness_m(level) > 0.0


def lnapl_specific_gravity(level: WaterLevel) -> float:
    """Product specific gravity (``product_density``; default 0.75)."""
    value = getattr(level, "product_density", None)
    return float(value) if value is not None else DEFAULT_LNAPL_SPECIFIC_GRAVITY


def lnapl_product_top_masl(
    level: WaterLevel,
    collar_rl: float,
    *,
    stick_up_m: float | None = None,
    md_to_elevation: MdToElevation | None = None,
    interface_masl: float | None = None,
) -> float | None:
    """Elevation (masl) of the air/product interface, ``None`` without product."""
    thickness = lnapl_thickness_m(level)
    if thickness <= 0.0:
        return None
    if interface_masl is None:
        interface_masl = water_head_masl(
            level,
            collar_rl,
            stick_up_m=stick_up_m,
            md_to_elevation=md_to_elevation,
            lnapl_corrected=False,
        )
    if level.elevation_masl is not None or md_to_elevation is None:
        return float(interface_masl) + thickness
    md_water = water_depth_along_hole_bgs(level, stick_up_m)
    if md_water < 0.0:
        # Whole product column inside the (vertical) stick-up.
        return float(interface_masl) + thickness
    md_product = md_water - thickness
    if md_product < 0.0:
        # Product top inside the stick-up, above the collar.
        return float(collar_rl) - md_product
    return float(md_to_elevation(md_product))


def lnapl_vertical_thickness_m(
    level: WaterLevel,
    collar_rl: float,
    *,
    stick_up_m: float | None = None,
    md_to_elevation: MdToElevation | None = None,
    interface_masl: float | None = None,
) -> float:
    """Vertical product thickness (m): the apparent thickness, or from the survey."""
    thickness = lnapl_thickness_m(level)
    if thickness <= 0.0:
        return 0.0
    if level.elevation_masl is not None or md_to_elevation is None:
        return thickness
    if interface_masl is None:
        interface_masl = water_head_masl(
            level,
            collar_rl,
            stick_up_m=stick_up_m,
            md_to_elevation=md_to_elevation,
            lnapl_corrected=False,
        )
    top = lnapl_product_top_masl(
        level,
        collar_rl,
        stick_up_m=stick_up_m,
        md_to_elevation=md_to_elevation,
        interface_masl=interface_masl,
    )
    return max(0.0, float(top) - float(interface_masl)) if top is not None else 0.0


def lnapl_head_correction_m(
    level: WaterLevel,
    collar_rl: float,
    *,
    stick_up_m: float | None = None,
    md_to_elevation: MdToElevation | None = None,
    interface_masl: float | None = None,
) -> float:
    """Head rise (m) from the LNAPL correction: vertical thickness × SG."""
    vertical = lnapl_vertical_thickness_m(
        level,
        collar_rl,
        stick_up_m=stick_up_m,
        md_to_elevation=md_to_elevation,
        interface_masl=interface_masl,
    )
    if vertical <= 0.0:
        return 0.0
    return vertical * lnapl_specific_gravity(level)


def lnapl_interval_masl(
    level: WaterLevel,
    collar_rl: float,
    *,
    stick_up_m: float | None = None,
    md_to_elevation: MdToElevation | None = None,
) -> tuple[float, float] | None:
    """``(product top, oil/water interface)`` elevations (masl) for the figure band."""
    if not has_lnapl(level):
        return None
    interface = water_head_masl(
        level,
        collar_rl,
        stick_up_m=stick_up_m,
        md_to_elevation=md_to_elevation,
        lnapl_corrected=False,
    )
    top = lnapl_product_top_masl(
        level,
        collar_rl,
        stick_up_m=stick_up_m,
        md_to_elevation=md_to_elevation,
        interface_masl=interface,
    )
    return (float(top), float(interface)) if top is not None else None


def format_lnapl_thickness_label(level: WaterLevel) -> str:
    """Figure label suffix, e.g. ``"LNAPL 0.32 m"`` (apparent thickness)."""
    return f"LNAPL {lnapl_thickness_m(level):.2f} m"


def lnapl_correction_note(levels: Sequence[WaterLevel]) -> str | None:
    """QA line naming the LNAPL-corrected wells, ``None`` when there are none.

    "Groundwater elevations corrected for LNAPL at MW-3 (0.32 m, SG 0.75)" —
    one entry per distinct well / thickness / SG, in reading order; dry / NM
    readings are skipped (no head is plotted).
    """
    entries: list[str] = []
    for level in levels:
        if not has_lnapl(level) or not is_measured_water_level(level):
            continue
        text = (
            f"{level.hole_id} ({lnapl_thickness_m(level):.2f} m, "
            f"SG {lnapl_specific_gravity(level):.2f})"
        )
        if text not in entries:
            entries.append(text)
    if not entries:
        return None
    return "Groundwater elevations corrected for LNAPL at " + ", ".join(entries)
