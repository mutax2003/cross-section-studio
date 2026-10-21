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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from models import ScreenInterval, VerticalGradient, WaterLevel

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
) -> float:
    """Head elevation (masl) of a water reading.

    * explicit ``elevation_masl`` is returned as given;
    * otherwise the along-hole depth below ground (``btoc`` minus stick-up) is
      converted with ``md_to_elevation`` (hole survey: collar − TVD) — vertical
      holes (``None``) use collar − depth, the legacy formula;
    * water above ground inside the casing (btoc depth < stick-up) sits
      ``stick_up − depth`` above the collar.
    """
    if level.elevation_masl is not None:
        return float(level.elevation_masl)
    if getattr(level, "depth_reference", "bgs") != "btoc" and md_to_elevation is None:
        return float(collar_rl) - float(level.depth)
    md_below_ground = water_depth_along_hole_bgs(level, stick_up_m)
    if md_below_ground < 0.0 or md_to_elevation is None:
        return float(collar_rl) - md_below_ground
    return float(md_to_elevation(md_below_ground))


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
    ``level.depth`` unchanged.
    """
    if (
        level.elevation_masl is None
        and getattr(level, "depth_reference", "bgs") != "btoc"
        and md_to_elevation is None
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
