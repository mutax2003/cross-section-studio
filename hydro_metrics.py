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

from models import Collar, DeviationReading, WaterLevel

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
