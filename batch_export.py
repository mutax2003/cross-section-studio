"""Multi-transect batch export and PDF binder helpers."""

from __future__ import annotations

import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from models import Collar, ConsultingTitleBlock, ParseResult
from parse_ops import subset_parse_result
from pipeline import (
    ALL_EXPORT_FORMATS,
    SectionGeometry,
    compute_section_geometry,
    render_cross_section_from_geometry,
)
from section_build_request import SectionBuildRequest

_GEOMETRY_MEMO_MAX = 8
_geometry_memo: OrderedDict[str, SectionGeometry] = OrderedDict()


def _memo_section_geometry(cache_key: str, factory) -> SectionGeometry:
    """Process-local LRU for multi-transect ZIP rebuilds (no Streamlit cache)."""
    cached = _geometry_memo.get(cache_key)
    if cached is not None:
        _geometry_memo.move_to_end(cache_key)
        return cached
    geometry = factory()
    _geometry_memo[cache_key] = geometry
    while len(_geometry_memo) > _GEOMETRY_MEMO_MAX:
        _geometry_memo.popitem(last=False)
    return geometry


def clear_batch_geometry_memo() -> None:
    """Clear the process-local geometry memo (tests / long-running workers)."""
    _geometry_memo.clear()


@dataclass(frozen=True)
class BatchTransectSpec:
    """One batch member: label + ordered holes (+ optional explicit transect polyline)."""

    label: str
    hole_ids: tuple[str, ...]
    transect_points: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        if len(self.hole_ids) < 2:
            raise ValueError(f"Batch transect {self.label!r}: requires at least two holes")
        if self.transect_points is not None and len(self.transect_points) < 2:
            raise ValueError(f"Batch transect {self.label!r}: transect_points needs ≥2 points")


def transect_points_from_collars(
    collars: Sequence[Collar],
    hole_ids: Sequence[str],
) -> tuple[tuple[float, float], ...]:
    """Build a hole-sequence transect polyline from collar easting/northing."""
    lookup = {collar.hole_id: collar for collar in collars}
    missing = [hole_id for hole_id in hole_ids if hole_id not in lookup]
    if missing:
        raise ValueError("Unknown collar(s): " + ", ".join(missing))
    return tuple((lookup[hole_id].easting, lookup[hole_id].northing) for hole_id in hole_ids)


def parse_batch_transect_lines(text: str) -> list[BatchTransectSpec]:
    """Parse lines of ``Label | hole1, hole2, …`` (pipe required).

    Blank lines are skipped. Raises ``ValueError`` on malformed rows.
    """
    specs: list[BatchTransectSpec] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "|" not in line:
            raise ValueError(
                f"Batch line {line!r}: expected 'Label | hole1, hole2, …' "
                "(pipe separates label from hole IDs)"
            )
        label_part, holes_part = line.split("|", 1)
        label = label_part.strip()
        hole_ids = tuple(
            part.strip() for part in holes_part.replace(";", ",").split(",") if part.strip()
        )
        if not label:
            raise ValueError(f"Batch line {line!r}: empty label")
        if len(hole_ids) < 2:
            raise ValueError(f"Batch line {line!r}: need at least two hole IDs")
        specs.append(BatchTransectSpec(label=label, hole_ids=hole_ids))
    return specs


def _consulting_for_spec(
    base: ConsultingTitleBlock | None,
    *,
    label: str,
    hole_ids: Sequence[str],
) -> ConsultingTitleBlock | None:
    if base is None:
        return ConsultingTitleBlock(
            section_label=label,
            transect_start_primary=hole_ids[0],
            transect_end_primary=hole_ids[-1],
        )
    return base.model_copy(
        update={
            "section_label": label or base.section_label,
            "transect_start_primary": hole_ids[0],
            "transect_end_primary": hole_ids[-1],
        }
    )


def prepare_batch_section_request(
    parse_result: ParseResult,
    base_request: SectionBuildRequest,
    spec: BatchTransectSpec,
) -> tuple[ParseResult, SectionBuildRequest]:
    """Subset workbook data and clone the base request for one batch transect."""
    subset = subset_parse_result(parse_result, spec.hole_ids)
    if len(subset.collars) < 2:
        raise ValueError(
            f"Batch transect {spec.label!r}: need ≥2 collars with data "
            f"(got {len(subset.collars)})"
        )
    if not subset.lithologies:
        raise ValueError(f"Batch transect {spec.label!r}: no lithology intervals")
    points = spec.transect_points or transect_points_from_collars(
        parse_result.collars, spec.hole_ids
    )
    title = base_request.section_title
    if spec.label and spec.label not in title:
        title = f"{base_request.section_title} — {spec.label}"
    consulting = _consulting_for_spec(
        base_request.consulting_title_block,
        label=spec.label,
        hole_ids=spec.hole_ids,
    )
    request = base_request.model_copy(
        update={
            "transect_points": tuple(points),
            "section_title": title,
            "consulting_title_block": consulting,
            "correlation_overrides": tuple(subset.correlation_overrides)
            + tuple(base_request.correlation_overrides),
            "water_levels": subset.water_levels,
            "screen_intervals": subset.screen_intervals,
            "vertical_gradients": subset.vertical_gradients,
            "faults": subset.faults,
            "unconformities": subset.unconformities,
            "environmental_readings": subset.environmental_readings,
            "deviation_readings": subset.deviation_readings,
        }
    )
    return subset, request


def build_one_transect_exports(
    parse_result: ParseResult,
    base_request: SectionBuildRequest,
    spec: BatchTransectSpec,
    *,
    export_formats: frozenset[str] | None = None,
) -> tuple[str, bytes, bytes, bytes]:
    """Rebuild one transect; reuse process-local geometry when payloads match."""
    subset, request = prepare_batch_section_request(parse_result, base_request, spec)
    formats = export_formats or ALL_EXPORT_FORMATS
    hole_ids = tuple(collar.hole_id for collar in subset.collars)
    geometry_key = request.geometry_cache_key(hole_ids)

    def _compute() -> SectionGeometry:
        return compute_section_geometry(
            subset.collars,
            subset.lithologies,
            request.transect_points,
            offset_warning_m=request.offset_warning_m,
            interpretation_mode=request.interpretation_mode,
            allow_pinch_outs=request.allow_pinch_outs,
            fail_on_overlaps=False,
            max_offset_for_interpolation_m=request.max_offset_for_interpolation_m,
            correlation_overrides=request.correlation_overrides,
            deviation_readings=request.deviation_readings,
            warn_on_correlation_gaps=False,
        )

    geometry = _memo_section_geometry(geometry_key, _compute)
    if request.fail_on_overlaps and geometry.overlap_pairs:
        raise ValueError(
            f"Polygon overlap detected ({len(geometry.overlap_pairs)} pair(s)); "
            "resolve correlation or set fail_on_overlaps=False to export."
        )
    result = render_cross_section_from_geometry(
        geometry,
        request.transect_points,
        vertical_exaggeration=request.vertical_exaggeration,
        show_hatches=request.show_hatches,
        show_legend=request.show_legend,
        title=request.section_title,
        interpretation_mode=request.interpretation_mode,
        water_levels=request.water_levels or None,
        uncertainty_spacing_m=request.uncertainty_spacing_m,
        uncertainty_offset_m=request.uncertainty_offset_m,
        faults=request.faults,
        unconformities=request.unconformities,
        environmental_readings=request.environmental_readings,
        figure_metadata=request.figure_metadata,
        show_ground_surface=request.show_ground_surface,
        interpolate_water_table=request.interpolate_water_table,
        show_water_elevation_labels=request.show_water_elevation_labels,
        show_water_legend=request.show_water_legend,
        show_dry_well_nm=request.show_dry_well_nm,
        water_interpolate_across_gaps=request.water_interpolate_across_gaps,
        environmental_parameters=request.environmental_parameters or None,
        show_parameter_labels=request.show_parameter_labels,
        parameter_interpolate_segments=request.parameter_interpolate_segments,
        parameter_interpolate_across_gaps=request.parameter_interpolate_across_gaps,
        parameter_draw_markers=request.parameter_draw_markers,
        parameter_marker_size=request.parameter_marker_size,
        parameter_draw_leaders=request.parameter_draw_leaders,
        parameter_label_include_units=request.parameter_label_include_units,
        column_header_detail=request.column_header_detail,
        show_scale_bar=request.show_scale_bar,
        show_ve_annotation=request.show_ve_annotation,
        show_parameter_legend_text=request.show_parameter_legend_text,
        export_font_family=request.export_font_family,
        export_font_size=request.export_font_size,
        selected_water_series_ids=request.selected_water_series_ids or None,
        water_line_solid=request.water_line_solid,
        legend_ncol=request.legend_ncol,
        chemistry_color_mode=request.chemistry_color_mode,
        chemistry_threshold_green_max=request.chemistry_threshold_green_max,
        chemistry_threshold_yellow_max=request.chemistry_threshold_yellow_max,
        render_layout=request.render_layout,
        track_width_m=request.track_width_m,
        auto_fit_track_width=request.auto_fit_track_width,
        elevation_mode=request.elevation_mode,
        raster_log_strips=request.raster_log_strips,
        export_formats=formats,
        consulting_title_block=request.consulting_title_block,
        screen_intervals=request.screen_intervals or None,
        vertical_gradients=request.vertical_gradients or None,
        export_framing=request.export_framing,
    )
    return spec.label, result.svg_bytes, result.png_bytes, result.pdf_bytes


def build_multi_transect_exports(
    parse_result: ParseResult,
    base_request: SectionBuildRequest,
    specs: Sequence[BatchTransectSpec],
    *,
    export_formats: frozenset[str] | None = None,
) -> list[tuple[str, bytes, bytes, bytes]]:
    """Rebuild each transect; return ``(stem, svg, png, pdf)`` entries for ZIP packaging."""
    if not specs:
        return []
    return [
        build_one_transect_exports(
            parse_result,
            base_request,
            spec,
            export_formats=export_formats,
        )
        for spec in specs
    ]


def _merge_pdfs_pypdf(section_pdfs: Sequence[bytes], *, cover_title: str) -> bytes | None:
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        return None
    writer = PdfWriter()
    cover_buf = BytesIO()
    with PdfPages(cover_buf) as pdf:
        cover, ax = plt.subplots(figsize=(8.5, 11.0))
        ax.axis("off")
        ax.text(
            0.5,
            0.55,
            cover_title,
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.45,
            f"{len(section_pdfs)} section(s)",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        pdf.savefig(cover, bbox_inches="tight")
        plt.close(cover)
    writer.append(PdfReader(BytesIO(cover_buf.getvalue())))
    for payload in section_pdfs:
        writer.append(PdfReader(BytesIO(payload)))
    out = BytesIO()
    writer.write(out)
    return out.getvalue()


def export_binder_pdf(section_pdfs: Sequence[bytes], *, cover_title: str = "Cross Section Report") -> bytes:
    """Combine prepared single-section PDF bytes into one binder document."""
    valid = [payload for payload in section_pdfs if payload]
    if not valid:
        return b""
    if len(valid) == 1 or len({payload for payload in valid}) == 1:
        return valid[0]
    merged = _merge_pdfs_pypdf(valid, cover_title=cover_title)
    if merged is not None:
        return merged
    # Fallback without pypdf: cover page only (individual PDFs still land in the ZIP).
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        cover, ax = plt.subplots(figsize=(8.5, 11.0))
        ax.axis("off")
        ax.text(
            0.5,
            0.55,
            cover_title,
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.45,
            f"{len(valid)} section(s) — install pypdf for full binder merge",
            ha="center",
            va="center",
            fontsize=11,
            transform=ax.transAxes,
        )
        pdf.savefig(cover, bbox_inches="tight")
        plt.close(cover)
    return buffer.getvalue()


def build_batch_zip(
    entries: Sequence[tuple[str, bytes, bytes, bytes]],
    *,
    binder_pdf: bytes | None = None,
) -> bytes:
    """Zip multiple transect exports. Each entry is (stem, svg, png, pdf)."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for stem, svg_bytes, png_bytes, pdf_bytes in entries:
            if svg_bytes:
                archive.writestr(f"{stem}.svg", svg_bytes)
            if png_bytes:
                archive.writestr(f"{stem}.png", png_bytes)
            if pdf_bytes:
                archive.writestr(f"{stem}.pdf", pdf_bytes)
        if binder_pdf:
            archive.writestr("report_binder.pdf", binder_pdf)
    buffer.seek(0)
    return buffer.getvalue()
