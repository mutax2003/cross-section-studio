"""Multi-transect batch export and PDF binder helpers."""

from __future__ import annotations

import hashlib
import re
import threading
import zipfile
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from export_framing import _sanitize_stem
from models import Collar, ConsultingTitleBlock, ParseResult
from parse_ops import subset_parse_result
from pipeline import (
    SectionGeometry,
    compute_section_geometry,
    render_cross_section_from_geometry,
)
from section_build_request import SectionBuildRequest

# Default batch deliverables skip SVG encode (usually the slowest); opt in via export_formats.
BATCH_DEFAULT_EXPORT_FORMATS = frozenset({"png", "pdf"})

_GEOMETRY_MEMO_MAX = 8
_geometry_memo: OrderedDict[str, SectionGeometry] = OrderedDict()
_geometry_memo_lock = threading.Lock()


def _memo_section_geometry(cache_key: str, factory) -> SectionGeometry:
    """Process-local LRU for multi-transect ZIP rebuilds (no Streamlit cache)."""
    with _geometry_memo_lock:
        cached = _geometry_memo.get(cache_key)
        if cached is not None:
            _geometry_memo.move_to_end(cache_key)
            return cached
    geometry = factory()
    with _geometry_memo_lock:
        _geometry_memo[cache_key] = geometry
        while len(_geometry_memo) > _GEOMETRY_MEMO_MAX:
            _geometry_memo.popitem(last=False)
    return geometry


def clear_batch_geometry_memo() -> None:
    """Clear the process-local geometry memo (tests / long-running workers)."""
    with _geometry_memo_lock:
        _geometry_memo.clear()


@dataclass(frozen=True)
class BatchTransectSpec:
    """One batch member: label + ordered holes (+ optional explicit transect polyline)."""

    label: str
    hole_ids: tuple[str, ...]
    transect_points: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        if len(set(self.hole_ids)) < 2:
            raise ValueError(
                f"Batch transect {self.label!r}: requires at least two distinct holes"
            )
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


@dataclass(frozen=True)
class BatchLineStatus:
    """Validation result for one non-blank line of the batch section-line editor."""

    line_number: int
    text: str
    label: str
    spec: BatchTransectSpec | None
    problem: str | None = None

    @property
    def ok(self) -> bool:
        return self.spec is not None and self.problem is None

    @property
    def message(self) -> str:
        """Short plain-language status, e.g. ``"C-C': MW-99 not in Collars"``."""
        name = self.label or f"Line {self.line_number}"
        if self.ok and self.spec is not None:
            return f"{name}: {len(self.spec.hole_ids)} holes, ready"
        return f"{name}: {self.problem}"


def validate_batch_transect_lines(
    text: str,
    known_hole_ids: Sequence[str] | None = None,
) -> list[BatchLineStatus]:
    """Check each ``Label | hole1, hole2, …`` line on its own; never raises.

    Unlike :func:`parse_batch_transect_lines`, a bad line does not stop the
    others: each non-blank line gets a :class:`BatchLineStatus` with either a
    ready ``spec`` or a plain-language ``problem`` (missing ``|``, no name,
    fewer than two holes, hole IDs not in Collars, repeated name). When
    ``known_hole_ids`` is None the Collars check is skipped.
    """
    known = None if known_hole_ids is None else set(known_hole_ids)
    statuses: list[BatchLineStatus] = []
    seen_labels: dict[str, int] = {}
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue

        def _bad(label: str, problem: str, _n=line_number, _line=line) -> None:
            statuses.append(BatchLineStatus(_n, _line, label, None, problem))

        if "|" not in line:
            _bad("", "add '|' between the name and the hole IDs (e.g. A-A' | MW-01, MW-02)")
            continue
        label_part, holes_part = line.split("|", 1)
        label = label_part.strip()
        hole_ids = tuple(
            part.strip() for part in holes_part.replace(";", ",").split(",") if part.strip()
        )
        if not label:
            _bad("", "give the section line a name before '|'")
            continue
        if label in seen_labels:
            _bad(label, f"name already used on line {seen_labels[label]}")
            continue
        seen_labels[label] = line_number
        if known is not None:
            unknown = [hole_id for hole_id in hole_ids if hole_id not in known]
            if unknown:
                _bad(label, ", ".join(unknown) + " not in Collars")
                continue
        repeated = sorted({hole_id for hole_id in hole_ids if hole_ids.count(hole_id) > 1})
        if repeated:
            _bad(label, ", ".join(repeated) + " listed more than once")
            continue
        if len(hole_ids) < 2:
            _bad(label, "needs at least 2 holes")
            continue
        statuses.append(
            BatchLineStatus(
                line_number, line, label, BatchTransectSpec(label=label, hole_ids=hole_ids)
            )
        )
    return statuses


def split_batch_transect_lines(
    text: str,
    known_hole_ids: Sequence[str] | None = None,
) -> tuple[list[BatchTransectSpec], list[BatchLineStatus]]:
    """Return ``(valid specs, skipped line statuses)`` so a batch can build what it can."""
    statuses = validate_batch_transect_lines(text, known_hole_ids)
    valid = [status.spec for status in statuses if status.ok and status.spec is not None]
    skipped = [status for status in statuses if not status.ok]
    return valid, skipped


_SECTION_ENDS_RE = re.compile(r"^\s*([A-Za-z0-9]+)\s*[-\u2013]\s*([A-Za-z0-9]+['\u2032\u2019]*)\s*$")


_TITLE_SECTION_LABEL_RE = re.compile(r"\b([A-Za-z0-9]{1,3})\s*[-\u2013]\s*\1['\u2032\u2019]")


def _section_end_letters(label: str) -> tuple[str, str]:
    """``"B-B'"`` → ``("B", "B'")``; anything else → no end letters.

    The end labels sit beside the edge boreholes' own names, so hole IDs must
    not be used here (they printed twice on every batch sheet).
    """
    match = _SECTION_ENDS_RE.match(label or "")
    if match is None:
        return "", ""
    return match.group(1), match.group(2)


def _consulting_for_spec(
    base: ConsultingTitleBlock | None,
    *,
    label: str,
    hole_ids: Sequence[str],
) -> ConsultingTitleBlock | None:
    start, end = _section_end_letters(label)
    # The base sheet's compass words describe the base section's direction,
    # not this transect's, so they are cleared rather than copied.
    ends = {
        "transect_start_primary": start,
        "transect_end_primary": end,
        "transect_start_secondary": "",
        "transect_end_secondary": "",
        "transect_start_label": "",
        "transect_end_label": "",
    }
    if base is None:
        return ConsultingTitleBlock(section_label=label, **ends)
    return base.model_copy(update={"section_label": label or base.section_label, **ends})



def label_in_title(label: str, title: str) -> bool:
    """True when ``label`` appears in ``title`` as a whole token.

    A plain substring test treated label "A" as present in "Bay Area Site"
    and dropped it from batch file names.
    """
    label = label.strip()
    if not label:
        return False
    return re.search(_label_pattern(label), title) is not None


def _label_pattern(label: str) -> str:
    return rf"(?<![A-Za-z0-9]){re.escape(label.strip())}(?![A-Za-z0-9])"


def _batch_section_title(base_request: SectionBuildRequest, label: str) -> str:
    """Title for one batch transect.

    The base title usually names the base section ("Site X A-A'"); swap that
    label for this line's instead of appending ("Site X A-A' — B-B'").
    """
    title = base_request.section_title
    if not label:
        return title
    block = base_request.consulting_title_block
    base_label = (block.section_label if block else "").strip()
    if base_label and base_label != label and label_in_title(base_label, title):
        return re.sub(_label_pattern(base_label), lambda _m: label, title, count=1)
    if label_in_title(label, title):
        return title
    # No title block (section-sheet styles): look for an "A-A'" style label.
    match = _TITLE_SECTION_LABEL_RE.search(title)
    if match is not None and _SECTION_ENDS_RE.match(label):
        return title[: match.start()] + label + title[match.end() :]
    return f"{title} — {label}"

def batch_section_title(base_request: SectionBuildRequest, label: str) -> str:
    """Public alias: the title one batch sheet prints (and is named after)."""
    return _batch_section_title(base_request, label)


def batch_cover_title(base_request: SectionBuildRequest, labels: Sequence[str]) -> str:
    """Binder cover / ZIP name for the whole batch, not the previewed section.

    "Site X A-A'" with lines A-A', B-B', C-C' → "Site X — Cross Sections A-A', B-B', C-C'".
    """
    title = base_request.section_title.strip()
    block = base_request.consulting_title_block
    base_label = (block.section_label if block else "").strip()
    if base_label and label_in_title(base_label, title):
        title = re.sub(_label_pattern(base_label), "", title)
    else:
        title = _TITLE_SECTION_LABEL_RE.sub("", title)
    title = title.strip(" -—–:,")
    shown = [label for label in labels if label]
    noun = "Cross Section" if len(shown) == 1 else "Cross Sections"
    sections = f"{noun} {', '.join(shown)}" if shown else "Cross Sections"
    return f"{title} — {sections}" if title else sections


def prepare_batch_section_request(
    parse_result: ParseResult,
    base_request: SectionBuildRequest,
    spec: BatchTransectSpec,
) -> tuple[ParseResult, SectionBuildRequest]:
    """Subset workbook data and clone the base request for one batch transect."""
    subset = subset_parse_result(parse_result, spec.hole_ids)
    if len({collar.hole_id for collar in subset.collars}) < 2:
        raise ValueError(
            f"Batch transect {spec.label!r}: need ≥2 collars with data "
            f"(got {len(subset.collars)})"
        )
    if not subset.lithologies:
        raise ValueError(f"Batch transect {spec.label!r}: no lithology intervals")
    points = spec.transect_points or transect_points_from_collars(
        parse_result.collars, spec.hole_ids
    )
    title = _batch_section_title(base_request, spec.label)
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
    formats = export_formats or BATCH_DEFAULT_EXPORT_FORMATS
    hole_ids = tuple(collar.hole_id for collar in subset.collars)
    # Key must cover the geology data too — geometry_cache_key hashes only the
    # request (transect/mode/overrides), and a re-uploaded workbook must not
    # serve stale polygons from the process-local memo.
    subset_digest = hashlib.sha256(subset.model_dump_json().encode("utf-8")).hexdigest()
    geometry_key = subset_digest + "|" + request.geometry_cache_key(hole_ids)

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
        chemistry_label_style=request.chemistry_label_style,
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


_LETTER_LANDSCAPE_IN = (11.0, 8.5)


def _cover_page_pdf(title: str, subtitle: str, *, page_size_in: tuple[float, float]) -> bytes:
    """One cover page at exactly ``page_size_in`` (no tight bbox: that cropped
    the page to the text, a non-standard ~489 x 624 pt sheet)."""
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        cover, ax = plt.subplots(figsize=page_size_in)
        ax.axis("off")
        ax.text(
            0.5,
            0.55,
            title,
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            wrap=True,
            transform=ax.transAxes,
        )
        ax.text(0.5, 0.45, subtitle, ha="center", va="center", fontsize=12, transform=ax.transAxes)
        pdf.savefig(cover)
        plt.close(cover)
    return buffer.getvalue()


def _merge_pdfs_pypdf(section_pdfs: Sequence[bytes], *, cover_title: str) -> bytes | None:
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        return None
    writer = PdfWriter()
    readers = [PdfReader(BytesIO(payload)) for payload in section_pdfs]
    page_size_in = _LETTER_LANDSCAPE_IN
    if readers and readers[0].pages:
        # Same sheet size and orientation as the section pages that follow.
        box = readers[0].pages[0].mediabox
        page_size_in = (float(box.width) / 72.0, float(box.height) / 72.0)
    cover = _cover_page_pdf(
        cover_title, f"{len(section_pdfs)} section(s)", page_size_in=page_size_in
    )
    writer.append(PdfReader(BytesIO(cover)))
    for reader in readers:
        writer.append(reader)
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
    return _cover_page_pdf(
        cover_title,
        f"{len(valid)} section(s) — install pypdf for full binder merge",
        page_size_in=_LETTER_LANDSCAPE_IN,
    )


def build_batch_zip(
    entries: Sequence[tuple[str, bytes, bytes, bytes]],
    *,
    binder_pdf: bytes | None = None,
) -> bytes:
    """Zip multiple transect exports. Each entry is (stem, svg, png, pdf)."""
    buffer = BytesIO()
    # The binder and readme names are taken up front so no section stem can
    # collide with them (a section labelled "report_binder").
    used: dict[str, int] = {"report_binder": 1, "readme": 1}
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for stem, svg_bytes, png_bytes, pdf_bytes in entries:
            # Sanitize (labels may carry path separators) and uniquify — zipfile
            # writes duplicate names silently and extractors keep only one.
            safe = _sanitize_stem(str(stem))
            key = safe.lower()
            count = used.get(key, 0)
            used[key] = count + 1
            if count:
                base = safe
                safe = f"{base}_{count + 1}"
                while safe.lower() in used:
                    count += 1
                    used[key] = count + 1
                    safe = f"{base}_{count + 1}"
                used[safe.lower()] = 1
            if svg_bytes:
                archive.writestr(f"{safe}.svg", svg_bytes)
            if png_bytes:
                archive.writestr(f"{safe}.png", png_bytes)
            if pdf_bytes:
                archive.writestr(f"{safe}.pdf", pdf_bytes)
        if binder_pdf:
            archive.writestr("report_binder.pdf", binder_pdf)
        archive.writestr("README.txt", _batch_readme(len(entries)).encode("utf-8"))
    buffer.seek(0)
    return buffer.getvalue()


def _batch_readme(section_count: int) -> str:
    """Plain-text note shipped in every batch ZIP (contents and attribution)."""
    from app_identity import COPYRIGHT_NOTICE, CREATED_BY

    return (
        f"Cross Section Studio batch export: {section_count} section line(s).\n"
        "Each section has a PNG (reports) and PDF (print); SVG (CAD) when selected.\n"
        "report_binder.pdf, when present, collects every section PDF behind a cover page.\n"
        f"{CREATED_BY}. {COPYRIGHT_NOTICE}\n"
    )
