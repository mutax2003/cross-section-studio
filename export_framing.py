"""Export framing, naming, and deliverable packaging helpers."""

from __future__ import annotations

import json
import re
import zipfile
from io import BytesIO
from typing import Literal, Mapping

_FILENAME_SAFE_RE = re.compile(r"[^\w\-]+")

from pydantic import BaseModel, Field

ExportPagePreset = Literal[
    "auto",
    "tight_fence",
    "title_block",
    "letter_portrait",
    "letter_landscape",
    "tabloid_landscape",
]

ExportFilenamePattern = Literal[
    "section_title",
    "project_figure_transect_rev",
]

PAGE_FIGSIZE_IN: dict[str, tuple[float, float]] = {
    "letter_portrait": (8.5, 11.0),
    "letter_landscape": (11.0, 8.5),
    "tabloid_landscape": (17.0, 11.0),
}


class ExportFramingConfig(BaseModel, frozen=True):
    """Cosmetic export controls — excluded from geometry cache."""

    page_preset: ExportPagePreset = "auto"
    margin_top_in: float = Field(default=0.0, ge=0.0, le=2.0)
    margin_bottom_in: float = Field(default=0.0, ge=0.0, le=2.0)
    margin_left_in: float = Field(default=0.0, ge=0.0, le=2.0)
    margin_right_in: float = Field(default=0.0, ge=0.0, le=2.0)
    export_dpi: int = Field(default=300, ge=72, le=600)
    show_draft_watermark: bool = False
    include_title_block: bool = True
    include_legend: bool = True
    include_water_table: bool = True
    include_qa_footer: bool = True
    fence_only: bool = False
    filename_pattern: ExportFilenamePattern = "section_title"
    export_revision: str = ""
    viewport_xmin: float | None = None
    viewport_xmax: float | None = None
    viewport_ymin: float | None = None
    viewport_ymax: float | None = None
    cad_svg_layers: bool = False

    def effective_page_preset(self, layout: str) -> ExportPagePreset:
        if self.fence_only:
            return "tight_fence"
        if self.page_preset != "auto":
            return self.page_preset
        if layout == "consulting_section":
            return "title_block"
        return "tight_fence"

    def pad_inches(self) -> float:
        return max(
            self.margin_top_in,
            self.margin_bottom_in,
            self.margin_left_in,
            self.margin_right_in,
            0.05,
        )


def savefig_kwargs(
    framing: ExportFramingConfig | None,
    *,
    layout: str,
) -> dict[str, object]:
    """Matplotlib savefig kwargs from export framing and layout."""
    config = framing or ExportFramingConfig()
    preset = config.effective_page_preset(layout)
    pad = config.pad_inches()
    if preset == "tight_fence":
        return {"bbox_inches": "tight", "pad_inches": pad}
    if preset == "title_block" or layout == "consulting_section":
        return {"bbox_inches": None, "pad_inches": pad}
    if preset in PAGE_FIGSIZE_IN:
        return {"bbox_inches": None, "pad_inches": pad}
    return {"bbox_inches": "tight", "pad_inches": pad}


def export_figsize_in(
    framing: ExportFramingConfig | None,
    *,
    layout: str,
    default: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    """Optional figure size override at export time."""
    config = framing or ExportFramingConfig()
    preset = config.effective_page_preset(layout)
    if preset in PAGE_FIGSIZE_IN:
        return PAGE_FIGSIZE_IN[preset]
    if preset == "title_block" or layout == "consulting_section":
        return default or PAGE_FIGSIZE_IN["letter_landscape"]
    return None


def _sanitize_stem(text: str, *, fallback: str = "cross_section") -> str:
    cleaned = _FILENAME_SAFE_RE.sub("_", text.strip())[:80].strip("_")
    return cleaned or fallback


def build_export_filename(
    *,
    pattern: ExportFilenamePattern | str,
    section_title: str,
    figure_number: str = "",
    project_number: str = "",
    transect_label: str = "",
    revision: str = "",
    draft: bool = False,
) -> str:
    """Return a sanitized filename stem for deliverables."""
    rev = revision.strip()
    if draft and rev and not rev.upper().startswith("DRAFT"):
        rev = f"DRAFT_{rev}"
    elif draft and not rev:
        rev = "DRAFT"

    if pattern == "project_figure_transect_rev":
        parts = [
            project_number or "project",
            figure_number or "fig",
            transect_label or _sanitize_stem(section_title, fallback="transect"),
        ]
        if rev:
            parts.append(rev)
        stem = "_".join(_sanitize_stem(part, fallback="x") for part in parts)
        return stem[:120].strip("_") or "cross_section"

    stem = _sanitize_stem(section_title)
    if rev:
        stem = f"{stem}_{_sanitize_stem(rev, fallback='rev')}"
    return stem[:120].strip("_") or "cross_section"


def build_report_package_bytes(
    *,
    stem: str,
    svg_bytes: bytes,
    png_bytes: bytes,
    pdf_bytes: bytes,
    metadata: Mapping[str, object],
    readme: str | None = None,
    docx_bytes: bytes | None = None,
) -> bytes:
    """Zip SVG, PNG, PDF, metadata, and optional DOCX."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if svg_bytes:
            archive.writestr(f"{stem}.svg", svg_bytes)
        if png_bytes:
            archive.writestr(f"{stem}.png", png_bytes)
        if pdf_bytes:
            archive.writestr(f"{stem}.pdf", pdf_bytes)
        if docx_bytes:
            archive.writestr(f"{stem}.docx", docx_bytes)
        archive.writestr(
            f"{stem}_metadata.json",
            json.dumps(metadata, indent=2, default=str).encode("utf-8"),
        )
        archive.writestr(
            "README_deliverable.txt",
            (readme or _default_readme(stem)).encode("utf-8"),
        )
    buffer.seek(0)
    return buffer.getvalue()


def _default_readme(stem: str) -> str:
    return (
        f"Cross Section Studio deliverable package: {stem}\n"
        "Contents: SVG (CAD), PNG (reports), PDF (print), metadata JSON.\n"
        "Import SVG into CAD; paste PNG into Word; file PDF for client binders.\n"
    )


def save_exports_to_directory(
    directory: str,
    *,
    stem: str,
    svg_bytes: bytes,
    png_bytes: bytes,
    pdf_bytes: bytes,
    metadata: Mapping[str, object],
    docx_bytes: bytes | None = None,
) -> list[str]:
    """Write export bytes to a project folder; returns written paths."""
    from pathlib import Path

    root = Path(directory).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    mapping = {
        f"{stem}.svg": svg_bytes,
        f"{stem}.png": png_bytes,
        f"{stem}.pdf": pdf_bytes,
        f"{stem}.docx": docx_bytes or b"",
        f"{stem}_metadata.json": json.dumps(metadata, indent=2, default=str).encode("utf-8"),
    }
    for name, payload in mapping.items():
        if not payload:
            continue
        path = root / name
        path.write_bytes(payload)
        written.append(str(path))
    return written


def png_clipboard_html(png_bytes: bytes) -> str:
    """Minimal HTML/JS to copy PNG bytes to the clipboard in the browser."""
    import base64

    if not png_bytes:
        return "<p>No PNG prepared yet.</p>"
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return (
        '<div><button id="copyPngBtn" type="button">Copy PNG to clipboard</button>'
        '<span id="copyPngStatus" style="margin-left:8px;"></span></div>'
        "<script>(function(){"
        "const b=document.getElementById('copyPngBtn'),s=document.getElementById('copyPngStatus');"
        "b.addEventListener('click',async()=>{"
        "try{const u=Uint8Array.from(atob('" + encoded + "'),c=>c.charCodeAt(0));"
        "await navigator.clipboard.write([new ClipboardItem({'image/png':new Blob([u],{type:'image/png'})})]);"
        "s.textContent='Copied';}catch(e){s.textContent='Copy failed — use Download PNG';}});"
        "})();</script>"
    )


def merge_framing_into_profile_updates(
    framing: ExportFramingConfig | None,
    profile_updates: dict[str, object],
) -> dict[str, object]:
    """Apply layer visibility toggles from export framing to render profile."""
    if framing is None:
        return profile_updates
    updates = dict(profile_updates)
    if framing.fence_only:
        updates["title_block"] = False
        updates["legend_in_title_block"] = False
        updates["show_scale_bar"] = False
        updates["show_ve_annotation"] = False
    elif not framing.include_title_block:
        updates["title_block"] = False
        updates["legend_in_title_block"] = False
    if not framing.include_legend:
        updates["legend_in_title_block"] = False
    if not framing.include_water_table:
        updates["show_water_elevation_labels"] = False
        updates["show_water_legend"] = False
        updates["interpolate_water_table_default"] = False
    return updates


def apply_export_page_size(
    fig,
    framing: ExportFramingConfig | None,
    *,
    layout: str,
) -> None:
    """Resize figure to letter/tabloid preset before raster/vector export."""
    size = export_figsize_in(framing, layout=layout)
    if size is not None:
        fig.set_size_inches(size[0], size[1], forward=True)


def apply_viewport_crop(fig, framing: ExportFramingConfig | None) -> None:
    """Restrict axis limits when viewport crop bounds are set."""
    if framing is None:
        return
    xmin = framing.viewport_xmin
    xmax = framing.viewport_xmax
    ymin = framing.viewport_ymin
    ymax = framing.viewport_ymax
    if xmin is None or xmax is None or ymin is None or ymax is None:
        return
    if xmin >= xmax or ymin >= ymax:
        return
    for axis in fig.axes:
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(ymin, ymax)


def apply_fixed_page_margins(
    fig,
    framing: ExportFramingConfig | None,
    *,
    layout: str,
) -> None:
    """Apply asymmetric margins on fixed-page presets (tight_fence uses pad_inches)."""
    if framing is None:
        return
    if framing.effective_page_preset(layout) == "tight_fence":
        return
    if (
        framing.margin_left_in
        == framing.margin_right_in
        == framing.margin_top_in
        == framing.margin_bottom_in
        == 0.0
    ):
        return
    width, height = fig.get_size_inches()
    if width <= 0 or height <= 0:
        return
    left = min(max(framing.margin_left_in / width, 0.0), 0.45)
    right = max(1.0 - min(framing.margin_right_in / width, 0.45), left + 0.05)
    bottom = min(max(framing.margin_bottom_in / height, 0.0), 0.45)
    top = max(1.0 - min(framing.margin_top_in / height, 0.45), bottom + 0.05)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)


def prepare_export_figure(
    fig,
    framing: ExportFramingConfig | None,
    *,
    layout: str,
) -> None:
    """Apply page size, margins, and viewport crop before savefig (not geometry)."""
    apply_export_page_size(fig, framing, layout=layout)
    apply_fixed_page_margins(fig, framing, layout=layout)
    apply_viewport_crop(fig, framing)


def apply_draft_watermark(fig, framing: ExportFramingConfig | None) -> None:
    """Overlay a semi-transparent DRAFT stamp on raster/PDF exports."""
    if framing is None or not framing.show_draft_watermark:
        return
    fig.text(
        0.5,
        0.5,
        "DRAFT",
        fontsize=48,
        color="red",
        alpha=0.25,
        ha="center",
        va="center",
        rotation=35,
        transform=fig.transFigure,
        zorder=1000,
    )


def annotate_svg_layers(svg_bytes: bytes) -> bytes:
    """Add Creator metadata hint for CAD layer import."""
    text = svg_bytes.decode("utf-8", errors="replace")
    if "Cross Section Studio CAD" in text:
        return svg_bytes
    if 'metadata={"Creator": "Cross Section Studio"}' in text:
        return text.replace(
            'metadata={"Creator": "Cross Section Studio"}',
            'metadata={"Creator": "Cross Section Studio CAD"}',
            1,
        ).encode("utf-8")
    return text.replace(
        "Cross Section Studio",
        "Cross Section Studio CAD",
        1,
    ).encode("utf-8")
