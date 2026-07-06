"""Multi-format export helpers for cross-section figures."""

from __future__ import annotations

from io import BytesIO
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from renderer import CrossSectionRenderer
from stratigraphy import GeologicalPolygon


def export_section_pdf(
    renderer: CrossSectionRenderer,
    polygons: list[GeologicalPolygon],
    projected_df: pd.DataFrame,
    *,
    collar_depths: dict[str, float] | None = None,
    water_levels: Sequence | None = None,
    lithology_codes: Sequence[str] | None = None,
    qa_lines: Sequence[str] = (),
    section_figure: Figure | None = None,
) -> bytes:
    """Build a two-page PDF: section figure + legend/metadata summary."""
    owns_figure = section_figure is None
    section_fig = section_figure or renderer.render(
        polygons,
        projected_df,
        collar_depths=collar_depths,
        water_levels=water_levels,
        lithology_codes=lithology_codes,
    )
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        pdf.savefig(section_fig, bbox_inches="tight")
        if owns_figure:
            plt.close(section_fig)

        summary_fig, summary_ax = plt.subplots(figsize=(8.5, 11.0))
        summary_ax.axis("off")
        summary_ax.set_title("Cross Section Studio — Report Summary", fontsize=14, fontweight="bold", loc="left")
        y = 0.92
        summary_ax.text(0.05, y, renderer.title, fontsize=12, fontweight="bold", transform=summary_ax.transAxes)
        y -= 0.05
        if renderer.disclaimer:
            summary_ax.text(0.05, y, renderer.disclaimer, fontsize=9, style="italic", transform=summary_ax.transAxes)
            y -= 0.05
        for line in renderer._metadata_footer_lines():
            summary_ax.text(0.05, y, line, fontsize=9, transform=summary_ax.transAxes)
            y -= 0.04
        if lithology_codes:
            summary_ax.text(0.05, y, "Lithology codes:", fontsize=10, fontweight="bold", transform=summary_ax.transAxes)
            y -= 0.04
            for code in lithology_codes:
                summary_ax.text(0.07, y, f"• {code}", fontsize=9, transform=summary_ax.transAxes)
                y -= 0.035
        if qa_lines:
            summary_ax.text(0.05, y, "QA notes:", fontsize=10, fontweight="bold", transform=summary_ax.transAxes)
            y -= 0.04
            for line in qa_lines[:12]:
                summary_ax.text(0.07, y, f"• {line}", fontsize=8, transform=summary_ax.transAxes)
                y -= 0.03
        pdf.savefig(summary_fig, bbox_inches="tight")
        plt.close(summary_fig)
    buffer.seek(0)
    return buffer.getvalue()
