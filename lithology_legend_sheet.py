"""Lithology legend sheet as a PNG, in the client legend 261023 style.

Two layouts, both drawn with the figures' own colours and hatch marks
(``constants.get_lithology_style`` + ``hatch_patterns``):

* ``"tall"`` (default) — the 261023 legend sheet look: tall vertical swatches
  with the name above each, in the client legend's order
  (``constants.LITHOLOGY_LEGEND_ORDER``), 13 per row.
* ``"rows"`` — one row per code: swatch, name, colour code and pattern, in
  separate columns sized from the longest name so text never prints over
  another column. Lists every palette code (legacy codes included).
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import hatch_patterns  # noqa: E402
from app_identity import COPYRIGHT_SHORT  # noqa: E402
from constants import (  # noqa: E402
    HATCH_LINE_COLOR,
    LITHOLOGY_LEGEND_ORDER,
    LITHOLOGY_SCHEME_ID,
    POLYGON_EDGE_COLOR,
    USGS_LITHOLOGY_COLORS,
    get_lithology_style,
)
from render_lock import RENDER_LOCK  # noqa: E402
from renderer_common import new_figure  # noqa: E402

LAYOUTS = ("tall", "rows")
DEFAULT_LAYOUT = "tall"
DEFAULT_TITLE = f"Lithology legend {LITHOLOGY_SCHEME_ID}"

_PATTERN_NAMES = {
    "": "plain",
    ".": "dots",
    "/": "45° dashes",
    "+": "plus marks",
    "O": "cobbles",
    "o": "small circles",
    "x": "cross-hatch",
    "-": "horizontal lines",
    "|": "vertical lines",
    "*": "stars",
    "\\": "back-slant lines",
}

# Row layout.
SWATCH_W_IN = 0.9
SWATCH_H_IN = 0.36
ROW_H_IN = 0.52
FONT_PT = 10

# Tall layout, proportioned like the 261023 sheet (CAD swatches 10 x 72 units
# on a 15-unit pitch, names 1.5 units tall above them).
TALL_SWATCH_W_IN = 0.55
TALL_SWATCH_H_IN = 3.9
TALL_PITCH_IN = 0.82
TALL_PER_ROW = 13
TALL_FONT_PT = 8
TALL_LABEL_CHARS = 11  # wrap names so neighbouring labels never touch
_HATCH_LINEWIDTH_PT = 0.65  # same hatch line as the figures


def pattern_name(hatch: str | None) -> str:
    """Plain words for a hatch string ("." -> "dots", "xxx" -> "cross-hatch")."""
    if not hatch:
        return "plain"
    names = list(dict.fromkeys(_PATTERN_NAMES[c] for c in hatch if c in _PATTERN_NAMES))
    return " + ".join(names) if names else "pattern"


def legend_codes(layout: str = DEFAULT_LAYOUT) -> list[str]:
    """Default codes: the client legend (tall) or every palette code (rows)."""
    if layout == "tall":
        return list(LITHOLOGY_LEGEND_ORDER)
    return list(USGS_LITHOLOGY_COLORS)


def tall_label(code: str) -> str:
    """Upper-case name wrapped on words for the tall layout ("SANDY CLAY\\nLOAM")."""
    return "\n".join(textwrap.wrap(code.upper(), TALL_LABEL_CHARS, break_long_words=False)) or code


def build_legend_sheet_png(
    codes: Sequence[str] | None = None,
    *,
    columns: int | None = None,
    dpi: int = 150,
    title: str | None = DEFAULT_TITLE,
    layout: str = DEFAULT_LAYOUT,
) -> bytes:
    """PNG legend sheet listing ``codes`` (default: see :func:`legend_codes`)."""
    # rc_context changes process-wide rcParams: hold the render lock so a
    # concurrent section render never draws with these hatch settings.
    with RENDER_LOCK, plt.rc_context(
        {"hatch.color": HATCH_LINE_COLOR, "hatch.linewidth": _HATCH_LINEWIDTH_PT}
    ):
        fig = build_legend_sheet_figure(codes, columns=columns, dpi=dpi, title=title, layout=layout)
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi)
        plt.close(fig)
    return buffer.getvalue()


def build_legend_sheet_figure(
    codes: Sequence[str] | None = None,
    *,
    columns: int | None = None,
    dpi: int = 150,
    title: str | None = DEFAULT_TITLE,
    layout: str = DEFAULT_LAYOUT,
):
    """The legend sheet as a matplotlib Figure (caller closes it).

    ``columns`` is swatches per row for ``"tall"`` (default 13) and the number
    of columns for ``"rows"`` (default 2).
    """
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown legend sheet layout {layout!r}; use one of {LAYOUTS}")
    hatch_patterns.install()
    codes = list(codes) if codes is not None else legend_codes(layout)
    if layout == "tall":
        return _tall_sheet(codes, per_row=columns or TALL_PER_ROW, dpi=dpi, title=title)
    return _row_sheet(codes, columns=columns or 2, dpi=dpi, title=title or "")


def _edge_colour(fill: str) -> str:
    """Swatches carry no outline (as on the CAD sheet) unless they are near white."""
    r, g, b = (int(fill[i : i + 2], 16) for i in (1, 3, 5))
    return POLYGON_EDGE_COLOR if 0.2126 * r + 0.7152 * g + 0.0722 * b > 235 else fill


def _tall_sheet(codes: list[str], *, per_row: int, dpi: int, title: str | None):
    per_row = max(1, int(per_row))
    rows = max(1, -(-len(codes) // per_row))
    label_lines = max((tall_label(code).count("\n") + 1 for code in codes), default=1)
    label_h_in = label_lines * TALL_FONT_PT * 1.25 / 72.0 + 0.12
    top_in = 0.55 if title else 0.25
    row_h_in = label_h_in + TALL_SWATCH_H_IN + 0.3
    width_in = 0.3 + min(per_row, max(1, len(codes))) * TALL_PITCH_IN + 0.1
    width_in = max(width_in, 3.0)
    height_in = top_in + rows * row_h_in + 0.3

    with RENDER_LOCK, plt.rc_context(
        {
            "hatch.color": HATCH_LINE_COLOR,
            "hatch.linewidth": _HATCH_LINEWIDTH_PT,
            "font.size": TALL_FONT_PT,
        }
    ):
        fig = new_figure((width_in, height_in), dpi=dpi)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, width_in)
        ax.set_ylim(height_in, 0)
        ax.axis("off")
        if title:
            ax.text(0.3, 0.3, title, fontsize=TALL_FONT_PT + 4, fontweight="bold", va="center")
        for index, code in enumerate(codes):
            row, col = divmod(index, per_row)
            centre_x = 0.3 + (col + 0.5) * TALL_PITCH_IN
            swatch_top = top_in + row * row_h_in + label_h_in
            style = get_lithology_style(code)
            ax.add_patch(
                Rectangle(
                    (centre_x - TALL_SWATCH_W_IN / 2, swatch_top),
                    TALL_SWATCH_W_IN,
                    TALL_SWATCH_H_IN,
                    facecolor=style.color,
                    edgecolor=_edge_colour(style.color),
                    linewidth=0.6,
                    hatch=style.hatch or None,
                )
            )
            label = ax.text(
                centre_x,
                swatch_top - 0.06,
                tall_label(code),
                ha="center",
                va="bottom",
                multialignment="center",
                fontsize=TALL_FONT_PT,
                linespacing=1.15,
            )
            label.set_gid(f"legend-label:{code}")
        ax.text(
            width_in - 0.15,
            height_in - 0.15,
            COPYRIGHT_SHORT,
            ha="right",
            va="center",
            fontsize=TALL_FONT_PT - 2,
            color="#64748B",
        )
    return fig


def _row_sheet(codes: list[str], *, columns: int, dpi: int, title: str):
    columns = max(1, int(columns))
    rows = max(1, -(-len(codes) // columns))
    longest = max((len(code) for code in codes), default=8)
    # Name column wide enough for the longest name at FONT_PT (approx. 0.6 em
    # per character), so names never run into the colour-code column.
    name_w_in = max(1.6, longest * FONT_PT * 0.6 / 72.0 + 0.2)
    code_w_in = 1.7
    col_w_in = 0.25 + SWATCH_W_IN + 0.2 + name_w_in + code_w_in
    width_in = columns * col_w_in + 0.3
    height_in = rows * ROW_H_IN + 1.0

    with RENDER_LOCK, plt.rc_context(
        {"hatch.color": HATCH_LINE_COLOR, "hatch.linewidth": _HATCH_LINEWIDTH_PT, "font.size": FONT_PT}
    ):
        fig = new_figure((width_in, height_in), dpi=dpi)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, width_in)
        ax.set_ylim(height_in, 0)
        ax.axis("off")
        ax.text(0.25, 0.4, title, fontsize=FONT_PT + 3, fontweight="bold", va="center")
        for index, code in enumerate(codes):
            col, row = divmod(index, rows)
            x0 = 0.25 + col * col_w_in
            y_mid = 0.85 + row * ROW_H_IN + ROW_H_IN / 2
            style = get_lithology_style(code)
            ax.add_patch(
                Rectangle(
                    (x0, y_mid - SWATCH_H_IN / 2),
                    SWATCH_W_IN,
                    SWATCH_H_IN,
                    facecolor=style.color,
                    edgecolor=POLYGON_EDGE_COLOR,
                    linewidth=0.6,
                    hatch=style.hatch or None,
                )
            )
            name_x = x0 + SWATCH_W_IN + 0.2
            ax.text(name_x, y_mid, code.upper(), va="center", fontsize=FONT_PT)
            ax.text(
                name_x + name_w_in,
                y_mid,
                f"{style.color.upper()} · {pattern_name(style.hatch)}",
                va="center",
                fontsize=FONT_PT - 1,
                color="#475569",
            )
        ax.text(
            width_in - 0.25,
            height_in - 0.2,
            COPYRIGHT_SHORT,
            ha="right",
            va="center",
            fontsize=FONT_PT - 3,
            color="#64748B",
        )
    return fig
