"""Full lithology legend sheet (every palette code) as a PNG.

The swatches use the same colours and template hatch marks as the figures
(``constants.get_lithology_style`` + ``hatch_patterns``). Each row has three
separate columns — swatch, name, colour code and pattern — sized from the
longest name so text never prints over another column.
"""

from __future__ import annotations

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
    POLYGON_EDGE_COLOR,
    USGS_LITHOLOGY_COLORS,
    get_lithology_style,
)

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

SWATCH_W_IN = 0.9
SWATCH_H_IN = 0.36
ROW_H_IN = 0.52
FONT_PT = 10


def pattern_name(hatch: str | None) -> str:
    """Plain words for a hatch string ("." -> "dots", "xxx" -> "cross-hatch")."""
    if not hatch:
        return "plain"
    names = list(dict.fromkeys(_PATTERN_NAMES[c] for c in hatch if c in _PATTERN_NAMES))
    return " + ".join(names) if names else "pattern"


def legend_codes() -> list[str]:
    """Every palette code, in palette order."""
    return list(USGS_LITHOLOGY_COLORS)


def build_legend_sheet_png(
    codes: Sequence[str] | None = None,
    *,
    columns: int = 2,
    dpi: int = 150,
    title: str = "Lithology legend",
) -> bytes:
    """PNG legend sheet listing ``codes`` (default: the whole palette)."""
    with plt.rc_context({"hatch.color": HATCH_LINE_COLOR, "hatch.linewidth": 0.65}):
        fig = build_legend_sheet_figure(codes, columns=columns, dpi=dpi, title=title)
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi)
        plt.close(fig)
    return buffer.getvalue()


def build_legend_sheet_figure(
    codes: Sequence[str] | None = None,
    *,
    columns: int = 2,
    dpi: int = 150,
    title: str = "Lithology legend",
):
    """The legend sheet as a matplotlib Figure (caller closes it)."""
    hatch_patterns.install()
    codes = list(codes) if codes is not None else legend_codes()
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

    with plt.rc_context(
        {"hatch.color": HATCH_LINE_COLOR, "hatch.linewidth": 0.65, "font.size": FONT_PT}
    ):
        fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
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
