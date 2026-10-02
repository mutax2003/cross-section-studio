"""Quick-preview chart layout draws environmental readings when enabled."""

from __future__ import annotations

from models import Collar, EnvironmentalReading, Lithology
from pipeline import build_cross_section
from render_profiles import CHART_PROFILE
from renderer import CrossSectionRenderer
from tests.conftest import assert_valid_svg, run_pipeline

_IDS = ("BH-01", "BH-02")
_VALUES = (120.0, 245.0, 390.0)
_TRANSECT = [(0.0, 0.0), (20.0, 0.0)]


def _collars() -> list[Collar]:
    return [
        Collar(hole_id=h, easting=i * 10.0, northing=0.0, elevation=100.0, total_depth=12.0)
        for i, h in enumerate(_IDS)
    ]


def _lithologies() -> list[Lithology]:
    return [Lithology(hole_id=h, from_depth=0.0, to_depth=12.0, lithology_code="Clay") for h in _IDS]


def _readings() -> list[EnvironmentalReading]:
    return [
        EnvironmentalReading(hole_id=h, parameter="Chloride", value=v, depth=2.0 + 3.0 * k)
        for h in _IDS
        for k, v in enumerate(_VALUES)
    ]


def _value_texts(figure) -> list[str]:
    expected = {f"{v:g}" for v in _VALUES}
    return [t.get_text() for ax in figure.axes for t in ax.texts if t.get_text().strip() in expected]


def _render_chart(*, show_markers: bool) -> tuple[CrossSectionRenderer, object]:
    projected, polygons, _ = run_pipeline(_collars(), _lithologies(), _TRANSECT, render_layout="chart")
    renderer = CrossSectionRenderer(
        render_profile=CHART_PROFILE.model_copy(
            update={"show_parameter_markers": show_markers, "show_parameter_labels": True}
        ),
        environmental_readings=_readings(),
        environmental_parameters=("Chloride",),
    )
    figure = renderer.render(polygons, projected, collar_depths={h: 12.0 for h in _IDS})
    return renderer, figure


def test_chart_layout_draws_reading_labels_and_legend() -> None:
    renderer, figure = _render_chart(show_markers=True)
    try:
        labels = _value_texts(figure)
        assert len(labels) == len(_IDS) * len(_VALUES)
        assert renderer.parameter_series_legend
        assert [entry["parameter"] for entry in renderer.parameter_series_legend] == ["Chloride"]
        # The stock chart cosmetics are untouched by the parameter override.
        assert renderer.profile.layout == "chart"
        assert renderer.profile.show_centerline is CHART_PROFILE.show_centerline
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_chart_layout_markers_off_draws_no_readings() -> None:
    renderer, figure = _render_chart(show_markers=False)
    try:
        assert _value_texts(figure) == []
        assert renderer.parameter_series_legend == []
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_pipeline_chart_layout_emits_reading_labels_in_svg() -> None:
    _, _, svg_bytes, _, _, _, _ = build_cross_section(
        _collars(),
        _lithologies(),
        _TRANSECT,
        render_layout="chart",
        environmental_readings=_readings(),
        environmental_parameters=("Chloride",),
    )
    assert_valid_svg(svg_bytes)
    svg = svg_bytes.decode("utf-8", errors="ignore")
    for value in _VALUES:
        assert f"{value:g}" in svg


def test_chart_layout_honours_depth_below_collar_mode() -> None:
    """The sidebar elevation-mode radio applies to the quick-preview chart."""
    collars = _collars()
    collars[1] = collars[1].model_copy(update={"elevation": 104.0})
    projected, polygons, _ = run_pipeline(collars, _lithologies(), _TRANSECT, render_layout="chart")
    renderer = CrossSectionRenderer(
        render_profile=CHART_PROFILE.model_copy(update={"y_axis_mode": "depth_below_collar"})
    )
    figure = renderer.render(polygons, projected, collar_depths={h: 12.0 for h in _IDS})
    try:
        ax = figure.axes[0]
        assert ax.get_ylabel().startswith("Depth below collar")
        assert ax.yaxis_inverted()
        # Both collars plot at depth 0 regardless of their RL difference.
        y_lo, y_hi = sorted(ax.get_ylim())
        assert y_lo <= 0.0 < y_hi
        # Scale bar text stays at the visual bottom (numerically deep end).
        bar = next(t for t in ax.texts if t.get_text().endswith(" m") and "TD" not in t.get_text())
        assert bar.get_position()[1] > 0.5 * y_hi
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)
