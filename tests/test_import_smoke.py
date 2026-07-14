"""Guard packaging and circular imports."""

from __future__ import annotations


def test_import_pipeline_and_result() -> None:
    from pipeline import CrossSectionResult, build_cross_section

    assert callable(build_cross_section)
    assert CrossSectionResult.__name__ == "CrossSectionResult"


def test_import_models_parsing_reexports() -> None:
    from typing import get_args

    from models import Collar, DataParser, InterpretationMode, subset_parse_result
    from parse_ops import apply_unit_order_fix
    from parsing import DataParser as Parser

    assert DataParser is Parser
    assert callable(subset_parse_result)
    assert callable(apply_unit_order_fix)
    assert Collar.__name__ == "Collar"
    assert set(get_args(InterpretationMode)) == {
        "borehole_only",
        "interpolated",
        "correlation_lines",
    }


def test_import_renderer_mixins() -> None:
    from renderer import CrossSectionRenderer
    from renderer_chart import ChartLayoutMixin
    from renderer_common import RendererGeometryMixin
    from renderer_consulting import ConsultingLayoutMixin
    from renderer_section_sheet import SectionSheetLayoutMixin

    assert issubclass(CrossSectionRenderer, ConsultingLayoutMixin)
    assert issubclass(CrossSectionRenderer, SectionSheetLayoutMixin)
    assert issubclass(CrossSectionRenderer, ChartLayoutMixin)
    assert issubclass(CrossSectionRenderer, RendererGeometryMixin)


def test_import_app_modules() -> None:
    import app_build
    import app_common
    import app_configure
    import app_generate
    import app_services
    import app_sidebar
    import app_state
    import app_styles
    import app_upload
    import app_validate
    import section_build_request
    from pipeline import ALL_EXPORT_FORMATS

    assert app_styles.APP_CSS
    assert callable(app_validate.render_validate_step)
    assert callable(app_configure.render_nl_transect_input)
    assert callable(app_generate.render_profile_and_downloads)
    assert callable(app_sidebar.render_sidebar)
    assert callable(app_upload.handle_workbook_upload)
    assert callable(app_build.collect_section_build_request)
    assert app_common._WORKFLOW_LABELS[0] == "Upload"
    assert section_build_request.SectionBuildRequest.__name__ == "SectionBuildRequest"
    assert callable(app_services.cached_build_section)
    assert callable(app_services.cached_build_section_exports)
    assert callable(app_services.cached_build_section_png)
    assert callable(app_services.cached_build_section_pdf)
    assert app_state.DEFAULT_SESSION
    assert "show_hatches" in app_state.DEFAULT_SESSION
    assert "enable_ai_suggestions" in app_state.DEFAULT_SESSION
    assert ALL_EXPORT_FORMATS == frozenset({"svg", "png", "pdf"})

    try:
        import app_menubar
    except ImportError:
        pass
    else:
        assert callable(app_menubar.render_menubar)