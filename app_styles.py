"""Shared Streamlit CSS for Cross Section Studio."""

APP_CSS = """
<style>
    :root {
        --brand-dark: #1e3a2f;
        --brand-mid: #2e6b4f;
        --brand-light: #3d8b5f;
        --surface: #ffffff;
        --border: #e2e8f0;
        --text: #1e293b;
        --muted: #64748b;
    }
    .app-hero {
        background: linear-gradient(135deg, var(--brand-dark) 0%, var(--brand-mid) 55%, var(--brand-light) 100%);
        padding: 0.85rem 1.15rem 0.7rem;
        border-radius: 12px;
        margin-bottom: 0.55rem;
        color: #f8fafc;
        box-shadow: 0 4px 14px rgba(30, 58, 47, 0.14);
    }
    .app-hero h1 { color: #f8fafc !important; margin: 0; font-size: 1.35rem; letter-spacing: -0.02em; }
    .app-hero p { margin: 0.2rem 0 0; opacity: 0.92; font-size: 0.84rem; line-height: 1.35; }
    .app-hero.compact {
        padding: 0.45rem 0.85rem 0.4rem;
        margin-bottom: 0.35rem;
    }
    .app-hero.compact h1 { font-size: 1.05rem; }
    .app-hero.compact p { display: none; }
    .app-hero.compact .workflow { margin-top: 0.3rem; }
    .app-hero.compact .workflow-step { padding: 0.28rem 0.4rem; font-size: 0.75rem; }
    /* Once a section exists: brand + stepper on ONE line so the figure is above the fold. */
    .app-hero.oneline {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        padding: 0.35rem 0.6rem 0.35rem 0.85rem;
    }
    .app-hero.oneline h1 { font-size: 0.98rem; white-space: nowrap; padding: 0; }
    .app-hero.oneline .workflow { flex: 1 1 auto; margin: 0; flex-wrap: nowrap; }
    .app-hero.oneline .workflow-step { padding: 0.18rem 0.35rem; }
    .generate-strip {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.65rem;
        background: linear-gradient(180deg, #f8fafc 0%, #fff 100%);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.4rem 0.75rem;
        margin: 0;
        font-size: 0.84rem;
        color: #334155;
    }
    .generate-strip .strip-title {
        font-size: 1rem !important;
        font-weight: 700;
        color: var(--text);
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.35;
    }
    .generate-strip .strip-status { flex: 1 1 12rem; color: #475569; }
    .generate-strip [data-testid="stHeaderActionElements"] { display: none; }
    /* Streamlit pulls markdown up by -1rem; undo it so the strip centres on the button. */
    [data-testid="stMarkdownContainer"]:has(> .generate-strip),
    [data-testid="stMarkdownContainer"]:has(> .profile-header) { margin-bottom: 0; }
    .generate-strip.is-stale { background: #fffbeb; border-color: #fde68a; }
    .generate-strip.is-stale .strip-status { color: #92400e; font-weight: 600; }
    .workflow {
        display: flex;
        gap: 0.35rem;
        flex-wrap: wrap;
        margin: 0.45rem 0 0.15rem;
    }
    /* Solid dark step background: white text stays >= 4.5:1 across the gradient. */
    .workflow-step {
        flex: 1 1 6.5rem;
        background: rgba(0,0,0,0.25);
        border: 1px solid rgba(255,255,255,0.22);
        border-radius: 8px;
        padding: 0.35rem 0.5rem;
        font-size: 0.75rem;
        color: #ffffff;
        text-align: center;
    }
    .workflow-step.active {
        background: rgba(255,255,255,0.95);
        color: var(--brand-dark);
        font-weight: 600;
        border-color: transparent;
    }
    .workflow-step.done { color: #ffffff; }
    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.8rem 0.9rem;
        text-align: center;
        min-height: 4.5rem;
    }
    .metric-card .value { font-size: 1.45rem; font-weight: 700; color: var(--text); line-height: 1.2; }
    .metric-card .label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.15rem; }
    .metric-card.ok { border-color: #86efac; background: linear-gradient(180deg, #f0fdf4 0%, #fff 100%); }
    .metric-card.warn { border-color: #fcd34d; background: linear-gradient(180deg, #fffbeb 0%, #fff 100%); }
    .metric-card.error { border-color: #fca5a5; background: linear-gradient(180deg, #fef2f2 0%, #fff 100%); }
    .section-card,
    .st-key-section_card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.6rem 1rem 0.6rem;
        margin-top: 0;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }
    .profile-header {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin: 0;
    }
    .chip {
        display: inline-block;
        padding: 0.22rem 0.55rem;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 600;
        border: 1px solid var(--border);
        background: #f8fafc;
        color: #334155;
    }
    .chip.brand { background: #ecfdf5; border-color: #a7f3d0; color: #065f46; }
    .chip.warn { background: #fffbeb; border-color: #fde68a; color: #92400e; }
    .chip.more { border-style: dashed; cursor: help; color: #475569; }
    .chip.more:focus-visible { outline: 2px solid var(--brand-mid); outline-offset: 2px; }
    /* Compact preview zoom, right-aligned beside the chips. */
    .st-key-svg_preview_zoom [data-testid="stButtonGroup"] { justify-content: flex-end; }
    .st-key-svg_preview_zoom button {
        min-height: 1.75rem !important;
        padding: 0.1rem 0.6rem !important;
    }
    .st-key-svg_preview_zoom button p { font-size: 0.8rem !important; }
    .legend-swatch {
        display: inline-block;
        width: 24px;
        height: 17px;
        border: 1px solid #334155;
        border-radius: 4px;
        margin-right: 8px;
        vertical-align: middle;
        background-size: 6px 6px, 6px 6px;
        background-position: 0 0, 3px 3px;
    }
    .legend-row { margin: 0.38rem 0; font-size: 0.84rem; color: #334155; line-height: 1.35; }
    .welcome-card {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px dashed #cbd5e1;
        border-radius: 14px;
        padding: 1.75rem 1.5rem;
        color: #475569;
    }
    .welcome-card h3 { margin: 0 0 0.5rem; color: #1e293b; }
    .welcome-steps { text-align: left; margin: 1rem auto 0; max-width: 34rem; }
    .welcome-steps li { margin: 0.35rem 0; }
    .stale-banner {
        background: linear-gradient(180deg, #fffbeb 0%, #fefce8 100%);
        border: 1px solid #fde68a;
        color: #92400e;
        border-radius: 10px;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.75rem;
        font-size: 0.88rem;
    }
    .stale-banner:focus-within {
        outline: 2px solid var(--brand-mid);
        outline-offset: 2px;
    }
    .next-step-coach {
        position: sticky;
        top: 0.35rem;
        z-index: 2;
        background: linear-gradient(180deg, #ecfdf5 0%, #f0fdf4 100%);
        border: 1px solid #a7f3d0;
        color: #065f46;
        border-radius: 10px;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.75rem;
        font-size: 0.88rem;
    }
    .next-step-coach:focus-within {
        outline: 2px solid var(--brand-mid);
        outline-offset: 2px;
    }
    .sidebar-section-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--muted);
        margin: 0.35rem 0 0.15rem;
        font-weight: 700;
    }
    .svg-frame {
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
        background: #fff;
    }
    .svg-frame--zoomed {
        overflow: auto;
        max-height: 80vh;
    }
    .svg-frame--zoomed:focus-visible {
        outline: 2px solid var(--brand-mid);
        outline-offset: 2px;
    }
    .app-menubar,
    .st-key-app_menubar {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.25rem 0.5rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
    }
    .app-menubar [data-testid="stHorizontalBlock"],
    .st-key-app_menubar [data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    .app-menubar button[kind="secondary"],
    .app-menubar button,
    .st-key-app_menubar [data-testid="stPopover"] button {
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        border: 1px solid transparent !important;
        background: transparent !important;
        color: var(--text) !important;
        min-height: 2rem !important;
    }
    .st-key-app_menubar [data-testid="stCaptionContainer"] { margin: 0; }
    .app-menubar button:hover,
    .st-key-app_menubar [data-testid="stPopover"] button:hover {
        background: #e2e8f0 !important;
        border-color: #cbd5e1 !important;
    }
    .app-menubar button:focus-visible,
    .st-key-app_menubar [data-testid="stPopover"] button:focus-visible {
        outline: 2px solid var(--brand-mid) !important;
        outline-offset: 2px !important;
        background: #ecfdf5 !important;
    }
    .menu-shortcut {
        float: right;
        color: var(--muted);
        font-size: 0.75rem;
        font-weight: 500;
        margin-left: 0.75rem;
    }
    .app-menu-accels,
    .st-key-menu_accels {
        position: absolute !important;
        width: 1px !important;
        height: 1px !important;
        overflow: hidden !important;
        clip: rect(0, 0, 0, 0) !important;
        white-space: nowrap !important;
        border: 0 !important;
        padding: 0 !important;
        margin: -1px !important;
        visibility: hidden !important;
    }
    .st-key-shortcut_bridge iframe { visibility: hidden !important; }
    /* Streamlit wraps each keyed container; take the WRAPPERS out of flow
       too, or each zero-height block still costs a 16px flex gap. */
    div:has(> .st-key-menu_accels),
    div:has(> .st-key-shortcut_bridge) {
        position: absolute !important;
        width: 1px !important;
        height: 1px !important;
        overflow: hidden !important;
    }
    /* Out of flow: an in-flow zero-height block still takes a flex gap. */
    .st-key-shortcut_bridge {
        position: absolute !important;
        width: 1px !important;
        height: 1px !important;
        overflow: hidden !important;
        margin: 0 !important;
    }
    div[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    div[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        font-weight: 600;
    }
    /* WCAG 2.2 AA: captions and success text >= 4.5:1 (Streamlit dims captions to ~4:1). */
    [data-testid="stCaptionContainer"] { opacity: 1 !important; color: #475569 !important; }
    [data-testid="stAlertContentSuccess"] { color: #166534 !important; }
    /* One solid, >=3:1 focus indicator for every interactive control. */
    button:focus-visible,
    summary:focus-visible,
    a:focus-visible,
    [role="switch"]:focus-visible,
    [role="radio"]:focus-visible,
    input[type="checkbox"]:focus-visible,
    [data-baseweb="select"] input:focus-visible {
        outline: 2px solid var(--brand-mid) !important;
        outline-offset: 2px !important;
    }
    /* Text/number inputs, selects and text areas: outline the whole field. */
    [data-baseweb="input"]:focus-within,
    [data-baseweb="select"] > div:focus-within,
    [data-baseweb="textarea"]:focus-within,
    textarea:focus-visible {
        outline: 2px solid var(--brand-mid) !important;
        outline-offset: 1px !important;
    }
    /* File uploader hint ("50MB per file · XLSX") at >= 4.5:1. */
    [data-testid="stFileUploaderDropzoneInstructions"] span { color: #475569 !important; }
    /* Reclaim default top padding so results sit higher on the page. */
    [data-testid="stMainBlockContainer"] { padding-top: 2.25rem; }
    /* Narrow windows and phones: keep File/Edit/View/Help on one row at their
       natural width (scroll sideways if needed) instead of truncating ("F…")
       or stacking into four full-width rows. */
    @media (max-width: 1100px) {
        .st-key-app_menubar [data-testid="stHorizontalBlock"] {
            flex-wrap: nowrap !important;
            overflow-x: auto;
            gap: 0.25rem;
        }
        .st-key-app_menubar [data-testid="stColumn"] {
            flex: 0 0 auto !important;
            width: auto !important;
            min-width: 0 !important;
        }
    }
    @media (max-width: 640px) {
        .st-key-app_menubar [data-testid="stCaptionContainer"] { display: none; }
        /* Downloads as a 2 × 2 grid rather than four stacked rows. */
        .st-key-section_card [data-testid="stHorizontalBlock"]:has([data-testid="stDownloadButton"]) {
            flex-wrap: wrap !important;
        }
        .st-key-section_card [data-testid="stHorizontalBlock"]:has([data-testid="stDownloadButton"]) > [data-testid="stColumn"] {
            min-width: calc(50% - 0.5rem) !important;
            flex: 1 1 calc(50% - 0.5rem) !important;
        }
        .app-hero.oneline { flex-wrap: wrap; }
        .app-hero.oneline .workflow { flex-wrap: wrap; }
    }
    @media (prefers-reduced-motion: reduce) {
        .workflow-step, .metric-card, .section-card {
            transition: none;
        }
    }
</style>
"""
