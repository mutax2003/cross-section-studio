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
    .app-hero.compact .workflow-step { padding: 0.28rem 0.4rem; font-size: 0.68rem; }
    .generate-strip {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.65rem;
        background: linear-gradient(180deg, #f8fafc 0%, #fff 100%);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.45rem 0.75rem;
        margin-bottom: 0.55rem;
        font-size: 0.84rem;
        color: #334155;
    }
    .generate-strip .strip-status { flex: 1 1 12rem; color: var(--muted); }
    .generate-strip .strip-status strong { color: var(--text); }
    .workflow {
        display: flex;
        gap: 0.35rem;
        flex-wrap: wrap;
        margin: 0.45rem 0 0.15rem;
    }
    .workflow-step {
        flex: 1 1 6.5rem;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.22);
        border-radius: 8px;
        padding: 0.35rem 0.5rem;
        font-size: 0.74rem;
        color: #e2e8f0;
        text-align: center;
    }
    .workflow-step.active {
        background: rgba(255,255,255,0.95);
        color: var(--brand-dark);
        font-weight: 600;
        border-color: transparent;
    }
    .workflow-step.done { opacity: 0.88; }
    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.8rem 0.9rem;
        text-align: center;
        min-height: 4.5rem;
    }
    .metric-card .value { font-size: 1.45rem; font-weight: 700; color: var(--text); line-height: 1.2; }
    .metric-card .label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.15rem; }
    .metric-card.ok { border-color: #86efac; background: linear-gradient(180deg, #f0fdf4 0%, #fff 100%); }
    .metric-card.warn { border-color: #fcd34d; background: linear-gradient(180deg, #fffbeb 0%, #fff 100%); }
    .metric-card.error { border-color: #fca5a5; background: linear-gradient(180deg, #fef2f2 0%, #fff 100%); }
    .section-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.15rem 0.65rem;
        margin-top: 0.65rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }
    .profile-header {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.35rem 0 0.85rem;
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
        font-size: 0.72rem;
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
    .app-menubar {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.2rem 0.45rem 0.35rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
    }
    .app-menubar [data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    .app-menubar button[kind="secondary"],
    .app-menubar button {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        border: 1px solid transparent !important;
        background: transparent !important;
        color: var(--text) !important;
        min-height: 1.85rem !important;
    }
    .app-menubar button:hover {
        background: #e2e8f0 !important;
        border-color: #cbd5e1 !important;
    }
    .app-menubar button:focus-visible {
        outline: 2px solid var(--brand-mid) !important;
        outline-offset: 2px !important;
        background: #ecfdf5 !important;
    }
    .menu-shortcut {
        float: right;
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 500;
        margin-left: 0.75rem;
    }
    .app-menu-accels {
        position: absolute !important;
        width: 1px !important;
        height: 1px !important;
        overflow: hidden !important;
        clip: rect(0, 0, 0, 0) !important;
        white-space: nowrap !important;
        border: 0 !important;
        padding: 0 !important;
        margin: -1px !important;
    }
    div[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    div[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        font-weight: 600;
    }
    @media (prefers-reduced-motion: reduce) {
        .workflow-step, .metric-card, .section-card {
            transition: none;
        }
    }
</style>
"""
