"""Process-wide render lock (leaf module — no app imports).

Matplotlib keeps process-wide state (rcParams, pyplot's figure registry). Two
Streamlit sessions rendering at once could reset each other's figure to a bare
``FigureCanvasBase`` ("no attribute get_renderer") or pick up another render's
temporary rcParams. Every render → export → close, and every helper that
changes rcParams while drawing, runs under this lock.
"""

from __future__ import annotations

import threading

RENDER_LOCK = threading.RLock()
