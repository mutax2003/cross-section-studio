"""Windows desktop entry point for Cross Section Studio (dev or PyInstaller bundle)."""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser

DEFAULT_PORT = 18501
PORT_SCAN_END = 18520


def _is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _pick_port(preferred: int, scan_end: int = PORT_SCAN_END) -> int:
    for port in range(preferred, scan_end + 1):
        if _is_port_free(port):
            return port
    raise RuntimeError(f"No free port in range {preferred}-{scan_end}")


def _resolve_port() -> int:
    env = os.environ.get("CROSS_SECTION_PORT")
    if env:
        preferred = int(env)
        if _is_port_free(preferred):
            return preferred
    return _pick_port(DEFAULT_PORT, PORT_SCAN_END)


def _run_streamlit(app_path: str, port: int) -> int:
    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        f"--server.port={port}",
    ]
    return stcli.main()


def main() -> int:
    from paths import app_root

    base = app_root()
    os.chdir(base)
    app_file = base / "app.py"
    if not app_file.is_file():
        print(f"Cross Section Studio: missing app.py in {base}", file=sys.stderr)
        return 1

    port = _resolve_port()
    url = f"http://localhost:{port}"

    def _open_browser() -> None:
        time.sleep(2.5)
        webbrowser.open(url)

    threading.Thread(target=_open_browser, daemon=True).start()
    print(f"Cross Section Studio starting at {url}")
    print("Close this window to stop the application.")
    return _run_streamlit(str(app_file), port) or 0


if __name__ == "__main__":
    raise SystemExit(main())
