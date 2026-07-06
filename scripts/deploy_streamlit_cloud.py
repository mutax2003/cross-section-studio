"""Deploy Cross Section Studio to Streamlit Community Cloud via API."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

API = "https://api.streamlit.io/v1/apps"
REPO = "mutax2003/cross-section-studio"
BRANCH = "main"
MAIN_FILE = "app.py"
APP_NAME = "cross-section-studio"


def main() -> int:
    token = os.environ.get("STREAMLIT_API_TOKEN", "").strip()
    if not token:
        print(
            "Set STREAMLIT_API_TOKEN (from share.streamlit.io → Settings → API tokens).",
            file=sys.stderr,
        )
        return 1

    body = json.dumps(
        {"repo": REPO, "branch": BRANCH, "mainFile": MAIN_FILE, "appName": APP_NAME}
    ).encode()
    req = urllib.request.Request(
        API,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            app = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        print(f"Deploy failed ({exc.code}): {detail}", file=sys.stderr)
        return 1

    print(json.dumps(app, indent=2))
    url = app.get("url")
    if url:
        print(f"\nLive app: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
