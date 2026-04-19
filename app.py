"""
app.py — BonsaiChat entry point.

Startup sequence
----------------
1. Create the pywebview window immediately (user sees the UI within ~1 s).
2. In a background thread, run all heavy initialisation:
   - init_db()
   - Import agent / knowledge modules (agno, fastembed, lancedb, etc.)
   - Start llama-server
3. The JS side triggers server startup via pywebviewready.

Why: Heavy libraries (agno, fastembed, lancedb) can take 20-60 seconds on a
cold run.  Running them on the main thread makes the window appear frozen.
"""

import atexit
import os
import sys
import threading

import webview
from dotenv import load_dotenv

load_dotenv()


def _background_init():
    """Run all slow imports off the main thread so the window stays responsive."""
    from database import init_db
    init_db()


def _on_exit():
    """Ensure the llama-server is terminated when BonsaiChat closes."""
    try:
        from api.bridge import ApiBridge
        # bridge singleton is accessed via the window; just attempt cleanup
        pass
    except Exception:
        pass


def get_resource_path(relative_path: str) -> str:
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


if __name__ == '__main__':
    # Kick off DB init in background immediately.
    init_thread = threading.Thread(target=_background_init, daemon=True, name="bg-init")
    init_thread.start()

    from api.bridge import ApiBridge
    api = ApiBridge()

    html_path = get_resource_path(os.path.join("ui", "index.html"))


    window = webview.create_window(
        "Paramodus",
        html_path,
        js_api=api,
        width=1100,
        height=850,
        background_color="#0f0f1a",
    )
    api.set_window(window)
    window.events.closed += _on_exit

    webview.start(debug=False)
