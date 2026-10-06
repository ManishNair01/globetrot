"""One place to switch logging on for the concierge and the API.

LOG_LEVEL in .env (default INFO) sets the level. Logs go to stderr, so they show in the terminal or in
the preview server's log. Tool calls are logged by tools.run_tool; never log keys or full user text.
"""
from __future__ import annotations

import logging
import os

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure() -> None:
    """Idempotent: safe to call from every Streamlit rerun and from the API startup."""
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").strip().upper(), logging.INFO)
    root = logging.getLogger()
    if not getattr(root, "_super_travel_configured", False):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
        root._super_travel_configured = True  # type: ignore[attr-defined]
    root.setLevel(level)
