"""The ASGI entrypoint uvicorn imports.

Separate from `app.py` so `--reload` has a stable import string to re-import,
and so constructing the app (which creates a `RunStore`, and with it the memory
holding every live run) happens exactly once per process.
"""

from __future__ import annotations

import os
from pathlib import Path

from resolution.app import ResolutionApp

_dist = os.environ.get("RESOLUTION_DIST")
app = ResolutionApp(dist=Path(_dist) if _dist else None)

__all__ = ["app"]
