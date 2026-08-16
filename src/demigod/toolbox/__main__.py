"""`python -m demigod.toolbox` -- the fallback when `/usr/local/bin/toolbox` is
missing (an older image, or a caller running from a source checkout)."""

from __future__ import annotations

from demigod.toolbox.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
