"""`uv run resolution` -- serve the re:SOLUTION UI and its event stream."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env_file(path: Path) -> list[str]:
    """Read `KEY=VALUE` lines into the environment. Returns the names it set.

    The README already tells you to put GOD-side credentials in `.env`, so the
    server reads it rather than making every caller remember to export first --
    a live run that dies on a missing key three phases in is an expensive way to
    learn you forgot `set -a`.

    Deliberately not python-dotenv: this is twenty lines against a new runtime
    dependency for the one process that serves a browser.

    Two rules worth stating. An existing environment variable always wins, so
    `ANTHROPIC_API_KEY=... uv run resolution` overrides the file rather than
    silently losing to it. And the return value is NAMES ONLY -- the caller
    prints it, and a secret that reaches a log is a secret you have to rotate.
    """

    if not path.is_file():
        return []
    loaded: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key or not value or key in os.environ:
            continue
        os.environ[key] = value
        loaded.append(key)
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description="re:SOLUTION web interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="restart on source changes (development)",
    )
    parser.add_argument(
        "--dist",
        type=Path,
        default=None,
        help="directory holding the built frontend (default: web/dist)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=REPO_ROOT / ".env",
        help="KEY=VALUE credentials to load (default: <repo>/.env)",
    )
    args = parser.parse_args()

    if args.dist:
        os.environ["RESOLUTION_DIST"] = str(args.dist.resolve())

    # flush=True: this is the line that tells you whether live runs will work,
    # and stdout block-buffers the moment it is redirected to a file -- so
    # without it the one message worth reading is the one that goes missing
    # exactly when you are reading logs instead of a terminal.
    loaded = load_env_file(args.env_file)
    if loaded:
        print(
            f"[resolution] loaded {', '.join(sorted(loaded))} from {args.env_file}",
            flush=True,
        )
    elif not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            f"[resolution] no ANTHROPIC_API_KEY (looked in {args.env_file}); "
            f"replay works, live runs are disabled",
            flush=True,
        )

    import uvicorn

    uvicorn.run(
        "resolution.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
        # SSE responses are long-lived by design; the default timeout would
        # tear one down mid-run.
        timeout_keep_alive=300,
    )


if __name__ == "__main__":
    main()
