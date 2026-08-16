"""Live check of the one thing the offline suite cannot reach: the vision API.

    uv run python scripts/preflight_vision.py

Sibling of `scripts/preflight_toolbox.py`, and it exists for the same reason
that one does: the offline tests pin every refusal, every parse and every piece
of wiring around `vision.read_image`, and none of that proves the request body
is a body the API accepts. Four things can be true with all 412 tests green and
the tool still dead in a live run:

  1. the model id was renamed or retired
  2. the account has no quota (this is what an unfunded key looks like, and it
     returns 429 rather than anything mentioning billing)
  3. $OPENAI_API_KEY is absent from the process that executes the tool
  4. the Responses request/response shape changed under us

Cheap on purpose: one small synthetic image, one call, a few hundred tokens.

IT ASSERTS ON WHAT THE MODEL SAW, not on HTTP 200. A 200 carrying a hedge
("I cannot determine the quadrant") is a failure -- it means the image did not
arrive in a form the model could read, which is exactly the bug a status-code
check waves through. So the image has one unambiguous feature and the check is
whether the answer names it.

Run it after a model bump, after rotating the key, and before the first
image-domain run of the day.
"""

from __future__ import annotations

import asyncio
import base64
import os
import struct
import sys
import zlib
from pathlib import Path

from reagents.tools import vision

REPO_ROOT = Path(__file__).resolve().parent.parent

SIZE = 240
QUADRANT = "bottom-left"
"""Where the black square goes. Also the answer, so the assertion is on the
image's content rather than on the shape of the reply."""

QUESTION = (
    "This image is white with one solid black square in exactly one quadrant. "
    "Which quadrant is it in? Answer with exactly one of: top-left, top-right, "
    "bottom-left, bottom-right."
)


def load_env() -> None:
    """Load .env, the same way `scripts/e2e_live.py` does.

    An exported variable only exists in the shell that exported it. A gitignored
    .env makes the run repeatable and delegatable; an existing environment
    variable still wins.
    """
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def probe_png() -> bytes:
    """An 8-bit greyscale PNG, written with zlib and struct.

    Hand-rolled rather than pulled from Pillow or matplotlib because neither is
    a dependency of this repo, and a preflight that needs `uv sync --extra
    something` to run is a preflight nobody runs.
    """
    rows = bytearray()
    for y in range(SIZE):
        rows.append(0)  # PNG per-row filter type: none
        for x in range(SIZE):
            in_square = (SIZE // 2 < y < SIZE - 20) and (20 < x < SIZE // 2)
            rows.append(0 if in_square else 255)

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return (
            struct.pack(">I", len(payload))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", SIZE, SIZE, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(rows), 9))
        + chunk(b"IEND", b"")
    )


def main() -> int:
    load_env()

    key = os.environ.get(vision.API_KEY_ENV, "")
    if not key:
        print(
            f"error: ${vision.API_KEY_ENV} is not set. Put it in .env as\n"
            f"  {vision.API_KEY_ENV}=sk-...\n"
            f"A bare key with no variable name on the line is silently skipped "
            f"by every loader that reads that file.",
            file=sys.stderr,
        )
        return 2
    print(f"[preflight] key ...{key[-6:]}, model {vision.DEFAULT_MODEL}")

    png = probe_png()
    print(f"[preflight] probe image {len(png)} bytes, square in the {QUADRANT}")

    try:
        result = asyncio.run(
            vision.read_image(
                {
                    "question": QUESTION,
                    "image_base64": base64.b64encode(png).decode(),
                    "media_type": "image/png",
                    "detail": "low",
                }
            )
        )
    except Exception as exc:
        print(f"[preflight] FAILED: {exc}", file=sys.stderr)
        return 1

    reading = result["reading"]
    print(f"[preflight] model    {result['model']}")
    print(f"[preflight] usage    {result['usage']}")
    print(f"[preflight] reading  {reading.strip()[:200]}")

    if result["truncated"]:
        print("[preflight] FAILED: answer was truncated", file=sys.stderr)
        return 1
    # Accept "bottom-left" and "bottom left"; reject anything that names a
    # different quadrant or declines to name one.
    normalized = reading.lower().replace(" ", "-")
    if QUADRANT not in normalized:
        print(
            f"[preflight] FAILED: expected {QUADRANT!r} in the reading. The "
            f"call succeeded but the model did not read the image.",
            file=sys.stderr,
        )
        return 1

    print("[preflight] ok -- the vision path is live end to end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
