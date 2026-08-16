"""Regenerate `web/src/components/PixelCreation.tsx` from the fresco.

The landing page's artwork is Michelangelo's *Creation of Adam* (1512, public
domain) reduced to three states -- solid, 50% dither, nothing -- to match the
1-bit poster the interface is themed after. The bitmap is committed as source so
the build needs no image pipeline; this script exists so the reduction is
reproducible and its parameters are arguable rather than mysterious.

    uv run python scripts/pixelate_fresco.py --download
    uv run python scripts/pixelate_fresco.py --source /tmp/adam_hands.jpg --width 140

Requires Pillow and numpy, which are NOT project dependencies -- this runs once
when someone wants different art, never during a build:

    uv run --with pillow --with numpy python scripts/pixelate_fresco.py --download

WHY SATURATION AND NOT BRIGHTNESS. The obvious reduction -- threshold the
luminance -- fails on this image, and fails misleadingly. The plaster behind the
hands is *brighter* than the skin (measured: V~0.95 against V~0.60-0.79), so a
bright-is-ink threshold renders the wall and drops the subject, while an
inverted one turns every crack in the plaster into an edge. Saturation
separates them cleanly (S~0.14 for plaster, S~0.25-0.57 for skin); luminance is
then only asked the easier question of lit-or-shaded *within* the hands.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# Wikimedia Commons, "Michelangelo - Creation of Adam (hand crop).jpg".
# Public domain: the work is from 1512, and a faithful photographic
# reproduction of a public-domain two-dimensional work carries no new copyright.
SOURCE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/1/1e/"
    "Michelangelo_-_Creation_of_Adam_%28hand_crop%29.jpg"
)
USER_AGENT = "reagents-resolution-ui/0.1 (fresco pixelation; one-off)"

COMPONENT = Path(__file__).resolve().parents[1] / "web/src/components/PixelCreation.tsx"

# Defaults chosen by eye against the rendered page, not by metric.
DEFAULT_CROP = (35, 95, 500, 285)  # the two hands, without the frame's edges
DEFAULT_WIDTH = 140  # cells; below ~100 the fingers stop separating
DEFAULT_SATURATION = 0.27  # skin vs plaster
DEFAULT_LIT = 0.42  # within the hands: solid vs dither
UNIT = 6  # SVG units per cell


def reduce_image(
    source: Path,
    *,
    width: int,
    crop: tuple[int, int, int, int],
    saturation: float,
    lit: float,
) -> list[str]:
    import numpy as np
    from PIL import Image, ImageFilter

    image = Image.open(source).convert("RGB").crop(crop)
    # Median first: the plaster is covered in fine cracks that survive
    # downsampling as speckle and read as noise rather than texture.
    image = image.filter(ImageFilter.MedianFilter(5))
    height = max(1, round(width * image.height / image.width))
    image = image.resize((width, height), Image.LANCZOS)

    pixels = np.asarray(image).astype(np.float32) / 255
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    value = pixels.max(axis=-1)
    spread = value - pixels.min(axis=-1)
    sat = np.where(value > 0, spread / np.maximum(value, 1e-6), 0)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue

    mask = sat >= saturation
    inside = luminance[mask]
    if inside.size == 0:
        raise SystemExit(f"saturation {saturation} selected no pixels; lower it")
    # Normalise within the mask, so "lit" means lit relative to the hand rather
    # than relative to the wall behind it.
    low, high = np.percentile(inside, 12), np.percentile(inside, 88)
    normalised = np.clip((luminance - low) / max(high - low, 1e-6), 0, 1)

    rows = [
        "".join(
            ("#" if normalised[y, x] >= lit else ":") if mask[y, x] else "."
            for x in range(width)
        )
        for y in range(height)
    ]
    return _trim(rows)


def _trim(rows: list[str]) -> list[str]:
    """Drop empty rows and columns so the art is flush with its own box."""

    rows = [row for row in rows if row.strip(".")]
    if not rows:
        raise SystemExit("the reduction produced nothing; loosen the thresholds")
    columns = [x for x in range(len(rows[0])) if any(row[x] != "." for row in rows)]
    return [row[min(columns) : max(columns) + 1] for row in rows]


def render_component(rows: list[str]) -> str:
    body = "\n".join(f"  '{row}'," for row in rows)
    return f"""/**
 * The Creation of Adam, at 1 bit.
 *
 * Michelangelo, 1512, public domain, by way of Wikimedia Commons -- reduced
 * to a {len(rows[0])}x{len(rows)} grid of three states: solid, 50% dither, and nothing.
 * The bitmap is committed rather than generated at build time; to change it, run
 * `scripts/pixelate_fresco.py`, which documents why the reduction segments on
 * saturation instead of brightness.
 *
 * `#` solid, `:` dithered, `.` background.
 */

const ART = [
{body}
] as const

const UNIT = {UNIT}
const HALF = UNIT / 2
const W = {len(rows[0])}
const H = {len(rows)}

type Cell = {{ x: number; y: number; solid: boolean }}

const CELLS: Cell[] = ART.flatMap((row, y) =>
  [...row].flatMap((ch, x) => (ch === '.' ? [] : [{{ x, y, solid: ch === '#' }}])),
)

export function PixelCreation({{ scale = 1 }}: {{ scale?: number }}) {{
  return (
    <svg
      width={{W * UNIT * scale}}
      height={{H * UNIT * scale}}
      viewBox={{`0 0 ${{W * UNIT}} ${{H * UNIT}}`}}
      role="img"
      aria-label="The Creation of Adam, as dithered pixel art"
      shapeRendering="crispEdges"
    >
      <defs>
        <pattern
          id="dither-creation"
          width={{UNIT}}
          height={{UNIT}}
          patternUnits="userSpaceOnUse"
        >
          <rect width={{HALF}} height={{HALF}} fill="#fff" />
          <rect x={{HALF}} y={{HALF}} width={{HALF}} height={{HALF}} fill="#fff" />
        </pattern>
      </defs>
      {{CELLS.map(({{ x, y, solid }}) => (
        <rect
          key={{`${{x}}:${{y}}`}}
          x={{x * UNIT}}
          y={{y * UNIT}}
          width={{UNIT}}
          height={{UNIT}}
          fill={{solid ? '#fff' : 'url(#dither-creation)'}}
        />
      ))}}
    </svg>
  )
}}
"""


def download(target: Path) -> Path:
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        target.write_bytes(response.read())
    print(f"downloaded {target} ({target.stat().st_size} bytes) from {SOURCE_URL}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="local copy of the fresco detail")
    parser.add_argument(
        "--download",
        action="store_true",
        help=f"fetch the public-domain source from {SOURCE_URL}",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--saturation", type=float, default=DEFAULT_SATURATION)
    parser.add_argument("--lit", type=float, default=DEFAULT_LIT)
    parser.add_argument(
        "--crop",
        default=",".join(str(v) for v in DEFAULT_CROP),
        help="left,top,right,bottom in source pixels",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="preview",
        help="write the bitmap to stdout instead of the component",
    )
    args = parser.parse_args()

    source = args.source
    if args.download:
        source = download(Path(args.source or "adam_hands.jpg"))
    if source is None or not source.exists():
        raise SystemExit("pass --source PATH, or --download to fetch it")

    crop = tuple(int(v) for v in args.crop.split(","))
    if len(crop) != 4:
        raise SystemExit("--crop takes left,top,right,bottom")

    rows = reduce_image(
        source,
        width=args.width,
        crop=crop,  # type: ignore[arg-type]
        saturation=args.saturation,
        lit=args.lit,
    )
    if args.preview:
        print("\n".join(rows))
        return

    COMPONENT.write_text(render_component(rows), encoding="utf-8")
    ink = sum(1 for row in rows for cell in row if cell != ".")
    print(
        f"wrote {COMPONENT.relative_to(COMPONENT.parents[3])}: "
        f"{len(rows[0])}x{len(rows)} grid, {ink} cells",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
