"""Pre-bake the image catalog. Run after ANY change to the registry or catalog.

    python scripts/bake.py            # bake all
    python scripts/bake.py --image demigod-data

Modal content-hashes image definitions, so the build cost is paid once. The
point of running this deliberately is that it is paid HERE, by you, and not by
whoever spawns the next DEMI_GOD.
"""

from __future__ import annotations

import argparse
import sys

from demigod.images import CATALOG
from demigod.runner.inside import MODAL_APP_NAME


def main() -> int:
    parser = argparse.ArgumentParser(description="Bake pre-baked images")
    parser.add_argument("--image", default=None, help="Bake only this image")
    args = parser.parse_args()

    import modal

    targets = [i for i in CATALOG if args.image in (None, i.name)]
    if not targets:
        print(f"error: no image named {args.image!r}", file=sys.stderr)
        return 2

    app = modal.App.lookup(MODAL_APP_NAME, create_if_missing=True)

    for image in targets:
        print(f"[bake] {image.name}: tools={sorted(image.tool_keys) or '[]'}")
        # Creating a throwaway sandbox is the cheapest way to force the image to
        # actually build; Image objects are otherwise lazy.
        sb = modal.Sandbox.create("true", app=app, image=image.build(), timeout=600)
        sb.wait()
        sb.terminate()
        print(f"[bake] {image.name}: built")

    print(f"[bake] {len(targets)} image(s) ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
