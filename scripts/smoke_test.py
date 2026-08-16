"""Run every registry tool's smoke test in the image that claims to provide it.

    python scripts/smoke_test.py              # all tools
    python scripts/smoke_test.py --tool pandas

This is the check that a registry entry is not a lie. An entry claiming
`pandas` in an image where the pip install silently resolved to something else
fails here, on your machine, instead of 20 minutes into a live DEMI_GOD run.

Required by `.claude/skills/add-tool-to-registry/SKILL.md` step 5. Do not add a
tool without a passing smoke test.
"""

from __future__ import annotations

import argparse
import sys

from demigod.images import resolve_image
from demigod.registry import REGISTRY, all_keys
from demigod.runner.inside import MODAL_APP_NAME


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test registry tools")
    parser.add_argument("--tool", default=None, help="Test only this tool key")
    args = parser.parse_args()

    if args.tool and args.tool not in REGISTRY:
        print(
            f"error: unknown tool {args.tool!r}. Valid: {all_keys()}",
            file=sys.stderr,
        )
        return 2

    keys = [args.tool] if args.tool else all_keys()
    if not keys:
        print("registry is empty")
        return 0

    import modal

    app = modal.App.lookup(MODAL_APP_NAME, create_if_missing=True)
    failures: list[str] = []

    for key in keys:
        entry = REGISTRY[key]
        if not entry.smoke_test:
            print(f"[smoke] {key}: SKIP (no smoke_test declared)")
            failures.append(f"{key}: no smoke_test declared")
            continue

        image = resolve_image([key])
        print(f"[smoke] {key}: in image {image.name} ...")

        sb = modal.Sandbox.create(app=app, image=image.build(), timeout=300)
        try:
            proc = sb.exec(*entry.smoke_test, timeout=120)
            out = proc.stdout.read()
            proc.wait()
            if proc.returncode == 0:
                print(f"[smoke] {key}: PASS {out.strip()}")
            else:
                err = proc.stderr.read()
                print(f"[smoke] {key}: FAIL (exit {proc.returncode})\n{err}")
                failures.append(f"{key}: exit {proc.returncode}")
        finally:
            sb.terminate()

    # A tool that also needs credentials cannot be fully proven by an offline
    # smoke test. Say so rather than implying more coverage than exists.
    for key in keys:
        if REGISTRY[key].secrets:
            print(
                f"[smoke] note: {key} declares secrets "
                f"{list(REGISTRY[key].secrets)}; the smoke test does not verify "
                "they are set in Modal."
            )

    if failures:
        print(f"\n{len(failures)} failure(s): {failures}", file=sys.stderr)
        return 1
    print(f"\nall {len(keys)} tool(s) passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
