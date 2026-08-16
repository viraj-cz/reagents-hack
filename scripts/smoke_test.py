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

from demigod.images import BASE, TOOLBOX_SMOKE_TEST, resolve_image
from demigod.registry import REGISTRY, all_keys
from demigod.runner.inside import MODAL_APP_NAME


def check_toolbox_cli(app, failures: list[str]) -> None:
    """The `toolbox` shim resolves, and `demigod.toolbox` imports in the image.

    Not a registry tool, so it needs its own pass -- but the same argument
    applies: an image where `toolbox` is missing produces an agent that reads
    `command not found` as "I have no tools" and stops trying, halfway through a
    billed run. Checked against `demigod-base` because the shim is installed in
    the first layer of every image in the catalog.

    It cannot be checked at BUILD time: the shim has to be written before
    `add_local_python_source`, so during the build there is no `demigod` package
    for `--help` to import.
    """
    import modal

    print(f"[smoke] toolbox CLI: in image {BASE.name} ...")
    sb = modal.Sandbox.create(app=app, image=BASE.build(), timeout=300)
    try:
        proc = sb.exec(*TOOLBOX_SMOKE_TEST, timeout=60)
        out = proc.stdout.read()
        proc.wait()
        if proc.returncode == 0 and "toolbox call" in out:
            print("[smoke] toolbox CLI: PASS")
        else:
            print(
                f"[smoke] toolbox CLI: FAIL (exit {proc.returncode})\n"
                f"{proc.stderr.read()}"
            )
            failures.append(f"toolbox CLI: exit {proc.returncode}")
    finally:
        sb.terminate()


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test registry tools")
    parser.add_argument("--tool", default=None, help="Test only this tool key")
    parser.add_argument(
        "--skip-toolbox",
        action="store_true",
        help="Skip the brokered-tool CLI check (saves one sandbox).",
    )
    args = parser.parse_args()

    if args.tool and args.tool not in REGISTRY:
        print(
            f"error: unknown tool {args.tool!r}. Valid: {all_keys()}",
            file=sys.stderr,
        )
        return 2

    keys = [args.tool] if args.tool else all_keys()

    import modal

    app = modal.App.lookup(MODAL_APP_NAME, create_if_missing=True)
    failures: list[str] = []

    if not args.skip_toolbox and not args.tool:
        check_toolbox_cli(app, failures)

    if not keys:
        print("registry is empty")
        return 1 if failures else 0

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
