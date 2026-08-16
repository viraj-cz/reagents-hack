"""THE ENTRY POINT. This is the "GOD" for now: whatever calls this with a spec.

    spawn-demigod --spec examples/pandas-demigod.json --run-id demo-1

Deliberately not an orchestrator. It spawns exactly one DEMI_GOD. Decomposition,
fan-out across domains, and synthesis of the results are the GOD's job and are
explicitly out of scope -- when they land, they will call `spawn_demigod()`
below N times and read N manifests. That is the whole integration surface.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from demigod.images import resolve_image
from demigod.layout import RunLayout
from demigod.result import DemiGodResult
from demigod.runner import get_runner
from demigod.spec import DemiGodSpec


def spawn_demigod(
    spec: DemiGodSpec, *, run_id: str, runner_kind: str = "inside"
) -> DemiGodResult:
    """Spawn one DEMI_GOD and return its manifest.

    The function the future GOD calls. Note what it does NOT take: an image, a
    volume, a prompt. All derived from the spec, so a caller cannot construct an
    inconsistent DEMI_GOD.
    """
    return get_runner(runner_kind).run(spec, run_id)


def _preflight(spec: DemiGodSpec, run_id: str) -> None:
    """Resolve everything resolvable before spending money on a sandbox.

    Unknown tool keys and uncovered tool sets both fail here, locally, in
    milliseconds -- rather than as an ImportError inside a live sandbox that has
    already been billed for image pull and startup.
    """
    image = resolve_image(spec.tools)
    layout = RunLayout(run_id=run_id, demigod_name=spec.name)
    print(
        f"[preflight] ok\n"
        f"  demigod : {spec.name}\n"
        f"  domain  : {spec.domain}\n"
        f"  tools   : {spec.tools or '(none)'}\n"
        f"  image   : {image.name} covers {sorted(image.tool_keys) or '[]'}\n"
        f"  volumes : {layout.shared_volume_name} (ro)\n"
        f"            {layout.out_volume_name} -> {layout.out_subpath}/\n"
        f"  limits  : {spec.max_lifetime_s}s wall, {spec.idle_timeout_s}s idle, "
        f"{spec.max_turns} turns, {spec.cpu} cpu, {spec.memory_mb}MiB"
    )


def _warn_on_missing_inputs(spec: DemiGodSpec, shared_args: list[str]) -> None:
    """Warn if the spec names input files that nothing uploaded this run.

    Cheap guard against the most wasteful failure mode: a fully-billed sandbox
    whose agent spends every turn looking for a CSV that was never put there.
    A warning, not an error -- shared/ may legitimately have been seeded by an
    earlier spawn onto the same run_id.
    """
    if not spec.files:
        return
    names = set()
    for p in shared_args:
        path = Path(p)
        if path.is_dir():
            names.update(f.relative_to(path).as_posix() for f in path.rglob("*"))
        else:
            names.add(path.name)
    missing = [f for f in spec.files if f not in names]
    if missing:
        print(
            f"[warn] spec names input files that were not uploaded this run: "
            f"{missing}\n"
            f"       Unless shared/ was already seeded on this run-id, the agent "
            f"will burn billed turns looking for them. Pass --shared.",
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="spawn-demigod",
        description="Spawn a single DEMI_GOD agent in an isolated Modal sandbox.",
    )
    parser.add_argument("--spec", required=True, help="Path to a DemiGodSpec JSON file")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Groups DEMI_GODs onto one volume. Defaults to a fresh id.",
    )
    parser.add_argument(
        "--runner",
        default="inside",
        choices=("inside", "outside"),
        help=(
            "Who drives the agent loop. PROVISIONAL -- see demigod/runner/. "
            "'outside' is not implemented yet."
        ),
    )
    parser.add_argument(
        "--shared",
        nargs="*",
        default=[],
        metavar="PATH",
        help=(
            "Local files/dirs to upload into the run volume's shared/ before "
            "spawning. This is the ONLY way input data reaches a DEMI_GOD -- "
            "shared/ is read-only at every mount. Paths named in the spec's "
            "`files` must land here."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the spec and resolve the image, then stop. No sandbox.",
    )
    args = parser.parse_args()

    spec_path = Path(args.spec)
    if not spec_path.exists():
        print(f"error: no such spec file: {spec_path}", file=sys.stderr)
        return 2

    try:
        spec = DemiGodSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"error: invalid spec {spec_path}:\n{e}", file=sys.stderr)
        return 2

    run_id = args.run_id or f"r{uuid.uuid4().hex[:8]}"

    try:
        _preflight(spec, run_id)
    except Exception as e:
        print(f"error: preflight failed:\n{e}", file=sys.stderr)
        return 2

    if args.dry_run:
        if args.shared:
            print(f"[dry-run] would upload to shared/: {args.shared}")
        print("[dry-run] stopping before sandbox creation")
        return 0

    # Seed shared/ BEFORE the sandbox exists -- it is mounted read-only, so it
    # cannot be populated from inside.
    if args.shared:
        layout = RunLayout(run_id=run_id, demigod_name=spec.name)
        try:
            uploaded = layout.seed_shared([Path(p) for p in args.shared])
        except Exception as e:
            print(f"error: uploading shared/ failed:\n{e}", file=sys.stderr)
            return 2
        print(f"[shared] uploaded {len(uploaded)} file(s): {uploaded}")

    _warn_on_missing_inputs(spec, args.shared)

    result = spawn_demigod(spec, run_id=run_id, runner_kind=args.runner)

    print("\n--- result ---")
    print(json.dumps(result.model_dump(), indent=2))
    layout = RunLayout(run_id, spec.name)
    print(
        f"\nartifacts: volume {layout.out_volume_name} under {layout.out_subpath}/\n"
        f"  uv run modal volume ls {layout.out_volume_name} {layout.out_subpath}"
    )
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
