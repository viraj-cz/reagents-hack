"""`god` -- launch a GOD run, then check in on it from anywhere.

    uv run god launch --problem simple --domains 2 --turns 12
    uv run god status <run_id>
    uv run god watch  <run_id>
    uv run god list
    uv run god logs   <run_id>
    uv run god followup <run_id> "also consider the outlet"
    uv run god terminate <run_id>

`launch` returns in seconds and the run continues without it. Everything else
is a read against the run's `modal.Dict`, which means it works from a different
machine, after a reboot, and after GOD's own sandbox has terminated -- the Dict
outlives the container that wrote it.

`status` deliberately does NOT read the volume. Volume writes are invisible
outside the writing container until GOD commits at the end, so polling `out/`
mid-run shows an empty directory and reads as "the agents produced nothing".
The Dict is the live channel; the volume is the archive.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from godbox.layout import GodRequest
from godbox.status import (
    STALE_AFTER_S,
    STATUS_SCHEMA,
    GodStatus,
    list_runs,
    push_followup,
    read_status,
)

WATCH_INTERVAL_S = 5.0


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m{seconds % 60:02d}s"


def _liveness(status: GodStatus) -> str:
    if status.is_terminal:
        return "finished"
    age = status.heartbeat_age_s
    if age is None:
        return "no heartbeat yet"
    if status.is_stale:
        return (
            f"STALE (no heartbeat for {_fmt_duration(age)}, "
            f"limit {int(STALE_AFTER_S)}s)"
        )
    return f"alive ({_fmt_duration(age)} since heartbeat)"


def _render(status: GodStatus, *, events: int = 8) -> str:
    lines = [
        f"run      : {status.run_id}",
        f"phase    : {status.phase}",
        f"liveness : {_liveness(status)}",
        f"elapsed  : {_fmt_duration(status.elapsed_s)}",
    ]
    if status.schema > STATUS_SCHEMA:
        lines.append(
            f"WARNING  : status schema {status.schema} is newer than this "
            f"client understands ({STATUS_SCHEMA}); fields may be missing"
        )
    if status.sandbox_id:
        lines.append(f"sandbox  : {status.sandbox_id}")
    volume = status.raw.get("artifact_volume")
    if volume:
        lines.append(f"artifacts: {volume} -> {status.raw.get('artifact_path')}")

    if status.domains:
        lines.append(f"domains  : {', '.join(status.domains)}")

    demigods = status.demigods
    if demigods:
        lines.append("demigods :")
        for name, row in sorted(demigods.items()):
            started = row.get("started_at")
            finished = row.get("finished_at")
            took = _fmt_duration(
                (finished - started) if (started and finished) else None
            )
            detail = f"conf={row.get('confidence')}" if row.get("confidence") else ""
            if row.get("error"):
                detail = f"error={str(row['error'])[:60]}"
            lines.append(
                f"  {row.get('status', '?'):<8} {name:<28} {took:>7}  {detail}"
            )

    pending = [f for f in status.followups if not f.get("delivered")]
    if status.followups:
        lines.append(
            f"followups: {len(status.followups)} sent, {len(pending)} not yet seen"
        )

    log = status.events[-events:]
    if log:
        lines.append(f"events   : (last {len(log)})")
        start = status.started_at or 0
        for event in log:
            offset = _fmt_duration((event.get("at") or start) - start)
            lines.append(f"  +{offset:>7}  {event.get('message', '')}")

    if status.error:
        lines.append(f"error    : {status.error}")
    if status.solution:
        lines.append("solution :")
        lines.append(json.dumps(status.solution, indent=2))
    return "\n".join(lines)


def _read_or_explain(run_id: str) -> GodStatus | None:
    try:
        return read_status(run_id)
    except Exception as exc:
        print(
            f"error: no status for run {run_id!r} ({type(exc).__name__}).\n"
            f"A run's Dict is created at launch, so this usually means the id "
            f"is wrong or the run was launched in a different Modal "
            f"environment. `uv run god list` shows what is there.",
            file=sys.stderr,
        )
        return None


# --- commands ----------------------------------------------------------------


def cmd_launch(args: argparse.Namespace) -> int:
    from godbox.launch import launch_god, new_run_id

    if args.problem_file:
        from pathlib import Path

        from reagents.contracts import NativeProblem

        problem = NativeProblem.model_validate_json(
            Path(args.problem_file).read_text(encoding="utf-8")
        )
    else:
        if args.problem == "flareguard":
            from benchmarks.flareguard import load_problem

            problem = load_problem()
        else:
            from reagents.toy import simple_problem, toy_problem

            problem = simple_problem() if args.problem == "simple" else toy_problem()

    request = GodRequest(
        run_id=args.run_id or new_run_id(),
        problem=problem,
        domain_count=args.domains,
        max_turns=args.turns,
        model=args.model,
        approved_write_tools=args.approve_write,
        approved_high_risk_tools=args.approve_high_risk,
        use_broker=args.broker,
        verifier_id=("flareguard-public-v1" if args.problem == "flareguard" else None),
        keep_alive_s=args.keep_alive,
    )
    handle = launch_god(
        request,
        max_lifetime_s=args.max_lifetime,
        idle_timeout_s=args.idle_timeout,
    )
    print()
    print(handle.describe())
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    status = _read_or_explain(args.run_id)
    if status is None:
        return 2
    if args.json:
        print(json.dumps(status.raw, indent=2, default=str))
    else:
        print(_render(status, events=args.events))
    return 0 if status.phase != "failed" else 1


def cmd_watch(args: argparse.Namespace) -> int:
    """Poll until the run reaches a terminal phase.

    A dumb poll rather than a subscription, because that is what the Dict
    supports and because it survives the connection dropping -- which a
    subscription would not, and which is the failure this whole design is
    about.
    """
    seen = 0
    while True:
        status = _read_or_explain(args.run_id)
        if status is None:
            return 2
        events = status.events
        for event in events[seen:]:
            offset = _fmt_duration((event.get("at") or 0) - (status.started_at or 0))
            phase = event.get("phase", "?")
            print(f"+{offset:>7}  [{phase}] {event.get('message', '')}")
        seen = len(events)
        if status.is_terminal:
            print()
            print(_render(status, events=0))
            return 0 if status.phase == "done" else 1
        if status.is_stale:
            print(f"  ... {_liveness(status)}")
        time.sleep(args.interval)


def cmd_list(args: argparse.Namespace) -> int:
    runs = list_runs(limit=args.limit)
    if not runs:
        print("no GOD runs found in this Modal environment")
        return 0
    print(f"{'run':<16} {'phase':<13} {'elapsed':>8}  liveness")
    for run_id in runs:
        try:
            status = read_status(run_id)
        except Exception:
            continue
        print(
            f"{run_id:<16} {status.phase:<13} "
            f"{_fmt_duration(status.elapsed_s):>8}  {_liveness(status)}"
        )
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    from godbox.launch import read_log

    status = _read_or_explain(args.run_id)
    if status is None:
        return 2
    try:
        text = read_log(args.run_id, status.sandbox_id)
    except Exception as exc:
        print(
            f"error: could not read the log ({type(exc).__name__}: {exc}).\n"
            f"GOD's log lives inside its sandbox and dies with it -- a finished "
            f"run has no log, by design. The durable record is "
            f"`god status {args.run_id}` and trace.json on volume "
            f"{status.raw.get('artifact_volume')}.",
            file=sys.stderr,
        )
        return 2
    lines = text.splitlines()
    for line in lines[-args.tail :] if args.tail else lines:
        print(line)
    return 0


def cmd_followup(args: argparse.Namespace) -> int:
    depth = push_followup(args.run_id, args.message)
    print(
        f"queued (inbox depth {depth}). GOD drains it at the next phase "
        f"boundary and records it in `god status {args.run_id}`."
    )
    return 0


def cmd_terminate(args: argparse.Namespace) -> int:
    import modal

    status = _read_or_explain(args.run_id)
    if status is None:
        return 2
    if not status.sandbox_id:
        print("no sandbox id recorded; nothing to terminate", file=sys.stderr)
        return 1
    try:
        modal.Sandbox.from_id(status.sandbox_id).terminate()
    except Exception as exc:
        print(f"terminate failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"terminated {status.sandbox_id}")
    print(
        "Note: DEMI_GOD sandboxes this run spawned are killed by their own "
        "idle_timeout, not by this -- GOD's `finally` cannot run once its "
        "container is gone."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="god", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    launch = sub.add_parser("launch", help="Start a GOD run and detach")
    launch.add_argument(
        "--problem",
        choices=("simple", "pathway", "flareguard"),
        default="simple",
        help="simple = 5-entity valve pipeline (cheap, for exercising the "
        "pipeline); pathway = the 9-entity glycolysis problem; flareguard = "
        "the complete-objective living-diagnostic design benchmark",
    )
    launch.add_argument(
        "--problem-file", default=None, help="Path to a NativeProblem JSON file"
    )
    launch.add_argument("--domains", type=int, default=2)
    launch.add_argument("--model", default="claude-opus-4-8")
    launch.add_argument(
        "--turns",
        type=int,
        default=12,
        help="Per-demigod turn cap. THE cost lever: every turn is an Anthropic call.",
    )
    launch.add_argument("--run-id", default=None)
    launch.add_argument("--max-lifetime", type=int, default=3600)
    launch.add_argument("--idle-timeout", type=int, default=600)
    launch.add_argument(
        "--keep-alive",
        type=int,
        default=0,
        help="Seconds GOD stays up after finishing, for poking at it. Billed.",
    )
    launch.add_argument("--approve-write", nargs="*", default=[], metavar="TOOL_ID")
    launch.add_argument("--approve-high-risk", nargs="*", default=[], metavar="TOOL_ID")
    launch.add_argument(
        "--broker",
        action="store_true",
        help="Publish scoped leases to the deployed TOOLBOX_BROKER. Off by default.",
    )
    launch.set_defaults(func=cmd_launch)

    status = sub.add_parser("status", help="Print one snapshot of a run")
    status.add_argument("run_id")
    status.add_argument("--json", action="store_true")
    status.add_argument("--events", type=int, default=8)
    status.set_defaults(func=cmd_status)

    watch = sub.add_parser("watch", help="Follow a run until it finishes")
    watch.add_argument("run_id")
    watch.add_argument("--interval", type=float, default=WATCH_INTERVAL_S)
    watch.set_defaults(func=cmd_watch)

    listing = sub.add_parser("list", help="Every run with a status Dict")
    listing.add_argument("--limit", type=int, default=100)
    listing.set_defaults(func=cmd_list)

    logs = sub.add_parser("logs", help="GOD's stdout, read out of its sandbox")
    logs.add_argument("run_id")
    logs.add_argument("--tail", type=int, default=80)
    logs.set_defaults(func=cmd_logs)

    followup = sub.add_parser("followup", help="Send a message to a running GOD")
    followup.add_argument("run_id")
    followup.add_argument("message")
    followup.set_defaults(func=cmd_followup)

    terminate = sub.add_parser("terminate", help="Kill a GOD sandbox now")
    terminate.add_argument("run_id")
    terminate.set_defaults(func=cmd_terminate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler: Any = args.func
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
