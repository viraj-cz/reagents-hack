"""INSIDE GOD's sandbox: drive the loop, report, self-destruct.

    python -u -m godbox.entrypoint --request /god/request.json

Never run by hand from a laptop -- `godbox/launch.py` execs this, and it
assumes the environment that launcher builds: `ANTHROPIC_API_KEY` and
`MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` from mounted Secrets, a request file, and
a status Dict already primed.

The analogue of `demigod/entrypoint.py`, one level up. That one runs an agent
loop and writes `result.json`; this one runs the God loop and writes
`solution.json` -- and, in between, spawns the sandboxes that produce the
results it integrates.

TWO OUTPUT CHANNELS, and the split is not cosmetic:

    modal.Dict    every phase transition, live. Readable from a laptop the
                  instant it is written, which the volume is not.
    out volume    solution.json and trace.json, uploaded once at the end.
                  Big, durable, outlives everything.

The solution goes to BOTH. Duplication is cheap and the failure modes are
disjoint: an upload can fail, and a Dict value has a size limit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from godbox.layout import (
    GOD_OUT_SUBDIR,
    REQUEST_PATH,
    SOLUTION_FILENAME,
    STAGING_DIR,
    TRACE_FILENAME,
    GodRequest,
    out_volume_name,
)
from godbox.status import Phase, StatusWriter
from godbox.trace_channel import QueueTracer

MAX_SOLUTION_CHARS = 60_000
"""Cap on what goes into the Dict copy of the solution. `Dict.put` raises
RequestSizeError on an oversized value, and losing the status channel to a
verbose integrator would hide the very run it was reporting. The volume copy is
never truncated."""


def _instrument(god: Any, status: StatusWriter) -> None:
    """Wrap God's four phases so the run reports where it is.

    The orchestrator has no logging of its own -- deliberately, it is a
    library. Rather than thread a callback through it, the phase objects are
    decorated in place. This is the same trick `scripts/e2e_live.py` uses, with
    one difference that matters: every wrapper takes `*args, **kwargs` and
    forwards them untouched.

    `e2e_live.py` pinned its integrate wrapper to the three positional
    arguments the integrator took at the time, and the orchestrator has since
    grown a `failed_domains=` keyword -- so that wrapper now raises TypeError
    at the last phase of a fully-paid-for run. Observability code must not be
    able to break the thing it observes.
    """
    planner_plan = god.planner.plan
    transformer_forward = god.transformer.forward
    integrator_integrate = god.integrator.integrate
    runtime_run = god.runtime.run

    async def plan(*args: Any, **kwargs: Any) -> Any:
        await status.set_phase(
            Phase.PLANNING,
            "inventing complete alternative representations",
        )
        specs = await planner_plan(*args, **kwargs)
        await status.set_domains([s.name for s in specs])
        for spec in specs:
            axes = ", ".join(a.value for a in spec.axes)
            await status.note(f"planned {spec.name!r} [{axes}] tools={spec.tool_ids}")
        await status.drain_followups()
        return specs

    async def forward(problem: Any, spec: Any, *args: Any, **kwargs: Any) -> Any:
        await status.set_phase(
            Phase.TRANSFORMING, f"projecting the problem into {spec.name!r}"
        )
        result = await transformer_forward(problem, spec, *args, **kwargs)
        # NOT "sealed": assert_sealed and find_spec_leaks run in the
        # orchestrator, outside this hook. Claiming a seal here would print a
        # reassuring lie about an envelope that is rejected moments later.
        await status.note(f"transformed {spec.name!r} (seal check pending)")
        return result

    async def run(envelope: Any, *args: Any, **kwargs: Any) -> Any:
        name = envelope.domain.name
        await status.set_phase(Phase.SPAWNING, f"spawning DEMI_GOD for {name!r}")
        await status.demigod(name, "running")
        result = await runtime_run(envelope, *args, **kwargs)
        await status.demigod(
            name,
            "ok" if result.status == "ok" else result.status,
            confidence=result.confidence,
            files=len(result.files),
            error=result.error,
        )
        await status.note(f"demigod {name!r} finished status={result.status}")
        await status.drain_followups()
        return result

    async def integrate(*args: Any, **kwargs: Any) -> Any:
        await status.set_phase(
            Phase.INTEGRATING, "comparing complete candidate artifacts"
        )
        return await integrator_integrate(*args, **kwargs)

    god.planner.plan = plan
    god.transformer.forward = forward
    god.integrator.integrate = integrate
    god.runtime.run = run


def _persist(run_id: str, solution: Any, trace: Any) -> str | None:
    """Stage solution.json and trace.json locally, then UPLOAD them.

    Upload, not a mounted write plus a commit. `Volume.commit()` raises
    `RuntimeError: commit() can only be called on a mounted volume inside a
    container` in a Sandbox -- it is a Modal *Function* API -- and a Sandbox's
    mount writes are otherwise flushed only when the sandbox terminates. Since
    GOD terminates itself moments later, that would technically work, and would
    also mean the answer is invisible right up until the process that produced
    it is gone. `batch_upload` is visible immediately. Measured both ways in
    `scripts/preflight_god.py`.

    Best-effort by design: the status Dict already carries the solution, so a
    failed upload costs the full trace, not the answer. Raising here would turn
    a storage hiccup into a failed run.
    """
    staging = Path(STAGING_DIR)
    solution_file = staging / SOLUTION_FILENAME
    trace_file = staging / TRACE_FILENAME
    try:
        staging.mkdir(parents=True, exist_ok=True)
        solution_file.write_text(
            json.dumps(solution.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
        trace_file.write_text(
            json.dumps(trace.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
    except Exception as exc:
        return f"could not stage artifacts in {staging}: {type(exc).__name__}: {exc}"

    try:
        import modal

        volume = modal.Volume.from_name(out_volume_name(run_id), create_if_missing=True)
        # force=True so a re-run of the same run_id overwrites rather than
        # failing halfway and leaving one of the two files stale.
        with volume.batch_upload(force=True) as batch:
            batch.put_file(solution_file, f"/{GOD_OUT_SUBDIR}/{SOLUTION_FILENAME}")
            batch.put_file(trace_file, f"/{GOD_OUT_SUBDIR}/{TRACE_FILENAME}")
    except Exception as exc:
        return (
            f"artifacts staged in {staging} but the upload to "
            f"{out_volume_name(run_id)} failed ({type(exc).__name__}: {exc}); "
            f"the solution is still in the status Dict"
        )
    return None


async def _solve(
    request: GodRequest, status: StatusWriter
) -> tuple[dict[str, Any], str, int]:
    """Run the loop and return (solution payload, summary, exit code).

    Deliberately does NOT write the terminal status itself. `StatusWriter`'s
    terminal methods are the SYNCHRONOUS ones, and calling a blocking Modal
    interface from inside a running event loop earns an `AsyncUsageWarning` and
    stalls the loop for the RPC. Returning the payload and letting `main()`
    write it keeps every terminal write on the sync side of `asyncio.run`,
    where the docstring on `complete()` already claims it is.
    """
    from reagents.contracts import Budget
    from reagents.demigod.sandbox_runtime import SandboxDemigodRuntime
    from reagents.god.orchestrator import God
    from reagents.llm.client import make_llm
    from reagents.llm.streaming import StreamingAnthropicLLM

    # The orchestrator's own event stream, shipped out of the sandbox on a
    # queue. `_instrument` below reports PHASES into the status Dict, which is
    # the right channel for state; this is the finer-grained trace -- tool
    # calls, per-demigod lanes, artifacts -- that a Dict capped at 200 entries
    # cannot hold. Best-effort: `QueueTracer` swallows its own failures, so a
    # trace channel that breaks cannot fail a paid-for run.
    tracer: Any = None
    try:
        tracer = QueueTracer(request.run_id)
    except Exception as exc:
        print(f"[trace] no live trace channel: {exc}", flush=True)

    toolbox = None
    if request.use_broker:
        from broker.session import modal_session

        toolbox = modal_session()

    verifier = None
    if request.verifier_id == "flareguard-public-v1":
        from benchmarks.flareguard import FlareGuardVerifier

        verifier = FlareGuardVerifier()
    elif request.verifier_id is not None:
        raise ValueError(f"unknown native verifier {request.verifier_id!r}")

    # Streamed when there is a channel to stream to. Without this the sandbox
    # used the non-streaming client and the whole planning phase -- 57 seconds
    # in the run that prompted this -- arrived as nothing at all between two
    # phase lines, which reads as a hang. `make_llm()` remains the fallback so
    # a run without a trace channel, or without a key, behaves as it always did.
    llm: Any = make_llm(request.model)
    if tracer is not None and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            llm = StreamingAnthropicLLM(tracer, request.model)
        except Exception as exc:
            print(
                f"[trace] falling back to the non-streaming client: {exc}",
                flush=True,
            )

    god = God(
        llm,
        domain_count=request.domain_count,
        # Operator approval arrives in the request and cannot be widened from
        # in here. An empty set means no write tool spawns.
        approved_write_tools=set(request.approved_write_tools),
        # THE SEAM, unchanged. GOD does not know it is itself in a sandbox;
        # this is the same runtime `scripts/e2e_live.py` passes from a laptop.
        runtime=SandboxDemigodRuntime(
            run_id=request.run_id,
            max_turns=request.max_turns,
            toolbox=toolbox,
            require_toolbox=request.use_broker,
            restrict_egress=request.use_broker,
            agent_model=request.model,
        ),
        verifier=verifier,
        budget=Budget(
            max_tokens=4096,
            max_steps=request.max_turns,
            wall_time_s=min(request.max_turns * 75, 1800),
            max_tool_calls=24,
        ),
        tracer=tracer,
    )
    _instrument(god, status)

    try:
        solution = await god.solve(request.problem)
    finally:
        # Flush before the sandbox goes away. Whatever the run did, the last
        # events are the ones a watcher most wants and the ones most likely to
        # be sitting in the buffer.
        if tracer is not None:
            tracer.close()
    trace = god.last_trace

    warning = _persist(request.run_id, solution, trace)
    if warning:
        await status.note(f"WARNING: {warning}")

    payload = solution.model_dump(mode="json")
    encoded = json.dumps(payload)
    if len(encoded) > MAX_SOLUTION_CHARS:
        payload = {
            "problem_id": solution.problem_id,
            "answer": solution.answer[:MAX_SOLUTION_CHARS],
            "confidence": solution.confidence,
            "truncated": True,
            "note": f"full solution on volume {out_volume_name(request.run_id)}",
        }

    summary = (
        f"complete: {len(trace.artifacts)} artifact(s), "
        f"{len(trace.failures)} failure(s), {len(trace.leaks)} leak(s)"
    )
    # Exit code mirrors `scripts/e2e_live.py`: no artifact means no usable
    # answer, whatever the integrator wrote.
    return payload, summary, 0 if trace.artifacts else 1


def _self_terminate(sandbox_id: str, keep_alive_s: int) -> None:
    """Kill our own sandbox. The LAST thing this process does.

    A GOD sandbox is long-lived on purpose, which means nothing outside it is
    holding a `finally` that would clean it up -- the launcher returned
    minutes or hours ago. So teardown has to come from in here, and it has to
    come after the status Dict and the volume are written, because the
    terminate RPC kills this container mid-call and nothing after it runs.

    `idle_timeout` on the sandbox is the backstop for the case where this never
    executes at all (SIGKILL, OOM). Between the two, a leaked GOD bills for
    minutes rather than the full `max_lifetime_s`.
    """
    if not sandbox_id:
        print(
            "[god] no sandbox id in the request; leaving teardown to idle_timeout",
            flush=True,
        )
        return
    if keep_alive_s > 0:
        print(f"[god] staying up {keep_alive_s}s before self-terminating", flush=True)
        time.sleep(keep_alive_s)
    try:
        import modal

        print(f"[god] terminating own sandbox {sandbox_id}", flush=True)
        modal.Sandbox.from_id(sandbox_id).terminate()
    except Exception as exc:
        print(
            f"[god] self-terminate failed ({type(exc).__name__}: {exc}); "
            f"idle_timeout will collect it",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="godbox.entrypoint",
        description="Run the God loop inside a Modal sandbox.",
    )
    parser.add_argument("--request", default=REQUEST_PATH)
    args = parser.parse_args()

    try:
        request = GodRequest.model_validate_json(
            Path(args.request).read_text(encoding="utf-8")
        )
    except Exception as exc:
        # No run_id yet, so there is no Dict to report into. stderr goes to
        # /god/god.log, which `god logs` can still read.
        print(f"[god] invalid request {args.request}: {exc}", file=sys.stderr)
        return 2

    status = StatusWriter(request.run_id)
    status.adopt()
    status.start_heartbeat()

    exit_code = 1
    try:
        # STARTING is written BEFORE the checks below, deliberately. It costs
        # one RPC and it is the difference between a status trail that reads
        # `launching -> starting -> failed` (GOD booted, then rejected the
        # environment) and one that reads `launching -> failed` (which is also
        # what a container that never came up at all looks like).
        asyncio.run(status.set_phase(Phase.STARTING, "GOD process up"))

        # ANTHROPIC_API_KEY comes from the mounted `demigod-anthropic` Secret.
        # Without it `make_llm()` silently returns the SCRIPTED toy client,
        # which would answer a real problem with canned text about glycolysis
        # and report success. Fail here instead.
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set inside the sandbox. The "
                "`demigod-anthropic` Secret is mounted by godbox.launch; if it "
                "is missing, GOD would silently fall back to the scripted toy "
                "LLM and report a fabricated answer."
            )
        payload, summary, exit_code = asyncio.run(_solve(request, status))
        # Outside the loop, on purpose -- see `_solve`.
        status.complete(payload, summary=summary)
    except BaseException as exc:
        # Includes KeyboardInterrupt and SystemExit on purpose: a GOD that dies
        # for ANY reason must leave a terminal phase behind, or every poller
        # sits on `spawning` until the heartbeat goes stale and has to guess.
        import traceback

        traceback.print_exc()
        # The traceback goes out FIRST. If the status channel is itself what
        # broke, `fail()` raises too -- and losing the real cause to the
        # secondary failure is how a five-minute diagnosis becomes an hour.
        try:
            status.fail(f"{type(exc).__name__}: {exc}")
        except Exception as report_exc:
            print(
                f"[god] could not write the failure to the status channel: "
                f"{type(report_exc).__name__}: {report_exc}",
                file=sys.stderr,
            )
        exit_code = 1
    finally:
        status.stop_heartbeat()
        _self_terminate(request.sandbox_id, request.keep_alive_s)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
