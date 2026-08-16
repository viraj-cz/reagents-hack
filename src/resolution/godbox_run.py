"""Watching a GOD that is running somewhere else.

The other two execution modes drive `God.solve()` in this process, so the
`TraceSink` handed to the orchestrator *is* the thing the browser subscribes to
-- no channel in between. When GOD runs in its own Modal sandbox that is no
longer possible, and this module is the seam that keeps the UI identical anyway:

    inside the sandbox   `godbox.trace_channel.QueueTracer` puts events on a
                         per-run `modal.Queue`
    here                 drain that queue and re-emit each event into the same
                         `Run` sink the in-process modes write to

So one event shape reaches the frontend regardless of where GOD ran, and the
stream, the tree and the transcripts work without knowing the difference.

TWO CHANNELS, TWO QUESTIONS, and conflating them is the bug this is written to
avoid. The queue answers *what just happened*; it cannot distinguish an idle GOD
from a dead one, because both look like an empty queue. The status Dict answers
*is this run over* -- it carries the phase, the heartbeat, the solution and the
error. Termination is decided by the Dict, always.
"""

from __future__ import annotations

import asyncio
from typing import Any

# How often to ask the Dict whether the run ended. The queue drain blocks for
# its own timeout, so this is a ceiling on how long a finished run can look
# like a running one, not a busy-poll interval.
STATUS_POLL_S = 3.0

# A terminal phase is not quite the end: the last trace events may still be in
# flight from the sandbox's flusher. Keep draining for a moment after the Dict
# says done, or the stream loses its own ending.
DRAIN_AFTER_TERMINAL_S = 6.0


def build_request(
    run_id: str,
    problem: Any,
    *,
    domain_count: int | None,
    max_turns: int,
) -> Any:
    from godbox.layout import GodRequest

    return GodRequest(
        run_id=run_id,
        problem=problem,
        domain_count=domain_count,
        max_turns=max_turns,
        # WITHOUT THIS every demigod is told its whole toolset is unreachable.
        # `GodRequest.use_broker` defaults to False, and the entrypoint reads it
        # as the single switch for three things at once: whether a lease is
        # published, whether the runtime requires one, and whether egress is
        # pinned. Leaving it unset produced three artifacts with `tool_trace=0`
        # and blockers reading "Domain tools ... are not reachable from this
        # environment" -- a run that looks successful and reasoned with nothing.
        #
        # The CLI opts OUT (`--no-broker`); a UI run has no reason to, since
        # brokered calls are the only tool evidence the agent does not author
        # itself.
        use_broker=True,
        # Operator approval travels with the request and nothing in the sandbox
        # can widen it. The UI has no approval flow yet, so it grants nothing --
        # which is the safe default, not an oversight: a demigod that wants a
        # write tool fails loudly instead of getting one by accident.
        approved_write_tools=[],
        approved_high_risk_tools=[],
    )


def launch(request: Any) -> Any:
    """Create the sandbox and hand over the task. Blocking; call in a thread."""

    from godbox.launch import launch_god

    return launch_god(request, verbose=False)


def read_status(run_id: str) -> Any:
    from godbox.status import read_status as _read

    return _read(run_id)


async def follow(
    run_id: str,
    emit: Any,
    *,
    status_poll_s: float = STATUS_POLL_S,
    drain_after_terminal_s: float = DRAIN_AFTER_TERMINAL_S,
) -> Any:
    """Pump one sandboxed run's trace into `emit` until the Dict says it ended.

    Returns the final `GodStatus`. Every Modal call is a blocking RPC, so all of
    them go through `asyncio.to_thread` -- this coroutine shares an event loop
    with the SSE responses feeding every other watcher, and one synchronous
    `get_many` here would stall all of them.
    """

    from godbox.trace_channel import drain

    terminal_at: float | None = None
    last_status: Any = None
    loop = asyncio.get_running_loop()
    next_status_check = 0.0

    while True:
        events = await asyncio.to_thread(drain, run_id)
        for event in _ordered(events):
            _relay(emit, event)

        now = loop.time()
        if now >= next_status_check:
            last_status = await asyncio.to_thread(read_status, run_id)
            next_status_check = now + status_poll_s
            if last_status.is_terminal and terminal_at is None:
                terminal_at = now

        if (
            terminal_at is not None
            and loop.time() - terminal_at >= drain_after_terminal_s
        ):
            # One last sweep: the sandbox's flusher may have raced the Dict.
            tail = await asyncio.to_thread(drain, run_id, timeout_s=0.5)
            for event in _ordered(tail):
                _relay(emit, event)
            return last_status

        if not events:
            await asyncio.sleep(0.2)


def _ordered(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sandbox order, not arrival order.

    `seq` is assigned by the writer inside the sandbox, so it is the only
    total order that reflects what actually happened. A drain returns whatever
    the queue hands back, and two flushes can interleave.
    """

    return sorted(events, key=lambda e: e.get("seq") or 0)


def _relay(emit: Any, event: dict[str, Any]) -> None:
    """Re-emit one relayed event, keeping the clock it arrived with.

    `t` was stamped inside the sandbox against GOD's own start. Dropping it and
    re-stamping on arrival is what made a whole batch share one timestamp.
    """

    at = event.get("t")
    emit(
        str(event.get("lane") or "GOD"),
        str(event.get("kind") or "NOTE"),
        str(event.get("message") or ""),
        data=event.get("data"),
        at=float(at) if isinstance(at, (int, float)) else None,
    )


def cleanup(run_id: str) -> None:
    from godbox.trace_channel import delete

    delete(run_id)
