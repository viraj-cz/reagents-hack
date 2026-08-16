"""The live trace channel: one `modal.Queue` per GOD run.

WHY A THIRD CHANNEL. `status.py` explains the split between the Dict (small,
hot, immediately visible) and the Volume (large, cold, committed once). Both are
right for what they carry and neither can carry a trace:

    modal.Dict    capped at MAX_EVENTS=200, and every append is a
                  read-modify-write of one value. A run emits hundreds of
                  events -- a single demigod's tool calls can exhaust the cap --
                  and `Dict.put` raises once the value grows past its limit.
    Volume        invisible until the sandbox terminates, which is precisely
                  the interval a live view exists to show.
    modal.Queue   append-only, read from anywhere the instant it is written,
                  and drained rather than accumulated. That is a trace.

The Dict stays the source of truth for *state* -- phase, heartbeat, solution --
because a queue is consumed and a late reader would find it empty. The queue
carries *events*, which are worth exactly nothing once you have missed them.

WHY A BACKGROUND THREAD. `TraceSink.emit` is synchronous by contract and is
called from inside GOD's event loop. Modal's blocking API inside `async def`
warns and stalls that loop (see the README's list of constraints), so emitting
straight to the queue would make observability slow the thing it observes. The
sink therefore appends to a list under a lock, and one flusher thread batches
puts. If the queue is unreachable the events are dropped and GOD keeps running:
a broken trace channel must never be able to fail a paid-for run.
"""

from __future__ import annotations

import threading
import time
from typing import Any

QUEUE_PREFIX = "god-trace-"

# One `put_many` per batch. Large enough that a chatty phase is a handful of
# RPCs, small enough to stay well inside Modal's per-value size limit.
BATCH_SIZE = 64
FLUSH_INTERVAL_S = 0.25

# The bound that matters: if nobody is draining, this is how many events are
# held before the oldest are dropped. Dropping is correct -- an unread trace is
# not worth growing GOD's memory for -- but it must be visible, hence `dropped`.
MAX_PENDING = 5000

# How long a drainer waits on an empty queue before returning control. Short
# enough to notice a finished run promptly, long enough not to spin.
POLL_TIMEOUT_S = 2.0


def queue_name(run_id: str) -> str:
    """One queue per run, named alongside the run's Dict and volume."""

    return f"{QUEUE_PREFIX}{run_id}"


def open_queue(run_id: str, *, create_if_missing: bool = False) -> Any:
    import modal

    return modal.Queue.from_name(
        queue_name(run_id), create_if_missing=create_if_missing
    )


class QueueTracer:
    """A `TraceSink` that ships events out of the sandbox, off the event loop.

    Satisfies the same protocol as `TerminalTracer`, so GOD cannot tell the
    difference and nothing in the orchestrator needs to know this exists.
    """

    def __init__(
        self,
        run_id: str,
        *,
        queue: Any | None = None,
        batch_size: int = BATCH_SIZE,
        flush_interval_s: float = FLUSH_INTERVAL_S,
        max_pending: int = MAX_PENDING,
    ) -> None:
        self.run_id = run_id
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s
        self.max_pending = max_pending
        self.started_at = time.monotonic()
        self.dropped = 0
        self.sent = 0
        self._seq = 0
        self._pending: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._queue = (
            queue if queue is not None else open_queue(run_id, create_if_missing=True)
        )
        self._thread = threading.Thread(
            target=self._run, name=f"god-trace-{run_id}", daemon=True
        )
        self._thread.start()

    # -- TraceSink ------------------------------------------------------
    def emit(
        self,
        lane: str,
        kind: str,
        message: str,
        *,
        data: Any | None = None,
    ) -> None:
        with self._lock:
            self._seq += 1
            self._pending.append(
                {
                    "seq": self._seq,
                    "t": round(time.monotonic() - self.started_at, 3),
                    "lane": lane,
                    "kind": kind,
                    "message": message,
                    "data": data,
                }
            )
            overflow = len(self._pending) - self.max_pending
            if overflow > 0:
                # Drop the OLDEST: a live view that has fallen behind wants the
                # present, and the volume keeps the complete record either way.
                del self._pending[:overflow]
                self.dropped += overflow

    # -- flusher --------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            self._flush_once()
            self._stop.wait(self.flush_interval_s)
        self._flush_once()

    def _flush_once(self) -> None:
        while True:
            with self._lock:
                if not self._pending:
                    return
                batch = self._pending[: self.batch_size]
                del self._pending[: len(batch)]
            try:
                self._queue.put_many(batch)
                self.sent += len(batch)
            except Exception:
                # The channel is best-effort by design. Losing the trace must
                # not lose the run, so the batch goes and GOD carries on.
                self.dropped += len(batch)

    def close(self, *, timeout_s: float = 5.0) -> None:
        """Flush what is left, then stop. Safe to call twice."""

        self._stop.set()
        self._thread.join(timeout=timeout_s)
        self._flush_once()


def drain(
    run_id: str,
    *,
    queue: Any | None = None,
    timeout_s: float = POLL_TIMEOUT_S,
    batch_size: int = BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Take whatever is waiting. `[]` means "nothing yet", not "finished".

    Whether a run is over is a question for the status Dict; this function
    cannot tell an idle GOD from a dead one, and pretending otherwise would put
    the end-of-run decision in the channel least able to make it.
    """

    handle = queue if queue is not None else open_queue(run_id)
    try:
        return handle.get_many(batch_size, block=True, timeout=timeout_s) or []
    except Exception:
        return []


def delete(run_id: str) -> None:
    """Best-effort cleanup once a run is terminal and its trace is persisted."""

    try:
        import modal

        modal.Queue.delete(queue_name(run_id))
    except Exception:
        pass


__all__ = ["QueueTracer", "delete", "drain", "open_queue", "queue_name"]
