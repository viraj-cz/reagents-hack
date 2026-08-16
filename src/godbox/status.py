"""The status channel: one `modal.Dict` per GOD run.

WHY A DICT AND NOT THE VOLUME. Modal Volume writes are not visible outside the
writing container until an explicit commit, so a caller polling `out/` while
GOD is still working sees an empty or stale directory -- which looks exactly
like "the agent produced nothing", the single most miserable bug in this repo's
history. A `modal.Dict` is cross-container shared state with no commit
semantics: a `put` here is readable from a laptop on the next RPC.

So the two channels split by what each is good at:

    modal.Dict   status, phase, heartbeat, per-demigod progress, the solution
                 summary. Small, hot, immediately visible, survives the sandbox.
    Volume       artifacts and the full trace. Large, cold, committed once at
                 the end.

SINGLE WRITER, BY CONSTRUCTION. Exactly one process writes a run's Dict: the
launcher primes it and then hands off, permanently, to the `godbox.entrypoint`
process inside the sandbox. That is what makes the in-memory mirrors below
(`_events`, `_demigods`) safe -- appending to a list in a Dict is otherwise a
read-modify-write race. The one exception is `followups`, which outside callers
append to and GOD drains; see `push_followup`.

The heartbeat thread is a second thread, not a second writer of the same keys:
it touches `heartbeat` and nothing else.

THE CONTRACT. Keys, and when each is written:

    key               written by            when
    ---------------   -------------------   ------------------------------------
    schema            launcher              once, at launch. Bump on any change.
    run_id            launcher              once
    app_name          launcher              once
    artifact_volume   launcher              once
    artifact_path     launcher              once
    sandbox_id        launcher              once, BEFORE exec -- so a run that
                                            never boots is still terminable
    started_at        launcher              once (unix seconds)
    problem_id        launcher              once
    domain_count      launcher              once (pinned by the operator, or
                                            null when GOD chose it; `domains`
                                            below is what was actually planned)
    phase             launcher, then GOD    every transition; see `Phase`
    updated_at        launcher, then GOD    every write
    heartbeat         GOD (ticker thread)   every HEARTBEAT_INTERVAL_S
    domains           GOD                   once, after planning
    demigods          GOD                   on each spawn start and finish
    events            GOD                   append-only, capped at MAX_EVENTS
    followups         outside callers, GOD  appended by `push_followup`, drained
                                            by GOD at phase boundaries
    solution          GOD                   once, at completion
    error             GOD                   once, on failure
    finished_at       GOD                   once, terminal

Values are plain JSON types on purpose. Modal will happily pickle a pydantic
model into a Dict, but then reading the status requires the writer's classes to
be importable -- and the whole point of this channel is that a bare laptop, or
a dashboard, can read it.

READ IT WITH: `read_status(run_id)`, or `god status <run_id>`.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

STATUS_SCHEMA = 1
"""Version of the key contract above. A reader that finds a higher number
should say so rather than silently misreport a run."""

DICT_PREFIX = "god-run-"
"""One Dict per run, named `god-run-<run_id>`. Mirrors the volume naming
(`demigod-run-<run_id>-out`) so a run's objects are greppable in
`modal dict list` / `modal volume list` as one family."""

HEARTBEAT_INTERVAL_S = 15.0
"""How often the ticker thread refreshes `heartbeat`."""

STALE_AFTER_S = 90.0
"""A run whose heartbeat is older than this is reported STALE rather than
running. Six missed ticks -- generous enough to absorb a slow Anthropic call
holding the GIL, tight enough that a dead container is obvious within
two minutes."""

MAX_EVENTS = 200
"""Events are a progress log, not an audit log; the full record goes to the
volume as trace.json. Capping keeps the Dict value small -- `Dict.put` raises
RequestSizeError on oversized values, and losing the whole status channel
because a chatty run overflowed one key would be a poor trade."""


class Phase(StrEnum):
    """Where a run is. Ordered as a run passes through them.

    A `StrEnum` so the value that goes into the Dict is a plain string in every
    serializer, including Modal's. `.value` is still used at every call site --
    relying on the implicit str-ness would make the stored contract depend on
    the enum class staying importable, which is exactly what the JSON-only
    rule in the module docstring rules out.
    """

    LAUNCHING = "launching"
    """Dict primed, sandbox being created. Written by the launcher."""
    STARTING = "starting"
    """GOD's process booted inside the sandbox and took over the channel."""
    PLANNING = "planning"
    TRANSFORMING = "transforming"
    SPAWNING = "spawning"
    """DEMI_GOD sandboxes are up. The long one; `demigods` tracks it."""
    INTEGRATING = "integrating"
    DONE = "done"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (Phase.DONE, Phase.FAILED)


def dict_name(run_id: str) -> str:
    """Name of the status Dict for a run."""
    return f"{DICT_PREFIX}{run_id}"


def run_id_from_dict_name(name: str) -> str | None:
    """Inverse of `dict_name`, or None if `name` is not one of ours."""
    if not name.startswith(DICT_PREFIX):
        return None
    return name[len(DICT_PREFIX) :]


def open_dict(run_id: str, *, create_if_missing: bool = False) -> Any:
    """Handle to a run's status Dict.

    Imported lazily so that the pure contract logic in this module -- phases,
    key names, staleness -- stays testable with no Modal client and no account.
    """
    import modal

    return modal.Dict.from_name(dict_name(run_id), create_if_missing=create_if_missing)


# --- read side ---------------------------------------------------------------


@dataclass(frozen=True)
class GodStatus:
    """A snapshot of one run, as seen from outside.

    Every field tolerates absence. A run whose sandbox died between
    `Sandbox.create` and the first status write still produces a usable
    snapshot -- one that says `launching`, with a stale heartbeat, which is
    exactly the diagnosis.
    """

    run_id: str
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def schema(self) -> int:
        return int(self.raw.get("schema", 0))

    @property
    def phase(self) -> str:
        return str(self.raw.get("phase", "unknown"))

    @property
    def is_terminal(self) -> bool:
        return self.phase in (Phase.DONE.value, Phase.FAILED.value)

    @property
    def sandbox_id(self) -> str | None:
        return self.raw.get("sandbox_id")

    @property
    def started_at(self) -> float | None:
        return self.raw.get("started_at")

    @property
    def finished_at(self) -> float | None:
        return self.raw.get("finished_at")

    @property
    def elapsed_s(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at else time.time()
        return end - self.started_at

    @property
    def heartbeat_age_s(self) -> float | None:
        beat = self.raw.get("heartbeat")
        return None if beat is None else time.time() - beat

    @property
    def is_stale(self) -> bool:
        """Alive-but-slow vs dead. Terminal runs are never stale -- a finished
        run legitimately stops beating."""
        if self.is_terminal:
            return False
        age = self.heartbeat_age_s
        return age is not None and age > STALE_AFTER_S

    @property
    def domains(self) -> list[str]:
        return list(self.raw.get("domains") or [])

    @property
    def demigods(self) -> dict[str, dict[str, Any]]:
        return dict(self.raw.get("demigods") or {})

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self.raw.get("events") or [])

    @property
    def followups(self) -> list[dict[str, Any]]:
        return list(self.raw.get("followups") or [])

    @property
    def solution(self) -> dict[str, Any] | None:
        return self.raw.get("solution")

    @property
    def error(self) -> str | None:
        return self.raw.get("error")


def read_status(run_id: str) -> GodStatus:
    """Read a run's whole status in one pass.

    `Dict.items()` rather than N `get`s: one RPC, and -- more importantly -- a
    single consistent-enough snapshot, so the printed phase and the printed
    demigod table cannot come from two different moments.

    Raises `modal.exception.NotFoundError` for an unknown run_id; the Dict is
    created by the launcher, so its absence means no such run was ever
    launched in this environment.
    """
    return GodStatus(run_id=run_id, raw=dict(open_dict(run_id).items()))


def list_runs(*, limit: int = 100) -> list[str]:
    """Run ids that have a status Dict, newest first.

    Cheap enough to be the default way of answering "what did I leave
    running?" -- it lists objects, it does not read them.
    """
    import modal

    found: list[tuple[Any, str]] = []
    for handle in modal.Dict.objects.list(max_objects=limit):
        run_id = run_id_from_dict_name(handle.name or "")
        if run_id is None:
            continue
        try:
            created = handle.info().created_at
        except Exception:
            created = None
        found.append((created, run_id))
    found.sort(key=lambda pair: (pair[0] is not None, pair[0]), reverse=True)
    return [run_id for _, run_id in found]


def push_followup(run_id: str, message: str) -> int:
    """Append a message to the run's follow-up inbox. Returns the queue depth.

    THE ONE PLACE an outside caller writes to a running GOD's Dict, and the
    reason the sandbox is long-lived rather than a fire-and-forget Function.

    PROVISIONAL, and honest about it: GOD drains this inbox at phase boundaries
    and records each message in `events`, so delivery is observable end to end.
    Acting on a follow-up mid-run -- re-planning, spawning another domain --
    needs a re-entrant orchestrator and is not built. What works today is the
    channel, not the behaviour.

    Read-modify-write, so two callers racing can lose a message. Follow-ups are
    typed by a human at human pace; a lock would cost more than it buys.
    """
    handle = open_dict(run_id)
    queue = list(handle.get("followups") or [])
    queue.append({"at": time.time(), "message": message, "delivered": False})
    handle.put("followups", queue)
    return len(queue)


# --- write side --------------------------------------------------------------


class StatusWriter:
    """GOD's end of the channel. One per run, one per process.

    Async methods exist because most writes happen from inside `God.solve()`'s
    event loop, and Modal's blocking interface called from an async context
    emits `AsyncUsageWarning` and stalls the loop for the RPC (~200ms measured).
    `.aio` is the documented rewrite. The sync methods are for the phases that
    are genuinely outside the loop -- launch, completion, and the heartbeat
    thread.
    """

    def __init__(self, run_id: str, *, backend: Any = None) -> None:
        self.run_id = run_id
        # `backend` is an injection point for offline tests: any object with
        # `put`/`get`. Production passes nothing and gets a modal.Dict.
        #
        # `create_if_missing` even though the launcher already primed it: a
        # GOD that cannot open its status channel cannot report that it cannot
        # report. Recreating a lost Dict costs one RPC and turns a total
        # blackout into a run that is merely missing its launch metadata.
        self._dict = (
            backend
            if backend is not None
            else open_dict(run_id, create_if_missing=True)
        )
        # In-memory mirrors of the append-only keys. Safe because this process
        # is the only writer of them; see the module docstring.
        self._events: list[dict[str, Any]] = []
        self._demigods: dict[str, dict[str, Any]] = {}
        self._phase: str = Phase.STARTING.value
        self._beat_stop: threading.Event | None = None
        self._beat_thread: threading.Thread | None = None
        # Demigods run concurrently, so two `demigod()` calls can be in flight
        # at once. Both send the WHOLE `_demigods` mirror, and without this the
        # older snapshot can land last and silently un-report a finished
        # demigod until the next write. A lock is cheaper than a sequence
        # number and cannot get the ordering wrong.
        #
        # Constructed outside a running loop on purpose: asyncio.Lock has not
        # bound a loop at construction since Python 3.10.
        self._lock = asyncio.Lock()

    # -- plumbing --

    def _put(self, key: str, value: Any) -> None:
        """Write from OUTSIDE an event loop. Never call this from `async def`.

        Modal's blocking interface used inside a running loop raises an
        `AsyncUsageWarning` and stalls the loop for the whole RPC. The public
        sync methods below (`complete`, `fail`) are therefore for the phases
        that genuinely run outside the loop -- and `godbox/entrypoint.py` is
        arranged so that stays true, returning its payload from `_solve` rather
        than writing the terminal status from inside `asyncio.run`.
        """
        self._dict.put(key, value)
        if key != "heartbeat":
            self._dict.put("updated_at", time.time())

    async def _aput(self, key: str, value: Any) -> None:
        """Write from inside the event loop.

        `.aio` rather than the blocking call: Modal's blocking interface used
        in an async context emits `AsyncUsageWarning` and stalls the loop for
        the duration of the RPC (~200ms measured), which would serialize the
        very demigod fan-out this is reporting on.
        """
        async with self._lock:
            aio = getattr(self._dict.put, "aio", None)
            if aio is None:  # test double, or a future SDK without .aio
                self._put(key, value)
                return
            await aio(key, value)
            if key != "heartbeat":
                await aio("updated_at", time.time())

    def adopt(self) -> None:
        """Take over a Dict the launcher primed, without losing its events.

        Called once, first thing inside the sandbox. Seeds the in-memory
        mirrors from what is already there so the launcher's `launching` event
        is not clobbered by GOD's first append.
        """
        try:
            self._events = list(self._dict.get("events") or [])
            self._demigods = dict(self._dict.get("demigods") or {})
        except Exception:
            # A missing or unreadable Dict must not stop the run. Losing
            # progress reporting is bad; refusing to reason because the
            # reporting channel hiccuped is worse.
            self._events, self._demigods = [], {}

    # -- heartbeat --

    def start_heartbeat(self, interval_s: float = HEARTBEAT_INTERVAL_S) -> None:
        """Begin proving liveness.

        Without this, "GOD is 20 minutes into a hard planning call" and "GOD's
        container was OOM-killed" look identical from outside: same phase, same
        `updated_at`, no output. The heartbeat is the only thing that separates
        them, which is why it runs on its own thread rather than being folded
        into the phase writes it would otherwise duplicate.

        Daemon thread: it must never keep the process alive past the run.
        """
        if self._beat_thread is not None:
            return
        stop = threading.Event()

        def tick() -> None:
            while not stop.wait(interval_s):
                # A transient RPC failure is swallowed: the next tick reports
                # liveness, whereas crashing this thread would permanently mark
                # a perfectly healthy run as stale.
                with contextlib.suppress(Exception):
                    self._dict.put("heartbeat", time.time())

        self._beat_stop = stop
        self._beat_thread = threading.Thread(
            target=tick, name=f"god-heartbeat-{self.run_id}", daemon=True
        )
        # Suppressed for the same reason the ticker suppresses: reporting is
        # not the job. A GOD that refused to start because its first heartbeat
        # RPC failed would trade a cosmetic problem for a total one.
        with contextlib.suppress(Exception):
            self._dict.put("heartbeat", time.time())
        self._beat_thread.start()

    def stop_heartbeat(self) -> None:
        if self._beat_stop is not None:
            self._beat_stop.set()
        self._beat_thread = None

    # -- phases and events --

    def _record(self, message: str) -> list[dict[str, Any]]:
        self._events.append(
            {"at": time.time(), "phase": self._phase, "message": message}
        )
        del self._events[:-MAX_EVENTS]
        return self._events

    async def set_phase(self, phase: Phase, message: str = "") -> None:
        self._phase = phase.value
        await self._aput("phase", phase.value)
        if message:
            await self._aput("events", self._record(message))

    async def note(self, message: str) -> None:
        """A progress line. Also printed, so the sandbox log and the status
        channel tell the same story."""
        print(f"[god] {message}", flush=True)
        await self._aput("events", self._record(message))

    async def set_domains(self, names: list[str]) -> None:
        await self._aput("domains", names)

    async def demigod(self, name: str, status: str, **fields: Any) -> None:
        """Upsert one demigod's row. `status` is running/ok/failed/timeout."""
        row = self._demigods.setdefault(name, {})
        row["status"] = status
        row.update({k: v for k, v in fields.items() if v is not None})
        row.setdefault("started_at", time.time())
        if status != "running":
            row["finished_at"] = time.time()
        await self._aput("demigods", self._demigods)

    async def drain_followups(self) -> list[str]:
        """Collect anything sent since the last check and acknowledge it.

        Marks rather than deletes: a follow-up the user typed should still be
        visible in `god status` after GOD has seen it, otherwise the send looks
        like it vanished.
        """
        aio = getattr(self._dict.get, "aio", None)
        raw = await aio("followups") if aio else self._dict.get("followups")
        queue = list(raw or [])
        fresh = [item for item in queue if not item.get("delivered")]
        if not fresh:
            return []
        for item in queue:
            item["delivered"] = True
        await self._aput("followups", queue)
        messages = [str(item.get("message", "")) for item in fresh]
        for message in messages:
            await self.note(f"FOLLOWUP received: {message}")
        return messages

    # -- terminal --

    def complete(self, solution: dict[str, Any], summary: str = "") -> None:
        """Terminal success. Sync: the event loop is finished by now."""
        self._phase = Phase.DONE.value
        if summary:
            self._put("events", self._record(summary))
        self._put("solution", solution)
        self._put("finished_at", time.time())
        self._put("phase", Phase.DONE.value)

    def fail(self, error: str) -> None:
        """Terminal failure.

        `phase` is written LAST on both terminal paths so a poller can never
        read `done` before the payload that makes `done` meaningful is there.
        """
        self._phase = Phase.FAILED.value
        self._put("events", self._record(f"FAILED: {error}"))
        self._put("error", error)
        self._put("finished_at", time.time())
        self._put("phase", Phase.FAILED.value)
