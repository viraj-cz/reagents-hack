"""One GOD run, watched by zero or more browsers.

Three things live here and nothing else does:

* `Run` -- the event log for a single `God.solve()`, plus the fan-out to
  whoever is subscribed. It is also the `TraceSink` handed to GOD, so there is
  no adapter between "the orchestrator emitted something" and "a tab sees it".
* `RunStore` -- run ids to runs, and the lifecycle around the task.
* `build_problem` -- free text to `NativeProblem`.

A run keeps its whole event log. That is what makes a reload cheap: a
subscriber that arrives at second 40 is sent every event that already happened
and then joins the live stream, so there is exactly one rendering path in the
frontend instead of one for "watched from the start" and one for "reconnected".
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import threading
import time
import traceback
import uuid
from collections.abc import AsyncIterator
from typing import Any

from reagents.contracts import NativeProblem
from reagents.god.orchestrator import God
from resolution.events import RunSnapshot, UiEvent, build_event, jsonable
from resolution.narration import make_llm

# A closed subscriber queue is an unread pipe by another name. Bound it, and
# drop the slowest reader's backlog rather than let one stalled tab grow the
# server's memory without limit.
SUBSCRIBER_QUEUE_MAX = 2048

# WHERE a demigod executes. The orchestrator is polymorphic over exactly this
# (`DemigodRuntimeProtocol`), so the UI exposes the choice rather than inventing
# a second one: `inprocess` reasons inside this server, `sandbox` gives each
# demigod its own Modal sandbox and routes its tools through the deployed
# broker. They are not interchangeable in cost or in what they prove -- a
# sandbox run is the only one that exercises image, secret, volume and lease
# plumbing, and it is the only one that can fail on any of them.
EXECUTION_INPROCESS = "inprocess"
EXECUTION_SANDBOX = "sandbox"
EXECUTION_GODBOX = "godbox"
EXECUTIONS = (EXECUTION_INPROCESS, EXECUTION_SANDBOX, EXECUTION_GODBOX)

# Per-demigod turn cap for a sandboxed GOD. THE cost lever: every turn is an
# Anthropic call, and a GOD sandbox that runs away is one nobody is watching.
DEFAULT_MAX_TURNS = 12


class Run:
    """The trace sink, the event log, and the fan-out for a single run."""

    def __init__(
        self,
        run_id: str,
        problem: NativeProblem,
        *,
        mode: str,
        domain_count: int,
        execution: str = EXECUTION_INPROCESS,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> None:
        self.run_id = run_id
        self.problem = problem
        self.mode = mode
        self.domain_count = domain_count
        self.execution = execution
        self.max_turns = max_turns
        self.events: list[UiEvent] = []
        self.snapshot = RunSnapshot(
            run_id=run_id,
            mode=mode,
            execution=execution,
            question=problem.question or problem.statement,
            started_at=time.time(),
        )
        self.task: asyncio.Task[Any] | None = None
        self._started_monotonic = time.monotonic()
        self._seq = 0
        self._lock = threading.Lock()
        self._subscribers: set[asyncio.Queue[UiEvent | None]] = set()
        self._loop = asyncio.get_event_loop()
        self._loop_thread = threading.get_ident()

    # -- TraceSink -------------------------------------------------------
    def emit(
        self,
        lane: str,
        kind: str,
        message: str,
        *,
        data: Any | None = None,
    ) -> None:
        """Synchronous by contract: a tool may call this from a worker thread.

        The fan-out therefore hops back onto the loop rather than touching an
        `asyncio.Queue` from whichever thread happened to be running the tool.
        """

        with self._lock:
            self._seq += 1
            event = build_event(
                seq=self._seq,
                elapsed_s=time.monotonic() - self._started_monotonic,
                lane=lane,
                kind=kind,
                message=message,
                data=data,
            )
            self.events.append(event)
            subscribers = list(self._subscribers)

        if not subscribers:
            return
        if threading.get_ident() == self._loop_thread:
            self._publish(event, subscribers)
        else:
            self._loop.call_soon_threadsafe(self._publish, event, subscribers)

    def _publish(
        self,
        event: UiEvent | None,
        subscribers: list[asyncio.Queue[UiEvent | None]],
    ) -> None:
        for queue in subscribers:
            # A reader this far behind is not worth stalling the run for; its
            # tab can reload and get the whole log back from `subscribe()`.
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(event)

    # -- subscription ----------------------------------------------------
    async def subscribe(self) -> AsyncIterator[UiEvent]:
        """Replay everything so far, then follow the live stream to the end."""

        queue: asyncio.Queue[UiEvent | None] = asyncio.Queue(SUBSCRIBER_QUEUE_MAX)
        with self._lock:
            backlog = list(self.events)
            finished = self.snapshot.status != "running"
            if not finished:
                self._subscribers.add(queue)
        try:
            for event in backlog:
                yield event
            if finished:
                return
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event
        finally:
            with self._lock:
                self._subscribers.discard(queue)

    def close(self) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
            self._subscribers.clear()
        self._publish(None, subscribers)

    # -- lifecycle -------------------------------------------------------
    async def _build_runtime(self) -> Any:
        """`None` means GOD's default in-process runtime."""

        if self.execution != EXECUTION_SANDBOX:
            return None

        def build() -> Any:
            from broker.session import modal_session
            from reagents.demigod.sandbox_runtime import SandboxDemigodRuntime

            # `modal_session()` resolves the deployed router's URL, which is a
            # network lookup -- hence the thread. Without the toolbox a demigod
            # is told its whole toolset is unreachable and reasons with nothing,
            # so the broker is not optional decoration here.
            return SandboxDemigodRuntime(
                run_id=self.run_id,
                toolbox=modal_session(),
                tracer=self,
            )

        self.emit(
            "GOD",
            "RUNTIME",
            "one Modal sandbox per demigod; tools brokered by lease",
        )
        return await asyncio.to_thread(build)

    async def _execute_in_godbox(self) -> None:
        """GOD in its own sandbox; this process only watches.

        Nothing of `God` runs here -- not the planner, not the integrator. The
        events reaching the browser are the same ones, re-emitted from the queue
        the sandbox writes to, so the stream and the tree cannot tell the
        difference. What ends the run is the status Dict, never an empty queue.
        """

        from resolution import godbox_run

        self.emit(
            "GOD",
            "RUNTIME",
            "GOD in its own Modal sandbox, spawning one sandbox per demigod",
        )
        request = godbox_run.build_request(
            self.run_id,
            self.problem,
            domain_count=self.domain_count,
            max_turns=self.max_turns,
        )
        handle = await asyncio.to_thread(godbox_run.launch, request)
        self.snapshot.sandbox_id = handle.sandbox_id
        self.emit(
            "GOD",
            "LAUNCHED",
            f"sandbox {handle.sandbox_id}",
            data={
                "sandbox_id": handle.sandbox_id,
                "app_name": handle.app_name,
                "status_dict": handle.status_dict,
                "artifact_volume": handle.artifact_volume,
            },
        )

        status = await godbox_run.follow(self.run_id, self.emit)
        solution = status.solution if status else None
        if solution:
            self.snapshot.solution = jsonable(solution)
            self.snapshot.status = "done"
        else:
            self.snapshot.status = "error"
            self.snapshot.error = (
                status.error if status and status.error else "GOD produced no solution"
            )
        self.snapshot.domains = [
            {"name": name} for name in (status.domains if status else [])
        ]
        await asyncio.to_thread(godbox_run.cleanup, self.run_id)

    async def execute(self) -> None:
        if self.execution == EXECUTION_GODBOX:
            try:
                await self._execute_in_godbox()
            except asyncio.CancelledError:
                self.snapshot.status = "cancelled"
                self.emit("GOD", "FAILURE", "run cancelled")
                raise
            except Exception as exc:
                self.snapshot.status = "error"
                self.snapshot.error = f"{type(exc).__name__}: {exc}"
                self.emit(
                    "GOD",
                    "FAILURE",
                    self.snapshot.error,
                    data={"traceback": traceback.format_exc()[-4000:]},
                )
            finally:
                self.snapshot.finished_at = time.time()
                self.emit(
                    "GOD",
                    "RUN END",
                    self.snapshot.status,
                    data=self.snapshot.to_json(),
                )
                self.close()
            return

        llm = make_llm(self.mode, self)
        try:
            runtime = await self._build_runtime()
        except Exception as exc:
            # A sandbox runtime that cannot be constructed is a setup failure --
            # no broker deployed, no Modal credentials -- and silently falling
            # back to in-process would answer the question the user did not ask.
            self.snapshot.status = "error"
            self.snapshot.error = f"could not reach Modal: {type(exc).__name__}: {exc}"
            self.emit("GOD", "FAILURE", self.snapshot.error)
            self.snapshot.finished_at = time.time()
            self.emit(
                "GOD",
                "RUN END",
                self.snapshot.status,
                data=self.snapshot.to_json(),
            )
            self.close()
            return

        god = God(llm, domain_count=self.domain_count, tracer=self, runtime=runtime)
        try:
            solution = await god.solve(self.problem)
            self.snapshot.solution = jsonable(solution.model_dump())
            self.snapshot.status = "done"
        except asyncio.CancelledError:
            self.snapshot.status = "cancelled"
            self.emit("GOD", "FAILURE", "run cancelled")
            raise
        except Exception as exc:
            self.snapshot.status = "error"
            self.snapshot.error = f"{type(exc).__name__}: {exc}"
            self.emit(
                "GOD",
                "FAILURE",
                self.snapshot.error,
                data={"traceback": traceback.format_exc()[-4000:]},
            )
        finally:
            self.snapshot.finished_at = time.time()
            self.snapshot.domains = [
                {
                    "name": spec.name,
                    "axes": [axis.value for axis in spec.axes],
                    "language": spec.language,
                    "tools": list(spec.tool_ids),
                }
                for spec in god.last_trace.specs
            ]
            self.emit(
                "GOD",
                "RUN END",
                self.snapshot.status,
                data=self.snapshot.to_json(),
            )
            self.close()

    def transcript(self, node: str) -> list[dict[str, Any]]:
        with self._lock:
            return [e.to_json() for e in self.events if e.node == node]


class RunStore:
    def __init__(self, *, max_runs: int = 32) -> None:
        self.runs: dict[str, Run] = {}
        self.order: list[str] = []
        self.max_runs = max_runs

    def create(
        self,
        problem: NativeProblem,
        *,
        mode: str,
        domain_count: int,
        execution: str = EXECUTION_INPROCESS,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> Run:
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        run = Run(
            run_id,
            problem,
            mode=mode,
            domain_count=domain_count,
            execution=execution,
            max_turns=max_turns,
        )
        run.task = asyncio.create_task(run.execute(), name=run_id)
        self.runs[run_id] = run
        self.order.append(run_id)
        self._evict()
        return run

    def get(self, run_id: str) -> Run | None:
        return self.runs.get(run_id)

    def list(self) -> list[dict[str, Any]]:
        return [self.runs[rid].snapshot.to_json() for rid in reversed(self.order)]

    def _evict(self) -> None:
        while len(self.order) > self.max_runs:
            stale = self.order.pop(0)
            run = self.runs.pop(stale, None)
            if run and run.task and not run.task.done():
                run.task.cancel()


_SENTENCE = re.compile(r"(?<=[.?!])\s+")


def build_problem(
    prompt: str,
    *,
    entities: list[str] | None = None,
    constraints: list[str] | None = None,
    problem_id: str | None = None,
) -> NativeProblem:
    """Free text to `NativeProblem`, without inventing anything.

    `entities` is the only field with teeth: `reagents.isolation.native_terms`
    reads it and nothing else, so it is the list of words a demigod must never
    see. Guessing at it would be worse than leaving it empty -- a wrong guess
    seals the wrong words and, worse, reports a run as sealed against terms the
    user never named. So it comes from the caller or it stays empty.
    """

    text = prompt.strip()
    if not text:
        raise ValueError("prompt is empty")
    question = _last_question(text)
    return NativeProblem(
        id=problem_id or _slug(text),
        statement=text,
        entities=[e.strip() for e in (entities or []) if e.strip()],
        constraints=[c.strip() for c in (constraints or []) if c.strip()],
        question=question,
    )


def _last_question(text: str) -> str:
    sentences = [s.strip() for s in _SENTENCE.split(text) if s.strip()]
    for sentence in reversed(sentences):
        if sentence.endswith("?"):
            return sentence
    return sentences[-1] if sentences else text


def _slug(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())[:4]
    return "-".join(words) or "problem"
