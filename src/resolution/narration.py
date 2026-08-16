"""LLM clients that let the UI watch text arrive, not just appear.

The orchestrator's own trace says *what stage GOD is in*. It deliberately never
carries model output, because its other consumer is a terminal and its contract
is "no chain-of-thought, bounded lines". A browser can afford the whole stream,
so these wrappers add one extra event kind -- `TEXT` -- carrying the model's
output as it is produced.

Two paths, and the difference between them is visible in the UI rather than
hidden:

* `reagents.llm.streaming.StreamingAnthropicLLM` streams genuinely, from the
  Anthropic API. It lives in `reagents` rather than here so a GOD running in
  its own Modal sandbox -- whose image ships `reagents` and not this package --
  streams too. Re-exported below for callers that already import it from here.
* `ReplayLLM` wraps a non-streaming client (in practice `ScriptedLLM`) and
  replays its already-complete answer in chunks. Every event it emits is
  tagged `simulated: true` and the frontend labels the run REPLAY, because a
  fake typing animation presented as live inference is a lie about latency and
  about where the answer came from.
"""

from __future__ import annotations

import asyncio
from typing import Any, TypeVar

from pydantic import BaseModel

from reagents.contracts import Budget
from reagents.llm.client import LLMClient
from reagents.llm.streaming import StreamingAnthropicLLM, lane_for_phase
from reagents.tools.registry import BoundToolPack
from reagents.tracing import NullTracer, TraceSink

T = TypeVar("T", bound=BaseModel)

# Big enough that a long answer does not become thousands of SSE frames, small
# enough that the text visibly flows rather than landing in paragraphs.
REPLAY_CHUNK_CHARS = 18
REPLAY_CHUNK_DELAY_S = 0.012


class _TextEmitter:
    def __init__(self, tracer: TraceSink | None) -> None:
        self.tracer = tracer or NullTracer()

    def emit_text(self, phase: str, delta: str, *, simulated: bool) -> None:
        if not delta:
            return
        self.tracer.emit(
            lane_for_phase(phase),
            "TEXT",
            delta,
            data={"phase": phase, "simulated": simulated},
        )

    def emit_open(self, phase: str, *, simulated: bool) -> None:
        self.tracer.emit(
            lane_for_phase(phase),
            "TEXT OPEN",
            phase,
            data={"phase": phase, "simulated": simulated},
        )

    def emit_close(self, phase: str, *, simulated: bool) -> None:
        self.tracer.emit(
            lane_for_phase(phase),
            "TEXT CLOSE",
            phase,
            data={"phase": phase, "simulated": simulated},
        )


class ReplayLLM(_TextEmitter):
    """Any `LLMClient`, with its output replayed as a paced character stream."""

    def __init__(
        self,
        inner: LLMClient,
        tracer: TraceSink | None = None,
        *,
        chunk_chars: int = REPLAY_CHUNK_CHARS,
        delay_s: float = REPLAY_CHUNK_DELAY_S,
    ) -> None:
        super().__init__(tracer)
        self.inner = inner
        self.chunk_chars = max(1, chunk_chars)
        self.delay_s = max(0.0, delay_s)

    def set_tracer(self, tracer: TraceSink) -> None:
        self.tracer = tracer

    async def complete(
        self,
        *,
        system: str,
        user: str,
        response_model: type[T],
        phase: str = "",
    ) -> T:
        result = await self.inner.complete(
            system=system, user=user, response_model=response_model, phase=phase
        )
        await self._replay(phase, _render(result))
        return result

    async def run_tool_loop(
        self,
        *,
        system: str,
        user: str,
        tools: BoundToolPack,
        response_model: type[T],
        budget: Budget,
        phase: str = "",
        response_schema: dict[str, Any] | None = None,
    ) -> tuple[T, list[dict[str, Any]]]:
        result, trace = await self.inner.run_tool_loop(
            system=system,
            user=user,
            tools=tools,
            response_model=response_model,
            budget=budget,
            phase=phase,
            response_schema=response_schema,
        )
        await self._replay(phase, _render(result))
        return result, trace

    async def _replay(self, phase: str, text: str) -> None:
        self.emit_open(phase, simulated=True)
        for start in range(0, len(text), self.chunk_chars):
            self.emit_text(
                phase, text[start : start + self.chunk_chars], simulated=True
            )
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
        self.emit_close(phase, simulated=True)


def _render(value: Any) -> str:
    dump = getattr(value, "model_dump_json", None)
    if callable(dump):
        return dump(indent=2)
    return str(value)


def make_llm(mode: str, tracer: TraceSink | None = None) -> LLMClient:
    """`live` needs a key and streams; `scripted` runs offline and replays."""

    if mode == "live":
        return StreamingAnthropicLLM(tracer)
    from reagents.llm.scripted import ScriptedLLM

    return ReplayLLM(ScriptedLLM.for_toy_pathway(), tracer)


__all__ = [
    "ReplayLLM",
    "StreamingAnthropicLLM",
    "lane_for_phase",
    "make_llm",
]
