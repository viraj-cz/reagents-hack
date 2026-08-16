"""LLM clients that let the UI watch text arrive, not just appear.

The orchestrator's own trace says *what stage GOD is in*. It deliberately never
carries model output, because its other consumer is a terminal and its contract
is "no chain-of-thought, bounded lines". A browser can afford the whole stream,
so these wrappers add one extra event kind -- `TEXT` -- carrying the model's
output as it is produced.

Two paths, and the difference between them is visible in the UI rather than
hidden:

* `StreamingAnthropicLLM` streams genuinely, from the Anthropic API.
* `ReplayLLM` wraps a non-streaming client (in practice `ScriptedLLM`) and
  replays its already-complete answer in chunks. Every event it emits is
  tagged `simulated: true` and the frontend labels the run REPLAY, because a
  fake typing animation presented as live inference is a lie about latency and
  about where the answer came from.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, TypeVar

from pydantic import BaseModel

from reagents.contracts import Budget
from reagents.llm.client import LLMClient
from reagents.tools.registry import BoundToolPack
from reagents.tracing import GOD_LANE, NullTracer, TraceSink, demigod_lane

T = TypeVar("T", bound=BaseModel)

# Big enough that a long answer does not become thousands of SSE frames, small
# enough that the text visibly flows rather than landing in paragraphs.
REPLAY_CHUNK_CHARS = 18
REPLAY_CHUNK_DELAY_S = 0.012


def lane_for_phase(phase: str) -> str:
    """`demigod:catalytic_dag` belongs to that demigod's lane; the rest is GOD's."""

    if phase.startswith("demigod:"):
        return demigod_lane(phase.removeprefix("demigod:"))
    return GOD_LANE


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


class StreamingAnthropicLLM:
    """`AnthropicLLM` with the planning call streamed instead of awaited whole.

    Only `complete()` is overridden. That is where GOD does its own thinking --
    inventing domains, projecting the problem, integrating the artifacts -- and
    it is the text a user actually wants to watch. `run_tool_loop` is inherited
    unchanged; a demigod's visible progress is its tool calls, which the
    registry already traces.
    """

    def __init__(
        self, tracer: TraceSink | None = None, model: str | None = None
    ) -> None:
        from reagents.llm.anthropic_client import DEFAULT_MODEL, AnthropicLLM

        self.inner = AnthropicLLM(model or DEFAULT_MODEL)
        self.emitter = _TextEmitter(tracer)

    def set_tracer(self, tracer: TraceSink) -> None:
        self.emitter = _TextEmitter(tracer)

    async def complete(
        self,
        *,
        system: str,
        user: str,
        response_model: type[T],
        phase: str = "",
    ) -> T:
        from reagents.llm.anthropic_client import (
            PLANNING_MAX_TOKENS,
            REFUSAL_RETRIES,
            _describe_refusal,
            _parse_model,
            _text_blocks,
        )
        from reagents.llm.client import LLMError

        schema = json.dumps(response_model.model_json_schema())
        system_full = (
            f"{system}\n\nRespond with JSON only matching this schema:\n{schema}"
        )

        last_refusal: str | None = None
        for _ in range(REFUSAL_RETRIES + 1):
            self.emitter.emit_open(phase, simulated=False)
            async with self.inner._client.messages.stream(
                model=self.inner.model,
                max_tokens=PLANNING_MAX_TOKENS,
                system=system_full,
                messages=[{"role": "user", "content": user}],
            ) as stream:
                async for delta in stream.text_stream:
                    self.emitter.emit_text(phase, delta, simulated=False)
                message = await stream.get_final_message()
            self.emitter.emit_close(phase, simulated=False)

            if getattr(message, "stop_reason", None) != "refusal":
                return _parse_model(_text_blocks(message), response_model, message)
            # Same false-positive handling as the non-streaming client: a
            # refusal truncates the JSON, so parsing it would report a
            # malformed response and hide the real cause.
            last_refusal = _describe_refusal(message)

        raise LLMError(
            f"{response_model.__name__}: the model refused "
            f"{REFUSAL_RETRIES + 1} times ({last_refusal})."
        )

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
        self.emitter.emit_open(phase, simulated=False)
        self.emitter.emit_text(phase, _render(result), simulated=False)
        self.emitter.emit_close(phase, simulated=False)
        return result, trace


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
