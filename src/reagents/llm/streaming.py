"""`AnthropicLLM`, with the planning call streamed instead of awaited whole.

WHY THIS IS IN `reagents` AND NOT IN A UI PACKAGE. It started in the web
interface, which was the only consumer that wanted token-level output. Then GOD
moved into its own Modal sandbox -- and that image ships `reagents`, `demigod`,
`godbox` and `broker`, not the UI. So a sandboxed GOD fell back to the
non-streaming client and the stream went silent for the entire planning phase:
57 seconds between two phase lines with nothing rendered, which is
indistinguishable from a hang and was duly reported as one.

The fix is placement, not more code. Streaming is a property of the Anthropic
client, so it belongs next to the Anthropic client, where every caller can have
it -- laptop, server, or sandbox.

WHAT IS STREAMED, AND WHAT IS NOT. `complete()` only. That is GOD's own
thinking: inventing domains, projecting the problem, integrating artifacts, and
the text a watcher actually wants to see arrive. `run_tool_loop` is delegated
unchanged, because a demigod's visible progress is its tool calls, which the
registry and the broker already trace.

The refusal handling is INHERITED rather than reimplemented. A refusal
truncates the response, so parsing one reports malformed JSON and hides the
cause; that logic lives in `anthropic_client` and this class calls it.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel

from reagents.contracts import Budget
from reagents.llm.client import LLMError
from reagents.tools.registry import BoundToolPack
from reagents.tracing import GOD_LANE, NullTracer, TraceSink, demigod_lane

T = TypeVar("T", bound=BaseModel)


def lane_for_phase(phase: str) -> str:
    """`demigod:catalytic_dag` belongs to that demigod's lane; the rest is GOD's."""

    if phase.startswith("demigod:"):
        return demigod_lane(phase.removeprefix("demigod:"))
    return GOD_LANE


class StreamingAnthropicLLM:
    """Satisfies `LLMClient`, and emits the model's text as it is produced.

    The extra events are `TEXT OPEN` / `TEXT` / `TEXT CLOSE` on the lane the
    phase belongs to. A sink that does not know them renders them as ordinary
    lines; `TerminalTracer` ignores them entirely, which is why the terminal
    view is unchanged by this class existing.
    """

    def __init__(
        self,
        tracer: TraceSink | None = None,
        model: str | None = None,
    ) -> None:
        from reagents.llm.anthropic_client import DEFAULT_MODEL, AnthropicLLM

        self.inner = AnthropicLLM(model or DEFAULT_MODEL)
        self.tracer = tracer or NullTracer()

    def set_tracer(self, tracer: TraceSink) -> None:
        self.tracer = tracer

    # -- text events -----------------------------------------------------
    def _emit(self, kind: str, phase: str, message: str) -> None:
        self.tracer.emit(
            lane_for_phase(phase),
            kind,
            message,
            data={"phase": phase, "simulated": False},
        )

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

        schema = json.dumps(response_model.model_json_schema())
        system_full = (
            f"{system}\n\nRespond with JSON only matching this schema:\n{schema}"
        )

        last_refusal: str | None = None
        for _ in range(REFUSAL_RETRIES + 1):
            self._emit("TEXT OPEN", phase, phase)
            async with self.inner._client.messages.stream(
                model=self.inner.model,
                max_tokens=PLANNING_MAX_TOKENS,
                system=system_full,
                messages=[{"role": "user", "content": user}],
            ) as stream:
                async for delta in stream.text_stream:
                    if delta:
                        self._emit("TEXT", phase, delta)
                message = await stream.get_final_message()
            self._emit("TEXT CLOSE", phase, phase)

            if getattr(message, "stop_reason", None) != "refusal":
                return _parse_model(_text_blocks(message), response_model, message)
            last_refusal = _describe_refusal(message)

        raise LLMError(
            f"{response_model.__name__}: the model refused "
            f"{REFUSAL_RETRIES + 1} times ({last_refusal}). A refusal truncates "
            f"the response, so this surfaces as incomplete JSON if unhandled."
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
        # Not streamed, but still shown: the loop's answer is worth reading even
        # though it arrived in one piece.
        dump = getattr(result, "model_dump_json", None)
        rendered = dump(indent=2) if callable(dump) else str(result)
        self._emit("TEXT OPEN", phase, phase)
        self._emit("TEXT", phase, rendered)
        self._emit("TEXT CLOSE", phase, phase)
        return result, trace


__all__ = ["StreamingAnthropicLLM", "lane_for_phase"]
