"""Optional Anthropic backend. Imported only when ANTHROPIC_API_KEY is set."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel

from reagents.contracts import Budget
from reagents.llm.client import LLMError
from reagents.tools.registry import BoundToolPack, UnboundToolError

T = TypeVar("T", bound=BaseModel)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


DEFAULT_MODEL = "claude-opus-4-8"
"""GOD's own loop: planner, transformer, integrator.

Was `claude-sonnet-4-20250514`, which is deprecated and now 404s:

    NotFoundError: Error code: 404 - {'type': 'not_found_error',
    'message': 'model: claude-sonnet-4-20250514'}

Use the bare alias, never a date-suffixed variant -- the suffixed forms are
snapshot IDs and guessing one 404s the same way. Note this model rejects
`temperature`/`top_p`/`top_k` and `thinking.budget_tokens` with a 400; this
client passes none of them, so no other change was needed.
"""

PLANNING_MAX_TOKENS = 16000
"""Was 4096. A planner turn emits N full DomainSpecs -- axes, invented
language, transform prompt, tool ids, and a JSON artifact_schema each -- and
truncating one mid-object surfaces as a JSON parse failure, not as a token
error. 16k is the non-streaming default; above that the SDK needs streaming."""


class AnthropicLLM:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError("install reagents[llm] to use Anthropic") from exc
        self._client = anthropic.AsyncAnthropic()
        self.model = model

    async def complete(
        self,
        *,
        system: str,
        user: str,
        response_model: type[T],
        phase: str = "",
    ) -> T:
        del phase
        schema = json.dumps(response_model.model_json_schema())
        message = await self._client.messages.create(
            model=self.model,
            max_tokens=PLANNING_MAX_TOKENS,
            system=f"{system}\n\nRespond with JSON only matching this schema:\n{schema}",
            messages=[{"role": "user", "content": user}],
        )
        return _parse_model(_text_blocks(message), response_model)

    async def run_tool_loop(
        self,
        *,
        system: str,
        user: str,
        tools: BoundToolPack,
        response_model: type[T],
        budget: Budget,
        phase: str = "",
    ) -> tuple[T, list[dict[str, Any]]]:
        del phase
        schema = json.dumps(response_model.model_json_schema())
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        anthropic_tools = _to_anthropic_tools(tools)
        trace: list[dict[str, Any]] = []
        system_full = (
            f"{system}\n\nWhen finished, respond with JSON only matching this schema:\n{schema}"
        )

        for _ in range(budget.max_steps):
            message = await self._client.messages.create(
                model=self.model,
                max_tokens=budget.max_tokens,
                system=system_full,
                messages=messages,
                tools=anthropic_tools,
            )
            if message.stop_reason == "tool_use":
                tool_results = []
                aliases = {_anthropic_tool_name(tool_id): tool_id for tool_id in tools.ids()}
                for block in message.content:
                    if getattr(block, "type", None) != "tool_use":
                        continue
                    try:
                        canonical_name = aliases.get(block.name, block.name)
                        result = await tools.acall(canonical_name, **dict(block.input))
                    except UnboundToolError:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        result = {"error": str(exc)}
                    trace.append(
                        {
                            "tool": canonical_name,
                            "input": dict(block.input),
                            "result": result,
                        }
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, default=str),
                        }
                    )
                messages.append({"role": "assistant", "content": message.content})
                messages.append({"role": "user", "content": tool_results})
                continue
            return _parse_model(_text_blocks(message), response_model), trace

        raise LLMError("demigod exhausted its step budget without an artifact")


def _to_anthropic_tools(tools: BoundToolPack) -> list[dict[str, Any]]:
    converted = []
    used_names: set[str] = set()
    for spec in tools.specs():
        transport_name = _anthropic_tool_name(spec.id)
        if transport_name in used_names:
            raise LLMError(
                f"tool IDs collide after Anthropic name normalization: {spec.id!r}"
            )
        used_names.add(transport_name)
        converted.append(
            {
                "name": transport_name,
                "description": spec.description,
                "input_schema": spec.parameters_schema or {"type": "object", "properties": {}},
            }
        )
    return converted


def _anthropic_tool_name(tool_id: str) -> str:
    """Map canonical capability IDs to Anthropic's portable tool-name alphabet."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "__", tool_id)
    if len(safe) > 64:
        raise LLMError(f"tool id is too long for Anthropic: {tool_id!r}")
    return safe


def _text_blocks(message: Any) -> str:
    parts = []
    for block in message.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


def _parse_model(text: str, response_model: type[T]) -> T:
    try:
        return response_model.model_validate_json(text)
    except Exception:
        match = _JSON_BLOCK.search(text)
        if not match:
            raise LLMError(f"no JSON in model response: {text[:200]}") from None
        return response_model.model_validate_json(match.group(0))
