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


class AnthropicLLM:
    def __init__(self, model: str = "claude-sonnet-4-20250514") -> None:
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
            max_tokens=4096,
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
                for block in message.content:
                    if getattr(block, "type", None) != "tool_use":
                        continue
                    try:
                        result = tools.call(block.name, **dict(block.input))
                    except UnboundToolError:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        result = {"error": str(exc)}
                    trace.append({"tool": block.name, "input": dict(block.input), "result": result})
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
    for spec in tools.specs():
        converted.append(
            {
                "name": spec.id,
                "description": spec.description,
                "input_schema": spec.parameters_schema or {"type": "object", "properties": {}},
            }
        )
    return converted


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
