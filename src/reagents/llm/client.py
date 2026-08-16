"""Thin LLM interface. God and demigods share the client, never a conversation."""

from __future__ import annotations

import os
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from reagents.contracts import Budget
from reagents.tools.registry import BoundToolPack

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    async def complete(
        self,
        *,
        system: str,
        user: str,
        response_model: type[T],
        phase: str = "",
    ) -> T: ...

    async def run_tool_loop(
        self,
        *,
        system: str,
        user: str,
        tools: BoundToolPack,
        response_model: type[T],
        budget: Budget,
        phase: str = "",
    ) -> tuple[T, list[dict[str, Any]]]: ...


def make_llm() -> LLMClient:
    """Anthropic when ANTHROPIC_API_KEY is set; otherwise the scripted toy client."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        from reagents.llm.anthropic_client import AnthropicLLM

        return AnthropicLLM()
    from reagents.llm.scripted import ScriptedLLM

    return ScriptedLLM.for_toy_pathway()
