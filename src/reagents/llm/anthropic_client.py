"""Optional Anthropic backend. Imported only when ANTHROPIC_API_KEY is set."""

from __future__ import annotations

import json
import re
import sys
from typing import Any, TypeVar

from pydantic import BaseModel

from reagents.contracts import Budget
from reagents.llm.client import LLMError
from reagents.tools.registry import BoundToolPack, UnboundToolError

T = TypeVar("T", bound=BaseModel)


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

REFUSAL_RETRIES = 1
"""Extra attempts after a `stop_reason: "refusal"`.

Refusals here are classifier false positives on abstract mathematics -- the run
that motivated this was cut off mid-way through tank state-update equations. One
retry is cheap and usually enough; more than that means the prompt itself is
tripping a classifier, and retrying harder will not fix that.
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
        system_full = (
            f"{system}\n\nRespond with JSON only matching this schema:\n{schema}"
        )

        last_refusal: str | None = None
        for attempt in range(REFUSAL_RETRIES + 1):
            message = await self._client.messages.create(
                model=self.model,
                max_tokens=PLANNING_MAX_TOKENS,
                system=system_full,
                messages=[{"role": "user", "content": user}],
            )
            if getattr(message, "stop_reason", None) != "refusal":
                return _parse_model(_text_blocks(message), response_model, message)

            # A safety classifier declined mid-generation. This is NOT a
            # malformed reply: the JSON is simply cut off wherever the
            # classifier fired, so parsing it reports a truncation whose real
            # cause is a refusal. Seen on a water-tank flow problem whose
            # content was abstract state-update equations -- i.e. a false
            # positive, and false positives are worth one retry.
            last_refusal = _describe_refusal(message)
            print(
                f"[llm] {response_model.__name__}: refusal on attempt "
                f"{attempt + 1}/{REFUSAL_RETRIES + 1} ({last_refusal})",
                file=sys.stderr,
                flush=True,
            )

        raise LLMError(
            f"{response_model.__name__}: the model refused "
            f"{REFUSAL_RETRIES + 1} times ({last_refusal}). A refusal truncates "
            f"the response, so this surfaces as incomplete JSON if unhandled. "
            f"If it persists, the prompt is triggering a classifier -- rephrase "
            f"it rather than retrying harder."
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
        del phase
        # An explicit schema wins. The caller uses it to nest a domain's
        # artifact shape inside `payload`, which is the difference between
        # showing the model one schema and showing it two that it has to guess
        # the relationship between.
        schema = json.dumps(response_schema or response_model.model_json_schema())
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        anthropic_tools = _to_anthropic_tools(tools)
        trace: list[dict[str, Any]] = []
        system_full = f"{system}\n\nWhen finished, respond with JSON only matching this schema:\n{schema}"

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
                aliases = {
                    _anthropic_tool_name(tool_id): tool_id for tool_id in tools.ids()
                }
                for block in message.content:
                    if getattr(block, "type", None) != "tool_use":
                        continue
                    try:
                        canonical_name = aliases.get(block.name, block.name)
                        result = await tools.acall(canonical_name, **dict(block.input))
                    except UnboundToolError:
                        raise
                    except Exception as exc:
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
            return _parse_model(_text_blocks(message), response_model, message), trace

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
                "input_schema": spec.parameters_schema
                or {"type": "object", "properties": {}},
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


def _extract_json_object(text: str) -> str | None:
    """First balanced top-level JSON object in `text`, or None.

    Replaces a greedy `\\{.*\\}` regex, which spans from the first `{` to the
    LAST `}` anywhere in the response. On a truncated reply that lands on a
    nested closing brace, the regex returns a fragment whose outer object never
    closes -- surfacing as `ValidationError: EOF while parsing an object`, which
    describes the symptom and hides the cause. Scanning for balance instead
    returns either a genuinely complete object or nothing, so truncation is
    reported as truncation.

    String-aware, so braces inside string values do not affect the depth count.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_model(text: str, response_model: type[T], message: Any = None) -> T:
    """Parse the model's reply, reporting WHY it failed when it does.

    `message` is optional only so the signature stays usable from tests; pass it
    wherever available. `stop_reason` is the single most useful field here --
    "max_tokens" means raise the budget, while "end_turn" on unparseable output
    means the model genuinely wrote something malformed. Without it, both look
    identical and the pydantic error points at neither.
    """
    try:
        return response_model.model_validate_json(text)
    except Exception:
        pass

    candidate = _extract_json_object(text)
    if candidate is None:
        stop = getattr(message, "stop_reason", None)
        if stop == "max_tokens":
            raise LLMError(
                f"{response_model.__name__}: response hit max_tokens before the "
                f"JSON object closed ({len(text)} chars). Raise max_tokens."
            ) from None
        raise LLMError(
            f"{response_model.__name__}: no complete JSON object in the response "
            f"(stop_reason={stop!r}, {len(text)} chars): {text[:300]}"
        ) from None

    try:
        return response_model.model_validate_json(candidate)
    except Exception as e:
        raise LLMError(
            f"{response_model.__name__}: extracted a complete JSON object but it "
            f"did not match the schema (stop_reason="
            f"{getattr(message, 'stop_reason', None)!r}): {e}"
        ) from e


def _describe_refusal(message: Any) -> str:
    """Human-readable summary of a refusal, including the category when present.

    `stop_details` is populated only for refusals and can still be None, so
    every access is guarded -- reading it unconditionally is its own crash.
    """
    details = getattr(message, "stop_details", None)
    if details is None:
        return "no stop_details"
    category = getattr(details, "category", None)
    explanation = getattr(details, "explanation", None)
    parts = [f"category={category!r}"]
    if explanation:
        parts.append(str(explanation)[:200])
    return "; ".join(parts)
