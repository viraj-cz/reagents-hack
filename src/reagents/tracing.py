"""User-facing orchestration progress plus internal structured events."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol, TextIO

GOD_LANE = "GOD"
_REDACTED_KEYS = {
    "authorization",
    "api_key",
    "apikey",
    "password",
    "secret",
    "token",
    "access_token",
    "refresh_token",
}


def demigod_lane(name: str) -> str:
    return f"DEMI:{name}"


class TraceSink(Protocol):
    """Synchronous event sink so it can instrument sync and async tools alike."""

    def emit(
        self,
        lane: str,
        kind: str,
        message: str,
        *,
        data: Any | None = None,
    ) -> None: ...


class NullTracer:
    def emit(
        self,
        lane: str,
        kind: str,
        message: str,
        *,
        data: Any | None = None,
    ) -> None:
        del lane, kind, message, data


@dataclass(frozen=True)
class TraceRecord:
    sequence: int
    elapsed_s: float
    lane: str
    kind: str
    message: str
    data: Any | None = None


class RecordingTracer:
    """In-memory sink for tests and alternate user interfaces."""

    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self.records: list[TraceRecord] = []

    def emit(
        self,
        lane: str,
        kind: str,
        message: str,
        *,
        data: Any | None = None,
    ) -> None:
        self.records.append(
            TraceRecord(
                sequence=len(self.records) + 1,
                elapsed_s=time.monotonic() - self.started_at,
                lane=lane,
                kind=kind,
                message=message,
                data=data,
            )
        )


class TerminalTracer:
    """Render a user-facing God and subagent progress narrative.

    Events are translated into concise status updates rather than printed as
    process logs. Subagent output is clearly lane-labelled because independent
    analyses may finish in any order. Model chain-of-thought is never emitted;
    ``REASON`` is the bounded justification attached to the returned artifact.
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        color: bool | None = None,
        max_detail_chars: int = 180,
        show_subagents: bool = True,
    ) -> None:
        self.stream = stream or sys.stdout
        self.started_at = time.monotonic()
        self.max_detail_chars = max_detail_chars
        self.show_subagents = show_subagents
        self._lock = threading.Lock()
        if color is None:
            color = bool(getattr(self.stream, "isatty", lambda: False)())
            color = color and "NO_COLOR" not in os.environ
        self.color = color

    def emit(
        self,
        lane: str,
        kind: str,
        message: str,
        *,
        data: Any | None = None,
    ) -> None:
        if lane == GOD_LANE:
            rendered = self._render_god(kind.upper(), message, _redact(data))
        elif self.show_subagents and lane.startswith("DEMI:"):
            rendered = self._render_subagent(
                lane.removeprefix("DEMI:"),
                kind.upper(),
                message,
                _redact(data),
            )
        else:
            return
        with self._lock:
            for style, line in rendered:
                if self.color:
                    line = _paint(style, line)
                print(line, file=self.stream, flush=True)

    def _render_god(
        self,
        kind: str,
        message: str,
        data: Any | None,
    ) -> list[tuple[str, str]]:
        detail = _truncate(_one_line(message), self.max_detail_chars)

        if kind == "START":
            return [("heading", "◆ God is analyzing the problem"), ("normal", f"  {detail}")]
        if kind == "CATALOG":
            return [("muted", f"  Checking capabilities — {detail}")]
        if kind == "PLAN" and not isinstance(data, list):
            return [("heading", "\n◇ Choosing useful representations"), ("muted", f"  {detail}")]
        if kind == "PLAN" and isinstance(data, list):
            lines: list[tuple[str, str]] = [
                ("normal", f"  Selected {len(data)} complete alternative representations:")
            ]
            for item in data:
                if not isinstance(item, dict):
                    continue
                tools = ", ".join(str(tool) for tool in item.get("tools", [])) or "no tools"
                lines.append(
                    (
                        "normal",
                        f"    • {item.get('name', 'unnamed')} — {item.get('axis', 'unspecified')}; "
                        f"tools: {tools}",
                    )
                )
            return lines
        if kind == "TRANSFORM":
            return []
        if kind == "SEALED":
            tools = [] if not isinstance(data, dict) else data.get("tools", [])
            axis = "" if not isinstance(data, dict) else str(data.get("axis", ""))
            suffix = f" — {axis}" if axis else ""
            if tools:
                suffix += f"; scoped to {', '.join(str(tool) for tool in tools)}"
            name = detail.removesuffix(" ready")
            return [("success", f"  ✓ Prepared {name}{suffix}")]
        if kind == "SPAWN":
            return [("heading", "\n◇ Running scoped analyses"), ("muted", f"  {detail}")]
        if kind == "COLLECT":
            if isinstance(data, dict):
                name = str(data.get("domain_name") or detail)
                if self.show_subagents:
                    return [("success", f"  ✓ God accepted {name}'s artifact")]
                confidence = data.get("confidence")
                calls = data.get("tool_calls")
                status = f"  ✓ {name} finished"
                if isinstance(confidence, (int, float)):
                    status += f" — {confidence:.0%} confidence"
                if isinstance(calls, int):
                    status += f", {calls} tool calls"
                lines = [("success", status)]
                conclusion = data.get("conclusion")
                if conclusion:
                    lines.append(
                        (
                            "muted",
                            f"    {summarize(conclusion, self.max_detail_chars - 4)}",
                        )
                    )
                return lines
            return [("success", f"  ✓ {detail}")]
        if kind in {"REJECT", "FAILURE"}:
            return [("warning", f"  ! {detail}")]
        if kind == "INTEGRATE":
            return [("heading", "\n◇ Comparing complete candidates"), ("muted", f"  {detail}")]
        if kind == "VERIFY":
            if data is None:
                return [
                    ("heading", "\n◇ Validating in the original domain"),
                    ("muted", f"  {detail}"),
                ]
            passed = isinstance(data, dict) and data.get("passed") is True
            return [("success" if passed else "warning", f"  {'✓' if passed else '!'} {detail}")]
        if kind == "DONE":
            confidence = None if not isinstance(data, dict) else data.get("confidence")
            artifacts = None if not isinstance(data, dict) else data.get("artifacts")
            failures = None if not isinstance(data, dict) else data.get("failures")
            summary = "  The integrated answer is ready"
            if isinstance(confidence, (int, float)):
                summary += f" at {confidence:.0%} confidence"
            facts = []
            if isinstance(artifacts, int):
                facts.append(f"{artifacts} artifacts combined")
            if isinstance(failures, int):
                facts.append(f"{failures} failures")
            lines = [("success_heading", "\n◆ Analysis complete"), ("normal", summary)]
            if facts:
                lines.append(("muted", f"  {' · '.join(facts)}"))
            return lines

        suffix = f" · {_compact(data, self.max_detail_chars)}" if data is not None else ""
        return [("normal", f"  {detail}{suffix}")]

    def _render_subagent(
        self,
        name: str,
        kind: str,
        message: str,
        data: Any | None,
    ) -> list[tuple[str, str]]:
        agent = summarize(name, 48)
        detail = summarize(message, self.max_detail_chars)
        prefix = f"  │ {agent} · "

        if kind == "START":
            return [("agent_heading", f"\n  ┌─ Subagent {agent} started")]
        if kind == "SCOPE":
            lines = [("normal", f"{prefix}Scope — {detail}")]
            if isinstance(data, dict):
                tools = data.get("tools", [])
                if tools:
                    lines.append(
                        ("muted", f"{prefix}Allowed tools — {', '.join(map(str, tools))}")
                    )
                max_steps = data.get("max_steps")
                max_calls = data.get("max_tool_calls")
                if isinstance(max_steps, int) or isinstance(max_calls, int):
                    limits = []
                    if isinstance(max_steps, int):
                        limits.append(f"{max_steps} reasoning steps")
                    if isinstance(max_calls, int):
                        limits.append(f"{max_calls} tool calls")
                    lines.append(("muted", f"{prefix}Budget — {', '.join(limits)}"))
            return lines
        if kind == "MODEL":
            return [("muted", f"{prefix}Reasoning in the sealed representation")]
        if kind == "TOOL CALL":
            argument_names: list[str] = []
            if isinstance(data, dict) and isinstance(data.get("arguments"), dict):
                argument_names = sorted(map(str, data["arguments"]))
            suffix = f" (inputs: {', '.join(argument_names)})" if argument_names else ""
            return [("tool", f"{prefix}→ Using {detail}{suffix}")]
        if kind == "TOOL RESULT":
            result = data.get("result") if isinstance(data, dict) else data
            return [
                (
                    "muted",
                    f"{prefix}← {detail} returned {_describe_result(result, self.max_detail_chars)}",
                )
            ]
        if kind in {"TOOL DENY", "TOOL ERROR"}:
            return [("warning", f"{prefix}! {detail}")]
        if kind == "REASON":
            return [("normal", f"{prefix}Reasoning summary — {detail}")]
        if kind == "ARTIFACT":
            return [("success", f"{prefix}Finding — {detail}")]
        if kind == "FAIL":
            return [("warning", f"  └─ Subagent {agent} failed — {detail}")]
        if kind == "DONE":
            return [("success", f"  └─ Subagent {agent} complete — {detail}")]

        return []


def summarize(value: Any, max_chars: int = 180) -> str:
    """Public bounded formatter used by lifecycle emitters."""

    return _truncate(_one_line(str(value)), max_chars)


def _compact(value: Any, max_chars: int) -> str:
    safe = _redact(value)
    try:
        rendered = json.dumps(safe, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        rendered = str(safe)
    return _truncate(_one_line(rendered), max_chars)


def _describe_result(value: Any, max_chars: int) -> str:
    """Describe a tool result without dumping potentially large tool output."""

    if isinstance(value, dict):
        keys = sorted(map(str, value))
        if not keys:
            return "an empty object"
        return _truncate(f"fields: {', '.join(keys)}", max_chars)
    if isinstance(value, (list, tuple)):
        return f"{len(value)} item{'s' if len(value) != 1 else ''}"
    if value is None:
        return "no payload"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    return _truncate(f"a {type(value).__name__} result", max_chars)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "<redacted>"
                if str(key).lower() in _REDACTED_KEYS
                else _redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def _one_line(value: str) -> str:
    return " ".join(value.split())


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 1:
        return "…"[:max_chars]
    return value[: max_chars - 1] + "…"


def _paint(style: str, line: str) -> str:
    colors = {
        "heading": "\033[1;36m",
        "success_heading": "\033[1;32m",
        "success": "\033[32m",
        "agent_heading": "\033[1;35m",
        "tool": "\033[36m",
        "warning": "\033[33m",
        "muted": "\033[2m",
        "normal": "",
    }
    prefix = colors.get(style, "")
    return f"{prefix}{line}\033[0m" if prefix else line
