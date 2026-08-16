"""Trace records in, browser-renderable events out.

`reagents.tracing` emits `(lane, kind, message, data)`. That is the right shape
for a terminal and the wrong shape for a tree: a lane is a string, and the UI
needs a node identity, a parent, and a coarse category it can style. This module
is the only place that translation happens, so the frontend never has to know
that `DEMI:catalytic_dag` is a lane rather than an id.

Redaction is repeated here rather than imported from `tracing`: the terminal
renderer redacts on its way to a TTY, and events leaving this process reach a
browser, which is a wider audience than a terminal. Two sinks, two independent
guarantees -- the SSE stream must not stop being redacted because a private
helper in another module was refactored.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

GOD_NODE = "god"
DEMI_PREFIX = "demi:"

_REDACTED_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "password",
    "refresh_token",
    "secret",
    "token",
}

# Coarse categories. The frontend styles on these; `kind` stays available for
# anything that needs the exact event.
GROUP_PHASE = "phase"  # GOD moving between stages of the loop
GROUP_TEXT = "text"  # model output arriving token by token
GROUP_TOOL = "tool"  # a capability was invoked, refused, or returned
GROUP_SPAWN = "spawn"  # a DEMI_GOD came into existence
GROUP_ARTIFACT = "artifact"  # a validated finding
GROUP_STATUS = "status"  # anything else worth a line

_GOD_PHASES = {
    "start": "analyzing",
    "catalog": "analyzing",
    "plan": "planning",
    "transform": "projecting",
    "sealed": "projecting",
    "spawn": "spawning",
    "collect": "collecting",
    "integrate": "integrating",
    "done": "done",
}

_GROUPS = {
    "tool_call": GROUP_TOOL,
    "tool_result": GROUP_TOOL,
    "tool_deny": GROUP_TOOL,
    "tool_error": GROUP_TOOL,
    "text": GROUP_TEXT,
    "artifact": GROUP_ARTIFACT,
    "reason": GROUP_ARTIFACT,
    "sealed": GROUP_SPAWN,
    "spawn": GROUP_SPAWN,
}


def node_id_for_lane(lane: str) -> str:
    """`GOD` -> `god`; `DEMI:catalytic_dag` -> `demi:catalytic_dag`."""

    if lane.startswith("DEMI:"):
        return DEMI_PREFIX + lane.removeprefix("DEMI:")
    return GOD_NODE


def normalize_kind(kind: str) -> str:
    return kind.strip().lower().replace(" ", "_")


def group_for(node: str, kind: str) -> str:
    if kind in _GROUPS:
        return _GROUPS[kind]
    if node == GOD_NODE and kind in _GOD_PHASES:
        return GROUP_PHASE
    if kind in {"start", "done", "fail", "failure", "reject", "scope", "model"}:
        return GROUP_STATUS
    return GROUP_STATUS


@dataclass(frozen=True)
class UiEvent:
    """One thing that happened, addressed to a node in the run tree."""

    seq: int
    t: float
    node: str
    lane: str
    kind: str
    group: str
    message: str
    data: Any | None = None
    phase: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data"] = jsonable(self.data)
        return payload


def build_event(
    *,
    seq: int,
    elapsed_s: float,
    lane: str,
    kind: str,
    message: str,
    data: Any | None,
) -> UiEvent:
    node = node_id_for_lane(lane)
    normalized = normalize_kind(kind)
    return UiEvent(
        seq=seq,
        t=round(elapsed_s, 3),
        node=node,
        lane=lane,
        kind=normalized,
        group=group_for(node, normalized),
        message=message,
        data=redact(data),
        phase=_GOD_PHASES.get(normalized) if node == GOD_NODE else None,
    )


@dataclass
class RunSnapshot:
    """Enough state to render a run that was already in flight when you opened it.

    A late subscriber gets this plus the full event log, so the first paint of a
    reconnecting tab is identical to the tab that watched from the start.
    """

    run_id: str
    status: str = "running"
    question: str = ""
    mode: str = "scripted"
    execution: str = "inprocess"
    sandbox_id: str | None = None
    started_at: float = 0.0
    finished_at: float | None = None
    error: str | None = None
    solution: dict[str, Any] | None = None
    domains: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return jsonable(asdict(self))


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "<redacted>" if str(key).lower() in _REDACTED_KEYS else redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [redact(item) for item in value]
    return value


def jsonable(value: Any) -> Any:
    """Best-effort conversion to something `json.dumps` accepts without a hook.

    Trace payloads carry whatever a tool returned, which includes pydantic
    models, enums and numpy-ish scalars. Serializing with `default=str` would
    work but turns a dict-shaped result into a quoted blob the UI cannot walk,
    so the structure is preserved and only the leaves are stringified.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        # JSON has no NaN/Infinity; a strict browser parser rejects the frame.
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return jsonable(dump(mode="json"))
        except Exception:
            return str(value)
    return str(value)
