"""Capability registry, leases, and execution broker.

God can inspect the complete catalog, but a demigod receives only a BoundToolPack
backed by an immutable lease. Executors may be local callables or lazy async
adapters such as remote MCP and container tools.
"""

from __future__ import annotations

import inspect
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from reagents.contracts import (
    Budget,
    CapabilityLease,
    RiskTier,
    ToolAccess,
    ToolProvider,
    ToolSpec,
)


class UnknownToolError(KeyError):
    """God asked to bind a tool that is not in the registry."""


class UnboundToolError(PermissionError):
    """A demigod requested a tool that is not in its envelope."""


class ToolPolicyError(PermissionError):
    """A bound tool call exceeded its lease or side-effect policy."""


class ToolExecutionError(RuntimeError):
    """A registered tool could not be executed by the selected runtime."""


AsyncExecutor = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass(frozen=True)
class Tool:
    id: str
    description: str
    parameters_schema: dict[str, Any]
    fn: Callable[..., Any] | None = None
    executor: AsyncExecutor | None = None
    output_schema: dict[str, Any] = field(default_factory=dict)
    namespace: str = "generic"
    provider: ToolProvider = ToolProvider.LOCAL
    access: ToolAccess = ToolAccess.COMPUTE
    risk_tier: RiskTier = RiskTier.LOW
    side_effects: tuple[str, ...] = ()
    cost_class: str = "free"
    latency_class: str = "interactive"
    defer_loading: bool = False

    def __post_init__(self) -> None:
        if self.fn is None and self.executor is None:
            raise ValueError(f"tool {self.id!r} needs fn or executor")
        if self.fn is not None and self.executor is not None:
            raise ValueError(f"tool {self.id!r} cannot have both fn and executor")

    def spec(self) -> ToolSpec:
        return ToolSpec(
            id=self.id,
            description=self.description,
            parameters_schema=self.parameters_schema,
            output_schema=self.output_schema,
            namespace=self.namespace,
            provider=self.provider,
            access=self.access,
            risk_tier=self.risk_tier,
            side_effects=list(self.side_effects),
            cost_class=self.cost_class,
            latency_class=self.latency_class,
            defer_loading=self.defer_loading,
        )

    def call_sync(self, **kwargs: Any) -> Any:
        if self.fn is None:
            raise ToolExecutionError(
                f"tool {self.id!r} is lazy/async; execute it with BoundToolPack.acall"
            )
        result = self.fn(**kwargs)
        if inspect.isawaitable(result):
            raise ToolExecutionError(
                f"tool {self.id!r} returned an awaitable from its sync adapter"
            )
        return result

    async def call_async(self, **kwargs: Any) -> Any:
        if self.executor is not None:
            return await self.executor(dict(kwargs))
        if self.fn is None:  # pragma: no cover - guarded by __post_init__
            raise ToolExecutionError(f"tool {self.id!r} has no executor")
        result = self.fn(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


@dataclass
class ToolBroker:
    """Enforce a lease at the only path from a demigod to an executor."""

    tools: dict[str, Tool]
    lease: CapabilityLease
    calls: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def _authorize(self, tool_id: str) -> Tool:
        if tool_id not in self.tools or tool_id not in self.lease.tool_ids:
            raise UnboundToolError(
                f"tool {tool_id!r} is not bound in this envelope; "
                f"allowed={sorted(self.lease.tool_ids)}"
            )
        if self.calls >= self.lease.max_calls:
            raise ToolPolicyError(
                f"lease {self.lease.lease_id!r} exhausted its {self.lease.max_calls} calls"
            )
        if time.monotonic() - self.started_at > self.lease.wall_time_s:
            raise ToolPolicyError(f"lease {self.lease.lease_id!r} expired")
        tool = self.tools[tool_id]
        if tool.access == ToolAccess.WRITE and not self.lease.allow_write:
            raise ToolPolicyError(f"lease does not permit write tool {tool_id!r}")
        self.calls += 1
        return tool

    def call(self, tool_id: str, **kwargs: Any) -> Any:
        return self._authorize(tool_id).call_sync(**kwargs)

    async def acall(self, tool_id: str, **kwargs: Any) -> Any:
        return await self._authorize(tool_id).call_async(**kwargs)


@dataclass
class BoundToolPack:
    """Callables a single demigod may invoke. Missing ids are a hard error."""

    _tools: dict[str, Tool] = field(default_factory=dict)
    _broker: ToolBroker | None = None

    def ids(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [self._tools[i].spec() for i in self._tools]

    def call(self, tool_id: str, **kwargs: Any) -> Any:
        if self._broker is None:  # pragma: no cover - packs come from ToolRegistry.bind
            raise ToolExecutionError("tool pack has no broker")
        return self._broker.call(tool_id, **kwargs)

    async def acall(self, tool_id: str, **kwargs: Any) -> Any:
        if self._broker is None:  # pragma: no cover - packs come from ToolRegistry.bind
            raise ToolExecutionError("tool pack has no broker")
        return await self._broker.acall(tool_id, **kwargs)

    @property
    def lease(self) -> CapabilityLease:
        if self._broker is None:  # pragma: no cover
            raise ToolExecutionError("tool pack has no broker")
        return self._broker.lease

    @property
    def calls_used(self) -> int:
        return self._broker.calls if self._broker else 0


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.id in self._tools:
            raise ValueError(f"duplicate tool id: {tool.id}")
        self._tools[tool.id] = tool

    def get(self, tool_id: str) -> Tool:
        try:
            return self._tools[tool_id]
        except KeyError as exc:
            raise UnknownToolError(tool_id) from exc

    def ids(self) -> list[str]:
        return sorted(self._tools)

    def specs(self, tool_ids: list[str] | None = None) -> list[ToolSpec]:
        ids = tool_ids if tool_ids is not None else self.ids()
        return [self.get(i).spec() for i in ids]

    def mint_lease(
        self,
        tool_ids: list[str],
        *,
        subject_id: str,
        budget: Budget | None = None,
        allow_write: bool = False,
    ) -> CapabilityLease:
        budget = budget or Budget()
        return CapabilityLease(
            lease_id=f"lease_{uuid.uuid4().hex}",
            subject_id=subject_id,
            tool_ids=list(tool_ids),
            max_calls=budget.max_tool_calls,
            wall_time_s=budget.wall_time_s,
            allow_write=allow_write,
        )

    def bind(
        self,
        tool_ids: list[str],
        *,
        lease: CapabilityLease | None = None,
        subject_id: str = "unscoped",
        budget: Budget | None = None,
    ) -> BoundToolPack:
        if not tool_ids:
            raise ValueError("cannot bind an empty tool pack")
        tools: dict[str, Tool] = {}
        for tool_id in tool_ids:
            tools[tool_id] = self.get(tool_id)
        effective_lease = lease or self.mint_lease(
            tool_ids,
            subject_id=subject_id,
            budget=budget,
        )
        if set(effective_lease.tool_ids) != set(tool_ids):
            raise ToolPolicyError(
                "lease tools must exactly match the requested bound tool pack"
            )
        return BoundToolPack(
            _tools=tools,
            _broker=ToolBroker(tools=tools, lease=effective_lease),
        )


def tool_jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def default_registry() -> ToolRegistry:
    from reagents.tools import builtins as builtin_impls

    registry = ToolRegistry()
    for tool in builtin_impls.all_tools():
        registry.register(tool)
    return registry
