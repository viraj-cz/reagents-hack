"""Tiering: which tools run in the router replica, and which get their own box.

THE DECISION THIS ENCODES. A single broker process serving every tool is wrong
in two directions at once. Run everything inline and one A100-class tool pins a
replica that is mostly serving 3ms graph rewrites; give every tool its own
container and a `find_cycles` call pays a cold start to add two integers.

So: cheap, pure-Python, dependency-free tools run INLINE in the router replica.
Heavy, GPU-bound, or dependency-conflicting tools are dispatched to a separate
Modal Function whose image and `gpu=` are its own. `ToolSpec.provider` already
models exactly this distinction -- LOCAL means "a callable in this process",
MCP and CONTAINER mean "something else executes it" -- so the tier is read off
the registry rather than configured twice.

WHY NOT A SANDBOX PER CALL. `modal.Sandbox.create` costs seconds even warm, and
a demigod makes tens of calls. A Modal Function scales to zero, keeps warm
replicas between calls within a run, and lets the GPU tool hold the GPU while
the router holds nothing.

HOW IT PLUGS IN. `prepare()` returns a `Tool`, not a result. The returned tool
is handed to `reagents.tools.registry.ToolBroker`, which stays the one and only
code path from a request to an executor -- so tiering cannot accidentally become
a way around the lease.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any

from reagents.contracts import ToolProvider
from reagents.tools.registry import Tool

RemoteCall = Callable[[str, dict[str, Any]], Awaitable[Any]]
"""`(tool_id, arguments) -> result`. Supplied by `broker.service`, where it is a
`modal.Function.remote.aio` call. Kept as a bare callable so this module -- and
therefore the whole request path -- imports no Modal and stays testable."""

INLINE_BY_DEFAULT = frozenset({ToolProvider.LOCAL})
"""LOCAL tools are in-process callables in the broker's own image. Anything else
already implies a hop, so making it a Modal hop costs nothing extra."""


@dataclass(frozen=True)
class DispatchPolicy:
    """Where each tool executes. One instance per broker container."""

    remote: RemoteCall | None = None
    """None means 'inline only'. A broker deployed without heavy executors is a
    legitimate configuration -- it just fails loudly if a CONTAINER tool is
    called, rather than silently running an untested inline path."""

    inline_providers: frozenset[ToolProvider] = INLINE_BY_DEFAULT

    force_remote: frozenset[str] = field(default_factory=frozenset)
    """Tool ids pushed out regardless of provider. The escape hatch for a LOCAL
    tool that turns out to be heavy -- a 40-second solve should not occupy the
    replica that other demigods are queued behind, and reclassifying its
    provider would be a lie about how it is implemented."""

    force_inline: frozenset[str] = field(default_factory=frozenset)
    """The other escape hatch, for a nominally-remote tool during local dev."""

    def is_inline(self, tool: Tool) -> bool:
        if tool.id in self.force_inline:
            return True
        if tool.id in self.force_remote:
            return False
        return tool.provider in self.inline_providers

    def prepare(self, tool: Tool) -> Tool:
        """The tool as the lease broker should execute it.

        Inline tools are returned untouched. Remote tools come back with their
        executor swapped for a call to the heavy Function -- same `Tool`, same
        id, same schema, same lease enforcement, different machine.
        """
        if self.is_inline(tool):
            return tool
        if self.remote is None:
            raise DispatchUnavailableError(
                f"tool {tool.id!r} is provider={tool.provider.value} and must run "
                f"outside the router, but this broker was configured without a "
                f"remote executor. Deploy broker.service with its executor "
                f"function, or add {tool.id!r} to force_inline."
            )
        remote = self.remote
        tool_id = tool.id

        async def _remote_executor(arguments: dict[str, Any]) -> Any:
            return await remote(tool_id, arguments)

        # `fn=None` is required, not decorative: Tool.__post_init__ rejects a
        # tool carrying both a callable and an executor.
        return replace(tool, fn=None, executor=_remote_executor)


class DispatchUnavailableError(RuntimeError):
    """A tool needs a tier this broker was not given."""


__all__ = [
    "INLINE_BY_DEFAULT",
    "DispatchPolicy",
    "DispatchUnavailableError",
    "RemoteCall",
]
