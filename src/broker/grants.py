"""Durable lease state: what a lease may do, and what it has already done.

WHY THIS EXISTS AT ALL. `reagents.tools.registry.ToolBroker` already enforces a
lease correctly -- max_calls, wall time, write permission -- but it does so with
two in-memory fields (`calls`, `started_at`) on one object in one process. The
TOOLBOX_BROKER is an autoscaling Modal Function: a lease's third call can land
on a replica that has never seen its first two. So the counters have to move out
of the process, and this module is where they live.

Nothing here re-implements the policy. `ToolBroker` remains the only code that
decides whether a call is allowed; this store just holds the two numbers it
needs across replicas. See `broker/router.py`.

THE COUNTER IS THE AUDIT LOG. `calls_used` is the length of the trace, not a
separate integer. That is not a trick to save a field -- it means the number the
broker enforces against and the number it reports to GOD cannot disagree, and it
removes the read-modify-write that a distributed counter would otherwise need.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from reagents.contracts import CapabilityLease


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


class Grant(BaseModel):
    """One lease, published by GOD before a demigod is spawned.

    Written once, read many times, never mutated. The immutability is what makes
    the store safe to shard: replicas only ever read this, and everything that
    changes during a run lives in the append-only trace.
    """

    lease: CapabilityLease
    issued_at: float = Field(default_factory=time.time)
    label: str = Field(
        "", description="Human tag for logs, e.g. the domain name. Never authority."
    )

    @property
    def lease_id(self) -> str:
        return self.lease.lease_id

    @property
    def expires_at(self) -> float:
        return self.issued_at + self.lease.wall_time_s

    def expires_in(self, now: float | None = None) -> float:
        return self.expires_at - (now if now is not None else time.time())


class TraceEntry(BaseModel):
    """One row of `DemiGodResult.tool_trace`, authored by the broker.

    `{tool, input, result}` is the shape `demigod.result` documents; the rest is
    provenance a consumer can ignore. Authored HERE rather than by the agent
    because the agent is the thing being audited -- a self-reported trace can
    omit the call whose result it disliked.
    """

    call_id: str = Field(default_factory=new_call_id)
    tool: str
    input: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    ok: bool = True
    error: dict[str, Any] | None = None
    at: float = Field(default_factory=time.time)
    duration_ms: float = 0.0
    metered: bool = True
    """False for refused calls. They are recorded (a demigod probing for tools
    it was not granted is exactly what an audit log is for) but they do not
    consume the call budget -- otherwise a typo'd tool name costs real work."""


@runtime_checkable
class GrantStore(Protocol):
    """The seam between the router and wherever lease state actually lives.

    Two implementations: `InMemoryGrantStore` (tests, local dev, the in-process
    runtime) and `broker.modal_store.ModalGrantStore` (Modal Dict + Queue). The
    router is written against this and imports no Modal, which is what makes the
    whole request path testable offline.
    """

    def publish(self, grant: Grant) -> None: ...

    def read(self, lease_id: str) -> Grant | None: ...

    def record(self, entry: TraceEntry, *, lease_id: str) -> None: ...

    def calls_used(self, lease_id: str) -> int: ...

    def trace(
        self, lease_id: str, *, include_refused: bool = False
    ) -> list[TraceEntry]: ...

    def revoke(self, lease_id: str) -> None: ...


class InMemoryGrantStore:
    """Single-process store. Correct, and correct is all it has to be.

    Used by every offline test and by the local (non-sandbox) runtime. It is
    NOT a fallback for the deployed broker: a Modal Function with
    `max_containers > 1` backed by this would give each replica its own budget,
    which is a lease that silently multiplies.
    """

    def __init__(self) -> None:
        self._grants: dict[str, Grant] = {}
        self._trace: dict[str, list[TraceEntry]] = {}

    def publish(self, grant: Grant) -> None:
        self._grants[grant.lease_id] = grant
        self._trace.setdefault(grant.lease_id, [])

    def read(self, lease_id: str) -> Grant | None:
        return self._grants.get(lease_id)

    def record(self, entry: TraceEntry, *, lease_id: str) -> None:
        self._trace.setdefault(lease_id, []).append(entry)

    def calls_used(self, lease_id: str) -> int:
        return sum(1 for e in self._trace.get(lease_id, []) if e.metered)

    def trace(
        self, lease_id: str, *, include_refused: bool = False
    ) -> list[TraceEntry]:
        entries = self._trace.get(lease_id, [])
        if include_refused:
            return list(entries)
        return [e for e in entries if e.metered]

    def revoke(self, lease_id: str) -> None:
        """Drop the grant, keep the trace.

        Revocation ends authority; it does not un-happen the calls. GOD reads
        the trace after the demigod is gone, so discarding it here would throw
        away the audit record at exactly the moment it becomes useful.
        """
        self._grants.pop(lease_id, None)


def mint_grant(
    lease: CapabilityLease, *, label: str = "", now: float | None = None
) -> Grant:
    """Wrap an existing lease for publication. Does NOT create authority.

    Leases are minted by `ToolRegistry.mint_lease` under GOD's policy checks
    (operator approval for write tools). If the broker could mint
    its own, a bug here would become a privilege escalation there.
    """
    return Grant(
        lease=lease,
        issued_at=now if now is not None else time.time(),
        label=label,
    )


__all__ = [
    "Grant",
    "GrantStore",
    "InMemoryGrantStore",
    "TraceEntry",
    "mint_grant",
    "new_call_id",
]
