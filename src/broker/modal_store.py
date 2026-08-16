"""`GrantStore` backed by Modal primitives. The only file here that imports Modal.

TWO OBJECTS, CHOSEN FOR THEIR CONCURRENCY, NOT THEIR CONVENIENCE:

`modal.Dict` holds the grants. A grant is written once by GOD and only ever read
by the broker, so there is no write contention to lose.

`modal.Queue` holds the trace, partitioned by lease id. `put` is atomic and
`iterate` is non-mutating (verified against modal 1.5.4), which is what lets
`calls_used` be `Queue.len(partition=lease_id)` -- the budget counter and the
audit log are the same object, so they cannot drift apart, and no replica ever
performs a read-modify-write on a shared integer.

THE ONE RACE THAT REMAINS, stated plainly: `calls_used` is read, then the tool
runs, then the entry is appended. Two calls on the SAME lease that overlap can
both see `used == max_calls - 1` and both proceed. A demigod is a single agent
issuing one bash command at a time, so the overlap window is not reachable in
the design as built; a future concurrent client would overshoot its budget by at
most the number of in-flight calls. Fixing it properly means a reservation
record per call, which costs a second round-trip on every call to close a hole
nothing can currently fall through.

RETENTION: queue partitions expire after `PARTITION_TTL_S`. A trace outlives the
run that produced it by a day, which is long enough for GOD to collect it and
short enough that a hackathon does not accumulate a permanent audit database.
"""

from __future__ import annotations

import contextlib
import hashlib
from typing import TYPE_CHECKING, Any

from broker.grants import Grant, TraceEntry

if TYPE_CHECKING:  # keep `modal` out of the import path for pure-logic tests
    import modal

GRANTS_DICT_NAME = "toolbox-grants"
TRACE_QUEUE_NAME = "toolbox-trace"

PARTITION_TTL_S = 24 * 3600
MAX_PARTITION_KEY = 64
"""Modal's limit, verified in `modal.queue._Queue.validate_partition_key`."""

REFUSED_SUFFIX = "!x"
"""Refused calls go to their own partition so they are audited without being
counted. `calls_used` is a partition length, so a refusal in the metered
partition would silently charge for a tool the caller never got."""


def _partition(lease_id: str, *, refused: bool = False) -> str:
    """Partition key for a lease, guaranteed inside Modal's 64-byte limit.

    Long lease ids are hashed rather than truncated: truncation makes two leases
    that share a prefix share a budget, which is the worst possible failure --
    silent, and in the permissive direction.
    """
    suffix = REFUSED_SUFFIX if refused else ""
    key = f"{lease_id}{suffix}"
    if len(key.encode("utf-8")) <= MAX_PARTITION_KEY:
        return key
    digest = hashlib.sha256(lease_id.encode("utf-8")).hexdigest()[:40]
    return f"h_{digest}{suffix}"


class ModalGrantStore:
    """Lease state that survives an autoscaling broker.

    Lazily hydrated: constructing this is free and touches no network, so a
    caller can build a router before it has Modal credentials and only pay on
    first use. That matters because `broker.service` constructs it at import
    time, inside a container that may be about to serve a health check.
    """

    def __init__(
        self,
        *,
        grants_name: str = GRANTS_DICT_NAME,
        trace_name: str = TRACE_QUEUE_NAME,
        create_if_missing: bool = True,
    ) -> None:
        self.grants_name = grants_name
        self.trace_name = trace_name
        self.create_if_missing = create_if_missing
        self._grants: Any = None
        self._trace: Any = None

    # --- lazy handles -------------------------------------------------------

    @property
    def grants(self) -> modal.Dict:
        import modal

        if self._grants is None:
            self._grants = modal.Dict.from_name(
                self.grants_name, create_if_missing=self.create_if_missing
            )
        return self._grants

    @property
    def traces(self) -> modal.Queue:
        import modal

        if self._trace is None:
            self._trace = modal.Queue.from_name(
                self.trace_name, create_if_missing=self.create_if_missing
            )
        return self._trace

    # --- GrantStore ---------------------------------------------------------

    def publish(self, grant: Grant) -> None:
        # Stored as JSON, not as a pickled model: the writer (GOD, on a laptop)
        # and the reader (a broker replica, on a Modal image) are different
        # processes that will drift in pydantic version long before they drift
        # in this schema.
        self.grants[grant.lease_id] = grant.model_dump_json()

    def read(self, lease_id: str) -> Grant | None:
        raw = self.grants.get(lease_id)
        if raw is None:
            return None
        try:
            return Grant.model_validate_json(raw)
        except Exception:
            return None

    def record(self, entry: TraceEntry, *, lease_id: str) -> None:
        self.traces.put(
            entry.model_dump_json(),
            partition=_partition(lease_id, refused=not entry.metered),
            partition_ttl=PARTITION_TTL_S,
        )

    def calls_used(self, lease_id: str) -> int:
        return self.traces.len(partition=_partition(lease_id))

    def trace(
        self, lease_id: str, *, include_refused: bool = False
    ) -> list[TraceEntry]:
        entries = self._read_partition(_partition(lease_id))
        if include_refused:
            entries += self._read_partition(_partition(lease_id, refused=True))
            entries.sort(key=lambda e: e.at)
        return entries

    def revoke(self, lease_id: str) -> None:
        """Drop the grant. The trace stays -- GOD reads it after the demigod dies.

        Idempotent: revocation runs in a `finally`, so it will be called again
        on paths that already revoked, and a KeyError there would mask the error
        that got us into the finally in the first place.
        """
        with contextlib.suppress(KeyError):
            self.grants.pop(lease_id)

    # --- internals ----------------------------------------------------------

    def _read_partition(self, partition: str) -> list[TraceEntry]:
        """Non-mutating read of a partition. Verified against modal 1.5.4.

        `Queue.iterate` is declared as an async generator but the blocking
        surface returns a sync one (`modal.queue.blocking_iterate`), so it is
        driven with a plain comprehension. `item_poll_timeout=0.0` means "stop
        at the end of what exists" rather than "wait for more" -- without it a
        caller collecting a finished run blocks for the poll window every time.

        Non-mutating is the load-bearing property: `get_many` would drain the
        partition, and the partition length IS the lease's call counter.
        """
        out: list[TraceEntry] = []
        for item in self.traces.iterate(partition=partition, item_poll_timeout=0.0):
            try:
                out.append(TraceEntry.model_validate_json(item))
            except Exception:
                continue
        return out


__all__ = [
    "GRANTS_DICT_NAME",
    "PARTITION_TTL_S",
    "TRACE_QUEUE_NAME",
    "ModalGrantStore",
]
