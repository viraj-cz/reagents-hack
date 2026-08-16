"""GOD's side of the broker: publish a lease, collect its trace, revoke it.

This is the whole GOD-facing API. Three calls, in the order the orchestrator
makes them:

    grant   -> before the sandbox exists, because a demigod must be handed its
               credential at spawn time and cannot be given one later
    collect -> after the sandbox is gone, because the trace outlives it
    revoke  -> unconditionally, because a lease that outlives its demigod is a
               credential lying around

WHY GOD PUBLISHES AND THE BROKER ONLY READS. Authority is minted by
`ToolRegistry.mint_lease` under GOD's policy checks -- operator approval for
write tools, for high-risk tools. If the broker could mint its own grants, a bug
in the request path would become privilege escalation rather than a 500.

WHY THE TRACE COMES FROM HERE AND NOT FROM THE AGENT. `DemiGodResult.tool_trace`
has been an empty list since the contract was written. The broker sees every
call, including the ones that were refused, so a trace assembled here cannot be
under-reported by the thing being audited.
"""

from __future__ import annotations

from typing import Any

from broker.grants import Grant, GrantStore, InMemoryGrantStore, mint_grant
from demigod.toolbox.protocol import ToolboxGrant
from reagents.contracts import CapabilityLease
from reagents.tools.registry import BoundToolPack


class ToolboxSession:
    """One broker endpoint plus its state store. Cheap; hold one per run."""

    def __init__(self, *, url: str, store: GrantStore) -> None:
        self.url = url.rstrip("/")
        self.store = store

    # --- before the spawn ---------------------------------------------------

    def grant(self, pack: BoundToolPack, *, label: str = "") -> ToolboxGrant:
        """Publish a bound pack's lease and return what the demigod is handed.

        Takes a `BoundToolPack`, not a list of ids, on purpose: the pack is the
        object GOD's orchestrator already produced after its approval checks,
        and its lease is the one those checks validated. Accepting ids here
        would be a second, unchecked path to authority.
        """
        return self.grant_lease(pack.lease, label=label)

    def grant_lease(self, lease: CapabilityLease, *, label: str = "") -> ToolboxGrant:
        record = mint_grant(lease, label=label)
        self.store.publish(record)
        return self.to_grant(record)

    def to_grant(self, record: Grant) -> ToolboxGrant:
        return ToolboxGrant(
            url=self.url,
            lease_id=record.lease_id,
            tool_ids=list(record.lease.tool_ids),
            expires_at=record.expires_at,
        )

    # --- after the run ------------------------------------------------------

    def collect_trace(
        self, lease_id: str, *, include_refused: bool = True
    ) -> list[dict[str, Any]]:
        """The rows for `DemiGodResult.tool_trace`, oldest first.

        Refused calls are INCLUDED by default. A demigod that spent six turns
        calling a tool it was never granted produced no results and no errors it
        chose to report -- the refusals are the only record that the turns went
        somewhere, and they are the signal that the domain was given the wrong
        tools.
        """
        entries = self.store.trace(lease_id, include_refused=include_refused)
        return [entry.model_dump(mode="json") for entry in entries]

    def calls_used(self, lease_id: str) -> int:
        return self.store.calls_used(lease_id)

    def revoke(self, lease_id: str) -> None:
        """End the lease's authority. The trace survives -- see `broker.grants`."""
        self.store.revoke(lease_id)


def local_session(url: str = "http://127.0.0.1:0") -> ToolboxSession:
    """In-process session. Tests, and any single-process runtime.

    NOT a fallback for a deployed broker: an `InMemoryGrantStore` behind an
    autoscaled Function gives every replica its own budget, which is a lease
    that silently multiplies.
    """
    return ToolboxSession(url=url, store=InMemoryGrantStore())


def modal_session(url: str | None = None) -> ToolboxSession:
    """The real one: Modal-backed state, pointed at the deployed router.

    Imports `broker.service` lazily so that constructing a local session -- the
    only thing the offline test suite does -- never builds a `modal.App`.
    """
    from broker.modal_store import ModalGrantStore

    if url is None:
        from broker.service import endpoint_url

        url = endpoint_url()
    return ToolboxSession(url=url, store=ModalGrantStore())


__all__ = ["ToolboxSession", "local_session", "modal_session"]
