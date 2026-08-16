"""TOOLBOX_BROKER -- the third sandbox type.

    GOD        plans, seals, mints leases, spawns
    DEMI_GOD   reasons in one invented domain, in its own sandbox
    BROKER     executes tools on behalf of a lease, and audits every call

WHAT PROBLEM THIS SOLVES. `reagents` tool ids resolve to in-process Python
callables behind a capability lease. A DEMI_GOD is a different process on a
different machine and cannot call them. Before this package, `reagents.demigod.
adapter` translated the handful of ids that happened to have a pip-package
equivalent and told the agent the rest were unreachable -- so in practice every
demigod reasoned with no tools at all.

THE SHAPE OF THE FIX. The broker holds the callables and the credentials; the
demigod holds a URL and a lease id. One HTTP hop, authenticated by the lease
itself, validated against the JSON Schema the tool already declares.

    demigod sandbox                    broker (modal.Function, autoscaled)
    +---------------------+            +-------------------------------+
    | toolbox call X -i f | --HTTPS--> | router  (asgi, cheap tools)   |
    |   Bearer lease_...  |            |   |-- inline: LOCAL tools     |
    | no modal token      |            |   '-- remote: exec_* per class|
    +---------------------+            +-------------------------------+

TRUST ASYMMETRY, WHICH IS THE POINT. Modal tokens are workspace-wide. A demigod
holding one could spawn sandboxes and read every sibling's output volume, which
would dissolve the isolation the whole system exists to provide. So it holds no
Modal credential, and it cannot be given one by accident: nothing shipped into
its image imports Modal. That is why `demigod/toolbox/` is stdlib-only and why
`broker/` is never added to a demigod image.

LAYOUT
    grants.py       durable lease state; the call counter IS the audit log
    router.py       the request path. Pure ASGI, no Modal, fully testable offline
    dispatch.py     which tools run inline and which get their own container
    modal_store.py  GrantStore over modal.Dict + modal.Queue
    service.py      the modal.App: router endpoint + one executor per tool class
    session.py      GOD's API: grant -> collect_trace -> revoke

The demigod-facing half is `src/demigod/toolbox/` -- deliberately in the other
package, because only that package is shipped into a sandbox.
"""

from __future__ import annotations

from broker.dispatch import DispatchPolicy, DispatchUnavailableError
from broker.grants import Grant, GrantStore, InMemoryGrantStore, TraceEntry, mint_grant
from broker.router import ToolboxRouter
from broker.session import ToolboxSession, local_session, modal_session

__all__ = [
    "DispatchPolicy",
    "DispatchUnavailableError",
    "Grant",
    "GrantStore",
    "InMemoryGrantStore",
    "ToolboxRouter",
    "ToolboxSession",
    "TraceEntry",
    "local_session",
    "mint_grant",
    "modal_session",
]
