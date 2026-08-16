"""Lease state and tiering: the two things the router delegates.

`InMemoryGrantStore` is tested directly because `ModalGrantStore` is the same
contract over `modal.Dict` + `modal.Queue` and cannot be exercised without an
account. What CAN be pinned offline is the semantics both must honour -- which
is exactly what these tests are for.
"""

from __future__ import annotations

import pytest
from conftest import request

from broker.dispatch import DispatchPolicy, DispatchUnavailableError
from broker.grants import Grant, InMemoryGrantStore, TraceEntry, mint_grant
from broker.session import local_session
from demigod.toolbox.protocol import ErrorCode, call_path
from reagents.contracts import Budget, ToolProvider
from reagents.tools.registry import Tool, ToolRegistry

# --- the counter is the audit log --------------------------------------------


def test_calls_used_counts_only_metered_entries():
    """The design in one assertion. `calls_used` is the length of the metered
    trace, so the number the broker enforces against and the number it reports
    to GOD cannot disagree."""
    store = InMemoryGrantStore()
    lease = ToolRegistry().mint_lease([], subject_id="s")
    store.publish(mint_grant(lease))

    store.record(TraceEntry(tool="a", metered=True), lease_id=lease.lease_id)
    store.record(TraceEntry(tool="b", metered=False), lease_id=lease.lease_id)
    store.record(TraceEntry(tool="c", metered=True), lease_id=lease.lease_id)

    assert store.calls_used(lease.lease_id) == 2
    assert [e.tool for e in store.trace(lease.lease_id)] == ["a", "c"]
    assert [e.tool for e in store.trace(lease.lease_id, include_refused=True)] == [
        "a",
        "b",
        "c",
    ]


def test_revoke_ends_authority_but_keeps_the_trace():
    """GOD reads the trace after the demigod is gone, so discarding it on revoke
    would throw the audit record away at the moment it becomes useful."""
    store = InMemoryGrantStore()
    lease = ToolRegistry().mint_lease([], subject_id="s")
    store.publish(mint_grant(lease))
    store.record(TraceEntry(tool="a"), lease_id=lease.lease_id)

    store.revoke(lease.lease_id)

    assert store.read(lease.lease_id) is None
    assert len(store.trace(lease.lease_id)) == 1


def test_expiry_is_derived_from_the_lease_not_stored_twice():
    lease = ToolRegistry().mint_lease(
        [], subject_id="s", budget=Budget(wall_time_s=30.0)
    )
    grant = Grant(lease=lease, issued_at=1000.0)
    assert grant.expires_at == 1030.0
    assert grant.expires_in(1025.0) == pytest.approx(5.0)
    assert grant.expires_in(1031.0) < 0


def test_mint_grant_does_not_create_authority():
    """Leases are minted GOD-side under operator-approval checks. If the broker
    could mint its own, a bug in the request path would become an escalation."""
    lease = ToolRegistry().mint_lease(["x"], subject_id="s", allow_write=True)
    grant = mint_grant(lease)
    assert grant.lease is lease
    assert grant.lease.allow_write is True


# --- the session, as GOD uses it ---------------------------------------------


def test_session_round_trip_produces_a_grant_a_trace_and_a_revocation():
    registry = ToolRegistry()
    registry.register(
        Tool(
            id="x.y",
            description="d",
            parameters_schema={"type": "object"},
            fn=lambda: 1,
        )
    )
    pack = registry.bind(["x.y"], subject_id="dg")
    session = local_session("https://broker.example")

    grant = session.grant(pack, label="a-domain")
    assert grant.lease_id == pack.lease.lease_id
    assert grant.tool_ids == ["x.y"]
    assert grant.base == "https://broker.example"

    session.store.record(TraceEntry(tool="x.y", result=1), lease_id=grant.lease_id)
    trace = session.collect_trace(grant.lease_id)
    assert trace[0]["tool"] == "x.y"
    assert trace[0]["result"] == 1

    session.revoke(grant.lease_id)
    assert session.store.read(grant.lease_id) is None


def test_collect_trace_includes_refusals_by_default():
    """A demigod that spent six turns calling a tool it was never granted
    produced nothing it would choose to report. The refusals are the only record
    that the turns went somewhere."""
    session = local_session()
    lease = ToolRegistry().mint_lease([], subject_id="s")
    session.store.publish(mint_grant(lease))
    session.store.record(
        TraceEntry(tool="denied", metered=False), lease_id=lease.lease_id
    )

    assert len(session.collect_trace(lease.lease_id)) == 1
    assert session.collect_trace(lease.lease_id, include_refused=False) == []


# --- the trace the broker actually writes ------------------------------------


def test_the_broker_records_every_call_including_refusals(router, store, lease_id):
    request(
        router,
        "POST",
        call_path("graph.add"),
        token=lease_id,
        body={"input": {"a": 1, "b": 2}},
    )
    request(router, "POST", call_path("graph.boom"), token=lease_id)
    request(router, "POST", call_path("graph.unleased"), token=lease_id)

    full = store.trace(lease_id, include_refused=True)
    assert [e.tool for e in full] == ["graph.add", "graph.boom", "graph.unleased"]

    ok, failed, refused = full
    assert ok.ok and ok.result == {"sum": 3} and ok.input == {"a": 1, "b": 2}
    assert not failed.ok and failed.error["code"] == ErrorCode.TOOL_ERROR
    assert not refused.metered
    assert refused.error["code"] == ErrorCode.UNBOUND_TOOL


def test_large_results_are_truncated_in_the_trace(store):
    """The trace says what was called. It is not a second copy of the artifact
    store, and a tool returning a big frame would otherwise put it in every
    DemiGodResult."""
    registry = ToolRegistry()
    registry.register(
        Tool(
            id="big.thing",
            description="d",
            parameters_schema={"type": "object", "properties": {}},
            fn=lambda: {"rows": ["x" * 100] * 200},
        )
    )
    lease = registry.mint_lease(["big.thing"], subject_id="s")
    store.publish(mint_grant(lease))
    from broker.router import ToolboxRouter

    router = ToolboxRouter(registry=registry, store=store)

    response = request(router, "POST", call_path("big.thing"), token=lease.lease_id)
    # The CALLER still gets the whole thing...
    assert len(response.payload["result"]["rows"]) == 200
    # ...only the trace is bounded.
    assert store.trace(lease.lease_id)[0].result["truncated"] is True


# --- tiering -----------------------------------------------------------------


def _tool(tool_id: str, provider: ToolProvider) -> Tool:
    return Tool(
        id=tool_id,
        description="d",
        parameters_schema={"type": "object"},
        fn=lambda **kw: {"inline": True, **kw},
        provider=provider,
    )


def test_local_tools_stay_inline():
    policy = DispatchPolicy()
    tool = _tool("a.b", ToolProvider.LOCAL)
    assert policy.is_inline(tool)
    assert policy.prepare(tool) is tool


@pytest.mark.parametrize("provider", [ToolProvider.CONTAINER, ToolProvider.MCP])
def test_non_local_tools_are_dispatched_out(provider):
    seen = {}

    async def remote(tool_id, arguments):
        seen["call"] = (tool_id, arguments)
        return {"inline": False}

    policy = DispatchPolicy(remote=remote)
    prepared = policy.prepare(_tool("heavy.solve", provider))

    # Same id, same schema, different machine -- and crucially still a `Tool`,
    # so ToolBroker remains the only path from a request to an executor.
    assert prepared.id == "heavy.solve"
    assert prepared.fn is None and prepared.executor is not None

    import asyncio

    assert asyncio.run(prepared.call_async(x=1)) == {"inline": False}
    assert seen["call"] == ("heavy.solve", {"x": 1})


def test_force_remote_pushes_out_a_local_tool():
    """The escape hatch for a LOCAL tool that turns out to be slow. Changing its
    `provider` instead would be a lie about how it is implemented."""

    async def remote(tool_id, arguments):
        return "remote"

    policy = DispatchPolicy(remote=remote, force_remote=frozenset({"a.b"}))
    assert not policy.is_inline(_tool("a.b", ToolProvider.LOCAL))


def test_a_broker_without_executors_names_the_missing_tier():
    policy = DispatchPolicy(remote=None)
    with pytest.raises(DispatchUnavailableError) as exc:
        policy.prepare(_tool("heavy.x", ToolProvider.CONTAINER))
    assert "force_inline" in str(exc.value)


def test_dispatch_failure_surfaces_as_upstream_error_not_a_500(store):
    """A misconfigured broker must still answer in the protocol. A 500 with an
    HTML body is unreadable to a demigod parsing JSON."""
    registry = ToolRegistry()
    registry.register(_tool("heavy.x", ToolProvider.CONTAINER))
    lease = registry.mint_lease(["heavy.x"], subject_id="s")
    store.publish(mint_grant(lease))
    from broker.router import ToolboxRouter

    router = ToolboxRouter(registry=registry, store=store)

    response = request(router, "POST", call_path("heavy.x"), token=lease.lease_id)
    assert response.status == 200
    assert response.payload["error"]["code"] == ErrorCode.UPSTREAM_ERROR
