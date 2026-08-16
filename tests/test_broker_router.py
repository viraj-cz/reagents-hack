"""Every way a brokered call can be refused, and the one way it succeeds.

No Modal, no network, no API key: the router is a plain ASGI callable and these
tests hand it a scope. That is the whole reason it was written without a
framework.

The refusals matter more than the happy path. A broker that leaks authority
fails silently and looks fine; a broker that refuses correctly is the only thing
standing between a demigod and a sibling's credentials.
"""

from __future__ import annotations

import pytest
from conftest import request

from broker.grants import InMemoryGrantStore, mint_grant
from broker.router import ToolboxRouter
from demigod.toolbox.protocol import (
    ErrorCode,
    call_path,
    describe_path,
)
from reagents.contracts import Budget
from reagents.tools.registry import ToolRegistry

CALL = call_path("graph.add")
ADD = {"input": {"a": 2, "b": 3}}


# --- authentication ----------------------------------------------------------


def test_health_needs_no_lease(router: ToolboxRouter):
    """The one unauthenticated route. Without it, 'broker is down' and 'my lease
    is bad' are the same observation from inside a sandbox."""
    response = request(router, "GET", "/v1/health")
    assert response.status == 200
    assert response.payload["ok"] is True


def test_no_token_is_401(router: ToolboxRouter):
    response = request(router, "GET", "/v1/tools")
    assert response.status == 401
    assert response.code == ErrorCode.UNAUTHORIZED


def test_unknown_token_is_401(router: ToolboxRouter, lease_id: str):
    response = request(router, "GET", "/v1/tools", token="lease_not_a_real_one")
    assert response.status == 401
    assert response.code == ErrorCode.UNAUTHORIZED


def test_malformed_authorization_header_is_401(router: ToolboxRouter, lease_id: str):
    """A bare token with no `Bearer` prefix. Accepting it would mean two auth
    formats to keep in sync, and the looser one always wins by accident."""
    response = request(
        router,
        "GET",
        "/v1/tools",
        headers=[(b"authorization", lease_id.encode())],
    )
    assert response.status == 401


def test_revoked_lease_is_401(
    router: ToolboxRouter, store: InMemoryGrantStore, lease_id: str
):
    """Revocation is what makes a lease safe to hand to a sandbox: the credential
    stops working the moment GOD says so, without waiting for wall time."""
    assert request(router, "GET", "/v1/tools", token=lease_id).status == 200
    store.revoke(lease_id)
    assert request(router, "GET", "/v1/tools", token=lease_id).status == 401


def test_expired_lease_is_410_not_401(
    registry: ToolRegistry, store: InMemoryGrantStore
):
    """A distinct code, because the two mean different things to an agent:
    401 might be a misconfiguration worth re-reading the grant file for, 410
    means the run is over and the manifest should be written now."""
    lease = registry.mint_lease(
        ["graph.add"], subject_id="s", budget=Budget(wall_time_s=10.0)
    )
    store.publish(mint_grant(lease, now=1000.0))
    router = ToolboxRouter(registry=registry, store=store, clock=lambda: 1011.0)

    response = request(router, "GET", "/v1/tools", token=lease.lease_id)
    assert response.status == 410
    assert response.code == ErrorCode.LEASE_EXPIRED


# --- what a lease can see ----------------------------------------------------


def test_list_shows_only_leased_tools(router: ToolboxRouter, lease_id: str):
    response = request(router, "GET", "/v1/tools", token=lease_id)
    ids = [t["id"] for t in response.payload["tools"]]
    assert ids == ["graph.add", "graph.boom"]
    assert "graph.unleased" not in ids
    assert "remote.publish" not in ids


def test_list_reports_the_remaining_budget(router: ToolboxRouter, lease_id: str):
    """An agent that cannot see its remaining calls spends them exploring and
    discovers the wall on the call that mattered."""
    before = request(router, "GET", "/v1/tools", token=lease_id).payload["lease"]
    assert before["calls_used"] == 0
    assert before["max_calls"] == 3

    request(router, "POST", CALL, token=lease_id, body=ADD)

    after = request(router, "GET", "/v1/tools", token=lease_id).payload["lease"]
    assert after["calls_used"] == 1


def test_describe_returns_the_input_schema(router: ToolboxRouter, lease_id: str):
    response = request(router, "GET", describe_path("graph.add"), token=lease_id)
    assert response.status == 200
    assert response.payload["tool"]["input_schema"]["required"] == ["a", "b"]


def test_unleased_tool_is_unbound_not_unknown(router: ToolboxRouter, lease_id: str):
    """MINIMAL DISCLOSURE, pinned. `graph.unleased` exists in the catalog, but
    saying so would let a demigod enumerate tools nobody granted it. Both the
    existent and non-existent cases must answer identically."""
    real = request(router, "GET", describe_path("graph.unleased"), token=lease_id)
    fake = request(router, "GET", describe_path("does.not.exist"), token=lease_id)

    assert real.status == fake.status == 403
    assert real.code == fake.code == ErrorCode.UNBOUND_TOOL
    assert "secret" not in real.message
    assert real.message.replace("graph.unleased", "X") == fake.message.replace(
        "does.not.exist", "X"
    )


def test_broker_missing_a_leased_tool_says_so(
    registry: ToolRegistry, store: InMemoryGrantStore
):
    """A lease naming a tool this deployment does not have is a broker/GOD
    version mismatch. It must be reported as such -- `unbound_tool` would send
    the agent looking for a mistake in its own lease."""
    lease = registry.mint_lease(["graph.add"], subject_id="s")
    lease.tool_ids.append("not.installed.here")
    store.publish(mint_grant(lease))
    router = ToolboxRouter(registry=registry, store=store)

    listed = request(router, "GET", "/v1/tools", token=lease.lease_id)
    assert [t["id"] for t in listed.payload["tools"]] == ["graph.add"]

    response = request(
        router, "POST", call_path("not.installed.here"), token=lease.lease_id
    )
    assert response.status == 404
    assert response.code == ErrorCode.UNKNOWN_TOOL


# --- write authority ---------------------------------------------------------


def test_write_tool_needs_a_write_lease(
    registry: ToolRegistry, store: InMemoryGrantStore
):
    """The same rule `ToolBroker` enforces in-process, now over the wire. Write
    authority is an operator decision made GOD-side; the broker must not be a
    second, looser gate on it."""
    readonly = registry.mint_lease(["remote.publish"], subject_id="reader")
    store.publish(mint_grant(readonly))
    router = ToolboxRouter(registry=registry, store=store)

    denied = request(
        router, "POST", call_path("remote.publish"), token=readonly.lease_id
    )
    assert denied.status == 403
    assert denied.code == ErrorCode.WRITE_DENIED

    writer = registry.mint_lease(
        ["remote.publish"], subject_id="writer", allow_write=True
    )
    store.publish(mint_grant(writer))
    allowed = request(
        router, "POST", call_path("remote.publish"), token=writer.lease_id
    )
    assert allowed.payload["ok"] is True
    assert allowed.payload["result"] == "published"


# --- budget ------------------------------------------------------------------


def test_calls_are_metered_and_the_lease_runs_out(router: ToolboxRouter, lease_id: str):
    for _ in range(3):
        assert request(router, "POST", CALL, token=lease_id, body=ADD).status == 200

    exhausted = request(router, "POST", CALL, token=lease_id, body=ADD)
    assert exhausted.status == 429
    assert exhausted.code == ErrorCode.LEASE_EXHAUSTED


def test_a_failing_tool_still_costs_a_call(
    router: ToolboxRouter, store: InMemoryGrantStore, lease_id: str
):
    """Otherwise an agent can exhaust the broker for free by calling badly, and
    a lease that only charges for success is not a budget."""
    response = request(router, "POST", call_path("graph.boom"), token=lease_id)
    assert response.status == 200
    assert response.payload["ok"] is False
    assert response.payload["error"]["code"] == ErrorCode.TOOL_ERROR
    assert store.calls_used(lease_id) == 1


def test_a_refused_call_costs_nothing(
    router: ToolboxRouter, store: InMemoryGrantStore, lease_id: str
):
    """A typo'd tool name must not cost real work. Refusals are audited in their
    own partition instead -- see test_broker_grants."""
    request(router, "POST", call_path("graph.unleased"), token=lease_id)
    request(router, "POST", CALL, token=lease_id, body={"input": {"a": "not an int"}})
    assert store.calls_used(lease_id) == 0


# --- input validation --------------------------------------------------------


def test_schema_violation_is_rejected_before_the_tool_runs(
    router: ToolboxRouter, lease_id: str
):
    """This is what gRPC would have bought, taken for free from a schema the
    tool already declares."""
    response = request(router, "POST", CALL, token=lease_id, body={"input": {"a": 1}})
    assert response.status == 422
    assert response.code == ErrorCode.INVALID_INPUT
    assert "b" in str(response.payload["error"]["detail"]["errors"])


def test_unexpected_argument_is_a_message_not_a_typeerror(
    router: ToolboxRouter, lease_id: str
):
    """Arguments become Python kwargs. Without this check the agent sees
    `TypeError: <lambda>() got an unexpected keyword argument` and has to guess
    which of its keys was wrong."""
    response = request(
        router,
        "POST",
        CALL,
        token=lease_id,
        body={"input": {"a": 1, "b": 2, "c": 3}},
    )
    assert response.status == 422
    errors = str(response.payload["error"]["detail"]["errors"])
    assert "'c'" in errors and "accepted" in errors


def test_invalid_input_is_the_only_retryable_refusal(lease_id: str):
    """Pinned because the CLI and client branch on it: an agent that retries a
    permanent refusal burns the turns it needed for its manifest."""
    from demigod.toolbox.protocol import RETRYABLE_CODES

    assert ErrorCode.INVALID_INPUT in RETRYABLE_CODES
    for permanent in (
        ErrorCode.LEASE_EXHAUSTED,
        ErrorCode.LEASE_EXPIRED,
        ErrorCode.UNBOUND_TOOL,
        ErrorCode.WRITE_DENIED,
        ErrorCode.UNAUTHORIZED,
    ):
        assert permanent not in RETRYABLE_CODES


# --- envelope handling -------------------------------------------------------


def test_bad_json_body_is_a_400(router: ToolboxRouter, lease_id: str):
    response = request(router, "POST", CALL, token=lease_id, raw_body=b"{not json")
    assert response.status == 400
    assert response.code == ErrorCode.BAD_REQUEST


def test_wrong_method_is_a_400(router: ToolboxRouter, lease_id: str):
    assert request(router, "GET", CALL, token=lease_id).status == 400


def test_unknown_route_is_a_400(router: ToolboxRouter, lease_id: str):
    assert request(router, "GET", "/v1/nope", token=lease_id).status == 400


def test_a_tool_that_raises_is_200_with_ok_false(router: ToolboxRouter, lease_id: str):
    """Not a 5xx. The CALL succeeded; the tool disagreed with its input, and the
    agent should read the message and adapt rather than conclude the broker is
    broken. Same convention MCP uses with `isError`."""
    response = request(router, "POST", call_path("graph.boom"), token=lease_id)
    assert response.status == 200
    assert response.payload["ok"] is False
    assert "disagreed with its input" in response.payload["error"]["message"]


def test_happy_path_returns_the_result_and_the_budget(
    router: ToolboxRouter, lease_id: str
):
    response = request(router, "POST", CALL, token=lease_id, body=ADD)
    assert response.status == 200
    assert response.payload["ok"] is True
    assert response.payload["result"] == {"sum": 5}
    assert response.payload["lease"]["calls_used"] == 1
    assert response.payload["call_id"].startswith("call_")


@pytest.mark.parametrize("tool_id", ["graph.add", "a.b.c", "weird tool/name"])
def test_tool_ids_survive_url_encoding(tool_id: str):
    """Ids contain dots today and could contain worse tomorrow. The path is
    built and parsed by the same module, so this pins the round trip."""
    from urllib.parse import unquote

    path = call_path(tool_id)
    assert unquote(path[len("/v1/tools/") : -len("/call")]) == tool_id
