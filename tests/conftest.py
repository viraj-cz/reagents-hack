"""Shared fixtures for the TOOLBOX_BROKER tests. Offline, always.

Nothing here touches Modal, the network beyond loopback, or an API key. The
broker's whole request path is a plain ASGI callable precisely so it can be
exercised this way -- see `broker/router.py`.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from broker.grants import InMemoryGrantStore, mint_grant
from broker.router import ToolboxRouter
from demigod.toolbox.protocol import auth_header
from reagents.contracts import Budget, ToolAccess
from reagents.tools.registry import Tool, ToolRegistry

# --- a small, fully-known registry -------------------------------------------
#
# Deliberately not `default_registry()`: these tests are about the broker, and
# pinning them to the builtins would make an unrelated edit to `builtins.py`
# fail the authorization suite.


def demo_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            id="graph.add",
            namespace="graph",
            description="Add two integers.\nSecond line, to test list rendering.",
            parameters_schema={
                "type": "object",
                "required": ["a", "b"],
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            },
            fn=lambda a, b: {"sum": a + b},
        )
    )
    registry.register(
        Tool(
            id="graph.boom",
            namespace="graph",
            description="Always raises. Stands in for a tool given bad data.",
            parameters_schema={"type": "object", "properties": {}},
            fn=_boom,
        )
    )
    registry.register(
        Tool(
            id="remote.publish",
            namespace="remote",
            description="A write tool.",
            parameters_schema={"type": "object", "properties": {}},
            fn=lambda: "published",
            access=ToolAccess.WRITE,
        )
    )
    registry.register(
        Tool(
            id="graph.unleased",
            namespace="graph",
            description="Exists in the catalog but is never leased.",
            parameters_schema={"type": "object", "properties": {}},
            fn=lambda: "secret",
        )
    )
    return registry


def _boom() -> None:
    raise ValueError("the tool disagreed with its input")


@pytest.fixture
def registry() -> ToolRegistry:
    return demo_registry()


@pytest.fixture
def store() -> InMemoryGrantStore:
    return InMemoryGrantStore()


@pytest.fixture
def router(registry: ToolRegistry, store: InMemoryGrantStore) -> ToolboxRouter:
    return ToolboxRouter(registry=registry, store=store)


@pytest.fixture
def lease_id(registry: ToolRegistry, store: InMemoryGrantStore) -> str:
    """A published lease over `graph.add` and `graph.boom`. Read-only, 3 calls."""
    lease = registry.mint_lease(
        ["graph.add", "graph.boom"],
        subject_id="test-demigod",
        budget=Budget(max_tool_calls=3, wall_time_s=300.0),
    )
    store.publish(mint_grant(lease, label="test-domain"))
    return lease.lease_id


# --- driving the ASGI app directly -------------------------------------------


class Response:
    def __init__(self, status: int, payload: dict[str, Any]) -> None:
        self.status = status
        self.payload = payload

    @property
    def code(self) -> str | None:
        return (self.payload.get("error") or {}).get("code")

    @property
    def message(self) -> str:
        return (self.payload.get("error") or {}).get("message", "")


def request(
    app: ToolboxRouter,
    method: str,
    path: str,
    *,
    token: str | None = None,
    body: dict[str, Any] | None = None,
    raw_body: bytes | None = None,
    headers: list[tuple[bytes, bytes]] | None = None,
) -> Response:
    """One HTTP request, with no HTTP. Drives the ASGI protocol by hand."""
    header_list = list(headers or [])
    if token is not None:
        for name, value in auth_header(token).items():
            header_list.append((name.lower().encode(), value.encode()))

    if raw_body is None:
        raw_body = json.dumps(body).encode() if body is not None else b""

    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": header_list,
    }
    sent: list[dict[str, Any]] = []
    delivered = {"done": False}

    async def receive() -> dict[str, Any]:
        if delivered["done"]:
            return {"type": "http.disconnect"}
        delivered["done"] = True
        return {"type": "http.request", "body": raw_body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    asyncio.run(app(scope, receive, send))

    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    payload_bytes = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )
    return Response(status, json.loads(payload_bytes))


# --- driving it over real HTTP, for the client and CLI -----------------------


class _Handler(BaseHTTPRequestHandler):
    """Minimal ASGI-over-http.server bridge. Enough for one JSON request."""

    app: ToolboxRouter

    def do_GET(self) -> None:
        self._serve("GET", b"")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self._serve("POST", self.rfile.read(length))

    def _serve(self, method: str, body: bytes) -> None:
        headers = [
            (k.lower().encode("latin-1"), v.encode("latin-1"))
            for k, v in self.headers.items()
        ]
        scope = {
            "type": "http",
            "method": method,
            "path": self.path.split("?", 1)[0],
            "headers": headers,
        }
        sent: list[dict[str, Any]] = []
        delivered = {"done": False}

        async def receive() -> dict[str, Any]:
            if delivered["done"]:
                return {"type": "http.disconnect"}
            delivered["done"] = True
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        asyncio.run(type(self).app(scope, receive, send))

        start = next(m for m in sent if m["type"] == "http.response.start")
        payload = b"".join(
            m.get("body", b"") for m in sent if m["type"] == "http.response.body"
        )
        self.send_response(start["status"])
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args: Any) -> None:
        """Silence. The default handler writes every request to stderr."""


@pytest.fixture
def live_broker(router: ToolboxRouter) -> Iterator[str]:
    """The real router, served over loopback HTTP. Returns its base URL.

    Worth the forty lines: it is the only way to exercise `ToolboxClient`'s
    urllib path, its HTTPError handling, and the CLI's exit codes against the
    actual server rather than against a mock that agrees with them by
    construction.
    """
    handler = type("BoundHandler", (_Handler,), {"app": router})
    server = HTTPServer(("127.0.0.1", 0), handler)
    # poll_interval well under the default 0.5s: `shutdown()` blocks for up to
    # one interval, and paying half a second per test to tear down a loopback
    # server is how a fast suite becomes one nobody runs.
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
