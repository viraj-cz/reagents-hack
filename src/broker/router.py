"""The broker's request path. A plain ASGI app -- no framework, no Modal.

Two consequences of that, both deliberate:

* The whole authorization path is testable offline, by calling this object with
  an ASGI scope. No account, no key, no network, no server. `tests/test_broker_
  router.py` exercises every refusal that way.
* The broker image needs no web framework. `modal.fastapi_endpoint` would drag
  FastAPI (verified: `modal/_runtime/asgi.py` imports it only for that path);
  `modal.asgi_app` accepts any ASGI callable, so the image is Python plus this
  repo.

WHAT IT DOES NOT DO. It does not decide policy. `reagents.tools.registry.
ToolBroker` -- already written, already tested -- remains the only code that
decides whether a call is allowed, and the only path from a request to an
executor. This module resolves the lease, seeds that broker from durable state,
translates its refusals into HTTP, and writes the trace.

MINIMAL DISCLOSURE. A tool that is not in the caller's lease is reported as
`unbound_tool` whether or not it exists in the catalog. A demigod's picture of
the world should be exactly its lease -- being able to probe for the existence
of `remote.publish` is a capability nobody granted it.
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote

from broker.dispatch import DispatchPolicy, DispatchUnavailableError
from broker.grants import Grant, GrantStore, TraceEntry, new_call_id
from demigod.schema import validate_payload
from demigod.toolbox.protocol import (
    API_PREFIX,
    AUTH_SCHEME,
    HEALTH_PATH,
    PROTOCOL_VERSION,
    STATUS_FOR_CODE,
    TOOLS_PATH,
    BrokeredTool,
    CallResponse,
    DescribeResponse,
    ErrorCode,
    ErrorResponse,
    LeaseStatus,
    ListResponse,
    ToolError,
)
from reagents.tools.registry import (
    Tool,
    ToolBroker,
    ToolPolicyError,
    ToolRegistry,
    UnboundToolError,
    UnknownToolError,
)

MAX_BODY_BYTES = 4 * 1024 * 1024
"""A tool argument object, not a file upload. Bounded so a runaway agent cannot
turn the broker into its own memory."""


class Refusal(Exception):
    """A request that will not produce a result, with the reason already typed."""

    def __init__(
        self, code: str, message: str, detail: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail or {}

    @property
    def status(self) -> int:
        return STATUS_FOR_CODE.get(self.code, 400)

    def as_error(self) -> ToolError:
        return ToolError(code=self.code, message=self.message, detail=self.detail)


@dataclass
class ToolboxRouter:
    """The ASGI application. One instance per broker container."""

    registry: ToolRegistry
    store: GrantStore
    dispatch: DispatchPolicy = field(default_factory=DispatchPolicy)
    """Inline-only by default: a router built without executor functions still
    serves every LOCAL tool, and names the missing tier if asked for another."""

    clock: Callable[[], float] = time.time
    """Injected so a test can expire a lease without sleeping through it."""

    # --- ASGI ---------------------------------------------------------------

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope["type"] != "http":  # pragma: no cover - websockets are not served
            return
        try:
            status, payload = await self._handle(scope, receive)
        except Refusal as refusal:
            status = refusal.status
            payload = ErrorResponse(error=refusal.as_error()).model_dump()
        except Exception as exc:
            traceback.print_exc()
            status = 500
            payload = ErrorResponse(
                error=ToolError(
                    code=ErrorCode.UPSTREAM_ERROR,
                    message=f"broker fault: {type(exc).__name__}: {exc}",
                )
            ).model_dump()
        await _respond(send, status, payload)

    @staticmethod
    async def _lifespan(receive: Callable, send: Callable) -> None:
        """Answer startup/shutdown so an ASGI server does not hang waiting.

        Nothing to do -- the registry and store are built at container start by
        `broker.service`, not per request -- but silence here reads as a stuck
        app to the server.
        """
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return

    # --- routing ------------------------------------------------------------

    async def _handle(
        self, scope: dict, receive: Callable
    ) -> tuple[int, dict[str, Any]]:
        method = scope.get("method", "GET").upper()
        path = scope.get("path", "/").rstrip("/") or "/"

        if path == HEALTH_PATH:
            return 200, {
                "ok": True,
                "protocol": PROTOCOL_VERSION,
                "service": "toolbox-broker",
            }

        if not path.startswith(API_PREFIX):
            raise Refusal(
                ErrorCode.BAD_REQUEST,
                f"no such route {path!r}; the API is under {API_PREFIX}",
            )

        grant = await self._authenticate(scope)

        if path == TOOLS_PATH:
            _require(method, "GET", path)
            return 200, (await self._list(grant)).model_dump()

        if path.startswith(f"{TOOLS_PATH}/"):
            rest = path[len(TOOLS_PATH) + 1 :]
            if rest.endswith("/call"):
                _require(method, "POST", path)
                tool_id = unquote(rest[: -len("/call")])
                body = await _read_body(receive)
                return 200, (await self._call(grant, tool_id, body)).model_dump()
            _require(method, "GET", path)
            return 200, self._describe(grant, unquote(rest)).model_dump()

        raise Refusal(ErrorCode.BAD_REQUEST, f"no such route {path!r}")

    # --- auth ---------------------------------------------------------------

    async def _authenticate(self, scope: dict) -> Grant:
        """Bearer token -> Grant. The token IS the lease id.

        Expiry is checked here rather than at call time so `toolbox list` on a
        dead lease says "expired" instead of listing tools that cannot be
        called.
        """
        token = _bearer(scope)
        if not token:
            raise Refusal(
                ErrorCode.UNAUTHORIZED,
                f"missing {AUTH_SCHEME} token; send your lease id as "
                f"`Authorization: {AUTH_SCHEME} <lease_id>`",
            )
        grant = await self._store(self.store.read, token)
        if grant is None:
            # Same message for "never existed" and "revoked". Distinguishing
            # them would let a caller test lease ids.
            raise Refusal(ErrorCode.UNAUTHORIZED, "unknown or revoked lease")
        if grant.expires_in(self.clock()) <= 0:
            raise Refusal(
                ErrorCode.LEASE_EXPIRED,
                f"lease expired {abs(grant.expires_in(self.clock())):.0f}s ago "
                f"({grant.lease.wall_time_s:.0f}s budget)",
            )
        return grant

    # --- handlers -----------------------------------------------------------

    async def _list(self, grant: Grant) -> ListResponse:
        tools: list[BrokeredTool] = []
        for tool_id in grant.lease.tool_ids:
            try:
                tools.append(_project(self.registry.get(tool_id)))
            except UnknownToolError:
                # The lease names a tool this broker does not have. That is a
                # deployment mismatch, not the demigod's problem -- omit it here
                # and let the call itself report unknown_tool with the detail.
                continue
        return ListResponse(lease=await self._status(grant), tools=tools)

    def _describe(self, grant: Grant, tool_id: str) -> DescribeResponse:
        return DescribeResponse(tool=_project(self._authorized_tool(grant, tool_id)))

    async def _call(
        self, grant: Grant, tool_id: str, body: dict[str, Any]
    ) -> CallResponse:
        call_id = new_call_id()
        arguments = _arguments(body)

        # Read ONCE. Three separate reads (budget check, lease seed, response)
        # would be three network round trips to the grant store per tool call,
        # on the critical path of an agent that is being billed per turn.
        used = await self._store(self.store.calls_used, grant.lease_id)

        try:
            tool = self._authorized_tool(grant, tool_id)
            self._check_budget(grant, used)
            _validate(tool, arguments)
        except Refusal as refusal:
            # Recorded, but NOT metered. A demigod probing for a tool it was
            # never granted belongs in the audit log; charging it a call would
            # mean a typo'd tool name costs real work.
            await self._store(
                self.store.record,
                TraceEntry(
                    call_id=call_id,
                    tool=tool_id,
                    input=arguments,
                    ok=False,
                    error=refusal.as_error().model_dump(),
                    metered=False,
                ),
                lease_id=grant.lease_id,
            )
            raise

        started = time.perf_counter()
        ok = True
        result: Any = None
        error: ToolError | None = None
        try:
            result = await self._execute(grant, tool, arguments, used)
        except (UnboundToolError, ToolPolicyError) as exc:
            # The lease broker refused after our pre-checks passed. Either the
            # budget was consumed by a concurrent call or the pre-checks and the
            # policy have drifted; both are worth reporting precisely.
            ok = False
            error = ToolError(
                code=ErrorCode.LEASE_EXHAUSTED,
                message=str(exc),
                detail={"authoritative": True},
            )
        except DispatchUnavailableError as exc:
            ok = False
            error = ToolError(code=ErrorCode.UPSTREAM_ERROR, message=str(exc))
        except Exception as exc:
            ok = False
            error = ToolError(
                code=ErrorCode.TOOL_ERROR,
                message=f"{type(exc).__name__}: {exc}",
                detail={"tool": tool_id},
            )
        duration_ms = (time.perf_counter() - started) * 1000

        # Metered whether or not the tool succeeded: it ran, it cost something,
        # and a lease that only charges for success is a lease an agent can
        # exhaust for free by calling badly.
        await self._store(
            self.store.record,
            TraceEntry(
                call_id=call_id,
                tool=tool_id,
                input=arguments,
                result=_summarize(result) if ok else None,
                ok=ok,
                error=error.model_dump() if error else None,
                duration_ms=duration_ms,
                metered=True,
            ),
            lease_id=grant.lease_id,
        )

        return CallResponse(
            ok=ok,
            tool=tool_id,
            call_id=call_id,
            result=result if ok else None,
            error=error,
            duration_ms=duration_ms,
            # `used + 1` rather than a fourth round trip: we just recorded
            # exactly one metered entry, so the count is known.
            lease=await self._status(grant, calls_used=used + 1),
        )

    # --- policy -------------------------------------------------------------

    def _authorized_tool(self, grant: Grant, tool_id: str) -> Tool:
        if tool_id not in grant.lease.tool_ids:
            raise Refusal(
                ErrorCode.UNBOUND_TOOL,
                f"tool {tool_id!r} is not in this lease",
                {"granted": sorted(grant.lease.tool_ids)},
            )
        try:
            tool = self.registry.get(tool_id)
        except UnknownToolError as exc:
            raise Refusal(
                ErrorCode.UNKNOWN_TOOL,
                f"tool {tool_id!r} is in your lease but this broker does not "
                f"have it installed. Report it as a blocker.",
            ) from exc
        if tool.access.value == "write" and not grant.lease.allow_write:
            raise Refusal(
                ErrorCode.WRITE_DENIED,
                f"tool {tool_id!r} writes, and this lease is read-only. Write "
                f"authority is an operator decision, not a retry.",
            )
        return tool

    def _check_budget(self, grant: Grant, used: int) -> None:
        if used >= grant.lease.max_calls:
            raise Refusal(
                ErrorCode.LEASE_EXHAUSTED,
                f"lease spent all {grant.lease.max_calls} calls",
                {"calls_used": used},
            )

    async def _execute(
        self, grant: Grant, tool: Tool, arguments: dict, used: int
    ) -> Any:
        """Run the call through `ToolBroker`, seeded from durable state.

        This is the point of the whole module. `ToolBroker` holds `calls` and
        `started_at` in memory, which is meaningless across autoscaled replicas
        -- so we hand it the numbers the store knows and let it apply exactly
        the policy it applies in-process. The enforcement code is the tested
        one; only where its two counters come from has changed.
        """
        elapsed = max(self.clock() - grant.issued_at, 0.0)
        prepared = self.dispatch.prepare(tool)
        broker = ToolBroker(
            tools={tool.id: prepared},
            lease=grant.lease,
            calls=used,
            started_at=time.monotonic() - elapsed,
        )
        return await broker.acall(tool.id, **arguments)

    async def _status(
        self, grant: Grant, *, calls_used: int | None = None
    ) -> LeaseStatus:
        if calls_used is None:
            calls_used = await self._store(self.store.calls_used, grant.lease_id)
        return LeaseStatus(
            lease_id=grant.lease_id,
            calls_used=calls_used,
            max_calls=grant.lease.max_calls,
            expires_in_s=max(grant.expires_in(self.clock()), 0.0),
            allow_write=grant.lease.allow_write,
        )

    # --- the one place that touches the store -------------------------------

    async def _store(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Every `GrantStore` call goes through here, off the event loop.

        FOUND LIVE, not reasoned about. The first deployed endpoint answered
        correctly and logged:

            AsyncUsageWarning: A blocking Modal interface is being used in an
            async context.  Original line: raw = self.grants.get(lease_id)

        `ModalGrantStore` talks to modal.Dict/Queue over Modal's blocking
        interface, and this router is ASGI. A blocking gRPC round trip on the
        event loop stalls every other demigod sharing the replica -- which is
        exactly the bottleneck a Modal Function was chosen over a single broker
        sandbox to avoid.

        `to_thread` rather than an async `GrantStore` protocol: the other
        implementation (`InMemoryGrantStore`, which is pure memory) and the
        other caller (GOD, which is synchronous at the boundary) would both have
        to grow an async half to serve one implementation's transport.
        """
        return await asyncio.to_thread(fn, *args, **kwargs)


# --- helpers -----------------------------------------------------------------


def _project(tool: Tool) -> BrokeredTool:
    """`Tool` -> what a demigod is allowed to see.

    `provider`, `cost_class` and `latency_class` are dropped: they describe how
    the broker executes a tool, and a demigod told that a tool is `container`
    and `expensive` will start reasoning about infrastructure it cannot see.
    """
    return BrokeredTool(
        id=tool.id,
        description=tool.description,
        input_schema=tool.parameters_schema,
        output_schema=tool.output_schema,
        access=tool.access.value,
    )


def _validate(tool: Tool, arguments: dict[str, Any]) -> None:
    """Server-side JSON Schema check. This is what gRPC would have bought us.

    Two checks, and the second is the one that matters in practice: arguments
    become Python kwargs, so an unexpected key is a `TypeError` from deep inside
    a tool rather than a message the agent can act on. Reported as
    `invalid_input`, which is the only retryable refusal in the protocol.
    """
    schema = tool.parameters_schema or {}
    errors = validate_payload(arguments, schema)

    properties = schema.get("properties")
    closed = (
        schema.get("type") == "object"
        and isinstance(properties, dict)
        and bool(properties)
        and schema.get("additionalProperties") is not True
    )
    if closed:
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            errors.append(
                f"$: unexpected argument(s) {unknown}; accepted: {sorted(properties)}"
            )

    if errors:
        raise Refusal(
            ErrorCode.INVALID_INPUT,
            f"input does not satisfy the schema for {tool.id!r}",
            {"errors": errors, "schema": schema},
        )


def _arguments(body: dict[str, Any]) -> dict[str, Any]:
    arguments = body.get("input", {})
    if not isinstance(arguments, dict):
        raise Refusal(
            ErrorCode.BAD_REQUEST,
            f"`input` must be a JSON object, got {type(arguments).__name__}",
        )
    return arguments


def _summarize(result: Any) -> Any:
    """Keep the trace bounded. GOD reads it; it is not the artifact store.

    A tool returning a 50k-row frame would otherwise put 50k rows into every
    `DemiGodResult.tool_trace`, and the trace exists to say what was called, not
    to duplicate the output the demigod already wrote to its volume.
    """
    encoded = json.dumps(result, default=str)
    if len(encoded) <= 4096:
        return result
    return {
        "truncated": True,
        "bytes": len(encoded),
        "preview": encoded[:2000],
    }


def _bearer(scope: dict) -> str | None:
    for raw_name, raw_value in scope.get("headers", []):
        if raw_name.decode("latin-1").lower() != "authorization":
            continue
        value = raw_value.decode("latin-1").strip()
        prefix = f"{AUTH_SCHEME} "
        if value.lower().startswith(prefix.lower()):
            return value[len(prefix) :].strip() or None
        return None
    return None


def _require(method: str, expected: str, path: str) -> None:
    if method != expected:
        raise Refusal(ErrorCode.BAD_REQUEST, f"{path} accepts {expected}, not {method}")


async def _read_body(receive: Callable[[], Awaitable[dict]]) -> dict[str, Any]:
    chunks: list[bytes] = []
    size = 0
    while True:
        message = await receive()
        if message["type"] != "http.request":  # pragma: no cover - client hangup
            break
        chunk = message.get("body", b"")
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            raise Refusal(
                ErrorCode.BAD_REQUEST, f"request body exceeds {MAX_BODY_BYTES} bytes"
            )
        chunks.append(chunk)
        if not message.get("more_body"):
            break
    raw = b"".join(chunks)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise Refusal(ErrorCode.BAD_REQUEST, f"body is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise Refusal(ErrorCode.BAD_REQUEST, "body must be a JSON object")
    return parsed


async def _respond(send: Callable, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, default=str).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


__all__ = ["MAX_BODY_BYTES", "Refusal", "ToolboxRouter"]
