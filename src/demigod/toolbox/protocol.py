"""THE WIRE CONTRACT between a DEMI_GOD and the TOOLBOX_BROKER.

Lives in `demigod`, not `reagents` or `broker`, for the same reason
`demigod.schema` does: it has to be importable INSIDE the sandbox, and the
dependency direction is one-way. `demigod` never imports `reagents`; the broker
imports `demigod`. So this module is the shared vocabulary and it is the
*smallest* half -- the demigod learns the shape of a request and nothing about
who serves it.

WHY HTTP/JSON AND NOT gRPC
--------------------------
Call volume is tens per demigod, not thousands, and latency is dominated by tool
execution rather than transport. What gRPC would buy is a typed contract, and we
get that for free: every tool already carries a `ToolSpec.parameters_schema`
(JSON Schema), which the broker validates server-side before dispatch. What
HTTP buys that gRPC does not is that a human can reproduce any call with `curl`,
which is worth a great deal when the client is an LLM writing bash.

WHY THE LEASE IS THE TOKEN
--------------------------
`Authorization: Bearer <lease_id>`. `CapabilityLease` already names the tools,
the call budget, the wall clock, and whether writes are permitted -- everything
a credential would have to say. Minting a second secret alongside it would mean
two things to revoke and two things to get out of sync.

WHAT A DEMI_GOD HOLDS
---------------------
A URL and a lease id. That is the entire capability surface. It never holds a
Modal token: Modal credentials are workspace-wide, so a demigod with one could
spawn sandboxes and read every sibling's output volume, which is precisely the
isolation this system is built to provide.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

PROTOCOL_VERSION = "1"
"""Bumped when a change would break an already-baked demigod image.

The demigod image is pre-baked and the broker is deployed separately, so the two
sides WILL drift. Every response carries this; the client warns loudly on a
mismatch rather than failing a live agent run on a field it did not need.
"""

API_PREFIX = "/v1"

# --- routes ------------------------------------------------------------------
#
# Path templates, formatted by both halves. Kept here so a typo is a shared
# constant rather than two string literals that agree today.

HEALTH_PATH = f"{API_PREFIX}/health"
"""Unauthenticated liveness. The only route that does not require a lease."""

TOOLS_PATH = f"{API_PREFIX}/tools"
"""GET -- the bound catalog plus the lease's remaining budget."""


def describe_path(tool_id: str) -> str:
    """GET -- one tool's full spec, including its JSON Schema."""
    return f"{TOOLS_PATH}/{_quote(tool_id)}"


def call_path(tool_id: str) -> str:
    """POST -- execute one tool. Body is `CallRequest`."""
    return f"{TOOLS_PATH}/{_quote(tool_id)}/call"


def _quote(tool_id: str) -> str:
    from urllib.parse import quote

    return quote(tool_id, safe="")


AUTH_HEADER = "Authorization"
AUTH_SCHEME = "Bearer"


def auth_header(lease_id: str) -> dict[str, str]:
    """The one header a demigod must send. There is no second credential."""
    return {AUTH_HEADER: f"{AUTH_SCHEME} {lease_id}"}


# --- error codes -------------------------------------------------------------
#
# A closed set, because the agent-facing CLI branches on them and an LLM reading
# free-text errors will invent a retry strategy for a permanent failure.


class ErrorCode:
    """Stable machine-readable reasons a call did not produce a result."""

    UNAUTHORIZED = "unauthorized"
    """No lease, malformed header, or a lease the broker has never issued."""

    LEASE_EXPIRED = "lease_expired"
    """Past `wall_time_s`. Not retryable; the run is over."""

    LEASE_EXHAUSTED = "lease_exhausted"
    """`max_calls` spent. Not retryable. Report it as a blocker."""

    UNBOUND_TOOL = "unbound_tool"
    """A real tool, but not in THIS lease. Not retryable."""

    WRITE_DENIED = "write_denied"
    """A write-access tool under a read-only lease. Operator decision."""

    UNKNOWN_TOOL = "unknown_tool"
    """No such tool anywhere in the broker's catalog. Check `toolbox list`."""

    INVALID_INPUT = "invalid_input"
    """Input failed the tool's `parameters_schema`. RETRYABLE -- fix and resend."""

    TOOL_ERROR = "tool_error"
    """The tool ran and raised. Retryable only if the input caused it."""

    UPSTREAM_ERROR = "upstream_error"
    """The broker itself failed (dispatch, transport). Retry once, then blocker."""

    BAD_REQUEST = "bad_request"
    """Malformed envelope: unparseable JSON, wrong method, unknown route."""


RETRYABLE_CODES = frozenset({ErrorCode.INVALID_INPUT, ErrorCode.UPSTREAM_ERROR})
"""Codes where trying again -- with DIFFERENT input -- can plausibly work.
Everything else means stop and record a blocker: an agent that retries an
exhausted lease burns its remaining turns discovering the same wall.

This is advice for the AGENT, not for the transport. See below."""

TRANSPORT_RETRY_CODES = frozenset({ErrorCode.UPSTREAM_ERROR})
"""Codes where resending the IDENTICAL request can work. Strictly smaller than
`RETRYABLE_CODES`, and the distinction is not pedantry: `invalid_input` is
retryable by a human or an agent who changes the arguments, but a client that
resends the same bad body just pays the latency twice and reports the same
error."""

# HTTP status per code. The CLI branches on `error.code`, never on the status,
# but a status that matches the meaning is what makes `curl -f` and every proxy
# in between behave sensibly.
STATUS_FOR_CODE: dict[str, int] = {
    ErrorCode.BAD_REQUEST: 400,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.WRITE_DENIED: 403,
    ErrorCode.UNBOUND_TOOL: 403,
    ErrorCode.UNKNOWN_TOOL: 404,
    ErrorCode.LEASE_EXPIRED: 410,
    ErrorCode.INVALID_INPUT: 422,
    ErrorCode.LEASE_EXHAUSTED: 429,
    ErrorCode.UPSTREAM_ERROR: 502,
}
"""TOOL_ERROR is deliberately absent: a tool that ran and raised is a 200 with
`ok=false`. The CALL succeeded, the tool disagreed with its input, and the agent
should read the message and adapt rather than treat it as a broken endpoint.
Same convention MCP uses with `isError`."""


# --- payloads ----------------------------------------------------------------


class ToolboxGrant(BaseModel):
    """Everything a DEMI_GOD is given in order to hold tools. Nothing else.

    Rides inside `DemiGodSpec`, so it is materialized into the sandbox by the
    same `spec.json` write the runner already performs, and it is visible in the
    manifest of what a demigod was authorized to do.
    """

    url: str = Field(..., description="Broker base URL, no trailing slash.")
    lease_id: str = Field(
        ...,
        description=(
            "The bearer token AND the CapabilityLease id. One concept: the "
            "lease already bounds tools, calls, wall time, and write access."
        ),
    )
    tool_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Advisory copy of what `toolbox list` will return. Present so the "
            "system prompt can name the tools without a network call at "
            "prompt-build time. The broker is authoritative."
        ),
    )
    expires_at: float | None = Field(
        None, description="Unix epoch seconds. Informational; the broker enforces."
    )

    @property
    def base(self) -> str:
        return self.url.rstrip("/")


class BrokeredTool(BaseModel):
    """One tool as a DEMI_GOD sees it. A projection of `reagents.ToolSpec`.

    Deliberately NOT ToolSpec itself: `provider` and `cost_class` describe how
    the BROKER executes a tool, which is not a demigod's business and would only
    invite it to reason about infrastructure it cannot see.
    """

    id: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    access: str = "compute"
    """read | compute | write. `write` calls need a write-enabled lease."""


class LeaseStatus(BaseModel):
    """The budget, as the broker currently sees it. Returned by `toolbox list`.

    An agent that cannot see its remaining calls spends them on exploration and
    then discovers the wall on the call that mattered.
    """

    lease_id: str
    calls_used: int
    max_calls: int
    expires_in_s: float | None = None
    allow_write: bool = False

    @property
    def calls_remaining(self) -> int:
        return max(self.max_calls - self.calls_used, 0)


class ListResponse(BaseModel):
    protocol: str = PROTOCOL_VERSION
    lease: LeaseStatus
    tools: list[BrokeredTool] = Field(default_factory=list)


class DescribeResponse(BaseModel):
    protocol: str = PROTOCOL_VERSION
    tool: BrokeredTool


class CallRequest(BaseModel):
    """POST body for a call. One field, so `--input file.json` can be the file.

    The arguments are nested under `input` rather than posted bare so the
    envelope has somewhere to grow (idempotency keys, per-call timeouts) without
    colliding with a tool that happens to have a parameter of the same name.
    """

    input: dict[str, Any] = Field(default_factory=dict)


class ToolError(BaseModel):
    code: str
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class CallResponse(BaseModel):
    """The result of one call, successful or not. One shape either way.

    Same rule as `DemiGodResult`: failure is a field, not a different type, so
    the CLI has one parse path and the trace has one row shape.
    """

    protocol: str = PROTOCOL_VERSION
    ok: bool
    tool: str
    call_id: str = ""
    result: Any = None
    error: ToolError | None = None
    duration_ms: float = 0.0
    lease: LeaseStatus | None = None


class ErrorResponse(BaseModel):
    """Non-200 body. Always JSON, never an HTML error page."""

    protocol: str = PROTOCOL_VERSION
    ok: bool = False
    error: ToolError


__all__ = [
    "API_PREFIX",
    "AUTH_HEADER",
    "AUTH_SCHEME",
    "HEALTH_PATH",
    "PROTOCOL_VERSION",
    "RETRYABLE_CODES",
    "STATUS_FOR_CODE",
    "TOOLS_PATH",
    "TRANSPORT_RETRY_CODES",
    "BrokeredTool",
    "CallRequest",
    "CallResponse",
    "DescribeResponse",
    "ErrorCode",
    "ErrorResponse",
    "LeaseStatus",
    "ListResponse",
    "ToolError",
    "ToolboxGrant",
    "auth_header",
    "call_path",
    "describe_path",
]
