"""The DEMI_GOD's side of the wire. Standard library only, on purpose.

This module runs inside a pre-baked image alongside an agent that is spending
billed turns. Every dependency it takes is a dependency that can fail to
install, drift between bakes, or collide with a tool package -- so it takes
none beyond `pydantic`, which the agent runtime already pins.

There is no Modal import here and there must never be one. A DEMI_GOD holds a
URL and a lease id; if it could import a Modal client it would be one env var
away from workspace-wide credentials, and workspace-wide credentials mean it can
enumerate its siblings' output volumes.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from demigod.toolbox.protocol import (
    HEALTH_PATH,
    PROTOCOL_VERSION,
    TOOLS_PATH,
    TRANSPORT_RETRY_CODES,
    CallRequest,
    CallResponse,
    DescribeResponse,
    ErrorCode,
    ListResponse,
    ToolboxGrant,
    ToolError,
    auth_header,
    call_path,
    describe_path,
)

ENV_URL = "TOOLBOX_URL"
ENV_LEASE = "TOOLBOX_LEASE"
ENV_GRANT_FILE = "TOOLBOX_GRANT"

DEFAULT_GRANT_FILE = "/run/toolbox.json"
"""Where the runner materializes the grant, next to `/run/spec.json`.

Belt and braces with the env vars. The agent composes bash, and bash composes
sub-shells, `env -i`, `sudo`, and heredocs -- any of which can lose an exported
variable. A file on a control-plane path outside both volume mounts cannot be
lost that way, and it is not in the agent's output dir so it never pollutes
`files`.
"""

DEFAULT_TIMEOUT_S = 180.0
"""Generous: the broker runs the tool synchronously and a heavy tool dispatched
to its own container may cold-start. Shorter than any sandbox idle timeout so a
hung call surfaces as an error the agent can record, not as a killed sandbox."""


class ToolboxError(RuntimeError):
    """A call did not produce a result, and retrying the same thing will not help.

    Carries the protocol's `ErrorCode` so a caller can branch on a closed set
    rather than on message text.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        detail: dict[str, Any] | None = None,
        status: int | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.detail = detail or {}
        self.status = status

    @property
    def retryable(self) -> bool:
        from demigod.toolbox.protocol import RETRYABLE_CODES

        return self.code in RETRYABLE_CODES


class ToolboxClient:
    """One lease, one broker. Synchronous -- the agent's bash call is too."""

    def __init__(
        self,
        url: str,
        lease_id: str,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        attempts: int = 2,
    ) -> None:
        if not url:
            raise ToolboxError(ErrorCode.BAD_REQUEST, "empty broker url")
        if not lease_id:
            raise ToolboxError(ErrorCode.UNAUTHORIZED, "empty lease id")
        self.base = url.rstrip("/")
        self.lease_id = lease_id
        self.timeout_s = timeout_s
        # Two, not five. A retry storm against an exhausted lease burns wall
        # clock the agent needs for its manifest, and every code except
        # UPSTREAM_ERROR is permanent by construction.
        self.attempts = max(1, attempts)
        self.protocol_warning: str | None = None

    # --- construction -------------------------------------------------------

    @classmethod
    def from_grant(cls, grant: ToolboxGrant, **kwargs: Any) -> ToolboxClient:
        return cls(grant.base, grant.lease_id, **kwargs)

    @classmethod
    def from_env(cls, **kwargs: Any) -> ToolboxClient:
        """Env vars first, then the grant file. Raises if neither is present.

        The error message names both mechanisms, because "no toolbox" and
        "toolbox misconfigured" look identical to an agent otherwise, and the
        first is a legitimate state (a demigod may be granted no tools at all).
        """
        url = os.environ.get(ENV_URL)
        lease = os.environ.get(ENV_LEASE)
        if url and lease:
            return cls(url, lease, **kwargs)

        path = Path(os.environ.get(ENV_GRANT_FILE) or DEFAULT_GRANT_FILE)
        if path.exists():
            try:
                grant = ToolboxGrant.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                raise ToolboxError(
                    ErrorCode.BAD_REQUEST, f"{path} is not a valid grant: {exc}"
                ) from exc
            return cls.from_grant(grant, **kwargs)

        raise ToolboxError(
            ErrorCode.UNAUTHORIZED,
            f"no toolbox grant: set {ENV_URL} and {ENV_LEASE}, or provide "
            f"{path}. If you were given no tools, this is expected -- record it "
            f"in `blockers` rather than working around it.",
        )

    # --- operations ---------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Unauthenticated. Distinguishes 'broker down' from 'lease refused'."""
        return self._request("GET", HEALTH_PATH, authed=False)

    def list_tools(self) -> ListResponse:
        return ListResponse.model_validate(self._request("GET", TOOLS_PATH))

    def describe(self, tool_id: str) -> DescribeResponse:
        return DescribeResponse.model_validate(
            self._request("GET", describe_path(tool_id))
        )

    def call(self, tool_id: str, arguments: dict[str, Any]) -> CallResponse:
        """Execute one tool. NEVER raises for a refused or failed call.

        Returns a `CallResponse` with `ok=False` instead, because the caller is
        an agent deciding what to do next and it needs the code and the message,
        not a traceback. Transport-level impossibility still raises.
        """
        body = CallRequest(input=arguments).model_dump()
        try:
            payload = self._request("POST", call_path(tool_id), body=body)
        except ToolboxError as exc:
            return CallResponse(
                ok=False,
                tool=tool_id,
                error=ToolError(code=exc.code, message=exc.message, detail=exc.detail),
            )
        return CallResponse.model_validate(payload)

    # --- transport ----------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        authed: bool = True,
    ) -> dict[str, Any]:
        url = f"{self.base}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if authed:
            headers.update(auth_header(self.lease_id))

        last: ToolboxError | None = None
        for attempt in range(self.attempts):
            try:
                return self._once(method, url, data, headers)
            except ToolboxError as exc:
                last = exc
                # TRANSPORT_RETRY_CODES, not RETRYABLE_CODES. `invalid_input` is
                # retryable by an agent that changes its arguments; resending
                # the identical body just pays the latency twice and reports the
                # same error, out of a turn budget that is the scarce resource.
                if (
                    exc.code not in TRANSPORT_RETRY_CODES
                    or attempt == self.attempts - 1
                ):
                    raise
                time.sleep(0.5 * (attempt + 1))
        raise last  # pragma: no cover - loop always returns or raises

    def _once(
        self,
        method: str,
        url: str,
        data: bytes | None,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise self._from_http_error(exc) from exc
        except urllib.error.URLError as exc:
            raise ToolboxError(
                ErrorCode.UPSTREAM_ERROR, f"cannot reach broker at {url}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise ToolboxError(
                ErrorCode.UPSTREAM_ERROR,
                f"broker did not answer within {self.timeout_s}s",
            ) from exc

        payload = self._decode(raw, url)
        self._check_protocol(payload)
        return payload

    def _from_http_error(self, exc: urllib.error.HTTPError) -> ToolboxError:
        """Turn a non-2xx into a typed error, preferring the broker's own code.

        Falls back to the status only when the body is not our JSON -- which
        means something between us and the broker answered (a proxy, a 404 page),
        and saying so beats reporting a bare status.
        """
        try:
            payload = json.loads(exc.read().decode("utf-8"))
            err = payload.get("error") or {}
            code = err.get("code")
            if code:
                return ToolboxError(
                    code,
                    err.get("message", ""),
                    detail=err.get("detail") or {},
                    status=exc.code,
                )
        except Exception:
            pass
        code = (
            ErrorCode.UNAUTHORIZED
            if exc.code in (401, 403)
            else ErrorCode.UPSTREAM_ERROR
        )
        return ToolboxError(code, f"broker returned HTTP {exc.code}", status=exc.code)

    @staticmethod
    def _decode(raw: bytes, url: str) -> dict[str, Any]:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ToolboxError(
                ErrorCode.UPSTREAM_ERROR,
                f"broker at {url} returned non-JSON ({raw[:120]!r})",
            ) from exc
        if not isinstance(payload, dict):
            raise ToolboxError(
                ErrorCode.UPSTREAM_ERROR, f"broker at {url} returned {type(payload)}"
            )
        return payload

    def _check_protocol(self, payload: dict[str, Any]) -> None:
        """Warn on a version mismatch; never fail on one.

        The demigod image is pre-baked and the broker is deployed separately, so
        drift is a matter of time. Killing a live agent run over a version field
        it did not read would be a strictly worse outcome than a warning.
        """
        seen = payload.get("protocol")
        if seen and seen != PROTOCOL_VERSION and self.protocol_warning is None:
            self.protocol_warning = (
                f"broker speaks protocol {seen}, this client speaks "
                f"{PROTOCOL_VERSION}; fields may be missing"
            )


__all__ = [
    "DEFAULT_GRANT_FILE",
    "ENV_GRANT_FILE",
    "ENV_LEASE",
    "ENV_URL",
    "ToolboxClient",
    "ToolboxError",
]
