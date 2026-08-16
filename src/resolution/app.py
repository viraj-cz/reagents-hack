"""The re:SOLUTION HTTP surface. A plain ASGI app, like `broker/router.py`.

No web framework, for the reason given in that module: this repo already ships
one ASGI app with no framework behind it, and the second one should not drag a
dependency in for four routes and a text/event-stream. It also means the whole
surface is callable from a test with a scope dict.

Routes
------
GET  /api/health                    liveness
GET  /api/presets                   problems the UI can start from
POST /api/uploads?name=x.csv        attach a table -> {id, profile, terms}
GET  /api/runs                      snapshots, newest first
POST /api/runs                      start a run -> {run_id}
GET  /api/runs/{id}                 snapshot + full event log
GET  /api/runs/{id}/stream          SSE: replay, then live, then close
GET  /api/runs/{id}/transcript/{n}  every event for one node
POST /api/runs/{id}/cancel          stop a run in flight
*                                   the built frontend, SPA-fallback to index
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import mimetypes
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from resolution.attachments import (
    MAX_UPLOAD_BYTES,
    AttachmentError,
    AttachmentStore,
)
from resolution.runs import (
    EXECUTION_GODBOX,
    EXECUTION_INPROCESS,
    EXECUTION_SANDBOX,
    EXECUTIONS,
    RunStore,
    build_problem,
    with_attachments,
)
from resolution.toy import PRESETS, preset_problem

# Long enough not to be chatty, short enough that a proxy idle timeout (often
# 60s) never fires first. A GOD phase can easily be silent for a minute.
HEARTBEAT_S = 15.0

WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"

_ALLOWED_ORIGIN_HOSTS = ("localhost", "127.0.0.1", "[::1]")


class ResolutionApp:
    def __init__(
        self, *, dist: Path | None = None, uploads: Path | None = None
    ) -> None:
        self.store = RunStore()
        self.attachments = AttachmentStore(uploads)
        self.dist = dist or WEB_DIST

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            await _lifespan(receive, send)
            return
        if scope["type"] != "http":
            return
        method: str = scope["method"]
        path: str = unquote(scope["path"])
        origin = _header(scope, b"origin")

        if method == "OPTIONS":
            await _respond(send, 204, b"", origin=origin)
            return

        try:
            if path.startswith("/api/"):
                await self._api(scope, receive, send, method, path, origin)
            else:
                await self._static(send, path)
        except _HttpError as exc:
            await _json(send, exc.status, {"error": exc.message}, origin=origin)

    # -- api ----------------------------------------------------------------
    async def _api(
        self,
        scope: dict,
        receive: Any,
        send: Any,
        method: str,
        path: str,
        origin: str | None,
    ) -> None:
        parts = [p for p in path.strip("/").split("/") if p]  # ["api", ...]

        if parts == ["api", "health"]:
            await _json(
                send, 200, {"ok": True, "runs": len(self.store.runs)}, origin=origin
            )
            return

        if parts == ["api", "presets"]:
            # The client cannot see the server's environment, and a live run
            # that fails on a missing key three seconds in is a worse answer
            # than a disabled button with the reason in its tooltip.
            live_ok, live_reason = live_support()
            await _json(
                send,
                200,
                {
                    "presets": PRESETS,
                    "live_available": live_ok,
                    "live_unavailable_reason": live_reason,
                },
                origin=origin,
            )
            return

        if parts == ["api", "uploads"] and method == "POST":
            # The file arrives as the raw request body with its name in the
            # query string, not as multipart/form-data. This app has no web
            # framework on purpose (see the module docstring), and a hand-rolled
            # multipart parser is a boundary-splitting bug generator for a
            # feature that never needs more than one file per request. `fetch`
            # sends a File as a body directly, so the client side is simpler too.
            name = _query(scope, "name")
            if not name:
                raise _HttpError(400, "upload needs ?name=<filename>")
            try:
                item = self.attachments.add(name, await _read_body(receive))
            except AttachmentError as exc:
                raise _HttpError(400, str(exc)) from exc
            await _json(send, 201, {"attachment": item.to_json()}, origin=origin)
            return

        if parts == ["api", "runs"] and method == "GET":
            await _json(send, 200, {"runs": self.store.list()}, origin=origin)
            return

        if parts == ["api", "runs"] and method == "POST":
            body = await _read_json(receive)
            await _json(send, 201, self._start(body), origin=origin)
            return

        if len(parts) >= 3 and parts[:2] == ["api", "runs"]:
            run = self.store.get(parts[2])
            if run is None:
                raise _HttpError(404, f"no such run: {parts[2]}")
            tail = parts[3:]

            if not tail and method == "GET":
                await _json(
                    send,
                    200,
                    {
                        "run": run.snapshot.to_json(),
                        "events": [e.to_json() for e in run.events],
                    },
                    origin=origin,
                )
                return
            if tail == ["stream"]:
                await self._stream(run, receive, send, origin)
                return
            if len(tail) == 2 and tail[0] == "transcript":
                await _json(
                    send,
                    200,
                    {"node": tail[1], "events": run.transcript(tail[1])},
                    origin=origin,
                )
                return
            if tail == ["cancel"] and method == "POST":
                if run.task and not run.task.done():
                    run.task.cancel()
                await _json(send, 200, {"cancelled": True}, origin=origin)
                return

        raise _HttpError(404, f"no route for {method} {path}")

    def _start(self, body: dict[str, Any]) -> dict[str, Any]:
        mode = str(body.get("mode") or "scripted")
        if mode not in {"scripted", "live"}:
            raise _HttpError(400, "mode must be 'scripted' or 'live'")
        if mode == "live":
            # Both preconditions, up front. The alternative is a run that starts,
            # streams two phases of narration and then dies on the first model
            # call -- which reads as "the orchestrator broke" rather than "this
            # server was never set up for live inference".
            _require_live_support()
        # `null`/absent means GOD decides how many domains the problem is worth,
        # including none at all. Checked with `is None` rather than falsiness so
        # an explicit 0 is still rejected by the range check below instead of
        # being silently read as "dynamic".
        raw_domains = body.get("domains")
        domain_count: int | None = None
        if raw_domains is not None:
            domain_count = int(raw_domains)
            if not 1 <= domain_count <= 5:
                raise _HttpError(400, "domains must be between 1 and 5")

        execution = str(body.get("execution") or EXECUTION_INPROCESS)
        if execution not in EXECUTIONS:
            raise _HttpError(400, f"execution must be one of {list(EXECUTIONS)}")
        if execution in (EXECUTION_SANDBOX, EXECUTION_GODBOX) and mode != "live":
            # A sandbox demigod runs the real agent against the real API using
            # its own mounted secret. Pairing that with a scripted GOD would
            # spend tokens replaying a recording -- the worst of both.
            raise _HttpError(400, f"{execution} execution requires mode 'live'")

        try:
            attached = self.attachments.resolve(list(body.get("attachments") or []))
        except AttachmentError as exc:
            raise _HttpError(400, str(exc)) from exc
        if attached and mode == "scripted":
            # A replay answers the recorded problem whatever it is handed, so
            # attaching a table to one would show the file accepted, mounted,
            # and silently ignored. Refuse instead of implying it was read.
            raise _HttpError(
                400, "replay mode cannot use attachments; switch to live"
            )

        preset = body.get("preset")
        if mode == "scripted":
            # A replay ignores the prompt: `ScriptedLLM` is keyed by phase name,
            # so it returns the recorded domains whatever the problem says. Run
            # it against arbitrary text and you get a plausible-looking analysis
            # of a problem nobody asked about -- so it is refused rather than
            # quietly answered.
            if not preset:
                raise _HttpError(
                    400,
                    "replay mode can only run a recorded preset; "
                    "pass `preset`, or use mode 'live' for your own problem",
                )
            if not _preset_supports(str(preset), "scripted"):
                raise _HttpError(400, f"no recording exists for preset: {preset}")

        if preset:
            problem = preset_problem(str(preset))
            if problem is None:
                raise _HttpError(400, f"unknown preset: {preset}")
            problem = with_attachments(problem, attached)
        else:
            try:
                problem = build_problem(
                    str(body.get("prompt") or ""),
                    entities=list(body.get("entities") or []),
                    constraints=list(body.get("constraints") or []),
                    attachments=attached,
                )
            except ValueError as exc:
                raise _HttpError(400, str(exc)) from exc

        run = self.store.create(
            problem,
            mode=mode,
            domain_count=domain_count,
            execution=execution,
            attachments=attached,
        )
        return {"run_id": run.run_id, "run": run.snapshot.to_json()}

    async def _stream(
        self, run: Any, receive: Any, send: Any, origin: str | None
    ) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/event-stream"),
                    (b"cache-control", b"no-cache, no-transform"),
                    (b"connection", b"keep-alive"),
                    # Nginx and friends buffer text/event-stream by default,
                    # which turns a live stream into one delivery at the end.
                    (b"x-accel-buffering", b"no"),
                    *_cors(origin),
                ],
            }
        )

        events = run.subscribe()
        disconnected = asyncio.Event()

        async def watch_disconnect() -> None:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    disconnected.set()
                    return

        _EXHAUSTED = object()

        async def next_event() -> Any:
            """One event, or the sentinel. Never raises StopAsyncIteration.

            A Task carrying StopAsyncIteration is awkward to inspect, and the
            caller only needs "is there more".
            """

            try:
                return await anext(events)
            except StopAsyncIteration:
                return _EXHAUSTED

        watcher = asyncio.create_task(watch_disconnect())
        # THE PENDING READ SURVIVES THE HEARTBEAT, and that is the whole point.
        # This was `asyncio.wait_for(anext(events), HEARTBEAT_S)`, which CANCELS
        # the read it is waiting on. The cancellation lands inside `subscribe()`
        # at `await queue.get()`, terminates the generator, and runs its
        # `finally` -- unsubscribing this reader. The next `anext()` then raised
        # StopAsyncIteration, so the loop broke and sent `event: end`, and the
        # client (which treats `end` as "the run is over") closed for good.
        #
        # The effect: any silence longer than HEARTBEAT_S permanently killed the
        # stream while looking healthy -- a keep-alive had just gone out. A live
        # run went quiet for 21s between its last token and `run_end`, so the
        # tab never received the event carrying the final answer and only showed
        # it after a reload replayed the log. `asyncio.wait` instead of
        # `wait_for` leaves the unfinished read pending across as many
        # heartbeats as the silence needs.
        pending: asyncio.Task[Any] | None = None
        try:
            while not disconnected.is_set():
                if pending is None:
                    pending = asyncio.ensure_future(next_event())
                finished, _ = await asyncio.wait(
                    {pending}, timeout=HEARTBEAT_S
                )
                if not finished:
                    # A comment frame: valid SSE, ignored by EventSource, and
                    # enough to keep every intermediary from closing the socket.
                    await send(
                        {
                            "type": "http.response.body",
                            "body": b": keep-alive\n\n",
                            "more_body": True,
                        }
                    )
                    continue
                event = pending.result()
                pending = None
                if event is _EXHAUSTED:
                    break
                await send(
                    {
                        "type": "http.response.body",
                        "body": _sse(event.to_json()),
                        "more_body": True,
                    }
                )
            await send(
                {
                    "type": "http.response.body",
                    "body": _sse({"kind": "stream_end"}, event="end"),
                    "more_body": False,
                }
            )
        finally:
            watcher.cancel()
            # Only now is cancelling the read correct: the client is gone, so
            # tearing the generator down is the intent rather than the accident.
            if pending is not None:
                pending.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pending
            await events.aclose()

    # -- static -------------------------------------------------------------
    async def _static(self, send: Any, path: str) -> None:
        index = self.dist / "index.html"
        if not index.exists():
            await _respond(
                send,
                503,
                _NO_BUILD.encode(),
                content_type=b"text/html; charset=utf-8",
            )
            return

        candidate = (self.dist / path.lstrip("/")).resolve()
        # A path like /../../etc/passwd resolves outside dist; SPA-fallback it
        # rather than serve it.
        inside = (
            self.dist.resolve() in candidate.parents or candidate == self.dist.resolve()
        )
        target = candidate if (inside and candidate.is_file()) else index
        guessed, _ = mimetypes.guess_type(target.name)
        content_type = (guessed or "application/octet-stream").encode()
        if target.suffix in {".html", ".js", ".css", ".json", ".svg"}:
            content_type += b"; charset=utf-8"
        cache = b"no-cache" if target == index else b"public, max-age=3600"
        await _respond(
            send,
            200,
            target.read_bytes(),
            content_type=content_type,
            extra_headers=[(b"cache-control", cache)],
        )


def _preset_supports(preset_id: str, mode: str) -> bool:
    return any(p["id"] == preset_id and mode in p["modes"] for p in PRESETS)


def live_support() -> tuple[bool, str | None]:
    """Can this process run live inference, and if not, what is missing?"""

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False, "live mode needs ANTHROPIC_API_KEY in the server's environment"
    if importlib.util.find_spec("anthropic") is None:
        return False, "live mode needs the llm extra: uv sync --extra llm --extra ui"
    return True, None


def _require_live_support() -> None:
    ok, reason = live_support()
    if not ok:
        raise _HttpError(400, reason or "live mode is unavailable")


class _HttpError(Exception):
    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(message)


def _sse(payload: dict[str, Any], event: str | None = None) -> bytes:
    prefix = f"event: {event}\n" if event else ""
    body = json.dumps(payload, ensure_ascii=False)
    return f"{prefix}data: {body}\n\n".encode()


def _header(scope: dict, name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key.lower() == name:
            return value.decode("latin-1")
    return None


def _cors(origin: str | None) -> list[tuple[bytes, bytes]]:
    """Echo only loopback origins: this is a local tool, not a public API."""

    if not origin:
        return []
    host = origin.split("://")[-1].split(":")[0]
    if host not in _ALLOWED_ORIGIN_HOSTS:
        return []
    return [
        (b"access-control-allow-origin", origin.encode()),
        (b"access-control-allow-headers", b"content-type"),
        (b"access-control-allow-methods", b"GET, POST, OPTIONS"),
        (b"vary", b"origin"),
    ]


def _query(scope: dict, key: str) -> str | None:
    raw = scope.get("query_string") or b""
    for field in raw.decode("latin-1").split("&"):
        name, _, value = field.partition("=")
        if unquote(name.replace("+", " ")) == key:
            return unquote(value.replace("+", " "))
    return None


async def _read_body(receive: Any) -> bytes:
    """Collect a raw request body, refusing one that will not fit anyway.

    The size check happens while reading rather than after: a client that
    ignores the documented limit should not first get to buffer a gigabyte in
    this process's memory.
    """

    chunks: list[bytes] = []
    total = 0
    while True:
        message = await receive()
        if message["type"] != "http.request":
            break
        chunk = message.get("body", b"")
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise _HttpError(413, f"upload exceeds {MAX_UPLOAD_BYTES} bytes")
        chunks.append(chunk)
        if not message.get("more_body"):
            break
    return b"".join(chunks)


async def _read_json(receive: Any) -> dict[str, Any]:
    chunks: list[bytes] = []
    while True:
        message = await receive()
        if message["type"] != "http.request":
            break
        chunks.append(message.get("body", b""))
        if not message.get("more_body"):
            break
    raw = b"".join(chunks)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _HttpError(400, f"invalid JSON body: {exc}") from exc
    if not isinstance(parsed, dict):
        raise _HttpError(400, "body must be a JSON object")
    return parsed


async def _respond(
    send: Any,
    status: int,
    body: bytes,
    *,
    content_type: bytes = b"text/plain; charset=utf-8",
    origin: str | None = None,
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", content_type),
                (b"content-length", str(len(body)).encode()),
                *_cors(origin),
                *(extra_headers or []),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def _json(
    send: Any, status: int, payload: dict, *, origin: str | None = None
) -> None:
    await _respond(
        send,
        status,
        json.dumps(payload, ensure_ascii=False).encode(),
        content_type=b"application/json; charset=utf-8",
        origin=origin,
    )


async def _lifespan(receive: Any, send: Any) -> None:
    while True:
        message = await receive()
        if message["type"] == "lifespan.startup":
            await send({"type": "lifespan.startup.complete"})
        elif message["type"] == "lifespan.shutdown":
            await send({"type": "lifespan.shutdown.complete"})
            return


_BUILD_COMMAND = "npm --prefix web install &amp;&amp; npm --prefix web run build"

_NO_BUILD = f"""<!doctype html>
<title>re:SOLUTION</title>
<body style="background:#2424c8;color:#fff;padding:48px;
             font:16px/1.6 'Helvetica Neue',Helvetica,Arial,sans-serif">
<h1 style="font-size:40px;margin:0;letter-spacing:-.03em">re:SOLUTION</h1>
<p>The API is running. The frontend has not been built yet.</p>
<pre style="background:rgba(255,255,255,.12);padding:16px">{_BUILD_COMMAND}</pre>
<p>For development, run <code>npm --prefix web run dev</code> and use its port.</p>
</body>
"""

app = ResolutionApp()
