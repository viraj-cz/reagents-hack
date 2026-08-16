"""The web interface, exercised without a browser or a server.

`ResolutionApp` is a plain ASGI callable for the same reason `broker/router.py`
is: it can be driven from a test with a scope dict, so the whole surface --
routing, the run lifecycle, and the SSE framing -- is checkable offline. The one
thing these tests deliberately do not stub is `God`: a scripted run is cheap and
the interesting failures are in the seam between the trace stream and the
events, which a stubbed orchestrator would hide.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import types
from unittest import mock

import pytest

from resolution.app import ResolutionApp
from resolution.events import build_event, jsonable, redact
from resolution.narration import lane_for_phase
from resolution.runs import RunStore, build_problem
from resolution.toy import preset_problem


async def call(app: ResolutionApp, method: str, path: str, body: dict | None = None):
    """Drive the ASGI app once and collect the whole response."""

    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [(b"origin", b"http://localhost:5273")],
    }
    payload = json.dumps(body or {}).encode()
    sent = [{"type": "http.request", "body": payload, "more_body": False}]
    received: list[dict] = []

    async def receive() -> dict:
        return sent.pop(0) if sent else {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        received.append(message)

    await app(scope, receive, send)
    start = received[0]
    chunks = b"".join(m.get("body", b"") for m in received[1:])
    return start["status"], chunks


async def read_stream(app: ResolutionApp, path: str) -> list[dict]:
    """Consume an SSE response to completion and parse every data frame."""

    scope = {"type": "http", "method": "GET", "path": path, "headers": []}
    frames: list[bytes] = []
    finished = asyncio.Event()

    async def receive() -> dict:
        await finished.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        if message["type"] == "http.response.body":
            frames.append(message.get("body", b""))
            if not message.get("more_body"):
                finished.set()

    await app(scope, receive, send)
    events = []
    for block in b"".join(frames).decode().split("\n\n"):
        # The terminating `event: end` frame is a stream-level marker, not a
        # run event; the client uses it to stop listening.
        if block.startswith("event: end"):
            continue
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))
    return events


async def test_health_and_presets() -> None:
    app = ResolutionApp()
    status, body = await call(app, "GET", "/api/health")
    assert status == 200
    assert json.loads(body)["ok"] is True

    status, body = await call(app, "GET", "/api/presets")
    payload = json.loads(body)
    assert status == 200
    assert {p["id"] for p in payload["presets"]} == {
        "shop-change",
        "release-schedule",
        "pfk-bottleneck",
    }
    # One per difficulty tier, in increasing order. The order is the point: the
    # list is meant to be read as a ramp, because what it demonstrates is a
    # fan-out that grows with the problem.
    assert [p["tier"] for p in payload["presets"]] == ["easy", "medium", "hard"]
    # The recorded scripts only cover one problem; the UI needs to know that
    # before it offers a replay that would fail with `no script for phase`.
    scripted = [p for p in payload["presets"] if "scripted" in p["modes"]]
    assert [p["id"] for p in scripted] == ["pfk-bottleneck"]


async def test_a_scripted_run_streams_and_finishes() -> None:
    app = ResolutionApp()
    status, body = await call(
        app, "POST", "/api/runs", {"preset": "pfk-bottleneck", "mode": "scripted"}
    )
    assert status == 201
    run_id = json.loads(body)["run_id"]

    events = await read_stream(app, f"/api/runs/{run_id}/stream")
    kinds = {(e["node"], e["kind"]) for e in events}

    # GOD's own phases, in one lane.
    assert ("god", "plan") in kinds
    assert ("god", "integrate") in kinds
    assert ("god", "run_end") in kinds
    # One lane per demigod, each with its own tool calls. This is the property
    # the tree view is built on: a node id derived from the lane, not parsed
    # out of a message.
    demigods = {node for node, _ in kinds if node.startswith("demi:")}
    assert demigods == {
        "demi:stoichiometric_flow",
        "demi:catalytic_dag",
        "demi:rate_orbit",
    }
    assert all((node, "tool_call") in kinds for node in demigods)

    run = app.store.get(run_id)
    assert run is not None
    await run.task
    assert run.snapshot.status == "done"
    assert run.snapshot.solution is not None
    assert run.snapshot.solution["confidence"] > 0

    # Sequence numbers are dense and ordered: a gap would mean a subscriber
    # silently lost a frame, which is exactly what the UI cannot detect.
    sequences = [e["seq"] for e in events]
    assert sequences == sorted(sequences)
    assert sequences == list(range(1, len(sequences) + 1))


async def test_a_late_subscriber_gets_the_whole_run() -> None:
    """Reload mid-run must not produce a half-rendered tree."""

    app = ResolutionApp()
    _, body = await call(app, "POST", "/api/runs", {"preset": "pfk-bottleneck"})
    run_id = json.loads(body)["run_id"]
    run = app.store.get(run_id)
    assert run is not None
    await run.task

    events = await read_stream(app, f"/api/runs/{run_id}/stream")
    assert events[-1]["kind"] == "run_end"
    assert len(events) == len(run.events)


async def test_unknown_routes_and_bad_requests_are_reported() -> None:
    app = ResolutionApp()
    status, _ = await call(app, "GET", "/api/runs/run-nope")
    assert status == 404
    status, _ = await call(
        app, "POST", "/api/runs", {"prompt": "x", "mode": "nonsense"}
    )
    assert status == 400
    status, _ = await call(
        app, "POST", "/api/runs", {"preset": "no-such-preset", "mode": "live"}
    )
    assert status == 400
    status, _ = await call(app, "POST", "/api/runs", {"preset": "pfk", "domains": 99})
    assert status == 400


async def test_live_mode_is_refused_before_a_run_starts(monkeypatch) -> None:
    """A missing key must fail the request, not the run.

    Starting a run that dies on its first model call reads as a broken
    orchestrator; a 400 reads as a server that was never configured for live
    inference, which is what it is.
    """

    app = ResolutionApp()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    status, body = await call(
        app, "POST", "/api/runs", {"prompt": "why", "mode": "live"}
    )
    assert status == 400
    assert "ANTHROPIC_API_KEY" in json.loads(body)["error"]
    assert not app.store.runs

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr("resolution.app.importlib.util.find_spec", lambda name: None)
    status, body = await call(
        app, "POST", "/api/runs", {"prompt": "why", "mode": "live"}
    )
    assert status == 400
    assert "llm" in json.loads(body)["error"]

    _, body = await call(app, "GET", "/api/presets")
    assert json.loads(body)["live_available"] is False


async def test_replay_refuses_a_prompt_it_cannot_actually_run() -> None:
    """The recording is keyed by phase, not by problem.

    `ScriptedLLM` happily returns the recorded glycolysis domains for *any*
    prompt, so a free-text replay produces a confident analysis of a problem
    nobody asked about. Refusing is the only honest answer.
    """

    app = ResolutionApp()
    status, body = await call(
        app, "POST", "/api/runs", {"prompt": "why is the bridge humming?"}
    )
    assert status == 400
    assert "recorded preset" in json.loads(body)["error"]
    assert not app.store.runs

    status, body = await call(
        app, "POST", "/api/runs", {"preset": "valve-bottleneck", "mode": "scripted"}
    )
    assert status == 400
    assert "no recording" in json.loads(body)["error"]


async def test_a_failed_run_becomes_a_ui_state_not_a_crash(monkeypatch) -> None:
    """An exception inside `God.solve` is a run state, not a 500 and not a hang.

    The stream must still terminate -- a subscriber waiting forever on a run
    that already died is the failure mode that matters here.
    """

    class ExplodingLLM:
        async def complete(self, **_: object) -> object:
            raise RuntimeError("model unavailable")

        async def run_tool_loop(self, **_: object) -> object:
            raise RuntimeError("model unavailable")

    monkeypatch.setattr("resolution.runs.make_llm", lambda mode, tracer: ExplodingLLM())

    app = ResolutionApp()
    _, body = await call(app, "POST", "/api/runs", {"preset": "pfk-bottleneck"})
    run_id = json.loads(body)["run_id"]
    run = app.store.get(run_id)
    assert run is not None
    await run.task

    assert run.snapshot.status == "error"
    assert "model unavailable" in (run.snapshot.error or "")
    events = await read_stream(app, f"/api/runs/{run_id}/stream")
    assert events[-1]["kind"] == "run_end"
    assert events[-1]["message"] == "error"


async def test_transcript_is_scoped_to_one_node() -> None:
    app = ResolutionApp()
    _, body = await call(app, "POST", "/api/runs", {"preset": "pfk-bottleneck"})
    run_id = json.loads(body)["run_id"]
    run = app.store.get(run_id)
    assert run is not None
    await run.task

    status, body = await call(
        app, "GET", f"/api/runs/{run_id}/transcript/demi:catalytic_dag"
    )
    payload = json.loads(body)
    assert status == 200
    assert payload["events"]
    assert {e["node"] for e in payload["events"]} == {"demi:catalytic_dag"}


def test_env_file_loading_prefers_the_real_environment(tmp_path, monkeypatch) -> None:
    """`.env` fills gaps; it never overrides what the caller exported.

    The names are returned rather than the values because the caller prints
    them, and a secret in a log is a secret you have to rotate.
    """

    from resolution.__main__ import load_env_file

    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "# a comment",
                "",
                "ANTHROPIC_API_KEY=sk-ant-from-file",
                'export QUOTED="quoted value"',
                "ALREADY_SET=from-file",
                "EMPTY=",
                "not-a-pair",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ALREADY_SET", "from-environment")

    loaded = load_env_file(env)

    assert sorted(loaded) == ["ANTHROPIC_API_KEY", "QUOTED"]
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-file"
    assert os.environ["QUOTED"] == "quoted value"
    assert os.environ["ALREADY_SET"] == "from-environment"
    assert "EMPTY" not in os.environ
    assert load_env_file(tmp_path / "absent") == []


def test_build_problem_never_invents_sealed_terms() -> None:
    """`entities` is the seal. A guessed one seals the wrong words."""

    problem = build_problem("A pump feeds a valve. Why is throughput flat?")
    assert problem.entities == []
    assert problem.question == "Why is throughput flat?"

    named = build_problem("text", entities=["pump A", " ", "valve B"])
    assert named.entities == ["pump A", "valve B"]

    with pytest.raises(ValueError):
        build_problem("   ")


def test_events_are_addressed_to_nodes_and_redacted() -> None:
    event = build_event(
        seq=1,
        elapsed_s=0.5,
        lane="DEMI:catalytic_dag",
        kind="TOOL CALL",
        message="build_graph",
        data={"arguments": {"api_key": "sk-secret", "nodes": ["v1"]}},
    ).to_json()

    assert event["node"] == "demi:catalytic_dag"
    assert event["kind"] == "tool_call"
    assert event["group"] == "tool"
    assert event["data"]["arguments"]["api_key"] == "<redacted>"
    assert event["data"]["arguments"]["nodes"] == ["v1"]
    # Redaction is by key at any depth, not only at the top level.
    assert redact({"a": [{"token": "t"}]}) == {"a": [{"token": "<redacted>"}]}


def test_jsonable_survives_what_a_tool_can_return() -> None:
    from reagents.contracts import Axis

    payload = jsonable({"axis": Axis.TOPOLOGY, "nan": float("nan"), "set": {1}})
    assert payload["axis"] == "topology"
    # JSON has no NaN literal; a strict browser parser rejects the whole frame.
    assert payload["nan"] is None
    assert payload["set"] == [1]
    json.dumps(payload)


def test_text_events_are_routed_to_the_lane_that_produced_them() -> None:
    assert lane_for_phase("demigod:catalytic_dag") == "DEMI:catalytic_dag"
    assert lane_for_phase("transform:catalytic_dag") == "GOD"
    assert lane_for_phase("integrate") == "GOD"


async def test_the_store_cancels_evicted_runs() -> None:
    store = RunStore(max_runs=1)
    first = store.create(
        preset_problem("pfk-bottleneck"), mode="scripted", domain_count=3
    )
    store.create(preset_problem("pfk-bottleneck"), mode="scripted", domain_count=3)
    assert store.get(first.run_id) is None
    assert first.task is not None
    await asyncio.gather(first.task, return_exceptions=True)
    assert first.task.cancelled() or first.snapshot.status in {"cancelled", "done"}


def test_relayed_events_keep_the_sandbox_clock() -> None:
    """A relayed event carries the time it happened, not the time it arrived.

    Re-stamping on arrival dated every event in a drained batch to the same
    instant: three demigods working in parallel for 212s collapsed into nine
    events sharing one timestamp, ordered by queue position.
    """
    from resolution.godbox_run import _relay

    seen: list[tuple[str, float | None]] = []

    def emit(lane, kind, message, *, data=None, at=None):
        seen.append((kind, at))

    _relay(emit, {"seq": 1, "t": 41.5, "lane": "GOD", "kind": "PLAN", "message": ""})
    _relay(emit, {"seq": 2, "lane": "GOD", "kind": "NOCLOCK", "message": ""})

    assert seen == [("PLAN", 41.5), ("NOCLOCK", None)]


def test_relayed_events_are_ordered_by_the_writer() -> None:
    """`seq` comes from inside the sandbox; a drain returns arrival order."""
    from resolution.godbox_run import _ordered

    batch = [{"seq": 3}, {"seq": 1}, {"seq": 2}]
    assert [e["seq"] for e in _ordered(batch)] == [1, 2, 3]


def test_godbox_requests_the_broker() -> None:
    """Unset `use_broker` tells every demigod its toolset is unreachable."""
    from reagents.toy import toy_problem
    from resolution.godbox_run import build_request

    request = build_request("run-x", toy_problem(), domain_count=2, max_turns=4)
    assert request.use_broker is True


async def test_sandbox_execution_requires_the_lease() -> None:
    """A lease that cannot be published must fail the run, not be worked around.

    The default prints one line and continues tool-less, and the artifact that
    comes back is confident and schema-valid. That is the shape of every
    tool-less run found so far.
    """
    import resolution.runs as runs_module
    from reagents.toy import toy_problem

    captured: dict[str, object] = {}

    class _Runtime:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    run = runs_module.Run(
        "run-x",
        toy_problem(),
        mode="live",
        domain_count=2,
        execution=runs_module.EXECUTION_SANDBOX,
    )
    module = types.ModuleType("reagents.demigod.sandbox_runtime")
    module.SandboxDemigodRuntime = _Runtime  # type: ignore[attr-defined]
    session = types.ModuleType("broker.session")
    session.modal_session = lambda: object()  # type: ignore[attr-defined]
    with mock.patch.dict(
        sys.modules,
        {"reagents.demigod.sandbox_runtime": module, "broker.session": session},
    ):
        await run._build_runtime()

    assert captured["require_toolbox"] is True


async def test_null_domains_means_god_chooses() -> None:
    """The UI's dynamic option sends `null`, and it must reach GOD as None.

    `int(body.get("domains") or 3)` used to turn every falsy value into a fixed
    3, so the dynamic option would have silently run as a pinned three-domain
    request -- indistinguishable, in the UI, from the button next to it.
    """

    app = ResolutionApp()
    status, body = await call(
        app, "POST", "/api/runs", {"preset": "pfk-bottleneck", "domains": None}
    )
    assert status == 201, body
    run = app.store.runs[json.loads(body)["run_id"]]
    assert run.domain_count is None

    # Omitted entirely is the same decision as an explicit null.
    status, body = await call(app, "POST", "/api/runs", {"preset": "pfk-bottleneck"})
    assert status == 201, body
    assert app.store.runs[json.loads(body)["run_id"]].domain_count is None


async def test_an_explicit_domain_count_is_still_pinned() -> None:
    """Dynamic is an option, not a takeover: 2/3/4 must still pin."""

    app = ResolutionApp()
    status, body = await call(
        app, "POST", "/api/runs", {"preset": "pfk-bottleneck", "domains": 2}
    )
    assert status == 201, body
    assert app.store.runs[json.loads(body)["run_id"]].domain_count == 2

    # 0 is not "dynamic by another name" -- it is out of range and refused.
    status, _ = await call(
        app, "POST", "/api/runs", {"preset": "pfk-bottleneck", "domains": 0}
    )
    assert status == 400
