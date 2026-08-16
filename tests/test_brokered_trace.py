"""A sandboxed demigod's tool calls happen out of this process's sight.

The broker authors the trace and it arrives with the result, so a consumer
counting live TOOL CALL events reported zero while six calls had been audited.

(The lease wall-clock tests that used to live here are gone: master fixed the
same problem in `godbox.entrypoint` with an explicit, bounded budget --
`min(max_turns * 75, 1800)` -- and a second mechanism advertising 3600 from the
runtime would have silently overridden that cap.)
"""

from __future__ import annotations

from demigod.result import DemiGodResult


def test_brokered_calls_are_replayed_as_tool_events() -> None:
    """A sandboxed demigod's calls happen out of this process's sight.

    The broker authors the trace and it arrives with the result, so a consumer
    counting live TOOL CALL events reported zero while six calls had been
    audited. Replayed after the fact: late, but ordered and complete -- and a
    refused call is a denial, not a silence.
    """

    from reagents.demigod.sandbox_runtime import SandboxDemigodRuntime
    from reagents.tracing import RecordingTracer

    tracer = RecordingTracer()
    runtime = SandboxDemigodRuntime(run_id="run-1", tracer=tracer)
    result = DemiGodResult.failure(
        status="failed", error="x", demigod_name="d", domain_name="d"
    )
    result.tool_trace = [
        {"tool": "build_graph", "input": {"nodes": []}, "result": {"edges": []}},
        {
            "tool": "publish",
            "ok": False,
            "metered": False,
            "error": {"message": "unbound_tool"},
        },
        {"tool": "simulate", "ok": False, "error": {"message": "boom"}},
    ]

    runtime._replay_tool_trace("DEMI:d", result)

    seen = [(r.kind, r.message) for r in tracer.records]
    assert seen[0] == ("TOOL CALL", "build_graph")
    assert seen[1] == ("TOOL RESULT", "build_graph")
    assert seen[2] == ("TOOL CALL", "publish")
    # Refused, not failed: the broker marks an unmetered call, and a demigod
    # probing for a tool nobody granted it should be visible as exactly that.
    assert seen[3][0] == "TOOL DENY"
    assert seen[5][0] == "TOOL ERROR"
    assert tracer.records[0].data == {"arguments": {"nodes": []}}


def test_live_calls_are_not_replayed_twice_at_the_end():
    """The poll and the end-of-run replay read the SAME broker trace.

    Without the emitted-count the final replay repeats every call a watcher
    already saw live, so a demigod that made two calls appears to have made
    four.
    """
    from demigod.result import DemiGodResult
    from reagents.demigod.sandbox_runtime import SandboxDemigodRuntime
    from reagents.tracing import RecordingTracer

    tracer = RecordingTracer()
    runtime = SandboxDemigodRuntime(run_id="r1", toolbox=None, tracer=tracer)
    entries = [
        {"tool": "solve", "input": {"x": 1}, "result": {"y": 2}, "ok": True},
        {"tool": "simplify", "input": {}, "result": {}, "ok": True},
    ]

    # Seen live, while the demigod was still running.
    runtime._emit_tool_entries("DEMI:d", entries)
    calls = [r for r in tracer.records if r.kind == "TOOL CALL"]
    assert [r.message for r in calls] == ["solve", "simplify"]

    # The audited trace comes back holding the same two, plus one more.
    result = DemiGodResult(
        claim="c",
        confidence=0.5,
        method="m",
        payload={},
        demigod_name="d",
        domain_name="d",
        status="ok",
        tool_trace=[*entries, {"tool": "cut", "input": {}, "result": {}, "ok": True}],
    )
    runtime._replay_tool_trace("DEMI:d", result)

    calls = [r for r in tracer.records if r.kind == "TOOL CALL"]
    assert [r.message for r in calls] == ["solve", "simplify", "cut"]


def test_a_broker_that_cannot_be_polled_does_not_fail_the_run():
    """Observability must not be able to break what it observes."""
    import asyncio

    from demigod.toolbox.protocol import ToolboxGrant
    from reagents.demigod.sandbox_runtime import SandboxDemigodRuntime
    from reagents.tracing import RecordingTracer

    class AngrySession:
        url = "https://broker.invalid"

        def collect_trace(self, lease_id, *, include_refused=True):
            raise RuntimeError("broker unreachable")

    tracer = RecordingTracer()
    runtime = SandboxDemigodRuntime(run_id="r1", toolbox=AngrySession(), tracer=tracer)
    grant = ToolboxGrant(url="https://broker.invalid", lease_id="lease-1")

    async def poll_briefly():
        task = asyncio.create_task(
            runtime._follow_lease("DEMI:d", grant, interval_s=0.01)
        )
        await asyncio.sleep(0.05)
        task.cancel()

    asyncio.run(poll_briefly())
    assert not [r for r in tracer.records if r.kind == "TOOL CALL"]
