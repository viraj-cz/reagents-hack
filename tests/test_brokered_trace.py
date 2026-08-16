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
