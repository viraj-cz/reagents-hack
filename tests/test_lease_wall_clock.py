"""The lease clock has to outlive the thing holding the lease.

`Budget.wall_time_s` defaults to 60s and the clock starts when GOD MINTS the
lease. In-process that is fine -- the pack is called microseconds later. In a
sandbox the same 60s is spent on container creation, image pull and the agent's
first turns, so the authority can expire before its holder has done anything.

Both failures below were observed in one live GOD-sandbox run:

    "Toolbox lease expired (60s budget) before any tool call could be made"
    "dimensional_check rejected array-valued terms ... and the 60s toolbox
     lease expired before a corrected object-shaped call could land"

The second is the worse one: the agent diagnosed its own mistake and was denied
the retry that would have worked.
"""

from __future__ import annotations

from typing import Any

import pytest

from demigod.result import DemiGodResult
from reagents.contracts import Budget, ContextEnvelope, NativeProblem
from reagents.god.orchestrator import God
from reagents.llm.scripted import ScriptedLLM
from reagents.tools.registry import BoundToolPack, default_registry
from reagents.toy import toy_domains, toy_problem


class RecordingRuntime:
    """Captures the lease the orchestrator hands it."""

    def __init__(self, lease_wall_time_s: float | None = None) -> None:
        if lease_wall_time_s is not None:
            self.lease_wall_time_s = lease_wall_time_s
        self.leases: list[Any] = []

    def set_tracer(self, tracer: Any) -> None:
        del tracer

    async def run(
        self,
        envelope: ContextEnvelope,
        tools: BoundToolPack,
        guard: Any | None = None,
    ) -> DemiGodResult:
        del guard
        self.leases.append(tools.lease)
        return DemiGodResult.failure(
            status="failed",
            error="not the point of this test",
            demigod_name=envelope.domain.name,
            domain_name=envelope.domain.name,
        )


async def _spawn_with(runtime: RecordingRuntime) -> list[Any]:
    god = God(ScriptedLLM.for_toy_pathway(), runtime=runtime, domain_count=3)
    await god.solve(toy_problem())
    assert runtime.leases, "the orchestrator never reached the runtime"
    return runtime.leases


async def test_a_runtime_that_states_its_lifetime_gets_a_matching_lease() -> None:
    leases = await _spawn_with(RecordingRuntime(lease_wall_time_s=3600))
    assert all(lease.wall_time_s == 3600 for lease in leases)


async def test_a_runtime_that_says_nothing_keeps_the_budget_default() -> None:
    """Silence must not widen anything. The in-process runtime declares no
    lifetime, and 60s is correct for it."""

    leases = await _spawn_with(RecordingRuntime())
    assert all(lease.wall_time_s == Budget().wall_time_s == 60.0 for lease in leases)


@pytest.mark.parametrize("bogus", [0, -1, None, "3600"])
async def test_a_nonsense_lifetime_is_ignored_rather_than_trusted(bogus: Any) -> None:
    """A runtime is not an authority on its own limits by assertion alone.

    Zero or negative would mint an already-expired lease; a string would crash
    the model copy. Either way the budget's own value stands.
    """

    leases = await _spawn_with(RecordingRuntime(lease_wall_time_s=bogus))
    assert all(lease.wall_time_s == 60.0 for lease in leases)


def test_the_sandbox_runtime_expires_with_its_sandbox() -> None:
    """Not 'big', but 'as long as the holder lives'. The bound that limits a
    demigod is max_calls, which this does not touch."""

    from demigod.spec import DemiGodSpec
    from reagents.demigod.sandbox_runtime import SandboxDemigodRuntime

    runtime = SandboxDemigodRuntime(run_id="run-1")
    assert (
        runtime.lease_wall_time_s == DemiGodSpec.model_fields["max_lifetime_s"].default
    )
    assert (
        SandboxDemigodRuntime(run_id="run-1", lease_wall_time_s=120).lease_wall_time_s
        == 120
    )


def test_the_other_lease_bounds_are_untouched() -> None:
    """Widening the clock must not widen authority in any other direction."""

    registry = default_registry()
    spec = toy_domains()[0]
    budget = Budget(wall_time_s=3600, max_tool_calls=4)
    pack = registry.bind(spec.tool_ids, subject_id=spec.name, budget=budget)

    assert pack.lease.wall_time_s == 3600
    assert pack.lease.max_calls == 4
    assert pack.lease.allow_write is False
    assert set(pack.lease.tool_ids) == set(spec.tool_ids)


def test_problem_fixture_is_the_one_the_scripts_cover() -> None:
    assert isinstance(toy_problem(), NativeProblem)


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
