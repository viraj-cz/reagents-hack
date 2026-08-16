import pytest

from reagents.contracts import DomainArtifact, NativeSolution
from reagents.demigod.runtime import DemigodDraft, DemigodRuntime
from reagents.god.orchestrator import God
from reagents.god.transformer import Transformer
from reagents.llm.scripted import ScriptedLLM
from reagents.tools.registry import default_registry
from reagents.toy import toy_domains, toy_problem


@pytest.mark.asyncio
async def test_god_solves_toy_pathway_end_to_end():
    problem = toy_problem()
    god = God(ScriptedLLM.for_toy_pathway())
    solution = await god.solve(problem)

    assert isinstance(solution, NativeSolution)
    assert solution.problem_id == "pfk-bottleneck"
    assert solution.confidence > 0
    assert set(solution.domain_contributions) == {
        "stoichiometric_flow",
        "catalytic_dag",
        "rate_orbit",
    }

    trace = god.last_trace
    assert {s.name for s in trace.specs} == {
        "stoichiometric_flow",
        "catalytic_dag",
        "rate_orbit",
    }
    assert {s.primary_axis.value for s in trace.specs} == {
        "conservation",
        "topology",
        "dynamics",
    }
    assert len(trace.envelopes) == 3
    assert len(trace.artifacts) == 3
    assert trace.failures == []
    assert trace.leaks == []
    assert {m.domain_name for m in trace.inverse_maps} == {
        "stoichiometric_flow",
        "catalytic_dag",
        "rate_orbit",
    }
    # Inverse maps stay on God's side; envelopes must not contain them.
    for envelope in trace.envelopes:
        dumped = envelope.model_dump()
        assert "symbol_to_native" not in dumped
        assert "hexokinase" not in str(dumped)


@pytest.mark.asyncio
async def test_demigod_uses_only_envelope_tools_and_validates_schema():
    problem = toy_problem()
    god = God(ScriptedLLM.for_toy_pathway())
    await god.solve(problem)
    for artifact in god.last_trace.artifacts:
        assert "findings" in artifact.payload
        assert "conclusion" in artifact.payload
        assert artifact.tool_trace
        used = {step["tool"] for step in artifact.tool_trace}
        spec = next(s for s in god.last_trace.specs if s.name == artifact.domain_name)
        assert used <= set(spec.tool_ids)


@pytest.mark.asyncio
async def test_demigod_tool_loop_cannot_reach_unbound_tools():
    spec = toy_domains()[0]
    llm = ScriptedLLM.for_toy_pathway()
    domain_problem, _ = await Transformer(llm).forward(toy_problem(), spec)
    envelope = God(llm).build_envelope(spec, domain_problem)

    def sneak(*, tools, **_):
        tools.call("find_cycles", graph={"nodes": [], "edges": []})
        return DemigodDraft(payload={"findings": [], "conclusion": "x"}, justification="x")

    runtime = DemigodRuntime(ScriptedLLM({f"demigod:{spec.name}": sneak}))
    pack = default_registry().bind(spec.tool_ids)
    result = await runtime.run(envelope, pack)
    assert not isinstance(result, DomainArtifact)
    assert "find_cycles" in result.reason
