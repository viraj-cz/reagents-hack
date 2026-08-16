import pytest

from reagents.contracts import NativeSolution
from demigod.result import DemiGodResult
from reagents.demigod.runtime import DemigodRuntime
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
        return DemiGodResult(
            claim="x",
            confidence=0.5,
            method="test stub",
            payload={"findings": [], "conclusion": "x"},
            justification="x",
        )

    runtime = DemigodRuntime(ScriptedLLM({f"demigod:{spec.name}": sneak}))
    pack = default_registry().bind(spec.tool_ids)
    result = await runtime.run(envelope, pack)
    assert result.status != "ok"
    assert "find_cycles" in result.error


@pytest.mark.asyncio
async def test_inprocess_runtime_enforces_schema_tool_call_minimum():
    spec = toy_domains()[0]
    llm = ScriptedLLM.for_toy_pathway()
    domain_problem, _ = await Transformer(llm).forward(toy_problem(), spec)
    envelope = God(llm).build_envelope(spec, domain_problem)
    envelope.artifact_schema = {**spec.artifact_schema, "x-min-tool-calls": 4}
    draft = DemiGodResult(
        claim="candidate",
        confidence=0.5,
        method="test stub",
        payload={
            "candidate_solution": {},
            "constraint_results": {},
            "certificate": {},
            "conclusion": "candidate",
        },
        justification="checked",
    )
    trace = [{"tool": spec.tool_ids[0]} for _ in range(3)]
    runtime = DemigodRuntime(
        ScriptedLLM({f"demigod:{spec.name}": lambda **_: (draft, trace)})
    )
    result = await runtime.run(envelope, default_registry().bind(spec.tool_ids))
    assert result.status == "failed"
    assert "at least 4 brokered tool calls; observed 3" in (result.error or "")


@pytest.mark.asyncio
async def test_one_domain_refusal_does_not_kill_the_whole_run():
    """A transform that fails must cost ONE domain, not the orchestration.

    Regression for a live run that exited non-zero with no answer at all: the
    Anthropic safety classifier refused the transform for the FIRST of two
    domains, `Transformer.forward` raised LLMError, and it propagated straight
    out of `God.solve()`. The second domain was healthy and never ran.

    That contradicts the premise the whole design rests on -- domains are
    independent -- so it is asserted here rather than left to a comment.
    """
    from reagents.llm.client import LLMError

    problem = toy_problem()
    god = God(ScriptedLLM.for_toy_pathway())

    real_forward = god.transformer.forward
    refused: list[str] = []

    async def forward_refusing_first(prob, spec, *args, **kwargs):
        # Refuse exactly one domain, the way a classifier would: on the first
        # one it sees, before any artifact exists.
        if not refused:
            refused.append(spec.name)
            raise LLMError(
                f"TransformDraft: the model refused 2 times (category='bio') "
                f"for {spec.name}"
            )
        return await real_forward(prob, spec, *args, **kwargs)

    god.transformer.forward = forward_refusing_first

    solution = await god.solve(problem)

    # The run produced an answer rather than raising.
    assert isinstance(solution, NativeSolution)
    trace = god.last_trace

    # The refused domain is recorded as a failure, not silently dropped: a
    # missing domain that nothing reports is indistinguishable from one that
    # was never planned.
    assert len(refused) == 1
    failed_names = {f.domain_name for f in trace.failures}
    assert refused[0] in failed_names
    assert any("refused" in (f.error or "") for f in trace.failures)

    # And the surviving domains still did their work.
    assert len(trace.artifacts) == len(trace.specs) - 1
    assert trace.artifacts, "every domain was lost to a single refusal"


@pytest.mark.asyncio
async def test_forbidden_list_leak_is_repaired_not_fatal():
    """A native term in `forbidden` costs the wording, not the domain.

    Regression for a live run that returned one artifact instead of two. The
    planner wrote `forbidden=['... the outlet ...']` -- an anti-leak warning
    that names the thing it forbids -- and the whole domain was discarded
    before its transform was even attempted. The final answer then listed the
    missing domain as a gap.

    `forbidden` is advisory text shown to the demigod, so the leak is real and
    must not be ignored; but it is not the domain definition, and the
    domain-agnostic wording says the same thing without naming anything.
    """
    from reagents.god.orchestrator import ABSTRACT_FORBIDDEN

    problem = toy_problem()
    god = God(ScriptedLLM.for_toy_pathway())

    real_plan = god.planner.plan

    async def plan_leaking_forbidden(prob, n=3):
        specs = await real_plan(prob, n=n)
        # Exactly the shape seen live: a warning naming a native entity.
        specs[0].forbidden = [f"Do not mention {prob.entities[0]} anywhere."]
        return specs

    god.planner.plan = plan_leaking_forbidden

    solution = await god.solve(problem)
    trace = god.last_trace

    # The domain survived rather than being discarded...
    assert len(trace.artifacts) == len(trace.specs), (
        "a leaky forbidden-list still cost a whole domain"
    )
    assert not any("forbidden" in (f.error or "") for f in trace.failures)
    # ...and it survived REPAIRED, not by ignoring the leak: the native term is
    # gone from what the demigod was shown.
    repaired = [e for e in trace.envelopes if e.domain.name == trace.specs[0].name]
    assert repaired, "the repaired domain produced no envelope"
    assert repaired[0].forbidden == list(ABSTRACT_FORBIDDEN)
    assert problem.entities[0] not in " ".join(repaired[0].forbidden)
    assert isinstance(solution, NativeSolution)
