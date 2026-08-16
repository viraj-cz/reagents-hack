"""Minimal walking test for the complete God -> demigods -> God lifecycle."""

import pytest

from reagents.god.orchestrator import God
from reagents.isolation import assert_sealed, native_terms
from reagents.llm.scripted import ScriptedLLM
from reagents.toy import toy_problem


@pytest.mark.asyncio
async def test_minimal_god_framework_end_to_end():
    """Three scoped representations must become one native-domain answer.

    The scripted LLM makes this deterministic and API-free, while the real planner,
    transformer, concurrent demigod runtime, tool registry/broker, artifact validator,
    isolation guard, and integrator all remain in the execution path.
    """
    problem = toy_problem()
    god = God(llm=ScriptedLLM.for_toy_pathway(), domain_count=3)

    solution = await god.solve(problem)
    trace = god.last_trace

    # God chose a genuinely orthogonal set of representation spaces.
    assert [spec.primary_axis.value for spec in trace.specs] == [
        "conservation",
        "topology",
        "dynamics",
    ]
    tool_sets = [set(spec.tool_ids) for spec in trace.specs]
    assert all(
        left.isdisjoint(right)
        for i, left in enumerate(tool_sets)
        for right in tool_sets[i + 1 :]
    )

    # Every demigod received only its sealed representation and exact tool pack.
    sealed_terms = native_terms(problem)
    assert len(trace.envelopes) == 3
    for envelope in trace.envelopes:
        spec = next(spec for spec in trace.specs if spec.name == envelope.domain.name)
        assert envelope.domain.transform_prompt == ""
        assert {tool.id for tool in envelope.tools} == set(spec.tool_ids)
        assert assert_sealed(envelope, sealed_terms) == []
        assert "symbol_to_native" not in envelope.model_dump()

    # The simple demigods called their scoped tools and returned valid artifacts.
    assert len(trace.artifacts) == 3
    assert trace.failures == []
    assert trace.leaks == []
    for artifact in trace.artifacts:
        spec = next(spec for spec in trace.specs if spec.name == artifact.domain_name)
        assert {call["tool"] for call in artifact.tool_trace} == set(spec.tool_ids)
        assert {"findings", "conclusion"} <= artifact.payload.keys()

    # God alone retained the inverse maps and translated the combined evidence back.
    assert len(trace.inverse_maps) == 3
    assert solution.problem_id == problem.id
    assert set(solution.domain_contributions) == {spec.name for spec in trace.specs}
    assert "phosphofructokinase" in solution.answer.lower()
    assert "ATP" in solution.answer
    assert solution.confidence > 0.8
    assert trace.solution == solution
