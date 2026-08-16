import pytest

from reagents.contracts import ContextEnvelope, DomainProblem
from reagents.demigod.runtime import DemigodRuntime, IsolationGuard
from reagents.god.orchestrator import God
from reagents.god.transformer import LeakError, Transformer
from reagents.isolation import assert_sealed, find_leaks, native_terms
from reagents.llm.scripted import ScriptedLLM
from reagents.tools.registry import default_registry
from reagents.toy import toy_domains, toy_problem


def test_native_terms_include_entities_but_not_short_tokens():
    terms = native_terms(toy_problem())
    assert "hexokinase" in terms
    assert "ATP" in terms
    assert "glucose" in terms
    assert "A" not in terms


def test_find_leaks_matches_whole_tokens_only():
    assert find_leaks("the hexokinase rate is large", {"hexokinase"}) == ["hexokinase"]
    assert find_leaks("hexokinasease is not a leak", {"hexokinase"}) == []
    assert find_leaks("no biological names here", {"hexokinase", "ATP"}) == []


@pytest.mark.asyncio
async def test_transform_is_sealed_against_native_names():
    problem = toy_problem()
    spec = toy_domains()[0]
    transformer = Transformer(ScriptedLLM.for_toy_pathway())
    domain_problem, inverse = await transformer.forward(problem, spec)
    leaks = find_leaks(
        f"{domain_problem.task}\n{domain_problem.notation_guide}\n{domain_problem.representation}",
        native_terms(problem),
    )
    assert leaks == []
    assert inverse.symbol_to_native["r1"] == "hexokinase"
    assert "hexokinase" not in domain_problem.task


@pytest.mark.asyncio
async def test_transform_retries_then_fails_on_persistent_leaks():
    from reagents.god.transformer import TransformDraft

    leaky = TransformDraft(
        representation={"enzyme": "hexokinase"},
        task="explain hexokinase",
        notation_guide="hexokinase is e1",
        symbol_to_native={"e1": "hexokinase"},
    )
    llm = ScriptedLLM({"transform:stoichiometric_flow": leaky})
    transformer = Transformer(llm)
    with pytest.raises(LeakError) as exc:
        await transformer.forward(toy_problem(), toy_domains()[0], max_retries=1)
    assert "hexokinase" in exc.value.leaks


@pytest.mark.asyncio
async def test_god_envelopes_contain_no_native_entity_names():
    problem = toy_problem()
    god = God(ScriptedLLM.for_toy_pathway())
    await god.solve(problem)
    terms = native_terms(problem)
    assert god.last_trace.envelopes
    for envelope in god.last_trace.envelopes:
        assert assert_sealed(envelope, terms) == []
        visible_tools = {t.id for t in envelope.tools}
        assert visible_tools == set(envelope.domain.tool_ids)
        assert visible_tools <= set(default_registry().ids())


@pytest.mark.asyncio
async def test_demigod_rejects_unsealed_envelope():
    problem = toy_problem()
    spec = toy_domains()[1]
    envelope = ContextEnvelope(
        domain=spec,
        problem=DomainProblem(
            domain_name=spec.name,
            representation={"nodes": ["hexokinase"]},
            task="cut hexokinase",
            notation_guide="hexokinase is a vertex",
        ),
        tools=default_registry().specs(spec.tool_ids),
        artifact_schema=spec.artifact_schema,
    )
    runtime = DemigodRuntime(ScriptedLLM.for_toy_pathway())
    pack = default_registry().bind(spec.tool_ids)
    result = await runtime.run(envelope, pack, guard=IsolationGuard(native_terms(problem)))
    assert result.isolation_violations
    assert "hexokinase" in result.isolation_violations
