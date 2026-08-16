import pytest

from reagents.contracts import Axis, DomainSpec
from reagents.god.planner import CriticVerdict, Planner, structural_critic
from reagents.llm.scripted import ScriptedLLM
from reagents.tools.registry import default_registry
from reagents.toy import ARTIFACT_SCHEMA, toy_domains, toy_problem


def _spec(name: str, axis: Axis, language: str, tools: list[str]) -> DomainSpec:
    return DomainSpec(
        name=name,
        axes=[axis],
        language=language,
        transform_prompt="project into symbols",
        tool_ids=tools,
        artifact_schema=ARTIFACT_SCHEMA,
    )


def test_structural_critic_accepts_toy_domains():
    verdict = structural_critic(toy_domains(), default_registry())
    assert verdict.ok, verdict.reasons


def test_structural_critic_rejects_shared_primary_axis():
    specs = [
        _spec("alpha_flow", Axis.TOPOLOGY, "cycle space of a signed incidence matrix", ["build_graph", "cut"]),
        _spec("beta_flow", Axis.TOPOLOGY, "generating functions over orbit representatives", ["simplify", "solve"]),
    ]
    verdict = structural_critic(specs, default_registry())
    assert not verdict.ok
    assert "alpha_flow" in verdict.colliding_names
    assert "beta_flow" in verdict.colliding_names


def test_structural_critic_rejects_high_tool_jaccard():
    specs = [
        _spec("alpha_flow", Axis.TOPOLOGY, "cycle space of a signed incidence matrix", ["build_graph", "cut"]),
        _spec("beta_sym", Axis.SYMMETRY, "generating functions over orbit representatives", ["build_graph", "cut"]),
    ]
    verdict = structural_critic(specs, default_registry())
    assert not verdict.ok
    assert any("Jaccard" in reason for reason in verdict.reasons)


def test_structural_critic_rejects_paraphrased_languages():
    specs = [
        _spec("alpha_flow", Axis.TOPOLOGY, "directed catalytic dependency graph of pools", ["build_graph", "cut"]),
        _spec("beta_dyn", Axis.DYNAMICS, "directed catalytic dependency graph of pools", ["simulate", "sample"]),
    ]
    verdict = structural_critic(specs, default_registry())
    assert not verdict.ok
    assert any("language overlap" in reason for reason in verdict.reasons)


@pytest.mark.asyncio
async def test_planner_returns_orthogonal_toy_set():
    planner = Planner(ScriptedLLM.for_toy_pathway(), default_registry())
    specs = await planner.plan(toy_problem(), n=3)
    assert len(specs) == 3
    assert {s.primary_axis for s in specs} == {Axis.CONSERVATION, Axis.TOPOLOGY, Axis.DYNAMICS}


@pytest.mark.asyncio
async def test_planner_regenerates_only_colliding_specs():
    from reagents.god.planner import InventedDomains

    good = toy_domains()
    colliding = [
        good[0],
        _spec(
            "also_conserved",
            Axis.CONSERVATION,
            "another conservation accounting of token balances",
            ["entropy", "compress"],
        ),
        good[1],
    ]
    replacements = [
        good[2],
        _spec(
            "shape_embed",
            Axis.GEOMETRY,
            "ranked embeddings of pool scalars on a line",
            ["embed", "distance"],
        ),
    ]
    llm = ScriptedLLM(
        {
            "invent": [
                InventedDomains(domains=colliding),
                InventedDomains(domains=replacements),
            ],
            "critic": CriticVerdict(ok=True),
        }
    )
    planner = Planner(llm, default_registry())
    specs = await planner.plan(toy_problem(), n=3)
    names = {s.name for s in specs}
    assert "also_conserved" not in names
    assert "stoichiometric_flow" not in names
    assert "catalytic_dag" in names
    assert {s.primary_axis for s in specs} == {Axis.TOPOLOGY, Axis.DYNAMICS, Axis.GEOMETRY}
