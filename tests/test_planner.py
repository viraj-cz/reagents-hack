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
        _spec(
            "alpha_flow",
            Axis.TOPOLOGY,
            "cycle space of a signed incidence matrix",
            ["build_graph", "cut"],
        ),
        _spec(
            "beta_flow",
            Axis.TOPOLOGY,
            "generating functions over orbit representatives",
            ["simplify", "solve"],
        ),
    ]
    verdict = structural_critic(specs, default_registry())
    assert not verdict.ok
    assert "alpha_flow" in verdict.colliding_names
    assert "beta_flow" in verdict.colliding_names


def test_structural_critic_rejects_high_tool_jaccard():
    specs = [
        _spec(
            "alpha_flow",
            Axis.TOPOLOGY,
            "cycle space of a signed incidence matrix",
            ["build_graph", "cut"],
        ),
        _spec(
            "beta_sym",
            Axis.SYMMETRY,
            "generating functions over orbit representatives",
            ["build_graph", "cut"],
        ),
    ]
    verdict = structural_critic(specs, default_registry())
    assert not verdict.ok
    assert any("Jaccard" in reason for reason in verdict.reasons)


def test_structural_critic_rejects_paraphrased_languages():
    specs = [
        _spec(
            "alpha_flow",
            Axis.TOPOLOGY,
            "directed catalytic dependency graph of pools",
            ["build_graph", "cut"],
        ),
        _spec(
            "beta_dyn",
            Axis.DYNAMICS,
            "directed catalytic dependency graph of pools",
            ["simulate", "sample"],
        ),
    ]
    verdict = structural_critic(specs, default_registry())
    assert not verdict.ok
    assert any("language overlap" in reason for reason in verdict.reasons)


@pytest.mark.asyncio
async def test_planner_returns_orthogonal_toy_set():
    planner = Planner(ScriptedLLM.for_toy_pathway(), default_registry())
    specs = await planner.plan(toy_problem(), n=3)
    assert len(specs) == 3
    assert {s.primary_axis for s in specs} == {
        Axis.CONSERVATION,
        Axis.TOPOLOGY,
        Axis.DYNAMICS,
    }


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
    assert {s.primary_axis for s in specs} == {
        Axis.TOPOLOGY,
        Axis.DYNAMICS,
        Axis.GEOMETRY,
    }


def test_structural_critic_flags_a_leaking_spec():
    """A native term in a scanned field must make the critic reject the spec.

    The critic is where rejection has a CONSEQUENCE: `Planner.plan` regenerates
    whatever the critic names. Before this, the leak check lived only in
    `God.solve`, which runs after planning has finished -- so a leaking domain
    was discarded rather than replaced, and a live run returned one artifact
    instead of two because of a word in a `forbidden` entry.
    """
    from reagents.isolation import native_terms

    problem = toy_problem()
    terms = native_terms(problem)
    leaked_entity = problem.entities[0]

    specs = [
        _spec(
            "alpha_flow",
            Axis.TOPOLOGY,
            "cycle space of a signed incidence matrix",
            ["build_graph", "cut"],
        )
    ]
    specs[0].forbidden = [f"Do not mention {leaked_entity}."]

    # Without `terms` this is the critic it always was: nothing to compare to.
    assert structural_critic(specs, default_registry()).ok

    verdict = structural_critic(specs, default_registry(), terms=terms)
    assert not verdict.ok
    assert "alpha_flow" in verdict.colliding_names
    assert any("leaked native terms" in r for r in verdict.reasons)
    assert any(leaked_entity in r for r in verdict.reasons), (
        "the reason must name the term, since it is fed back to the model"
    )


@pytest.mark.asyncio
async def test_plan_regenerates_a_leaking_spec_and_says_why():
    """The leak must reach the model as feedback, not just cause a retry.

    Regenerating without saying what was wrong is asking the model to guess,
    and it will reproduce the same leak. `Transformer.forward` already feeds its
    leaks back for exactly this reason; the planner now does the same.
    """
    from reagents.isolation import find_spec_leaks, native_terms

    problem = toy_problem()
    terms = native_terms(problem)
    planner = Planner(ScriptedLLM.for_toy_pathway(), default_registry())

    real_invent = planner.invent
    prompts_seen: list[list[str]] = []
    calls = {"n": 0}

    async def invent_leaking_first(prob, n, **kwargs):
        prompts_seen.append(list(kwargs.get("rejected_because") or []))
        specs = await real_invent(prob, n, **kwargs)
        calls["n"] += 1
        if calls["n"] == 1:
            # Leak on the first attempt only, the way the live planner did.
            # A COPY, not a mutation: ScriptedLLM hands back the same DomainSpec
            # objects every call, so mutating one would make the "regenerated"
            # spec the same leaking instance and the loop could never converge.
            specs[0] = specs[0].model_copy(
                update={"forbidden": [f"Do not mention {prob.entities[0]}."]}
            )
        return specs

    planner.invent = invent_leaking_first

    specs = await planner.plan(problem, n=3)

    assert calls["n"] > 1, "the leaking plan was accepted without regenerating"
    # The second call was told what was wrong, naming the offending term.
    feedback = [p for p in prompts_seen if p]
    assert feedback, "regenerated without telling the model why"
    assert any("leaked native terms" in r for round_ in feedback for r in round_)

    # And what comes back is clean.
    for spec in specs:
        assert not find_spec_leaks(spec, terms), f"{spec.name} still leaks"
