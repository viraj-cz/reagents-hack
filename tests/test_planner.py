import pytest

from reagents.contracts import Axis, DomainSpec
from reagents.god.planner import (
    CriticVerdict,
    InventedDomains,
    Planner,
    structural_critic,
)
from reagents.llm.client import LLMError
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


def test_reasoning_contract_requires_compute_and_auditable_experiments(
    monkeypatch,
):
    monkeypatch.setenv("REAGENTS_ENABLE_NORMAN_V2_BENCHMARK", "1")
    registry = default_registry()
    contract = {
        "artifact_required_keys": [
            "candidate_solution",
            "constraint_results",
            "certificate",
            "conclusion",
            "hypotheses",
            "experiments",
            "model_comparison",
            "validation",
        ],
        "minimum_broker_calls": 4,
        "minimum_model_configurations": 2,
        "required_tool_prefix": "screen2.",
        "required_compute_suffix": "_lab",
    }
    weak = _spec(
        "weak_geometry",
        Axis.GEOMETRY,
        "metric geometry of opaque vectors",
        ["screen2.training_manifest", "distance"],
    )
    verdict = structural_critic([weak], registry, reasoning_contract=contract)
    assert not verdict.ok
    assert any("x-min-tool-calls" in reason for reason in verdict.reasons)
    assert any("screen2.*_lab" in reason for reason in verdict.reasons)

    properties = dict(weak.artifact_schema["properties"])
    properties.update(
        {
            "hypotheses": {"type": "array"},
            "experiments": {"type": "array", "minItems": 1},
            "model_comparison": {"type": "array", "minItems": 2},
            "validation": {"type": "object"},
        }
    )
    strong = weak.model_copy(
        update={
            "tool_ids": ["screen2.geometry_lab", "distance"],
            "artifact_schema": {
                **weak.artifact_schema,
                "required": contract["artifact_required_keys"],
                "properties": properties,
                "x-min-tool-calls": 4,
            },
        }
    )
    verdict = structural_critic([strong], registry, reasoning_contract=contract)
    assert verdict.ok, verdict.reasons


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
    leaked_entity = problem.entities[0]
    planner = Planner(ScriptedLLM.for_toy_pathway(), default_registry())

    real_invent = planner.invent
    prompts_seen: list[list[str]] = []
    calls = {"n": 0}

    async def invent_leaking_first(prob, n, **kwargs):
        # `prob` here is the ANONYMISED problem -- prob.entities[0] is "e0", not
        # a native name. The leak is injected from the ORIGINAL problem on
        # purpose: this test exercises the critic backstop, and the only way to
        # reach it now is to bypass the anonymiser the way a real leak never
        # can. That the test needs this is itself the point.
        prompts_seen.append(list(kwargs.get("rejected_because") or []))
        specs = await real_invent(prob, n, **kwargs)
        calls["n"] += 1
        if calls["n"] == 1:
            # Leak on the first attempt only, the way the live planner did.
            # A COPY, not a mutation: ScriptedLLM hands back the same DomainSpec
            # objects every call, so mutating one would make the "regenerated"
            # spec the same leaking instance and the loop could never converge.
            specs[0] = specs[0].model_copy(
                update={"forbidden": [f"Do not mention {leaked_entity}."]}
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


def test_shared_infrastructure_does_not_count_as_tool_overlap():
    """The exact domain set that made a live run give up entirely.

    Primary axes conservation / dynamics / causality -- all distinct -- and
    three unrelated languages. Rejected round after round until PlanError,
    solely because two of them both picked `build_graph` and `solve` out of a
    14-tool catalog. `solve` appeared in 8 of the 11 domains invented across
    those rounds; sharing it says nothing about orthogonality.
    """
    specs = [
        _spec(
            "series_flux_ledger",
            Axis.CONSERVATION,
            "directed capacity graph with per-edge flux weights and a ledger",
            ["build_graph", "simplify", "solve"],
        ),
        _spec(
            "gain_regime_response",
            Axis.DYNAMICS,
            "piecewise transfer-function over an input-multiplier axis",
            ["dimensional_check", "simulate", "solve"],
        ),
        _spec(
            "cause_chain_propagation",
            Axis.CAUSALITY,
            "signed influence chain with propagation-blocking gates",
            ["build_graph", "match_motif", "cut", "solve"],
        ),
    ]
    verdict = structural_critic(specs, default_registry())
    assert verdict.ok, verdict.reasons


def test_identical_toolsets_are_still_a_collision():
    """The fix must not become a way to pass by sharing EVERYTHING.

    If every tool is shared infrastructure then nothing is distinctive, and a
    naive implementation scores that as zero overlap -- exactly backwards. The
    fallback to raw Jaccard is what stops it.
    """
    specs = [
        _spec(
            "alpha_min",
            Axis.CONSERVATION,
            "a semiring of nonnegative rate values under a min-operator",
            ["solve", "simplify"],
        ),
        _spec(
            "beta_min",
            Axis.DYNAMICS,
            "orbit decomposition over a finite permutation group action",
            ["solve", "simplify"],
        ),
    ]
    verdict = structural_critic(specs, default_registry())
    assert not verdict.ok
    assert any("Jaccard" in r for r in verdict.reasons)


@pytest.mark.asyncio
async def test_exhausted_rounds_returns_a_best_effort_plan_not_an_exception():
    """Running out of rounds must not throw away every domain invented.

    Live, this raised PlanError after five model calls and eighty seconds and
    the operator got nothing at all -- for a set whose only objection was a
    tool-overlap proxy. The compromises are reported instead, so a caller that
    wants strictness can still refuse.
    """
    planner = Planner(ScriptedLLM.for_toy_pathway(), default_registry())

    # A critic that is never satisfied, so the loop always exhausts.
    import reagents.god.planner as mod

    original = mod.structural_critic
    try:
        mod.structural_critic = lambda *a, **k: CriticVerdict(
            ok=False, colliding_names=[], reasons=["synthetic objection"]
        )
        specs = await planner.plan(toy_problem(), n=3, max_rounds=2)
    finally:
        mod.structural_critic = original

    assert specs, "returned nothing at all"
    assert planner.last_plan_compromises, (
        "returned a compromised plan without recording what was wrong"
    )


@pytest.mark.asyncio
async def test_planner_retries_one_malformed_invention():
    class FlakyPlannerLLM:
        def __init__(self):
            self.invent_calls = 0

        async def complete(self, *, phase, **_):
            if phase == "invent":
                self.invent_calls += 1
                if self.invent_calls == 1:
                    raise LLMError("complete JSON did not match the schema")
                return InventedDomains(domains=toy_domains())
            return CriticVerdict(ok=True)

    llm = FlakyPlannerLLM()
    specs = await Planner(llm, default_registry()).plan(toy_problem(), n=3)
    assert len(specs) == 3
    assert llm.invent_calls == 2


@pytest.mark.asyncio
async def test_planner_may_invent_no_domains_at_all():
    """An empty plan is a decision, not a failure.

    The count used to be an input the planner had to hit. Now it judges it, and
    "none" is a legal judgement -- the caller answers the problem itself. The
    critic must not run: there is nothing to call orthogonal, and every round
    would re-ask a question already answered."""

    class CountingLLM:
        def __init__(self):
            self.phases: list[str] = []

        async def complete(self, *, phase, **_):
            self.phases.append(phase)
            return InventedDomains(
                domains=[],
                rationale="one arithmetic step; a foreign language buys nothing",
            )

    llm = CountingLLM()
    planner = Planner(llm, default_registry())
    specs = await planner.plan(toy_problem())

    assert specs == []
    assert llm.phases == ["invent"], "the critic was asked to judge an empty set"
    assert "arithmetic" in planner.last_rationale
    assert planner.last_plan_compromises == []


@pytest.mark.asyncio
async def test_planner_leaves_the_count_open_when_none_is_pinned():
    """No number reaches the model unless an operator pinned one."""
    seen: dict[str, str] = {}

    class PromptCapturingLLM:
        async def complete(self, *, phase, user, **_):
            seen[phase] = user
            if phase == "invent":
                return InventedDomains(domains=toy_domains())
            return CriticVerdict(ok=True)

    await Planner(PromptCapturingLLM(), default_registry()).plan(toy_problem())
    assert "Decide how many domains" in seen["invent"]
    assert "exactly" not in seen["invent"].splitlines()[0]


@pytest.mark.asyncio
async def test_planner_pins_the_count_when_an_operator_asks_for_one():
    seen: dict[str, str] = {}

    class PromptCapturingLLM:
        async def complete(self, *, phase, user, **_):
            seen[phase] = user
            if phase == "invent":
                return InventedDomains(domains=toy_domains())
            return CriticVerdict(ok=True)

    specs = await Planner(PromptCapturingLLM(), default_registry()).plan(
        toy_problem(), n=2
    )
    assert "Invent exactly 2 domains" in seen["invent"]
    assert len(specs) == 2, "a pinned count must also cap what comes back"


@pytest.mark.asyncio
async def test_single_domain_plan_skips_the_orthogonality_critic():
    """One domain has no pair to be orthogonal to.

    Asking anyway spends a call and invites an objection about a set of one.
    Structural checks still run -- they are per-spec, not pairwise."""

    class CountingLLM:
        def __init__(self):
            self.phases: list[str] = []

        async def complete(self, *, phase, **_):
            self.phases.append(phase)
            if phase == "invent":
                return InventedDomains(domains=toy_domains()[:1])
            return CriticVerdict(ok=False, reasons=["should never be asked"])

    llm = CountingLLM()
    specs = await Planner(llm, default_registry()).plan(toy_problem())
    assert len(specs) == 1
    assert llm.phases == ["invent"]


@pytest.mark.asyncio
async def test_pinned_count_refuses_an_empty_plan():
    """Pinning n asks for n domains, not for an opinion about whether to bother.

    Routing an empty response to the direct answer here would let a run that
    was told to spawn 3 demigods spawn none and still report success."""
    from reagents.god.planner import PlanError

    class EmptyLLM:
        async def complete(self, *, phase, **_):
            return InventedDomains(domains=[])

    with pytest.raises(PlanError, match="pinned to 3"):
        await Planner(EmptyLLM(), default_registry()).plan(toy_problem(), n=3)
