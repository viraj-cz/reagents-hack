"""Invent representation domains, then reject sets that are not orthogonal."""

from __future__ import annotations

import sys
from typing import Any

from pydantic import BaseModel, Field

from reagents.contracts import Axis, DomainSpec, NativeProblem
from reagents.god.anonymize import anonymize_problem
from reagents.isolation import find_spec_leaks, native_terms
from reagents.llm.client import LLMClient, LLMError
from reagents.tools.registry import ToolRegistry, UnknownToolError, tool_jaccard
from reagents.tracing import GOD_LANE, NullTracer, TraceSink

JACCARD_THRESHOLD = 0.3
LANGUAGE_OVERLAP_THRESHOLD = 0.5
DEFAULT_DOMAIN_COUNT = 3
MAX_PLAN_ROUNDS = 4
COMPLETE_ARTIFACT_KEYS = frozenset(
    {"candidate_solution", "constraint_results", "certificate", "conclusion"}
)

INVENT_SYSTEM = """You are God. You do not solve the problem.
You invent representation domains so isolated demigods can reason in a foreign language.

This is NOT work decomposition. Every domain is an alternative coordinate system for
the COMPLETE native problem. Every demigod must be able to return an independently
complete candidate solution; God will compare alternative proofs rather than assemble
partial answers.

Each domain MUST:
- have a short identifier name (snake_case)
- sit on 1-2 axes from the allowed axis list, with a distinct primary axis
- name a representation language (graph, algebra, orbits, measures, rewrite system, ...)
  not a strategy ("think harder about pathways" is illegal)
- choose 2-4 tools from the allowed tool list only
- use tool descriptions to bind one coherent representation family; when the
  catalog is large enough, keep tool sets disjoint across domains
- include `vision.read_image` in EVERY domain whose evidence is an image
  (micrographs, plate photographs, gels and blots, slides, chromatograms,
  scanned figures or plots, instrument screenshots). It is the only tool that
  can read a picture, so a domain that needs one and does not hold it produces
  an agent guessing at pixels it never looked at. This is the exception to
  keeping tool sets disjoint: share it across every domain that needs it rather
  than making one domain the designated looker -- each artifact must stand
  alone.
- include an artifact JSON schema requiring candidate_solution (object),
  constraint_results (object), certificate (object), and conclusion (string)
- list abstract forbidden rules

NAMING RULE, AND IT IS CHECKED MECHANICALLY. Four fields are scanned for the
problem's own entity names: `name`, `language`, `forbidden`, and
`artifact_schema` (keys and values, at every depth). A domain whose scan comes
back non-empty is thrown away and regenerated, so a single borrowed noun costs
the whole domain.

The trap is `forbidden`: a rule that says "do not mention the <entity>" names
the entity, and is a leak. Write the rules without referring to anything
specific -- "use only symbols defined in the representation" says the same
thing and scans clean. The same applies to a schema property named after an
entity, and to a domain name built from one.

You may reason ABOUT the entities to choose good representations; you may not
carry their names into these four fields.

The transform_prompt must explicitly preserve every input, constraint, objective, and
required output while changing only the representation language. Reject any language
that makes only one aspect of the problem easier but cannot express a full solution.

When a reasoning contract is supplied, it is mandatory rather than advisory:
- require every artifact key it lists in artifact_schema.required
- copy minimum_broker_calls to artifact_schema.x-min-tool-calls
- make experiments an array and model_comparison an array, with the requested minimum
  number of configurations reflected by minItems
- choose at least one tool matching required_tool_prefix and
  required_compute_suffix for every domain
- express its remaining requirements as checkable artifact fields

Cover distinct axes. Do not invent executable tools."""

CRITIC_SYSTEM = """You are God's orthogonality critic.
Reject a set of domains if any two are paraphrases, share a primary axis, or describe a
strategy instead of a representation. Return colliding domain names to regenerate.
Accept only if the languages are genuinely different representations and EACH domain's
artifact is a complete candidate solution, never a partial contribution."""


class InventedDomains(BaseModel):
    domains: list[DomainSpec]


class CriticVerdict(BaseModel):
    ok: bool
    colliding_names: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class PlanError(RuntimeError):
    pass


def _token_set(text: str) -> set[str]:
    return {
        tok
        for tok in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
        if len(tok) > 2
    }


def language_overlap(a: str, b: str) -> float:
    ta, tb = _token_set(a), _token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def distinctive_tools(specs: list[DomainSpec]) -> dict[str, list[str]]:
    """Each spec's tools with the set's shared infrastructure removed.

    A tool that most domains pick carries no information about whether any two
    of them are orthogonal. Raw Jaccard cannot tell that apart: it scores
    sharing `solve` -- which 8 of 11 invented domains reached for -- exactly
    like sharing `entropy`, which one did.

    That is not hypothetical. On the water-tank problem with a 14-tool local
    catalog, three domains with primary axes conservation / dynamics /
    causality and completely unrelated languages (a min-algebra semiring, a
    ledger graph, a system of rate equations) were rejected round after round
    until the planner gave up, solely because two of them both picked
    `build_graph` and `solve`. The metric was measuring how small the catalog
    is, not how similar the approaches are.

    So overlap is judged on what is left after removing tools at least half the
    set uses. If that leaves a spec with nothing distinctive, the caller falls
    back to raw Jaccard -- otherwise domains that genuinely picked identical
    toolsets would score 0 and pass, which is the opposite of the intent.
    """
    if not specs:
        return {}
    counts: dict[str, int] = {}
    for spec in specs:
        for tool in set(spec.tool_ids):
            counts[tool] = counts.get(tool, 0) + 1
    shared_threshold = max(2, (len(specs) + 1) // 2)
    common = {t for t, c in counts.items() if c >= shared_threshold}
    return {s.name: [t for t in s.tool_ids if t not in common] for s in specs}


def structural_critic(
    specs: list[DomainSpec],
    registry: ToolRegistry,
    *,
    jaccard_threshold: float = JACCARD_THRESHOLD,
    language_threshold: float = LANGUAGE_OVERLAP_THRESHOLD,
    terms: set[str] | None = None,
    reasoning_contract: dict[str, Any] | None = None,
) -> CriticVerdict:
    reasons: list[str] = []
    colliding: set[str] = set()

    # A LEAK IS A REGENERATION TRIGGER, not a later fatality. This check used to
    # live only in `God.solve`, which runs after planning has finished -- so the
    # planner's own regeneration loop, already sitting right here and already
    # willing to throw a spec away and ask for another, never learned that a
    # spec had leaked. A domain was simply lost.
    #
    # Cost of that, observed live: the planner wrote `forbidden=['...outlet...']`
    # and `transient_saturation_dynamics` was discarded before its transform,
    # leaving the run to report "the dedicated dynamics domain produced no
    # artifact" as a gap. One artifact instead of two, for a word.
    #
    # `terms` is optional so existing callers keep working; when omitted this is
    # exactly the critic it was before.
    if terms:
        for spec in specs:
            spec_leaks = find_spec_leaks(spec, terms)
            if spec_leaks:
                detail = ", ".join(
                    f"{field}={found}" for field, found in sorted(spec_leaks.items())
                )
                reasons.append(f"{spec.name}: leaked native terms ({detail})")
                colliding.add(spec.name)

    names = [s.name for s in specs]
    if len(names) != len(set(names)):
        reasons.append("duplicate domain names")
        seen: set[str] = set()
        for name in names:
            if name in seen:
                colliding.add(name)
            seen.add(name)

    primary: dict[Axis, str] = {}
    for spec in specs:
        owner = primary.get(spec.primary_axis)
        if owner is not None:
            reasons.append(
                f"shared primary axis {spec.primary_axis.value}: {owner} vs {spec.name}"
            )
            colliding.add(spec.name)
            colliding.add(owner)
        else:
            primary[spec.primary_axis] = spec.name

    contract = reasoning_contract or {}
    required_by_contract = set(contract.get("artifact_required_keys") or [])
    minimum_calls = int(contract.get("minimum_broker_calls") or 0)
    minimum_models = int(contract.get("minimum_model_configurations") or 0)
    required_prefix = str(contract.get("required_tool_prefix") or "")
    required_suffix = str(contract.get("required_compute_suffix") or "")

    for spec in specs:
        required = set(spec.artifact_schema.get("required") or [])
        missing_artifacts = sorted(
            (COMPLETE_ARTIFACT_KEYS | required_by_contract) - required
        )
        if missing_artifacts:
            reasons.append(
                f"{spec.name}: artifact schema is partial; missing {missing_artifacts}"
            )
            colliding.add(spec.name)
        if (
            minimum_calls
            and spec.artifact_schema.get("x-min-tool-calls") != minimum_calls
        ):
            reasons.append(
                f"{spec.name}: artifact schema must set x-min-tool-calls="
                f"{minimum_calls}"
            )
            colliding.add(spec.name)
        properties = spec.artifact_schema.get("properties") or {}
        for field, minimum in (
            ("experiments", 1 if "experiments" in required_by_contract else 0),
            ("model_comparison", minimum_models),
        ):
            if not minimum:
                continue
            field_schema = properties.get(field) or {}
            if (
                field_schema.get("type") != "array"
                or int(field_schema.get("minItems") or 0) < minimum
            ):
                reasons.append(
                    f"{spec.name}: {field} must be an array with minItems>={minimum}"
                )
                colliding.add(spec.name)
        if (
            required_prefix
            and required_suffix
            and not any(
                tool_id.startswith(required_prefix)
                and tool_id.endswith(required_suffix)
                for tool_id in spec.tool_ids
            )
        ):
            reasons.append(
                f"{spec.name}: must select a compute tool matching "
                f"{required_prefix}*{required_suffix}"
            )
            colliding.add(spec.name)
        try:
            registry.bind(spec.tool_ids)
        except (UnknownToolError, ValueError) as exc:
            reasons.append(f"{spec.name}: bad tools ({exc})")
            colliding.add(spec.name)

    distinctive = distinctive_tools(specs)
    for i, a in enumerate(specs):
        for b in specs[i + 1 :]:
            # Judged on distinctive tools, not raw ones -- see distinctive_tools
            # for the run this cost. Falls back to raw when either side has
            # nothing distinctive, so two domains that picked the SAME toolset
            # are still caught rather than scoring a vacuous 0.
            da, db = distinctive.get(a.name, []), distinctive.get(b.name, [])
            if da and db:
                jac = tool_jaccard(da, db)
            else:
                jac = tool_jaccard(a.tool_ids, b.tool_ids)
            if jac > jaccard_threshold:
                reasons.append(
                    f"tool Jaccard {jac:.2f} > {jaccard_threshold}: "
                    f"{a.name} vs {b.name}"
                )
                colliding.add(a.name)
                colliding.add(b.name)
            overlap = language_overlap(a.language, b.language)
            if overlap > language_threshold:
                reasons.append(
                    f"language overlap {overlap:.2f} > {language_threshold}: "
                    f"{a.name} vs {b.name}"
                )
                colliding.add(a.name)
                colliding.add(b.name)

    return CriticVerdict(
        ok=not reasons, colliding_names=sorted(colliding), reasons=reasons
    )


class Planner:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        tracer: TraceSink | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.last_plan_compromises: list[str] = []
        """Critic objections the returned plan still carries, or empty.

        Non-empty means `plan` ran out of rounds and returned its best set
        anyway. Readable by a caller that would rather fail than proceed."""
        self.tracer = tracer or NullTracer()
        self.last_symbol_map: dict[str, str] = {}
        """Symbol -> native entity for the most recent plan. GOD's alone.

        Kept for diagnostics: a planner decision reads as "e3 is the bottleneck"
        and this is what turns that back into something a human can check. It
        must never enter an envelope, and there is no envelope field for it."""

    async def invent(
        self,
        problem: NativeProblem,
        n: int,
        *,
        avoid: list[DomainSpec] | None = None,
        forbidden_axes: list[Axis] | None = None,
        reserved_tool_ids: list[str] | None = None,
        rejected_because: list[str] | None = None,
    ) -> list[DomainSpec]:
        avoid = avoid or []
        forbidden_axes = forbidden_axes or []
        reserved_tool_ids = reserved_tool_ids or []
        feedback = rejected_because or []
        catalog = [
            {"id": spec.id, "description": spec.description}
            for spec in self.registry.specs()
        ]
        user = (
            f"Invent exactly {n} domains for this native problem.\n\n"
            f"id: {problem.id}\n"
            f"statement: {problem.statement}\n"
            f"entities: {problem.entities}\n"
            f"constraints: {problem.constraints}\n"
            f"question: {problem.question}\n\n"
            f"required_outputs: {problem.required_outputs}\n"
            f"reasoning_contract: "
            f"{problem.inputs.get('reasoning_contract', {})}\n\n"
            f"Allowed axes: {[a.value for a in Axis]}\n"
            f"Allowed tool catalog: {catalog}\n"
            f"Do not use primary axes: {[a.value for a in forbidden_axes]}\n"
            f"Tool IDs reserved by accepted domains; do not select them: "
            f"{reserved_tool_ids}\n"
            f"Prior rejected domain names/languages to stay distinct from: "
            f"{[{'name': s.name, 'language': s.language} for s in avoid]}\n"
            f"Rejection reasons to correct: {feedback}\n"
        )
        for attempt in range(2):
            try:
                invented = await self.llm.complete(
                    system=INVENT_SYSTEM,
                    user=user,
                    response_model=InventedDomains,
                    phase="invent",
                )
                return invented.domains[:n]
            except LLMError as exc:
                retryable = "did not match the schema" in str(
                    exc
                ) or "no complete JSON object" in str(exc)
                if attempt or not retryable:
                    raise
                self.tracer.emit(
                    GOD_LANE,
                    "REPAIR",
                    "planner response was incomplete; requesting one corrected draft",
                    data=str(exc)[:300],
                )
                user += (
                    "\nYour previous response was invalid: "
                    f"{str(exc)[:1200]}\nReturn exactly {n} complete domains. "
                    "Every domain must include every field required by the schema, "
                    "especially axes, language, tool_ids, transform_prompt, and "
                    "artifact_schema.\n"
                )
        raise AssertionError("unreachable")

    async def llm_critic(self, specs: list[DomainSpec]) -> CriticVerdict:
        payload: list[dict[str, Any]] = [
            {
                "name": s.name,
                "axes": [a.value for a in s.axes],
                "language": s.language,
                "tool_ids": s.tool_ids,
            }
            for s in specs
        ]
        return await self.llm.complete(
            system=CRITIC_SYSTEM,
            user=f"Judge orthogonality of these domains:\n{payload}",
            response_model=CriticVerdict,
            phase="critic",
        )

    async def plan(
        self,
        problem: NativeProblem,
        n: int = DEFAULT_DOMAIN_COUNT,
        max_rounds: int = MAX_PLAN_ROUNDS,
    ) -> list[DomainSpec]:
        # Namespace loaders are inert until planning. Provider failures are recorded
        # on the registry so local reasoning remains available during outages.
        await self.registry.load_deferred()
        # `terms` from the ORIGINAL problem, `planning_problem` without them.
        # The planner is shown symbols, so it cannot copy a native name into a
        # spec; the critic below still checks against the real terms, as a
        # backstop rather than as the defence. Two independent mechanisms, and
        # the cheap one is no longer the only one.
        #
        # The transform is deliberately NOT given this: it needs the real
        # problem to project, and its output is checked by find_leaks. Only
        # planning -- which produces free text that survives into the envelope
        # -- is done blind.
        terms = native_terms(problem)
        planning_problem, self.last_symbol_map = anonymize_problem(problem)
        reasoning_contract = planning_problem.inputs.get("reasoning_contract", {})
        specs = await self.invent(planning_problem, n)
        # Best seen so far, so exhausting the rounds returns something rather
        # than nothing. Scored by how many objections the critic raised.
        best: list[DomainSpec] = specs
        best_reasons: list[str] = ["not yet judged"]
        for round_index in range(max_rounds):
            structural = structural_critic(
                specs,
                self.registry,
                terms=terms,
                reasoning_contract=reasoning_contract,
            )
            verdict = structural
            if structural.ok:
                verdict = await self.llm_critic(specs)
                if verdict.ok:
                    self.last_plan_compromises = []
                    return specs
            if best_reasons == ["not yet judged"] or len(verdict.reasons) < len(
                best_reasons
            ):
                best, best_reasons = specs, verdict.reasons
            self.tracer.emit(
                GOD_LANE,
                "REPLAN",
                f"domain set rejected on round {round_index + 1}/{max_rounds}",
                data=list(verdict.reasons),
            )
            colliding = set(verdict.colliding_names)
            if not colliding:
                colliding = {s.name for s in specs}
            kept = [s for s in specs if s.name not in colliding]
            forbidden_axes = [s.primary_axis for s in kept]
            reserved_tool_ids = sorted(
                {tool_id for spec in kept for tool_id in spec.tool_ids}
            )
            replacements = await self.invent(
                planning_problem,
                len(colliding),
                avoid=specs,
                forbidden_axes=forbidden_axes,
                reserved_tool_ids=reserved_tool_ids,
                rejected_because=verdict.reasons,
            )
            specs = kept + replacements

        # ROUNDS EXHAUSTED, AND THAT IS NOT A REASON TO RETURN NOTHING. This
        # used to `raise PlanError`, throwing away every domain invented across
        # every round -- five model calls and eighty seconds, for a set whose
        # only remaining objection was a tool-overlap proxy.
        #
        # Seen live on the water-tank problem with containers off: the three
        # domains had primary axes conservation / dynamics / causality, all
        # distinct, and languages that shared nothing. They were rejected solely
        # because two of them both picked `solve` and `build_graph` out of a
        # 14-tool catalog. Orthogonal by every criterion that matters, discarded
        # for a metric artifact.
        #
        # The compromises are recorded rather than swallowed: a caller that
        # wants strictness can read `last_plan_compromises` and refuse, but the
        # default is a usable plan with its flaws stated. A partial answer beats
        # a stack trace.
        if not best:
            raise PlanError(
                f"the planner returned no domains at all across {max_rounds} rounds"
            )
        self.last_plan_compromises = best_reasons
        print(
            f"[planner] returning a best-effort domain set after {max_rounds} "
            f"rounds; unresolved: {best_reasons}",
            file=sys.stderr,
            flush=True,
        )
        return best
