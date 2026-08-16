"""Invent representation domains, then reject sets that are not orthogonal."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from reagents.contracts import Axis, DomainSpec, NativeProblem
from reagents.god.anonymize import anonymize_problem
from reagents.isolation import find_spec_leaks, native_terms
from reagents.llm.client import LLMClient
from reagents.tools.registry import ToolRegistry, UnknownToolError, tool_jaccard

JACCARD_THRESHOLD = 0.3
LANGUAGE_OVERLAP_THRESHOLD = 0.5
DEFAULT_DOMAIN_COUNT = 3
MAX_PLAN_ROUNDS = 4

INVENT_SYSTEM = """You are God. You do not solve the problem.
You invent representation domains so isolated demigods can reason in a foreign language.

Each domain MUST:
- have a short identifier name (snake_case)
- sit on 1-2 axes from the allowed axis list, with a distinct primary axis
- name a representation language (graph, algebra, orbits, measures, rewrite system, ...)
  not a strategy ("think harder about pathways" is illegal)
- choose 2-4 tools from the allowed tool list only
- include an artifact JSON schema with required findings (array) and conclusion (string)
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

Cover distinct axes. Do not invent executable tools."""

CRITIC_SYSTEM = """You are God's orthogonality critic.
Reject a set of domains if any two are paraphrases, share a primary axis, or describe a
strategy instead of a representation. Return colliding domain names to regenerate.
Accept only if the languages are genuinely different representations."""


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


def structural_critic(
    specs: list[DomainSpec],
    registry: ToolRegistry,
    *,
    jaccard_threshold: float = JACCARD_THRESHOLD,
    language_threshold: float = LANGUAGE_OVERLAP_THRESHOLD,
    terms: set[str] | None = None,
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

    for spec in specs:
        try:
            registry.bind(spec.tool_ids)
        except (UnknownToolError, ValueError) as exc:
            reasons.append(f"{spec.name}: bad tools ({exc})")
            colliding.add(spec.name)

    for i, a in enumerate(specs):
        for b in specs[i + 1 :]:
            jac = tool_jaccard(a.tool_ids, b.tool_ids)
            if jac > jaccard_threshold:
                reasons.append(
                    f"tool Jaccard {jac:.2f} > {jaccard_threshold}: {a.name} vs {b.name}"
                )
                colliding.add(a.name)
                colliding.add(b.name)
            overlap = language_overlap(a.language, b.language)
            if overlap > language_threshold:
                reasons.append(
                    f"language overlap {overlap:.2f} > {language_threshold}: {a.name} vs {b.name}"
                )
                colliding.add(a.name)
                colliding.add(b.name)

    return CriticVerdict(
        ok=not reasons, colliding_names=sorted(colliding), reasons=reasons
    )


class Planner:
    def __init__(self, llm: LLMClient, registry: ToolRegistry) -> None:
        self.llm = llm
        self.registry = registry
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
        rejected_because: list[str] | None = None,
    ) -> list[DomainSpec]:
        avoid = avoid or []
        forbidden_axes = forbidden_axes or []
        retry = ""
        if rejected_because:
            retry = (
                "The previous attempt was rejected for these reasons. Fix them "
                "rather than varying the wording:\n"
                + "\n".join(f"- {r}" for r in rejected_because)
                + "\n\n"
            )
        user = (
            f"{retry}"
            f"Invent exactly {n} domains for this native problem.\n\n"
            f"id: {problem.id}\n"
            f"statement: {problem.statement}\n"
            f"entities: {problem.entities}\n"
            f"constraints: {problem.constraints}\n"
            f"question: {problem.question}\n\n"
            f"Allowed axes: {[a.value for a in Axis]}\n"
            f"Allowed tools: {self.registry.ids()}\n"
            f"Do not use primary axes: {[a.value for a in forbidden_axes]}\n"
            f"Do not reuse names: {[s.name for s in avoid]}\n"
            f"Existing languages to stay away from: {[s.language for s in avoid]}\n"
        )
        invented = await self.llm.complete(
            system=INVENT_SYSTEM,
            user=user,
            response_model=InventedDomains,
            phase="invent",
        )
        return invented.domains[:n]

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
        specs = await self.invent(planning_problem, n)
        for _ in range(max_rounds):
            structural = structural_critic(specs, self.registry, terms=terms)
            verdict = structural
            if structural.ok:
                verdict = await self.llm_critic(specs)
                if verdict.ok:
                    return specs
            colliding = set(verdict.colliding_names)
            if not colliding:
                colliding = {s.name for s in specs}
            kept = [s for s in specs if s.name not in colliding]
            forbidden_axes = [s.primary_axis for s in kept]
            replacements = await self.invent(
                planning_problem,
                len(colliding),
                avoid=specs,
                forbidden_axes=forbidden_axes,
                # WHY the previous attempt was rejected. Regenerating without it
                # is asking the model to guess, and it will happily reproduce
                # the same leak -- the Transformer already feeds its leaks back
                # for exactly this reason (see transformer.forward).
                rejected_because=verdict.reasons,
            )
            specs = kept + replacements
        raise PlanError("could not invent an orthogonal domain set")
