"""Invent representation domains, then reject sets that are not orthogonal."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from reagents.contracts import Axis, DomainSpec, NativeProblem
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
- list abstract forbidden rules (do not paste native entity names)

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
    return {tok for tok in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if len(tok) > 2}


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
) -> CriticVerdict:
    reasons: list[str] = []
    colliding: set[str] = set()

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
            reasons.append(f"shared primary axis {spec.primary_axis.value}: {owner} vs {spec.name}")
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
                reasons.append(f"tool Jaccard {jac:.2f} > {jaccard_threshold}: {a.name} vs {b.name}")
                colliding.add(a.name)
                colliding.add(b.name)
            overlap = language_overlap(a.language, b.language)
            if overlap > language_threshold:
                reasons.append(
                    f"language overlap {overlap:.2f} > {language_threshold}: {a.name} vs {b.name}"
                )
                colliding.add(a.name)
                colliding.add(b.name)

    return CriticVerdict(ok=not reasons, colliding_names=sorted(colliding), reasons=reasons)


class Planner:
    def __init__(self, llm: LLMClient, registry: ToolRegistry) -> None:
        self.llm = llm
        self.registry = registry

    async def invent(
        self,
        problem: NativeProblem,
        n: int,
        *,
        avoid: list[DomainSpec] | None = None,
        forbidden_axes: list[Axis] | None = None,
    ) -> list[DomainSpec]:
        avoid = avoid or []
        forbidden_axes = forbidden_axes or []
        user = (
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
        specs = await self.invent(problem, n)
        for _ in range(max_rounds):
            structural = structural_critic(specs, self.registry)
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
                problem,
                len(colliding),
                avoid=specs,
                forbidden_axes=forbidden_axes,
            )
            specs = kept + replacements
        raise PlanError("could not invent an orthogonal domain set")
