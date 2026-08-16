"""Translate domain artifacts back into the native field and resolve conflicts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from reagents.contracts import (
    DemiGodResult,
    InverseMap,
    NativeProblem,
    NativeSolution,
)
from reagents.llm.client import LLMClient

INTEGRATE_SYSTEM = """You are God integrating demigod artifacts.
You have the inverse maps from domain symbols back to native entities.
Each artifact is intended to be a COMPLETE alternative solution in a different
coordinate system. Translate each candidate into the native field, compare their
constraint results and certificates, resolve conflicts, and select or reconcile a
complete answer to the original question. Never combine incomplete fragments into an
answer no demigod actually established. Do not invent new domain reasoning.

`domain_contributions` must contain an entry ONLY for a domain whose artifact
appears below. Domains listed as failed produced nothing: do not describe what
they found, do not infer what they would have found, and do not give them an
entry. Attributing a finding to a domain that produced no artifact presents
unvalidated reasoning as if it had been checked."""


class IntegrationDraft(BaseModel):
    answer: str
    structured_answer: dict = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    domain_contributions: dict[str, str] = Field(default_factory=dict)
    conflicts: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class Integrator:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def integrate(
        self,
        problem: NativeProblem,
        artifacts: list[DemiGodResult],
        inverse_maps: list[InverseMap],
        failed_domains: list[str] | None = None,
    ) -> NativeSolution:
        # Only the inverse maps of domains that actually produced an artifact.
        # inverse_maps carries an entry per SEALED domain while artifacts carries
        # one per SUCCESSFUL domain, so passing them raw showed the integrator a
        # domain name with no artifact behind it -- and it filled the blank in,
        # attributing invented findings to a demigod that had failed.
        produced = {a.domain_name for a in artifacts}
        maps = {
            m.domain_name: m.symbol_to_native
            for m in inverse_maps
            if m.domain_name in produced
        }
        user = (
            f"Native problem id: {problem.id}\n"
            f"Statement: {problem.statement}\n"
            f"Question: {problem.question}\n"
            f"Entities: {problem.entities}\n"
            f"Native inputs: {problem.inputs}\n"
            f"Required outputs: {problem.required_outputs}\n"
            f"Native answer JSON schema: {problem.answer_schema}\n"
        )
        if problem.constraints:
            user += "\nConstraints (apply to the final answer):\n"
            for constraint in problem.constraints:
                user += f"- {constraint}\n"
            user += (
                "If a constraint requires a JSON object, `answer` must be that "
                "object as compact JSON text — no markdown fences, no surrounding "
                "prose. Put explanations in domain_contributions.\n"
            )
        user += f"\nInverse maps: {maps}\n\nArtifacts:\n"
        for artifact in artifacts:
            # `confidence` and `unknowns` are new to the integrator: the first
            # lets it weight conflicting claims instead of treating every
            # artifact as equally certain, the second tells it what was left
            # undetermined so it lands in `gaps` rather than being invented.
            user += (
                f"\n--- {artifact.domain_name} ---\n"
                f"payload: {artifact.payload}\n"
                f"justification: {artifact.justification}\n"
                f"confidence: {artifact.confidence}\n"
                f"unknowns: {artifact.unknowns}\n"
            )
            if artifact.isolation_violations:
                user += (
                    f"CAUTION: this artifact used native terms "
                    f"{artifact.isolation_violations}, so part of its reasoning "
                    f"may have left its domain. Weigh it accordingly.\n"
                )

        # Named explicitly so the integrator does not have to infer absence.
        if failed_domains:
            user += (
                f"\nThese domains produced NO artifact and must not appear in "
                f"domain_contributions: {sorted(failed_domains)}\n"
            )
        draft = await self.llm.complete(
            system=INTEGRATE_SYSTEM,
            user=user,
            response_model=IntegrationDraft,
            phase="integrate",
        )
        return NativeSolution(
            problem_id=problem.id,
            answer=draft.answer,
            structured_answer=draft.structured_answer,
            confidence=draft.confidence,
            domain_contributions=draft.domain_contributions,
            conflicts=draft.conflicts,
            gaps=draft.gaps,
        )
