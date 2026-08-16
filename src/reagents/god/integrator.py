"""Translate domain artifacts back into the native field and resolve conflicts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from reagents.contracts import (
    DomainArtifact,
    InverseMap,
    NativeProblem,
    NativeSolution,
)
from reagents.llm.client import LLMClient

INTEGRATE_SYSTEM = """You are God integrating demigod artifacts.
You have the inverse maps from domain symbols back to native entities.
Translate each artifact into the native field, resolve conflicts, and answer the
original question. Do not invent new domain reasoning. Integration is translation
plus conflict resolution, not concatenation of essays."""


class IntegrationDraft(BaseModel):
    answer: str
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
        artifacts: list[DomainArtifact],
        inverse_maps: list[InverseMap],
    ) -> NativeSolution:
        maps = {m.domain_name: m.symbol_to_native for m in inverse_maps}
        user = (
            f"Native problem id: {problem.id}\n"
            f"Statement: {problem.statement}\n"
            f"Question: {problem.question}\n"
            f"Entities: {problem.entities}\n\n"
            f"Inverse maps: {maps}\n\n"
            f"Artifacts:\n"
        )
        for artifact in artifacts:
            user += (
                f"\n--- {artifact.domain_name} ---\n"
                f"payload: {artifact.payload}\n"
                f"justification: {artifact.justification}\n"
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
            confidence=draft.confidence,
            domain_contributions=draft.domain_contributions,
            conflicts=draft.conflicts,
            gaps=draft.gaps,
        )
