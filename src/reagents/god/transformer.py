"""Forward transform into a domain and God-only inverse maps."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from reagents.contracts import DomainProblem, DomainSpec, InverseMap, NativeProblem
from reagents.isolation import domain_problem_text, find_leaks, native_terms
from reagents.llm.client import LLMClient

TRANSFORM_SYSTEM = """You are God performing a Fourier-like projection.
Rewrite the native problem into the given domain language.

Rules:
- The representation, task, and notation_guide MUST use only invented symbols.
- Never copy a native entity name, or any recognisable variant of one.
- Put the mapping from invented symbols back to native names in symbol_to_native only.
- representation is structured data in the domain language (graph, equations, measures, ...).
- task is what the demigod must produce, stated only in domain notation."""


class TransformDraft(BaseModel):
    representation: dict[str, Any]
    task: str
    notation_guide: str
    symbol_to_native: dict[str, str] = Field(default_factory=dict)


class LeakError(RuntimeError):
    def __init__(self, leaks: list[str]) -> None:
        self.leaks = leaks
        super().__init__(f"transform leaked native terms: {leaks}")


class Transformer:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def forward(
        self,
        problem: NativeProblem,
        spec: DomainSpec,
        *,
        max_retries: int = 2,
    ) -> tuple[DomainProblem, InverseMap]:
        terms = native_terms(problem)
        last_leaks: list[str] = []
        for attempt in range(max_retries + 1):
            draft = await self.llm.complete(
                system=TRANSFORM_SYSTEM,
                user=_transform_user(problem, spec, last_leaks),
                response_model=TransformDraft,
                phase=f"transform:{spec.name}",
            )
            domain_problem = DomainProblem(
                domain_name=spec.name,
                representation=draft.representation,
                task=draft.task,
                notation_guide=draft.notation_guide,
            )
            leaks = find_leaks(domain_problem_text(domain_problem), terms)
            if not leaks:
                inverse = InverseMap(domain_name=spec.name, symbol_to_native=draft.symbol_to_native)
                return domain_problem, inverse
            last_leaks = leaks
        raise LeakError(last_leaks)


def _transform_user(problem: NativeProblem, spec: DomainSpec, leaks: list[str]) -> str:
    retry = ""
    if leaks:
        retry = (
            f"\nPrevious draft leaked these native terms: {leaks}. "
            "Replace every one with an invented symbol.\n"
        )
    return (
        f"{retry}"
        f"Domain name: {spec.name}\n"
        f"Axes: {[a.value for a in spec.axes]}\n"
        f"Language: {spec.language}\n"
        f"Transform instructions: {spec.transform_prompt}\n\n"
        f"Native statement: {problem.statement}\n"
        f"Native entities: {problem.entities}\n"
        f"Native constraints: {problem.constraints}\n"
        f"Native question: {problem.question}\n"
    )
