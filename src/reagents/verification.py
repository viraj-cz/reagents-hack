"""Native-field verification after alternative domain solutions are integrated."""

from __future__ import annotations

import inspect
from typing import Any, Protocol

from pydantic import BaseModel, Field

from reagents.contracts import NativeProblem, NativeSolution


class VerificationReport(BaseModel):
    """Deterministic checks authored outside the model reasoning loop."""

    passed: bool
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    checks: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class NativeVerifier(Protocol):
    """A benchmark or application-owned checker in the original domain."""

    def verify(
        self, problem: NativeProblem, solution: NativeSolution
    ) -> VerificationReport | Any: ...


async def verify_solution(
    verifier: NativeVerifier,
    problem: NativeProblem,
    solution: NativeSolution,
) -> VerificationReport:
    raw = verifier.verify(problem, solution)
    if inspect.isawaitable(raw):
        raw = await raw
    if isinstance(raw, VerificationReport):
        return raw
    return VerificationReport.model_validate(raw)


__all__ = ["NativeVerifier", "VerificationReport", "verify_solution"]
