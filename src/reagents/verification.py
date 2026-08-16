"""Native-field verification after alternative domain solutions are integrated."""

from __future__ import annotations

import inspect
from typing import Any, Protocol

from pydantic import BaseModel, Field

from reagents.contracts import DemiGodResult, NativeProblem, NativeSolution


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


async def finalize_solution(
    verifier: NativeVerifier,
    problem: NativeProblem,
    solution: NativeSolution,
    artifacts: list[DemiGodResult],
) -> NativeSolution:
    """Apply an optional deterministic native-schema projection.

    A model integrator can correctly select an artifact yet summarize a large
    structured payload instead of copying it. Benchmark-owned finalization may
    fill that transport-level gap from accepted artifacts, but receives no
    private truth and performs no new reasoning.
    """

    finalizer = getattr(verifier, "finalize", None)
    if not callable(finalizer):
        return solution
    raw = finalizer(problem, solution, artifacts)
    if inspect.isawaitable(raw):
        raw = await raw
    if isinstance(raw, NativeSolution):
        return raw
    return NativeSolution.model_validate(raw)


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


__all__ = [
    "NativeVerifier",
    "VerificationReport",
    "finalize_solution",
    "verify_solution",
]
