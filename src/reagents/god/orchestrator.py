"""God loop: plan, critic, transform, parallel spawn, integrate."""

from __future__ import annotations

import asyncio

from reagents.contracts import (
    ContextEnvelope,
    DemigodFailure,
    DomainArtifact,
    DomainProblem,
    DomainSpec,
    NativeProblem,
    NativeSolution,
    OrchestrationTrace,
)
from reagents.demigod.runtime import DemigodRuntime, IsolationGuard
from reagents.god.integrator import Integrator
from reagents.god.planner import Planner
from reagents.god.transformer import LeakError, Transformer
from reagents.isolation import assert_sealed, native_terms
from reagents.llm.client import LLMClient
from reagents.tools.registry import ToolRegistry, default_registry

ABSTRACT_FORBIDDEN = [
    "Use only symbols defined in the representation.",
    "Do not refer to biological proper names, genes, proteins, or metabolites.",
]


class God:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry | None = None,
        domain_count: int = 3,
    ) -> None:
        self.llm = llm
        self.registry = registry or default_registry()
        self.domain_count = domain_count
        self.planner = Planner(llm, self.registry)
        self.transformer = Transformer(llm)
        self.integrator = Integrator(llm)
        self.runtime = DemigodRuntime(llm)
        self.last_trace = OrchestrationTrace()

    def build_envelope(self, spec: DomainSpec, problem: DomainProblem) -> ContextEnvelope:
        pack = self.registry.bind(spec.tool_ids)
        # transform_prompt is God's instruction to itself; it must not enter the envelope.
        sealed_spec = spec.model_copy(update={"transform_prompt": ""})
        return ContextEnvelope(
            domain=sealed_spec,
            problem=problem,
            tools=pack.specs(),
            artifact_schema=spec.artifact_schema,
            forbidden=list(spec.forbidden) or list(ABSTRACT_FORBIDDEN),
        )

    async def solve(self, problem: NativeProblem) -> NativeSolution:
        terms = native_terms(problem)
        guard = IsolationGuard(terms)
        specs = await self.planner.plan(problem, n=self.domain_count)

        envelopes: list[ContextEnvelope] = []
        inverse_maps = []
        leaks: list[str] = []
        failures: list[DemigodFailure] = []

        for spec in specs:
            try:
                domain_problem, inverse = await self.transformer.forward(problem, spec)
            except LeakError as exc:
                leaks.extend(exc.leaks)
                failures.append(
                    DemigodFailure(
                        domain_name=spec.name,
                        reason="transform could not seal the domain problem",
                        isolation_violations=exc.leaks,
                    )
                )
                continue
            envelope = self.build_envelope(spec, domain_problem)
            found = assert_sealed(envelope, terms)
            if found:
                leaks.extend(found)
                failures.append(
                    DemigodFailure(
                        domain_name=spec.name,
                        reason="envelope leaked native terms",
                        isolation_violations=found,
                    )
                )
                continue
            envelopes.append(envelope)
            inverse_maps.append(inverse)

        results = await asyncio.gather(*[_spawn(self, env, guard) for env in envelopes])
        artifacts: list[DomainArtifact] = []
        for result in results:
            if isinstance(result, DomainArtifact):
                artifacts.append(result)
            else:
                failures.append(result)

        if artifacts:
            solution = await self.integrator.integrate(problem, artifacts, inverse_maps)
        else:
            solution = NativeSolution(
                problem_id=problem.id,
                answer="No demigod produced a usable artifact.",
                confidence=0.0,
                gaps=[f.reason for f in failures],
            )

        self.last_trace = OrchestrationTrace(
            specs=specs,
            inverse_maps=inverse_maps,
            envelopes=envelopes,
            artifacts=artifacts,
            failures=failures,
            leaks=leaks,
            solution=solution,
        )
        return solution


async def _spawn(god: God, envelope: ContextEnvelope, guard: IsolationGuard):
    pack = god.registry.bind(envelope.domain.tool_ids)
    return await god.runtime.run(envelope, pack, guard=guard)
