"""God loop: plan, critic, transform, parallel spawn, integrate."""

from __future__ import annotations

import asyncio

from reagents.contracts import (
    ContextEnvelope,
    DemiGodResult,
    DomainProblem,
    DomainSpec,
    NativeProblem,
    NativeSolution,
    OrchestrationTrace,
    RiskTier,
    ToolAccess,
)
from reagents.demigod.runtime import (
    DemigodRuntime,
    DemigodRuntimeProtocol,
    IsolationGuard,
)
from reagents.god.integrator import Integrator
from reagents.god.planner import Planner
from reagents.god.transformer import LeakError, Transformer
from reagents.isolation import assert_sealed, find_spec_leaks, native_terms
from reagents.llm.client import LLMClient, LLMError
from reagents.tools.registry import ToolRegistry, default_registry
from reagents.tracing import GOD_LANE, NullTracer, TraceSink, demigod_lane, summarize

# Deliberately domain-AGNOSTIC. These once named genes, proteins and
# metabolites, from when this system was biology-only. That was wrong twice
# over: GOD invents the domain at runtime, so naming one field's vocabulary
# does not generalise -- and repeating that vocabulary in every transform
# prompt tripped Anthropic's `bio` safety classifier, which refused the call
# and truncated the response. The system's own anti-leak instruction was the
# thing causing the refusal, on a problem about water tanks.
ABSTRACT_FORBIDDEN = [
    "Use only symbols defined in the representation.",
    "Do not refer to any entity by its name from the original problem.",
]


class God:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry | None = None,
        domain_count: int = 3,
        approved_write_tools: set[str] | None = None,
        approved_high_risk_tools: set[str] | None = None,
        runtime: DemigodRuntimeProtocol | None = None,
        tracer: TraceSink | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry or default_registry()
        self.domain_count = domain_count
        self.tracer = tracer or NullTracer()
        # Write authority is an operator decision, never something the planner or
        # demigod can grant itself. IDs must match the discovered catalog exactly.
        self.approved_write_tools = frozenset(approved_write_tools or set())
        self.approved_high_risk_tools = frozenset(
            approved_high_risk_tools or set()
        )
        self.planner = Planner(llm, self.registry)
        self.transformer = Transformer(llm)
        self.integrator = Integrator(llm)
        # WHERE a demigod executes is injected, not hardcoded. The default runs
        # it in this process (fast, no infrastructure, right for tests and the
        # scripted demo); SandboxDemigodRuntime gives each one its own Modal
        # sandbox. Both satisfy DemigodRuntimeProtocol and return DemiGodResult,
        # so nothing else in this file knows which is in use.
        self.runtime: DemigodRuntimeProtocol = runtime or DemigodRuntime(
            llm, tracer=self.tracer
        )
        set_tracer = getattr(self.runtime, "set_tracer", None)
        if callable(set_tracer):
            set_tracer(self.tracer)
        self.last_trace = OrchestrationTrace()

    def build_envelope(self, spec: DomainSpec, problem: DomainProblem) -> ContextEnvelope:
        tool_specs = self.registry.specs(spec.tool_ids)
        # transform_prompt is God's instruction to itself; it must not enter the envelope.
        sealed_spec = spec.model_copy(update={"transform_prompt": ""})
        return ContextEnvelope(
            domain=sealed_spec,
            problem=problem,
            tools=tool_specs,
            artifact_schema=spec.artifact_schema,
            forbidden=list(spec.forbidden) or list(ABSTRACT_FORBIDDEN),
        )

    async def solve(self, problem: NativeProblem) -> NativeSolution:
        self.tracer.emit(
            GOD_LANE,
            "START",
            f"problem={problem.id}; {summarize(problem.question)}",
        )
        deferred = self.registry.deferred_namespaces()
        catalog_note = f"{len(self.registry.ids())} local capabilities available"
        if deferred:
            catalog_note += f"; {len(deferred)} remote catalogs will be checked"
        else:
            catalog_note += "; no remote catalogs configured"
        self.tracer.emit(GOD_LANE, "CATALOG", catalog_note)
        terms = native_terms(problem)
        guard = IsolationGuard(terms)
        self.tracer.emit(
            GOD_LANE,
            "PLAN",
            f"Looking for {self.domain_count} representations that simplify different parts of the problem",
        )
        specs = await self.planner.plan(problem, n=self.domain_count)
        self.tracer.emit(
            GOD_LANE,
            "PLAN",
            f"selected {len(specs)} domains",
            data=[
                {
                    "name": spec.name,
                    "axis": spec.primary_axis.value,
                    "tools": spec.tool_ids,
                }
                for spec in specs
            ],
        )

        envelopes: list[ContextEnvelope] = []
        inverse_maps = []
        leaks: list[str] = []
        failures: list[DemiGodResult] = []

        for spec in specs:
            # Check the PLANNER's own output first. assert_sealed below scans
            # the whole envelope, so a term the planner wrote into `language`
            # or `artifact_schema` would otherwise only surface after a
            # transform call had been spent -- and would read as a transform
            # leak, which it is not.
            spec_leaks = find_spec_leaks(spec, terms)
            if spec_leaks:
                flat = sorted({t for ts in spec_leaks.values() for t in ts})
                leaks.extend(flat)
                detail = ", ".join(f"{f}={ts}" for f, ts in spec_leaks.items())
                failures.append(
                    _sealing_failure(
                        spec.name,
                        f"planner leaked native terms into the domain spec ({detail})",
                        flat,
                    )
                )
                self.tracer.emit(
                    GOD_LANE,
                    "REJECT",
                    f"{spec.name} domain specification leaked native terms",
                    data=flat,
                )
                continue
            self.tracer.emit(
                GOD_LANE,
                "TRANSFORM",
                f"projecting native problem into {spec.name} ({spec.primary_axis.value})",
            )
            try:
                domain_problem, inverse = await self.transformer.forward(problem, spec)
            except LeakError as exc:
                self.tracer.emit(
                    GOD_LANE,
                    "REJECT",
                    f"{spec.name} could not be sealed",
                    data=exc.leaks,
                )
                leaks.extend(exc.leaks)
                failures.append(
                    _sealing_failure(
                        spec.name,
                        "transform could not seal the domain problem",
                        exc.leaks,
                    )
                )
                continue
            except LLMError as exc:
                # One domain's transform failing must not end the run. This
                # propagated and killed a live run at the FIRST of two domains:
                # a refusal on `throughput_ceiling_orbits` meant the second,
                # perfectly healthy domain was never even attempted, and the
                # whole orchestration exited non-zero with no answer at all.
                #
                # The entire design premise is that domains are independent, so
                # treating one transform failure as fatal contradicts it -- and
                # partial recombination is exactly what `failed_domains` on the
                # integrator exists to describe.
                #
                # Scoped to LLMError deliberately: that is the "the model would
                # not cooperate" class (refusal, truncation, unparseable JSON),
                # all of which are recoverable by dropping this domain. A
                # genuine bug in here should still crash loudly rather than be
                # silently downgraded to a missing domain.
                self.tracer.emit(
                    GOD_LANE,
                    "REJECT",
                    f"{spec.name} transform failed; continuing without it",
                    data=str(exc)[:200],
                )
                failures.append(
                    _sealing_failure(spec.name, f"transform failed: {exc}")
                )
                continue
            envelope = self.build_envelope(spec, domain_problem)
            found = assert_sealed(envelope, terms)
            if found:
                self.tracer.emit(
                    GOD_LANE,
                    "REJECT",
                    f"{spec.name} envelope leaked native terms",
                    data=found,
                )
                leaks.extend(found)
                failures.append(
                    _sealing_failure(spec.name, "envelope leaked native terms", found)
                )
                continue
            envelopes.append(envelope)
            inverse_maps.append(inverse)
            self.tracer.emit(
                GOD_LANE,
                "SEALED",
                f"{spec.name} ready",
                data={
                    "axis": spec.primary_axis.value,
                    "language": spec.language,
                    "representation_keys": sorted(domain_problem.representation),
                    "tools": spec.tool_ids,
                },
            )

        # as_completed, not gather: demigods run in parallel and are collected
        # in whatever order they finish, so a slow domain does not hold up the
        # ones already done. gather would block on the slowest before GOD could
        # look at anything.
        self.tracer.emit(
            GOD_LANE,
            "SPAWN",
            f"{len(envelopes)} independent analyses are working in parallel",
        )
        artifacts: list[DemiGodResult] = []
        pending = [_spawn(self, env, guard) for env in envelopes]
        for finished in asyncio.as_completed(pending):
            result = await finished
            # One shape for success and failure; partition on status, never on
            # type. See demigod.result for why the union was collapsed.
            if result.status == "ok":
                artifacts.append(result)
                conclusion = result.payload.get("conclusion") or result.payload.get("claim")
                self.tracer.emit(
                    GOD_LANE,
                    "COLLECT",
                    f"accepted artifact from {result.domain_name}",
                    data={
                        "domain_name": result.domain_name,
                        "confidence": result.confidence,
                        "conclusion": conclusion,
                        "tool_calls": len(result.tool_trace),
                    },
                )
            else:
                failures.append(result)
                self.tracer.emit(
                    GOD_LANE,
                    "FAILURE",
                    f"{result.domain_name}: {result.error or 'unspecified failure'}",
                )

        if artifacts:
            self.tracer.emit(
                GOD_LANE,
                "INTEGRATE",
                f"Translating {len(artifacts)} domain artifacts back into the original problem",
            )
            solution = await self.integrator.integrate(
                problem,
                artifacts,
                inverse_maps,
                failed_domains=[f.domain_name for f in failures if f.domain_name],
            )
        else:
            solution = NativeSolution(
                problem_id=problem.id,
                answer="No demigod produced a usable artifact.",
                confidence=0.0,
                gaps=[f.error or "unspecified failure" for f in failures],
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
        self.tracer.emit(
            GOD_LANE,
            "DONE",
            "integrated answer ready",
            data={
                "confidence": solution.confidence,
                "artifacts": len(artifacts),
                "failures": len(failures),
                "leaks": len(leaks),
            },
        )
        return solution


def _sealing_failure(
    domain_name: str, reason: str, violations: list[str] | None = None
) -> DemiGodResult:
    """A failure raised GOD-side, before (or instead of) any demigod running.

    Same shape as a failure the demigod itself returns, so `OrchestrationTrace`
    holds one type and a consumer never has to ask where the failure came from.
    """
    from reagents.demigod.adapter import slugify_domain_name

    return DemiGodResult.failure(
        status="failed",
        error=reason,
        demigod_name=slugify_domain_name(domain_name),
        domain_name=domain_name,
        isolation_violations=violations,
    )


async def _spawn(
    god: God, envelope: ContextEnvelope, guard: IsolationGuard
) -> DemiGodResult:
    lane = demigod_lane(envelope.domain.name)
    write_tools = {
        spec.id for spec in envelope.tools if spec.access == ToolAccess.WRITE
    }
    unapproved = write_tools - god.approved_write_tools
    if unapproved:
        god.tracer.emit(
            lane,
            "FAIL",
            "write tools lack operator approval",
            data=sorted(unapproved),
        )
        return _sealing_failure(
            envelope.domain.name,
            f"write tools require operator approval: {sorted(unapproved)}",
        )
    high_risk_tools = {
        spec.id for spec in envelope.tools if spec.risk_tier == RiskTier.HIGH
    }
    unapproved_high_risk = high_risk_tools - god.approved_high_risk_tools
    if unapproved_high_risk:
        god.tracer.emit(
            lane,
            "FAIL",
            "high-risk tools lack operator approval",
            data=sorted(unapproved_high_risk),
        )
        return _sealing_failure(
            envelope.domain.name,
            "high-risk tools require operator approval: "
            f"{sorted(unapproved_high_risk)}",
        )
    pack = god.registry.bind(
        envelope.domain.tool_ids,
        subject_id=envelope.domain.name,
        budget=envelope.budget,
        allow_write=bool(write_tools),
        tracer=god.tracer,
    )
    return await god.runtime.run(envelope, pack, guard=guard)
