"""God loop: plan, critic, transform, parallel spawn, integrate."""

from __future__ import annotations

import asyncio

from reagents.contracts import (
    Budget,
    ContextEnvelope,
    DemiGodResult,
    DomainProblem,
    DomainSpec,
    NativeProblem,
    NativeSolution,
    OrchestrationTrace,
    ToolAccess,
)
from reagents.demigod.runtime import (
    DemigodRuntime,
    DemigodRuntimeProtocol,
    IsolationGuard,
)
from reagents.god.direct import answer_directly
from reagents.god.integrator import Integrator
from reagents.god.planner import Planner
from reagents.god.transformer import (
    IncompleteProjectionError,
    LeakError,
    Transformer,
)
from reagents.isolation import assert_sealed, find_spec_leaks, native_terms
from reagents.llm.client import LLMClient, LLMError
from reagents.tools.registry import ToolRegistry, default_registry
from reagents.tracing import GOD_LANE, NullTracer, TraceSink, demigod_lane, summarize
from reagents.verification import NativeVerifier, finalize_solution, verify_solution

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
        domain_count: int | None = None,
        approved_write_tools: set[str] | None = None,
        runtime: DemigodRuntimeProtocol | None = None,
        tracer: TraceSink | None = None,
        verifier: NativeVerifier | None = None,
        budget: Budget | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry or default_registry()
        self.domain_count = domain_count
        """How many domains to invent, or None to let the planner decide.

        NONE IS THE DEFAULT, and it is the interesting setting: the number of
        demigods becomes a judgement about the problem rather than a constant
        applied to every problem alike. A simple question can be answered with
        no demigod at all; a hard one can be attacked from as many independent
        representations as genuinely differ.

        An integer pins it, which is worth having for cost control (`preflight`
        wants exactly one sandbox) but is a ceiling and a floor at once: it
        will spawn three demigods on arithmetic if it is told three."""

        self.tracer = tracer or NullTracer()
        self.verifier = verifier
        self.budget = budget or Budget()
        # Write authority is an operator decision, never something the planner or
        # demigod can grant itself. IDs must match the discovered catalog exactly.
        self.approved_write_tools = frozenset(approved_write_tools or set())
        self.planner = Planner(llm, self.registry, tracer=self.tracer)
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

    def build_envelope(
        self, spec: DomainSpec, problem: DomainProblem
    ) -> ContextEnvelope:
        tool_specs = self.registry.specs(spec.tool_ids)
        # transform_prompt is God's instruction to itself; it must not enter
        # the envelope.
        sealed_spec = spec.model_copy(update={"transform_prompt": ""})
        return ContextEnvelope(
            domain=sealed_spec,
            problem=problem,
            tools=tool_specs,
            artifact_schema=spec.artifact_schema,
            budget=self.budget,
            forbidden=list(spec.forbidden) or list(ABSTRACT_FORBIDDEN),
        )

    async def _check_natively(
        self,
        problem: NativeProblem,
        solution: NativeSolution,
        artifacts: list[DemiGodResult],
    ) -> NativeSolution:
        """Optional benchmark-owned finalization and verification.

        Shared by both endings -- an integrated multi-domain answer and a direct
        one -- because the check is on the answer in the native field and knows
        nothing about how many demigods produced it. A direct answer passes
        `artifacts=[]`, which a finalizer sees as "nothing to copy from" and
        leaves alone; the verifier still runs, which is the point. Skipping
        verification for direct answers would exempt the least-scrutinised path
        in the system from the only deterministic check it has.
        """
        if self.verifier is None:
            return solution
        self.tracer.emit(
            GOD_LANE,
            "FINALIZE",
            "Projecting accepted artifacts into the required native schema",
        )
        solution = await finalize_solution(self.verifier, problem, solution, artifacts)
        self.tracer.emit(
            GOD_LANE,
            "VERIFY",
            "Checking the candidate in the original problem domain",
        )
        report = await verify_solution(self.verifier, problem, solution)
        solution.verification = report.model_dump(mode="json")
        if not report.passed:
            solution.confidence = min(solution.confidence, 0.5)
            solution.gaps = [
                *solution.gaps,
                *[f"native verification: {error}" for error in report.errors],
            ]
        self.tracer.emit(
            GOD_LANE,
            "VERIFY",
            "native checks passed" if report.passed else "native checks found gaps",
            data={
                "passed": report.passed,
                "score": report.score,
                "checks": report.checks,
                "errors": report.errors,
            },
        )
        return solution

    async def _answer_directly(self, problem: NativeProblem) -> NativeSolution:
        """The zero-domain ending: God answers, nothing is spawned.

        Reached only when the planner returns an empty set, which it does when
        no invented representation would earn the sandbox it costs. See
        `god/direct.py` for why this is not the integrator with no artifacts.
        """
        rationale = self.planner.last_rationale
        self.tracer.emit(
            GOD_LANE,
            "DIRECT",
            "answering in the native field without spawning a demigod",
            data=rationale or None,
        )
        solution = await answer_directly(self.llm, problem, rationale=rationale)
        solution = await self._check_natively(problem, solution, [])
        self.last_trace = OrchestrationTrace(solution=solution, direct=True)
        self.tracer.emit(
            GOD_LANE,
            "DONE",
            "direct answer ready",
            data={"confidence": solution.confidence, "artifacts": 0, "domains": 0},
        )
        return solution

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
            (
                "Deciding how many coordinate systems, if any, this problem is "
                "worth splitting into"
                if self.domain_count is None
                else f"Looking for {self.domain_count} coordinate systems that "
                "each preserve and simplify the complete objective"
            ),
        )
        specs = await self.planner.plan(problem, n=self.domain_count)
        if not specs:
            # NO DOMAIN EARNED ITS SANDBOX. Answer natively and return; there is
            # nothing to transform, seal, spawn or integrate. The trace still
            # gets written, with empty spec/artifact lists, so a consumer can
            # tell this apart from a run whose demigods all failed by the fact
            # that `failures` is empty too.
            return await self._answer_directly(problem)
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
            # A leak confined to `forbidden` is REPAIRABLE, and repairing it is
            # strictly better than losing the domain. That field is a list of
            # warnings shown to the demigod -- so a planner that writes "do not
            # mention the outlet" really does leak "outlet", and the check is
            # right to see it. But it is advisory text, not the domain
            # definition: `language`, `transform_prompt` and `artifact_schema`
            # are what make the domain what it is, and none of them are touched
            # by swapping in the domain-agnostic wording that says the same
            # thing without naming anything.
            #
            # Observed live, and it cost half a run: the planner leaked
            # `forbidden=['outlet']`, the whole `transient_saturation_dynamics`
            # domain was discarded before the transform, and the final answer
            # had to report "the dedicated dynamics domain produced no
            # artifact" as a gap. One artifact instead of two, because a
            # warning list mentioned a word.
            if spec_leaks and set(spec_leaks) == {"forbidden"}:
                self.tracer.emit(
                    GOD_LANE,
                    "REPAIR",
                    f"{spec.name} forbidden-list named native terms; "
                    f"replaced with domain-agnostic wording",
                    data=sorted({t for ts in spec_leaks.values() for t in ts}),
                )
                spec = spec.model_copy(update={"forbidden": list(ABSTRACT_FORBIDDEN)})
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
                f"projecting native problem into {spec.name} "
                f"({spec.primary_axis.value})",
                # Structured, not only in the sentence: a non-terminal consumer
                # builds a node per domain and must not have to parse a name
                # back out of prose.
                data={"domain_name": spec.name, "axis": spec.primary_axis.value},
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
            except IncompleteProjectionError as exc:
                self.tracer.emit(
                    GOD_LANE,
                    "REJECT",
                    f"{spec.name} dropped part of the complete objective",
                    data=exc.errors,
                )
                failures.append(
                    _sealing_failure(
                        spec.name,
                        "transform did not preserve the complete objective: "
                        f"{exc.errors}",
                    )
                )
                continue
            except LLMError as exc:
                # Domains are independent: a refusal, truncation, or malformed
                # model response costs one representation, not the whole run.
                # Genuine implementation errors still propagate loudly.
                self.tracer.emit(
                    GOD_LANE,
                    "REJECT",
                    f"{spec.name} transform failed; continuing without it",
                    data=str(exc)[:200],
                )
                failures.append(_sealing_failure(spec.name, f"transform failed: {exc}"))
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
                    "domain_name": spec.name,
                    "axis": spec.primary_axis.value,
                    "language": spec.language,
                    "representation_keys": sorted(domain_problem.representation),
                    "objective_obligations": len(
                        domain_problem.projection_manifest.objective_ids
                    ),
                    "required_outputs": len(
                        domain_problem.projection_manifest.output_ids
                    ),
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
                conclusion = result.payload.get("conclusion") or result.payload.get(
                    "claim"
                )
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
                f"Translating {len(artifacts)} domain artifacts back into the "
                "original problem",
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

        if artifacts:
            solution = await self._check_natively(problem, solution, artifacts)

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
    pack = god.registry.bind(
        envelope.domain.tool_ids,
        subject_id=envelope.domain.name,
        budget=envelope.budget,
        allow_write=bool(write_tools),
        tracer=god.tracer,
    )
    return await god.runtime.run(envelope, pack, guard=guard)
