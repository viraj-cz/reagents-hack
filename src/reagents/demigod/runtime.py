"""Isolated demigod: fresh messages, envelope-only tools, schema-validated artifact."""

from __future__ import annotations

from typing import Protocol

from demigod.result import DemiGodResult, result_json_schema
from reagents.contracts import ContextEnvelope
from reagents.demigod.adapter import slugify_domain_name
from reagents.isolation import envelope_visible_text, find_leaks
from reagents.llm.client import LLMClient
from reagents.schema import validate_payload
from reagents.tools.registry import BoundToolPack, UnboundToolError
from reagents.tracing import NullTracer, TraceSink, demigod_lane, summarize

DEMIGOD_SYSTEM = """You are a demigod. You reason only in the representation language
in your envelope. You may call only the listed tools. You must return one artifact
matching the given schema. Your artifact is an independently complete candidate
solution to every projected obligation, not one portion for another agent to finish.
It must include constraint results, robustness or counterexamples, and a checkable
certificate.

You do not know the original problem. You do not speak to other demigods.
If a fact is not in the envelope, it does not exist.
Write justification only in the domain language.

Your final reply is one JSON object with exactly two keys: `payload`, holding
an object matching the artifact schema, and `justification`, a string. The
artifact schema describes what goes INSIDE `payload` -- do not return it as
your whole reply, and do not echo the schema itself back."""


class DemigodRuntimeProtocol(Protocol):
    """WHERE a demigod executes. The one thing God is polymorphic over.

    Two implementations: `DemigodRuntime` (this module, in GOD's own process)
    and `SandboxDemigodRuntime` (its own Modal sandbox). Both return the single
    `DemiGodResult` shape for success and failure alike, so the orchestrator
    partitions on `status` and never learns which runtime it is holding.

    Implementations must NOT raise for demigod-level failure. A demigod that
    fails is a manifest with `status != "ok"`, not an exception -- one failing
    domain must never take down the fan-out.
    """

    async def run(
        self,
        envelope: ContextEnvelope,
        tools: BoundToolPack,
        guard: IsolationGuard | None = None,
    ) -> DemiGodResult: ...


class IsolationGuard:
    """God-side leak list. Never serialized into the envelope."""

    def __init__(self, sealed_terms: set[str]) -> None:
        self.sealed_terms = sealed_terms

    def check(self, text: str) -> list[str]:
        return find_leaks(text, self.sealed_terms)


class DemigodRuntime:
    def __init__(self, llm: LLMClient, tracer: TraceSink | None = None) -> None:
        self.llm = llm
        self.tracer = tracer or NullTracer()

    def set_tracer(self, tracer: TraceSink) -> None:
        self.tracer = tracer

    async def run(
        self,
        envelope: ContextEnvelope,
        tools: BoundToolPack,
        guard: IsolationGuard | None = None,
    ) -> DemiGodResult:
        name = envelope.domain.name
        lane = demigod_lane(name)
        self.tracer.emit(lane, "START", "isolated reasoning stream opened")
        self.tracer.emit(
            lane,
            "SCOPE",
            f"axis={envelope.domain.primary_axis.value}; "
            f"language={envelope.domain.language}",
            data={
                "tools": [spec.id for spec in envelope.tools],
                "max_steps": envelope.budget.max_steps,
                "max_tool_calls": envelope.budget.max_tool_calls,
            },
        )

        def fail(reason: str, violations: list[str] | None = None) -> DemiGodResult:
            self.tracer.emit(lane, "FAIL", summarize(reason), data=violations)
            return DemiGodResult.failure(
                status="failed",
                error=reason,
                demigod_name=slugify_domain_name(name),
                domain_name=name,
                isolation_violations=violations,
            )

        visible = envelope_visible_text(envelope)
        if guard:
            leaks = guard.check(visible)
            if leaks:
                return fail("envelope was not sealed before spawn", leaks)

        allowed = {spec.id for spec in envelope.tools}
        bound = set(tools.ids())
        if allowed != bound:
            return fail(
                f"bound tools {sorted(bound)} != envelope tools {sorted(allowed)}"
            )

        user = _envelope_user(envelope)
        self.tracer.emit(lane, "MODEL", "reasoning/tool loop started")
        try:
            result, trace = await self.llm.run_tool_loop(
                system=DEMIGOD_SYSTEM,
                user=user,
                tools=tools,
                # THE contract, the same one the sandbox runtime uses. There
                # used to be a second model here -- `DemigodDraft`, with just
                # `payload` and `justification` -- and the drift was not
                # hypothetical: it had no `confidence` field, so every
                # in-process artifact was hardcoded to 0.5 and the integrator
                # weighed a brilliant result exactly like a doubtful one. The
                # scripted demigods even wrote `confidence` INSIDE the payload,
                # where nothing read it.
                #
                # `result_json_schema` nests the domain's artifact shape inside
                # `payload`, so the agent sees one schema instead of a generic
                # `payload: object` plus an unrelated "Artifact schema" line it
                # has to guess the relationship between. Guessing wrong is what
                # produced `payload Field required` on a completed artifact.
                response_model=DemiGodResult,
                response_schema=result_json_schema(envelope.artifact_schema),
                budget=envelope.budget,
                phase=f"demigod:{name}",
            )
        except UnboundToolError as exc:
            return fail(str(exc))
        except Exception as exc:
            return fail(str(exc))

        minimum_tool_calls = int(envelope.artifact_schema.get("x-min-tool-calls") or 0)
        if len(trace) < minimum_tool_calls:
            return fail(
                "artifact requires at least "
                f"{minimum_tool_calls} brokered tool calls; observed {len(trace)}"
            )

        schema_errors = validate_payload(result.payload, envelope.artifact_schema)
        if schema_errors:
            return fail(f"artifact failed schema: {schema_errors}")

        # Graded, not fatal -- see SandboxDemigodRuntime for why. The envelope
        # check above stays a hard gate; this one is a quality signal, because
        # by now the reasoning has already happened.
        artifact_leaks: list[str] = []
        if guard:
            artifact_leaks = guard.check(f"{result.justification}\n{result.payload}")

        # Envelope fields are RUNNER-owned and were stripped from the schema the
        # agent saw, so they are set here rather than trusted from the reply --
        # the same rule the sandbox runner applies when it reads result.json.
        result.demigod_name = slugify_domain_name(name)
        result.domain_name = name
        result.status = "ok"
        result.tool_trace = trace
        result.isolation_violations = artifact_leaks
        if artifact_leaks:
            result.blockers = [
                *result.blockers,
                f"used native terms {artifact_leaks}; some reasoning may "
                f"have left the domain",
            ]

        conclusion = result.payload.get("conclusion") or result.claim
        self.tracer.emit(lane, "REASON", summarize(result.justification))
        if conclusion:
            confidence = result.confidence
            message = summarize(conclusion)
            message = f"confidence={confidence}; {message}"
            self.tracer.emit(lane, "ARTIFACT", message)
        self.tracer.emit(
            lane,
            "DONE",
            f"artifact validated; tool_calls={len(trace)}",
        )
        return result


def _envelope_user(envelope: ContextEnvelope) -> str:
    tool_lines = []
    for spec in envelope.tools:
        tool_lines.append(f"- {spec.id}: {spec.description}")
    forbidden = envelope.forbidden or envelope.domain.forbidden
    return (
        f"Domain: {envelope.domain.name}\n"
        f"Axes: {[a.value for a in envelope.domain.axes]}\n"
        f"Language: {envelope.domain.language}\n"
        f"Notation: {envelope.problem.notation_guide}\n"
        f"Representation: {envelope.problem.representation}\n"
        f"Projection manifest: {envelope.problem.projection_manifest.model_dump()}\n"
        f"Task: {envelope.problem.task}\n"
        f"Forbidden: {forbidden}\n"
        f"Tools:\n" + "\n".join(tool_lines) + "\n"
        f"Artifact schema -- this is the shape of `payload`, not of your "
        f"whole reply: {envelope.artifact_schema}\n"
        f"Budget steps: {envelope.budget.max_steps}\n"
    )
