"""Isolated demigod: fresh messages, envelope-only tools, schema-validated artifact."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from demigod.result import DemiGodResult
from reagents.contracts import ContextEnvelope
from reagents.demigod.adapter import slugify_domain_name
from reagents.isolation import envelope_visible_text, find_leaks
from reagents.llm.client import LLMClient
from reagents.schema import validate_payload
from reagents.tools.registry import BoundToolPack, UnboundToolError

DEMIGOD_SYSTEM = """You are a demigod. You reason only in the representation language
in your envelope. You may call only the listed tools. You must return one artifact
matching the given schema.

You do not know the original problem. You do not speak to other demigods.
If a fact is not in the envelope, it does not exist.
Write justification only in the domain language."""


class DemigodDraft(BaseModel):
    payload: dict
    justification: str


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
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def run(
        self,
        envelope: ContextEnvelope,
        tools: BoundToolPack,
        guard: IsolationGuard | None = None,
    ) -> DemiGodResult:
        name = envelope.domain.name

        def fail(reason: str, violations: list[str] | None = None) -> DemiGodResult:
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
        try:
            draft, trace = await self.llm.run_tool_loop(
                system=DEMIGOD_SYSTEM,
                user=user,
                tools=tools,
                response_model=DemigodDraft,
                budget=envelope.budget,
                phase=f"demigod:{name}",
            )
        except UnboundToolError as exc:
            return fail(str(exc))
        except Exception as exc:  # noqa: BLE001 — tool/LLM failures become manifests
            return fail(str(exc))

        schema_errors = validate_payload(draft.payload, envelope.artifact_schema)
        if schema_errors:
            return fail(f"artifact failed schema: {schema_errors}")

        # Graded, not fatal -- see SandboxDemigodRuntime for why. The envelope
        # check above stays a hard gate; this one is a quality signal, because
        # by now the reasoning has already happened.
        artifact_leaks: list[str] = []
        if guard:
            artifact_leaks = guard.check(f"{draft.justification}\n{draft.payload}")

        return DemiGodResult(
            claim=draft.justification,
            # This runtime has no calibrated self-assessment to offer: the draft
            # carries no confidence field. 1.0 would be a lie and 0.0 reads as
            # failure, so a schema-valid, leak-free artifact sits in the middle.
            # The sandbox runtime gets a real number from the agent.
            confidence=0.5,
            payload=draft.payload,
            method="in-process demigod runtime (reagents.demigod.runtime)",
            justification=draft.justification,
            tool_trace=trace,
            isolation_violations=artifact_leaks,
            blockers=(
                [
                    f"used native terms {artifact_leaks}; some reasoning may "
                    f"have left the domain"
                ]
                if artifact_leaks
                else []
            ),
            demigod_name=slugify_domain_name(name),
            domain_name=name,
            status="ok",
        )


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
        f"Task: {envelope.problem.task}\n"
        f"Forbidden: {forbidden}\n"
        f"Tools:\n" + "\n".join(tool_lines) + "\n"
        f"Artifact schema: {envelope.artifact_schema}\n"
        f"Budget steps: {envelope.budget.max_steps}\n"
    )
