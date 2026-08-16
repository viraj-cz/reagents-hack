"""Isolated demigod: fresh messages, envelope-only tools, schema-validated artifact."""

from __future__ import annotations

from pydantic import BaseModel

from reagents.contracts import (
    ContextEnvelope,
    DemigodFailure,
    DomainArtifact,
)
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
    ) -> DomainArtifact | DemigodFailure:
        visible = envelope_visible_text(envelope)
        if guard:
            leaks = guard.check(visible)
            if leaks:
                return DemigodFailure(
                    domain_name=envelope.domain.name,
                    reason="envelope was not sealed before spawn",
                    isolation_violations=leaks,
                )

        allowed = {spec.id for spec in envelope.tools}
        bound = set(tools.ids())
        if allowed != bound:
            return DemigodFailure(
                domain_name=envelope.domain.name,
                reason=f"bound tools {sorted(bound)} != envelope tools {sorted(allowed)}",
            )

        user = _envelope_user(envelope)
        try:
            draft, trace = await self.llm.run_tool_loop(
                system=DEMIGOD_SYSTEM,
                user=user,
                tools=tools,
                response_model=DemigodDraft,
                budget=envelope.budget,
                phase=f"demigod:{envelope.domain.name}",
            )
        except UnboundToolError as exc:
            return DemigodFailure(domain_name=envelope.domain.name, reason=str(exc))
        except Exception as exc:  # noqa: BLE001 — convert tool/LLM failures into artifacts
            return DemigodFailure(domain_name=envelope.domain.name, reason=str(exc))

        schema_errors = validate_payload(draft.payload, envelope.artifact_schema)
        if schema_errors:
            return DemigodFailure(
                domain_name=envelope.domain.name,
                reason=f"artifact failed schema: {schema_errors}",
            )

        artifact = DomainArtifact(
            domain_name=envelope.domain.name,
            payload=draft.payload,
            justification=draft.justification,
            tool_trace=trace,
        )
        if guard:
            leaks = guard.check(f"{draft.justification}\n{draft.payload}")
            if leaks:
                return DemigodFailure(
                    domain_name=envelope.domain.name,
                    reason="artifact leaked native terms",
                    isolation_violations=leaks,
                )
        return artifact


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
