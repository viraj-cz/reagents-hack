"""Envelope sealing: native-field terms must never reach a demigod."""

from __future__ import annotations

import json
import re
from typing import Any

from reagents.contracts import (
    ContextEnvelope,
    DomainProblem,
    DomainSpec,
    NativeProblem,
)

# Tokens this short collide with algebra (A, B, x) and are not treated as leaks.
_MIN_TERM_LEN = 3


def native_terms(problem: NativeProblem) -> set[str]:
    """Terms that identify the original field. God keeps these; demigods must not see them."""
    terms: set[str] = set()
    for raw in (*problem.entities,):
        cleaned = raw.strip()
        if len(cleaned) >= _MIN_TERM_LEN:
            terms.add(cleaned)
    return terms


def find_leaks(text: str, terms: set[str]) -> list[str]:
    """Return native terms that appear as whole tokens in `text`."""
    if not text or not terms:
        return []
    found: list[str] = []
    for term in sorted(terms, key=len, reverse=True):
        pattern = rf"(?<![\w-]){re.escape(term)}(?![\w-])"
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(term)
    return found


def _flatten(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True)


def domain_problem_text(problem: DomainProblem) -> str:
    return "\n".join(
        [
            problem.domain_name,
            problem.task,
            problem.notation_guide,
            _flatten(problem.representation),
        ]
    )


def envelope_visible_text(envelope: ContextEnvelope) -> str:
    """Concatenate every string a demigod can read from the envelope."""
    parts = [
        envelope.domain.name,
        envelope.domain.language,
        envelope.domain.transform_prompt,
        *envelope.domain.forbidden,
        *envelope.forbidden,
        domain_problem_text(envelope.problem),
        _flatten(envelope.artifact_schema),
    ]
    for tool in envelope.tools:
        parts.append(tool.id)
        parts.append(tool.description)
        parts.append(_flatten(tool.parameters_schema))
    return "\n".join(parts)


def assert_sealed(envelope: ContextEnvelope, terms: set[str]) -> list[str]:
    """Return leak terms found in demigod-visible envelope text."""
    return find_leaks(envelope_visible_text(envelope), terms)


def spec_visible_text(spec: DomainSpec) -> str:
    """The PLANNER-authored text a demigod will eventually see.

    Deliberately excludes `transform_prompt`: it is God's instruction to itself,
    describes the native problem by design, and the orchestrator blanks it
    before it ever reaches an envelope.
    """
    return "\n".join(
        [
            spec.name,
            spec.language,
            *spec.forbidden,
            _flatten(spec.artifact_schema),
        ]
    )


def find_spec_leaks(spec: DomainSpec, terms: set[str]) -> dict[str, list[str]]:
    """Leaks in the planner's own output, attributed to the field they came from.

    Worth checking separately, and early. `assert_sealed` scans the whole
    envelope, so a term the PLANNER wrote into `language` or `artifact_schema`
    is only caught after a transform call has already been spent on that domain
    -- and the failure reads as if the transform leaked, which it did not.

    Attribution matters because the fix differs by field: a leak in `language`
    means the invented representation is not actually abstract, while one in
    `artifact_schema` usually means a property was named after a native entity.
    """
    fields = {
        "name": [spec.name],
        "language": [spec.language],
        "forbidden": list(spec.forbidden),
        "artifact_schema": [_flatten(spec.artifact_schema)],
    }
    found: dict[str, list[str]] = {}
    for field, texts in fields.items():
        leaks = find_leaks("\n".join(texts), terms)
        if leaks:
            found[field] = leaks
    return found
