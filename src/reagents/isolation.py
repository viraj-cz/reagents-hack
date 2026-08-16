"""Envelope sealing: native-field terms must never reach a demigod."""

from __future__ import annotations

import json
import re
from typing import Any

from reagents.contracts import ContextEnvelope, DomainProblem, NativeProblem

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
