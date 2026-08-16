"""Strip native entity names from a problem before the planner ever sees it.

WHY THIS EXISTS, RATHER THAN JUST CHECKING HARDER
-------------------------------------------------
`structural_critic` catches a planner leak and regenerates the spec, which is a
rate reduction: the planner is still shown every native term and asked not to
repeat it, so not leaking takes restraint and restraint sometimes fails. A live
run lost a whole domain to `forbidden=['... the outlet ...']`.

This removes the channel instead. A planner that never saw "outlet" cannot
write "outlet", so the check becomes a backstop rather than the defence.

WHAT SURVIVES, AND WHY THAT IS THE POINT
----------------------------------------
`native_terms` derives from `problem.entities` alone, so those are exactly the
strings replaced -- and exactly the strings the seal is checked against. General
vocabulary is untouched:

    "Pump A moves water from tank 1 to tank 2"
     -> "e0 moves water from e2 to e3"

"water", "moves" and the sentence structure remain, so the planner can still
tell a flow problem from a folding problem and pick sensible tools, while the
identifiers it could leak are gone. The anonymisation is deliberately partial:
it is exactly as wide as the check it defends against, no wider.

This is NOT the seal a demigod gets. The transform still does the real
projection into an invented language; this only ensures GOD's own planning step
cannot contaminate the spec it writes.
"""

from __future__ import annotations

import re
from typing import Any

from reagents.contracts import NativeProblem
from reagents.isolation import find_leaks, native_terms


class AnonymizationError(RuntimeError):
    """Raised when a term survives substitution.

    Loud on purpose. The entire value of this module is the guarantee that the
    planner cannot see a native term, and a silent partial substitution would
    leave that guarantee believed but false -- worse than not having it, since
    the leak check downstream would then be the only defence while everyone
    assumed there were two.
    """


def anonymize_problem(problem: NativeProblem) -> tuple[NativeProblem, dict[str, str]]:
    """Return the problem with entity names replaced by symbols, plus the map.

    The map is GOD's alone. It is returned so a caller can relate a planner
    decision back to the real entity, and must never travel to a demigod -- the
    envelope has no field for it, which is the structural reason it cannot.

    Substitution is longest-first, for the same reason `find_leaks` scans that
    way: with "tank 1" and "tank" both present, replacing the short one first
    would leave "e5 1" and the longer term would never match.
    """
    terms = native_terms(problem)
    if not terms:
        return problem, {}

    ordered = sorted(terms, key=len, reverse=True)
    symbol_of = {term: f"e{i}" for i, term in enumerate(sorted(terms))}

    def scrub(text: str) -> str:
        for term in ordered:
            # Same boundary rule as find_leaks, so anything that check would
            # flag is something this substitution removes. Two different
            # notions of "a whole token" would leave a gap between them.
            pattern = rf"(?<![\w-]){re.escape(term)}(?![\w-])"
            text = re.sub(pattern, symbol_of[term], text, flags=re.IGNORECASE)
        return text

    def scrub_value(value: Any) -> Any:
        """Scrub native labels in every planner-visible structured field."""
        if isinstance(value, str):
            return scrub(value)
        if isinstance(value, list):
            return [scrub_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(scrub_value(item) for item in value)
        if isinstance(value, dict):
            return {scrub(str(key)): scrub_value(item) for key, item in value.items()}
        return value

    anonymized = NativeProblem(
        id=scrub(problem.id),
        statement=scrub(problem.statement),
        entities=[symbol_of[t.strip()] for t in problem.entities if t.strip() in terms],
        sensitive_terms=[
            symbol_of[t.strip()] for t in problem.sensitive_terms if t.strip() in terms
        ],
        constraints=[scrub(c) for c in problem.constraints],
        question=scrub(problem.question),
        inputs=scrub_value(problem.inputs),
        required_outputs=[scrub(item) for item in problem.required_outputs],
        answer_schema=scrub_value(problem.answer_schema),
    )

    # The guarantee, asserted rather than assumed. `native_terms` of the
    # ANONYMISED problem is now the symbol set, so the original terms are what
    # must be absent -- checked against every field a planner is shown.
    residue = find_leaks(
        "\n".join(
            [
                anonymized.id,
                anonymized.statement,
                *anonymized.constraints,
                anonymized.question,
                *anonymized.entities,
                *anonymized.sensitive_terms,
                str(anonymized.inputs),
                *anonymized.required_outputs,
                str(anonymized.answer_schema),
            ]
        ),
        terms,
    )
    if residue:
        raise AnonymizationError(
            f"native terms survived anonymisation: {residue}. The planner would "
            f"have been shown terms it is checked against, so the guarantee this "
            f"module exists to provide would be false."
        )

    return anonymized, {v: k for k, v in symbol_of.items()}
