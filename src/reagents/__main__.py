"""Run the walking skeleton: python -m reagents"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from reagents.god.orchestrator import God
from reagents.isolation import assert_sealed, native_terms
from reagents.llm.client import make_llm
from reagents.llm.scripted import ScriptedLLM
from reagents.toy import toy_problem


def main() -> None:
    parser = argparse.ArgumentParser(description="God-to-demigod orchestration")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use Anthropic (requires ANTHROPIC_API_KEY) instead of the scripted toy client",
    )
    args = parser.parse_args()

    if args.live:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("--live requires ANTHROPIC_API_KEY")
        llm = make_llm()
    else:
        llm = ScriptedLLM.for_toy_pathway()

    solution = asyncio.run(_run(llm))
    print(json.dumps(solution.model_dump(), indent=2))


async def _run(llm) -> object:
    problem = toy_problem()
    god = God(llm)
    solution = await god.solve(problem)
    terms = native_terms(problem)
    print("domains:", [s.name for s in god.last_trace.specs])
    print("axes:", [[a.value for a in s.axes] for s in god.last_trace.specs])
    for envelope in god.last_trace.envelopes:
        leaks = assert_sealed(envelope, terms)
        print(f"envelope {envelope.domain.name} sealed: {not leaks}")
        if leaks:
            print("  leaks:", leaks)
    print("artifacts:", [a.domain_name for a in god.last_trace.artifacts])
    print("failures:", [f.model_dump() for f in god.last_trace.failures])
    return solution


if __name__ == "__main__":
    main()
