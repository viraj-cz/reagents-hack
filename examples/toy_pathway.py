"""End-to-end toy pathway: invent 3 orthogonal domains, spawn demigods, integrate."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reagents.god.orchestrator import God
from reagents.isolation import assert_sealed, native_terms
from reagents.llm.scripted import ScriptedLLM
from reagents.toy import toy_problem


async def main() -> None:
    problem = toy_problem()
    god = God(ScriptedLLM.for_toy_pathway())
    solution = await god.solve(problem)
    terms = native_terms(problem)

    print("problem:", problem.id)
    print("question:", problem.question)
    print()
    for spec in god.last_trace.specs:
        print(f"domain {spec.name}  axis={spec.primary_axis.value}  tools={spec.tool_ids}")
    print()
    for envelope in god.last_trace.envelopes:
        leaks = assert_sealed(envelope, terms)
        status = "SEALED" if not leaks else f"LEAKED {leaks}"
        print(f"envelope {envelope.domain.name}: {status}")
        print(f"  visible tools: {[t.id for t in envelope.tools]}")
    print()
    print(json.dumps(solution.model_dump(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
