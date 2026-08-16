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
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Show concise God progress updates while the analysis runs",
    )
    subparsers = parser.add_subparsers(dest="command")
    tools_parser = subparsers.add_parser("tools", help="Inspect capability setup")
    tools_subparsers = tools_parser.add_subparsers(dest="tools_command", required=True)
    tools_subparsers.add_parser("list", help="List locally registered capability cards")
    doctor_parser = tools_subparsers.add_parser("doctor", help="Check runtimes, packages, and MCP credentials")
    doctor_parser.add_argument(
        "--connect",
        action="store_true",
        help="Connect to configured MCP servers and discover their tool schemas",
    )
    args = parser.parse_args()

    if args.command == "tools":
        asyncio.run(_run_tools_command(args))
        return

    if args.live:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("--live requires ANTHROPIC_API_KEY")
        llm = make_llm()
    else:
        llm = ScriptedLLM.for_toy_pathway()

    if args.trace:
        from reagents.tracing import TerminalTracer

        tracer = TerminalTracer()
    else:
        tracer = None
    solution = asyncio.run(_run(llm, tracer=tracer))
    if args.trace:
        print(f"\nFinal answer ({solution.confidence:.0%} confidence):\n{solution.answer}")
    else:
        print(json.dumps(solution.model_dump(), indent=2))


async def _run_tools_command(args) -> None:
    if args.tools_command == "list":
        from reagents.tools.registry import default_registry

        registry = default_registry()
        payload = {
            "tools": [spec.model_dump(mode="json") for spec in registry.specs()],
            "deferred_namespaces": registry.deferred_namespaces(),
        }
    else:
        from reagents.tools.doctor import doctor_report

        payload = await doctor_report(connect=args.connect)
    print(json.dumps(payload, indent=2))


async def _run(llm, tracer=None) -> object:
    problem = toy_problem()
    god = God(llm, tracer=tracer)
    solution = await god.solve(problem)
    if tracer is None:
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
