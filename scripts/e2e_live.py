"""End-to-end: GOD plans and seals -> DEMI_GODs in sandboxes -> GOD integrates.

    uv run python scripts/e2e_live.py                 # 2 domains, 6 turns each
    uv run python scripts/e2e_live.py --domains 3 --turns 10

The only path that exercises the whole system at once. Everything below it has
been proven separately -- images, mounts, the filesystem API, the agent loop,
the manifest round trip, nested spawning -- but the span from God's planner
through the adapter into a sandbox and back into the integrator has not.

COST. Every God phase and every demigod turn is an Anthropic call; Modal
compute is cents beside that. Defaults are deliberately small: 2 domains, 6
turns. Rough shape of one run at defaults:

    1 planner call + 2 transform calls + 1 integrate call   (God, sonnet)
    2 sandboxes x <=6 turns                                 (the demigods)

Raise --domains/--turns only once the pipeline is known to work.

Progress is printed as `>>> STAGE` lines so a watcher can follow a live run
without wading through streamed agent output.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

from reagents.demigod.sandbox_runtime import SandboxDemigodRuntime
from reagents.god.orchestrator import God
from reagents.llm.client import make_llm
from reagents.toy import toy_problem

_T0 = time.monotonic()


def stage(message: str) -> None:
    """A progress marker. Prefixed so a monitor can filter to just these."""
    print(f">>> [{time.monotonic() - _T0:6.1f}s] {message}", flush=True)


def instrument(god: God) -> None:
    """Wrap God's phases so a live run reports where it is.

    The orchestrator has no logging of its own -- deliberately, it is a library.
    Rather than thread a callback through it, this decorates the four phase
    objects in place. Test-harness code: it belongs here, not in the loop.
    """
    planner_plan = god.planner.plan
    transformer_forward = god.transformer.forward
    integrator_integrate = god.integrator.integrate
    runtime_run = god.runtime.run

    async def plan(problem, n):
        stage(f"PLAN: inventing {n} orthogonal domains")
        specs = await planner_plan(problem, n=n)
        for s in specs:
            axes = ", ".join(a.value for a in s.axes)
            stage(f"PLAN: '{s.name}' [{axes}] tools={s.tool_ids}")
        return specs

    async def forward(problem, spec):
        stage(f"TRANSFORM: projecting problem into '{spec.name}'")
        result = await transformer_forward(problem, spec)
        stage(f"TRANSFORM: '{spec.name}' sealed")
        return result

    async def run(envelope, tools, guard=None):
        name = envelope.domain.name
        stage(f"SPAWN: '{name}' -> sandbox")
        result = await runtime_run(envelope, tools, guard=guard)
        if result.status == "ok":
            stage(
                f"DONE: '{name}' status=ok confidence={result.confidence} "
                f"files={len(result.files)}"
            )
        else:
            stage(f"DONE: '{name}' status={result.status} error={result.error}")
        return result

    async def integrate(problem, artifacts, inverse_maps):
        stage(f"INTEGRATE: recombining {len(artifacts)} artifact(s)")
        return await integrator_integrate(problem, artifacts, inverse_maps)

    god.planner.plan = plan
    god.transformer.forward = forward
    god.integrator.integrate = integrate
    god.runtime.run = run


async def run_once(domains: int, turns: int, run_id: str) -> int:
    problem = toy_problem()
    stage(f"START run_id={run_id} domains={domains} turns={turns}")
    stage(f"PROBLEM: {problem.id} -- {problem.question}")

    god = God(
        make_llm(),
        domain_count=domains,
        # The seam. Swap for the default in-process runtime and the same God
        # loop runs without any infrastructure at all.
        runtime=SandboxDemigodRuntime(run_id=run_id, max_turns=turns),
    )
    instrument(god)

    solution = await god.solve(problem)

    stage("COMPLETE")
    trace = god.last_trace
    print("\n" + "=" * 72)
    print(f"artifacts: {len(trace.artifacts)}   failures: {len(trace.failures)}")
    if trace.leaks:
        print(f"isolation leaks: {trace.leaks}")
    for f in trace.failures:
        print(f"  FAILED {f.domain_name}: {f.error}")
    print("=" * 72)
    print(json.dumps(solution.model_dump(), indent=2))

    for artifact in trace.artifacts:
        print(f"\n--- {artifact.domain_name} (sandbox) ---")
        print(f"  confidence : {artifact.confidence}")
        print(f"  payload    : {json.dumps(artifact.payload)[:400]}")
        print(f"  files      : {artifact.files}")
        print(f"  unknowns   : {artifact.unknowns[:3]}")

    # Artifacts survive on the volume regardless of what the integrator said.
    print(f"\nartifacts on volume: uv run modal volume ls demigod-run-{run_id}-out")
    return 0 if trace.artifacts else 1


def load_env_file() -> None:
    """Load .env if present, so the key does not have to live in the shell.

    An exported variable only exists in the shell that exported it, which makes
    the run unrepeatable and un-delegatable -- anyone (or anything) driving this
    script from a different process has no key. A gitignored .env fixes that
    once. Existing environment variables win, so an explicit export still
    overrides the file.
    """
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv(env_path, override=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domains", type=int, default=2)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    load_env_file()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "error: ANTHROPIC_API_KEY not found for God's own loop (planner, "
            "transformer, integrator). The DEMI_GODs read theirs from the Modal "
            "secret `demigod-anthropic` instead -- Modal secrets are write-only, "
            "so that one cannot be reused here.\n"
            "Put it in a gitignored .env at the repo root (preferred -- any "
            "process can then run this):\n"
            "  echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env\n"
            "or export it in your shell:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...",
            file=sys.stderr,
        )
        return 2

    # `anthropic` lives in the optional `llm` extra, so a plain `uv sync` leaves
    # it out and make_llm() explodes on its first call -- AFTER a run_id and
    # volumes have been created. Fail here instead, with the fix.
    try:
        import anthropic  # noqa: F401
    except ModuleNotFoundError:
        print(
            "error: the `anthropic` package is not installed. It is in the "
            "optional `llm` extra, which `uv sync` does not install by default:\n"
            "  uv sync --extra llm",
            file=sys.stderr,
        )
        return 2

    run_id = args.run_id or f"e2e{uuid.uuid4().hex[:6]}"
    return asyncio.run(run_once(args.domains, args.turns, run_id))


if __name__ == "__main__":
    raise SystemExit(main())
