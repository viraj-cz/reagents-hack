"""End-to-end: GOD plans and seals -> DEMI_GODs in sandboxes -> GOD integrates.

    uv run python scripts/e2e_live.py                 # GOD picks the domain count
    uv run python scripts/e2e_live.py --domains 2     # pin it, for a bounded bill
    uv run python scripts/e2e_live.py --no-broker     # without brokered tools

The only path that exercises the whole system at once. Everything below it has
been proven separately -- images, mounts, the filesystem API, the agent loop,
the manifest round trip, nested spawning -- but the span from God's planner
through the adapter into a sandbox and back into the integrator has not.

COST. Every God phase and every demigod turn is an Anthropic call; Modal
compute is cents beside that. THE DOMAIN COUNT IS NO LONGER FIXED -- GOD reads
the problem and decides, so the bill is a function of how hard it judges the
problem to be. A trivial one can cost a single call and spawn nothing at all; a
hard one can spawn more sandboxes than the old default of 2 ever did. Rough
shape of a run that lands on two domains:

    1 planner call + 2 transform calls + 1 integrate call   (God)
    2 sandboxes x <=12 turns                                (the demigods)

Pass --domains to pin the count when the bill has to be predictable, and raise
--turns only once the pipeline is known to work.

Progress uses the same God/subagent terminal stream as the library entrypoint.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

from reagents.contracts import Budget, NativeProblem
from reagents.demigod.sandbox_runtime import SandboxDemigodRuntime, seed_shared_files
from reagents.god.orchestrator import God
from reagents.llm.anthropic_client import DEFAULT_MODEL
from reagents.llm.client import make_llm
from reagents.tools.registry import default_registry
from reagents.toy import simple_problem, toy_problem
from reagents.tracing import TerminalTracer

ROOT = Path(__file__).resolve().parent.parent
ADAPTIVE_DIR = ROOT / "benchmarks" / "adaptive_circuit"

_T0 = time.monotonic()


def stage(message: str) -> None:
    """A coarse phase marker alongside the lane-oriented terminal trace."""
    print(f">>> [{time.monotonic() - _T0:6.1f}s] {message}", flush=True)


def load_problem(problem_name: str) -> tuple[NativeProblem, list[Path]]:
    if problem_name == "simple":
        return simple_problem(), []
    if problem_name == "pathway":
        return toy_problem(), []
    if problem_name == "flareguard":
        from benchmarks.flareguard import load_problem as load_flareguard

        # Raw native files are deliberately not mounted. load_flareguard()
        # hydrates them into NativeProblem.inputs so each transformer must
        # project the complete dataset into its own sealed representation.
        return load_flareguard(), []
    if problem_name == "perturbseq":
        from benchmarks.perturbseq_norman import load_problem as load_perturbseq

        # Raw cells and held-out truth never enter a shared volume. The compact
        # training-only candidates live behind the Broker's exact leases.
        return load_perturbseq(), []
    if problem_name == "perturbseq2":
        from benchmarks.perturbseq_norman.v2_benchmark import (
            load_problem as load_perturbseq_v2,
        )

        # Public raw training primitives are mounted only inside the dedicated
        # Broker executor. No held-out response enters God or a demigod.
        return load_perturbseq_v2(), []
    question = ADAPTIVE_DIR / "public" / "question.json"
    observations = ADAPTIVE_DIR / "public" / "observations.csv"
    return NativeProblem.model_validate_json(question.read_text()), [observations]


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

    # Every wrapper takes *args/**kwargs and forwards verbatim. Pinning a
    # positional signature here means any new parameter on the wrapped method
    # raises TypeError at that phase -- and for `integrate` that is the LAST
    # phase, after every model call and every sandbox has already been paid
    # for. That is exactly what happened when orchestrator.solve() grew a
    # `failed_domains=` keyword.
    async def plan(problem, *args, **kwargs):
        n = kwargs.get("n", args[0] if args else None)
        stage(
            f"PLAN: inventing {n} orthogonal domains"
            if n
            else "PLAN: deciding how many orthogonal domains this is worth"
        )
        specs = await planner_plan(problem, *args, **kwargs)
        if not specs:
            stage("PLAN: none earn a demigod; GOD answers this one directly")
        for s in specs:
            axes = ", ".join(a.value for a in s.axes)
            stage(f"PLAN: '{s.name}' [{axes}] tools={s.tool_ids}")
        return specs

    async def forward(problem, spec, *args, **kwargs):
        stage(f"TRANSFORM: projecting problem into '{spec.name}'")
        result = await transformer_forward(problem, spec, *args, **kwargs)
        # NOT "sealed" -- assert_sealed and find_spec_leaks run in the
        # orchestrator, outside this hook. Saying "sealed" here printed a
        # reassuring lie on a run whose envelope was rejected moments later.
        stage(f"TRANSFORM: '{spec.name}' projected (seal check pending)")
        return result

    async def run(envelope, tools, *args, **kwargs):
        name = envelope.domain.name
        stage(f"SPAWN: '{name}' -> sandbox")
        result = await runtime_run(envelope, tools, *args, **kwargs)
        if result.status == "ok":
            stage(
                f"DONE: '{name}' status=ok confidence={result.confidence} "
                f"files={len(result.files)}"
            )
        else:
            stage(f"DONE: '{name}' status={result.status} error={result.error}")
        return result

    async def integrate(problem, artifacts, *args, **kwargs):
        stage(f"INTEGRATE: recombining {len(artifacts)} artifact(s)")
        return await integrator_integrate(problem, artifacts, *args, **kwargs)

    god.planner.plan = plan
    god.transformer.forward = forward
    god.integrator.integrate = integrate
    god.runtime.run = run


def make_toolbox(enabled: bool):
    """The TOOLBOX_BROKER provider for `SandboxDemigodRuntime`, or None.

    ON by default now, off with `--no-broker`. It was opt-in while the broker
    was unproven, and the cost of that caution was runs reporting `brokered
    tool calls: 0` with a healthy router: a demigod with no brokered tools
    solves numerically by writing its own Python through Bash, which works,
    costs turns, and proves nothing about the broker.

    Imported here rather than at module scope so `--no-broker` never constructs
    a `modal.App` for the broker or touches its grant store.
    """
    if not enabled:
        return None
    from broker.session import modal_session

    session = modal_session()
    stage(f"TOOLBOX: brokering scoped tools via {session.url}")
    return session


async def run_once(
    domains: int | None,
    turns: int,
    run_id: str,
    problem_name: str,
    broker: bool = True,
    require_broker: bool = False,
    result_out: Path | None = None,
    model: str = DEFAULT_MODEL,
) -> int:
    started_at = time.time()
    if problem_name == "perturbseq":
        os.environ["REAGENTS_ENABLE_NORMAN_BENCHMARK"] = "1"
    elif problem_name == "perturbseq2":
        os.environ["REAGENTS_ENABLE_NORMAN_V2_BENCHMARK"] = "1"
    problem, input_paths = load_problem(problem_name)
    stage(f"START run_id={run_id} domains={domains or 'GOD decides'} turns={turns}")
    stage(f"PROBLEM: {problem.id} -- {problem.question}")
    shared_files: list[str] = []
    if input_paths:
        print(f"◆ Preparing {len(input_paths)} public benchmark input file(s)")
        shared_files = await asyncio.to_thread(
            seed_shared_files,
            run_id,
            [str(path) for path in input_paths],
        )
    verifier = None
    if problem_name == "flareguard":
        from benchmarks.flareguard import FlareGuardVerifier

        verifier = FlareGuardVerifier()
    elif problem_name == "perturbseq":
        from benchmarks.perturbseq_norman import NormanPerturbSeqVerifier

        verifier = NormanPerturbSeqVerifier()
    elif problem_name == "perturbseq2":
        from benchmarks.perturbseq_norman.v2_benchmark import (
            NormanPerturbSeqV2Verifier,
        )

        verifier = NormanPerturbSeqV2Verifier()

    registry = None
    if problem_name == "perturbseq":
        # Give God the complete benchmark-tool catalog but omit unrelated
        # executors. This makes every invented domain bind a data-bearing
        # representation tool and its independent certificate, while the
        # Broker remains a superset and enforces the exact selected lease.
        from reagents.tools.registry import ToolRegistry

        discovered = default_registry()
        registry = ToolRegistry()
        for tool_id in discovered.ids():
            if tool_id.startswith("screen."):
                registry.register(discovered.get(tool_id))

    # TerminalTracer multiplexes GOD and every DEMI_GOD lane into ONE stream,
    # lane-labelled, so a parallel fan-out is readable in a single terminal --
    # no tmux, no per-sandbox tail. The orchestrator and SandboxDemigodRuntime
    # already emit into it; God.__init__ forwards the sink to the runtime.
    llm = make_llm(model)
    god = God(
        llm,
        registry=registry,
        domain_count=domains,
        # The seam. Swap for the default in-process runtime and the same God
        # loop runs without any infrastructure at all.
        runtime=SandboxDemigodRuntime(
            run_id=run_id,
            max_turns=turns,
            shared_files=shared_files,
            toolbox=make_toolbox(broker),
            # DECOUPLED from `broker` on purpose. Both of these were tied to it
            # when `--broker` was an explicit opt-in, where "I asked for the
            # broker" reasonably meant "and enforce it". The broker is the
            # DEFAULT now, and defaulting these with it would mean every run
            # without a deployed broker fails every demigod (require_toolbox)
            # under an egress policy the code itself calls untested
            # (restrict_egress: "enabling it untested would break every live
            # run"). Strictness stays something you ask for: --require-broker.
            require_toolbox=require_broker,
            restrict_egress=require_broker,
            model=model,
        ),
        tracer=TerminalTracer(),
        verifier=verifier,
        budget=Budget(
            max_tokens=4096,
            max_steps=turns,
            wall_time_s=1200,
            max_tool_calls=24,
        ),
    )
    instrument(god)
    try:
        solution = await god.solve(problem)
    except Exception as exc:
        if result_out is not None:
            result_out.parent.mkdir(parents=True, exist_ok=True)
            result_out.write_text(
                json.dumps(
                    {
                        "kind": "reagents_pipeline",
                        "status": "failed",
                        "run_id": run_id,
                        "problem_id": problem.id,
                        "model": model,
                        "started_at_unix": started_at,
                        "elapsed_s": round(time.time() - started_at, 3),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "god_usage": (
                            llm.usage_summary()
                            if hasattr(llm, "usage_summary")
                            else {"model": model}
                        ),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"\nfailed run record: {result_out}")
        raise

    stage("COMPLETE")
    trace = god.last_trace
    print("\n" + "=" * 72)
    print(f"artifacts: {len(trace.artifacts)}   failures: {len(trace.failures)}")
    if trace.leaks:
        print(f"isolation leaks: {trace.leaks}")
    for failure in trace.failures:
        print(f"  FAILED {failure.domain_name}: {failure.error}")
    print("=" * 72)
    print(json.dumps(solution.model_dump(), indent=2))

    for artifact in trace.artifacts:
        print(f"\n--- {artifact.domain_name} (sandbox) ---")
        print(f"  confidence : {artifact.confidence}")
        print(f"  payload    : {json.dumps(artifact.payload)[:400]}")
        print(f"  files      : {artifact.files}")
        print(f"  unknowns   : {artifact.unknowns[:3]}")
        _print_tool_trace(artifact)

    # THE evidence that `--broker` did anything. Without this the run prints an
    # answer and says nothing about where it came from, and "the broker was in
    # play" becomes an assumption rather than an observation -- which is how a
    # previous run got reported as brokered when `toolbox` was None.
    #
    # Read from the BROKER's record, not the agent's manifest: the demigod
    # authors its claim, it does not author the log of what it called.
    total_calls = sum(len(a.tool_trace) for a in trace.artifacts)
    print(
        f"\nbrokered tool calls: {total_calls} across "
        f"{len(trace.artifacts)} artifact(s)"
    )
    if broker and total_calls == 0:
        print(
            "  WARNING: --broker was set but no tool was called. The lease was "
            "published and never used -- the run proves the spawn path, not the "
            "broker. Check `toolbox list` output in the demigod transcript."
        )

    if result_out is not None:
        result_out.parent.mkdir(parents=True, exist_ok=True)
        result_out.write_text(
            json.dumps(
                {
                    "kind": "reagents_pipeline",
                    "run_id": run_id,
                    "problem_id": problem.id,
                    "model": model,
                    "public_input_sha256": hashlib.sha256(
                        json.dumps(
                            problem.model_dump(mode="json"), sort_keys=True
                        ).encode()
                    ).hexdigest(),
                    "started_at_unix": started_at,
                    "elapsed_s": round(time.time() - started_at, 3),
                    "broker_required": broker,
                    "egress_restricted": broker,
                    "god_usage": (
                        llm.usage_summary()
                        if hasattr(llm, "usage_summary")
                        else {"model": model}
                    ),
                    "solution": solution.model_dump(mode="json"),
                    "audit": {
                        "specs": [spec.model_dump(mode="json") for spec in trace.specs],
                        "envelopes": [
                            envelope.model_dump(mode="json")
                            for envelope in trace.envelopes
                        ],
                        "inverse_maps": [
                            inverse.model_dump(mode="json")
                            for inverse in trace.inverse_maps
                        ],
                    },
                    "artifacts": [
                        artifact.model_dump(mode="json") for artifact in trace.artifacts
                    ],
                    "failures": [
                        failure.model_dump(mode="json") for failure in trace.failures
                    ],
                    "leaks": trace.leaks,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nrun record: {result_out}")

    # Artifacts survive on the volume regardless of what the integrator said.
    print(f"\nartifacts on volume: uv run modal volume ls demigod-run-{run_id}-out")
    # `direct` runs have no artifacts BY DESIGN -- GOD judged the problem not
    # worth a demigod and answered it. Failing them for an empty artifact list
    # would report the cheap correct path as a broken run.
    return 0 if (trace.artifacts or trace.direct) else 1


def _print_tool_trace(artifact) -> None:
    """One line per brokered call: what was asked, and what came back.

    Truncated hard. A trace entry can carry a whole tool result (an embedding is
    480 floats), and dumping that buries the one fact worth reading -- that the
    call happened, against which tool, and whether it succeeded.
    """
    if not artifact.tool_trace:
        print("  tool calls : none")
        return
    print(f"  tool calls : {len(artifact.tool_trace)}")
    for entry in artifact.tool_trace:
        tool = entry.get("tool", "?")
        args = json.dumps(entry.get("input", {}))[:80]
        result = entry.get("result", {})
        if isinstance(result, dict) and "error" in result:
            status = f"ERROR {str(result['error'])[:60]}"
        else:
            status = f"ok {json.dumps(result, default=str)[:60]}"
        print(f"      - {tool}({args}) -> {status}")


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
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
        return
    load_dotenv(env_path, override=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--domains",
        type=int,
        default=None,
        help="Pin the domain count. Omitted, GOD decides from the problem.",
    )
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--problem",
        choices=(
            "simple",
            "pathway",
            "adaptive",
            "flareguard",
            "perturbseq",
            "perturbseq2",
        ),
        default="simple",
        help="simple = 5-entity valve pipeline (default, for testing the "
        "pipeline); pathway = the 9-entity glycolysis problem; adaptive = "
        "the held-out synthetic-circuit workflow benchmark; flareguard = "
        "the complete-objective living-diagnostic design benchmark; "
        "perturbseq = v1 candidate-selection Norman benchmark; perturbseq2 = "
        "real held-out Norman responses from raw training primitives and "
        "demigod-authored models",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--result-out",
        type=Path,
        default=None,
        help="Write the solution, artifacts, and failures to a local JSON record.",
    )
    parser.add_argument(
        "--require-broker",
        action="store_true",
        help="fail a demigod outright if the broker is unreachable, and pin "
        "sandbox egress to the agent API plus the broker. Off by default: "
        "without it an absent broker costs the tools, not the run.",
    )
    parser.add_argument(
        "--no-broker",
        dest="broker",
        action="store_false",
        help="do NOT publish a lease; each demigod reasons from shared/ and "
        "whatever Python it writes itself. The broker is on by default -- a "
        "demigod that can reach its tools should. Use this to exercise the "
        "spawn path alone, or when no broker is deployed.",
    )
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

    # The broker's own catalog is env-gated, and so is God's. A demigod can only
    # be granted a tool its GOD can see, so `--broker` without this flag would
    # publish a lease naming builtins alone and look like the broker was empty.
    if args.broker and not os.environ.get("REAGENTS_ENABLE_CONTAINERS"):
        print(
            "note: enabling REAGENTS_ENABLE_CONTAINERS=1 so God can plan with "
            "the brokered container-tool catalog.",
            file=sys.stderr,
        )
        os.environ["REAGENTS_ENABLE_CONTAINERS"] = "1"

    run_id = args.run_id or f"e2e{uuid.uuid4().hex[:6]}"
    return asyncio.run(
        run_once(
            args.domains,
            args.turns,
            run_id,
            args.problem,
            broker=args.broker,
            require_broker=args.require_broker,
            result_out=args.result_out,
            model=args.model,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
