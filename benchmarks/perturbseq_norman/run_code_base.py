"""Run one native Claude Code agent with no Broker or domain tools."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

from benchmarks.perturbseq_norman.benchmark import load_problem, score_solution
from demigod.egress import allowlist
from demigod.spawn import spawn_demigod
from demigod.spec import DEFAULT_AGENT_MODEL, DemiGodSpec, Problem


def _public_hash() -> str:
    public = load_problem().model_dump(mode="json")
    return hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()


def context() -> str:
    problem = load_problem()
    public = problem.model_dump(mode="json")
    return (
        "Solve the following frozen problem directly in its native biological "
        "representation. You have the standard local Claude Code file and shell "
        "utilities, but no Broker lease, domain tools, internet, retrieval, "
        "subagents, or additional evidence. The broker_interfaces field is "
        "contextual metadata only: do not call those names or claim their "
        "outputs. Use only the supplied public information and biological "
        "reasoning.\n\n"
        f"Public problem:\n{json.dumps(public, indent=2)}"
    )


def run(*, model: str, turns: int, output: Path) -> dict[str, Any]:
    run_id = f"norman-code-base-{uuid.uuid4().hex[:8]}"
    spec = DemiGodSpec(
        name="native-norman-code-base",
        domain="native perturbational transcriptomics",
        domain_name="native_norman_code_base",
        model=model,
        toolbox=None,
        tools=[],
        # Anthropic endpoints only. No Broker or arbitrary internet egress.
        egress_domains=allowlist(),
        problem=Problem(
            context=context(),
            goal=(
                "Independently produce one complete native prediction for all "
                "12 held-out combinations without any domain tool."
            ),
            success_criteria=list(load_problem().required_outputs),
        ),
        artifact_schema=load_problem().answer_schema,
        miscellaneous={
            "baseline_kind": "claude_code_no_domain_tools",
            "broker_available": False,
            "web_available": False,
        },
        max_turns=turns,
        max_lifetime_s=1500,
        idle_timeout_s=180,
    )
    started = time.time()
    result = spawn_demigod(spec, run_id=run_id)
    record = {
        "kind": "single_claude_code_no_domain_tools",
        "run_id": run_id,
        "model": model,
        "turns": turns,
        "toolbox": None,
        "domain_tools_provided": [],
        "claude_code_tools": ["Read", "Write", "Edit", "Glob", "Grep", "Bash"],
        "egress_domains": allowlist(),
        "public_input_sha256": _public_hash(),
        "elapsed_s": round(time.time() - started, 3),
        "result": result.model_dump(mode="json"),
        "score": None,
        "response_committed_before_private_scoring": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if result.status == "ok":
        record["score"] = score_solution(
            {"structured_answer": result.payload}, private=True
        )
    else:
        record["score"] = {
            "passed": False,
            "score_10": 0.0,
            "errors": [result.error or f"agent status was {result.status}"],
        }
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = run(model=args.model, turns=args.turns, output=args.output)
    print(
        json.dumps(
            {key: value for key, value in record.items() if key != "result"},
            indent=2,
        )
    )
    return 0 if record["score"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
