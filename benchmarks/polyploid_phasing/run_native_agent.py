"""One native Claude Code agent with local Python, but no transformed tools."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

from benchmarks.polyploid_phasing.benchmark import (
    PUBLIC_DIR,
    load_problem,
    score_solution,
)
from demigod.egress import allowlist
from demigod.spawn import spawn_demigod
from demigod.spec import DEFAULT_AGENT_MODEL, DemiGodSpec, Problem
from reagents.demigod.sandbox_runtime import seed_shared_files


def _public_hash() -> str:
    public = {
        "problem": load_problem().model_dump(mode="json"),
        "observations": json.loads((PUBLIC_DIR / "observations.json").read_text()),
    }
    return hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()


def run(*, model: str, turns: int, output: Path) -> dict[str, Any]:
    run_id = f"polyploid-native-{uuid.uuid4().hex[:8]}"
    problem = load_problem()
    files = seed_shared_files(run_id, [str(PUBLIC_DIR / "observations.json")])
    spec = DemiGodSpec(
        name="native-polyploid-code-agent",
        domain="native polyploid computational genomics",
        domain_name="native_polyploid_code_agent",
        model=model,
        toolbox=None,
        tools=["pandas"],
        egress_domains=allowlist(),
        problem=Problem(
            context=(
                f"{problem.statement}\n\nConstraints:\n- "
                + "\n- ".join(problem.constraints)
                + f"\n\nQuestion:\n{problem.question}\n\n"
                "The complete public matrix is shared/observations.json. Work in "
                "the native field using local Python. No Broker, specialized "
                "phasing package, web, private reference, or subagents are available."
            ),
            goal="Return four complete dosage-consistent haplotypes and phase blocks.",
            success_criteria=list(problem.required_outputs),
        ),
        files=files,
        artifact_schema=problem.answer_schema,
        miscellaneous={"baseline_kind": "native_code_no_domain_tools"},
        max_turns=turns,
        max_lifetime_s=6750,
        idle_timeout_s=180,
    )
    started = time.time()
    result = spawn_demigod(spec, run_id=run_id)
    record = {
        "kind": "native_claude_code_polyploid",
        "run_id": run_id,
        "model": model,
        "turns": turns,
        "toolbox": None,
        "domain_tools_provided": [],
        "public_input_sha256": _public_hash(),
        "elapsed_s": round(time.time() - started, 3),
        "result": result.model_dump(mode="json"),
        "score": None,
        "response_committed_before_private_scoring": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if result.status == "ok":
        record["score"] = score_solution(result.payload, private=True)
    else:
        record["score"] = {
            "passed": False,
            "score_10": 0.0,
            "errors": [result.error or result.status],
        }
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument("--turns", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = run(model=args.model, turns=args.turns, output=args.output)
    print(json.dumps({k: v for k, v in record.items() if k != "result"}, indent=2))
    return 0 if record["score"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
