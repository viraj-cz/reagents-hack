"""Run native-representation Claude Code controls for FlareGuard.

These controls use the same pinned model, turn cap, Modal sandbox runner, raw
public inputs, and exact Broker tool grants as the pipeline demigods. They omit
only God's invented representations and integration. Private fixtures are read
locally after every sandbox has terminated and are never mounted or prompted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from benchmarks.flareguard.benchmark import (
    PUBLIC_DIR,
    load_problem,
    normalize_design,
    score_solution,
)
from broker.session import modal_session
from demigod.egress import allowlist
from demigod.layout import RunLayout
from demigod.result import DemiGodResult
from demigod.spawn import spawn_demigod
from demigod.spec import DemiGodSpec, Problem
from reagents.contracts import Budget
from reagents.tools.registry import default_registry

DEFAULT_MODEL = "claude-opus-4-8"
PUBLIC_FILES = (
    PUBLIC_DIR / "trajectories.csv",
    PUBLIC_DIR / "parts.json",
    PUBLIC_DIR / "compatibility.json",
)


def _public_hash() -> str:
    problem = load_problem()
    return hashlib.sha256(
        json.dumps(problem.model_dump(mode="json"), sort_keys=True).encode()
    ).hexdigest()


def _problem_context() -> str:
    problem = load_problem()
    return (
        f"{problem.statement}\n\nConstraints:\n- "
        + "\n- ".join(problem.constraints)
        + f"\n\nQuestion:\n{problem.question}\n\n"
        "Solve directly in the native biological representation. Use the three "
        "public files listed below. Return every required output in payload."
    )


def _run_sample(
    index: int,
    *,
    run_id: str,
    files: list[str],
    model: str,
    turns: int,
    tool_ids: list[str],
) -> DemiGodResult:
    name = f"native-control-{index:02d}"
    registry = default_registry()
    budget = Budget(
        max_tokens=4096,
        max_steps=turns,
        wall_time_s=1200,
        max_tool_calls=24,
    )
    pack = registry.bind(tool_ids, subject_id=name, budget=budget)
    session = modal_session()
    grant = session.grant(pack, label=name)
    try:
        spec = DemiGodSpec(
            name=name,
            domain="native biological circuit design",
            domain_name=name,
            model=model,
            toolbox=grant,
            egress_domains=allowlist(grant.base),
            problem=Problem(
                context=_problem_context(),
                goal="Return one independently complete native FlareGuard design.",
                success_criteria=list(load_problem().required_outputs),
            ),
            files=files,
            artifact_schema=load_problem().answer_schema,
            max_turns=turns,
            max_lifetime_s=1500,
            idle_timeout_s=180,
        )
        result = spawn_demigod(spec, run_id=run_id)
        result.tool_trace = session.collect_trace(grant.lease_id)
        return result
    finally:
        session.revoke(grant.lease_id)


def _score(result: DemiGodResult) -> dict[str, Any]:
    if result.status != "ok":
        return {"passed": False, "score_10": 0.0, "errors": [result.error]}
    return score_solution({"structured_answer": result.payload}, private=True)


def _majority(results: list[DemiGodResult]) -> tuple[int | None, dict[str, Any]]:
    designs: list[tuple[int, str]] = []
    for index, result in enumerate(results):
        if result.status != "ok":
            continue
        try:
            normalized = normalize_design(result.payload)
        except (KeyError, TypeError, ValueError):
            continue
        designs.append((index, json.dumps(normalized, sort_keys=True)))
    if not designs:
        return None, {"passed": False, "score_10": 0.0, "errors": ["no valid vote"]}
    counts = Counter(encoded for _, encoded in designs)
    winner = counts.most_common(1)[0][0]
    selected = next(index for index, encoded in designs if encoded == winner)
    return selected, _score(results[selected])


def run_controls(
    *,
    samples: int,
    turns: int,
    model: str,
    tool_ids: list[str],
    output: Path,
) -> dict[str, Any]:
    os.environ.setdefault("REAGENTS_ENABLE_CONTAINERS", "1")
    run_id = f"fg-control-{uuid.uuid4().hex[:8]}"
    layout = RunLayout(run_id=run_id, demigod_name="native-control-01")
    files = layout.seed_shared(list(PUBLIC_FILES))
    started = time.time()
    with ThreadPoolExecutor(max_workers=samples) as pool:
        futures = [
            pool.submit(
                _run_sample,
                index,
                run_id=run_id,
                files=files,
                model=model,
                turns=turns,
                tool_ids=tool_ids,
            )
            for index in range(1, samples + 1)
        ]
        results = [future.result() for future in futures]
    scores = [_score(result) for result in results]
    majority_index, majority_score = _majority(results)
    record = {
        "kind": "native_independent_controls",
        "run_id": run_id,
        "model": model,
        "samples": samples,
        "turns": turns,
        "tool_ids": tool_ids,
        "public_input_sha256": _public_hash(),
        "elapsed_s": round(time.time() - started, 3),
        "single_sample_score": scores[0],
        "majority_selected_sample": (
            None if majority_index is None else majority_index + 1
        ),
        "majority_score": majority_score,
        "oracle_best_sample_score": max(
            (score.get("score_10", 0.0) for score in scores), default=0.0
        ),
        "results": [result.model_dump(mode="json") for result in results],
        "scores": scores,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tool", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = run_controls(
        samples=args.samples,
        turns=args.turns,
        model=args.model,
        tool_ids=args.tool,
        output=args.output,
    )
    print(json.dumps({k: v for k, v in record.items() if k != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
