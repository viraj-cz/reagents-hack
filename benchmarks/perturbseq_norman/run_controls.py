"""Run matched native-representation Claude controls for Norman Perturb-seq."""

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

from benchmarks.perturbseq_norman.benchmark import (
    TARGET_IDS,
    load_problem,
    normalize_predictions,
    score_solution,
)
from broker.session import modal_session
from demigod.egress import allowlist
from demigod.result import DemiGodResult
from demigod.spawn import spawn_demigod
from demigod.spec import DemiGodSpec, Problem
from reagents.contracts import Budget
from reagents.tools.registry import default_registry

DEFAULT_MODEL = "claude-opus-4-8"
TOOL_IDS = (
    "screen.algebra_candidate",
    "screen.algebra_certificate",
    "screen.geometry_candidate",
    "screen.geometry_certificate",
    "screen.graph_candidate",
    "screen.graph_certificate",
    "screen.information_candidate",
    "screen.information_certificate",
)


def _public_hash() -> str:
    problem = load_problem()
    return hashlib.sha256(
        json.dumps(problem.model_dump(mode="json"), sort_keys=True).encode()
    ).hexdigest()


def _context() -> str:
    problem = load_problem()
    return (
        f"{problem.statement}\n\nConstraints:\n- "
        + "\n- ".join(problem.constraints)
        + f"\n\nQuestion:\n{problem.question}\n\n"
        f"Public inputs:\n{json.dumps(problem.inputs, indent=2)}\n\n"
        "Solve directly in the native biological representation. The Broker "
        "contains training-only summaries; call its tools for all T01-T12. "
        "Return predictions in payload matching the exact answer schema."
    )


def _run_sample(index: int, *, run_id: str, model: str, turns: int) -> DemiGodResult:
    name = f"native-norman-{index:02d}"
    registry = default_registry()
    budget = Budget(
        max_tokens=4096, max_steps=turns, wall_time_s=1200, max_tool_calls=24
    )
    pack = registry.bind(list(TOOL_IDS), subject_id=name, budget=budget)
    session = modal_session()
    grant = session.grant(pack, label=name)
    try:
        spec = DemiGodSpec(
            name=name,
            domain="native perturbational transcriptomics",
            domain_name=name,
            model=model,
            toolbox=grant,
            egress_domains=allowlist(grant.base),
            problem=Problem(
                context=_context(),
                goal=(
                    "Produce one independently complete prediction for all 12 "
                    "held-out combinations without an invented representation."
                ),
                success_criteria=list(load_problem().required_outputs),
            ),
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


def _ensemble(
    results: list[DemiGodResult],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    valid = []
    for result in results:
        if result.status != "ok":
            continue
        predictions, errors = normalize_predictions(
            {"structured_answer": result.payload}
        )
        if not errors:
            valid.append(predictions)
    if not valid:
        return None, {"passed": False, "score_10": 0.0, "errors": ["no valid samples"]}
    combined = []
    for target_id in TARGET_IDS:
        rows = [
            next(item for item in sample if item["target_id"] == target_id)
            for sample in valid
        ]
        values = [
            sum(row["predicted_delta"][feature] for row in rows) / len(rows)
            for feature in range(64)
        ]
        classes = Counter(row["interaction_class"] for row in rows)
        interaction_class = sorted(classes, key=lambda label: (-classes[label], label))[
            0
        ]
        combined.append(
            {
                "target_id": target_id,
                "predicted_delta": values,
                "interaction_class": interaction_class,
                "confidence": sum(row["confidence"] for row in rows) / len(rows),
                "falsifier": (
                    "Independent native samples disagree or held-out pseudobulk falls "
                    "outside their predicted direction."
                ),
            }
        )
    answer = {
        "predictions": combined,
        "method_summary": (
            f"Arithmetic ensemble of {len(valid)} independently sampled native "
            "Claude agents."
        ),
    }
    return answer, score_solution({"structured_answer": answer}, private=True)


def run_controls(
    *, samples: int, turns: int, model: str, output: Path
) -> dict[str, Any]:
    os.environ["REAGENTS_ENABLE_NORMAN_BENCHMARK"] = "1"
    run_id = f"norman-control-{uuid.uuid4().hex[:8]}"
    started = time.time()
    with ThreadPoolExecutor(max_workers=samples) as pool:
        futures = [
            pool.submit(_run_sample, index, run_id=run_id, model=model, turns=turns)
            for index in range(1, samples + 1)
        ]
        results = [future.result() for future in futures]
    scores = [_score(result) for result in results]
    ensemble_answer, ensemble_score = _ensemble(results)
    record = {
        "kind": "native_independent_controls",
        "run_id": run_id,
        "model": model,
        "samples": samples,
        "turns": turns,
        "tool_ids": list(TOOL_IDS),
        "public_input_sha256": _public_hash(),
        "freeze_manifest": json.loads(
            (Path(__file__).parent / "public" / "freeze_manifest.json").read_text()
        ),
        "elapsed_s": round(time.time() - started, 3),
        "single_sample_score": scores[0],
        "independent_ensemble_answer": ensemble_answer,
        "independent_ensemble_score": ensemble_score,
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = run_controls(
        samples=args.samples, turns=args.turns, model=args.model, output=args.output
    )
    print(
        json.dumps(
            {key: value for key, value in record.items() if key != "results"}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
