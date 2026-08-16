"""Privately score and compare a pipeline run with matched Claude controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.perturbseq_norman.benchmark import TARGET_IDS, score_solution


def _artifact_answer(artifact: dict) -> dict:
    """Project one sealed artifact for post-run, oracle-only diagnostics."""

    candidates = artifact.get("payload", {}).get("candidate_solution", {})
    if set(TARGET_IDS).issubset(candidates):
        ordered = [(target_id, candidates[target_id]) for target_id in TARGET_IDS]
    elif len(candidates) == len(TARGET_IDS):
        ordered = list(
            zip(
                TARGET_IDS,
                (candidates[key] for key in sorted(candidates)),
                strict=True,
            )
        )
    else:
        return {"structured_answer": {"predictions": []}}
    return {
        "structured_answer": {
            "predictions": [
                {
                    "target_id": target_id,
                    "predicted_delta": candidate.get(
                        "delta_vector", candidate.get("displacement_vector")
                    ),
                    "interaction_class": candidate.get(
                        "interaction_class", candidate.get("regime_label")
                    ),
                    "confidence": candidate.get(
                        "confidence", artifact.get("confidence", 0.0)
                    ),
                    "falsifier": candidate.get(
                        "falsifier", "Post-run domain artifact diagnostic."
                    ),
                }
                for target_id, candidate in ordered
            ]
        }
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pipeline", type=Path)
    parser.add_argument("controls", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--code-base", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    pipeline = json.loads(args.pipeline.read_text(encoding="utf-8"))
    controls = json.loads(args.controls.read_text(encoding="utf-8"))
    base = json.loads(args.base.read_text(encoding="utf-8")) if args.base else None
    code_base = (
        json.loads(args.code_base.read_text(encoding="utf-8"))
        if args.code_base
        else None
    )
    artifact_diagnostics = [
        {
            "domain_name": artifact.get("domain_name"),
            "confidence": artifact.get("confidence"),
            "score": score_solution(_artifact_answer(artifact), private=True),
        }
        for artifact in pipeline.get("artifacts", [])
    ]
    valid_artifacts = [
        item for item in artifact_diagnostics if item["score"].get("passed")
    ]
    best_artifact = max(
        valid_artifacts,
        key=lambda item: item["score"]["score_10"],
        default=None,
    )
    summary = {
        "model": pipeline.get("model"),
        "public_hash_match": pipeline.get("public_input_sha256")
        == controls.get("public_input_sha256"),
        "private_commitment": controls.get("freeze_manifest", {}).get(
            "private_expected_sha256"
        ),
        "pipeline": {
            "score": score_solution(pipeline, private=True),
            "elapsed_s": pipeline.get("elapsed_s"),
            "god_usage": pipeline.get("god_usage"),
            "demigod_runtime": [
                artifact.get("miscellaneous", {}).get("runtime", {})
                for artifact in pipeline.get("artifacts", [])
            ],
            "tool_calls": sum(
                len(artifact.get("tool_trace", []))
                for artifact in pipeline.get("artifacts", [])
            ),
            "artifact_diagnostics_oracle_only": artifact_diagnostics,
            "best_artifact_oracle_only": (
                {
                    "domain_name": best_artifact["domain_name"],
                    "score_10": best_artifact["score"]["score_10"],
                }
                if best_artifact
                else None
            ),
        },
        "base_claude_no_tools": base.get("score") if base else None,
        "base_claude_code_no_domain_tools": (
            code_base.get("score") if code_base else None
        ),
        "single_native_tool_augmented": controls.get("single_sample_score"),
        "independent_native_ensemble": controls.get("independent_ensemble_score"),
        "oracle_best_native_diagnostic": controls.get("oracle_best_sample_score"),
        "controls_elapsed_s": controls.get("elapsed_s"),
    }
    rendered = json.dumps(summary, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
