"""Privately score and compare a pipeline run with matched Claude controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.perturbseq_norman.benchmark import score_solution


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pipeline", type=Path)
    parser.add_argument("controls", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    pipeline = json.loads(args.pipeline.read_text(encoding="utf-8"))
    controls = json.loads(args.controls.read_text(encoding="utf-8"))
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
        },
        "single_native": controls.get("single_sample_score"),
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
