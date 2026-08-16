"""Privately grade and compare one pipeline run with native controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.flareguard.benchmark import score_solution


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pipeline", type=Path)
    parser.add_argument("controls", type=Path)
    args = parser.parse_args()
    pipeline = json.loads(args.pipeline.read_text(encoding="utf-8"))
    controls = json.loads(args.controls.read_text(encoding="utf-8"))
    pipeline_score = score_solution(pipeline, private=True)
    summary = {
        "model": pipeline.get("model"),
        "public_hash_match": pipeline.get("public_input_sha256")
        == controls.get("public_input_sha256"),
        "pipeline": {
            "score": pipeline_score,
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
        "majority_native": controls.get("majority_score"),
        "oracle_best_native_score": controls.get("oracle_best_sample_score"),
        "controls_elapsed_s": controls.get("elapsed_s"),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
