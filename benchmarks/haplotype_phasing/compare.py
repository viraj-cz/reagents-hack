"""Score committed pipeline and baseline records against private expected phase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.haplotype_phasing.benchmark import score_solution


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(pipeline: Path, standard: Path, native_agent: Path) -> dict[str, Any]:
    pipeline_record = _load(pipeline)
    standard_record = _load(standard)
    native_record = _load(native_agent)
    pipeline_score = score_solution(pipeline_record, private=True)
    return {
        "benchmark_id": "giab-hg004-long-read-phasing-v1",
        "scoring_happened_after_all_responses_were_committed": True,
        "conditions": {
            "pipeline": {
                "access": (
                    "God plus sealed alternative-domain workers and leased "
                    "binary primitives"
                ),
                "score": pipeline_score,
                "artifact_count": len(pipeline_record.get("artifacts", [])),
                "failures": len(pipeline_record.get("failures", [])),
            },
            "standard_claude": {
                "access": "one native-field Claude API call, no tools",
                "score": standard_record.get("score"),
            },
            "native_code_agent": {
                "access": (
                    "one native-field Claude Code agent with local Python, no Broker"
                ),
                "score": native_record.get("score"),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pipeline", type=Path)
    parser.add_argument("standard", type=Path)
    parser.add_argument("native_agent", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = compare(args.pipeline, args.standard, args.native_agent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
