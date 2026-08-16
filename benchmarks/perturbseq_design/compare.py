"""Score a pipeline record beside frozen deterministic baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.perturbseq_design.benchmark import score_solution


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pipeline", type=Path)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pipeline = json.loads(args.pipeline.read_text(encoding="utf-8"))
    result = {
        "pipeline": score_solution(pipeline),
        "deterministic_baselines": json.loads(
            args.baselines.read_text(encoding="utf-8")
        ),
    }
    if args.base:
        base = json.loads(args.base.read_text(encoding="utf-8"))
        result["single_claude"] = base.get("score", score_solution(base))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
